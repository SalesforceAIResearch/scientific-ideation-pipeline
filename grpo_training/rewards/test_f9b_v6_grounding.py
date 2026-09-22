"""Offline tests for the F9-B v6 GROUNDING GATE (contrib_i *= g_i).

No live judge: _judge_chat is patched to return queued raw strings, exactly as in
test_f9b_v1_impute.py. Covers the four things that can silently break training:

  1. the g mapping itself (the rubric min-max, load_bearing 1..5 -> 0.00..1.00,
     plus the saturating/floored variant that stays reachable by config)
  2. F9B_GROUNDING=0 is INERT -- byte-identical score and zero judge calls
  3. a judge failure IMPUTES (a Bedrock outage must be a GRPO no-op, never a
     grounding penalty), and a missing document block does the same
  4. all FOUR return sites emit an identical key set -- ragged reward_extra_info
     arrays break verl's W&B logging

    F9B_GROUNDING=1 python grpo_training/rewards/test_f9b_v6_grounding.py

The env var is set by the runner below, since the module reads it at import.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

# Read at import time by the module under test; force it ON before importing.
os.environ["F9B_GROUNDING"] = "1"

import f9b_impdist_len_v6 as R  # noqa: E402
from common import grounding  # noqa: E402
from common.ideas import Grounding, Idea  # noqa: E402


def _idea(tag: str, gtype: str = "cross_domain_transfer") -> Idea:
    return Idea(motivation=f"mot {tag}", problem=f"prob {tag}",
                insight_claim=f"insight {tag}",
                grounding=(Grounding(source=f"doc {tag}", supports=f"sup {tag}"),),
                grounding_type=gtype)


def _gjson(lb: int, faithful: int = 3) -> str:
    return json.dumps({"source_exists": 4, "support_faithful": faithful,
                       "load_bearing": lb, "type_correct": 4, "rationale": "r"})


def _patch_judge(by_label):
    """Patch _judge_chat to serve queued replies PER LABEL.

    Keyed by label because impact / distance / grounding all fan out concurrently,
    so a single flat queue would be order-dependent and flaky. A queued None
    simulates a timeout (what _judge_chat returns on APITimeoutError)."""
    queues = {k: list(v) for k, v in by_label.items()}
    calls = {k: 0 for k in queues}

    async def fake(client, model, messages, *, max_tokens, scijudge_style,
                   label="judge"):
        q = queues.get(label, [])
        i = calls.get(label, 0)
        calls[label] = i + 1
        # last reply repeats, so a test only has to enumerate what it cares about
        if not q:
            return None
        return q[i] if i < len(q) else q[-1]

    R._judge_chat = fake
    R._impact_client = lambda: None
    R._idea_client = lambda: None
    return calls


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


DOCS = ["doc zero body", "doc one body"]


# --------------------------------------------------------------------------- #
# 1. the mapping
# --------------------------------------------------------------------------- #
def test_features_to_g_mapping():
    """DEFAULT map is the rubric's own 1-5 min-max: g = (load_bearing - 1) / 4."""
    g = lambda lb: grounding.features_to_g({"load_bearing": lb},
                                           "cross_domain_transfer")
    assert g(1) == 0.00, g(1)   # "stands completely without the cited docs"
    assert g(2) == 0.25, g(2)
    assert g(3) == 0.50, g(3)
    assert g(4) == 0.75, g(4)
    assert g(5) == 1.00, g(5)
    # judge failure / no features -> floor from THIS function, but the reward
    # imputes instead of using it (see test_grounding_failure_imputes)
    assert grounding.features_to_g(None, "contradiction") == grounding.LB_FLOOR
    # an invalid grounding_type IS real signal -> floor
    assert grounding.features_to_g({"load_bearing": 5}, None) == grounding.LB_FLOOR
    # ...unless the schema has no such field (SI shim)
    assert grounding.features_to_g({"load_bearing": 5}, None,
                                   require_type=False) == 1.0
    print("PASS: features_to_g is the rubric min-max, 1..5 -> 0.00/0.25/0.50/0.75/1.00")


def test_judged_claim_is_a_hook_and_v0_reads_insight_claim():
    """WHICH FIELD goes in the rubric's `=== INSIGHT ===` slot is a property of the
    SCHEMA, so it is a rebindable hook (like _grounding_text_of), and v0's binding
    must stay `insight_claim`.

    v0's prompt requires the claim prose to state what the cited docs do and the
    BRIDGE from them to the claim (prompts/ideation.py:57-72), so the claim is the
    right subject there. SI's core_claim is specified to stand ALONE ("do not just
    restate the gap"), which is why the SI shim rebinds this to problem_gap -- see
    test_si_shim_judges_the_problem_gap."""
    idea = _idea("0")
    assert R._judged_claim_of(idea) == idea.insight_claim

    # ...and the judge really is handed that text (the hook is wired into _grounding,
    # not merely defined). Also assert the OTHER fields land in their own slots, so a
    # future edit cannot quietly swap the subject for the whole card.
    seen = {}

    async def fake(client, model, messages, *, max_tokens, scijudge_style,
                   label="judge"):
        seen["prompt"] = messages[1]["content"]
        return _gjson(3)

    real_chat, real_client = R._judge_chat, R._idea_client
    try:
        R._judge_chat = fake
        R._idea_client = lambda: None   # else _grounding resolves a REAL judge URL
        feats, failed = _run(R._grounding(idea, DOCS, asyncio.Semaphore(1)))
    finally:
        R._judge_chat, R._idea_client = real_chat, real_client
    assert not failed and feats is not None
    p = seen["prompt"]
    assert idea.insight_claim in p                 # the INSIGHT slot
    assert idea.grounding_text() in p              # the DECLARED GROUNDING slot
    assert "doc zero body" in p                    # the SOURCE DOCUMENTS slot
    assert idea.problem not in p                   # v0 does NOT judge the problem
    print("PASS: _judged_claim_of is a hook and v0 judges insight_claim")


def test_si_shim_judges_the_problem_gap():
    """The SI shim must rebind BOTH schema hooks, or the gate silently measures the
    wrong thing on an SI arm. Asserted by reading the shim source rather than
    importing it (importing rebinds v6's globals process-wide and would corrupt every
    later test in this file), so this is a guard against the binding being dropped."""
    src = (Path(__file__).parent / "f9b_impdist_len_v6_si_bedrock.py").read_text()
    assert "_v6._judged_claim_of = lambda idea: idea.problem_gap" in src, \
        "SI shim must judge problem_gap, not core_claim"
    assert "_v6._grounding_text_of = lambda idea: idea.provenance_text()" in src
    assert "_v6.GROUNDING_REQUIRE_TYPE = False" in src
    print("PASS: the SI shim rebinds the judged field to problem_gap")


def test_no_grounding_type_omits_the_type_rubric():
    """A schema WITHOUT grounding_type (SI) must have the type feature OMITTED, not
    rendered as "(none declared / ungrounded)".

    The old placeholder told the judge the idea was UNGROUNDED inside the same prompt
    that asks whether its citation is decorative -- a leading premise that biases
    load_bearing, the one feature the reward prices. It also asked for a score on a
    field the card was never asked to have."""
    si = grounding.build_messages("GAP", "INSPIRATION", None, DOCS)[1]["content"]
    assert "ungrounded" not in si, "must not tell the judge the idea is ungrounded"
    assert "type_correct" not in si          # not in the rubric, not in the JSON
    assert "GROUNDING TYPE" not in si        # no section, no intro clause
    assert "Score three" in si
    assert "\n\n\n" not in si                # spacing closes up cleanly
    for slot in ("=== INSIGHT ===", "=== DECLARED GROUNDING ===",
                 "=== SOURCE DOCUMENTS ==="):
        assert slot in si, slot
    # ...and the DECLARED GROUNDING / DOCUMENTS the judge sees are untouched
    assert "INSPIRATION" in si and "doc zero body" in si

    # A v0 card WITH a type is unchanged: four features, type section present.
    v0 = grounding.build_messages("CLAIM", "GND", "contradiction", DOCS)[1]["content"]
    assert "Score four" in v0
    assert "=== DECLARED GROUNDING TYPE ===\ncontradiction" in v0
    assert '"type_correct": 1 | 2 | 3 | 4 | 5,' in v0
    # An ABSENT type on a schema that HAS the field is real signal, so it keeps the
    # original wording -- ask_type is explicit here, which is what the reward passes.
    v0_none = grounding.build_messages("CLAIM", "GND", None, DOCS,
                                       ask_type=True)[1]["content"]
    assert "(none declared / ungrounded)" in v0_none
    print("PASS: no-type schemas omit the type rubric instead of declaring it ungrounded")


def test_parse_features_matches_what_was_asked():
    """parse_features must agree with build_messages about whether type_correct was
    requested: if the prompt never asked for it, its absence is CORRECT, and treating
    it as a missing field would turn every SI reply into a judge failure -> imputed ->
    the gate silently inert on the whole arm."""
    three = {"source_exists": 4, "support_faithful": 3, "load_bearing": 2,
             "rationale": "r"}
    assert grounding.parse_features(three, require_type=False) is not None
    assert grounding.parse_features(three) is None          # strict by default
    # a volunteered type is still kept
    got = grounding.parse_features({**three, "type_correct": 5}, require_type=False)
    assert got and got["type_correct"] == 5
    # fidelity averages over the features PRESENT, so 3-feature results stay on 1-5
    assert grounding.fidelity({"source_exists": 4, "support_faithful": 3,
                               "load_bearing": 2}) == 3.0
    # out-of-range still fails, with or without the type
    assert grounding.parse_features({**three, "load_bearing": 9},
                                    require_type=False) is None
    print("PASS: parse_features requires exactly the features build_messages asked for")


def test_features_to_g_saturating_variant():
    """The tuned variant stays reachable by config (LO=1, HI=3, FLOOR=0.25): the
    clip is what makes it SATURATE and FLOOR, and it is the fallback documented in
    launch_grpo_v6_grounding_onenode_h200.sbatch if the default map proves harsh."""
    g = lambda lb: grounding.features_to_g({"load_bearing": lb},
                                           "cross_domain_transfer",
                                           lo=1.0, hi=3.0, floor=0.25)
    assert (g(1), g(2), g(3)) == (0.25, 0.50, 1.00), (g(1), g(2), g(3))
    assert g(4) == 1.00 and g(5) == 1.00     # saturating above HI
    print("PASS: the saturating/floored variant still maps 1/2/3 -> 0.25/0.50/1.00")


# --------------------------------------------------------------------------- #
# 2. inert when off
# --------------------------------------------------------------------------- #
def test_gate_off_is_inert():
    """F9B_GROUNDING=0 -> every g == 1.0, score identical to pre-gate v6, and NOT
    ONE grounding judge call fired. This is the regression that lets the flag ship
    dark."""
    ideas = [_idea("0"), _idea("1"), _idea("2")]
    try:
        R.GROUNDING_ON = False
        calls = _patch_judge({"impact": ["Answer: A", "Answer: B"],
                              "distance": ['{"similarity": 0.5}'],
                              "grounding": [_gjson(1)]})
        off = _run(R._score_set(ideas, "gold", "answer", documents=DOCS))
        assert calls.get("grounding", 0) == 0, calls  # zero grounding calls fired
        assert off["grounding_g_mean"] == 1.0, off["grounding_g_mean"]
        assert off["grounding_fail"] == 0.0
        assert all(g == 1.0 for g in off["_gs"]), off["_gs"]

        # same set, gate ON, judge says load_bearing=2 -> every contribution x0.25
        # (lb=2 rather than 1: under the default FLOOR=0, lb=1 gives g=0 and the
        # ratio test would pass trivially for a body that is zero for any reason)
        R.GROUNDING_ON = True
        _patch_judge({"impact": ["Answer: A", "Answer: B"],
                      "distance": ['{"similarity": 0.5}'],
                      "grounding": [_gjson(2)]})
        on = _run(R._score_set(ideas, "gold", "answer", documents=DOCS))
        body_off = off["score"] - R.SCHEMA_BONUS + off["len_penalty"]
        body_on = on["score"] - R.SCHEMA_BONUS + on["len_penalty"]
        assert body_off > 1e-6, body_off      # the ratio below must mean something
        assert abs(body_on - 0.25 * body_off) < 1e-9, (body_on, body_off)
        assert on["load_bearing_mean"] == 2.0, on["load_bearing_mean"]
        assert on["grounding_fail"] == 0.0

        # lb=1 zeroes the body outright at the default FLOOR=0
        _patch_judge({"impact": ["Answer: A", "Answer: B"],
                      "distance": ['{"similarity": 0.5}'],
                      "grounding": [_gjson(1)]})
        dead = _run(R._score_set(ideas, "gold", "answer", documents=DOCS))
        assert abs(dead["score"] - R.SCHEMA_BONUS + dead["len_penalty"]) < 1e-9, dead["score"]
    finally:
        R.GROUNDING_ON = True
    print("PASS: gate off fires no judge calls and leaves the reward unchanged; "
          "gate on scales body by g (lb=2 -> x0.25, lb=1 -> 0)")


def test_gate_scales_per_idea_not_per_set():
    """The drift concentrates in the TAIL of the set, so g must be PER IDEA."""
    ideas = [_idea("0"), _idea("1"), _idea("2")]
    # grounding queue is consumed in idea order (one task per idea, created in order)
    _patch_judge({"impact": ["Answer: A", "Answer: B"],
                  "distance": ['{"similarity": 0.5}'],
                  "grounding": [_gjson(5), _gjson(3), _gjson(1)]})
    res = _run(R._score_set(ideas, "gold", "answer", documents=DOCS))
    assert res["_gs"] == [1.0, 0.5, 0.0], res["_gs"]
    assert abs(res["load_bearing_mean"] - (5 + 3 + 1) / 3) < 1e-9
    # unpriced tripwires still logged
    assert res["support_faithful_mean"] == 3.0, res["support_faithful_mean"]
    assert res["source_exists_mean"] == 4.0
    print("PASS: g is per-idea (1.0 / 0.50 / 0.00) and all four features log")


# --------------------------------------------------------------------------- #
# 3. failures impute, never penalize
# --------------------------------------------------------------------------- #
def test_grounding_failure_imputes():
    """Judge fails for idea 2 while ideas 0-1 succeed at lb=5 (g=1.0). The failed
    idea must be imputed to the set-mean of successes (1.0), NOT dropped to floor:
    an outage is missing data, not evidence of a decorative citation. Under the
    default FLOOR=0 that distinction is the difference between a no-op and zeroing
    the idea's entire contribution."""
    R._G_WIN.clear()
    ideas = [_idea("0"), _idea("1"), _idea("2")]
    # 2 retries per idea; idea2 needs both to fail -> None,None after 2 good ideas
    _patch_judge({"impact": ["Answer: A", "Answer: B"],
                  "distance": ['{"similarity": 0.5}'],
                  "grounding": [_gjson(5), _gjson(5), None, None]})
    res = _run(R._score_set(ideas, "gold", "answer", documents=DOCS))
    assert res["grounding_fail"] == 1.0, res["grounding_fail"]
    assert res["_gs"] == [1.0, 1.0, 1.0], res["_gs"]  # imputed to set-mean
    assert res["_lbs"][2] is None, res["_lbs"]
    print("PASS: grounding judge failure imputes to the set-mean, not the floor")


def test_total_grounding_blackout_uses_window():
    """Every call fails -> fall back to the trailing window of REAL g, and with an
    empty window (cold start) to GROUNDING_DEFAULT. Either way, uniform across the
    set: uniform g means the group advantage is unchanged, i.e. a GRPO no-op."""
    R._G_WIN.clear()
    R._G_WIN.extend([1.0, 1.0, 0.5, 0.5])  # window mean 0.75
    ideas = [_idea("0"), _idea("1")]
    _patch_judge({"impact": ["Answer: A", "Answer: B"],
                  "distance": ['{"similarity": 0.5}'],
                  "grounding": [None]})
    res = _run(R._score_set(ideas, "gold", "answer", documents=DOCS))
    assert res["grounding_fail"] == 2.0, res["grounding_fail"]
    assert res["_gs"] == [0.75, 0.75], res["_gs"]
    R._G_WIN.clear()
    print("PASS: total grounding blackout -> trailing-window mean, uniform")


def test_no_documents_imputes_and_is_not_a_failure():
    """A prompt whose corpus will not parse must NOT be judged against an empty
    document block (that scores every idea at floor). It imputes -- and it is not
    counted in grounding_fail, which stays a clean Bedrock-health metric."""
    R._G_WIN.clear()
    calls = _patch_judge({"impact": ["Answer: A", "Answer: B"],
                          "distance": ['{"similarity": 0.5}'],
                          "grounding": [_gjson(1)]})
    res = _run(R._score_set([_idea("0"), _idea("1")], "gold", "answer",
                            documents=[]))
    assert calls.get("grounding", 0) == 0, calls    # no call fired
    assert res["grounding_fail"] == 0.0, res["grounding_fail"]  # not a judge failure
    # cold-start window -> GROUNDING_DEFAULT, NOT GROUNDING_FLOOR: with the default
    # FLOOR=0 a floor fill would zero a set nobody ever judged.
    assert res["_gs"] == [R.GROUNDING_DEFAULT] * 2, res["_gs"]
    assert R.GROUNDING_DEFAULT > 0.0, R.GROUNDING_DEFAULT
    print("PASS: absent documents impute without inflating grounding_fail")


def test_invalid_grounding_type_gets_floor():
    """An idea that declared no valid grounding_type IS real signal -> floor, and
    it costs no judge call to know that... but we still judge (the judge sees the
    declared type and scores type_correct), so just assert the g lands at floor."""
    ideas = [_idea("0"), _idea("1", gtype="not_a_real_type")]
    _patch_judge({"impact": ["Answer: A", "Answer: B"],
                  "distance": ['{"similarity": 0.5}'],
                  "grounding": [_gjson(5), _gjson(5)]})
    res = _run(R._score_set(ideas, "gold", "answer", documents=DOCS))
    assert res["_gs"] == [1.0, R.GROUNDING_FLOOR], res["_gs"]
    print("PASS: invalid grounding_type -> floor even at load_bearing=5")


def test_grounding_default_is_not_the_floor():
    """Guards the pairing of the two constants. GROUNDING_DEFAULT is the cold-start
    IMPUTATION fill and must stay strictly above GROUNDING_FLOOR: setting FLOOR=0
    (the rubric min-max default) while filling with the floor would silently convert
    a Bedrock outage into a total grounding penalty on every set until the trailing
    window refills. Regression-only -- this is a config invariant, not behaviour."""
    assert R.GROUNDING_DEFAULT > R.GROUNDING_FLOOR, (R.GROUNDING_DEFAULT,
                                                     R.GROUNDING_FLOOR)
    R._G_WIN.clear()
    assert R._g_window_mean() == R.GROUNDING_DEFAULT
    R._G_WIN.extend([0.75, 0.25])
    assert R._g_window_mean() == 0.5      # real g wins once the window has any
    R._G_WIN.clear()
    print("PASS: GROUNDING_DEFAULT is above the floor and yields to real g")


# --------------------------------------------------------------------------- #
# 4. document parsing
# --------------------------------------------------------------------------- #
def test_parse_documents():
    prompt = (
        "You are a scientist.\n\n"
        "SOURCE DOCUMENTS\n"
        "================\n"
        "\nDocument 1:\n"
        "Title: A\nAbstract: aaa\n"
        "\nDocument 2:\n"
        "Title: B\nAbstract: bbb\n"
        "\nDocument 3:\n"
        "Title: C\nAbstract: ccc\n"
        "\nNow produce ideas."
    )
    docs = R._documents_of({"query": prompt})
    assert len(docs) == 3, docs
    assert docs[0].startswith("Title: A"), docs[0]
    assert "Title: C" in docs[2]
    # malformed / absent -> [] so the caller IMPUTES rather than judging vs nothing
    assert R._documents_of({"query": "no documents here at all"}) == []
    assert R._documents_of({}) == []
    assert R._documents_of(None) == []
    print("PASS: _documents_of splits a real prompt and returns [] when malformed")


def test_parse_documents_on_real_rollout_prompt():
    """Same parse against a real RL parquet if one is on disk (skipped if not)."""
    import glob
    import os
    parquet_glob = os.environ.get(
        "F9B_TEST_RL_PARQUET_GLOB",
        os.path.join(os.environ.get("DATA_DIR", "./data"), "rl", "train.parquet"),
    )
    hits = sorted(glob.glob(parquet_glob))
    if not hits:
        print("SKIP: RL parquet not present")
        return
    try:
        import pandas as pd
    except Exception:
        print("SKIP: pandas unavailable")
        return
    df = pd.read_parquet(hits[0]).head(20)
    counts = []
    for _, row in df.iterrows():
        ei = row["extra_info"]
        docs = R._documents_of({"query": ei["query"]})
        counts.append((len(docs), int(ei["num_documents"])))
    bad = [c for c in counts if c[0] != c[1]]
    assert not bad, f"parsed != num_documents for {len(bad)}/{len(counts)}: {bad[:5]}"
    print(f"PASS: parsed doc count == num_documents on {len(counts)}/{len(counts)} real prompts")


# --------------------------------------------------------------------------- #
# 5. key-set parity across all four return sites
# --------------------------------------------------------------------------- #
def test_all_return_sites_have_identical_keys():
    """verl stacks reward_extra_info into arrays; a key missing from ONE rollout
    makes them ragged and breaks W&B logging for the whole run."""
    ideas = [_idea(str(i)) for i in range(R.K)]
    _patch_judge({"impact": ["Answer: A", "Answer: B"],
                  "distance": ['{"similarity": 0.5}'],
                  "grounding": [_gjson(3)]})
    pub = lambda d: {k for k in d if not k.startswith("_")}

    scored = pub(_run(R._score_set(ideas, "gold", "answer", documents=DOCS)))
    imputed = pub(R._imputed_set(ideas, "answer"))

    # format-gate fail and the exception handler, via the public entry point
    bad_fmt = pub(_run(R.compute_score("ds", "not a valid rollout", "gold", {})))

    async def boom(*a, **k):
        raise RuntimeError("judge exploded")
    real_score_set = R._score_set
    good = ("<think></think><answer>" +
            json.dumps([{"problem": f"p{i}", "insight": {"claim": f"c{i}",
                        "grounding": [{"source": "s", "supports": "x"}],
                        "grounding_type": "contradiction"},
                        "motivation": f"m{i}"} for i in range(R.K)]) +
            "</answer>")
    try:
        R._score_set = boom
        errored = pub(_run(R.compute_score("ds", good, "gold", {"query": ""})))
    finally:
        R._score_set = real_score_set

    for name, keys in (("imputed", imputed), ("format-fail", bad_fmt),
                       ("exception", errored)):
        assert keys == scored, (name, sorted(scored ^ keys))
    for k in ("grounding_g_mean", "grounding_fail", "load_bearing_mean",
              "support_faithful_mean", "source_exists_mean", "type_correct_mean"):
        assert k in scored, k
    print(f"PASS: all 4 return sites emit the same {len(scored)} keys "
          "(incl. the 6 grounding keys)")


def test_mindist_cap():
    """F9B_MINDIST_CAP clamps the diversity factor (default off)."""
    assert R._capped_mindist(0.8) == 0.8   # default: MINDIST_CAP == 0 -> off
    try:
        R.MINDIST_CAP = 0.6
        assert R._capped_mindist(0.8) == 0.6
        assert R._capped_mindist(0.4) == 0.4
    finally:
        R.MINDIST_CAP = 0.0
    print("PASS: _capped_mindist is off by default and clamps when set")


if __name__ == "__main__":
    test_features_to_g_mapping()
    test_judged_claim_is_a_hook_and_v0_reads_insight_claim()
    test_si_shim_judges_the_problem_gap()
    test_no_grounding_type_omits_the_type_rubric()
    test_parse_features_matches_what_was_asked()
    test_features_to_g_saturating_variant()
    test_gate_off_is_inert()
    test_gate_scales_per_idea_not_per_set()
    test_grounding_failure_imputes()
    test_total_grounding_blackout_uses_window()
    test_no_documents_imputes_and_is_not_a_failure()
    test_invalid_grounding_type_gets_floor()
    test_grounding_default_is_not_the_floor()
    test_parse_documents()
    test_parse_documents_on_real_rollout_prompt()
    test_all_return_sites_have_identical_keys()
    test_mindist_cap()
    print("\nAll f9b v6 grounding-gate tests passed.")
