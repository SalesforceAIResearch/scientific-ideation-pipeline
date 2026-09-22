#!/usr/bin/env python3
"""Combined evaluator: run a battery of judges (novelty / impact / feasibility /
soundness / contribution / diversity) over ONE prediction file and write a single
per-idea record with all signals.

Reuses the standalone judges' loaders + scoring (no duplicated prompt/call
logic): common definitions live in grpo_training/common/, the per-idea scoring
lives in eval/{novelty_judge,pairwise_judge}.py, and this script just orchestrates
them so you get one combined output instead of several runs.

Every per-idea signal is OPT-IN. Novelty (our home-grown per-doc overlap judge)
is enabled via --with-novelty -- contribution-vs-sources / soundness already
cover idea novelty vs its sources, so a normal run leaves it off. Impact uses a
separate SciJudge client and is SKIPPED unless --with-impact is set AND a gold
idea is available per record (jun22 has none yet -> impact stays a placeholder).

Three outputs only (all derived from --out, so the --out name disambiguates runs
over the same dataset, e.g. summary_batched.jsonl vs summary.jsonl):
  (a) <out>_with_evals.jsonl  a copy of the INPUT file (same rows/order),
      each row gaining an "evals" list of its per-idea eval records
      {instance_id, idea_id, novelty:{...}, impact:{...}|null,
       feasibility:{...}|null, soundness:{...}|null, contribution:{...}|null, meta}.
  (b) <out>_with_evals.xlsx   a FLAT one-row-per-idea sheet for human eval
      (unrolled via export_evals: candidate idea text + gold + all scores as
      plain columns). Optional; skipped if openpyxl is absent.
  (c) <out>                   the run summary (a single {"summary": {...}} line).

Diversity (per-response pairwise distance among the K ideas) is opt-in via
--with-diversity. Impact stays opt-in via --with-impact.

Two more eval-only signals are opt-in:
  --with-feasibility  per idea: predicts compute + human-eval effort as two 1-5
                      tiers (feasibility_judge). Runs concurrently with
                      novelty/impact on a DEDICATED model (--feasibility-*,
                      default gateway/claude-opus-4-8), separate from the gpt-oss
                      judge -- mirroring how impact uses its own SciJudge client.
  --with-soundness    per idea: ScholarEval literature-grounded method soundness
                      (0-10, soundness_judge). SLOW (~3-4 min/idea, needs GROBID
                      on :8070); runs as its own phase after diversity and is
                      resumable via --soundness-save-root + --resume.
  --with-contribution per idea: ScholarEval contribution-vs-sources -- how novel
                      the idea is relative to the specific documents it was
                      conditioned on (avg/min novelty, -10..10, LOW = overlap;
                      contribution_source_judge). Pure LLM (~1-2 min/idea, no
                      GROBID/S2); runs as its own phase. Sources come from the
                      record's --contribution-sources-key (default
                      input_full_summaries_str).

To keep a run cheap, subset the RECORDS up front (applied once before any judge, so
novelty/diversity all see the same records):
  --domain LABEL    keep only a domain (matches the coarse `domain`, e.g. "Language",
                    or the arXiv `primary`, e.g. "cs.CL"); repeatable.
  --per-domain N    stratified cap: first N records of EACH --domain.
  --max-records N   flat cap on total records (counts records, not ideas like --limit).

Run:
  python grpo_training/eval/run_all_evals.py \
      --dataset preds.jsonl --out summary.jsonl \
      --input-format ideation-answer --backend vllm --limit 2

  # first 100 Language + first 100 Physics, with diversity:
  python grpo_training/eval/run_all_evals.py \
      --dataset preds.jsonl --out summary.jsonl --backend vllm \
      --input-format ideation-answer --with-diversity \
      --domain Language --domain Physics --per-domain 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt_dur(seconds: float) -> str:
    """Compact h/m/s duration for progress + ETA lines."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


async def _run_phase_with_eta(tag: str, coros: List[Any], *, on_result) -> List[Any]:
    """Await ``coros`` concurrently, printing a live ``[tag]`` progress + wall-clock
    ETA line as each finishes (throughput-based, self-correcting). ``on_result`` is
    called with each result the moment it lands (e.g. to log the score). Returns the
    results (order NOT preserved -- these phases key back by instance_id anyway)."""
    total = len(coros)
    t0 = time.perf_counter()
    out: List[Any] = []
    for done, fut in enumerate(asyncio.as_completed(coros), start=1):
        res = await fut
        out.append(res)
        if on_result is not None:
            on_result(res)
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(f"[{tag}] {done}/{total}  elapsed {_fmt_dur(elapsed)}  "
              f"eta ~{_fmt_dur(remaining)}  ({elapsed / done:.0f}s/idea)")
    return out

_GRPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GRPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    clients, distance, grounding, ideas, ideas_si, new_grounding, novelty,
    scijudge, soundness,
)
from eval import diversity_judge as DJ  # noqa: E402
from eval import export_evals  # noqa: E402
from eval import feasibility_judge as FJ  # noqa: E402
from eval import novelty_judge as NJ  # noqa: E402
# soundness_judge is imported LAZILY inside run_async (only when --with-soundness):
# importing it pulls in the ScholarEval project, which need not be installed for a
# plain novelty/diversity run.

DEFAULT_MODELS = {"vllm": "openai/gpt-oss-120b", "gateway": "claude-sonnet-4-6",
                  "bedrock": "openai.gpt-oss-120b-1:0"}
# Impact (SciJudge) default when routing through the gateway. Kept separate from
# DEFAULT_MODELS["gateway"] (novelty/diversity) so impact can default to a
# stronger model without changing the novelty judge.
SCIJUDGE_GATEWAY_DEFAULT = "claude-opus-4-8"
TEMPERATURE = 0.0
MAX_TOKENS = 1536
MAX_CONCURRENT = 5


async def _score_impact(cand_text: str, gold_text: str, *, sci_client, sci_model,
                        today: str, max_tokens: int) -> Dict[str, Any]:
    """Impact = SciJudge candidate-vs-gold across BOTH A/B orderings -> {0,0.5,1}.
    Returns None impact if no gold idea is available for this record. Also returns
    the rendered prompt(s) + summed token usage for per-feature metadata."""
    if not gold_text or not cand_text:
        return {"impact": None, "order1_ans": None, "order2_ans": None,
                "note": "no gold idea", "prompt": None,
                "usage": clients.usage_dict(None)}

    async def _winner(a, b):
        messages = scijudge.build_messages(a, b, today)
        try:
            resp = await sci_client.chat.completions.create(
                model=sci_model, messages=messages,
                **scijudge.sampling_kwargs(max_tokens, sci_model))
            raw = resp.choices[0].message.content or ""
            return (scijudge.parse_answer(raw), raw, clients.usage_dict(resp), messages)
        except Exception as e:
            return None, f"<exception>{type(e).__name__}: {e}</exception>", clients.usage_dict(None), messages
    # ordering 1: cand=A, gold=B -> cand wins iff 'A';  ordering 2: gold=A, cand=B -> 'B'
    ans1, raw1, u1, m1 = await _winner(cand_text, gold_text)
    ans2, raw2, u2, _ = await _winner(gold_text, cand_text)
    prompt = "\n".join(f"[{m['role']}]\n{m['content']}" for m in m1)
    # When BOTH orderings failed to yield a parseable A/B verdict (connection
    # error, truncation, malformed output), there is no signal -- record impact
    # as None so it's excluded from the mean, NOT winrate_a(None,None)=0.0 which
    # would masquerade a failed call as a real "candidate lost both" score.
    impact = (None if ans1 is None and ans2 is None
              else scijudge.winrate_a(ans1, ans2))
    return {"impact": impact,
            "order1_ans": ans1, "order2_ans": ans2,
            "order1_raw": raw1, "order2_raw": raw2,
            "prompt": prompt, "usage": clients.add_usage(u1, u2)}


def _dump_messages(tag: str, messages) -> None:
    """Print a judge's chat messages verbatim so the log shows exactly what was
    sent (used for the index-0 debug self-check)."""
    print(f"\n----- DEBUG {tag}: PROMPT SENT -----")
    for m in messages:
        print(f"[{m['role']}]\n{m['content']}")
    print(f"----- end {tag} prompt -----")


def _feasibility_idea_text(idea_card: Dict[str, Any]) -> str:
    """Render the candidate idea as the 'Problem: … Insight: … Motivation: …' card
    the standalone feasibility judge scores (ideas.Idea.card_text()), so the in-loop
    feasibility call sees the exact same input shape as feasibility_judge.py."""
    summ = idea_card.get("summary") if isinstance(idea_card, dict) else None
    summ = summ if isinstance(summ, dict) else {}
    return (
        f"Problem: {str(summ.get('problem') or '').strip()}\n"
        f"Insight: {str(summ.get('insight') or '').strip()}\n"
        f"Motivation: {str(summ.get('motivation') or '').strip()}"
    )


async def _score_idea(nov_inst: Dict[str, Any], *,
                      client, model, mode, temperature, gen_config, doc_sem,
                      gold_text: Optional[str], sci_client, sci_model, today,
                      sci_max_tokens, prices: Dict[str, Dict[str, Optional[float]]],
                      feas_client=None, feas_model: Optional[str] = None,
                      feas_gen_config: Optional[Dict[str, Any]] = None,
                      with_novelty: bool = True,
                      si: bool = False,
                      debug: bool = False) -> Dict[str, Any]:
    """Score one idea on, when enabled, our own novelty judge (gpt-oss,
    concurrent) and, if a gold idea + SciJudge client are available, impact
    (SciJudge), plus feasibility on its own client.

    When ``debug`` is set (we pass it only for the FIRST idea), the exact prompts
    sent to every judge and the parsed results are printed to the log as a
    self-check that the inputs/outputs are wired as expected."""
    refs = nov_inst.get("references", [])
    # --- novelty (reuse novelty_judge's per-doc / batched scorers) -- OPT-IN ---
    # Our home-grown novelty judge is skipped unless --with-novelty; contribution/
    # soundness cover idea-vs-source overlap without it. When skipped, per_doc stays
    # empty and novelty.aggregate([]) yields the null (all-None) novelty shape.
    do_novelty = with_novelty

    tasks = []
    if do_novelty:
        if mode == "batched":
            nov_task = NJ._score_batched(nov_inst["idea"], refs, client=client,
                                         model=model, temperature=temperature,
                                         gen_config=gen_config)
        else:
            nov_task = NJ._score_per_doc(nov_inst["idea"], refs, client=client,
                                         model=model, temperature=temperature,
                                         gen_config=gen_config, doc_sem=doc_sem)
        tasks.append(nov_task)
    # --- impact (SciJudge vs gold) only when enabled + gold present ---
    # Candidate must be the FULL idea string (motivation+problem+insight), matching
    # the gold target_idea_str shape -- NOT just the insight clause, or the judge
    # compares a fragment against a full idea and the candidate loses every time.
    cand_text = _candidate_idea_text(nov_inst["idea"], si=si)
    do_impact = sci_client is not None and bool(gold_text)
    if do_impact:
        tasks.append(_score_impact(cand_text, gold_text, sci_client=sci_client,
                                   sci_model=sci_model, today=today,
                                   max_tokens=sci_max_tokens))

    # --- feasibility (compute + human-eval tiers) on its own gateway client ---
    # Reuses feasibility_judge._score_one so there is no duplicated prompt/call
    # logic; runs concurrently with novelty/impact. Its model/client is
    # SEPARATE from the gpt-oss judge (dedicated gateway model), mirroring impact.
    do_feas = feas_client is not None
    if do_feas:
        feas_inst = {"idea_text": _feasibility_idea_text(nov_inst["idea"])}
        tasks.append(FJ._score_one(feas_inst, client=feas_client, model=feas_model,
                                   temperature=temperature,
                                   gen_config=feas_gen_config or {}))

    if debug:
        print("\n========== DEBUG: index-0 self-check ==========")
        print(f"instance_id: {nov_inst.get('instance_id')}  mode={mode}")
        # novelty prompt(s): per-doc sends one call per ref; show the first.
        if not do_novelty:
            print("NOVELTY: skipped (pass --with-novelty)")
        elif mode == "batched":
            _dump_messages("NOVELTY(batched)", novelty.build_batched_messages(nov_inst["idea"], refs))
        elif refs:
            _dump_messages("NOVELTY(per-doc, ref[0] of %d)" % len(refs),
                           novelty.build_per_doc_messages(nov_inst["idea"], refs[0]))
        else:
            print("NOVELTY: no references for this idea")
        if do_impact:
            _dump_messages("IMPACT(cand=A, gold=B)",
                           scijudge.build_messages(cand_text, gold_text, today))
        else:
            print(f"IMPACT: skipped (sci_client={sci_client is not None}, "
                  f"gold_present={bool(gold_text)})")
        if do_feas:
            _dump_messages("FEASIBILITY",
                           FJ.feas.build_judge_messages(_feasibility_idea_text(nov_inst["idea"])))
        else:
            print("FEASIBILITY: skipped (pass --with-feasibility)")

    results = await asyncio.gather(*tasks)
    # tasks were appended in order: [(novelty?), (impact?), (feasibility?)]
    idx = 0
    if do_novelty:
        per_doc = results[idx]; idx += 1
    else:
        per_doc = []  # novelty.aggregate([]) -> the null novelty shape
    impact = None
    if do_impact:
        impact = results[idx]; idx += 1
    feasibility = None
    if do_feas:
        feasibility = results[idx]; idx += 1
    nov_agg = novelty.aggregate(per_doc)
    if debug:
        print("\n----- DEBUG: PARSED RESULTS -----")
        print(f"novelty: score={nov_agg.get('novelty_score')} "
              f"max_overlap={nov_agg.get('max_overlap')} per_doc={per_doc}")
        print(f"impact: {impact}")
        print(f"feasibility: {feasibility}")
        print("========== end index-0 self-check ==========\n")

    # --- per-feature metadata: prompt input + tokens + cost ---
    # novelty: sum usage across the per-doc calls; keep each doc's prompt.
    nov_usage = clients.add_usage(*[d.get("usage") or clients.usage_dict(None)
                                    for d in per_doc]) if per_doc else clients.usage_dict(None)
    meta = {
        "novelty": _feature_meta(
            model, [d.get("prompt") for d in per_doc], nov_usage, prices.get(model)),
        "impact": (_feature_meta(
            sci_model, impact.get("prompt"), impact.get("usage"),
            prices.get(sci_model)) if impact else None),
        "feasibility": (_feature_meta(
            feas_model, feasibility.get("prompt"), feasibility.get("usage"),
            prices.get(feas_model)) if feasibility else None),
    }
    # keep records lean: prompts/usage now live in `meta`, drop the raw copies.
    per_doc_clean = [{k: v for k, v in d.items() if k not in ("prompt", "usage")}
                     for d in per_doc]
    impact_clean = ({k: v for k, v in impact.items() if k not in ("prompt", "usage")}
                    if impact else None)
    feas_clean = ({k: v for k, v in feasibility.items() if k not in ("prompt", "usage")}
                  if feasibility else None)
    return {
        "instance_id": nov_inst.get("instance_id"),
        "idea_id": nov_inst["idea"].get("id"),
        "novelty": {**nov_agg, "per_doc": per_doc_clean},
        "impact": impact_clean,
        "feasibility": feas_clean,
        "meta": meta,
    }


def _feature_meta(model, prompt, usage, price) -> Dict[str, Any]:
    """One feature's metadata: model, prompt input, token usage, dollar cost.
    `prompt` may be a single string or a list of strings (per-doc novelty).
    `price` is {'in':$/1M, 'out':$/1M} or None (local vLLM -> cost None)."""
    usage = usage or clients.usage_dict(None)
    price = price or {}
    cost = clients.token_cost(usage, price_in=price.get("in"), price_out=price.get("out"))
    return {"model": model, "prompt": prompt, "usage": usage, "cost_usd": cost}


def _strip_grounding(idea_text: str) -> str:
    """Drop the grounding citation list + grounding_type from a gold idea string,
    keeping only motivation/problem/insight. Impact compares idea SUBSTANCE; the
    candidate text (_candidate_idea_text) carries no grounding, so we strip it from
    gold too -- otherwise gold looks more substantiated (its citations read as
    rigor) and the pairwise judge skews against the candidate. Grounding quality
    is already its own metric. Truncates at the first 'grounding:'/'grounding_type:'
    line (tolerating leading '-'/whitespace)."""
    kept = []
    for line in (idea_text or "").splitlines():
        stripped = line.lstrip(" -\t").lower()
        if stripped.startswith("grounding:") or stripped.startswith("grounding_type:"):
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


def _gold_by_record(dataset: Path, *, si: bool = False) -> Dict[str, str]:
    """Map a record's id (target_arxiv_id / index / recN, the SAME prefix the
    loaders use to build instance_ids) -> its gold idea text (target_idea_str),
    rendered to match the candidate's shape. Empty map if no record carries a
    gold idea.

    v0: strip the grounding block, keeping motivation/problem/insight.
    si:  the stored gold is a flat FOUR-field block (problem_gap/core_claim/
         method/inspiration). ideas_si.render_si_card drops `inspiration` and
         renders the SAME 3-field "Problem gap:/Core claim:/Method:" card that
         SIIdea.card_text() (hence _candidate_idea_text with si=True) gives the
         candidate. Passing the stored string verbatim would show the impact
         judge a 4-field gold against a 3-field candidate -- a systematic format
         asymmetry on the one comparison that skews it toward gold; render_si_card
         exists precisely to remove it."""
    gold: Dict[str, str] = {}
    with dataset.open() as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rec_id = r.get("target_arxiv_id") or r.get("index") or f"rec{line_no}"
            g = r.get("target_idea_str")
            if g:
                if si:
                    # SI gold is a 4-field card; render_si_card drops inspiration.
                    # GiantsBench gold is a free-form <insight> with none of those
                    # labels, so render returns "" -- fall back to the raw prose
                    # rather than silently skipping every impact comparison.
                    rendered = ideas_si.render_si_card(str(g))
                    gold[str(rec_id)] = rendered if rendered else str(g).strip()
                else:
                    gold[str(rec_id)] = _strip_grounding(str(g))
    return gold


def _cutoff_by_record(dataset: Path) -> Dict[str, str]:
    """Map a record's id (same prefix the loaders use for instance_ids) -> its
    `cutoff_date`, so soundness can pass ScholarEval a per-idea literature cutoff.
    Empty/absent cutoffs are skipped (ScholarEval treats None as no cutoff)."""
    cutoff: Dict[str, str] = {}
    with dataset.open() as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rec_id = r.get("target_arxiv_id") or r.get("index") or f"rec{line_no}"
            c = r.get("cutoff_date")
            if c:
                cutoff[str(rec_id)] = str(c)
    return cutoff


def _sources_by_record(dataset: Path, sources_key: str) -> Dict[str, List[Any]]:
    """Map a record's id (same prefix the loaders use for instance_ids) -> the
    list of source documents the idea was conditioned on, so the contribution
    judge can compare each idea against exactly its own sources. ``sources_key``
    is the record field holding them (default input_full_summaries_str: a list of
    per-document summary strings). A bare string is wrapped in a one-element list;
    records with no sources are omitted (contribution is skipped for them)."""
    srcs: Dict[str, List[Any]] = {}
    with dataset.open() as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rec_id = r.get("target_arxiv_id") or r.get("index") or f"rec{line_no}"
            v = r.get(sources_key)
            if isinstance(v, list) and v:
                srcs[str(rec_id)] = v
            elif isinstance(v, str) and v.strip():
                srcs[str(rec_id)] = [v]
    return srcs


def _ideas_by_iid(dataset: Path, k: Optional[int], *, si: bool = False) -> Dict[str, Any]:
    """Map instance_id -> the parsed `ideas.Idea`, re-parsing the dataset with the
    SAME instance_id scheme as NJ._load_ideation_answers.

    The per-idea loop above scores `idea_to_eval_card` cards, which DROP the
    grounding list / grounding_type / raw dict. The grounding + pointwise-soundness
    phases need those, so we recover the full Idea object here (cheap: one re-parse,
    keyed identically so results merge back 1:1).

    `si=True` uses the SI parser (ideas_si.parse_ideas_si) so v1 cards
    {problem_gap, core_claim, method, inspiration} parse to SIIdea rather than
    None -- must match the parser NJ.load_instances used, or the keys won't merge."""
    parse = ideas_si.parse_ideas_si if si else ideas.parse_ideas
    out: Dict[str, Any] = {}
    with dataset.open() as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            answers, raws = rec.get("answer_str"), rec.get("raw_output")
            gens: List[str] = []
            for src in (answers, raws):
                if isinstance(src, list):
                    gens.extend([s for s in src if isinstance(s, str)])
                    break
                if isinstance(src, str):
                    gens.append(src)
                    break
            rec_id = rec.get("target_arxiv_id") or rec.get("index") or f"rec{line_no}"
            for s_idx, gen in enumerate(gens):
                for i_idx, idea in enumerate(parse(gen, k=k)):
                    out[f"{rec_id}::s{s_idx}::idea{i_idx}"] = idea
    return out


async def _score_grounding(idea, documents: List[Any], *, client, model: str,
                           gen_config: Dict[str, Any], require_type: bool,
                           thinking_budget: Optional[int] = None,
                           ) -> Dict[str, Any]:
    """Grounding fidelity for one idea via common.grounding. Returns the
    load-bearing value + its [0,1] multiplier `g` (features_to_g) + graded
    `overall` + raw features. `g` is the load-bearing headline the report shows.
    None-valued on a persistent judge failure (excluded from means, never
    defaulted -- same policy as diversity/novelty).

    SI ideas (ideas_si.SIIdea) never populate the base Idea's `grounding` list
    (SI has no structured citation list -- `inspiration` carries the provenance
    prose instead), so idea.grounding_text() always renders "(no grounding
    provided)" for them and idea.insight_claim (core_claim) is not the field
    the SI reward gate judges either. Mirror the SI training reward's rebind
    (grpo_training/rewards/f9b_impdist_len_v6_si_bedrock.py:
    _judged_claim_of -> idea.problem_gap, _grounding_text_of ->
    idea.provenance_text(), asserted by test_f9b_v6_grounding.py) instead of
    silently feeding the v0 defaults to every SI idea.

    ``thinking_budget``: Gemini only. None = API default/dynamic think; 0 = off;
    N = hard cap. Ignored for non-gemini models."""
    si_idea = isinstance(idea, ideas_si.SIIdea)
    judged_claim = idea.problem_gap if si_idea else idea.insight_claim
    grounding_src = idea.provenance_text() if si_idea else idea.grounding_text()
    gtype = idea.grounding_type
    # ask_type mirrors build_messages' default: ask for type_correct iff a
    # grounding_type was actually declared, and parse it with the same strictness.
    ask_type = gtype is not None and str(gtype).strip() != ""
    messages = grounding.build_messages(
        judged_claim, grounding_src, gtype,
        [str(d) for d in (documents or [])], ask_type=ask_type)
    call = {"model": model, "messages": messages, **gen_config}
    if clients.accepts_temperature(model):
        call["temperature"] = 0.0
    call.update(new_grounding.gemini_thinking_extra(model, thinking_budget))
    err = None
    for _ in range(2):
        try:
            resp = await client.chat.completions.create(**call)
            raw = resp.choices[0].message.content or ""
            obj = grounding.extract_json(raw)
            feats = (grounding.parse_features(obj, require_type=ask_type)
                     if isinstance(obj, dict) else None)
            if feats is not None:
                return {
                    "load_bearing": feats.get("load_bearing"),
                    "g": grounding.features_to_g(feats, gtype, require_type=require_type),
                    "overall": grounding.overall_score(feats, gtype),
                    "fidelity": grounding.fidelity(feats),
                    "grounding_type": gtype, "features": feats, "error": None,
                    "claim_field": "problem_gap" if si_idea else "insight_claim",
                }
            err = "unparsed"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return {"load_bearing": None, "g": None, "overall": None, "fidelity": None,
            "grounding_type": gtype, "features": None, "error": err,
            "claim_field": "problem_gap" if si_idea else "insight_claim"}


async def _score_new_grounding(
    idea, documents: List[Any], *, client, model: str,
    gen_config: Dict[str, Any], require_type: bool,
    lb_thinking_budget: Optional[int],
    sf_thinking_budget: Optional[int],
    with_sf: bool = True,
) -> Dict[str, Any]:
    """Dual-head reporting judge (common.new_grounding).

    Head A: LB_ONLY (no docs) -> load_bearing.
    Head B: SF_ONLY vagueness anchors (with docs) -> support_faithful.
    `source_exists` is not scored -- see new_grounding's docstring. Combined into
    the same result shape as _score_grounding.

    `with_sf=False` runs head A ONLY. `g` is computed from load_bearing alone
    (grounding.features_to_g), so head B never moves the gate -- it is pure
    reporting, and on SI cards it is also the expensive head (docs in context:
    ~3.2k prompt tokens vs head A's ~0.7k). Skipping it halves the grounding call
    count and cuts ~80% of its tokens.
    """
    si_idea = isinstance(idea, ideas_si.SIIdea)
    judged_claim = idea.problem_gap if si_idea else idea.insight_claim
    grounding_src = idea.provenance_text() if si_idea else idea.grounding_text()
    gtype = idea.grounding_type
    ask_type = gtype is not None and str(gtype).strip() != ""
    docs = [str(d) for d in (documents or [])]
    claim_field = "problem_gap" if si_idea else "insight_claim"

    async def _one(messages, thinking_budget: Optional[int]):
        call: Dict[str, Any] = {"model": model, "messages": messages, **gen_config}
        if clients.accepts_temperature(model):
            call["temperature"] = 0.0
        call.update(new_grounding.gemini_thinking_extra(model, thinking_budget))
        err = None
        for _ in range(2):
            try:
                resp = await client.chat.completions.create(**call)
                return resp.choices[0].message.content or "", None
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
        return "", err

    lb_msgs = new_grounding.build_lb_only_messages(judged_claim, grounding_src)

    # Sequential so --max-concurrent ≈ gateway concurrency (not 2x).
    lb_raw, lb_err = await _one(lb_msgs, lb_thinking_budget)
    lb = new_grounding.parse_lb_only(lb_raw) if lb_raw else None
    if lb is None and lb_raw and not lb_err:
        lb_err = "unparsed"

    faith = faith_err = None
    if with_sf:
        faith_msgs = new_grounding.build_faithfulness_messages(
            judged_claim, grounding_src, docs, gtype, ask_type=ask_type)
        faith_raw, faith_err = await _one(faith_msgs, sf_thinking_budget)
        faith = (new_grounding.parse_faithfulness(faith_raw, require_type=ask_type)
                 if faith_raw else None)
        if faith is None and faith_raw and not faith_err:
            faith_err = "unparsed"

    out = new_grounding.combine_heads(
        lb, faith, gtype, require_type=require_type,
        lb_error=lb_err, faith_error=faith_err, sf_requested=with_sf)
    out["claim_field"] = claim_field
    return out


async def _score_soundness_pointwise(idea, *, client, model: str, max_tokens: int
                                     ) -> Dict[str, Any]:
    """Pointwise soundness (1-4 -> [0,1]) for one idea via common.soundness.
    Uses the idea's raw dict so render_idea_to_slots sees the real schema fields.
    `reward` in [0,1] is the headline; None on persistent failure."""
    hypothesis, experiment = soundness.render_idea_to_slots(idea.raw or {})
    messages = soundness.build_messages(hypothesis, experiment)
    call = {"model": model, "messages": messages,
            **soundness.sampling_kwargs(max_tokens, 0.0, model)}
    err = None
    for _ in range(2):
        try:
            resp = await client.chat.completions.create(**call)
            raw = resp.choices[0].message.content or ""
            rating = soundness.parse_rating(raw)
            if rating is not None:
                return {"rating": rating,
                        "reward": soundness.rating_to_reward(rating), "error": None}
            err = "unparsed"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return {"rating": None, "reward": None, "error": err}


def _rec_id_of(instance_id: str) -> str:
    """instance_id is '<rec_id>::s<n>::idea<n>' -> recover '<rec_id>'."""
    return (instance_id or "").split("::", 1)[0]


def _sample_idx_of(instance_id: str) -> int:
    """instance_id is '<rec_id>::s<n>::idea<n>' -> recover the sample index <n>.
    Defaults to 0 when there is no '::s' segment."""
    for part in (instance_id or "").split("::"):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])
    return 0


def _candidate_idea_text(idea_card: Dict[str, Any], *, si: bool = False) -> str:
    """Render the candidate idea as a full multi-field string for impact,
    mirroring the gold target_idea_str shape so SciJudge compares two complete
    ideas (not a fragment vs. a full idea). Falls back to whatever fields present.

    The idea_card summary always carries v0-named slots (problem/insight/
    motivation) -- for SI, idea_to_eval_card mapped SIIdea's gap/claim/method onto
    them. In SI mode we render them under the SAME SI labels ("Problem gap:/Core
    claim:/Method:") that ideas_si.render_si_card gives the gold, so both sides of
    the pairwise judge are byte-for-byte format-identical (label set AND casing).
    A label/casing mismatch between the two sides is itself a judge signal, so
    this symmetry is load-bearing, not cosmetic."""
    summ = idea_card.get("summary") if isinstance(idea_card, dict) else None
    summ = summ if isinstance(summ, dict) else {}
    # (display label, summary key) -- keys are v0-named in both modes.
    fields = (("Problem gap", "problem"), ("Core claim", "insight"),
              ("Method", "motivation")) if si else \
             (("motivation", "motivation"), ("problem", "problem"),
              ("insight", "insight"))
    parts = []
    for label, key in fields:
        v = str(summ.get(key) or "").strip()
        if v:
            parts.append(f"{label}: {v}")
    return "\n".join(parts)


def _filter_dataset(dataset: Path, *, domains: Optional[List[str]],
                    per_domain: Optional[int], max_records: Optional[int]
                    ) -> tuple[Path, Optional[Path], int, int, Dict[str, int]]:
    """Pre-filter the prediction JSONL to a RECORD subset BEFORE any loader touches
    it, so novelty/diversity/gold all see the identical records (their
    instance_ids stay aligned).

    Selection, applied in order:
      --domain LABEL   keep only records whose coarse `domain` (e.g. "Language")
                       OR arXiv `primary` (e.g. "cs.CL") matches, case-insensitive.
                       Repeatable to keep several domains.
      --per-domain N   cap EACH selected domain to its first N records. This is the
                       stratified knob: `--domain Language --domain Physics
                       --per-domain 100` -> first 100 Language + first 100 Physics.
                       Requires --domain (a per-domain cap over "everything" is
                       ambiguous; use --max-records for a flat cap).
      --max-records N  a final flat cap on the total kept (applied after the above).

    When nothing is set, returns the original path untouched (no temp file). Returns
    (path_to_use, tmp_path_or_None, n_total, n_kept, kept_by_domain)."""
    if not domains and per_domain is None and max_records is None:
        n = sum(1 for line in dataset.open() if line.strip())
        return dataset, None, n, n, {}

    wanted = {d.strip().lower() for d in (domains or [])}
    # Track which selector each kept record counts against so --per-domain can cap
    # each domain independently. A record is bucketed under the wanted label it
    # matched (domain preferred over primary); with no --domain, a single "*" bucket.
    kept: List[str] = []
    per_bucket: Dict[str, int] = {}
    kept_by_domain: Dict[str, int] = {}
    n_total = 0
    with dataset.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            r = json.loads(line)
            dom = str(r.get("domain") or "").lower()
            prim = str(r.get("primary") or "").lower()
            if wanted:
                labels = {dom, prim} & wanted
                if not labels:
                    continue
                bucket = dom if dom in wanted else next(iter(labels))
            else:
                bucket = "*"
            if per_domain is not None and per_bucket.get(bucket, 0) >= per_domain:
                continue
            kept.append(line)
            per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
            report_key = str(r.get("domain") or r.get("primary") or "?")
            kept_by_domain[report_key] = kept_by_domain.get(report_key, 0) + 1
            if max_records is not None and len(kept) >= max_records:
                break

    tmp = Path(tempfile.mkstemp(prefix="run_all_evals_filtered_", suffix=".jsonl")[1])
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""))
    return tmp, tmp, n_total, len(kept), kept_by_domain


async def run_async(dataset: Path, out: Path, *, backend: str, model: str,
                    mode: str, input_format: str, temperature: float,
                    max_tokens: int, max_concurrent: int, k: Optional[int],
                    limit: Optional[int], with_impact: bool, with_diversity: bool,
                    with_novelty: bool = True,
                    domains: Optional[List[str]], per_domain: Optional[int],
                    max_records: Optional[int],
                    judge_url: Optional[str], scijudge_url: Optional[str],
                    scijudge_model: Optional[str], scijudge_max_tokens: int,
                    scijudge_backend: str = "vllm",
                    with_feasibility: bool = False,
                    feasibility_backend: str = "gateway",
                    feasibility_model: Optional[str] = None,
                    feasibility_url: Optional[str] = None,
                    with_soundness: bool = False,
                    soundness_model: Optional[str] = None,
                    soundness_save_root: Optional[Path] = None,
                    soundness_concurrency: int = 3,
                    with_contribution: bool = False,
                    contribution_model: Optional[str] = None,
                    contribution_concurrency: int = 4,
                    contribution_sources_key: str = "input_full_summaries_str",
                    with_grounding: bool = False,
                    grounding_backend: str = "vllm",
                    grounding_model: Optional[str] = None,
                    grounding_url: Optional[str] = None,
                    grounding_max_tokens: int = 1536,
                    grounding_sources_key: str = "input_full_summaries_str",
                    grounding_require_type: bool = False,
                    grounding_thinking_budget: Optional[int] = None,
                    with_new_grounding: bool = False,
                    new_grounding_lb_thinking_budget: Optional[int] = 0,
                    new_grounding_sf_thinking_budget: Optional[int] = None,
                    new_grounding_with_sf: bool = True,
                    with_soundness_pointwise: bool = False,
                    soundness_pw_backend: str = "gateway",
                    soundness_pw_model: Optional[str] = None,
                    soundness_pw_url: Optional[str] = None,
                    soundness_pw_max_tokens: int = 6144,
                    soundness_pw_concurrency: int = 5,
                    prices: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
                    checkpoint: Optional[Path] = None, resume: bool = False,
                    keep_checkpoints: bool = False) -> dict:
    prices = prices or {}
    # Record-level pre-filter (domain slice + per-domain / flat cap). Every loader
    # below reads `dataset`, so filtering once here keeps all judges on the SAME
    # record subset (their instance_ids stay aligned).
    dataset, _tmp_ds, n_total_recs, n_kept_recs, kept_by_domain = _filter_dataset(
        dataset, domains=domains, per_domain=per_domain, max_records=max_records)
    if domains or per_domain is not None or max_records is not None:
        print(f"[filter] domains={domains or 'any'} per_domain={per_domain} "
              f"max_records={max_records}: kept {n_kept_recs}/{n_total_recs} records "
              f"{kept_by_domain}")
    # Load the ideas via the novelty loader (drives the per-idea loop).
    nov_insts = NJ.load_instances(dataset, input_format=input_format, k=k)
    if limit is not None:
        nov_insts = nov_insts[:limit]
    # instance_id -> novelty instance, so later phases (soundness) can recover the
    # idea card without re-parsing the dataset.
    nov_by_iid = {n["instance_id"]: n for n in nov_insts}

    out.parent.mkdir(parents=True, exist_ok=True)
    # gpt-oss client for novelty + diversity. judge_url overrides discovery.
    # Only built when actually needed: novelty or diversity (both opt-in).
    # An impact-only run (--with-impact, no --with-diversity) needs no gpt-oss endpoint,
    # so don't force one -- otherwise resolve_base_url raises "No judge endpoint"
    # for a judge we never call.
    need_gpt_judge = with_novelty or with_diversity
    client = (clients.build_async_client(backend, model, base_url=judge_url)
              if need_gpt_judge else None)
    gen_config = clients.generation_config(model, max_tokens)

    # SI (v1) format: gold + candidate must render as the SAME 3-field SI card
    # (Problem gap/Core claim/Method), or the pairwise impact judge sees a
    # 4-field/lowercase gold vs a 3-field candidate and skews toward gold.
    si_format = input_format == "ideation-si"

    # Optional SciJudge client for impact-vs-gold.
    gold_map: Dict[str, str] = (_gold_by_record(dataset, si=si_format)
                                if with_impact else {})
    sci_client = sci_model = None
    today = datetime.now().strftime("%Y-%m-%d")
    if with_impact:
        if not gold_map:
            print("[impact] --with-impact set but NO record has target_idea_str; "
                  "impact will be null for every idea.")
        # Impact judge model: default per backend -- the self-hosted SciJudge-30B
        # on vllm, or a hosted gpt-/claude- model when routing through the gateway.
        if scijudge_model:
            sci_model = scijudge_model
        elif scijudge_backend == "gateway":
            sci_model = SCIJUDGE_GATEWAY_DEFAULT
        else:
            sci_model = scijudge.DEFAULT_MODEL
        # vllm: scijudge_url overrides the endpoint-file discovery.
        # gateway: an LLM API (gpt-/claude-), routed + keyed by common.gateway;
        # scijudge_url overrides the default gateway base URL for that family.
        sci_client = clients.build_async_client(scijudge_backend, sci_model,
                                                base_url=scijudge_url)

    # Optional feasibility client (dedicated model, separate from the gpt-oss judge).
    feas_client = feas_model = feas_gen_config = None
    if with_feasibility:
        feas_model = feasibility_model or FJ.DEFAULT_MODELS.get(feasibility_backend)
        feas_client = clients.build_async_client(feasibility_backend, feas_model,
                                                 base_url=feasibility_url)
        feas_gen_config = clients.generation_config(feas_model, FJ.MAX_TOKENS)

    sem = asyncio.Semaphore(max_concurrent)
    doc_sem = asyncio.Semaphore(max_concurrent)

    # --- incremental checkpoint: append each per-idea record the moment it's
    # scored, so a crash AFTER scoring (e.g. a bad --out path at the final write)
    # never throws away the judge calls. Default: <out-stem>.ckpt.jsonl. With
    # resume=True, ideas already in the checkpoint are reused (skips re-scoring).
    ckpt = checkpoint or out.with_name(out.stem + ".ckpt.jsonl")
    done_by_iid: Dict[str, Dict[str, Any]] = {}
    if resume and ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            iid = r.get("instance_id")
            if iid:
                done_by_iid[iid] = r
        print(f"[ckpt] resume: {len(done_by_iid)} idea(s) already scored in {ckpt}")
    else:
        # fresh run: start a clean checkpoint (truncate any stale file).
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("", encoding="utf-8")
    ckpt_lock = asyncio.Lock()

    async def one(idx: int, nov_inst: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            iid = nov_inst.get("instance_id") or f"#{idx}"
            cached = done_by_iid.get(iid)
            if cached is not None:
                print(f"[{idx}/{len(nov_insts)}] {iid}  (cached)")
                return cached
            gold_text = gold_map.get(_rec_id_of(iid)) if with_impact else None
            print(f"[{idx}/{len(nov_insts)}] {iid}")
            rec = await _score_idea(
                nov_inst, client=client, model=model, mode=mode,
                temperature=temperature, gen_config=gen_config, doc_sem=doc_sem,
                gold_text=gold_text, sci_client=sci_client, sci_model=sci_model,
                today=today, sci_max_tokens=scijudge_max_tokens, prices=prices,
                feas_client=feas_client, feas_model=feas_model,
                feas_gen_config=feas_gen_config, with_novelty=with_novelty,
                si=si_format, debug=(idx == 1))
            ns = rec["novelty"]["novelty_score"]
            imp = rec["impact"]["impact"] if rec.get("impact") else None
            feas = rec.get("feasibility") or {}
            print(f"    novelty={ns if ns is None else round(ns,2)} "
                  f"impact={imp} feas_compute={feas.get('compute_tier')} "
                  f"feas_human={feas.get('human_eval_tier')}")
            out_rec = {"model": model, "backend": backend, "mode": mode,
                       "timestamp": datetime.now().isoformat(), **rec}
            # append to the checkpoint before returning (one JSON line per idea).
            async with ckpt_lock:
                with ckpt.open("a", encoding="utf-8") as cf:
                    cf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            return out_rec

    records = [await t for t in
               [asyncio.create_task(one(i + 1, n)) for i, n in enumerate(nov_insts)]]

    # --- diversity (set-level: the K ideas of one response vs. each other) ---
    # Unlike novelty/impact (per idea), diversity groups a response's
    # ideas and scores all C(K,2) pairs with the shared distance.py rubric. Reuses
    # the SAME gpt-oss client, so no extra endpoint. Opt-in via --with-diversity.
    div_records: List[Dict[str, Any]] = []
    div_by_rec_sample: Dict[str, Dict[int, Dict[str, Any]]] = {}
    if with_diversity:
        # ONE si flag for both calls below: _load_groups picks the card PARSER and
        # score_group picks the RUBRIC, and they have to agree. Passing si only to
        # the parser (the pre-2026-08-13 behaviour) rendered SI cards and then
        # scored them under common/distance.py's v0 rubric -- see
        # diversity_judge.rubric_for for why that fails silently.
        div_si = (input_format == "ideation-si")
        groups = DJ._load_groups(dataset, k=k, si=div_si)
        if limit is not None:
            # keep only responses whose record was actually scored above, so the
            # diversity rows align with the (limited) per-idea rows.
            scored_recs = {_rec_id_of(r.get("instance_id") or "") for r in records}
            groups = [g for g in groups if g["rec_id"] in scored_recs]
        pair_sem = asyncio.Semaphore(max_concurrent)
        # diversity checkpoint, parallel to the per-idea one (keyed by rec::sample).
        div_ckpt = ckpt.with_name(ckpt.stem + ".div" + ckpt.suffix)
        div_rubric = DJ.rubric_for(div_si)[0].RUBRIC_VERSION
        done_div: Dict[str, Dict[str, Any]] = {}
        if resume and div_ckpt.exists():
            stale_div = 0
            for line in div_ckpt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                # A checkpoint row is only reusable if it was scored under the SAME
                # rubric we are running now. Rows written before the SI-rubric fix
                # carry distance-pairwise-v2 while an SI run wants
                # distance-pairwise-si-v1; resuming onto them would mix two
                # incomparable distance scales inside one model's mean and, because
                # both parse cleanly, leave no trace in the output.
                if (r.get("diversity") or {}).get("rubric_version") != div_rubric:
                    stale_div += 1
                    continue
                done_div[f"{r.get('instance_id')}::s{r.get('sample_idx')}"] = r
            print(f"[ckpt] resume: {len(done_div)} diversity row(s) already scored"
                  + (f"; {stale_div} discarded (rubric != {div_rubric})" if stale_div else ""))
        else:
            div_ckpt.write_text("", encoding="utf-8")
        div_lock = asyncio.Lock()

        async def one_div(idx: int, group: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                dkey = f"{group['rec_id']}::s{group['sample_idx']}"
                cached = done_div.get(dkey)
                if cached is not None:
                    print(f"[div {idx}/{len(groups)}] {dkey}  (cached)")
                    return cached
                rec = await DJ.score_group(
                    group, client=client, model=model, temperature=temperature,
                    gen_config=gen_config, pair_sem=pair_sem, prices=prices,
                    # keep_pairs=True so the raw per-pair 1-5 features are persisted
                    # for downstream distribution analysis, not just the aggregated
                    # mean/min/max/by_field. Keys are the rubric's:
                    # problem_sim/insight_sim/motivation_sim under v0,
                    # problem_gap_sim/core_claim_sim/method_sim under SI.
                    keep_pairs=True, si=div_si)
                d = rec["diversity"]
                mp = d["mean_pairwise"]
                print(f"[div {idx}/{len(groups)}] {group['rec_id']}::s{group['sample_idx']}"
                      f"  ideas={d['n_ideas']} mean_div="
                      f"{mp if mp is None else round(mp, 3)}")
                async with div_lock:
                    with div_ckpt.open("a", encoding="utf-8") as cf:
                        cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return rec

        div_records = [await t for t in
                       [asyncio.create_task(one_div(i + 1, g))
                        for i, g in enumerate(groups)]]
        for r in div_records:
            div_by_rec_sample.setdefault(r["instance_id"], {})[r["sample_idx"]] = r

    # --- grounding fidelity ---
    # Opt-in via --with-grounding (production common.grounding) OR
    # --with-new-grounding (dual-head common.new_grounding for reporting).
    # Idea objects are recovered by re-parsing the dataset.
    do_grounding = with_grounding or with_new_grounding
    if do_grounding or with_soundness_pointwise:
        idea_by_iid = _ideas_by_iid(dataset, k, si=(input_format == "ideation-si"))
    grd_model = None
    if do_grounding:
        if with_grounding and with_new_grounding:
            print("[grounding] both --with-grounding and --with-new-grounding set; "
                  "using dual-head new_grounding")
        use_new = with_new_grounding
        grd_model = grounding_model or DEFAULT_MODELS.get(grounding_backend)
        grd_client = clients.build_async_client(grounding_backend, grd_model,
                                                base_url=grounding_url)
        grd_gen_config = clients.generation_config(grd_model, grounding_max_tokens)
        sources_by_rec = _sources_by_record(dataset, grounding_sources_key)
        grd_ckpt = ckpt.with_name(
            ckpt.stem + (".grdnew" if use_new else ".grd") + ckpt.suffix)
        # Which claim field THIS run's ideas are judged on (_score_grounding's
        # si_idea branch) -- ideas in one run are uniformly SI or v0, so this is
        # a single expected tag. A cached row with no/mismatched "claim_field"
        # predates the SI grounding-field fix (it silently scored SI ideas'
        # insight_claim against "(no grounding provided)", see _score_grounding's
        # docstring) and must be discarded, not resumed from, mirroring the
        # diversity checkpoint's rubric_version staleness check above.
        grd_claim_field = "problem_gap" if input_format == "ideation-si" else "insight_claim"
        expect_rubric = (new_grounding.RUBRIC_VERSION if use_new
                         else grounding.RUBRIC_VERSION)
        done_grd: Dict[str, Dict[str, Any]] = {}
        if resume and grd_ckpt.exists():
            stale_grd = 0
            for line in grd_ckpt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not r.get("instance_id"):
                    continue
                result = r.get("result", r)
                if result.get("claim_field") != grd_claim_field:
                    stale_grd += 1
                    continue
                got_rubric = result.get("rubric_version") or grounding.RUBRIC_VERSION
                if use_new and got_rubric != expect_rubric:
                    stale_grd += 1
                    continue
                # A row scored WITHOUT head B carries no support_faithful, and the
                # rubric_version can't tell the two apart (same prompts, one fewer
                # call). Resuming an --new-grounding-no-sf checkpoint into an SF run
                # would silently leave SF null on every resumed idea, so treat the
                # head set as part of staleness. The reverse (SF row resumed into a
                # no-SF run) is harmless -- it's a superset -- so only reject the
                # direction that loses data.
                if use_new and new_grounding_with_sf:
                    head_b = (result.get("heads") or {}).get("faithfulness_vagueness") or {}
                    if head_b.get("skipped"):
                        stale_grd += 1
                        continue
                done_grd[r["instance_id"]] = result
            print(f"[ckpt] resume: {len(done_grd)} grounding idea(s) already scored"
                  + (f"; {stale_grd} discarded (claim_field/rubric/head-set stale)"
                     if stale_grd else ""))
        else:
            grd_ckpt.write_text("", encoding="utf-8")
        grd_sem = asyncio.Semaphore(max_concurrent)
        grd_lock = asyncio.Lock()
        head_label = (
            (f"new_grounding {'dual-head' if new_grounding_with_sf else 'LB-ONLY'} "
             f"(LB think={new_grounding_lb_thinking_budget}"
             + (f", SF think={new_grounding_sf_thinking_budget}"
                if new_grounding_with_sf else ", head B skipped") + ")")
            if use_new
            else f"common.grounding (think={grounding_thinking_budget})")
        print(f"[grounding] scoring {len(records)} idea(s) with {grd_model}; "
              f"sources_key={grounding_sources_key}; rubric={head_label}")

        async def _score_grd(r):
            iid = r["instance_id"]
            cached = done_grd.get(iid)
            if cached is not None:
                return iid, cached
            idea = idea_by_iid.get(iid)
            if idea is None:
                return iid, {"load_bearing": None, "g": None, "overall": None,
                             "fidelity": None, "grounding_type": None,
                             "features": None, "error": "idea not recovered",
                             "claim_field": grd_claim_field,
                             "rubric_version": expect_rubric}
            docs = sources_by_rec.get(_rec_id_of(iid or ""), [])
            async with grd_sem:
                if use_new:
                    res = await _score_new_grounding(
                        idea, docs, client=grd_client, model=grd_model,
                        gen_config=grd_gen_config,
                        require_type=grounding_require_type,
                        lb_thinking_budget=new_grounding_lb_thinking_budget,
                        sf_thinking_budget=new_grounding_sf_thinking_budget,
                        with_sf=new_grounding_with_sf)
                else:
                    res = await _score_grounding(
                        idea, docs, client=grd_client, model=grd_model,
                        gen_config=grd_gen_config,
                        require_type=grounding_require_type,
                        thinking_budget=grounding_thinking_budget)
                    res.setdefault("rubric_version", grounding.RUBRIC_VERSION)
            async with grd_lock:
                with grd_ckpt.open("a", encoding="utf-8") as cf:
                    cf.write(json.dumps({"instance_id": iid, "result": res},
                                        ensure_ascii=False) + "\n")
            return iid, res

        grd_by_iid: Dict[str, Dict[str, Any]] = dict(done_grd)

        def _log_grd(item):
            iid, res = item
            grd_by_iid[iid] = res
            feats = res.get("features") or {}
            print(f"[grounding] {iid}  load_bearing={res.get('load_bearing')} "
                  f"sf={feats.get('support_faithful')} g={res.get('g')}"
                  + (f"  ERR({res['error']})" if res.get("error") else ""))

        await _run_phase_with_eta("grounding",
                                  [_score_grd(r) for r in records],
                                  on_result=_log_grd)
        for r in records:
            r["grounding"] = grd_by_iid.get(r["instance_id"])
            if r.get("grounding") is not None:
                r.setdefault("meta", {})["grounding"] = {
                    "model": grd_model,
                    "rubric_version": expect_rubric,
                    "variant": "new_grounding" if use_new else "grounding",
                }

    # --- pointwise soundness (common.soundness: scale-4 rating -> [0,1]) ---
    # Per idea, own phase, merged back by instance_id. DISTINCT from the ScholarEval
    # --with-soundness path below: this is the validated pointwise judge
    # (best on SoundnessBench = claude-sonnet-5), the same one wired into the reward.
    # Opt-in via --with-soundness-pointwise.
    snd_pw_model = None
    if with_soundness_pointwise:
        snd_pw_model = soundness_pw_model or soundness.DEFAULT_MODEL
        snd_pw_client = clients.build_async_client(soundness_pw_backend, snd_pw_model,
                                                   base_url=soundness_pw_url)
        snd_pw_ckpt = ckpt.with_name(ckpt.stem + ".sndpw" + ckpt.suffix)
        done_snd_pw: Dict[str, Dict[str, Any]] = {}
        if resume and snd_pw_ckpt.exists():
            for line in snd_pw_ckpt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("instance_id"):
                    done_snd_pw[r["instance_id"]] = r.get("result", r)
            print(f"[ckpt] resume: {len(done_snd_pw)} pointwise-soundness idea(s) already scored")
        else:
            snd_pw_ckpt.write_text("", encoding="utf-8")
        snd_pw_sem = asyncio.Semaphore(max(1, soundness_pw_concurrency))
        snd_pw_lock = asyncio.Lock()
        print(f"[soundness-pw] scoring {len(records)} idea(s) with {snd_pw_model}")

        async def _score_snd_pw(r):
            iid = r["instance_id"]
            cached = done_snd_pw.get(iid)
            if cached is not None:
                return iid, cached
            idea = idea_by_iid.get(iid)
            if idea is None:
                return iid, {"rating": None, "reward": None,
                             "error": "idea not recovered"}
            async with snd_pw_sem:
                res = await _score_soundness_pointwise(
                    idea, client=snd_pw_client, model=snd_pw_model,
                    max_tokens=soundness_pw_max_tokens)
            async with snd_pw_lock:
                with snd_pw_ckpt.open("a", encoding="utf-8") as cf:
                    cf.write(json.dumps({"instance_id": iid, "result": res},
                                        ensure_ascii=False) + "\n")
            return iid, res

        snd_pw_by_iid: Dict[str, Dict[str, Any]] = dict(done_snd_pw)

        def _log_snd_pw(item):
            iid, res = item
            snd_pw_by_iid[iid] = res
            print(f"[soundness-pw] {iid}  rating={res.get('rating')} "
                  f"reward={res.get('reward')}"
                  + (f"  ERR({res['error']})" if res.get("error") else ""))

        await _run_phase_with_eta("soundness-pw",
                                  [_score_snd_pw(r) for r in records],
                                  on_result=_log_snd_pw)
        for r in records:
            r["soundness_pointwise"] = snd_pw_by_iid.get(r["instance_id"])
            if r.get("soundness_pointwise") is not None:
                r.setdefault("meta", {})["soundness_pointwise"] = {"model": snd_pw_model}

    # --- soundness (ScholarEval: literature-grounded method soundness) ---
    # Per idea, but SEPARATE from the async gather above: ScholarEval is blocking
    # (subprocess + GROBID PDF parsing, ~3-4 min/idea) with its own thread pool,
    # save-root, and resume, so we run it as its own phase and merge results back
    # onto the per-idea records by instance_id. Opt-in via --with-soundness.
    snd_by_iid: Dict[str, Dict[str, Any]] = {}
    snd_model = None
    if with_soundness:
        from eval import soundness_judge as SJ  # lazy: pulls in ScholarEval
        if not SJ._grobid_alive():
            raise RuntimeError(
                f"GROBID is not responding at {SJ.GROBID_URL}. Start it first:\n"
                f"    bash {SJ._SCHOLAREVAL_ROOT}/scripts/start_grobid.sh")
        snd_model = soundness_model or SJ.DEFAULT_MODEL
        snd_root = soundness_save_root or out.with_name(out.stem + "_soundness")
        snd_root.mkdir(parents=True, exist_ok=True)
        # Build one soundness instance per already-scored idea, carrying the same
        # instance_id so results merge back 1:1. cutoff_date comes from the source
        # record when present (re-read cheaply from the dataset).
        cutoff_by_rec = _cutoff_by_record(dataset)
        snd_insts = [{
            "instance_id": r["instance_id"],
            "idea_id": r.get("idea_id"),
            "idea_text": _feasibility_idea_text(  # same Problem/Insight/Motivation card
                (nov_by_iid.get(r["instance_id"]) or {}).get("idea", {})),
            "cutoff_date": cutoff_by_rec.get(_rec_id_of(r["instance_id"] or "")),
        } for r in records]
        print(f"[soundness] scoring {len(snd_insts)} idea(s) with {snd_model}; "
              f"save_root={snd_root} concurrency={soundness_concurrency}")

        snd_sem = asyncio.Semaphore(max(1, soundness_concurrency))

        async def _score_snd(inst):
            async with snd_sem:  # bound by the shared GROBID container
                res = await asyncio.to_thread(
                    SJ._score_one, inst, model=snd_model, save_root=snd_root,
                    resume=resume)
            return inst["instance_id"], res

        def _log_snd(item):
            iid, res = item
            snd_by_iid[iid] = res
            print(f"[soundness] {iid}  score={res.get('soundness_score')}"
                  + (f"  ERR({res['error']})" if res.get("error") else ""))

        await _run_phase_with_eta("soundness",
                                  [_score_snd(inst) for inst in snd_insts],
                                  on_result=_log_snd)
        # attach onto each per-idea record so it flows into the enriched output.
        # stamp the scoring model into meta (soundness runs in its own phase, so
        # _score_idea's meta didn't know it) -> the flat export can surface it.
        for r in records:
            r["soundness"] = snd_by_iid.get(r["instance_id"])
            if r.get("soundness") is not None:
                r.setdefault("meta", {})["soundness"] = {"model": snd_model}

    # --- contribution-vs-sources (ScholarEval: idea novelty vs its OWN sources) ---
    # Like soundness, this wraps ScholarEval and is blocking (LLM-only, no GROBID /
    # no Semantic Scholar, ~1-2 min/idea), so it runs as its own phase after the
    # async gather and merges back by instance_id. Unlike soundness it needs the
    # per-record SOURCES the idea was conditioned on (contribution_sources_key).
    # Opt-in via --with-contribution.
    con_by_iid: Dict[str, Dict[str, Any]] = {}
    con_model = None
    if with_contribution:
        from eval import contribution_source_judge as CJ  # lazy: pulls in ScholarEval
        con_model = contribution_model or CJ.DEFAULT_MODEL
        sources_by_rec = _sources_by_record(dataset, contribution_sources_key)
        if not sources_by_rec:
            print(f"[contribution] --with-contribution set but NO record carries "
                  f"'{contribution_sources_key}'; contribution will be null for every idea.")
        # one contribution instance per already-scored idea, same instance_id so
        # results merge back 1:1. Skip ideas whose record has no sources.
        con_insts = []
        for r in records:
            iid = r["instance_id"]
            sources = sources_by_rec.get(_rec_id_of(iid or ""))
            if not sources:
                continue
            con_insts.append({
                "instance_id": iid,
                "idea_id": r.get("idea_id"),
                "query": _candidate_idea_text((nov_by_iid.get(iid) or {}).get("idea", {}), si=si_format),
                "sources": sources,
            })
        print(f"[contribution] scoring {len(con_insts)} idea(s) with {con_model}; "
              f"sources_key={contribution_sources_key} concurrency={contribution_concurrency}")

        # crash-safe checkpoint (parallel to the per-idea one): append each scored
        # idea to <ckpt-stem>.con.jsonl the moment it's done, and on resume reuse
        # what's already there. Unlike phase 1, ScholarEval's contribution path
        # writes no per-instance artifact, so this file is the ONLY progress signal
        # for this phase (`wc -l` = ideas done). Deleted on full success alongside
        # the others.
        con_ckpt = ckpt.with_name(ckpt.stem + ".con" + ckpt.suffix)
        if resume and con_ckpt.exists():
            for line in con_ckpt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("instance_id"):
                    con_by_iid[r["instance_id"]] = r.get("result", r)
            print(f"[ckpt] resume: {len(con_by_iid)} contribution idea(s) already scored")
        else:
            con_ckpt.write_text("", encoding="utf-8")
        # drop ideas already scored in a prior run so we don't re-hit the gateway.
        con_insts = [i for i in con_insts if i["instance_id"] not in con_by_iid]
        if resume and con_by_iid:
            print(f"[contribution] {len(con_insts)} idea(s) remaining after resume")

        con_sem = asyncio.Semaphore(max(1, contribution_concurrency))

        async def _score_con(inst):
            async with con_sem:  # pure LLM -> bounded only by the gateway
                res = await asyncio.to_thread(CJ._score_one, inst, model=con_model)
            return inst["instance_id"], res

        def _log_con(item):
            iid, res = item
            con_by_iid[iid] = res
            # append before returning (serial: _run_phase_with_eta drives this from
            # a single loop, so no lock needed).
            with con_ckpt.open("a", encoding="utf-8") as cf:
                cf.write(json.dumps({"instance_id": iid, "result": res},
                                    ensure_ascii=False) + "\n")
            print(f"[contribution] {iid}  avg={res.get('avg_score')} "
                  f"min={res.get('min_score')}"
                  + (f"  ERR({res['error']})" if res.get("error") else ""))

        await _run_phase_with_eta("contribution",
                                  [_score_con(inst) for inst in con_insts],
                                  on_result=_log_con)
        # attach onto each per-idea record (None where the record had no sources).
        # stamp the scoring model into meta (own phase -> not in _score_idea's meta).
        for r in records:
            r["contribution"] = con_by_iid.get(r["instance_id"])
            if r.get("contribution") is not None:
                r.setdefault("meta", {})["contribution"] = {"model": con_model}

    # --- combined summary ---
    nov_scored = [r for r in records if r["novelty"]["novelty_score"] is not None]
    imp_scored = [r for r in records if r.get("impact") and r["impact"].get("impact") is not None]
    mean_nov = (sum(r["novelty"]["novelty_score"] for r in nov_scored) / len(nov_scored)
                if nov_scored else None)
    mean_imp = (sum(r["impact"]["impact"] for r in imp_scored) / len(imp_scored)
                if imp_scored else None)
    # feasibility: two independent 1-5 tiers (compute / human-eval effort).
    feas_scored = [r for r in records
                   if r.get("feasibility") and r["feasibility"].get("compute_tier") is not None]

    def _feas_mean(key):
        vals = [r["feasibility"][key] for r in feas_scored
                if r["feasibility"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    mean_feas_compute = _feas_mean("compute_tier")
    mean_feas_human = _feas_mean("human_eval_tier")
    # soundness: ScholarEval mean-of-per-method score (0-10).
    snd_scored = [r for r in records
                  if r.get("soundness") and r["soundness"].get("soundness_score") is not None]
    mean_snd = (sum(r["soundness"]["soundness_score"] for r in snd_scored) / len(snd_scored)
                if snd_scored else None)
    # contribution: two novelty-vs-sources scores (-10..10, LOW = overlap).
    con_scored = [r for r in records
                  if r.get("contribution") and r["contribution"].get("avg_score") is not None]

    def _con_mean(key):
        vals = [r["contribution"][key] for r in con_scored
                if r["contribution"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    mean_con_avg = _con_mean("avg_score")
    mean_con_min = _con_mean("min_score")
    # grounding: load-bearing multiplier g in [0,1] (the headline).
    grd_scored = [r for r in records
                  if r.get("grounding") and r["grounding"].get("g") is not None]
    mean_grd_g = (sum(r["grounding"]["g"] for r in grd_scored) / len(grd_scored)
                  if grd_scored else None)
    # pointwise soundness: reward in [0,1] from the 1-4 rating.
    snd_pw_scored = [r for r in records
                     if r.get("soundness_pointwise")
                     and r["soundness_pointwise"].get("reward") is not None]
    mean_snd_pw = (sum(r["soundness_pointwise"]["reward"] for r in snd_pw_scored)
                   / len(snd_pw_scored) if snd_pw_scored else None)

    # --- aggregate token usage + cost across every feature/call ---
    def _feat_totals(feature: str):
        usages, cost = [], 0.0
        any_cost = False
        for r in records:
            m = (r.get("meta") or {}).get(feature)
            if not m:
                continue
            usages.append(m.get("usage") or clients.usage_dict(None))
            if m.get("cost_usd") is not None:
                cost += m["cost_usd"]; any_cost = True
        return clients.add_usage(*usages) if usages else clients.usage_dict(None), \
               (cost if any_cost else None)
    cost_by_feature = {}
    for feat in ("novelty", "impact", "feasibility"):
        u, c = _feat_totals(feat)
        cost_by_feature[feat] = {"usage": u, "cost_usd": c}
    # diversity usage lives on div_records (one per response, not per idea).
    if with_diversity:
        div_usages = [(r.get("meta") or {}).get("diversity", {}).get("usage")
                      or clients.usage_dict(None) for r in div_records]
        div_costs = [(r.get("meta") or {}).get("diversity", {}).get("cost_usd")
                     for r in div_records]
        any_cost = any(c is not None for c in div_costs)
        cost_by_feature["diversity"] = {
            "usage": clients.add_usage(*div_usages) if div_usages else clients.usage_dict(None),
            "cost_usd": sum(c for c in div_costs if c is not None) if any_cost else None}

    # --- diversity rollup (per-response -> corpus mean + per-field) ---
    div_summary = DJ.summarize(div_records) if with_diversity else "skipped (not requested)"

    # --- parsed-instance counts: how much came out of the raw predictions ---
    #   n_responses = distinct (record, sample) generations parsed
    #   n_ideas     = total ideas parsed across all responses (== per-idea rows)
    #   n_records   = distinct source records that produced >=1 parseable idea
    responses = {(_rec_id_of(r.get("instance_id") or ""), _sample_idx_of(r.get("instance_id") or ""))
                 for r in records}
    parsed = {
        "n_records": len({_rec_id_of(r.get("instance_id") or "") for r in records}),
        "n_responses": len(responses),
        "n_ideas": len(records),
    }

    summary = {
        "n_ideas": len(records),
        "parsed": parsed,
        "novelty": ({"n_scored": len(nov_scored), "mean_novelty_score": mean_nov,
                     "rubric_version": novelty.RUBRIC_VERSION}
                    if with_novelty else "skipped (not requested)"),
        "impact": ({"n_scored": len(imp_scored), "mean_impact": mean_imp,
                    "model": sci_model} if with_impact else "skipped (not requested)"),
        "feasibility": ({"n_scored": len(feas_scored),
                         "mean_compute_tier": mean_feas_compute,
                         "mean_human_eval_tier": mean_feas_human,
                         "model": feas_model, "scale_version": FJ.feas.SCALE_VERSION}
                        if with_feasibility else "skipped (not requested)"),
        "soundness": ({"n_scored": len(snd_scored),
                       "n_error": sum(1 for r in records
                                      if r.get("soundness") and r["soundness"].get("error")),
                       "mean_soundness_score": mean_snd,
                       "model": snd_model}
                      if with_soundness else "skipped (not requested)"),
        "contribution": ({"n_scored": len(con_scored),
                          "n_error": sum(1 for r in records
                                         if r.get("contribution") and r["contribution"].get("error")),
                          "mean_avg_score": mean_con_avg,
                          "mean_min_score": mean_con_min,
                          "model": con_model}
                         if with_contribution else "skipped (not requested)"),
        "grounding": ({"n_scored": len(grd_scored),
                       "n_error": sum(1 for r in records
                                      if r.get("grounding") and r["grounding"].get("error")),
                       "mean_g": mean_grd_g, "model": grd_model,
                       "rubric_version": (
                           (records[0].get("meta") or {}).get("grounding", {})
                           .get("rubric_version")
                           if records else None
                       ) or (new_grounding.RUBRIC_VERSION if with_new_grounding
                             else grounding.RUBRIC_VERSION),
                       "variant": ("new_grounding" if with_new_grounding
                                   else "grounding")}
                      if (with_grounding or with_new_grounding)
                      else "skipped (not requested)"),
        "soundness_pointwise": ({"n_scored": len(snd_pw_scored),
                                 "n_error": sum(1 for r in records
                                                if r.get("soundness_pointwise")
                                                and r["soundness_pointwise"].get("error")),
                                 "mean_reward": mean_snd_pw, "model": snd_pw_model}
                                if with_soundness_pointwise else "skipped (not requested)"),
        "diversity": div_summary,
        "cost": cost_by_feature,
        "model": model, "backend": backend, "mode": mode,
    }
    # --- three outputs only ---
    #   (a) <dataset>_with_evals.jsonl : input copy + per-row "evals"
    #   (b) <dataset>_with_evals.xlsx  : same, as a spreadsheet (optional/openpyxl)
    #   (c) <out>                      : the summary (the path passed via --out)
    # Build the enriched input rows once (each input record gets an "evals" list =
    # the per-idea eval records whose instance_id maps back to it via rec_id; same
    # row count/order as the input, nothing dropped or mutated).
    # evals is a LIST OF LISTS per row: outer index = sample (s<n>), inner list =
    # that sample's per-idea eval records. So a record with multiple samples for
    # the same instance gets evals[sample_idx] = [idea0_eval, idea1_eval, ...].
    by_rec_sample: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for rec in records:
        iid = rec.get("instance_id") or ""
        by_rec_sample.setdefault(_rec_id_of(iid), {}).setdefault(
            _sample_idx_of(iid), []).append(rec)
    enriched_rows: List[Dict[str, Any]] = []
    with dataset.open() as fin:
        for line_no, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rec_id = str(r.get("target_arxiv_id") or r.get("index") or f"rec{line_no}")
            samples = by_rec_sample.get(rec_id, {})
            # When --limit is set we only evaluated a prefix of the ideas, so only
            # write back the rows that actually got scored (skip the untouched tail)
            # rather than emitting all input rows with empty evals.
            if limit is not None and not samples:
                continue
            # dense list-of-lists ordered by sample index (gaps -> empty lists)
            n_samples = (max(samples) + 1) if samples else 0
            r["evals"] = [samples.get(s, []) for s in range(n_samples)]
            # diversity is per (record, sample) -> one list entry per sample,
            # parallel to evals[sample_idx]; None where that sample wasn't scored.
            if with_diversity:
                div_samples = div_by_rec_sample.get(rec_id, {})
                r["diversity"] = [div_samples.get(s, {}).get("diversity")
                                  for s in range(n_samples)]
            enriched_rows.append(r)

    # (a) input copy + evals (JSONL) -- written first so a missing openpyxl or a
    #     bad summary serialize can't block the core output.
    enriched_path = out.with_name(out.stem + "_with_evals.jsonl")
    with enriched_path.open("w") as fout:
        for r in enriched_rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    # (c) summary (the --out path).
    with out.open("w") as f:
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")

    # (b) FLAT one-row-per-idea Excel for human eval/validation. Instead of a dense
    #     sheet whose single cell holds the whole nested "evals" blob, export_evals
    #     unrolls it: one row per (record, sample, idea) with the candidate idea
    #     text, gold, and all novelty/impact scores in plain columns.
    #     Optional: skipped with a warning if openpyxl is absent.
    enriched_xlsx = out.with_name(out.stem + "_with_evals.xlsx")
    n_idea_rows = export_evals.write_xlsx(enriched_rows, enriched_xlsx)
    if n_idea_rows is None:
        enriched_xlsx = None
        print("[excel] skipped .xlsx export (openpyxl/pandas missing); "
              "pip install openpyxl to enable.")
    n_rows = len(enriched_rows)
    print("\n=== combined eval ===")
    print(f"  parsed:          {parsed['n_ideas']} ideas from "
          f"{parsed['n_responses']} responses / {parsed['n_records']} records")
    print(f"  ideas:           {summary['n_ideas']}")
    if not with_novelty:
        print(f"  novelty:         skipped (pass --with-novelty)")
    elif mean_nov is not None:
        print(f"  mean novelty:    {mean_nov:.3f}  (0=not novel, 1=fully novel)")
    if with_impact:
        print(f"  mean impact:     {mean_imp}  ({len(imp_scored)}/{len(records)} had gold)")
    else:
        print(f"  impact:          skipped (pass --with-impact)")
    if with_feasibility:
        fc = mean_feas_compute; fh = mean_feas_human
        print(f"  mean feasibility: compute={fc if fc is None else round(fc,2)} "
              f"human_eval={fh if fh is None else round(fh,2)}  (1-5; "
              f"{len(feas_scored)}/{len(records)} scored)")
    else:
        print(f"  feasibility:     skipped (pass --with-feasibility)")
    if with_soundness:
        print(f"  mean soundness:  {mean_snd if mean_snd is None else round(mean_snd,2)}"
              f"  (0-10; {len(snd_scored)}/{len(records)} scored)")
    else:
        print(f"  soundness:       skipped (pass --with-soundness)")
    if with_contribution:
        ca = mean_con_avg; cm = mean_con_min
        print(f"  mean contribution: avg={ca if ca is None else round(ca,2)} "
              f"min={cm if cm is None else round(cm,2)}  (-10..10, low=overlap; "
              f"{len(con_scored)}/{len(records)} scored)")
    else:
        print(f"  contribution:    skipped (pass --with-contribution)")
    if with_grounding or with_new_grounding:
        print(f"  mean grounding g: {mean_grd_g if mean_grd_g is None else round(mean_grd_g,3)}"
              f"  (0-1 load-bearing; {len(grd_scored)}/{len(records)} scored"
              f"; {'new_grounding' if with_new_grounding else 'grounding'})")
    else:
        print(f"  grounding:       skipped (pass --with-grounding or --with-new-grounding)")
    if with_soundness_pointwise:
        print(f"  mean soundness(pw): {mean_snd_pw if mean_snd_pw is None else round(mean_snd_pw,3)}"
              f"  (0-1 from 1-4 rating; {len(snd_pw_scored)}/{len(records)} scored)")
    else:
        print(f"  soundness(pw):   skipped (pass --with-soundness-pointwise)")
    if with_diversity:
        md = div_summary["mean_diversity"]
        print(f"  mean diversity:  {md if md is None else round(md, 3)}  "
              f"(0=identical, 1=unrelated; {div_summary['n_scored']}/"
              f"{div_summary['n_responses']} responses had >=2 ideas)")
        print(f"    by field:      {div_summary['mean_by_field']}")
    else:
        print(f"  diversity:       skipped (pass --with-diversity)")
    print("  tokens/cost:")
    for feat, cb in cost_by_feature.items():
        tot = cb["usage"].get("total_tokens")
        print(f"    {feat:9s} total_tokens={tot}  cost_usd={cb['cost_usd']}")
    print(f"  -> {enriched_path}  (input copy + per-row evals, {n_rows} rows)")
    if enriched_xlsx is not None:
        print(f"  -> {enriched_xlsx}  (flat one-row-per-idea, {n_idea_rows} rows)")
    print(f"  -> {out}  (summary)")
    if _tmp_ds is not None:
        _tmp_ds.unlink(missing_ok=True)  # drop the filtered temp copy
    # all outputs written successfully -> the checkpoint(s) have served their
    # crash-safety purpose; drop them so they don't confuse a later fresh run.
    #
    # keep_checkpoints inverts that for INCREMENTAL scoring. --resume is gated on
    # ckpt.exists(), so deleting on success means the next run over a superset of
    # this dataset re-scores everything already paid for: a clean finish destroys
    # exactly the state an append-only dataset needs. Keeping them costs ~120MB per
    # 5k records and makes the next run score only the new rows.
    if keep_checkpoints:
        print(f"[ckpt] keeping checkpoints (--keep-checkpoints) next to {out}")
        return summary
    ckpt.unlink(missing_ok=True)
    if with_diversity:
        ckpt.with_name(ckpt.stem + ".div" + ckpt.suffix).unlink(missing_ok=True)
    if with_contribution:
        ckpt.with_name(ckpt.stem + ".con" + ckpt.suffix).unlink(missing_ok=True)
    if with_grounding:
        ckpt.with_name(ckpt.stem + ".grd" + ckpt.suffix).unlink(missing_ok=True)
    if with_new_grounding:
        ckpt.with_name(ckpt.stem + ".grdnew" + ckpt.suffix).unlink(missing_ok=True)
    if with_soundness_pointwise:
        ckpt.with_name(ckpt.stem + ".sndpw" + ckpt.suffix).unlink(missing_ok=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run novelty / impact / feasibility / soundness / contribution / diversity over one file.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True,
                    help="summary output path; the input-copy+evals files are "
                         "written alongside as <out>_with_evals.{jsonl,xlsx}")
    ap.add_argument("--backend", choices=clients.VALID_BACKENDS, default="vllm")
    ap.add_argument("--model", default=None,
                    help="judge model for novelty+diversity; default per backend")
    ap.add_argument("--mode", choices=["per-doc", "batched"], default="per-doc",
                    help="novelty document-batching mode")
    ap.add_argument("--input-format",
                    choices=["eval-set", "ideation-answer", "ideation-si"],
                    default="ideation-answer")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--k", type=int, default=None, help="max ideas per generation (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="first N ideas (debug)")
    # --- record-level subsetting (cheaper runs / stratified domain slices) ---
    ap.add_argument("--domain", action="append", default=[], metavar="LABEL",
                    help="keep only records whose `domain` OR arXiv `primary` "
                         "matches (case-insensitive), e.g. --domain Language or "
                         "--domain cs.CL. Repeatable to keep several domains.")
    ap.add_argument("--per-domain", type=int, default=None, metavar="N",
                    help="cap EACH --domain to its first N records (stratified), "
                         "e.g. --domain Language --domain Physics --per-domain 100 "
                         "-> first 100 Language + first 100 Physics. Requires --domain.")
    ap.add_argument("--max-records", type=int, default=None, metavar="N",
                    help="flat cap on total RECORDS scored (distinct source docs), "
                         "applied after --domain/--per-domain. Counts records, not "
                         "ideas (unlike --limit).")
    # --- novelty (our home-grown per-doc overlap judge; OPT-IN) ---
    # Skipped by default: contribution-vs-sources / soundness cover idea novelty
    # against its sources, so a normal run does diversity without it.
    ap.add_argument("--with-novelty", action="store_true",
                    help="also score our home-grown novelty judge (per-doc overlap "
                         "vs the idea's references, gpt-oss). Off by default.")
    # --- impact (SciJudge vs gold) ---
    ap.add_argument("--with-impact", action="store_true",
                    help="also score impact (SciJudge vs the record's gold idea); "
                         "needs target_idea_str per record + a SciJudge endpoint")
    # diversity (per-response pairwise distance among the K ideas, reusing the
    # gpt-oss judge + the F9-B distance.py rubric; no extra endpoint). OPT-IN like
    # novelty/impact -- off by default so an impact-only run needs no gpt-oss endpoint.
    ap.add_argument("--with-diversity", action="store_true",
                    help="also score per-response diversity (C(K,2) pairwise "
                         "distances among a response's K ideas, gpt-oss). Off by default.")
    ap.add_argument("--judge-url", default=None,
                    help="explicit gpt-oss vLLM URL for novelty+diversity "
                         "(else discovered via endpoint file / env)")
    ap.add_argument("--scijudge-backend", choices=clients.VALID_BACKENDS,
                    default="vllm",
                    help="backend for the impact judge (default: vllm -> the "
                         "self-hosted SciJudge-30B). Use 'gateway' to score impact "
                         "with a hosted LLM API (gpt-/claude-) instead.")
    ap.add_argument("--scijudge-url", default=None,
                    help="explicit base URL for the impact judge: a SciJudge-30B "
                         "vLLM URL (vllm backend; else discovered via "
                         "SCIJUDGE_ENDPOINT_FILE / env) or a gateway override URL.")
    ap.add_argument("--scijudge-model", default=None,
                    help="impact judge model; default per --scijudge-backend "
                         "(vllm -> SciJudge-30B, gateway -> claude-opus-4-8).")
    ap.add_argument("--scijudge-max-tokens", type=int, default=2048)
    # --- feasibility (compute + human-eval tiers; dedicated model) ---
    ap.add_argument("--with-feasibility", action="store_true",
                    help="also score feasibility (compute + human-eval 1-5 tiers) "
                         "with a dedicated model; runs concurrently per idea.")
    ap.add_argument("--feasibility-backend", choices=clients.VALID_BACKENDS,
                    default="gateway",
                    help="backend for the feasibility judge (default: gateway).")
    ap.add_argument("--feasibility-model", default=None,
                    help="feasibility judge model; default per backend "
                         "(gateway -> claude-opus-4-8, vllm -> gpt-oss-120b).")
    ap.add_argument("--feasibility-url", default=None,
                    help="explicit base URL for a vllm feasibility backend.")
    # --- soundness (ScholarEval; slow, needs GROBID) ---
    ap.add_argument("--with-soundness", action="store_true",
                    help="also score soundness via ScholarEval (needs GROBID on "
                         ":8070; ~3-4 min/idea). Runs as a separate phase, resumable "
                         "via --soundness-save-root + --resume.")
    ap.add_argument("--soundness-model", default=None,
                    help="ScholarEval llm_engine label (default: GPT-4.1).")
    ap.add_argument("--soundness-save-root", default=None,
                    help="root dir for ScholarEval artifacts (default: "
                         "<out-stem>_soundness). One subdir per instance; enables resume.")
    ap.add_argument("--soundness-concurrency", type=int, default=3,
                    help="concurrent soundness evals (bounded by shared GROBID; 3 safe).")
    # --- contribution-vs-sources (ScholarEval; LLM-only, fast, no GROBID/S2) ---
    ap.add_argument("--with-contribution", action="store_true",
                    help="also score contribution-vs-sources (idea novelty vs the "
                         "specific documents it was conditioned on) via ScholarEval. "
                         "Pure LLM (~1-2 min/idea), runs as a separate phase.")
    ap.add_argument("--contribution-model", default=None,
                    help="ScholarEval llm_engine label for contribution (default: GPT-4.1).")
    ap.add_argument("--contribution-concurrency", type=int, default=4,
                    help="concurrent contribution evals (pure LLM; bounded by the gateway).")
    ap.add_argument("--contribution-sources-key", default="input_full_summaries_str",
                    help="record field holding the source documents the idea was "
                         "conditioned on (default: input_full_summaries_str).")
    # --- grounding (common.grounding OR dual-head common.new_grounding) ---
    ap.add_argument("--with-grounding", action="store_true",
                    help="also score grounding fidelity (load-bearing multiplier g "
                         "in [0,1]) via common.grounding over the idea's sources. "
                         "Runs as its own phase, resumable via --resume.")
    ap.add_argument("--with-new-grounding", action="store_true",
                    help="reporting dual-head grounding via common.new_grounding: "
                         "LB_ONLY (no docs, think0 by default) for load_bearing + "
                         "vagueness-spliced full template (with docs) for "
                         "support_faithful/source_exists. Prefer this over "
                         "--with-grounding for n1k3 reporting. Checkpoint: .grdnew")
    ap.add_argument("--new-grounding-lb-thinking-budget", type=int, default=0,
                    help="Gemini thinkingBudget for the LB_ONLY head "
                         "(default 0 = off). Ignored for non-gemini models. "
                         "Pass a negative value to omit the budget (API default).")
    ap.add_argument("--new-grounding-sf-thinking-budget", type=int, default=None,
                    help="Gemini thinkingBudget for the faithfulness head "
                         "(default: omit = API default think). Pass 0 to disable.")
    ap.add_argument("--new-grounding-no-sf", action="store_true",
                    help="skip head B (support_faithful) and score load_bearing "
                         "only. `g` comes from load_bearing alone, so this does not "
                         "change any gate -- it drops a reporting column. Head B is "
                         "the head that needs the documents (~3.2k prompt tokens vs "
                         "~0.7k), so this halves grounding calls and cuts ~80%% of "
                         "its tokens.")
    ap.add_argument("--grounding-backend", choices=clients.VALID_BACKENDS,
                    default="vllm",
                    help="backend for the grounding judge (default: vllm -> gpt-oss).")
    ap.add_argument("--grounding-model", default=None,
                    help="grounding judge model; default per backend.")
    ap.add_argument("--grounding-url", default=None,
                    help="explicit base URL for a vllm grounding backend.")
    ap.add_argument("--grounding-max-tokens", type=int, default=1536)
    ap.add_argument("--grounding-sources-key", default="input_full_summaries_str",
                    help="record field holding the source documents (default: "
                         "input_full_summaries_str).")
    ap.add_argument("--grounding-require-type", action="store_true",
                    help="require a valid grounding_type for a non-floor g (v0 "
                         "schema). Off by default: SI cards have no grounding_type, "
                         "and requiring it would pin every SI idea to the floor.")
    ap.add_argument("--grounding-thinking-budget", type=int, default=None,
                    help="Gemini thinkingBudget for --with-grounding "
                         "(omit=API default, 0=off, N=cap). Ignored for "
                         "non-gemini models. For --with-new-grounding use "
                         "--new-grounding-*-thinking-budget instead.")
    # --- pointwise soundness (common.soundness scale-4; own phase) ---
    ap.add_argument("--with-soundness-pointwise", action="store_true",
                    help="also score pointwise soundness (1-4 -> [0,1] reward) via "
                         "common.soundness (the validated pointwise judge, distinct "
                         "from the slow ScholarEval --with-soundness).")
    ap.add_argument("--soundness-pw-backend", choices=clients.VALID_BACKENDS,
                    default="gateway",
                    help="backend for the pointwise soundness judge (default: gateway).")
    ap.add_argument("--soundness-pw-model", default=None,
                    help="pointwise soundness judge model (default: common.soundness "
                         "DEFAULT_MODEL; for the reward-validated judge use claude-sonnet-5).")
    ap.add_argument("--soundness-pw-url", default=None,
                    help="explicit base URL for a vllm pointwise soundness backend.")
    ap.add_argument("--soundness-pw-max-tokens", type=int, default=6144,
                    help="max tokens for the pointwise soundness judge (>=6144 for "
                         "sonnet-5 so the JSON body fits after thinking).")
    ap.add_argument("--soundness-pw-concurrency", type=int, default=16,
                    help="concurrent pointwise-soundness evals (bounded by the gateway).")
    # --- crash-safe checkpointing ---
    ap.add_argument("--checkpoint", default=None,
                    help="path to append each scored idea to as it completes "
                         "(default <out-stem>.ckpt.jsonl). Deleted on success.")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="do NOT delete the per-phase checkpoints on success. Needed "
                         "for incremental scoring: --resume only reads a checkpoint "
                         "that still exists, so a clean run otherwise forces the next "
                         "run over a superset file to re-score every idea.")
    ap.add_argument("--resume", action="store_true",
                    help="reuse ideas already in the checkpoint (skip re-scoring "
                         "them); pair with the same --out/--checkpoint as the "
                         "crashed run.")
    # --- pricing for cost metadata ($/1M tokens; local vLLM -> omit -> cost null) ---
    ap.add_argument("--price", action="append", default=[], metavar="MODEL:IN,OUT",
                    help="per-model $/1M-token rates for cost accounting, e.g. "
                         "--price openai/gpt-oss-120b:0.5,1.5 (repeatable). Models "
                         "without a rate report cost_usd=null.")
    args = ap.parse_args()
    if args.per_domain is not None and not args.domain:
        ap.error("--per-domain requires --domain (use --max-records for a flat cap "
                 "with no domain filter).")
    model = args.model or DEFAULT_MODELS[args.backend]
    prices = _parse_prices(args.price)
    _t0 = time.time()
    print(f"[timer] start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    asyncio.run(run_async(
        Path(args.dataset), Path(args.out), backend=args.backend, model=model,
        mode=args.mode, input_format=args.input_format, temperature=args.temperature,
        max_tokens=args.max_tokens, max_concurrent=args.max_concurrent,
        k=args.k, limit=args.limit, with_impact=args.with_impact,
        with_diversity=args.with_diversity, with_novelty=args.with_novelty,
        domains=args.domain or None, per_domain=args.per_domain,
        max_records=args.max_records,
        judge_url=args.judge_url, scijudge_url=args.scijudge_url,
        scijudge_model=args.scijudge_model, scijudge_max_tokens=args.scijudge_max_tokens,
        scijudge_backend=args.scijudge_backend,
        with_feasibility=args.with_feasibility,
        feasibility_backend=args.feasibility_backend,
        feasibility_model=args.feasibility_model, feasibility_url=args.feasibility_url,
        with_soundness=args.with_soundness, soundness_model=args.soundness_model,
        soundness_save_root=(Path(args.soundness_save_root)
                             if args.soundness_save_root else None),
        soundness_concurrency=args.soundness_concurrency,
        with_contribution=args.with_contribution,
        contribution_model=args.contribution_model,
        contribution_concurrency=args.contribution_concurrency,
        contribution_sources_key=args.contribution_sources_key,
        with_grounding=args.with_grounding,
        grounding_backend=args.grounding_backend,
        grounding_model=args.grounding_model,
        grounding_url=args.grounding_url,
        grounding_max_tokens=args.grounding_max_tokens,
        grounding_sources_key=args.grounding_sources_key,
        grounding_require_type=args.grounding_require_type,
        grounding_thinking_budget=args.grounding_thinking_budget,
        with_new_grounding=args.with_new_grounding,
        new_grounding_lb_thinking_budget=(
            None if args.new_grounding_lb_thinking_budget < 0
            else args.new_grounding_lb_thinking_budget),
        new_grounding_sf_thinking_budget=args.new_grounding_sf_thinking_budget,
        new_grounding_with_sf=not args.new_grounding_no_sf,
        with_soundness_pointwise=args.with_soundness_pointwise,
        soundness_pw_backend=args.soundness_pw_backend,
        soundness_pw_model=args.soundness_pw_model,
        soundness_pw_url=args.soundness_pw_url,
        soundness_pw_max_tokens=args.soundness_pw_max_tokens,
        soundness_pw_concurrency=args.soundness_pw_concurrency,
        prices=prices,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        resume=args.resume,
        keep_checkpoints=args.keep_checkpoints))
    _elapsed = time.time() - _t0
    print(f"[timer] end   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"| elapsed {_elapsed:.1f}s ({_elapsed / 60:.1f} min)", flush=True)


def _parse_prices(specs: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """Parse repeated --price MODEL:IN,OUT into {model: {'in':float,'out':float}}."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for spec in specs:
        try:
            model, rates = spec.split(":", 1)
            pin, pout = (rates.split(",") + [""])[:2]
            out[model.strip()] = {"in": float(pin) if pin else None,
                                  "out": float(pout) if pout else None}
        except Exception:
            raise SystemExit(f"bad --price {spec!r}; expected MODEL:IN,OUT")
    return out


if __name__ == "__main__":
    main()
