#!/usr/bin/env python3
"""Score the FULL pooled-35 pairwise diversity matrix for an arm-B eval file.

Arm B pooled 5 ideas x 7 draws = ~35 ideas per instance, but --with-diversity in
run_all_evals scored diversity PER DRAW (7 blocks of C(5,2)=10 pairs each). The
cross-draw distances -- the ones a greedy / creative-utility walk over the pooled
set needs -- were never computed. This fills them in: for each instance it scores
ALL C(n,2) pairs of the pooled ideas with the SAME F9-B rubric the within-draw
pass used -- common.distance (distance-pairwise-v2) for v0/ideation-answer, or
common.distance_si (distance-pairwise-si-v1) for --input-format ideation-si, per
rubric_for(si) -- so the resulting matrix is on one consistent scale per format
and drops straight into the CU / greedy-ordering plots.

Idea order in the pool = draw-then-idea (draw 0's ideas, then draw 1's, ...),
which is EXACTLY the order evals[]/diversity[] are stored in, so pooled index g
lines up with the per-idea impact/novelty already on disk (verified: parsed ideas
== evals == diversity n_ideas per draw for every model).

Seeding: the within-draw pairs already scored in the source file's diversity[]
are mapped to their pooled (i,j) and REUSED, so only the cross-draw pairs are sent
to the judge (~525 of 595 per instance).

Checkpointing (see always-index-and-checkpoint memory): every scored pair is
appended to <out>.ckpt.jsonl as {instance_id, i, j, distance, by_field, error}
the instant it returns. Re-running resumes from the checkpoint -- nothing is
re-scored. The final per-instance pool file is rebuilt from the checkpoint at the
end (and can be rebuilt any time with --rebuild-only).

Usage:
  synthesis_env/bin/python qd_tradeoff/score_pool_diversity.py \
      --in  qd_tradeoff/evals/b_sample_sweep/gpt-5.5-2026-04-23/n7_summary_with_evals.jsonl \
      --out qd_tradeoff/evals/b_sample_sweep/gpt-5.5-2026-04-23/pool_diversity.jsonl \
      --backend vllm --model openai/gpt-oss-120b --judge-url http://localhost:8001 \
      --max-concurrent 32
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_QD = Path(__file__).resolve().parent
_REPO = _QD.parent
_GRPO = _REPO / "grpo_training"
for _p in (str(_REPO), str(_GRPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import clients, distance, distance_si, ideas, ideas_si  # noqa: E402

# Bare field names for the per-field distance breakdown (mirrors distance.py's
# FEATURE_KEYS/weight-table field names, without the trailing "_sim").
_FIELDS = ("problem", "insight", "motivation")           # v0 rubric
_FIELDS_SI = ("problem_gap", "core_claim", "method")     # SI (v1) rubric


def rubric_for(si: bool) -> Tuple[Any, Tuple[str, ...]]:
    """(rubric module, bare field names) for the SI or v0 pairwise-distance rubric.

    `si=True` MUST select common.distance_si, not common.distance -- mirrors
    grpo_training/eval/diversity_judge.py's rubric_for (see that module's
    docstring for the full rationale). Both modules expose the same interface
    (build_messages / extract_json / parse_features / weighted_field_distance /
    feature_to_distance / RUBRIC_VERSION), so picking the wrong one does not
    raise -- it silently scores SI cards on the v0 prompt/weights instead of
    distance_si's, drifting from the SI training reward and human-val distance
    (which both use distance_si)."""
    return (distance_si, _FIELDS_SI) if si else (distance, _FIELDS)


# --------------------------------------------------------------------------- #
# Load: pool each instance's ideas (draw-then-idea order) + seed known distances
# --------------------------------------------------------------------------- #
def _pool_instance(rec: Dict[str, Any], *, si: bool = False) -> Tuple[str, List[str], Dict[Tuple[int, int], float]]:
    """Return (instance_id, pooled card list, seed{(i,j)->distance}).

    Pooled cards come from parsing answer_str per draw (draw-then-idea order).
    Seeds come from the source diversity[] blocks: each draw d's pairs (local
    a,b) map to pooled (offset_d + a, offset_d + b).

    `si=True` selects the SI (v1) parser: SI cards are {problem_gap, core_claim,
    method, inspiration}, which common.ideas.parse_ideas rejects wholesale (0
    ideas -> empty pool -> all-null pool_diversity). ideas_si.parse_ideas_si
    builds SIIdea, mirroring diversity_judge._load_groups' si branch."""
    parse = ideas_si.parse_ideas_si if si else ideas.parse_ideas
    inst = str(rec.get("target_arxiv_id") or rec.get("index"))
    answers = rec.get("answer_str") or []
    per_draw_cards: List[List[str]] = []
    for gen in answers:
        parsed = parse(gen, k=None) if isinstance(gen, str) else []
        per_draw_cards.append([idea.card_text() for idea in parsed])

    # pooled offsets
    offsets, off = [], 0
    for cards in per_draw_cards:
        offsets.append(off)
        off += len(cards)
    pooled = [c for cards in per_draw_cards for c in cards]

    # seed from existing within-draw diversity pairs
    seed: Dict[Tuple[int, int], float] = {}
    for d, dv in enumerate(rec.get("diversity") or []):
        if not dv:
            continue
        base = offsets[d] if d < len(offsets) else None
        if base is None:
            continue
        n_local = len(per_draw_cards[d]) if d < len(per_draw_cards) else 0
        for p in dv.get("pairs") or []:
            a, b, dist = p.get("i"), p.get("j"), p.get("distance")
            if a is None or b is None or dist is None:
                continue
            if a >= n_local or b >= n_local:
                continue  # parse drift; skip seed, will be re-scored if needed
            gi, gj = base + a, base + b
            if gi > gj:
                gi, gj = gj, gi
            seed[(gi, gj)] = float(dist)
    return inst, pooled, seed


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def _load_ckpt(ckpt_path: Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    done: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    if not ckpt_path.exists():
        return done
    with ckpt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[(r["instance_id"], r["i"], r["j"])] = r
    return done


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
async def _score_pair(client, model, gen_config, card_a, card_b, *,
                      rubric=distance, fields=_FIELDS) -> Dict[str, Any]:
    """One pair -> {distance, by_field, error}. distance=None on persistent fail
    (excluded from the mean, never defaulted -- same policy as diversity_judge).

    `rubric`/`fields` come from rubric_for(si): distance + _FIELDS for v0,
    distance_si + _FIELDS_SI for SI. MUST match the `si` used to build card_a/
    card_b (see rubric_for / _pool_instance)."""
    messages = rubric.build_messages(card_a, card_b)
    call = {"model": model, "messages": messages, **gen_config}
    if clients.accepts_temperature(model):
        call["temperature"] = 0.0
    err = None
    for _ in range(2):
        try:
            resp = await client.chat.completions.create(**call)
            raw = resp.choices[0].message.content or ""
            obj = rubric.extract_json(raw)
            feats = rubric.parse_features(obj) if isinstance(obj, dict) else None
            if feats is not None:
                by_field = {f: rubric.feature_to_distance(int(feats[f"{f}_sim"]))
                            for f in fields}
                return {"distance": rubric.weighted_field_distance(feats),
                        "by_field": by_field, "error": None}
            err = "unparsed"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return {"distance": None, "by_field": None, "error": err}


def _aggregate(pooled_n: int, dist_map: Dict[Tuple[int, int], Optional[float]]) -> Dict[str, Any]:
    scored = [(ij, d) for ij, d in dist_map.items() if d is not None]
    dists = [d for _, d in scored]
    n_pairs = pooled_n * (pooled_n - 1) // 2
    if not dists:
        return {"mean_pairwise": None, "min_pairwise": None, "max_pairwise": None,
                "n_ideas": pooled_n, "n_pairs": n_pairs, "n_pairs_scored": 0}
    return {"mean_pairwise": sum(dists) / len(dists),
            "min_pairwise": min(dists), "max_pairwise": max(dists),
            "n_ideas": pooled_n, "n_pairs": n_pairs, "n_pairs_scored": len(dists)}


def _rebuild(records_meta, ckpt, out_path, rubric=distance):
    """Write per-instance pool files from the checkpoint + seeds."""
    with out_path.open("w") as f:
        for inst, pooled_n, all_pairs in records_meta:
            dist_map: Dict[Tuple[int, int], Optional[float]] = {}
            pairs_out = []
            for (i, j) in all_pairs:
                r = ckpt.get((inst, i, j))
                d = r["distance"] if r else None
                dist_map[(i, j)] = d
                pairs_out.append({"i": i, "j": j, "distance": d,
                                  "by_field": (r or {}).get("by_field"),
                                  "error": (r or {}).get("error")})
            agg = _aggregate(pooled_n, dist_map)
            f.write(json.dumps({
                "instance_id": inst, "n_ideas": pooled_n,
                "rubric_version": rubric.RUBRIC_VERSION,
                **agg, "pairs": pairs_out}, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}  ({len(records_meta)} instances)")


async def main() -> None:
    ap = argparse.ArgumentParser(description="pooled-35 pairwise diversity for one arm-B file")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="vllm", choices=clients.VALID_BACKENDS)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--judge-url", default=None,
                    help="judge endpoint override (normally only needed for vLLM)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-concurrent", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N instances (debug)")
    ap.add_argument("--domain", action="append", default=[], metavar="LABEL",
                    help="keep only records whose coarse `domain` (e.g. Language) "
                         "OR arXiv `primary` (e.g. cs.CL) matches, case-insensitive. "
                         "Repeatable. Mirrors run_all_evals.py --domain so the pool "
                         "is scored over the SAME instance subset the judges saw.")
    ap.add_argument("--input-format", choices=["ideation-answer", "ideation-si"],
                    default="ideation-answer",
                    help="ideation-si uses the v1/SI parser (else the v0 parser "
                         "returns 0 ideas for SI cards -> all-null pool)")
    ap.add_argument("--rebuild-only", action="store_true",
                    help="skip scoring; just rebuild the pool file from the checkpoint")
    args = ap.parse_args()

    inp, out = Path(args.inp), Path(args.out)
    si = args.input_format == "ideation-si"
    rubric, fields = rubric_for(si)
    print(f"[pool-diversity] rubric={rubric.RUBRIC_VERSION} (si={si})")
    ckpt_path = out.with_suffix(out.suffix + ".ckpt.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # pool every instance + collect seeds and the full pair list
    records_meta = []       # (inst, pooled_n, all_pairs[(i,j)])
    seeds_all: Dict[Tuple[str, int, int], float] = {}
    with inp.open() as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if args.domain:
        wanted = {d.strip().lower() for d in args.domain}
        n_before = len(recs)
        recs = [r for r in recs
                if {str(r.get("domain") or "").lower(),
                    str(r.get("primary") or "").lower()} & wanted]
        print(f"[domain] {sorted(wanted)}: kept {len(recs)}/{n_before} instances")
    if args.limit is not None:
        recs = recs[:args.limit]
    for rec in recs:
        inst, pooled, seed = _pool_instance(rec, si=si)
        n = len(pooled)
        all_pairs = list(combinations(range(n), 2))
        records_meta.append((inst, n, all_pairs))
        for (i, j), d in seed.items():
            seeds_all[(inst, i, j)] = d

    ckpt = _load_ckpt(ckpt_path)
    # Discard judge-scored pairs from a different rubric (e.g. a stale v0-scored
    # checkpoint re-read after switching --input-format to ideation-si, or vice
    # versa) -- MUST be dropped, not resumed from, mirroring diversity_judge.py's
    # rubric_version staleness check. Missing rubric_version predates this tag
    # and is treated as stale too. Seed rows (from the source file's own
    # diversity[] blocks, merged in below) carry no rubric_version and are
    # exempt -- they are re-derived fresh from --in every run.
    stale = [k for k, r in ckpt.items() if r.get("rubric_version") != rubric.RUBRIC_VERSION]
    for k in stale:
        del ckpt[k]
    if stale:
        print(f"[pool-diversity] discarded {len(stale)} cached pair(s) scored under a "
              f"different rubric (expected {rubric.RUBRIC_VERSION})")

    # Failed calls are not completed work. Drop them from the in-memory checkpoint
    # so --resume retries them; successful replacement rows appended below win when
    # the checkpoint is loaded again.
    failed = [k for k, r in ckpt.items()
              if r.get("distance") is None and r.get("error")]
    for k in failed:
        del ckpt[k]
    if failed:
        print(f"[pool-diversity] retrying {len(failed)} previously failed pair(s)")

    # merge seeds into ckpt view (seeds are authoritative for within-draw pairs)
    for key, d in seeds_all.items():
        if key not in ckpt:
            ckpt[key] = {"instance_id": key[0], "i": key[1], "j": key[2],
                         "distance": d, "by_field": None, "error": None, "seed": True}

    if args.rebuild_only:
        _rebuild(records_meta, ckpt, out, rubric=rubric)
        return

    # figure out which pairs still need scoring
    pooled_cards = {}
    for rec in recs:
        inst, pooled, _ = _pool_instance(rec, si=si)
        pooled_cards[inst] = pooled

    todo = []  # (inst, i, j)
    for inst, n, all_pairs in records_meta:
        for (i, j) in all_pairs:
            if (inst, i, j) not in ckpt:
                todo.append((inst, i, j))

    total_pairs = sum(len(ap_) for _, _, ap_ in records_meta)
    print(f"instances={len(records_meta)}  total_pairs={total_pairs}  "
          f"seeded/cached={total_pairs - len(todo)}  to_score={len(todo)}")

    if not todo:
        _rebuild(records_meta, ckpt, out, rubric=rubric)
        return

    client = clients.build_async_client(args.backend, args.model, base_url=args.judge_url)
    gen_config = clients.generation_config(args.model, args.max_tokens)
    sem = asyncio.Semaphore(args.max_concurrent)
    ckpt_f = ckpt_path.open("a")
    write_lock = asyncio.Lock()
    done_ct = [0]

    async def worker(inst, i, j):
        cards = pooled_cards[inst]
        async with sem:
            res = await _score_pair(client, args.model, gen_config, cards[i], cards[j],
                                    rubric=rubric, fields=fields)
        row = {"instance_id": inst, "i": i, "j": j,
               "distance": res["distance"], "by_field": res["by_field"],
               "error": res["error"], "rubric_version": rubric.RUBRIC_VERSION}
        async with write_lock:
            ckpt_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            ckpt[(inst, i, j)] = row
            done_ct[0] += 1
            if done_ct[0] % 200 == 0:
                print(f"  scored {done_ct[0]}/{len(todo)}  (last {inst} {i},{j} "
                      f"d={res['distance']}{' ERR' if res['error'] else ''})")

    await asyncio.gather(*[asyncio.create_task(worker(*t)) for t in todo])
    ckpt_f.close()
    n_err = sum(1 for r in ckpt.values() if r.get("error"))
    print(f"done scoring. checkpoint pairs={len(ckpt)}  errors={n_err}")
    _rebuild(records_meta, ckpt, out, rubric=rubric)


if __name__ == "__main__":
    asyncio.run(main())
