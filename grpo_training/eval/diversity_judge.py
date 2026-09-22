#!/usr/bin/env python3
"""Diversity judge: how DIFFERENT are the K ideas WITHIN one response from each
other? (set-level, idea-vs-idea -- NOT idea-vs-source like novelty, NOT idea-vs-
gold like impact).

Reuses the F9-B pairwise-distance rubric in common/distance.py so the offline
diversity metric and the training reward's diversity signal share ONE definition
and cannot drift. Each ordered response (one record, one sample) contributes its
own diversity score, computed over all C(K,2) idea pairs:

  - per pair we score three 1-5 SIMILARITY features (problem/insight/motivation)
    and invert+weight them into a single distance d(u_i,u_j) in [0,1] (5==same
    -> 0, 1==unrelated -> 1; weights motivation 0.1 / problem 0.3 / insight 0.6).
  - the response's OVERALL diversity = mean over all pairs of d(u_i,u_j).
  - we ALSO report per-field diversity (mean over pairs of each field's inverted
    distance) so you can see whether the spread is in the problem, the insight,
    or the motivation.

Input formats mirror the other judges:
  ideation-answer  a prediction JSONL from inference/run_inference.py: each line
                   has answer_str (or raw_output) with K ideas per sample. We
                   parse the ideas (common.ideas) and score each (record, sample)
                   group's ideas against each other. A response with <2 parseable
                   ideas gets diversity=None (no pairs to compare).

Run:
  python grpo_training/eval/diversity_judge.py \
      --dataset preds.jsonl --out /tmp/div.jsonl \
      --input-format ideation-answer --backend vllm
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_GRPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GRPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import clients, distance, distance_si, ideas, ideas_si  # noqa: E402

DEFAULT_MODELS = {"vllm": "openai/gpt-oss-120b", "gateway": "claude-sonnet-4-6"}
TEMPERATURE = 0.0
MAX_TOKENS = 1536
MAX_CONCURRENT = 5

# The per-field distance keys we report (bare field names, mirroring the weights).
_FIELDS = ("problem", "insight", "motivation")           # v0 rubric
_FIELDS_SI = ("problem_gap", "core_claim", "method")     # SI (v1) rubric


def rubric_for(si: bool) -> Tuple[Any, Tuple[str, ...]]:
    """(rubric module, bare field names) for the SI or v0 pairwise-distance rubric.

    `si=True` MUST select common.distance_si, not common.distance. Both modules
    expose the same four-function interface (build_messages / extract_json /
    parse_features / weighted_field_distance / feature_to_distance / RUBRIC_VERSION),
    so the wrong one does not raise -- it silently asks the wrong question and the
    run completes with plausible numbers:

      - the SI branch of _load_groups already renders 3-field SI cards
        (Problem gap / Core claim / Method), so the v0 rubric would ask the judge
        to rate `motivation_sim` -- outward-facing stakes -- on a card that has no
        motivation field, and parse_features returns None on the missing key for
        any judge that answers honestly (-> distance=None, dropped from the mean).
      - v0 weights insight 0.6 / problem 0.3 / motivation 0.1; distance_si weights
        its three slots EQUALLY. So even when the v0 keys are echoed back, the
        reduction differs from the one the SI rewards
        (rewards/f9b_impdist_len_v{6,10}_si_bedrock.py, darling_group_v10_si_bedrock.py)
        train against -- they all rebind their `distance` global to distance_si.
        Scoring an SI run under v0 measures it on a rubric it never optimized for.

    This mirrors those reward shims' `_v6.distance = distance_si` rebind, so the
    offline diversity metric and the training signal share ONE definition."""
    return (distance_si, _FIELDS_SI) if si else (distance, _FIELDS)


# --------------------------------------------------------------------------- #
# Dataset loading -> one GROUP per (record, sample), each holding its K idea cards
# --------------------------------------------------------------------------- #
def _load_groups(path: Path, *, k: Optional[int], si: bool = False) -> List[Dict[str, Any]]:
    """Expand a prediction JSONL into per-response idea groups.

    Diversity is set-level, so unlike the per-idea judges we DON'T flatten to one
    instance per idea -- we keep each (record, sample)'s ideas together. Returns a
    list of {rec_id, sample_idx, cards:[idea card_text, ...]}.

    `si=True` selects the SI (v1) parser: SI cards are {problem_gap, core_claim,
    method, inspiration}, which common.ideas.parse_ideas rejects wholesale (0
    ideas -> 0 pairs -> diversity n_scored=0). ideas_si.parse_ideas_si builds
    SIIdea (whose card_text renders the 3-field card), mirroring the per-idea
    judges' ideation-si branch in novelty_judge.load_instances."""
    parse = ideas_si.parse_ideas_si if si else ideas.parse_ideas
    groups: List[Dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            answers = rec.get("answer_str")
            raws = rec.get("raw_output")
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
                parsed = parse(gen, k=k)
                groups.append({
                    "rec_id": str(rec_id),
                    "sample_idx": s_idx,
                    "cards": [idea.card_text() for idea in parsed],
                })
    return groups


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
async def _call_judge(client, model: str, messages, *, temperature: float,
                      gen_config: dict) -> Tuple[str, Dict[str, Optional[int]]]:
    """Return (content, usage_dict). Mirrors novelty_judge._call_judge so both
    judges hit the client the same way (temperature skipped for gpt-5.x)."""
    call_kwargs: Dict[str, Any] = {"model": model, "messages": messages, **gen_config}
    if clients.accepts_temperature(model):
        call_kwargs["temperature"] = temperature
    resp = await client.chat.completions.create(**call_kwargs)
    return resp.choices[0].message.content or "", clients.usage_dict(resp)


async def _score_pair(card_a: str, card_b: str, i: int, j: int, *, client, model,
                      temperature, gen_config, pair_sem,
                      rubric=distance, fields=_FIELDS) -> Dict[str, Any]:
    """Score one idea pair -> per-field distances + weighted overall d(u_i,u_j).
    Retries once on an unparsed reply; a persistent failure yields distance=None
    (the caller drops it from the mean rather than defaulting, so a judge failure
    doesn't spuriously inflate OR collapse diversity).

    `rubric`/`fields` come from rubric_for(si): distance + _FIELDS for v0,
    distance_si + _FIELDS_SI for SI. `f"{field}_sim"` reconstructs the feature key
    under both (problem->problem_sim, problem_gap->problem_gap_sim)."""
    messages = rubric.build_messages(card_a, card_b)
    rec: Dict[str, Any] = {"i": i, "j": j, "features": None, "distance": None,
                           "by_field": None, "error": None,
                           "prompt": "\n".join(f"[{m['role']}]\n{m['content']}"
                                               for m in messages),
                           "usage": clients.usage_dict(None)}
    async with pair_sem:
        for _ in range(2):  # retry once on unparsed
            try:
                raw, usage = await _call_judge(client, model, messages,
                                               temperature=temperature,
                                               gen_config=gen_config)
                rec["usage"] = clients.add_usage(rec["usage"], usage)
                obj = rubric.extract_json(raw)
                feats = rubric.parse_features(obj) if isinstance(obj, dict) else None
                if feats is not None:
                    rec["features"] = feats
                    rec["distance"] = rubric.weighted_field_distance(feats)
                    rec["by_field"] = {
                        field: rubric.feature_to_distance(int(feats[f"{field}_sim"]))
                        for field in fields}
                    rec["error"] = None
                    return rec
                rec["error"] = "unparsed"
            except Exception as e:  # noqa: BLE001 -- record, don't crash the batch
                rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def _aggregate(pairs: List[Dict[str, Any]], n_ideas: int,
               fields: Tuple[str, ...] = _FIELDS) -> Dict[str, Any]:
    """Reduce a response's scored pairs -> overall + per-field diversity.
    overall = mean over pairs of d(u_i,u_j); by_field[f] = mean over pairs of the
    field's inverted distance. None distances (unparsed pairs) are excluded."""
    scored = [p for p in pairs if p.get("distance") is not None]
    if not scored:
        return {"mean_pairwise": None, "min_pairwise": None, "max_pairwise": None,
                "by_field": {f: None for f in fields},
                "n_ideas": n_ideas, "n_pairs": len(pairs), "n_pairs_scored": 0}
    dists = [p["distance"] for p in scored]
    by_field: Dict[str, Optional[float]] = {}
    for field in fields:
        vals = [p["by_field"][field] for p in scored
                if p.get("by_field") and p["by_field"].get(field) is not None]
        by_field[field] = (sum(vals) / len(vals)) if vals else None
    return {"mean_pairwise": sum(dists) / len(dists),
            "min_pairwise": min(dists), "max_pairwise": max(dists),
            "by_field": by_field,
            "n_ideas": n_ideas, "n_pairs": len(pairs), "n_pairs_scored": len(scored)}


async def score_group(group: Dict[str, Any], *, client, model, temperature,
                      gen_config, pair_sem, prices: Dict[str, Dict[str, Optional[float]]],
                      keep_pairs: bool = True, si: bool = False) -> Dict[str, Any]:
    """Score one (record, sample) response's diversity over all C(K,2) idea pairs.
    A response with <2 parseable ideas has no pairs -> diversity=None.

    `si=True` selects the distance_si rubric -- it MUST match the `si=` passed to
    _load_groups (see rubric_for). The emitted rubric_version records which one ran,
    so a mismatched run is identifiable after the fact rather than silent."""
    rubric, fields = rubric_for(si)
    cards = group["cards"]
    pairs_idx = list(combinations(range(len(cards)), 2))
    pair_recs = await asyncio.gather(*[
        _score_pair(cards[i], cards[j], i, j, client=client, model=model,
                    temperature=temperature, gen_config=gen_config, pair_sem=pair_sem,
                    rubric=rubric, fields=fields)
        for i, j in pairs_idx])
    agg = _aggregate(pair_recs, n_ideas=len(cards), fields=fields)

    usage = clients.add_usage(*[p.get("usage") or clients.usage_dict(None)
                                for p in pair_recs]) if pair_recs else clients.usage_dict(None)
    price = prices.get(model) or {}
    cost = clients.token_cost(usage, price_in=price.get("in"), price_out=price.get("out"))
    diversity = {**agg, "rubric_version": rubric.RUBRIC_VERSION}
    if keep_pairs:
        diversity["pairs"] = [{k: v for k, v in p.items()
                               if k not in ("prompt", "usage")} for p in pair_recs]
    return {
        "instance_id": group["rec_id"],
        "sample_idx": group["sample_idx"],
        "diversity": diversity,
        "meta": {"diversity": {"model": model, "usage": usage, "cost_usd": cost}},
    }


async def run_async(dataset: Path, out: Path, *, backend: str, model: str,
                    input_format: str, temperature: float, max_tokens: int,
                    max_concurrent: int, k: Optional[int], limit: Optional[int],
                    judge_url: Optional[str],
                    prices: Optional[Dict[str, Dict[str, Optional[float]]]] = None) -> dict:
    if input_format not in ("ideation-answer", "ideation-si"):
        raise SystemExit("diversity_judge only supports --input-format ideation-answer "
                         "/ ideation-si (it needs the K ideas of each response together).")
    prices = prices or {}
    si = (input_format == "ideation-si")
    rubric, _fields = rubric_for(si)
    print(f"[diversity] rubric={rubric.RUBRIC_VERSION} (si={si})")
    groups = _load_groups(dataset, k=k, si=si)
    if limit is not None:
        groups = groups[:limit]

    out.parent.mkdir(parents=True, exist_ok=True)
    client = clients.build_async_client(backend, model, base_url=judge_url)
    gen_config = clients.generation_config(model, max_tokens)
    group_sem = asyncio.Semaphore(max_concurrent)
    pair_sem = asyncio.Semaphore(max_concurrent)

    async def one(idx: int, group: Dict[str, Any]) -> Dict[str, Any]:
        async with group_sem:
            tag = f"{group['rec_id']}::s{group['sample_idx']}"
            rec = await score_group(group, client=client, model=model,
                                    temperature=temperature, gen_config=gen_config,
                                    pair_sem=pair_sem, prices=prices, si=si)
            d = rec["diversity"]
            mp = d["mean_pairwise"]
            print(f"[{idx}/{len(groups)}] {tag}  ideas={d['n_ideas']} "
                  f"pairs={d['n_pairs_scored']}/{d['n_pairs']} "
                  f"mean_div={mp if mp is None else round(mp, 3)}")
            return {"model": model, "backend": backend,
                    "timestamp": datetime.now().isoformat(), **rec}

    records = [await t for t in
               [asyncio.create_task(one(i + 1, g)) for i, g in enumerate(groups)]]

    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(records)
    print("\n=== diversity ===")
    print(f"  responses:        {summary['n_responses']}")
    print(f"  scored:           {summary['n_scored']} "
          f"(>=2 parseable ideas)")
    md = summary["mean_diversity"]
    if md is not None:
        print(f"  mean diversity:   {md:.3f}  (0=all identical, 1=all unrelated)")
    print(f"  by field:         {summary['mean_by_field']}")
    print(f"  -> {out}  ({len(records)} response rows)")
    return summary


def summarize(records: List[Dict[str, Any]], si: Optional[bool] = None) -> Dict[str, Any]:
    """Corpus-level rollup over per-response diversity records.

    `si=None` INFERS the rubric from the records' own `rubric_version` rather than
    taking it from the caller, so run_all_evals (which calls summarize(records) with
    no flag) cannot pair SI per-response records with v0 field names and report
    by_field as all-None. Explicit si=True/False overrides."""
    if si is None:
        versions = {(r.get("diversity") or {}).get("rubric_version") for r in records}
        si = any(v == distance_si.RUBRIC_VERSION for v in versions if v)
    rubric, fields = rubric_for(si)
    scored = [r for r in records if r["diversity"]["mean_pairwise"] is not None]
    mean_div = (sum(r["diversity"]["mean_pairwise"] for r in scored) / len(scored)
                if scored else None)
    by_field: Dict[str, Optional[float]] = {}
    for field in fields:
        vals = [r["diversity"]["by_field"].get(field) for r in scored
                if r["diversity"]["by_field"].get(field) is not None]
        by_field[field] = (sum(vals) / len(vals)) if vals else None
    return {"n_responses": len(records), "n_scored": len(scored),
            "mean_diversity": mean_div, "mean_by_field": by_field,
            "rubric_version": rubric.RUBRIC_VERSION}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score per-response idea diversity (pairwise distance).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=clients.VALID_BACKENDS, default="vllm")
    ap.add_argument("--model", default=None,
                    help="judge model; default per backend")
    ap.add_argument("--input-format", choices=["ideation-answer", "ideation-si"],
                    default="ideation-answer")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--k", type=int, default=None,
                    help="max ideas per response to score (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N responses (debug)")
    ap.add_argument("--judge-url", default=None,
                    help="explicit vLLM URL (else discovered via endpoint file / env)")
    ap.add_argument("--price", action="append", default=[], metavar="MODEL:IN,OUT",
                    help="per-model $/1M-token rates for cost accounting (repeatable)")
    args = ap.parse_args()
    model = args.model or DEFAULT_MODELS[args.backend]
    prices = _parse_prices(args.price)
    asyncio.run(run_async(
        Path(args.dataset), Path(args.out), backend=args.backend, model=model,
        input_format=args.input_format, temperature=args.temperature,
        max_tokens=args.max_tokens, max_concurrent=args.max_concurrent,
        k=args.k, limit=args.limit, judge_url=args.judge_url, prices=prices))


def _parse_prices(specs: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
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
