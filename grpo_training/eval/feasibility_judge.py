#!/usr/bin/env python3
"""Feasibility evaluator (EVAL-ONLY): how much would executing an idea cost?

Given an IDEA-ONLY description (problem / claim / method, no results or hardware)
the judge predicts two anchored 1-5 tiers -- accelerator compute (A100-GPU-hours)
and human-evaluation effort (person-hours). This is an evaluation-only signal: it
does NOT feed the F9-B reward, so -- unlike novelty/grounding -- its rubric and prompt live
next to this judge (feasibility_prompts.py) rather than in grpo_training/common/
(which is reserved for reward-shared code). The validation harness under
reward/validation/feasibility/ imports the prompt from here. This file is the
runner + CLI.

Backends: --backend gateway --model claude-opus-4-8 | gemini-2.5-flash
          --backend vllm (any self-hosted OpenAI-compatible model)

Input formats (--input-format):
  summaries        a directory of <arxiv_id>.json render_templates records; the
                   idea text is each record's `abstract` (rendered T0-T3 fields).
                   This is what reward/validation/feasibility validates against.
  eval-set         JSON list of {instance_id, idea_text} (or {idea:{...}} cards).
  ideation-answer  prediction JSONL from inference/run_inference.py; parses the K
                   ideas per record (common.ideas) and judges each idea card.

Run:
  # validate the prompt against the 2025 target summaries
  python eval/feasibility_judge.py \
      --dataset reward/validation/feasibility/data/summaries \
      --input-format summaries --backend gateway --model claude-opus-4-8 \
      --out reward/validation/feasibility/data/pred.opus.jsonl

  python eval/feasibility_judge.py --dataset preds.jsonl \
      --input-format ideation-answer --backend gateway \
      --model gemini-2.5-flash --out feas.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_EVAL_DIR = Path(__file__).resolve().parent
_GRPO_ROOT = _EVAL_DIR.parent
_REPO_ROOT = _GRPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_GRPO_ROOT), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import clients, ideas  # noqa: E402
import feasibility_prompts as feas  # noqa: E402  (sibling: grpo_training/eval/)

DEFAULT_MODELS = {"vllm": "openai/gpt-oss-120b", "gateway": "claude-opus-4-8"}
TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_CONCURRENT = 5


# --------------------------------------------------------------------------- #
# Dataset loading -> uniform instances: {instance_id, idea_id, idea_text, domain}
# --------------------------------------------------------------------------- #
def _load_summaries(path: Path) -> List[Dict[str, Any]]:
    """A directory of render_templates <arxiv_id>.json records; idea text is the
    rendered `abstract`. Skips errored / empty records."""
    if not path.is_dir():
        raise ValueError(f"--input-format summaries expects a directory, got {path}")
    out: List[Dict[str, Any]] = []
    for fp in sorted(path.glob("*.json")):
        if fp.name.startswith("_") or fp.name == "render_summary.json":
            continue
        try:
            rec = json.loads(fp.read_text())
        except Exception:
            continue
        idea_text = (rec.get("abstract") or "").strip()
        if rec.get("error") or not idea_text:
            continue
        aid = rec.get("arxiv_id") or fp.stem
        out.append({"instance_id": aid, "idea_id": aid, "idea_text": idea_text,
                    "domain": rec.get("domain"), "template": rec.get("template")})
    return out


def _load_eval_set(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: --input-format eval-set expects a JSON list.")
    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(data):
        iid = rec.get("instance_id") or f"#{i}"
        text = rec.get("idea_text")
        if not text and isinstance(rec.get("idea"), dict):
            parsed = ideas.parse_ideas(json.dumps([rec["idea"]]), k=1)
            text = parsed[0].card_text() if parsed else ""
        text = (text or "").strip()
        if not text:
            continue
        out.append({"instance_id": iid, "idea_id": rec.get("id") or iid,
                    "idea_text": text, "domain": rec.get("domain")})
    return out


def _load_ideation_answers(path: Path, *, k: Optional[int]) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gens: List[str] = []
            for src in (rec.get("answer_str"), rec.get("raw_output")):
                if isinstance(src, list):
                    gens.extend([s for s in src if isinstance(s, str)]); break
                if isinstance(src, str):
                    gens.append(src); break
            rec_id = rec.get("target_arxiv_id") or rec.get("index") or f"rec{line_no}"
            for s_idx, gen in enumerate(gens):
                for i_idx, idea in enumerate(ideas.parse_ideas(gen, k=k)):
                    instances.append({
                        "instance_id": f"{rec_id}::s{s_idx}::idea{i_idx}",
                        "idea_id": f"{rec_id}-s{s_idx}-i{i_idx}",
                        "idea_text": idea.card_text(),
                        "domain": rec.get("domain"),
                    })
    return instances


def load_instances(path: Path, *, input_format: str, k: Optional[int]) -> List[Dict[str, Any]]:
    if input_format == "summaries":
        return _load_summaries(path)
    if input_format == "eval-set":
        return _load_eval_set(path)
    if input_format == "ideation-answer":
        return _load_ideation_answers(path, k=k)
    raise ValueError(f"unknown --input-format {input_format!r}")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced {...} object out of a raw completion."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_prediction(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate a judge JSON into {compute_tier, human_eval_tier, *_estimate,
    rationale}. Returns None if either tier is missing / out of the 1-5 range."""
    if not isinstance(obj, dict):
        return None
    if not (feas.valid_tier(obj.get("compute_tier"))
            and feas.valid_tier(obj.get("human_eval_tier"))):
        return None
    return {
        "compute_tier": int(obj["compute_tier"]),
        "human_eval_tier": int(obj["human_eval_tier"]),
        "compute_estimate": str(obj.get("compute_estimate") or "").strip(),
        "human_eval_estimate": str(obj.get("human_eval_estimate") or "").strip(),
        "rationale": str(obj.get("rationale") or "").strip(),
    }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
async def _score_one(inst: Dict[str, Any], *, client, model, temperature,
                     gen_config) -> Dict[str, Any]:
    """Predict the two feasibility tiers for one idea. Mirrors the shape the
    other eval judges return (parsed fields + error + prompt + usage)."""
    messages = feas.build_judge_messages(inst.get("idea_text", ""))
    call_kwargs: Dict[str, Any] = {"model": model, "messages": messages, **gen_config}
    if clients.accepts_temperature(model):
        call_kwargs["temperature"] = temperature
    pred = None
    error = None
    usage = clients.usage_dict(None)
    prompt = "\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
    for _ in range(2):  # retry once on unparsed
        try:
            resp = await client.chat.completions.create(**call_kwargs)
            usage = clients.add_usage(usage, clients.usage_dict(resp))
            pred = parse_prediction(_extract_json(resp.choices[0].message.content or ""))
            if pred is not None:
                error = None
                break
            error = "unparsed"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
    fields = pred or {"compute_tier": None, "human_eval_tier": None,
                      "compute_estimate": "", "human_eval_estimate": "",
                      "rationale": ""}
    return {**fields, "error": error, "prompt": prompt, "usage": usage}


async def run_async(dataset: Path, out: Path, *, backend: str, model: str,
                    input_format: str, temperature: float, max_tokens: int,
                    max_concurrent: int, k: Optional[int], limit: Optional[int]) -> dict:
    instances = load_instances(dataset, input_format=input_format, k=k)
    if limit is not None:
        instances = instances[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)

    client = clients.build_async_client(backend, model)
    gen_config = clients.generation_config(model, max_tokens)
    sem = asyncio.Semaphore(max_concurrent)

    async def one(idx: int, inst: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            iid = inst.get("instance_id") or f"#{idx}"
            print(f"[{idx}/{len(instances)}] start  {iid}")
            res = await _score_one(inst, client=client, model=model,
                                   temperature=temperature, gen_config=gen_config)
            print(f"[{idx}/{len(instances)}] done   {iid}  "
                  f"compute={res['compute_tier']}  human_eval={res['human_eval_tier']}"
                  + (f"  ERR({res['error']})" if res.get("error") else ""))
            # keep records lean: drop the (large) rendered prompt from the row
            res_clean = {kk: vv for kk, vv in res.items() if kk != "prompt"}
            return {
                "instance_id": inst.get("instance_id"),
                "idea_id": inst.get("idea_id"),
                "domain": inst.get("domain"),
                "model": model, "backend": backend,
                "scale_version": feas.SCALE_VERSION,
                "timestamp": datetime.now().isoformat(), **res_clean,
            }

    records = [await t for t in
               [asyncio.create_task(one(i + 1, inst)) for i, inst in enumerate(instances)]]

    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    scored = [r for r in records if r.get("compute_tier") is not None]
    def _mean(key):
        vals = [r[key] for r in scored if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    summary = {"n_instances": len(records), "n_scored": len(scored),
               "mean_compute_tier": _mean("compute_tier"),
               "mean_human_eval_tier": _mean("human_eval_tier"),
               "model": model, "backend": backend,
               "scale_version": feas.SCALE_VERSION}
    print("\n=== feasibility judge ===")
    print(f"  instances: {summary['n_instances']}   scored: {summary['n_scored']}")
    if summary["mean_compute_tier"] is not None:
        print(f"  mean compute tier:    {summary['mean_compute_tier']:.2f}  (1-5)")
        print(f"  mean human-eval tier: {summary['mean_human_eval_tier']:.2f}  (1-5)")
    print(f"  -> {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Feasibility (compute + human-eval) LLM judge.")
    ap.add_argument("--dataset", required=True,
                    help="summaries dir | eval-set JSON | ideation-answer JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=clients.VALID_BACKENDS, default="gateway")
    ap.add_argument("--model", default=None,
                    help="default: claude-opus-4-8 (gateway) / gpt-oss-120b (vllm)")
    ap.add_argument("--input-format",
                    choices=["summaries", "eval-set", "ideation-answer"],
                    default="summaries")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--k", type=int, default=ideas.DEFAULT_K)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    model = args.model or DEFAULT_MODELS[args.backend]
    asyncio.run(run_async(
        Path(args.dataset), Path(args.out), backend=args.backend, model=model,
        input_format=args.input_format, temperature=args.temperature,
        max_tokens=args.max_tokens, max_concurrent=args.max_concurrent,
        k=args.k, limit=args.limit))


if __name__ == "__main__":
    main()
