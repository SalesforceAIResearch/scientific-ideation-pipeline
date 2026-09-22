#!/usr/bin/env python3
"""Novelty evaluator: is a target idea novel w.r.t. a population of source docs?

Async LLM-as-judge over a dataset, writing one JSONL verdict per instance. The
rubric / prompts / parsing / scoring all live in `common.novelty` (single source
of truth, shared with the F9-B reward); this file is just the runner + CLI.

Backends (--backend):
  vllm     self-hosted gpt-oss-120b (default on-cluster); endpoint via *_BASE_URL
           / endpoint file (see common.scijudge.resolve_base_url).
  gateway  hosted claude-sonnet-4-6 via the SF/eng-ai gateways (mirrors the Mac
           reference judge); needs the SF CA bundle + egress.

Input formats (--input-format):
  eval-set         the reference dataset: a JSON list of
                   {instance_id, idea:{...}, references:[{...}]}.
  ideation-answer  a prediction JSONL from inference/run_inference.py: each line
                   has answer_str (or raw_output) + input_documents; we parse the
                   K ideas (common.ideas) and emit one eval instance per idea,
                   scoring it against the record's source documents.

Run:
  python eval/novelty_judge.py --dataset eval_set.json --out verdicts.jsonl --backend vllm
  python eval/novelty_judge.py --dataset preds.jsonl  --out verdicts.jsonl \
      --input-format ideation-answer --mode batched --backend gateway
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make grpo_training importable as `common.*` whether run from repo root or here.
_GRPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GRPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import clients, ideas, ideas_si, novelty  # noqa: E402

DEFAULT_MODELS = {"vllm": "openai/gpt-oss-120b", "gateway": "claude-sonnet-4-6"}
TEMPERATURE = 0.0
MAX_TOKENS = 1536
MAX_CONCURRENT = 5


# --------------------------------------------------------------------------- #
# Dataset loading -> a uniform list of instances {instance_id, idea, references}
# --------------------------------------------------------------------------- #
def _load_eval_set(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: --input-format eval-set expects a JSON list.")
    return data


def _docs_from_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build reference-doc dicts from a prediction record. Prefers a structured
    `references` list; falls back to splitting the `input_documents` block
    (format_documents joins with 'Document N:' headers)."""
    refs = rec.get("references")
    if isinstance(refs, list) and refs:
        return refs
    block = rec.get("input_documents") or ""
    docs: List[Dict[str, Any]] = []
    if block:
        # split on the "Document N:" headers produced by prompts.format_documents
        import re
        parts = re.split(r"\n\s*Document\s+\d+\s*:\s*\n", "\n" + block)
        for i, body in enumerate([p for p in parts if p.strip()]):
            docs.append({"id": f"doc{i}", "title": f"Document {i + 1}",
                         "abstract": body.strip()})
    return docs


def _load_ideation_answers(path: Path, *, k: Optional[int],
                           si: bool = False) -> List[Dict[str, Any]]:
    """Expand a prediction JSONL into per-idea novelty instances.

    Each record may carry answer_str / raw_output as a list (one per sample) or a
    string. We parse every sample's K ideas and emit one instance per idea,
    scored against that record's source documents.

    `si=True` selects the SI (v1) parser: SI cards are {problem_gap, core_claim,
    method, inspiration}, which common.ideas.parse_ideas rejects wholesale (0
    ideas -> every judge scores None). ideas_si.parse_ideas_si builds SIIdea,
    which subclasses Idea with the SI fields mapped onto v0's slots, so
    idea_to_eval_card and every downstream judge keep working unchanged."""
    parse = ideas_si.parse_ideas_si if si else ideas.parse_ideas
    instances: List[Dict[str, Any]] = []
    refs_cache: Dict[int, List[Dict[str, Any]]] = {}
    with path.open() as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            refs = _docs_from_record(rec)
            refs_cache[line_no] = refs
            answers = rec.get("answer_str")
            raws = rec.get("raw_output")
            # normalize to a list of generation strings
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
                    instances.append({
                        "instance_id": f"{rec_id}::s{s_idx}::idea{i_idx}",
                        "idea": ideas.idea_to_eval_card(
                            idea, idea_id=f"{rec_id}-s{s_idx}-i{i_idx}"),
                        "references": refs,
                    })
    return instances


def load_instances(path: Path, *, input_format: str, k: Optional[int]) -> List[Dict[str, Any]]:
    if input_format == "eval-set":
        return _load_eval_set(path)
    if input_format == "ideation-answer":
        return _load_ideation_answers(path, k=k)
    if input_format == "ideation-si":
        return _load_ideation_answers(path, k=k, si=True)
    raise ValueError(f"unknown --input-format {input_format!r}")


# --------------------------------------------------------------------------- #
# Scoring one instance
# --------------------------------------------------------------------------- #
async def _call_judge(client, model: str, messages, *, temperature: float,
                      gen_config: dict):
    """Return (content, usage_dict). usage carries prompt/completion/total tokens
    so callers can record per-feature token metadata."""
    call_kwargs: Dict[str, Any] = {"model": model, "messages": messages, **gen_config}
    if clients.accepts_temperature(model):
        call_kwargs["temperature"] = temperature
    resp = await client.chat.completions.create(**call_kwargs)
    return resp.choices[0].message.content or "", clients.usage_dict(resp)


def _prompt_text(messages) -> str:
    """Flatten chat messages into a single string for the metadata log."""
    return "\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


def _empty_per_doc(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"ref_id": r.get("id"), "ref_title": r.get("title"),
             "role": r.get("role"), "features": None, "overlap": None,
             "error": "unparsed", "prompt": None,
             "usage": clients.usage_dict(None)} for r in refs]


async def _score_per_doc(idea, refs, *, client, model, temperature, gen_config,
                         doc_sem) -> List[Dict[str, Any]]:
    async def one(ref):
        async with doc_sem:
            messages = novelty.build_per_doc_messages(idea, ref)
            rec = {"ref_id": ref.get("id"), "ref_title": ref.get("title"),
                   "role": ref.get("role"), "features": None, "overlap": None,
                   "error": None, "prompt": _prompt_text(messages),
                   "usage": clients.usage_dict(None)}
            for _ in range(2):  # retry once on unparsed
                try:
                    raw, usage = await _call_judge(
                        client, model, messages,
                        temperature=temperature, gen_config=gen_config)
                    rec["usage"] = clients.add_usage(rec["usage"], usage)
                    obj = novelty.extract_json(raw)
                    feats = novelty.parse_features(obj) if isinstance(obj, dict) else None
                    if feats is not None:
                        rec["features"] = feats
                        rec["overlap"] = novelty.overlap(feats)
                        rec["error"] = None
                        return rec
                    rec["error"] = "unparsed"
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
            return rec

    return [await t for t in [asyncio.create_task(one(r)) for r in refs]]


async def _score_batched(idea, refs, *, client, model, temperature, gen_config
                         ) -> List[Dict[str, Any]]:
    per_doc = _empty_per_doc(refs)
    messages = novelty.build_batched_messages(idea, refs)
    prompt = _prompt_text(messages)
    try:
        raw, usage = await _call_judge(
            client, model, messages,
            temperature=temperature, gen_config=gen_config)
        # batched = ONE call for all docs; attribute the whole usage to doc 0 so
        # we don't double-count, and stamp the shared prompt on every doc.
        for i, d in enumerate(per_doc):
            d["prompt"] = prompt
            d["usage"] = usage if i == 0 else clients.usage_dict(None)
        obj = novelty.extract_json(raw)
        docs = obj.get("documents") if isinstance(obj, dict) else None
        if isinstance(docs, list):
            for entry in docs:
                if not isinstance(entry, dict):
                    continue
                idx = entry.get("index")
                if not isinstance(idx, int) or not (0 <= idx < len(refs)):
                    continue
                feats = novelty.parse_features(entry)
                if feats is not None:
                    per_doc[idx]["features"] = feats
                    per_doc[idx]["overlap"] = novelty.overlap(feats)
                    per_doc[idx]["error"] = None
    except Exception as e:
        for d in per_doc:
            d["error"] = f"{type(e).__name__}: {e}"
    return per_doc


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
async def run_async(dataset: Path, out: Path, *, backend: str, model: str,
                    mode: str, input_format: str, temperature: float,
                    max_tokens: int, max_concurrent: int, k: Optional[int],
                    limit: Optional[int]) -> dict:
    instances = load_instances(dataset, input_format=input_format, k=k)
    if limit is not None:
        instances = instances[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)

    client = clients.build_async_client(backend, model)
    gen_config = clients.generation_config(model, max_tokens)
    inst_sem = asyncio.Semaphore(max_concurrent if mode == "batched" else 1)
    doc_sem = asyncio.Semaphore(max_concurrent)

    async def one(idx: int, inst: Dict[str, Any]) -> Dict[str, Any]:
        async with inst_sem:
            iid = inst.get("instance_id") or f"#{idx}"
            refs = inst.get("references", [])
            print(f"[{idx}/{len(instances)}] start  {iid}  ({len(refs)} refs, {mode})")
            if mode == "batched":
                per_doc = await _score_batched(
                    inst["idea"], refs, client=client, model=model,
                    temperature=temperature, gen_config=gen_config)
            else:
                per_doc = await _score_per_doc(
                    inst["idea"], refs, client=client, model=model,
                    temperature=temperature, gen_config=gen_config, doc_sem=doc_sem)
            agg = novelty.aggregate(per_doc)
            ns = f"{agg['novelty_score']:.2f}" if agg["novelty_score"] is not None else "-"
            print(f"[{idx}/{len(instances)}] done   {iid}  novelty={ns}  "
                  f"({agg['n_scored']}/{agg['n_docs']} scored)")
            return {
                "instance_id": inst.get("instance_id"),
                "idea_id": inst["idea"].get("id"),
                "idea_title": inst["idea"].get("title"),
                "model": model, "backend": backend, "mode": mode,
                "rubric_version": novelty.RUBRIC_VERSION,
                "timestamp": datetime.now().isoformat(),
                "per_doc": per_doc, **agg,
            }

    records = [await t for t in
               [asyncio.create_task(one(i + 1, inst)) for i, inst in enumerate(instances)]]

    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    scored = [r for r in records if r.get("novelty_score") is not None]
    mean_novelty = (sum(r["novelty_score"] for r in scored) / len(scored)
                    if scored else None)
    summary = {"n_instances": len(records), "n_scored": len(scored),
               "mean_novelty_score": mean_novelty, "model": model,
               "backend": backend, "mode": mode,
               "rubric_version": novelty.RUBRIC_VERSION}
    print("\n=== novelty judge ===")
    print(f"  instances:    {summary['n_instances']}")
    print(f"  scored:       {summary['n_scored']}")
    print(f"  backend/mode: {backend} / {mode}")
    if mean_novelty is not None:
        print(f"  mean novelty: {mean_novelty:.3f}  (0=not novel, 1=fully novel)")
    print(f"  -> {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Population-based novelty LLM judge.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=clients.VALID_BACKENDS, default="vllm")
    ap.add_argument("--model", default=None,
                    help="judge model; defaults per backend (vllm: gpt-oss-120b, "
                         "gateway: claude-sonnet-4-6)")
    ap.add_argument("--mode", choices=["per-doc", "batched"], default="per-doc")
    ap.add_argument("--input-format", choices=["eval-set", "ideation-answer"],
                    default="eval-set")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--k", type=int, default=ideas.DEFAULT_K,
                    help="max ideas per generation to score (ideation-answer)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    model = args.model or DEFAULT_MODELS[args.backend]
    asyncio.run(run_async(
        Path(args.dataset), Path(args.out), backend=args.backend, model=model,
        mode=args.mode, input_format=args.input_format, temperature=args.temperature,
        max_tokens=args.max_tokens, max_concurrent=args.max_concurrent,
        k=args.k, limit=args.limit))


if __name__ == "__main__":
    main()
