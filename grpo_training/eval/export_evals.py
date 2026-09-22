#!/usr/bin/env python3
"""Unroll the nested ``*_with_evals.jsonl`` into a FLAT one-row-per-idea table for
human evaluation / validation.

The enriched JSONL keeps each input record intact with an ``evals`` field shaped
as a list-of-lists (``evals[sample_idx][idea_idx]`` -> one eval record). That is
faithful but unreadable as a spreadsheet (a single cell holds the whole nested
blob). This module flattens it: ONE row per (record, sample, idea), with the
candidate idea text, the gold idea, and every novelty / grounding / impact score
pulled up into plain columns.

Used two ways:
  - imported by ``run_all_evals.py`` (``export_rows`` / ``write_xlsx``) so the run
    writes this flat sheet instead of a dense, nested one.
  - standalone CLI to re-export an existing enriched JSONL:
        python eval/export_evals.py --in foo_with_evals.jsonl --out foo_flat.xlsx
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    # openpyxl's own definition of the chars Excel forbids -- reuse it so our
    # sanitizer stays in sync with what the writer will reject.
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except Exception:  # openpyxl absent -> write_xlsx returns None before we need it
    ILLEGAL_CHARACTERS_RE = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _idea_card(row: Dict[str, Any], sample_idx: int, idea_idx: int) -> Dict[str, Any]:
    """The parsed candidate idea (motivation/problem/insight/grounding_type) for a
    given (sample, idea), pulled from ``row['ideas']`` which runs parallel to
    ``row['evals']``. Returns {} when unavailable."""
    ideas = row.get("ideas")
    if not isinstance(ideas, list) or sample_idx >= len(ideas):
        return {}
    sample = ideas[sample_idx]
    if not isinstance(sample, list) or idea_idx >= len(sample):
        return {}
    card = sample[idea_idx]
    return card if isinstance(card, dict) else {}


def _grounding_list_str(card: Dict[str, Any]) -> Optional[str]:
    """Render the idea's declared grounding [{source, supports}] as readable text."""
    g = card.get("grounding")
    if not isinstance(g, list):
        return None
    parts = []
    for item in g:
        if isinstance(item, dict):
            src = item.get("source", "")
            sup = item.get("supports", "")
            parts.append(f"{src} -> {sup}")
    return " | ".join(parts) if parts else None


def _per_doc_features(nov: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The per-document 1-5 feature breakdown (problem/insight/motivation match +
    rationale) for every reference doc, tagged with the doc id/title. This is the
    detail behind novelty_score (which only keeps the max-overlap doc)."""
    out: List[Dict[str, Any]] = []
    for d in (nov.get("per_doc") or []):
        if not isinstance(d, dict):
            continue
        out.append({
            "ref_id": d.get("ref_id"),
            "ref_title": d.get("ref_title"),
            "overlap": d.get("overlap"),
            "features": d.get("features"),
        })
    return out


def _flatten_idea(row: Dict[str, Any], ev: Dict[str, Any],
                  sample_idx: int, idea_idx: int) -> Dict[str, Any]:
    """One flat row: identity + candidate idea text + gold + source docs + the
    selected novelty/grounding/impact result columns."""
    card = _idea_card(row, sample_idx, idea_idx)
    nov = ev.get("novelty") or {}
    grnd = ev.get("grounding") or {}
    imp = ev.get("impact") if isinstance(ev.get("impact"), dict) else {}
    feas = ev.get("feasibility") if isinstance(ev.get("feasibility"), dict) else {}
    snd = ev.get("soundness") if isinstance(ev.get("soundness"), dict) else {}
    con = ev.get("contribution") if isinstance(ev.get("contribution"), dict) else {}
    # which model scored each field (from the record's per-feature `meta`); None
    # for a field that wasn't scored or predates meta stamping.
    meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}

    def _judge_model(field: str) -> Optional[str]:
        m = meta.get(field)
        return m.get("model") if isinstance(m, dict) else None

    return {
        # --- identity ---
        "record_id": row.get("target_arxiv_id") or row.get("index"),
        "instance_id": ev.get("instance_id"),
        "idea_id": ev.get("idea_id"),
        "sample_idx": sample_idx,
        "idea_idx": idea_idx,
        # --- candidate idea (for the human rater) ---
        "cand_motivation": card.get("motivation"),
        "cand_problem": card.get("problem"),
        "cand_insight": card.get("insight_claim"),
        "cand_grounding_type": card.get("grounding_type"),
        "cand_grounding": _grounding_list_str(card),
        # full un-broken candidate idea (this sample/idea index) so the rater can
        # see EVERYTHING -- incl. the raw grounding list -- not just the split-out
        # fields above.
        "cand_raw": json.dumps(card, ensure_ascii=False) if card else None,
        # --- gold idea (title + idea text in ONE column) ---
        "gold_idea": "\n".join(
            x for x in (row.get("target_title"), row.get("target_idea_str")) if x),
        # --- source documents the idea was generated from ---
        "input_documents": row.get("input_documents"),
        # --- novelty: headline + the driving ref + the per-doc 1-5 features ---
        "novelty_score": nov.get("novelty_score"),
        "novelty_judge_model": _judge_model("novelty"),
        "novelty_most_similar_ref": json.dumps(
            nov.get("most_similar_ref"), ensure_ascii=False),
        "novelty_per_doc_features": json.dumps(
            _per_doc_features(nov), ensure_ascii=False),
        # --- grounding: headline + fidelity + the 4 features (1-5) + rationale ---
        "grounding_overall": grnd.get("overall"),
        "grounding_fidelity": grnd.get("fidelity"),
        "grounding_features": json.dumps(grnd.get("features"), ensure_ascii=False),
        # --- impact: the two ordering verdicts ---
        "impact_order1_ans": imp.get("order1_ans"),
        "impact_order2_ans": imp.get("order2_ans"),
        "impact_judge_model": _judge_model("impact"),
        # --- feasibility: the two 1-5 tiers + estimates + rationale ---
        "feasibility_compute_tier": feas.get("compute_tier"),
        "feasibility_human_eval_tier": feas.get("human_eval_tier"),
        "feasibility_compute_estimate": feas.get("compute_estimate"),
        "feasibility_human_eval_estimate": feas.get("human_eval_estimate"),
        "feasibility_rationale": feas.get("rationale"),
        "feasibility_judge_model": _judge_model("feasibility"),
        # --- soundness: ScholarEval mean-of-per-method score (0-10) ---
        "soundness_score": snd.get("soundness_score"),
        "soundness_n_methods": snd.get("n_methods"),
        "soundness_judge_model": _judge_model("soundness"),
        # --- contribution: idea novelty vs its source docs (-10..10, low=overlap) ---
        "contribution_avg_score": con.get("avg_score"),
        "contribution_min_score": con.get("min_score"),
        "contribution_min_score_document": con.get("min_score_document"),
        "contribution_n_sources": con.get("n_sources"),
        "contribution_judge_model": _judge_model("contribution"),
    }


def export_rows(enriched_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten enriched input rows (each with an ``evals`` list-of-lists) into a
    flat list of one-row-per-idea dicts, in (record, sample, idea) order."""
    flat: List[Dict[str, Any]] = []
    for row in enriched_rows:
        evals = row.get("evals") or []
        for sample_idx, sample in enumerate(evals):
            if not isinstance(sample, list):
                continue
            for idea_idx, ev in enumerate(sample):
                if isinstance(ev, dict):
                    flat.append(_flatten_idea(row, ev, sample_idx, idea_idx))
    return flat


def _xlsx_safe(value: Any) -> Any:
    """Strip XML-illegal control chars from strings so openpyxl won't raise
    IllegalCharacterError. Excel/openpyxl forbid most C0 control codes (except
    tab/newline/CR); model-generated idea & doc text occasionally contains them
    (e.g. stray \\x0b, \\x0c, or mojibake), which would otherwise abort the whole
    sheet write on a single bad cell. Non-strings pass through untouched."""
    if isinstance(value, str):
        return ILLEGAL_CHARACTERS_RE.sub("", value)
    return value


def write_xlsx(enriched_rows: List[Dict[str, Any]], out_path: Path) -> Optional[int]:
    """Write the flat one-row-per-idea sheet. Returns the row count, or None if
    openpyxl/pandas is missing (caller can warn and continue)."""
    flat = export_rows(enriched_rows)
    try:
        import pandas as pd
    except ImportError:
        return None
    flat = [{k: _xlsx_safe(v) for k, v in row.items()} for row in flat]
    pd.DataFrame(flat).to_excel(out_path, index=False)
    return len(flat)


def write_jsonl(enriched_rows: List[Dict[str, Any]], out_path: Path) -> int:
    """Write the flat rows as JSONL (no extra deps). Returns the row count."""
    flat = export_rows(enriched_rows)
    with out_path.open("w") as f:
        for r in flat:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(flat)


def _read_enriched(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unroll a *_with_evals.jsonl into a flat one-row-per-idea table.")
    ap.add_argument("--in", dest="inp", required=True,
                    help="enriched JSONL (input copy + per-row 'evals')")
    ap.add_argument("--out", required=True,
                    help="output path; .xlsx -> spreadsheet, else JSONL")
    args = ap.parse_args()
    rows = _read_enriched(Path(args.inp))
    out = Path(args.out)
    if out.suffix.lower() == ".xlsx":
        n = write_xlsx(rows, out)
        if n is None:
            raise SystemExit("openpyxl/pandas not installed; use a .jsonl --out "
                             "or `pip install openpyxl`.")
    else:
        n = write_jsonl(rows, out)
    print(f"wrote {n} idea rows -> {out}")


if __name__ == "__main__":
    main()
