"""Dual-head grounding judge for *reporting* (not training rewards).

Two separate prompts, validated in human_val:

  1. load_bearing     -- LB_ONLY_TEMPLATE (no source docs; think0 recommended).
  2. support_faithful -- SF_ONLY_TEMPLATE (docs required), the VAGUENESS anchors.

`source_exists` is deliberately NOT scored. It was measured flat and
uncorrelated with the failure this judge exists to catch (grounding.py's
features_to_g comment block: early->late 2.45 -> 2.38, corr -0.045; it is >=3 on
98% of ideas, so every gate that ANDed it was a no-op). Re-add later by restoring
grounding.TEMPLATE's source_exists block here and putting the key back in
_SF_REQUIRED.

Head B does not score load_bearing at all -- the LB_ONLY head owns that column,
and asking for a second, differently-conditioned LB only invited the judge to
emit an anchor-less number that combine_heads then threw away.

Does NOT edit common.grounding. Training / F9-B rewards keep using grounding.py.

RUBRIC_VERSION bumps when either prompt text changes so eval checkpoints
invalidate correctly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from common import grounding

RUBRIC_VERSION = "new-grounding-dual-v1"

# --------------------------------------------------------------------------- #
# Head A: load-bearing only, no documents (human_val lb_only_nodocs).
# --------------------------------------------------------------------------- #
LB_ONLY_SYSTEM = (
    "You are an expert research scientist auditing whether a research idea's "
    "central insight is LOAD-BEARING on the grounding the author declared. "
    "You do NOT see the source documents — judge from the insight and the "
    "declared grounding prose alone. Be skeptical of decorative citations "
    "(generic gap / connect / extend language with no concrete finding). "
    "Return only a single JSON object."
)

LB_ONLY_TEMPLATE = """Below is a research INSIGHT and the GROUNDING the author declared for it
(SI `inspiration` prose: which sources, what they are said to support).

Score ONE feature, 1-5. USE THE FULL SCALE.

  load_bearing -- would the insight COLLAPSE if the cited documents were
    removed? (decorative citation test)
      1 = decorative: the insight stands completely without the cited docs
      2 = the docs add color but the insight does not depend on them
      3 = the docs support the insight but it could be re-grounded elsewhere
      4 = the insight largely depends on the specific cited findings
      5 = the insight strictly requires the cited docs; remove them and it falls apart

Judge from the text alone: does the declared grounding name concrete findings /
methods / results that the insight needs, or is it template rhetoric that would
read the same with the citations deleted?

=== INSIGHT ===
{insight_claim}

=== DECLARED GROUNDING ===
{grounding_block}

Return ONLY a JSON object of the form:
{{
  "load_bearing": 1 | 2 | 3 | 4 | 5,
  "rationale": "1-3 sentences on whether the grounding is decorative or load-bearing."
}}
Do not output anything outside the JSON object."""


def build_lb_only_messages(insight_claim: str, grounding_text: str
                           ) -> List[Dict[str, str]]:
    prompt = LB_ONLY_TEMPLATE.format(
        insight_claim=(insight_claim or "").strip() or "(no insight provided)",
        grounding_block=grounding_text or "(no grounding provided)",
    )
    return [{"role": "system", "content": LB_ONLY_SYSTEM},
            {"role": "user", "content": prompt}]


def parse_lb_only(raw: str) -> Optional[int]:
    obj = grounding.extract_json(raw)
    if not isinstance(obj, dict):
        return None
    return grounding.clamp_score(obj.get("load_bearing"))


# --------------------------------------------------------------------------- #
# Head B: support_faithful ONLY, docs on, VAGUENESS anchors.
# Anchor text matches human_val/score/ab_rubric_load_bearing.py::VAGUENESS.
# --------------------------------------------------------------------------- #
# Written out in full rather than spliced out of grounding.TEMPLATE by string
# index: the splice cut from "  support_faithful --" to "=== INSIGHT ===", which
# swallowed the load_bearing anchors while leaving "load_bearing" in the JSON
# schema and {n_features} reading "three" for a two-feature prompt.
SF_ONLY_SYSTEM = (
    "You are an expert research scientist auditing whether a research idea's "
    "declared grounding FAITHFULLY represents the source documents it cites. "
    "You are given the insight, the grounding the author declared, and the full "
    "source documents. Judge fidelity of grounding -- not novelty, quality, or "
    "impact. Be skeptical of claims too vague to check against a source. "
    "Return only a single JSON object."
)

VAGUENESS_BLOCK = """  support_faithful -- extract the SPECIFIC factual/mechanistic claim the
    GROUNDING text makes about each cited source, then check it against that
    source's actual content. A claim too generic/vague to check is a FAILURE
    here, not a safe default -- vagueness is not faithfulness.
      1 = misrepresents the source (the source does not say this), OR the
          claim is too generic/vague to verify against the source at all
      2 = weak/strained connection to what the source actually shows
      3 = partially supported; the source backs some but not all of the claim
      4 = well supported with a minor gap or overstatement
      5 = a SPECIFIC, checkable claim that the source clearly and fully
          establishes
"""

SF_ONLY_TEMPLATE = """Below is a research INSIGHT, the GROUNDING the author declared for it (one or
more source references with what each is said to support), and the full
SOURCE DOCUMENTS available to the author.

Judge how faithfully the insight is grounded in the cited sources. Score ONE
feature, 1-5. USE THE FULL SCALE.

{vagueness_block}
=== INSIGHT ===
{insight_claim}

=== DECLARED GROUNDING ===
{grounding_block}

=== SOURCE DOCUMENTS ===
{documents_block}

Return ONLY a JSON object of the form:
{{
  "support_faithful": 1 | 2 | 3 | 4 | 5,
  "rationale": "1-3 sentences citing which source supports (or fails to support) the insight."
}}
Do not output anything outside the JSON object."""

_SF_REQUIRED = ("support_faithful",)


def build_faithfulness_messages(
    insight_claim: str,
    grounding_text: str,
    documents: List[str],
    grounding_type: Optional[str] = None,
    *,
    ask_type: Optional[bool] = None,
) -> List[Dict[str, str]]:
    """Docs-on support_faithful prompt with the vagueness anchors.

    `grounding_type` / `ask_type` are accepted for call-site compatibility and
    ignored: this head scores neither type_correct nor load_bearing, so there is
    no type block to splice. Head A owns load_bearing.
    """
    prompt = SF_ONLY_TEMPLATE.format(
        vagueness_block=VAGUENESS_BLOCK,
        insight_claim=(insight_claim or "").strip() or "(no insight provided)",
        grounding_block=grounding_text or "(no grounding provided)",
        documents_block=grounding.documents_block(documents),
    )
    return [{"role": "system", "content": SF_ONLY_SYSTEM},
            {"role": "user", "content": prompt}]


def parse_faithfulness(raw: str, *, require_type: bool = False
                       ) -> Optional[Dict[str, Any]]:
    """Parse support_faithful (+ rationale). None if SF is missing/out of range.

    Deliberately NOT grounding.parse_features: that requires all of
    source_exists / support_faithful / load_bearing and returns None if any is
    absent, so it would reject every reply to this one-feature prompt.
    `require_type` is accepted and ignored (this head never asks for the type).
    """
    obj = grounding.extract_json(raw)
    if not isinstance(obj, dict):
        return None
    feats: Dict[str, Any] = {}
    for k in _SF_REQUIRED:
        s = grounding.clamp_score(obj.get(k))
        if s is None:
            return None
        feats[k] = s
    rat = obj.get("rationale")
    if isinstance(rat, str) and rat.strip():
        feats["rationale"] = rat.strip()
    return feats


def combine_heads(
    lb: Optional[int],
    faith_feats: Optional[Dict[str, Any]],
    grounding_type: Optional[str],
    *,
    require_type: bool = False,
    lb_error: Optional[str] = None,
    faith_error: Optional[str] = None,
    sf_requested: bool = True,
) -> Dict[str, Any]:
    """Merge dual-head outputs into the same shape as run_all_evals grounding rows.

    `features` carries LB from head A and SF from head B. `g` / fidelity use the
    combined feature dict (LB-centric gate unchanged).

    `sf_requested=False` means head B was NEVER CALLED (--new-grounding-no-sf), so
    a missing SF is expected, not a judge failure: it must not set `error`, or
    every row reads as half-broken and the resume path can't tell a deliberate
    LB-only run from an outage.
    """
    feats: Optional[Dict[str, Any]] = None
    if lb is not None or faith_feats is not None:
        feats = {}
        if faith_feats:
            for k in ("support_faithful", "rationale"):
                if k in faith_feats:
                    feats[k] = faith_feats[k]
        if lb is not None:
            feats["load_bearing"] = lb

    err = None
    if not sf_requested:
        err = lb_error or ("lb_only_failed" if lb is None else None)
    elif lb is None and faith_feats is None:
        err = lb_error or faith_error or "both_heads_failed"
    elif lb is None:
        err = lb_error or "lb_only_failed"
    elif faith_feats is None:
        err = faith_error or "faithfulness_failed"

    g = (grounding.features_to_g(feats, grounding_type, require_type=require_type)
         if feats and feats.get("load_bearing") is not None else None)
    return {
        "load_bearing": lb,
        "g": g,
        "overall": grounding.overall_score(feats, grounding_type) if feats else None,
        "fidelity": grounding.fidelity(feats) if feats else None,
        "grounding_type": grounding_type,
        "features": feats,
        "error": err,
        "rubric_version": RUBRIC_VERSION,
        "heads": {
            "lb_only": {"load_bearing": lb, "error": lb_error},
            "faithfulness_vagueness": {
                "support_faithful": None if not faith_feats else faith_feats.get("support_faithful"),
                "source_exists": None,  # not scored; see module docstring
                "error": faith_error,
                "skipped": not sf_requested,
            },
        },
    }


def gemini_thinking_extra(model: str, thinking_budget: Optional[int]
                          ) -> Dict[str, Any]:
    """extra_body for Gemini thinkingBudget; {} if N/A or budget is None (default)."""
    if thinking_budget is None:
        return {}
    if "gemini" not in (model or "").lower():
        return {}
    return {"extra_body": {"thinkingConfig": {"thinkingBudget": int(thinking_budget)}}}
