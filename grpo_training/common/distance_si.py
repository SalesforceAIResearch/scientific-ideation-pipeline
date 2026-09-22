"""Pairwise SIMILARITY -> DISTANCE rubric for SI (v1) idea cards.

The v1 counterpart of `common/distance.py`. Same public interface --
`build_messages(idea_a, idea_b)` and `distance_from_raw(raw)` -- so the v6 reward
reaches it through the same two attribute lookups and needs no edit; the SI reward
shim just rebinds v6's `distance` module global to this one.

Two deliberate differences from the v0 rubric:

1. FEATURES are the three judged SI slots: problem_gap_sim / core_claim_sim /
   method_sim. The v0 rubric's `motivation_sim` scored OUTWARD-FACING STAKES (who
   is affected, what becomes possible), which has no SI counterpart -- SI's third
   field is the METHOD, i.e. how the claim would be tested. So that anchor block
   is retargeted to paradigm-level methodological similarity rather than renamed.

2. WEIGHTS are EQUAL across the three slots (1/3 each = a plain mean), where v0
   weights insight 0.6 / problem 0.3 / motivation 0.1. This makes distance easier
   to earn than under v0 -- two cards sharing a claim but differing in gap and
   method score ~0.67 instead of ~0.4 -- so the diversity term is more permissive
   about claim-level repetition, and is NOT comparable across v0 and v1 runs.
   If `mindist_mean` runs high while cards read as claim-duplicates, re-weight
   toward core_claim here (this table is the only place to change).

The `[label]` trap this rubric guards against: both problem_gap and method open
with a bracketed taxonomy label drawn from a closed 7-value list, so with K=3
ideas a shared label is common by chance. Without an explicit instruction, a
judge reads two `[Empirical Mapping]` methods as near-duplicates and the
diversity term collapses. Both label-bearing anchor blocks say so outright.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Reuse novelty's JSON extractor so v0 and v1 rubrics parse replies identically.
# NOT importing NOVELTY_FIELD_WEIGHTS: v1 weights are equal (see module docstring).
from common.novelty import extract_json

RUBRIC_VERSION = "distance-pairwise-si-v1"

# Similarity feature keys, in scoring order. MUST match the JSON keys named in
# PAIR_TEMPLATE's output block: parse_features requires ALL of them and returns
# None on any miss, which the reward converts to DISTANCE_DEFAULT=1.0 -- i.e. a
# name mismatch here silently maxes out diversity for every pair, visible only in
# the distance_fail metric. The offline test asserts the two agree.
FEATURE_KEYS = ["problem_gap_sim", "core_claim_sim", "method_sim"]

# Feature key -> bare slot name used by the weight table.
_FEATURE_TO_FIELD = {
    "problem_gap_sim": "problem_gap",
    "core_claim_sim": "core_claim",
    "method_sim": "method",
}

# Equal weights: the three slot distances are averaged. weighted_field_distance
# normalizes by the weight sum, so 1/3 each and 1.0 each are equivalent; thirds
# are written out to make the intent unambiguous at a glance.
SI_FIELD_WEIGHTS = {
    "problem_gap": 1.0 / 3.0,
    "core_claim": 1.0 / 3.0,
    "method": 1.0 / 3.0,
}


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
# Verbatim from common/distance.py -- it names no fields, so it is schema-neutral
# and shared intent between v0 and v1 by construction.
SYSTEM_MESSAGE = (
    "You are an expert research scientist judging how SIMILAR two research ideas "
    "are to each other. You are not judging quality, novelty, or impact -- only "
    "how much the two ideas overlap in substance. Two ideas can be phrased very "
    "differently yet propose the same thing, or share surface wording yet differ "
    "in their core contribution. Return only a single JSON object."
)

PAIR_TEMPLATE = """Below are two research ideas, IDEA A and IDEA B. Judge how SIMILAR they are
to each other, field by field.

Each idea is a card with three fields: a problem gap, a core claim, and a method.
The problem gap and the method each OPEN WITH A BRACKETED [label] drawn from a
short fixed taxonomy. Those labels are coarse categories -- there are only seven
of each -- so two ideas sharing a label is common and means very little on its
own. Judge the PROSE that follows the label, not the label itself.

Score three features, each on a 1-5 scale. USE THE FULL SCALE -- every point
1-5 is a distinct judgment; do not default to 1/3/5.

  problem_gap_sim -- do the two ideas target the SAME gap: the same open
    question, tension, or unresolved difficulty?
      1 = unrelated gaps; different questions entirely
      2 = same broad area or task family, but a clearly different gap
      3 = related / adjacent gaps; the framing or what is missing differs in a
          way that matters
      4 = nearly the same gap, differing only in scope, domain, or a secondary
          condition
      5 = exactly the same gap
      NOTE: an identical bracketed label with substantively different prose is
      at most a 2 -- the label alone is not evidence of similarity.

  core_claim_sim -- do the two ideas make the SAME central claim: the testable
    assertion the idea lives or dies on?
      1 = entirely different claims; no mechanistic overlap
      2 = only a loosely related claim or a generic shared building block
          (e.g. both assume attention matters), not the specific assertion
      3 = a partial overlap: one is a weaker/narrower version of the other, or
          they share a premise but not the central assertion
      4 = mostly the same claim; one adds a condition, generalization, or
          framing the other lacks
      5 = the same claim; confirming one would largely confirm the other

  method_sim -- do the two ideas propose the SAME approach for testing the
    claim: the same paradigm AND the same concrete instrument (what is built,
    measured, derived, or optimized, and on what)?
      1 = unrelated approaches; different paradigm and different instrument
      2 = same broad paradigm but a different instrument, or a shared generic
          tool (e.g. both fine-tune a model) doing different work
      3 = related approaches; overlapping instrument but a different design,
          target, or evidence it would produce
      4 = largely the same approach, differing in scope, dataset, or one step
      5 = the same approach; the two would be executed nearly identically
      NOTE: an identical bracketed label with substantively different prose is
      at most a 2 -- the label alone is not evidence of similarity.

=== IDEA A ===
{idea_a}

=== IDEA B ===
{idea_b}

Return ONLY a JSON object of the form:
{{
  "problem_gap_sim": 1 | 2 | 3 | 4 | 5,
  "core_claim_sim": 1 | 2 | 3 | 4 | 5,
  "method_sim": 1 | 2 | 3 | 4 | 5,
  "rationale": "1-3 sentences citing what specifically does or does not overlap."
}}
Do not output anything outside the JSON object."""


def build_messages(idea_a: str, idea_b: str) -> list[dict]:
    """Chat messages asking the judge how similar two SI cards are, slot by slot.

    ``idea_a`` / ``idea_b`` are rendered SI cards; use
    ``common.ideas_si.SIIdea.card_text()`` so the judge sees the same 3-field
    shape the impact judge scores (inspiration excluded)."""
    prompt = PAIR_TEMPLATE.format(idea_a=idea_a, idea_b=idea_b)
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}]


# --------------------------------------------------------------------------- #
# Parsing + scoring  (same contract as common/distance.py)
# --------------------------------------------------------------------------- #
def clamp_score(v: Any) -> Optional[int]:
    try:
        s = int(v)
    except Exception:
        return None
    return s if 1 <= s <= 5 else None


def parse_features(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate a {problem_gap_sim, core_claim_sim, method_sim, rationale} block.
    Returns None if any of the three similarity scores is missing / out of range."""
    if not isinstance(obj, dict):
        return None
    feats: Dict[str, Any] = {}
    for k in FEATURE_KEYS:
        s = clamp_score(obj.get(k))
        if s is None:
            return None
        feats[k] = s
    feats["rationale"] = str(obj.get("rationale") or "").strip()
    return feats


def feature_to_distance(sim: int) -> float:
    """Invert a 1-5 similarity into a [0,1] distance: 5 (identical) -> 0;
    1 (unrelated) -> 1. Identical map to v0's distance/novelty."""
    return (5 - sim) / 4.0


def weighted_field_distance(feats: Dict[str, Any],
                            weights: Optional[Dict[str, float]] = None) -> float:
    """Combine the three per-slot distances into one d(u_i, u_j) in [0,1].

    Default weights are EQUAL (SI_FIELD_WEIGHTS), so this is the plain mean of
    the three inverted similarities."""
    weights = weights or SI_FIELD_WEIGHTS
    total = 0.0
    wsum = 0.0
    for fkey, field in _FEATURE_TO_FIELD.items():
        if fkey in feats and isinstance(feats[fkey], (int, float)):
            w = weights.get(field, 0.0)
            total += w * feature_to_distance(int(feats[fkey]))
            wsum += w
    return total / wsum if wsum else 0.0


def distance_from_raw(raw: str) -> Optional[float]:
    """Parse a judge reply into d(u_i,u_j) in [0,1], or None if it didn't parse.
    The caller owns the LLM call and decides what a None means (the v6 reward
    treats it as DISTANCE_DEFAULT and counts a distance_fail)."""
    obj = extract_json(raw)
    feats = parse_features(obj) if isinstance(obj, dict) else None
    if feats is None:
        return None
    return weighted_field_distance(feats)
