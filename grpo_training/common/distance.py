"""Distance rubric: how DIFFERENT are two generated ideas from each other?
(pairwise, idea-vs-idea -- NOT idea-vs-source like novelty, NOT idea-vs-gold
like impact).

This is the F9-B diversity signal d(u_i, u_j): within one rollout the policy
emits K ideas, and the reward rewards each idea by its distance to the NEAREST
EARLIER idea (min_{j<i} d). A set of 8 near-duplicate ideas should collapse to
~0 diversity credit; a set that genuinely spans the space keeps full credit.

Kept TRANSPORT-AGNOSTIC like its siblings (common/novelty.py, common/scijudge.py):
this module holds only the rubric text, the prompt builder, the parser, and the
scoring math. The OpenAI client lives in the caller (rewards/f9b.py, or an async
eval) so a sync reward and an async eval share ONE definition and cannot drift.

Per (idea_i, idea_j) we score three 1-5 SIMILARITY features -- the same three
fields novelty scores, with the same F9-B weights (insight carries the signal):
  - problem_sim     -- same problem / research question?
  - insight_sim     -- same core insight / contribution?
  - motivation_sim  -- same motivation (outward-facing stakes), not the gap?

Reduction (mirrors novelty's invert-and-weight, so distance and novelty speak the
same language): per-field similarity 1-5 -> per-field distance in [0,1] via
(5 - sim)/4 (5 == identical -> distance 0; 1 == unrelated -> distance 1), then a
weighted sum with the F9-B weights (motivation 0.1, problem 0.3, insight 0.6):

  d(u_i, u_j) = 0.1*d_mot + 0.3*d_prob + 0.6*d_ins                   in [0, 1]

So two ideas with the SAME insight phrased with different motivations are still
close (low distance) -- which is the whole point of decomposing per field rather
than embedding the whole JSON blob.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Reuse novelty's JSON extractor / per-field weights / invert math so the two
# rubrics are provably the same shape (one source of truth for the field model).
from common.novelty import (
    NOVELTY_FIELD_WEIGHTS as FIELD_WEIGHTS,
    extract_json,
)

RUBRIC_VERSION = "distance-pairwise-v2"

# Similarity feature keys, in scoring order. Parallel to novelty's "_match" keys
# but named "_sim" because here we score similarity of two IDEAS (not overlap of
# an idea with a source document).
FEATURE_KEYS = ["problem_sim", "insight_sim", "motivation_sim"]

# Map a feature key -> its bare field name for the weighted reduction (the
# weights themselves are shared with novelty: motivation 0.1, problem 0.3,
# insight 0.6).
_FEATURE_TO_FIELD = {
    "problem_sim": "problem",
    "insight_sim": "insight",
    "motivation_sim": "motivation",
}


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_MESSAGE = (
    "You are an expert research scientist judging how SIMILAR two research ideas "
    "are to each other. You are not judging quality, novelty, or impact -- only "
    "how much the two ideas overlap in substance. Two ideas can be phrased very "
    "differently yet propose the same thing, or share surface wording yet differ "
    "in their core contribution. Return only a single JSON object."
)

PAIR_TEMPLATE = """Below are two research ideas, IDEA A and IDEA B. Judge how SIMILAR they are
to each other, field by field.

Score three features, each on a 1-5 scale. USE THE FULL SCALE -- every point
1-5 is a distinct judgment; do not default to 1/3/5.

  problem_sim -- do the two ideas tackle the SAME problem / research question?
      1 = unrelated problems; different questions entirely
      2 = same broad area or task family, but a clearly different problem
      3 = related / adjacent problems; framing or objective differs in a way
          that matters
      4 = nearly the same problem, differing only in scope, domain, or a
          secondary condition
      5 = exactly the same problem statement

  insight_sim -- do the two ideas share the SAME core insight, method, or key
    contribution?
      1 = entirely different insights; no mechanistic overlap
      2 = only a loosely related idea or a generic shared building block
          (e.g. both use attention), not the specific insight
      3 = a partial overlap: one is a weaker/narrower version of the other, or
          they share a component but not the central claim
      4 = mostly the same insight; one adds a step, generalization, or framing
          the other lacks
      5 = the same core insight; the two ideas would be largely interchangeable
          as contributions

  motivation_sim -- do the two ideas share the SAME motivation -- the
    OUTWARD-FACING STAKES (who is affected, what becomes possible downstream,
    what limitation would be lifted), NOT merely the same gap in prior work?
      1 = unrelated stakes; different beneficiary and consequence
      2 = loosely related: a high-level concern in common but a different
          beneficiary or downstream consequence
      3 = related but distinct motivation / angle on who benefits
      4 = largely the same stakes, framed or scoped somewhat differently
      5 = exactly the same motivation; same beneficiary, same downstream consequence

=== IDEA A ===
{idea_a}

=== IDEA B ===
{idea_b}

Return ONLY a JSON object of the form:
{{
  "problem_sim": 1 | 2 | 3 | 4 | 5,
  "insight_sim": 1 | 2 | 3 | 4 | 5,
  "motivation_sim": 1 | 2 | 3 | 4 | 5,
  "rationale": "1-3 sentences citing what specifically does or does not overlap."
}}
Do not output anything outside the JSON object."""


def build_messages(idea_a: str, idea_b: str) -> list[dict]:
    """Chat messages asking the judge how similar two ideas are, field by field.

    ``idea_a`` / ``idea_b`` are the rendered idea cards (problem + insight +
    motivation); use ``common.ideas.Idea.card_text()`` to produce them so the
    judge sees the same shape novelty/impact use."""
    prompt = PAIR_TEMPLATE.format(idea_a=idea_a, idea_b=idea_b)
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}]


# --------------------------------------------------------------------------- #
# Parsing + scoring
# --------------------------------------------------------------------------- #
def clamp_score(v: Any) -> Optional[int]:
    try:
        s = int(v)
    except Exception:
        return None
    return s if 1 <= s <= 5 else None


def parse_features(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate a {problem_sim, insight_sim, motivation_sim, rationale} block.
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
    """Invert a 1-5 similarity into a [0,1] distance: 5 (identical) -> 0 distance;
    1 (unrelated) -> 1 distance.  d_field = (5 - sim) / 4 (mirrors novelty's
    feature_to_novelty, so distance and novelty share the same 1-5 -> [0,1] map)."""
    return (5 - sim) / 4.0


def weighted_field_distance(feats: Dict[str, Any],
                            weights: Optional[Dict[str, float]] = None) -> float:
    """Combine the three per-field distances into one d(u_i, u_j) in [0,1] using
    the F9-B weights (motivation 0.1, problem 0.3, insight 0.6 -- shared with
    novelty). ``feats`` uses the "_sim" feature keys (as returned by
    parse_features)."""
    weights = weights or FIELD_WEIGHTS
    total = 0.0
    wsum = 0.0
    for fkey, field in _FEATURE_TO_FIELD.items():
        if fkey in feats and isinstance(feats[fkey], (int, float)):
            w = weights.get(field, 0.0)
            total += w * feature_to_distance(int(feats[fkey]))
            wsum += w
    return total / wsum if wsum else 0.0


def distance_from_raw(raw: str) -> Optional[float]:
    """End-to-end convenience: parse a judge reply into d(u_i,u_j) in [0,1], or
    None if it didn't parse. The caller (reward / eval) owns the LLM call and
    decides what to do with a None (e.g. retry, or treat as max distance)."""
    obj = extract_json(raw)
    feats = parse_features(obj) if isinstance(obj, dict) else None
    if feats is None:
        return None
    return weighted_field_distance(feats)
