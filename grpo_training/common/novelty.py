"""Novelty rubric: how much of a target idea is already present in a population
of source documents? (population-based, NOT pairwise).

Ported from the reference `evals/novelty/novelty_judge.py` (Mac), but kept
TRANSPORT-AGNOSTIC: this module holds only the rubric text, the prompt builders,
the parsers, and the scoring math. The OpenAI client lives in the caller
(eval/novelty_judge.py, evals/, or rewards/f9b.py later) so a sync reward and an
async eval can share one definition.

Per (document, idea) we score three 1-5 overlap features:
  - problem_match     -- same problem / question as the idea?
  - insight_match     -- already contains the idea's core insight / contribution?
  - motivation_match  -- same motivation (outward-facing stakes), not the gap?

Instance-level reduction (the "try batching max" design): novelty is the inverse
of the MAX overlap across the population -- if ANY single doc already covers the
idea, it is not novel.

  overlap(idea)        = max_d  mean(problem, insight, motivation)   in [1, 5]
  novelty_score(idea)  = (5 - overlap(idea)) / 4                      in [0, 1]

(The 1-5 overlap is rescaled to [0,1] via (5 - x)/4, NOT 1 - x/5 -- the latter
caps a fully-novel idea at 0.8. Matches feature_to_novelty / grounding / distance.)

F9-B reconciliation (the reward consumes the SAME rubric, different reduction):
the per-field novelty the reward needs is N_field = (5 - feature_field)/4, combined
as a weighted sum with weights (motivation 0.1, problem 0.3, insight 0.6). See
`feature_to_novelty` / `weighted_field_novelty` at the bottom. Eval reports
max-overlap novelty; the reward reads per-field N -- one rubric, two reductions.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

RUBRIC_VERSION = "novelty-population-v2"

# Feature keys, in the order the rubric scores them. The trailing "_match"
# matches the reference output schema.
FEATURE_KEYS = ["problem_match", "insight_match", "motivation_match"]

# F9-B per-field novelty weights (insight carries the most signal). Keyed by the
# bare field name (problem/insight/motivation), NOT the "_match" feature key.
NOVELTY_FIELD_WEIGHTS = {"motivation": 0.1, "problem": 0.3, "insight": 0.6}

# Map a feature key -> its bare field name for the weighted reduction.
_FEATURE_TO_FIELD = {
    "problem_match": "problem",
    "insight_match": "insight",
    "motivation_match": "motivation",
}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM_MESSAGE = (
    "You are an expert research scientist assessing the NOVELTY of a research "
    "idea relative to prior work. You will be shown a target idea and one or "
    "more source documents that predate it. Your job is to judge how much of "
    "the idea is ALREADY PRESENT in the source document(s). You are not "
    "judging quality, writing, or impact -- only overlap of substance. "
    "Return only a single JSON object."
)

# --- per-doc mode: one (document, idea) pair per call ---------------------- #
PER_DOC_TEMPLATE = """Below is a SOURCE DOCUMENT and a TARGET IDEA. The source predates the idea.
Judge how much of the target idea is already present in the source document.

Score three features, each on a 1-5 scale. USE THE FULL SCALE -- every point
1-5 is a distinct judgment; do not default to 1/3/5.

  problem_match -- does the source tackle the SAME problem / question as the
    target idea?
      1 = unrelated problem; the source addresses a different question entirely
          and would not be cited as prior work on this problem
      2 = same broad research area or task family, but a clearly different
          problem (e.g. both in machine translation, but one is decoding speed
          and the other is low-resource data)
      3 = related / adjacent problem: overlapping but the framing, setting, or
          objective differs in a way that matters
      4 = nearly the same problem, differing only in scope, domain, or a
          secondary condition (e.g. same task on a different modality/language)
      5 = exactly the same problem statement; they target the identical
          question under the same conditions

  insight_match -- does the source already contain the CORE INSIGHT, method,
    or key contribution of the target idea?
      1 = none of the idea's core insight is present; no mechanistic overlap
      2 = only a loosely related idea or a generic shared building block is
          present (e.g. both use attention), not the specific insight
      3 = a partial or weaker version of the insight is present, or it appears
          as a minor component rather than the central claim
      4 = most of the core insight is present; the source nearly proposes it
          but is missing a key step, generalization, or framing the idea adds
      5 = the source essentially already proposes this idea's core insight;
          the idea's contribution would be largely anticipated by it

  motivation_match -- does the source share the same MOTIVATION as the target
    idea -- the OUTWARD-FACING STAKES: who is affected, what becomes possible
    downstream, what limitation would be lifted or cost reduced? (This is the
    stakes of the work, NOT merely the same gap in prior work.)
      1 = unrelated stakes; a different beneficiary and a different downstream
          consequence
      2 = loosely related stakes: a high-level concern in common, but a
          different beneficiary or downstream consequence
      3 = a related but distinct motivation; overlapping stakes but a different
          angle on who benefits or what it unlocks
      4 = largely the same stakes -- same beneficiary and consequence -- framed
          or scoped somewhat differently
      5 = exactly the same motivation; the same beneficiary stands to gain the
          same downstream consequence

=== SOURCE DOCUMENT ===
Title: {ref_title}
{ref_body}

=== TARGET IDEA ===
Title: {idea_title}
{idea_body}

Return ONLY a JSON object of the form:
{{
  "problem_match": 1 | 2 | 3 | 4 | 5,
  "insight_match": 1 | 2 | 3 | 4 | 5,
  "motivation_match": 1 | 2 | 3 | 4 | 5,
  "rationale": "1-3 sentences citing what specifically does or does not overlap."
}}
Do not output anything outside the JSON object."""

# --- batched mode: idea vs whole population in one call --------------------- #
BATCHED_TEMPLATE = """Below is a TARGET IDEA followed by a POPULATION of SOURCE DOCUMENTS that all
predate it. For EACH source document, judge how much of the target idea is
already present in it.

Score three features per document, each on a 1-5 scale. USE THE FULL SCALE --
every point 1-5 is a distinct judgment; do not default to 1/3/5.

  problem_match -- does the source tackle the SAME problem / question as the
    target idea?
      1 = unrelated problem
      2 = same broad area/task family, clearly different problem
      3 = related / adjacent problem, framing or objective differs
      4 = nearly the same problem, differs only in scope/domain/condition
      5 = exactly the same problem statement

  insight_match -- does the source already contain the CORE INSIGHT, method,
    or key contribution of the target idea?
      1 = none of the core insight is present
      2 = only a loosely related idea or generic shared building block
      3 = a partial/weaker version, or a minor component not the central claim
      4 = most of the insight; missing a key step, generalization, or framing
      5 = source essentially already proposes this idea's core insight

  motivation_match -- does the source share the same MOTIVATION as the target
    idea -- the OUTWARD-FACING STAKES (who is affected, what becomes possible
    downstream), NOT merely the same gap in prior work?
      1 = unrelated stakes; different beneficiary and consequence
      2 = loosely related: shared high-level concern, different beneficiary/consequence
      3 = related but distinct motivation / angle on who benefits
      4 = largely the same stakes, framed or scoped differently
      5 = exactly the same motivation; same beneficiary, same downstream consequence

=== TARGET IDEA ===
Title: {idea_title}
{idea_body}

=== POPULATION OF SOURCE DOCUMENTS ===
{population_block}

Return ONLY a JSON object mapping each document's index to its scores:
{{
  "documents": [
    {{"index": 0, "problem_match": 1-5, "insight_match": 1-5, "motivation_match": 1-5, "rationale": "..."}},
    {{"index": 1, "problem_match": 1-5, "insight_match": 1-5, "motivation_match": 1-5, "rationale": "..."}}
    // one entry per source document, in order
  ]
}}
Do not output anything outside the JSON object."""


# --------------------------------------------------------------------------- #
# Document / idea rendering
# --------------------------------------------------------------------------- #
def card_block(summary: Dict[str, Any]) -> str:
    """Render a {problem, insight, motivation} card."""
    return (
        f"Problem: {str(summary.get('problem', '') or '').strip()}\n"
        f"Insight: {str(summary.get('insight', '') or '').strip()}\n"
        f"Motivation: {str(summary.get('motivation', '') or '').strip()}"
    )


def doc_block(doc: Dict[str, Any]) -> str:
    """Preferred text for a document (idea or reference): the structured summary
    card when present (from summarize.py / common.ideas.idea_to_eval_card), else
    the raw abstract / tldr."""
    summary = doc.get("summary")
    if isinstance(summary, dict) and summary.get("problem"):
        return card_block(summary)
    abstract = (doc.get("abstract") or "").strip()
    if abstract:
        return f"Abstract:\n{abstract}"
    return f"Summary:\n{(doc.get('tldr') or '').strip()}"


def population_block(refs: List[Dict[str, Any]]) -> str:
    chunks = []
    for i, ref in enumerate(refs):
        chunks.append(
            f"--- DOCUMENT {i} ---\n"
            f"Title: {ref.get('title') or '(untitled)'}\n"
            f"{doc_block(ref)}"
        )
    return "\n\n".join(chunks)


def build_per_doc_messages(idea: Dict[str, Any], ref: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = PER_DOC_TEMPLATE.format(
        ref_title=ref.get("title") or "(untitled)",
        ref_body=doc_block(ref),
        idea_title=idea.get("title") or "(untitled)",
        idea_body=doc_block(idea),
    )
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}]


def build_batched_messages(idea: Dict[str, Any], refs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    prompt = BATCHED_TEMPLATE.format(
        idea_title=idea.get("title") or "(untitled)",
        idea_body=doc_block(idea),
        population_block=population_block(refs),
    )
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def extract_json(raw: str) -> Optional[Any]:
    """Pull the intended JSON object out of a (possibly chatty) model reply.

    Scans for every balanced top-level {...} span and returns the LAST one that
    parses -- reasoning/opus models sometimes emit a first object, then prose
    ("wait, let me fix that"), then a corrected object. A greedy {.*} match would
    span both blocks plus prose and fail; the last-balanced-span approach returns
    the model's final answer."""
    if not raw:
        return None
    candidates: List[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(raw[start:i + 1])
    parsed: Optional[Any] = None
    for cand in candidates:
        try:
            parsed = json.loads(cand)  # keep the last one that parses
        except Exception:
            continue
    return parsed


def clamp_score(v: Any) -> Optional[int]:
    try:
        s = int(v)
    except Exception:
        return None
    return s if 1 <= s <= 5 else None


def parse_features(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate a {problem_match, insight_match, motivation_match, rationale} block.
    Returns None if any of the three feature scores is missing / out of range."""
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


def overlap(feats: Dict[str, Any]) -> float:
    """Per-document overlap = mean of the feature scores, in [1, 5]."""
    return sum(feats[k] for k in FEATURE_KEYS) / len(FEATURE_KEYS)


# --------------------------------------------------------------------------- #
# Instance-level aggregation (eval): novelty = MIN over docs of whole-card
# weighted novelty -- SHARED with the reward via reduce_novelty (see below).
# --------------------------------------------------------------------------- #
def aggregate(per_doc: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce per-document scores into an instance-level novelty score.

    novelty_score(idea) = MIN over documents of the doc's whole-card weighted
                          novelty (reduce_novelty) in [0, 1] -- IDENTICAL logic to
                          the f9b reward, so eval and training agree. 0 = not
                          novel, 1 = fully novel. `per_doc` entries carry a
                          "features" dict (parse_features output) and, for
                          reporting continuity, an "overlap" float (unweighted
                          1-5 mean). max_overlap/mean_overlap are kept purely as
                          descriptive reporting fields.
    """
    scored = [d for d in per_doc if d.get("features") is not None]
    if not scored:
        return {"max_overlap": None, "mean_overlap": None,
                "novelty_score": None, "most_similar_ref": None,
                "n_scored": 0, "n_docs": len(per_doc)}
    # whole-doc weighted novelty per doc; instance novelty = MIN (the most-similar
    # doc, i.e. lowest per-doc novelty, bounds the idea).
    doc_novs = [(d, weighted_field_novelty(d["features"])) for d in scored]
    least_novel, novelty_score = min(doc_novs, key=lambda t: t[1])
    overlaps = [d["overlap"] for d in scored if d.get("overlap") is not None]
    return {
        "max_overlap": max(overlaps) if overlaps else None,
        "mean_overlap": (sum(overlaps) / len(overlaps)) if overlaps else None,
        "novelty_score": novelty_score,
        "most_similar_ref": {"ref_id": least_novel.get("ref_id"),
                             "ref_title": least_novel.get("ref_title"),
                             "overlap": least_novel.get("overlap")},
        "n_scored": len(scored),
        "n_docs": len(per_doc),
    }


# --------------------------------------------------------------------------- #
# F9-B reconciliation (reward reduction: per-field N, weighted)
# --------------------------------------------------------------------------- #
def feature_to_novelty(match: int) -> float:
    """Invert a 1-5 overlap feature into a [0,1] novelty: 5 (fully present) -> 0
    novel; 1 (absent) -> 1 fully novel. N_field = 1 - match/5 ... but with the
    1-5 scale that gives [0, 0.8]; rescale so 1->1 and 5->0:  (5 - match) / 4."""
    return (5 - match) / 4.0


def weighted_field_novelty(feats: Dict[str, Any],
                           weights: Optional[Dict[str, float]] = None) -> float:
    """Combine the three per-field novelties into one N(u) in [0,1] using F9-B
    weights (motivation 0.1, problem 0.3, insight 0.6). `feats` uses the "_match"
    feature keys (as returned by parse_features)."""
    weights = weights or NOVELTY_FIELD_WEIGHTS
    total = 0.0
    wsum = 0.0
    for fkey, field in _FEATURE_TO_FIELD.items():
        if fkey in feats and isinstance(feats[fkey], (int, float)):
            w = weights.get(field, 0.0)
            total += w * feature_to_novelty(int(feats[fkey]))
            wsum += w
    return total / wsum if wsum else 0.0


def reduce_novelty(feats_list):
    """Instance-level novelty, SHARED by the reward (f9b) and the eval pipeline
    so the two never drift. Score each document's WHOLE card via
    weighted_field_novelty (0.1/0.3/0.6), then take the MIN across documents --
    the single MOST-SIMILAR prior document bounds how novel the idea can be.

    Whole-document, NOT per-field cross-doc max: the least-novel value must come
    from ONE document, so an idea is penalised on problem+insight+motivation
    TOGETHER against its closest prior work. The old per-field max took the worst
    field from possibly THREE different docs -- strictly harsher (weighted_sum of
    per-field maxes <= min of per-doc weighted_sums) and near-unreachable-novel in
    practice. `feats_list` = parse_features() dicts; None entries (unparsed) are
    skipped. Returns None if nothing scored (caller imputes)."""
    novs = [weighted_field_novelty(f) for f in feats_list if f is not None]
    return min(novs) if novs else None
