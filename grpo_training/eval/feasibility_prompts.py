"""Feasibility scale + prompts -- the SINGLE source of truth shared by the
eval-only judge (grpo_training/eval/feasibility_judge.py) and the ground-truth
extractor (extract_ground_truth.py).

Feasibility of a research idea is scored on TWO independent axes, each an
anchored 1-5 ordinal tier:

  * compute      -- how much accelerator compute the work needs, in A100-GPU-hours
  * human_eval   -- how much human-evaluation effort it needs, in person-hours

Why anchored ordinals (and not raw GPU-hours / dollars): the judge only ever
sees an IDEA-ONLY summary (no results, numbers, or hardware), so predicting an
exact figure is false precision. The 1-5 tiers give robust agreement metrics
(exact / +-1 / quadratic-weighted kappa) while the log-scaled anchors preserve
magnitude. GPU-hours (not dollars) is the base unit because it is what ML papers
actually report and it is hardware-normalizable; dollars are time-varying and
rarely stated. Each prediction also carries a free-text point `*_estimate` and a
`rationale`, which are informative but NOT scored against the ground truth.

The extractor reads the FULL paper (methods + experiments + tables) and maps the
REPORTED compute/eval onto the same two tiers, so judge and ground truth live on
one identical scale. When a paper does not report an axis, the extractor sets the
corresponding `*_not_reported` flag and that axis is excluded from agreement
metrics for that paper.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Bump when the tier bands or extraction/judge prompts change materially, so
# stale caches (gt/*.json, summaries/*.json) can be told apart from fresh ones.
SCALE_VERSION = "feasibility-anchored-1to5-v1"

# --------------------------------------------------------------------------- #
# The two 1-5 anchor tables. Each entry: tier -> (short label, band, guidance).
# Bands are log-scaled by ~1 order of magnitude per tier.
# --------------------------------------------------------------------------- #
COMPUTE_TIERS: Dict[int, Dict[str, str]] = {
    1: {"label": "trivial",
        "band": "< 10 A100-GPU-hours",
        "guidance": "single GPU for hours; API-only calls, prompting studies, "
                    "small fine-tunes, or purely analytical/no-training work."},
    2: {"label": "modest",
        "band": "10-100 A100-GPU-hours",
        "guidance": "one GPU node for about a day; a moderate fine-tune, a small "
                    "model trained from scratch, or many API/inference runs."},
    3: {"label": "substantial",
        "band": "100-1,000 A100-GPU-hours",
        "guidance": "multiple GPUs for several days; sizeable fine-tuning, "
                    "medium pretraining, or a large sweep of training runs."},
    4: {"label": "heavy",
        "band": "1,000-10,000 A100-GPU-hours",
        "guidance": "multi-node cluster for about a week; large-scale training, "
                    "RLHF pipelines, or extensive multi-model experiments."},
    5: {"label": "frontier",
        "band": "> 10,000 A100-GPU-hours",
        "guidance": "large cluster for weeks or more; frontier-scale pretraining "
                    "or industrial-scale training infrastructure."},
}

HUMAN_EVAL_TIERS: Dict[int, Dict[str, str]] = {
    1: {"label": "none",
        "band": "0 person-hours",
        "guidance": "fully automatic evaluation; no human annotation or study "
                    "at all (metrics, automatic benchmarks, LLM-as-judge only)."},
    2: {"label": "light",
        "band": "< 10 person-hours",
        "guidance": "a couple of annotators on a small sample; a quick rating "
                    "task, spot-check, or pilot with a handful of participants."},
    3: {"label": "moderate",
        "band": "10-100 person-hours",
        "guidance": "a moderate annotation effort or a small controlled user "
                    "study; tens of participants or a few trained annotators."},
    4: {"label": "large",
        "band": "100-1,000 person-hours",
        "guidance": "large-scale annotation or a multi-session user study; many "
                    "participants, several trained annotators, or expert review."},
    5: {"label": "massive",
        "band": "> 1,000 person-hours",
        "guidance": "very large or longitudinal human effort; big crowd studies, "
                    "large expert panels, or sustained multi-week evaluation."},
}


def _tier_block(tiers: Dict[int, Dict[str, str]]) -> str:
    """Render an anchor table as a bulleted 1-5 list for a prompt."""
    return "\n".join(
        f"  {t}. {tiers[t]['label']} ({tiers[t]['band']}): {tiers[t]['guidance']}"
        for t in sorted(tiers)
    )


COMPUTE_BLOCK = _tier_block(COMPUTE_TIERS)
HUMAN_EVAL_BLOCK = _tier_block(HUMAN_EVAL_TIERS)


def valid_tier(v: Any) -> bool:
    """True iff v is (coercible to) an integer tier in 1..5."""
    try:
        return int(v) in (1, 2, 3, 4, 5)
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Ground-truth EXTRACTION prompt (reads the FULL paper).
# --------------------------------------------------------------------------- #
EXTRACT_SYSTEM = (
    "You are an expert at reading machine-learning and computer-science papers "
    "and extracting the RESOURCES the authors actually used. You are given the "
    "full text (methods, experiments, and tables) of ONE paper. Report the "
    "compute the authors used to run their experiments and the human-evaluation "
    "effort they expended, then map each onto a fixed 1-5 tier. Base every "
    "judgement ONLY on what the paper states or clearly implies; never guess "
    "beyond the evidence. If the paper does not report an axis at all, say so "
    "explicitly with the not_reported flag rather than inventing a number. "
    "Return only a single JSON object."
)

EXTRACT_USER = """Read the paper below and extract its resource footprint on two axes.

═══════════════════════════════════════
COMPUTE TIERS (accelerator compute, in A100-GPU-hours)
═══════════════════════════════════════
{compute_block}

═══════════════════════════════════════
HUMAN-EVALUATION TIERS (human effort, in person-hours)
═══════════════════════════════════════
{human_eval_block}

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════
- COMPUTE: find any reported GPU/TPU type, count, and wall-clock or GPU-hours;
  model sizes trained; number/size of training runs. Convert to approximate
  A100-GPU-hours (state your conversion in `compute_evidence`) and pick the tier.
  If the work does no training and only calls APIs / does analysis, that is
  tier 1. If NO compute is reported anywhere, set compute_not_reported=true and
  still give your best tier from what the method implies.
- HUMAN_EVAL: find any human annotation, rating, or user study -- number of
  participants/annotators, items rated, sessions, hours. Convert to approximate
  person-hours (state it in `human_eval_evidence`) and pick the tier. Fully
  automatic evaluation is tier 1. If NO human evaluation is reported, set
  human_eval_not_reported=true and use tier 1.
- Quote or closely paraphrase the exact sentence(s) you relied on in each
  `*_evidence` field. Keep evidence short (1-3 snippets).

═══════════════════════════════════════
PAPER
═══════════════════════════════════════

Title:
{title}

Full text (may be truncated):
{full_text}

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

Return ONLY a JSON object of exactly this shape (no prose, no code fences):

{{
  "compute_tier": <integer 1-5>,
  "compute_estimate": "<short A100-GPU-hour estimate, e.g. '~256 GPU-hours'>",
  "compute_evidence": "<the sentence(s) + your conversion you relied on>",
  "compute_not_reported": <true|false>,
  "human_eval_tier": <integer 1-5>,
  "human_eval_estimate": "<short person-hour estimate, e.g. '~40 person-hours'>",
  "human_eval_evidence": "<the sentence(s) + your conversion you relied on>",
  "human_eval_not_reported": <true|false>
}}
"""


def build_extract_messages(title: str, full_text: str) -> List[Dict[str, str]]:
    """Assemble the ground-truth extraction chat messages for one full paper.

    Uses .replace (not str.format) on the paper fields so any literal braces in
    the paper text cannot break formatting; the tier blocks and JSON skeleton
    are injected first via the {..._block} placeholders."""
    user = (
        EXTRACT_USER
        .replace("{compute_block}", COMPUTE_BLOCK)
        .replace("{human_eval_block}", HUMAN_EVAL_BLOCK)
        .replace("{title}", title or "(untitled)")
        .replace("{full_text}", full_text or "")
    )
    return [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# JUDGE prompt (reads an IDEA-ONLY summary; predicts the same two tiers).
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "You are a research-feasibility assessor. You are given an IDEA-ONLY "
    "description of a research proposal -- its problem, claim, and method, with "
    "NO results, numbers, or hardware details. Estimate how much this idea would "
    "cost to EXECUTE on two axes: accelerator compute (A100-GPU-hours) and "
    "human-evaluation effort (person-hours). Reason from the method: what would "
    "have to be trained or run, and what human effort (if any) the idea's own "
    "claim genuinely requires. Judge the MINIMUM resources needed to execute the "
    "idea as described -- NOT an ideal, exhaustive, or 'to be safe' study. Many "
    "ideas need no human evaluation at all (they are checked automatically). "
    "Map each axis onto a fixed 1-5 tier. Return only a single JSON object."
)

JUDGE_USER = """Estimate the feasibility (resource cost to execute) of the research idea below.

═══════════════════════════════════════
COMPUTE TIERS (accelerator compute, in A100-GPU-hours)
═══════════════════════════════════════
{compute_block}

═══════════════════════════════════════
HUMAN-EVALUATION TIERS (human effort, in person-hours)
═══════════════════════════════════════
{human_eval_block}

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════
- Judge the MINIMUM resources needed to execute the idea AS DESCRIBED, not the
  most thorough study you could imagine. Do NOT add a human study "to be safe."
- COMPUTE: would it need training a model (how big?), a large inference sweep,
  or only API calls / analysis? Pick the tier whose A100-GPU-hour band fits.
- HUMAN_EVAL: count only human effort the idea genuinely requires, from two
  sources: (a) human work described IN the method itself -- building a
  human-annotated dataset, collecting human data, or hiring raters/experts --
  which you should count at face value; and (b) evaluating the central claim
  when no automatic metric could stand in (subjective quality, user behaviour,
  human preference). Do NOT invent an evaluation study for a claim that is
  automatically measurable: if the idea can be checked with metrics, benchmarks,
  LLM-as-judge, or simulated users, its human_eval is tier 1 (0 person-hours).
  In short: score the human effort the method NAMES, but never add a "to be
  safe" study on top. When the method names no human effort and the claim is
  automatically measurable, choose tier 1.
- Give a short point `*_estimate` and a one-sentence `rationale`.

═══════════════════════════════════════
RESEARCH IDEA
═══════════════════════════════════════

{idea_text}

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

Return ONLY a JSON object of exactly this shape (no prose, no code fences):

{{
  "compute_tier": <integer 1-5>,
  "compute_estimate": "<short A100-GPU-hour estimate>",
  "human_eval_tier": <integer 1-5>,
  "human_eval_estimate": "<short person-hour estimate>",
  "rationale": "<one sentence justifying both tiers>"
}}
"""


def build_judge_messages(idea_text: str) -> List[Dict[str, str]]:
    """Assemble the feasibility-judge chat messages for one idea-only summary."""
    user = (
        JUDGE_USER
        .replace("{compute_block}", COMPUTE_BLOCK)
        .replace("{human_eval_block}", HUMAN_EVAL_BLOCK)
        .replace("{idea_text}", idea_text or "")
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]
