"""F9-B v6 for the SI (v1) card schema, distance judge on AWS Bedrock.

THE REWARD IS NOT REDEFINED HERE. Like f9b_impdist_len_v6_bedrock.py, this module
imports f9b_impdist_len_v6 and re-exports its `compute_score` unchanged. v6's math
-- SCHEMA_BONUS + sum_i quality_i * mindist_i^beta - len_penalty, the impute tiers,
the curriculum, the card-only length penalty -- is single-sourced, so v0 and v1
scores are produced by the same code and cannot drift apart.

WHY THIS IS A SHIM AND NOT A FORK: the v0 and v1 schemas line up 3+1 on the same
slot semantics (gap / claim / third field / provenance), and v6 was already built
so that provenance is REQUIRED BY THE FORMAT GATE but INVISIBLE to the impact judge
and UNTAXED by the length penalty. v1's `inspiration` is the exact analogue of v0's
grounding, so only four things are v0-specific: the parser, the card labels, the
gate's provenance check, and the distance rubric's field names. All four are
module-global lookups v6 resolves at CALL time, so all four can be rebound:

    v6 call site                          rebound to
    parse_ideas   (compute_score:831)     common.ideas_si.parse_ideas_si
    _format_value (compute_score:832)     _format_value_si (below)
    distance      (_distance:544)         common.distance_si
    _idea_client / IDEA_JUDGE_MODEL       Bedrock (same as the v0 shim)

The card labels need no rebinding at all: SIIdea subclasses Idea and overrides
card_text(), so v6's _card_concat / _impact / _distance / sampling seed and its
per-idea log line (which prints `u.insight_claim` = core_claim) all follow along.

WHAT IS DELIBERATELY DIFFERENT FROM v0, both because the data says so:
  * K=3 (v0 used 5): every train_multi row carries num_ideas=3, and v6's gate
    hard-fails len(ideas) != K. Set F9B_NUM_IDEAS=3.
  * F9B_LEN_TARGET=834 (v0: 1670): the K=3 SFT set p90 over the three judged
    fields, same construction as v6's v0 target.
  * Distance weights are EQUAL across the three slots (see common/distance_si.py),
    where v0 weights the claim 0.6. The diversity term is therefore NOT comparable
    across v0 and v1 runs.

USE IT:
    REWARD_FN=.../rewards/f9b_impdist_len_v6_si_bedrock.py
    F9B_NUM_IDEAS=3 F9B_LEN_TARGET=834
Everything else is v6's env surface. Bedrock knobs are inherited unchanged from
f9b_impdist_len_v6_bedrock (F9B_BEDROCK_MODEL / _REGION / _MAX_RETRIES).

`F9B_NUM_IDEAS=3` IS NOT OPTIONAL. K is read from the env at v6's import time and
defaults to 8; if it isn't 3, the gate fails `len(ideas) != K` on every SI rollout
and the run reads as a broken reward from step 0 rather than as a misconfiguration.
launch_grpo_v6_si_onenode_h200.sbatch defaults it correctly.

ONE SHIM PER PROCESS. Rebinding mutates the shared v6 module dict, so importing
BOTH this and f9b_impdist_len_v6_bedrock leaves v6 pointing at whichever imported
last -- the v0 reward would then score v0 rollouts with the SI parser and gate
everything to 0.0. verl loads exactly one REWARD_FN per job, so this is safe in
production; it only matters for tests and notebooks that import both.

THE <think> BLOCK IS REWARD-INVISIBLE BY DESIGN -- and this is why with-think
underperforms forced-empty-think under RL. v6 scores only the committed K cards;
the reasoning trace is required by the format gate but never read by the impact
judge, distance judge, or length penalty. The reward is therefore INDIFFERENT to
whether the trace did any work, so RL is free to make it decorative (and does:
run2605 collapses genuine deliberation 44.8%->6.9% by step 600). This is the
expected behavior for reward shaping on an open-ended task, not a bug -- see
thinking_design/test_v2/related_work.md. Load-bearing summary:
  - Sprague 2024 (2409.12183): CoT helps mainly math/symbolic, thin on open-ended.
  - Pfau 2024 (2404.15758): a CoT that isn't rewarded can be pure filler.
  - Shao 2025, Spurious Rewards (2506.10947): RLVR gains can be reweighting, not
    caused reasoning -- cuts against naive best-of-M.
  - Baker 2025 (2503.11926): rewarding the trace TEXT directly -> obfuscation
    (this is why a deliberation-rubric multiplier on <think> was vetoed).
  - Lightman 2023 (2305.20050) / Guan 2024 (2412.16339): the fix is to make the
    reasoning STRUCTURALLY required for the reward (step/spec satisfaction), not
    to grade whether the trace looks deliberative. Motivates the best-of-M redesign.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

_GRPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_GRPO_ROOT.parent), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import distance_si  # noqa: E402
from common.ideas_si import (  # noqa: E402
    SIIdea,
    has_valid_method_label,
    has_valid_opportunity_label,
    parse_ideas_si,
)
from rewards import f9b_impdist_len_v6 as _v6  # noqa: E402
# Importing the v0 Bedrock shim rebinds v6's distance client to Bedrock as an
# import side effect, and gives us its client factory / model id without
# duplicating the pooling logic (one lru_cached client per worker process, sized
# from v6's own timeout/concurrency constants).
from rewards.f9b_impdist_len_v6_bedrock import (  # noqa: E402
    BEDROCK_JUDGE_MODEL,
    _bedrock_idea_client,
)


# --------------------------------------------------------------------------- #
# SI format gate
# --------------------------------------------------------------------------- #
def _format_value_si(solution_str: str, ideas: List[SIIdea]) -> Optional[float]:
    """SI format gate (BINARY). None if schema-valid (proceed to scoring), else the
    terminal reward 0.0 -- with NO judge calls fired.

    Mirrors f9b_impdist_len_v6._format_value, keeping its two schema-agnostic
    checks (think+answer tags present; exactly K parseable ideas) and replacing
    its three v0 grounding checks -- each an unconditional fail on an SI card --
    with the SI provenance + taxonomy checks.

    Why gate the labels and `inspiration` at all: neither is visible to the impact
    judge and neither is priced by the length penalty, so both cost tokens and earn
    nothing. That asymmetry is exactly what made the policy drop grounding under
    v4. The gate is what makes emitting them non-optional. Labels are validated
    NORMALIZED (see ideas_si.normalize_label) so a wording slip the SFT data itself
    contains -- "[Synthesis]" for "[Synthesis / Unification]" -- is not punished."""
    text = solution_str or ""
    if not (_v6._THINK_RE.search(text) and _v6._ANSWER_RE.search(text)):
        return 0.0
    if len(ideas) != _v6.K:
        return 0.0
    for idea in ideas:
        # All four SI fields non-empty. The parser already enforces this, but the
        # gate re-asserts it so it is self-contained and readable as the contract.
        if not (idea.problem_gap and idea.core_claim
                and idea.method and idea.inspiration):
            return 0.0
        # Both label-bearing fields must OPEN with a label from their closed
        # taxonomy (or the prompt's explicit [Other] escape hatch).
        if not has_valid_opportunity_label(idea.problem_gap):
            return 0.0
        if not has_valid_method_label(idea.method):
            return 0.0
    return None


# --- the actual swap: rebind the globals v6 resolves at call time ------------ #
# compute_score does `parse_ideas(...)` then `_format_value(...)`; _distance does
# `distance.build_messages(...)` / `distance.distance_from_raw(...)`. All are
# module-dict lookups, so rebinding here redirects them without editing v6.
# The impact path (_impact_client / IMPACT_MODEL / common.scijudge) is untouched:
# SciJudge is schema-agnostic (its prompt names no fields) and stays self-hosted.
_v6.parse_ideas = parse_ideas_si
_v6._format_value = _format_value_si
_v6.distance = distance_si
# GROUNDING GATE (v6's F9B_GROUNDING, off unless a launcher enables it): the judge
# is handed the card's DECLARED provenance. v0 renders an `insight.grounding` list;
# SI has free-text `inspiration` in its place, so point the hook at that. Also drop
# v6's grounding_type requirement -- SIIdea leaves grounding_type empty by
# construction, so keeping it on would pin every SI idea to the g floor.
# VALIDATED ON REAL SI ROLLOUTS (job 492, v8 smoke, 8 steps x 128 sets): the rubric's
# source_exists / support_faithful / load_bearing questions all read `inspiration` as
# the citation, which is a looser target than a structured {source, supports} list --
# but grounding_fail stayed 0.000 (one step at 0.008) and g came back GRADED, not
# defaulted. Two calibration notes for whoever prices this next:
#   * ROLLOUTS GROUND LOWER THAN THE SFT PREDICTIONS THIS WAS CALIBRATED ON.
#     load_bearing 1.94-2.17 and g 0.234 (val@step0) / 0.265-0.321 (train) under
#     LO=1 HI=5 FLOOR=0, vs the 2.4 / 0.36 measured on greedy step-112 predictions in
#     thinking/grounding_plan.md step 4. Temperature 1.0 is the difference. Use ~0.29,
#     not 0.36, as the SI step-0 expectation.
#   * support_faithful is ~1.23-1.32, i.e. NEAR ITS FLOOR from step 0, so it has
#     little headroom left to detect the assert-dependence gaming mode it was meant to
#     act as a tripwire for.
_v6._grounding_text_of = lambda idea: idea.provenance_text()
# ...and the JUDGED INSIGHT is the problem_gap, NOT the core_claim. The two prompts
# put the document relationship in different fields. v0 requires the claim prose to
# state "what the cited docs do" and "the bridge" from docs to claim, so its
# insight_claim is what the decorative-citation test should read. SI's core_claim is
# specified as a standalone falsifiable bet -- "the couple of sentences the whole idea
# lives or dies on... do not just restate the gap" -- while problem_gap is the field
# "drawn from across the source documents... that connects them and that none resolves
# on its own" (prompts/ideation.py:365-382). Asking "would core_claim collapse without
# the docs?" would score the SCHEMA rather than the drift: SI is told to write that
# field so it stands alone, so a perfectly grounded card reads decorative.
# The gap is also what the model must SHIFT in order to drift off-document, which is
# the behaviour the gate exists to price.
_v6._judged_claim_of = lambda idea: idea.problem_gap
_v6.GROUNDING_REQUIRE_TYPE = False
# Belt-and-braces: the v0 shim's import already did these two, but state them
# here so this file is readable on its own and order-independent.
_v6._idea_client = _bedrock_idea_client
_v6.IDEA_JUDGE_MODEL = BEDROCK_JUDGE_MODEL

# v6's compute_score, unmodified. Re-exporting the function object (not a wrapper)
# keeps inspect.iscoroutinefunction true so verl's async reward manager awaits it
# on its own long-lived loop.
compute_score = _v6.compute_score

# Re-export the knobs a launcher or test may want to assert on.
K = _v6.K
LEN_COEF = _v6.LEN_COEF
LEN_TARGET = _v6.LEN_TARGET
IMPACT_MODEL = _v6.IMPACT_MODEL

__all__ = ["compute_score", "BEDROCK_JUDGE_MODEL", "IMPACT_MODEL",
           "K", "LEN_COEF", "LEN_TARGET", "_format_value_si"]
