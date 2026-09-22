"""F9-B v6 (impact x distance + CARD-ONLY LENGTH PENALTY): v4, retargeted.

Forked from f9b_impdist_len_v4.py. IDENTICAL reward EXCEPT the verbosity penalty
is measured over the CARD text only ({Problem, Insight, Motivation} joined across
the K ideas -- exactly what the impact judge scores), NOT the whole <answer> JSON
block. Grounding (source/supports), grounding_type, and JSON scaffolding are NO
LONGER counted, so the penalty can no longer be relieved by shrinking/dropping
grounding (the v4 failure mode: grounding earns zero reward yet was taxed, so RL
cut it). LEN_TARGET is a per-SET CARD budget, calibrated with
scripts/measure_train_tokens.py. Default 1670 = train_multi.jsonl whole-set card
p90 (Qwen3 tokens), i.e. the SFT target's own K=5 length -- so RL holds length at
the SFT-init baseline rather than being pushed below it. The terser alternative is
the gold anchor: per-idea gold card p90 ~= 256 tok x K=5 ~= 1300/set, which applies
genuine downward pressure (multi cards run ~34% longer per idea than gold).
Qwen3 and OLMo-3 tokenize this prose within ~0.5% (1670 vs 1664), so the target
does NOT need rescaling between the two policies.

Identical reward to rewards/f9b_impdist_v0.py -- linear impact quality times the
DECOMPOSED (0.1/0.3/0.6) idea-vs-idea min-distance diversity term, novelty dropped
-- with ONE addition: a VERBOSITY penalty on the length of the whole <answer> JSON
block, subtracted from the score so the policy is nudged toward an offline-estimated
target answer length WITHOUT shedding ideas. One rollout emits K ideas as a JSON
list inside <answer>...</answer>; this scores the whole set:

    R(U) =  0.0                                                if schema invalid (no think/answer tags OR != K parseable ideas)
            0.05 + sum_i  I(u_i)*min_{j<i} d(u_i,u_j)  - P(U)  if schema valid

      f(u)        = I(u)                            per-idea quality == impact (LINEAR)
      I(u)        in {0, 0.5, 1}                    impact (SciJudge, idea vs gold, both A/B)
      d(u_i,u_j)  = 0.1*d_mot + 0.3*d_prob + 0.6*d_ins   distance (LLM judge, idea vs idea)
      min_{j<i} d = 1 for the first idea (distance from the empty set)
      P(U)        = LEN_COEF * min(max(0, answer_len/LEN_TARGET - (1+LEN_TOL)), LEN_CAP)

The length term P(U) is the ONLY difference from v0. It is:
  * MEASURED over the ENTIRE <answer> JSON block -- every idea's fields, its
    grounding citations, AND the JSON structure itself (keys/brackets/quotes) --
    but NOT the <think> reasoning block. It targets verbosity, NOT the idea count
    (the K-gate already makes dropping ideas terminal; with K fixed the block
    target is ~K x per-idea).
  * ONE-SIDED: it only ever punishes going OVER target; being short is never
    rewarded (that would collapse to terse, low-content ideas).
  * NORMALIZED by the target (LEN_COEF is scale-free), DEADBANDED (LEN_TOL slack),
    and CAPPED (LEN_CAP bounds the worst case so a runaway-length set cannot blow
    up GRPO's group-relative advantage).
  * OFF by default (F9B_LEN_COEF=0) -- with the default this file is byte-for-byte
    equivalent in reward to v0, just with two extra logged-only fields.

Everything else is copied verbatim from f9b_impdist_v0: the FINDING-7 set-mean
imputation of dropped judge calls, the hardened httpx clients (structured timeouts
+ keepalive expiry), the endpoint-split concurrency caps, the deterministic
distance sub-sampling, the optional impact-first curriculum, and the single-file
append-only error log.

Cost per rollout ~ 2*K impact + (sampled) distance judge calls (length adds no
judge calls -- it is measured locally from the generation). Impact hits SciJudge
(SCIJUDGE_* / SCIJUDGE_ENDPOINT_FILE); distance hits the idea judge (F9B_JUDGE_* /
JUDGE_ENDPOINT_FILE, e.g. gpt-oss-120b). Rubric/parse math lives in common/.

verl wiring (per-sample naive reward manager; one call == one full K-idea set):
    custom_reward_function.path=.../grpo_training/rewards/f9b_impdist_len_v4.py
    custom_reward_function.name=compute_score

compute_score returns a DICT so verl logs the components independently
(reward_extra_info): score, format, n_ideas, impact/quality/mindist means,
answer_len, len_penalty, impact_fail. novelty_mean is logged as a constant
0.0 so W&B panels shared with the f9b runs don't error on a missing key.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

# Make the grpo_training root importable whether run as a script or loaded by
# verl via custom_reward_function.path (importlib sets __file__ to the abs path).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import clients, distance, grounding, scijudge  # noqa: E402  (novelty import dropped)
from common.ideas import DEFAULT_K, GROUNDING_TYPES, Idea, extract_answer_str, parse_ideas  # noqa: E402

# --------------------------------------------------------------------------- #
# Config (env-overridable; all have sane on-cluster defaults)
# --------------------------------------------------------------------------- #
K = int(os.environ.get("F9B_NUM_IDEAS", str(DEFAULT_K)))            # required idea count
SCHEMA_BONUS = float(os.environ.get("F9B_SCHEMA_BONUS", "0.05"))  # additive bonus for a schema-valid set (binary: 0.0 otherwise)

# Distance uses the "idea judge" (a general chat model, e.g. gpt-oss-120b served
# on JUDGE_PORT). Impact uses the SciJudge endpoint. (Novelty, which also used the
# idea judge in f9b, is gone -- so the idea judge now serves distance only.)
IDEA_JUDGE_MODEL = os.environ.get("F9B_JUDGE_MODEL", "openai/gpt-oss-120b")
IMPACT_MODEL = scijudge.DEFAULT_MODEL

DISTANCE_MAX_TOKENS = int(os.environ.get("F9B_DISTANCE_MAX_TOKENS", "1024"))
IMPACT_MAX_TOKENS = int(os.environ.get("SCIJUDGE_MAX_TOKENS", "2048"))
JUDGE_TEMPERATURE = float(os.environ.get("F9B_JUDGE_TEMPERATURE", "0.0"))
# Per-CALL wall-clock (s). Default 120: a healthy 2048-tok SciJudge decode is
# ~10-16s on an H200, so 120 covers decode + prefill + a modest queue margin and
# fails a wedged/half-open socket FAST. A large timeout (the old 600) only ever
# masked wedged pools -- it let requests pile into the queue and deepened the
# collapse (run 26070). Keep this small and lean on the per-worker concurrency
# cap + per-set budget below to bound the worst case instead.
REQUEST_TIMEOUT = float(os.environ.get("F9B_TIMEOUT", "120"))
# Drop idle pooled sockets aggressively. A judge we hit in BURSTS (val) doesn't
# benefit from long-lived keepalive, and a stale/half-open pooled socket is
# exactly what wedged run 26070 -- a short expiry forces a fresh re-dial instead
# of awaiting a dead connection for the full REQUEST_TIMEOUT.
KEEPALIVE_EXPIRY = float(os.environ.get("F9B_KEEPALIVE_EXPIRY", "5.0"))
# PER-WORKER in-flight judge caps, split by ENDPOINT. These are now MODULE-LEVEL
# (one semaphore per worker process / event loop, shared across ALL _score_set
# calls) -- NOT per-set as before. With the native-async reward, verl awaits many
# sets concurrently on one loop, so a per-set cap never bound anything and offered
# concurrency to the judges was effectively unbounded (val bursts drove SciJudge
# to its 256 max_num_seqs ceiling). The true offered concurrency is
# (num_workers x cap); with ~8 reward workers the defaults below target a global
# ~128 impact / ~80 distance, high enough not to slow steady-state training but
# low enough to keep the val-burst queue shallow so the tight REQUEST_TIMEOUT is
# safe. gpt-oss-120b (distance) is heavier per call, so its cap is lower.
MAX_CONCURRENT_IDEA = int(os.environ.get("F9B_MAX_CONCURRENT_IDEA", "10"))    # gpt-oss: distance (per worker)
MAX_CONCURRENT_IMPACT = int(os.environ.get("F9B_MAX_CONCURRENT_IMPACT", "16"))  # SciJudge: impact (per worker)
# Back-compat: the old single knob, if set, seeds BOTH (preserves prior behaviour).
_legacy_cc = os.environ.get("F9B_MAX_CONCURRENT")
if _legacy_cc:
    MAX_CONCURRENT_IDEA = MAX_CONCURRENT_IMPACT = int(_legacy_cc)
# httpx pool must admit both endpoints in-flight at once.
MAX_CONCURRENT = MAX_CONCURRENT_IDEA + MAX_CONCURRENT_IMPACT    # httpx pool sizing only
# Per-SET wall-clock budget (s): a hard ceiling on how long scoring ONE K-idea set
# may take, no matter what goes wrong downstream (retry pileup, semaphore
# contention, a half-wedged pool). On expiry we cancel the set's outstanding judge
# calls and IMPUTE the remainder (counted as failures so the watchdog sees it),
# guaranteeing a reward set can never outlast a rollout and wedge the trainer.
# 240s (2x REQUEST_TIMEOUT): a HEALTHY set under full fleet load already runs ~145s
# (measured via judge_load_test.py at 8-worker scale), so a tighter budget trips
# healthy-but-queued sets; 240 covers one slow wave (2x the 120s call timeout) while
# still bounding a genuinely wedged set. This is a SAFETY ceiling, not the primary
# load control -- that's keeping the val burst small (VAL_ROLLOUT_N=1) so it drains
# in ~75s. Set F9B_SET_BUDGET=0 to disable.
SET_BUDGET = float(os.environ.get("F9B_SET_BUDGET", "240"))
# Impact ordering count. Normally 2 (cand-vs-gold + gold-vs-cand, position-bias
# controlled -> {0,0.5,1}). Set F9B_IMPACT_SINGLE_ORDER=1 to fire ONE randomized
# ordering per idea instead (halves SciJudge load at the cost of a binary {0,1}
# per-idea signal). NOTE: this is a GLOBAL knob -- the reward worker cannot tell
# validation from training, so it cannot be scoped to val only; leave it OFF for
# training runs and only enable it for load tests / eval-only passes.
IMPACT_SINGLE_ORDER = os.environ.get("F9B_IMPACT_SINGLE_ORDER", "0") not in ("0", "false", "False", "")
TODAY = os.environ.get("SCIJUDGE_TODAY") or datetime.now().strftime("%Y-%m-%d")

# Failed judge calls arrive as None; _score_set IMPUTES them (MISSING DATA, not
# evidence the idea lost). Impact-judge failures CLUSTER per-set (outage/socket
# wedge), so imputation is three-tier (see _score_set): set-mean when successful
# calls are a majority, else the trailing-WINDOW mean of recent real impacts, and
# IMPACT_DEFAULT only at cold start. Kept at 0.5 (neutral), NOT 0.0, so a blackout
# never spuriously zeros quality. (FINDING 7, f9b_v1_curriculum.md; revised.)
IMPACT_DEFAULT = float(os.environ.get("F9B_IMPACT_DEFAULT", "0.5"))
# Trust a set-mean impute only when successful impact calls are at least this
# fraction of the set; below it (thin sample OR total blackout) fall back to the
# trailing-window mean. Failures cluster, so a lone survivor is nearly as noisy as
# a blackout -- don't let it dictate the set. 0.5 => require a surviving majority.
IMPACT_IMPUTE_MIN_FRAC = float(os.environ.get("F9B_IMPACT_IMPUTE_MIN_FRAC", "0.5"))
IMPACT_RETRIES = int(os.environ.get("F9B_IMPACT_RETRIES", "1"))
# Floor applied inside quality q=max(I, FLOOR). With LINEAR q=I this is OFF by
# default (0.0): a genuine impact=0 correctly contributes 0. (Under f9b's
# q=sqrt(N*I) the floor mattered; here it's an optional experiment knob.)
IMPACT_FLOOR = float(os.environ.get("F9B_IMPACT_FLOOR", "0.0"))
# Distance default 1.0 == "treat an unscored pair as fully distinct" so a judge
# failure never spuriously collapses diversity credit to 0. (Same FINDING 7 caveat
# as f9b_v1: inflationary under a timeout storm; left as-is, flagged.)
DISTANCE_DEFAULT = float(os.environ.get("F9B_DISTANCE_DEFAULT", "1.0"))
# Diversity is now the ONLY idea-judge consumer. mindist is a MIN, so sampling
# <=K earlier ideas per idea almost always preserves the true nearest neighbour at
# a fraction of the calls. 0 => keep the exhaustive C(n,2) matrix.
DISTANCE_SAMPLE_K = int(os.environ.get("F9B_DISTANCE_SAMPLE_K", "3"))

# --- LENGTH REGULARIZATION: a verbosity penalty on the impact x distance score
# (the ONLY thing this file adds over v0). Nudges CARD length toward an
# offline-measured target WITHOUT touching idea count (the K-gate already makes
# dropping ideas terminal, so shedding ideas to dodge the penalty is impossible --
# and with K fixed the set target is ~K x per-idea).
# Measured over the CARD text ONLY -- the K ideas' problem/insight/motivation
# joined, i.e. exactly what the impact judge scores. Grounding citations, the JSON
# scaffolding (keys/brackets/quotes), and the <think> block are ALL excluded, so
# the policy cannot relieve the penalty by cutting anything that earns reward
# elsewhere (that was the v4 failure mode). The penalty is
# ONE-SIDED (never rewards being short), NORMALIZED by the target (so LEN_COEF is
# scale-free), DEADBANDED (LEN_TOL slack so it doesn't fight natural variance), and
# CAPPED (a runaway-length set can't blow up GRPO's group-relative advantage):
#   answer_len = tokens of the K cards joined (NOT the whole <answer> block)
#   over       = max(0, answer_len/LEN_TARGET - (1 + LEN_TOL))   # fraction past tolerated target
#   penalty    = LEN_COEF * min(over, LEN_CAP)
#   score     -= penalty
# OFF by default (F9B_LEN_COEF=0): with the default the reward is identical to v0.
LEN_COEF = float(os.environ.get("F9B_LEN_COEF", "0.0"))       # 0 => length penalty OFF
# Per-SET CARD budget. 1670 = train_multi.jsonl whole-set card p90 in Qwen3 tokens
# (scripts/measure_train_tokens.py) -- the SFT target's own length, so RL HOLDS the
# SFT-init baseline. Use 1300 (gold p90 256/idea x K=5) to instead push shorter.
LEN_TARGET = float(os.environ.get("F9B_LEN_TARGET", "1670"))
LEN_TOL = float(os.environ.get("F9B_LEN_TOL", "0.2"))        # deadband: no penalty until answer_len > target*(1+tol)
LEN_CAP = float(os.environ.get("F9B_LEN_CAP", "2.0"))        # cap on the overage FRACTION (bounds the worst-case penalty)
# Tokenizer for measuring idea length: an HF name or local path. Set it so the
# count is in REAL tokens (matching a target you estimate offline with the same
# tokenizer); if unset OR it fails to load we fall back to whitespace WORD count
# (then estimate LEN_TARGET in words to match). Loaded lazily + cached, so the
# cost is amortized and trivial next to the network-bound judge calls.
LEN_TOKENIZER = os.environ.get("F9B_LEN_TOKENIZER", "").strip()

# --- GROUNDING GATE: a MULTIPLICATIVE per-idea factor g_i in [FLOOR, 1] on each
# idea's contribution, from the `load_bearing` feature of common.grounding's rubric
# (judged by the idea judge -- gpt-oss-120b -- against the SOURCE DOCUMENTS).
#
#     contrib_i = impact_i * mindist_i^beta * g_i
#
# WHY THIS EXISTS (job 328, 668 steps): score climbed 0.61 -> 3.95 and impact
# 0.16 -> 0.93 while the ideas progressively stopped being about their documents.
# Judged on 368 rollout ideas, `load_bearing` fell 2.99 -> 1.76 and the share of
# ideas at the "stands completely without the cited docs" floor went 15.5% -> 57.5%.
# Cause: NOTHING in v6 read grounding. v6 fixed the v4 failure's "grounding was
# TAXED" half (card-only length penalty) but never its "grounding EARNS ZERO" half,
# so grounding stayed presence-gated only. Once impact saturates (~step 170) the
# only remaining way to raise score is to push ideas apart, and the cheapest way to
# make K ideas mutually distant is to stop anchoring them to the same documents:
# corr(insight document-precision, mindist) = -0.927.
#
# WHY MULTIPLICATIVE: an ADDITIVE grounding term is a tax the policy routes around
# -- exactly the v4 failure. Multiplying means an ungrounded idea cannot collect
# diversity credit at all, so drifting off-document stops being the cheap win.
#
# WHY PER-IDEA: the drift is concentrated in the TAIL of the set (idea 1 was 7%
# off-topic late vs ideas 4-5 at ~60%), because each later idea only needs to be
# distant from the earlier ones. A set-mean g cannot penalize the specific ideas
# that are drifting.
#
# See common.grounding.features_to_g for the shape (graded, saturating at HI so
# there is no gradient toward keyword-stuffing or toward merely ASSERTING
# dependence, floored above 0 so within-group advantage variance survives) and for
# why load_bearing ALONE beats `fidelity`, the binary E gate, or load_bearing +
# support_faithful.
#
# OFF by default (F9B_GROUNDING=0): with the default this file fires NO grounding
# judge calls and every g_i is 1.0, so the reward is byte-for-byte what it was.
GROUNDING_ON = os.environ.get("F9B_GROUNDING", "0").strip().lower() in ("1", "true", "yes")
# Reuses the IDEA judge (same model/client/semaphore as distance) so enabling this
# needs no new endpoint. The Bedrock shim rebinds _idea_client/IDEA_JUDGE_MODEL and
# this call site follows along for free.
GROUNDING_MAX_TOKENS = int(os.environ.get("F9B_GROUNDING_MAX_TOKENS", "1536"))
# Defaults are the RUBRIC'S OWN 1-5 min-max: g = (load_bearing - 1) / 4, so
# 1/2/3/4/5 -> 0.00/0.25/0.50/0.75/1.00 and the multiplier is readable straight off
# load_bearing_mean. The saturating/floored variant (HI=3, FLOOR=0.25) is one env
# override away; common.grounding.features_to_g documents what it buys.
GROUNDING_LO = float(os.environ.get("F9B_GROUNDING_LO", "1.0"))
GROUNDING_HI = float(os.environ.get("F9B_GROUNDING_HI", "5.0"))
GROUNDING_FLOOR = float(os.environ.get("F9B_GROUNDING_FLOOR", "0.0"))
GROUNDING_RETRIES = int(os.environ.get("F9B_GROUNDING_RETRIES", "2"))
# Cold-start imputation fill, used ONLY before the trailing window has any real g.
# Kept SEPARATE from GROUNDING_FLOOR and neutral (0.5 == load_bearing 3 under the
# default map), for the same reason as IMPACT_DEFAULT: with the default FLOOR=0 a
# floor-valued fill would ZERO a set whose judge was merely unreachable, turning an
# outage into a total grounding penalty instead of a GRPO no-op.
GROUNDING_DEFAULT = float(os.environ.get("F9B_GROUNDING_DEFAULT", "0.5"))
# Require a valid grounding_type for g > floor. TRUE for the v0 schema (an absent
# type is real signal). The SI shim sets this False: SI cards have no grounding_type
# field at all, so every idea would otherwise pin to the floor.
GROUNDING_REQUIRE_TYPE = os.environ.get(
    "F9B_GROUNDING_REQUIRE_TYPE", "1").strip().lower() in ("1", "true", "yes")
# Trailing window of real g values, for imputing judge failures (mirrors _IMP_WIN).
GROUNDING_WINDOW = int(os.environ.get("F9B_GROUNDING_WINDOW", "256"))
# Fraction of a set's grounding calls that must succeed before we trust the
# set-mean as the imputation fill (mirrors IMPACT_IMPUTE_MIN_FRAC).
GROUNDING_IMPUTE_MIN_FRAC = float(os.environ.get("F9B_GROUNDING_IMPUTE_MIN_FRAC", "0.5"))

# --- MINDIST CAP: clamp the per-idea diversity factor before it is applied. 328
# drove mindist to 0.796, above what K genuinely grounded ideas drawn from 3-6
# related papers can reach -- the excess is bought by leaving the documents. This
# bounds the term so the gate and the cap can be evaluated independently.
# 0 => OFF (no clamp), which is the default.
MINDIST_CAP = float(os.environ.get("F9B_MINDIST_CAP", "0.0"))

# --- CURRICULUM (impact-first): a STEPWISE diversity exponent beta on the
# distance term, gated on a TRAILING WINDOW of recent impact ONLY (f9b_impdist has
# no novelty signal, so the gate is single-condition). Stage 1 keeps beta low so
# the reward is IMPACT-DOMINANT while the policy learns to beat gold; once the
# window-mean impact clears T_IMP, beta ratchets to BETA_HI and the full
# min-distance diversity term switches on. This mirrors f9b_v1's curriculum with
# the novelty condition removed.
#   i_bar  < T_IMP  -> beta = BETA_LO  (impact-dominant; distance barely matters)
#   i_bar >= T_IMP  -> beta = BETA_HI  (full diversity, == the fixed-beta impdist)
# F9B_CURRICULUM=0 (DEFAULT) keeps the ORIGINAL impdist behaviour: fixed beta =
# BETA_HI, full distance from step 0. Set F9B_CURRICULUM=1 to enable staging.
CURRICULUM = os.environ.get("F9B_CURRICULUM", "0") not in ("0", "false", "False", "")
BETA_LO = float(os.environ.get("F9B_BETA_LO", "0.2"))   # stage-1 exponent (impact-dominant)
# Clamp BETA_LO strictly > 0: at beta==0 Python evaluates 0.0**0 as 1.0, so a
# zero-distance (duplicate) pair would score as MAXIMALLY diverse -- backwards.
BETA_LO = max(BETA_LO, 1e-3)
BETA_HI = float(os.environ.get("F9B_BETA_HI", "1.0"))   # stage-2 exponent (full diversity)
# Advance when the trailing-window mean impact clears this. NOTE: impact_mean has
# sat ~0.2 early in impdist runs, so 0.35 requires genuine improvement before
# distance turns on; if impact never clears it the run simply stays impact-only
# (a safe fallback, just not the full curriculum).
T_IMP = float(os.environ.get("F9B_T_IMP", "0.35"))
WINDOW = int(os.environ.get("F9B_WINDOW", "640"))       # trailing samples driving the gate
# Ratchet: once advanced, never fall back (a transient impact dip won't re-hide
# distance). Set F9B_BETA_RATCHET=0 to let beta track the window both ways.
BETA_RATCHET = os.environ.get("F9B_BETA_RATCHET", "1") not in ("0", "false", "False", "")
_IMP_WIN = collections.deque(maxlen=WINDOW)
_BETA_STATE = {"max": BETA_LO}   # ratcheted beta high-water mark


def _curriculum_beta() -> float:
    """Current diversity exponent from the trailing-impact window. Fixed BETA_HI if
    the curriculum is disabled; BETA_LO until the window fills (so early,
    high-variance steps do not trip the gate); then BETA_HI once i_bar >= T_IMP.
    Ratcheted by default so it never falls back."""
    if not CURRICULUM:
        return BETA_HI
    if len(_IMP_WIN) < WINDOW:
        beta = BETA_LO
    else:
        i_bar = sum(_IMP_WIN) / len(_IMP_WIN)
        beta = BETA_HI if i_bar >= T_IMP else BETA_LO
    if BETA_RATCHET:
        beta = max(beta, _BETA_STATE["max"])
        _BETA_STATE["max"] = beta
    return beta


def _update_quality_window(impact_mean: float) -> None:
    """Append THIS set's impact mean to the trailing window (called AFTER reading
    beta, so a sample never gates on its own impact)."""
    _IMP_WIN.append(impact_mean)


# Trailing window of SUCCESSFULLY JUDGED per-idea g values. Used ONLY to impute
# grounding-judge failures: a timeout is MISSING DATA, and filling it with the
# recent real mean makes an outage a GRPO no-op (imputed ~= group mean -> advantage
# ~= 0) instead of a spurious grounding penalty. Same rationale as _IMP_WIN's role
# in the FINDING-7 impact imputation.
_G_WIN = collections.deque(maxlen=GROUNDING_WINDOW)


def _g_window_mean() -> float:
    """Recent real per-idea g, or GROUNDING_DEFAULT at cold start (no samples yet).

    NOT GROUNDING_FLOOR: the default floor is 0.0, and filling an unjudged set with
    0.0 would zero its whole body -- an outage must be a no-op, not a penalty."""
    return (sum(_G_WIN) / len(_G_WIN)) if _G_WIN else GROUNDING_DEFAULT


DEBUG_DIR = os.environ.get("F9B_DEBUG_DIR") or os.environ.get("SCIJUDGE_DEBUG_DIR")

# Throttled console dump of a full scored set (1 of every N completed rollouts).
_LOG_EVERY = max(1, int(os.environ.get("F9B_LOG_EVERY", "5")))
_LOG_PROMPT_CHARS = int(os.environ.get("F9B_LOG_PROMPT_CHARS", "1200"))
_log_lock = threading.Lock()
_completed = 0

_THINK_RE = re.compile(r"(?:<think>)?.*</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>.*</answer>", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Clients (one per judge endpoint; built lazily, reused across calls)
# --------------------------------------------------------------------------- #
# Max times AsyncOpenAI re-dials (fresh connection) on a failed/timed-out call.
# Default 1 (was 2): retries against a wedged pool just re-block and multiply load
# on an already-slow queue. One re-dial is enough to shake off a transient blip;
# beyond that we'd rather fail fast and let imputation/the watchdog handle it.
_JUDGE_MAX_RETRIES = int(os.environ.get("F9B_JUDGE_MAX_RETRIES", "1"))


def _hardened_async_client() -> "httpx.AsyncClient":
    """httpx client that bounds dead/half-open sockets, not just slow responses.

    A STRUCTURED timeout adds a `pool` bound (waiting to acquire/reuse a
    connection) and a SHORT `keepalive_expiry` drops idle pooled sockets so a
    wedged one is re-dialed instead of awaited forever (the step-56 CLOSE-WAIT
    hang / the run-26070 pool wedge)."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            REQUEST_TIMEOUT,            # default (read) bound
            connect=min(10.0, REQUEST_TIMEOUT),
            write=min(30.0, REQUEST_TIMEOUT),
            pool=min(10.0, REQUEST_TIMEOUT),
        ),
        limits=httpx.Limits(max_connections=MAX_CONCURRENT + 8,
                            max_keepalive_connections=MAX_CONCURRENT,
                            keepalive_expiry=KEEPALIVE_EXPIRY),
    )


def _vllm_client(base_url: Optional[str], endpoint_file: Optional[str]) -> AsyncOpenAI:
    """Self-hosted vLLM judge client (mirrors clients.build_async_client's vllm
    branch) but with the hardened http client + retries. Both judges in this
    deployment (SciJudge-30B impact, gpt-oss-120b distance) are vLLM."""
    url = scijudge.resolve_base_url(base_url, endpoint_file)
    return AsyncOpenAI(base_url=url, api_key="EMPTY", default_headers={},
                       http_client=_hardened_async_client(),
                       max_retries=_JUDGE_MAX_RETRIES)


@lru_cache(maxsize=1)
def _idea_client() -> AsyncOpenAI:
    """Client for the distance judge. Resolves its URL from F9B_JUDGE_* / JUDGE_*
    (independent of SCIJUDGE_*, so the two judges can live on different
    endpoints)."""
    base_url = (os.environ.get("F9B_JUDGE_BASE_URL", "").strip()
                or os.environ.get("JUDGE_BASE_URL", "").strip() or None)
    endpoint_file = os.environ.get("JUDGE_ENDPOINT_FILE", "").strip() or None
    # gpt-oss-* matches gateway.is_hosted (the "gpt" prefix) but is a self-hosted
    # vLLM judge -> clients._looks_local catches it. Honor an explicit
    # F9B_JUDGE_BACKEND override; else treat hosted-and-not-local as gateway.
    backend = os.environ.get("F9B_JUDGE_BACKEND", "").strip().lower()
    if backend not in clients.VALID_BACKENDS:
        backend = ("gateway" if (clients.gateway.is_hosted(IDEA_JUDGE_MODEL)
                                 and not clients._looks_local(IDEA_JUDGE_MODEL))
                   else "vllm")
    if backend == "gateway":
        return clients.build_async_client("gateway", IDEA_JUDGE_MODEL,
                                          base_url=base_url, timeout=REQUEST_TIMEOUT)
    # vllm: pass the resolved URL as base_url so SCIJUDGE_BASE_URL never leaks in.
    return _vllm_client(base_url, endpoint_file)


@lru_cache(maxsize=1)
def _impact_client() -> AsyncOpenAI:
    """Client for the SciJudge impact judge (SCIJUDGE_* / endpoint file)."""
    if scijudge.is_gateway_model(IMPACT_MODEL):
        return clients.build_async_client("gateway", IMPACT_MODEL, timeout=REQUEST_TIMEOUT)
    return _vllm_client(None, None)


# Single append-only error log + running counter (replaces the one-file-per-failure
# writer that caused a Lustre MDS metadata storm on job 25344). One cached file
# handle, atomic single-line O_APPEND writes (safe across Ray workers), count
# echoed every 100 errors so a judge problem is visible at the step it STARTS.
_debug_lock = threading.Lock()
_debug_fh = None
_judge_error_count = 0


def _write_debug(prefix: str, body: str) -> None:
    """Append ONE line ('<ts>\t#<n>\t<prefix>\t<body>') per failure to
    DEBUG_DIR/f9b_errors.log and bump a running counter. Never creates a file per
    call. Best-effort: a logging failure must never break scoring."""
    global _debug_fh, _judge_error_count
    if not DEBUG_DIR:
        return
    line = (body or "").replace("\n", " | ").strip()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _debug_lock:
        _judge_error_count += 1
        n = _judge_error_count
        try:
            if _debug_fh is None:
                root = Path(DEBUG_DIR)
                root.mkdir(parents=True, exist_ok=True)
                _debug_fh = open(root / "f9b_errors.log", "a", encoding="utf-8")
            _debug_fh.write(f"{ts}\t#{n}\t{prefix}\t{line}\n")
            _debug_fh.flush()
        except Exception:
            pass  # never let debug logging break scoring
    if n % 100 == 0:
        print(f"[f9b-impdist-len] judge/scoring error count = {n} "
              f"(latest {prefix}: {line[:120]})", flush=True)


# --------------------------------------------------------------------------- #
# Format gate
# --------------------------------------------------------------------------- #
def _format_value(solution_str: str, ideas: List[Idea]) -> Optional[float]:
    """Format gate (BINARY). Returns None if schema-valid (proceed to scoring),
    else the terminal reward 0.0 -- for BOTH missing think/answer tags AND tags
    present but not exactly K parseable ideas. The schema bonus is added in
    _score_set, not here."""
    text = solution_str or ""
    if not (_THINK_RE.search(text) and _ANSWER_RE.search(text)):
        return 0.0
    if len(ideas) != K:
        return 0.0
    # Every idea must carry REAL grounding: a declared grounding_type from the
    # closed vocabulary AND >=1 citation whose source and supports are both
    # non-empty. The instruction asks for this, but the impact judge only sees
    # the {problem, insight, motivation} card -- so without this gate the policy
    # learns to drop grounding entirely (it costs tokens, earns no reward). A
    # bare grounding_type with an empty/absent grounding list (or stub citations)
    # fails the gate -> terminal 0.0, same as a missing <answer>.
    for idea in ideas:
        if idea.grounding_type not in GROUNDING_TYPES:
            return 0.0
        if not idea.grounding:
            return 0.0
        if not all(g.source and g.supports for g in idea.grounding):
            return 0.0
    return None


# --------------------------------------------------------------------------- #
# Length regularization (per-idea verbosity penalty)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _length_tokenizer():
    """Cached HF tokenizer for idea-length measurement, or None to fall back to
    word count. Lazy so an import-only worker pays nothing; a load failure DEGRADES
    to word-count (logged once) rather than crashing the reward."""
    if not LEN_TOKENIZER:
        return None
    try:
        from transformers import AutoTokenizer  # noqa: WPS433
        return AutoTokenizer.from_pretrained(LEN_TOKENIZER)
    except Exception as exc:
        _write_debug("f9b_len_tokenizer", f"{type(exc).__name__}: {exc}")
        return None


def _token_len(text: str) -> int:
    """Length of `text` in REAL tokens (if LEN_TOKENIZER loaded) else in whitespace
    words. Empty -> 0. Never raises: a tokenizer hiccup falls back to word count so
    length measurement can't break scoring."""
    text = text or ""
    tok = _length_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(text.split())


def _card_concat(ideas):
    # v6 length-measurement text: the {Problem, Insight, Motivation} cards the
    # impact judge actually scores, joined across the K ideas. Grounding is
    # EXCLUDED by construction (card_text() never renders it), so the penalty
    # prices only judge-visible verbosity and cannot be gamed by cutting grounding.
    return chr(10).join(u.card_text() for u in ideas)


def _length_penalty(answer_text: str) -> Tuple[float, float]:
    """(answer_len, penalty) for a rollout. answer_len is the token (or word) count
    of the WHOLE <answer> JSON block -- every idea's fields, its grounding, and the
    JSON structure itself. Penalty is one-sided / normalized / deadbanded / capped,
    and is 0 whenever the feature is OFF (LEN_COEF<=0) or answer_len is within
    LEN_TARGET*(1+LEN_TOL). answer_len is still returned when OFF so it can be logged
    for offline target calibration."""
    answer_len = float(_token_len(answer_text))
    if LEN_COEF <= 0.0 or LEN_TARGET <= 0.0:
        return answer_len, 0.0
    over = max(0.0, answer_len / LEN_TARGET - (1.0 + LEN_TOL))
    penalty = LEN_COEF * min(over, LEN_CAP)
    return answer_len, penalty


# --------------------------------------------------------------------------- #
# Per-signal scoring (async; each owns its retry + parse + reduce)
# --------------------------------------------------------------------------- #
async def _judge_chat(client: AsyncOpenAI, model: str, messages, *,
                      max_tokens: int, scijudge_style: bool,
                      label: str = "judge") -> Optional[str]:
    """One chat completion -> content string (or None on error). scijudge_style
    uses scijudge.sampling_kwargs (top_k/top_p tuned for the RM); otherwise the
    generic per-model generation_config + temperature."""
    try:
        if scijudge_style:
            kw = scijudge.sampling_kwargs(max_tokens, model)
        else:
            kw = dict(clients.generation_config(model, max_tokens))
            if clients.accepts_temperature(model):
                kw["temperature"] = JUDGE_TEMPERATURE
        resp = await client.chat.completions.create(model=model, messages=messages, **kw)
        return resp.choices[0].message.content or ""
    except Exception as exc:
        _write_debug(f"f9b_judge_error:{label}", f"{type(exc).__name__}: {exc}")
        return None


async def _impact(idea: Idea, gold: str, sem: asyncio.Semaphore) -> Optional[float]:
    """I(u) in {0,0.5,1}: SciJudge win-rate of the idea vs the gold idea across
    both A/B orderings. No gold -> None (imputed). Each ordering RETRIES on failure;
    if either ordering still can't be parsed we return None (a partial signal is
    position-biased) -> _score_set imputes the set-mean instead of scoring 0.0."""
    gold = (gold or "").strip()
    if not gold:
        return None
    cand = idea.card_text()

    async def winner(a: str, b: str) -> Optional[str]:
        messages = scijudge.build_messages(a, b, TODAY)
        for _ in range(IMPACT_RETRIES):  # retry each ordering on timeout/unparsed
            async with sem:
                raw = await _judge_chat(_impact_client(), IMPACT_MODEL, messages,
                                        max_tokens=IMPACT_MAX_TOKENS, scijudge_style=True,
                                        label="impact")
            ans = scijudge.parse_answer(raw or "")
            if ans is not None:
                return ans
        return None

    # SINGLE-ORDER mode (F9B_IMPACT_SINGLE_ORDER=1): fire ONE randomized ordering
    # (halves SciJudge load). Randomizing which side cand takes de-biases position
    # bias in expectation across the batch; the per-idea signal is binary {0,1}.
    if IMPACT_SINGLE_ORDER:
        if random.random() < 0.5:
            ans = await winner(cand, gold)      # cand=A -> cand wins iff 'A'
            return None if ans is None else (1.0 if ans == "A" else 0.0)
        ans = await winner(gold, cand)          # cand=B -> cand wins iff 'B'
        return None if ans is None else (1.0 if ans == "B" else 0.0)

    # ordering 1: cand=A, gold=B (cand wins iff 'A'); ordering 2 swapped (iff 'B').
    ans1, ans2 = await asyncio.gather(winner(cand, gold), winner(gold, cand))
    if ans1 is None or ans2 is None:
        return None  # missing/partial signal is position-biased -> impute
    return scijudge.winrate_a(ans1, ans2)


async def _distance(idea_i: Idea, idea_j: Idea, sem: asyncio.Semaphore) -> Tuple[float, bool]:
    """d(u_i,u_j) in [0,1]: one idea-vs-idea similarity judge call, inverted +
    F9-B weighted. Parse failure -> DISTANCE_DEFAULT (treat as fully distinct so a
    failure never spuriously collapses diversity credit)."""
    messages = distance.build_messages(idea_i.card_text(), idea_j.card_text())
    for _ in range(2):
        async with sem:
            raw = await _judge_chat(_idea_client(), IDEA_JUDGE_MODEL, messages,
                                    max_tokens=DISTANCE_MAX_TOKENS, scijudge_style=False,
                                    label="distance")
        d = distance.distance_from_raw(raw or "")
        if d is not None:
            return d, False
    return DISTANCE_DEFAULT, True  # parse/timeout failure -> counted as distance_fail


# --------------------------------------------------------------------------- #
# Grounding gate: source documents + the load_bearing judge
# --------------------------------------------------------------------------- #
# The rendered prompt's document block, e.g. "\nDocument 3:\n". The prompt puts the
# corpus after a "SOURCE DOCUMENTS" banner (prompts/ideation.py), so we anchor on
# that first to avoid splitting on a "Document N:" the POLICY happened to echo.
_DOC_MARKER = "SOURCE DOCUMENTS"
_DOC_SPLIT_RE = re.compile(r"\n\s*Document\s+\d+\s*:\s*\n")


@lru_cache(maxsize=256)
def _parse_documents_cached(query: str) -> Tuple[str, ...]:
    """Source documents parsed out of the rendered prompt, in order.

    CACHED because every rollout in a GRPO group shares one prompt: at
    rollout_n=8 x train_batch=12 a step is 96 rollouts over only 12 DISTINCT
    prompts, so this does 12 splits instead of 96. Keyed on the whole prompt
    string (hashable, and identical across the group by construction).

    Returns () when the block is absent or unsplittable -- the caller treats that
    as "no documents", which IMPUTES rather than judging against nothing."""
    if not query:
        return tuple()
    i = query.find(_DOC_MARKER)
    block = query[i + len(_DOC_MARKER):] if i >= 0 else query
    parts = _DOC_SPLIT_RE.split("\n" + block)
    docs = [p.strip() for p in parts if p and p.strip()]
    # parts[0] is whatever preceded the first "Document 1:" header (the banner's
    # rule characters). Drop it: a real doc list always splits into >1 piece.
    if len(docs) > 1:
        docs = docs[1:]
    elif i < 0:
        return tuple()  # no marker AND no split -> this isn't an ideation prompt
    return tuple(docs)


def _documents_of(extra_info: Optional[Dict[str, Any]]) -> List[str]:
    """Source documents for this rollout, from extra_info['query'] (the rendered
    prompt, which carries the corpus inline -- verified present in the RL parquet
    and in job 328's logs). No dataset change was needed to get these."""
    query = str((extra_info or {}).get("query", "") or "")
    return list(_parse_documents_cached(query))


def _grounding_text_of(idea: Idea) -> str:
    """The idea's DECLARED provenance, rendered for the grounding judge.

    Indirection so the SI shim can rebind it: the v0 schema has an
    `insight.grounding` list (Idea.grounding_text()), whereas SI carries a free-text
    `inspiration` field. See f9b_impdist_len_v6_si_bedrock.py."""
    return idea.grounding_text()


def _judged_claim_of(idea: Idea) -> str:
    """The text the grounding rubric treats as THE INSIGHT ("would this collapse if
    the cited documents were removed?").

    A hook, for the same reason as _grounding_text_of: WHICH FIELD carries the
    document relationship is a property of the SCHEMA, and only the shim knows which
    schema is in play. v0's prompt requires the claim prose itself to state "what the
    docs do" and "the bridge" from docs to claim (prompts/ideation.py:57-72), so
    `insight_claim` is the right subject there.

    SI's prompt splits those apart on purpose: `core_claim` is "the couple of
    sentences the whole idea lives or dies on... do not just restate the gap" -- a
    standalone falsifiable bet -- while the document relationship lives in
    `problem_gap` ("drawn from across the source documents... that connects them and
    that none resolves on its own"). Scoring SI's `core_claim` for load_bearing would
    therefore measure the SCHEMA (a field written to stand alone) rather than DRIFT,
    and would read low even for a perfectly grounded card. The SI shim rebinds this."""
    return idea.insight_claim


async def _grounding(idea: Idea, documents: List[str],
                     sem: asyncio.Semaphore) -> Tuple[Optional[Dict[str, Any]], bool]:
    """One grounding-rubric judge call -> (features | None, failed).

    Mirrors _distance's shape (retry, shared idea-judge client + semaphore), but
    returns the RAW FEATURE DICT rather than a reduced score so all four features
    can be logged -- only `load_bearing` is priced (see common.grounding), and
    `support_faithful` / `source_exists` ride along free as tripwires for the
    policy learning to ASSERT dependence it doesn't have.

    NO DOCUMENTS -> (None, False): a prompt whose corpus we cannot parse is MISSING
    DATA, not a grounding failure. grounding.build_messages would happily render
    "(no source documents provided)" and the judge would score against nothing, so
    we refuse the call and let the caller impute. `False` (not a failure) keeps the
    grounding_fail metric meaning "the JUDGE failed", not "the prompt was odd"."""
    if not documents:
        return None, False
    # GROUNDING_REQUIRE_TYPE is off for schemas with no grounding_type field (the SI
    # shim). Passing it to build_messages OMITS the type rubric entirely rather than
    # rendering "(none declared / ungrounded)", which would tell the judge the idea is
    # ungrounded in the same breath as asking whether its citation is decorative --
    # biasing the one feature the reward prices. parse_features must agree, or the
    # (correctly) absent type_correct key reads as a truncated reply.
    messages = grounding.build_messages(_judged_claim_of(idea), _grounding_text_of(idea),
                                        idea.grounding_type, documents,
                                        ask_type=GROUNDING_REQUIRE_TYPE)
    for _ in range(GROUNDING_RETRIES):
        async with sem:
            raw = await _judge_chat(_idea_client(), IDEA_JUDGE_MODEL, messages,
                                    max_tokens=GROUNDING_MAX_TOKENS,
                                    scijudge_style=False, label="grounding")
        obj = grounding.extract_json(raw or "")
        feats = (grounding.parse_features(obj, require_type=GROUNDING_REQUIRE_TYPE)
                 if isinstance(obj, dict) else None)
        if feats is not None:
            return feats, False
    return None, True  # parse/timeout failure -> counted as grounding_fail, imputed


# --------------------------------------------------------------------------- #
# Per-worker concurrency caps (module-level, shared across ALL sets)
# --------------------------------------------------------------------------- #
# One semaphore per (endpoint, event-loop). Created lazily on first use so it
# binds to the worker's persistent loop (asyncio.Semaphore binds to the running
# loop, not import-time). Keyed by loop id so a test harness that spins a fresh
# loop gets its own -- but in a reward worker there is exactly ONE long-lived
# loop, so this is effectively a single per-worker cap shared by every rollout.
# This is the real backpressure: it bounds offered concurrency to (workers x cap)
# instead of (sets-in-flight x calls-per-set), keeping the val-burst queue shallow
# enough that the tight REQUEST_TIMEOUT never fires on healthy-but-queued calls.
_SEM_CACHE: Dict[Tuple[str, int], asyncio.Semaphore] = {}


def _get_sem(kind: str, size: int) -> asyncio.Semaphore:
    # INVARIANT (do NOT refactor away): the get_running_loop() call MUST stay, and
    # this MUST only ever be called from inside the awaited compute_score. It is the
    # guard that keeps the per-worker loop-binding correct: it raises RuntimeError if
    # ever run with no loop (e.g. hoisted to import time), so a wrong-loop binding
    # can never happen silently. Do NOT precompute _SEM_CACHE at module load or
    # construct the Semaphore anywhere a loop isn't running. (A Semaphore actually
    # binds on first SUSPEND, not construction, but keying by the running loop here
    # is what guarantees each loop gets its own -- the same reason the @lru_cache'd
    # httpx judge clients must also be built lazily inside the loop.)
    loop = asyncio.get_running_loop()
    key = (kind, id(loop))
    sem = _SEM_CACHE.get(key)
    if sem is None:
        sem = asyncio.Semaphore(size)
        _SEM_CACHE[key] = sem
    return sem


# --------------------------------------------------------------------------- #
# Set-level assembly
# --------------------------------------------------------------------------- #
def _capped_mindist(md: float) -> float:
    """mindist clamped to MINDIST_CAP (0 => no clamp). Bounds how much credit pure
    mutual-distance can buy; see the MINDIST_CAP comment above."""
    if MINDIST_CAP and MINDIST_CAP > 0.0:
        return min(md, MINDIST_CAP)
    return md


# The grounding feature keys logged as unpriced means. `load_bearing` is the one the
# gate prices; the rest ride along free (one judge call returns all four) as
# tripwires -- notably support_faithful, which would fall if the policy learns to
# ASSERT dependence it does not have.
_G_LOG_KEYS = ("load_bearing", "support_faithful", "source_exists", "type_correct")


def _grounding_metrics(gs: List[float], feats: List[Optional[Dict[str, Any]]],
                       n_fail: float) -> Dict[str, float]:
    """The grounding block of reward_extra_info. MUST return the SAME KEY SET on
    every path (scored / imputed / format-fail / exception) or verl's
    reward_extra_info arrays go ragged and W&B logging breaks."""
    out: Dict[str, float] = {
        "grounding_g_mean": (sum(gs) / len(gs)) if gs else 1.0,
        "grounding_fail": float(n_fail),
    }
    for k in _G_LOG_KEYS:
        vals = [f[k] for f in feats if isinstance(f, dict) and isinstance(f.get(k), (int, float))]
        # 0.0 == "not judged" (the rubric's real range is 1-5, so 0 is unambiguous).
        out[f"{k}_mean"] = (sum(vals) / len(vals)) if vals else 0.0
    return out



async def _score_set(ideas: List[Idea], gold: str, answer_text: str,
                     documents: Optional[List[str]] = None) -> Dict[str, Any]:
    """Score one K-idea set under a hard per-set wall-clock budget.

    Delegates to _score_set_impl, but wraps it in SET_BUDGET so scoring can never
    outlast a rollout: on timeout we cancel outstanding judge calls and impute the
    whole set (counted as failures so the metric + watchdog see it). ``answer_text``
    is the raw <answer> JSON block, measured for the length penalty. ``documents``
    are the source documents the grounding gate judges against (empty/None => the
    gate imputes; it is also skipped entirely when F9B_GROUNDING=0)."""
    if SET_BUDGET and SET_BUDGET > 0:
        try:
            return await asyncio.wait_for(
                _score_set_impl(ideas, gold, answer_text, documents), timeout=SET_BUDGET)
        except (asyncio.TimeoutError, TimeoutError):
            _write_debug("f9b_set_budget",
                         f"set exceeded SET_BUDGET={SET_BUDGET}s -> impute all ({len(ideas)} ideas)")
            return _imputed_set(ideas, answer_text)
    return await _score_set_impl(ideas, gold, answer_text, documents)


def _imputed_set(ideas: List[Idea], answer_text: str) -> Dict[str, Any]:
    """Degraded result for a set that blew the per-set budget: neutral (imputed)
    impact for every idea, max distance, ALL judge calls counted as failures. Keeps
    the exact key shape _score_set_impl returns so verl's reward_extra_info arrays
    stay rectangular. This is MISSING DATA (a NO-OP-ish reward), not a 0.0 penalty."""
    n = len(ideas)
    win_mean = (sum(_IMP_WIN) / len(_IMP_WIN)) if _IMP_WIN else IMPACT_DEFAULT
    impacts = [win_mean] * n
    qualities = [max(IMPACT_FLOOR, v) for v in impacts]
    mindists = [1.0] * n  # DISTANCE_DEFAULT semantics: unscored pair == fully distinct
    impact_mean = win_mean
    beta = _curriculum_beta()
    # Grounding is imputed too (trailing-window mean, or 1.0 with the gate off), so a
    # blackout set is not penalized for grounding it never got judged on.
    g_fill = _g_window_mean() if GROUNDING_ON else 1.0
    gs = [g_fill] * n
    contribs = [qualities[i] * (_capped_mindist(mindists[i]) ** beta) * gs[i]
                for i in range(n)]
    _update_quality_window(impact_mean)
    # every sampled pair failed -> mirror _score_set_impl's pair count
    pairs = 0
    for i in range(1, n):
        prev = i if not (0 < DISTANCE_SAMPLE_K < i) else DISTANCE_SAMPLE_K
        pairs += prev
    # length is judge-independent, so log answer_len for visibility, but keep the
    # imputed set a NO-OP: do NOT subtract a length penalty on a judge-blackout set.
    answer_len, _ = _length_penalty(_card_concat(ideas))  # v6: card-only
    return {
        "score": SCHEMA_BONUS + sum(contribs),
        "format": 1.0,
        "format_valid": 1.0,
        "n_ideas": float(n),
        "novelty_mean": 0.0,
        "impact_mean": impact_mean,
        "quality_mean": (sum(qualities) / n) if n else 0.0,
        "mindist_mean": (sum(mindists[1:]) / (n - 1)) if n > 1 else 0.0,
        "answer_len": answer_len,
        "len_penalty": 0.0,
        "curriculum_beta": beta,
        "impact_win": win_mean,
        "impact_fail": float(n),        # all impact calls treated as failed
        "distance_fail": float(pairs),  # all distance pairs treated as failed
        **_grounding_metrics(gs, [None] * n, float(n) if GROUNDING_ON else 0.0),
        "_impacts": impacts, "_qualities": qualities, "_mindists": mindists,
        "_gs": gs, "_lbs": [None] * n,
    }


async def _score_set_impl(ideas: List[Idea], gold: str, answer_text: str,
                          documents: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run all judge calls for one K-idea set concurrently, then combine into R.

    R = SCHEMA_BONUS + sum_i  I(u_i) * min_{j<i} d(u_i,u_j)^beta * g_i - length_penalty.
    ``answer_text`` is the raw <answer> JSON block, measured for the length term.
    ``documents`` are the source documents for the grounding gate (g_i == 1.0 for
    every idea when F9B_GROUNDING=0, making the reward identical to pre-gate v6)."""
    sem_idea = _get_sem("idea", MAX_CONCURRENT_IDEA)      # distance -> gpt-oss (per worker)
    sem_impact = _get_sem("impact", MAX_CONCURRENT_IMPACT)  # impact -> SciJudge (per worker)
    n = len(ideas)

    imp_tasks = [asyncio.create_task(_impact(u, gold, sem_impact)) for u in ideas]
    # For each idea, the earlier ideas it is compared against for mindist. With
    # DISTANCE_SAMPLE_K>0 we sample <=K earlier ideas (deterministically, seeded by
    # the set text so an identical rollout scores identically) instead of all i.
    _seed = int(hashlib.md5("\x1e".join(u.card_text() for u in ideas).encode()).hexdigest()[:8], 16)
    _rng = random.Random(_seed)
    earlier_of: Dict[int, List[int]] = {}
    for i in range(1, n):
        prev = list(range(i))
        if 0 < DISTANCE_SAMPLE_K < len(prev):
            prev = sorted(_rng.sample(prev, DISTANCE_SAMPLE_K))
        earlier_of[i] = prev
    pairs = sorted({(j, i) for i in range(1, n) for j in earlier_of[i]})
    dist_tasks = {p: asyncio.create_task(_distance(ideas[p[0]], ideas[p[1]], sem_idea))
                  for p in pairs}
    # Grounding: one call per idea, fired CONCURRENTLY with impact/distance and
    # sharing sem_idea (same judge endpoint as distance). Per-idea, not per-set,
    # because the drift concentrates in the tail of the set.
    docs = list(documents or [])
    gnd_tasks = ([asyncio.create_task(_grounding(u, docs, sem_idea)) for u in ideas]
                 if GROUNDING_ON else [])

    # If the per-set budget (asyncio.wait_for) cancels us mid-flight, CancelledError
    # lands at whichever await is live; any tasks not yet awaited would otherwise be
    # orphaned and keep holding a semaphore slot. Cancel them all in a finally so a
    # budget trip fully releases this set's judge work before we impute.
    all_tasks = list(imp_tasks) + list(dist_tasks.values()) + list(gnd_tasks)
    try:
        impacts_raw = await asyncio.gather(*imp_tasks)
        dist_raw = {p: await t for p, t in dist_tasks.items()}
        gnd_raw = await asyncio.gather(*gnd_tasks) if gnd_tasks else []
    except (asyncio.CancelledError, asyncio.TimeoutError):
        for t in all_tasks:
            if not t.done():
                t.cancel()
        raise
    dist = {p: v for p, (v, _f) in dist_raw.items()}
    dist_fail = sum(1 for _v, f in dist_raw.values() if f)  # gpt-oss (distance) failures

    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0

    # FINDING 7 (revised): a failed judge call arrives as None -- MISSING DATA, not
    # evidence the idea lost to gold. Failures CLUSTER per-set (judge outage/socket
    # wedge): in run 26045 ~4.6% of sets were TOTAL blackouts and another ~7.9% had
    # only a MINORITY of calls survive, so a naive set-mean is driven by 1-2 noisy
    # survivors. Impute in three tiers so an outage becomes a GRPO NO-OP (imputed
    # ~= group mean -> advantage ~= 0) instead of a false penalty or a gradient
    # dictated by a lone survivor:
    #   successful >= MIN_FRAC of set -> set-mean of successful calls (trustworthy)
    #   0 < successful <  MIN_FRAC    -> trailing-WINDOW mean (set sample too thin)
    #   0 successful (total blackout) -> trailing-WINDOW mean (was fixed default)
    #   window empty (cold start)     -> default (IMPACT_DEFAULT)
    def _impute(vals: List[Optional[float]], default: float):
        ok = [v for v in vals if v is not None]
        n_fail = len(vals) - len(ok)
        win_mean = (sum(_IMP_WIN) / len(_IMP_WIN)) if _IMP_WIN else default
        # trust the set-mean only when successful calls are a real majority of the
        # set; else fall back to recent real impact, never a lone survivor/constant.
        if ok and len(ok) >= math.ceil(IMPACT_IMPUTE_MIN_FRAC * len(vals)):
            fill = mean(ok)
        else:
            fill = win_mean
        return [v if v is not None else fill for v in vals], n_fail

    impacts, imp_fail = _impute(impacts_raw, IMPACT_DEFAULT)

    # GROUNDING: same three-tier imputation as impact, for the same reason -- a
    # Bedrock outage is MISSING DATA and must be a GRPO no-op, never a grounding
    # penalty. Note the asymmetry with features_to_g: an idea that declared an
    # INVALID grounding_type gets `floor` (real signal, judged); an idea whose
    # JUDGE CALL failed gets the imputed fill (no signal).
    gnd_feats: List[Optional[Dict[str, Any]]] = [None] * n
    gs: List[float] = [1.0] * n
    gnd_fail = 0.0
    if GROUNDING_ON:
        g_raw: List[Optional[float]] = []
        for i in range(n):
            feats, failed = gnd_raw[i] if i < len(gnd_raw) else (None, True)
            # `not docs` also lands here (_grounding short-circuits to (None, False)):
            # judging against an empty document block would score every idea at floor,
            # so it is imputed like a failure but NOT counted as one -- grounding_fail
            # must mean "the judge failed", so it stays a clean Bedrock-health metric.
            if feats is None:
                g_raw.append(None)
                gnd_fail += 1.0 if failed else 0.0
                continue
            gnd_feats[i] = feats
            g_raw.append(grounding.features_to_g(feats, ideas[i].grounding_type,
                                                 lo=GROUNDING_LO, hi=GROUNDING_HI,
                                                 floor=GROUNDING_FLOOR,
                                                 require_type=GROUNDING_REQUIRE_TYPE))
        ok_g = [v for v in g_raw if v is not None]
        win_g = _g_window_mean()
        if ok_g and len(ok_g) >= math.ceil(GROUNDING_IMPUTE_MIN_FRAC * len(g_raw)):
            fill_g = mean(ok_g)
        else:
            fill_g = win_g
        gs = [v if v is not None else fill_g for v in g_raw]
        _G_WIN.extend(ok_g)  # window tracks REAL g only, never imputed fills

    # quality == impact (LINEAR; no novelty factor). Optional IMPACT_FLOOR (off by
    # default) only lifts the term used in the reward, not the logged impact_mean.
    qualities, mindists = [], []
    for i in range(n):
        q = max(IMPACT_FLOOR, impacts[i])
        md = 1.0 if i == 0 else min(dist[(j, i)] for j in earlier_of[i])
        qualities.append(q)
        mindists.append(md)

    # CURRICULUM: read beta from the trailing impact window BEFORE recording THIS
    # sample (a set never gates on its own impact), apply it, then update the window.
    impact_mean = mean(impacts)
    beta = _curriculum_beta()
    # GROUNDING GATE: g_i multiplies the whole contribution, so an ungrounded idea
    # cannot collect diversity credit at all (an ADDITIVE grounding term is a tax the
    # policy routes around -- the documented v4 failure). gs is all 1.0 when
    # F9B_GROUNDING=0, making this line identical to pre-gate v6.
    contribs = [qualities[i] * (_capped_mindist(mindists[i]) ** beta) * gs[i]
                for i in range(n)]
    _update_quality_window(impact_mean)
    body = sum(contribs)
    # LENGTH REGULARIZATION: subtract a one-sided verbosity penalty on the whole
    # <answer> block (0 by default / when within target). Judge-independent, so it
    # applies on every scored set; see _length_penalty. THIS is the only reward
    # difference from v0.
    answer_len, len_penalty = _length_penalty(_card_concat(ideas))  # v6: card-only
    score = SCHEMA_BONUS + body - len_penalty
    return {
        "score": score,
        "format": 1.0,
        "format_valid": 1.0,  # logged-only: 1 iff set passed the format gate
        "n_ideas": float(n),
        "novelty_mean": 0.0,  # logged-only constant (novelty removed) -- keeps shared W&B panels alive
        "impact_mean": impact_mean,
        "quality_mean": mean(qualities),
        "mindist_mean": mean(mindists[1:]),  # exclude the trivial first==1
        "answer_len": answer_len,            # logged-only: <answer> block token count driving the penalty
        "len_penalty": len_penalty,          # logged-only: length penalty subtracted from score
        "curriculum_beta": beta,             # logged-only: current diversity exponent
        "impact_win": (sum(_IMP_WIN)/len(_IMP_WIN)) if _IMP_WIN else 0.0,  # logged-only: trailing-window mean impact driving beta
        # logged-only (FINDING 7): per-set impact judge-failure count -> verl averages
        # into reward/impact_fail/mean so the timeout rate is visible per-step in W&B.
        "impact_fail": float(imp_fail),
        # logged-only: per-set gpt-oss distance judge-failure count (was SILENT;
        # each failure fell back to DISTANCE_DEFAULT=1.0 -> spuriously inflated diversity).
        "distance_fail": float(dist_fail),
        # logged-only: grounding gate. grounding_g_mean is the priced multiplier;
        # the four *_mean features are UNPRICED tripwires (the judge returns all four
        # in one call, so logging them is free). Once load_bearing is priced the policy
        # has an incentive to ASSERT dependence without it being true -- if
        # support_faithful_mean starts falling, fold it into g_i.
        **_grounding_metrics(gs, gnd_feats, gnd_fail),
        "_impacts": impacts, "_qualities": qualities, "_mindists": mindists,
        "_gs": gs,
        # per-idea raw load_bearing (None == judge failed -> g was imputed), for the
        # sample dumps only; stripped before verl sees the dict.
        "_lbs": [(f.get("load_bearing") if isinstance(f, dict) else None) for f in gnd_feats],
    }


# --------------------------------------------------------------------------- #
# async-in-sync bridge: verl calls compute_score synchronously, one rollout each
# --------------------------------------------------------------------------- #
# Reuse ONE event loop for the whole reward-worker thread via verl's own helper
# (verl.utils.ray_utils.get_event_loop). The old asyncio.run(coro)-per-rollout
# CREATES AND CLOSES a fresh loop each call; the @lru_cache'd judge clients' httpx
# pools bind to the FIRST loop and are orphaned the moment it closes, so every later
# rollout re-dials and leaks sockets -- the leak that tanked run 25968 (fail rate
# 0->1.48 over 13h, 194 TIME-WAIT to the judge ports while the judge sat idle).
# get_event_loop fetches-or-creates the thread's loop and NEVER closes it, so the
# cached pools stay valid. See f9b_v1.py + test_f9b_v1_loop_reuse.py.
try:  # verl is always importable where the reward runs; guard only for offline tests
    from verl.utils.ray_utils import get_event_loop as _get_event_loop
except Exception:  # pragma: no cover - fallback mirrors verl's get_event_loop exactly
    def _get_event_loop():
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop


def _run_async(coro):
    """Run an async coroutine from verl's sync reward worker on the thread's
    persistent event loop (fetched once, never closed), so the cached judge clients'
    connection pools survive across all rollouts (no per-call asyncio.run teardown)."""
    return _get_event_loop().run_until_complete(coro)


def _maybe_log(extra_info, ideas, gold, result: Dict[str, Any]) -> None:
    global _completed
    with _log_lock:
        idx = _completed
        _completed += 1
    if idx % _LOG_EVERY != 0:
        return
    q, im, md = result["_qualities"], result["_impacts"], result["_mindists"]
    lines = ["=" * 88,
             f"[f9b-impdist-len set] score={result['score']:.3f} | n={len(ideas)} | "
             f"impact={result['impact_mean']:.3f} quality={result['quality_mean']:.3f} "
             f"mindist={result['mindist_mean']:.3f} | "
             f"answer_len={result.get('answer_len', 0.0):.0f} len_pen={result.get('len_penalty', 0.0):.3f} | "
             f"judge_fail imp={int(result.get('impact_fail', 0))} dist={int(result.get('distance_fail', 0))}"
             + (f" gnd={int(result.get('grounding_fail', 0))} | "
                f"g={result.get('grounding_g_mean', 1.0):.3f} lb={result.get('load_bearing_mean', 0.0):.2f}"
                if GROUNDING_ON else "")]
    prompt = str((extra_info or {}).get("query", "") or "")
    if prompt:
        lines += ["[prompt]", prompt[:_LOG_PROMPT_CHARS]]
    if gold:
        lines += ["[gold]", gold[:_LOG_PROMPT_CHARS]]
    gs_, lbs_ = result.get("_gs") or [], result.get("_lbs") or []
    for i, u in enumerate(ideas):
        gtag = ""
        if GROUNDING_ON and i < len(gs_):
            lb = lbs_[i] if i < len(lbs_) else None
            gtag = f" lb={'--' if lb is None else lb} g={gs_[i]:.3f}"
        lines.append(f"--- idea {i} | f={q[i]:.3f} (I={im[i]:.3f}) mindist={md[i]:.3f}{gtag} ---")
        lines.append(u.insight_claim)
    lines.append("=" * 88)
    print("\n".join(lines), flush=True)


async def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                        **kwargs) -> Dict[str, Any]:
    """F9-B (impact x distance + length penalty) reward for one rollout (one K-idea
    set). Returns a dict so verl logs the components; "score" is what GRPO optimizes.

    NATIVE ASYNC: verl's experimental reward manager detects the coroutine
    (inspect.iscoroutinefunction) and awaits this on its single long-lived event
    loop, so the cached judge httpx pools stay bound to one loop. Do NOT wrap the
    judge calls in _run_async here -- that reintroduced the cross-thread-loop
    pool-binding bug that wedged run 26070 (357k APITimeoutError, idle judges)."""
    del data_source, kwargs
    try:
        ideas = parse_ideas(solution_str or "", k=K)
        gate = _format_value(solution_str or "", ideas)
        if gate is not None:  # malformed -> terminal reward, NO judge calls
            # keys aligned with the valid branch so verl's reward_extra_info arrays
            # stay rectangular (ragged arrays break W&B logging).
            return {"score": gate, "format": gate, "format_valid": 0.0,
                    "n_ideas": float(len(ideas)),
                    "novelty_mean": 0.0, "impact_mean": 0.0,
                    "quality_mean": 0.0, "mindist_mean": 0.0,
                    "answer_len": 0.0, "len_penalty": 0.0,
                    "curriculum_beta": _curriculum_beta(),
                    "impact_win": (sum(_IMP_WIN)/len(_IMP_WIN)) if _IMP_WIN else 0.0,
                    "impact_fail": 0.0, "distance_fail": 0.0,  # no judge calls fired
                    # neutral (no gate applied), not floor -- nothing was judged
                    **_grounding_metrics([], [], 0.0)}

        gold = (ground_truth or "").strip()
        answer_text = extract_answer_str(solution_str or "") or ""  # raw <answer> block for the length penalty
        # The grounding judge needs the raw source documents; extra_info["query"] is
        # the fully-rendered prompt and already carries them inline, so no dataset
        # change is needed. Empty -> _grounding short-circuits and g is imputed.
        documents = _documents_of(extra_info)
        result = await _score_set(ideas, gold, answer_text, documents=documents)
        _maybe_log(extra_info, ideas, gold, result)
        # strip the private _-prefixed arrays before handing back to verl
        return {k: v for k, v in result.items() if not k.startswith("_")}
    except Exception as exc:
        _write_debug("f9b_error", f"{type(exc).__name__}: {exc}")
        return {"score": 0.0, "format": 0.0, "format_valid": 0.0, "n_ideas": 0.0,
                "novelty_mean": 0.0, "impact_mean": 0.0,
                "quality_mean": 0.0, "mindist_mean": 0.0,
                "answer_len": 0.0, "len_penalty": 0.0,
                "curriculum_beta": _curriculum_beta(),
                "impact_win": (sum(_IMP_WIN)/len(_IMP_WIN)) if _IMP_WIN else 0.0,
                "impact_fail": 0.0, "distance_fail": 0.0,
                **_grounding_metrics([], [], 0.0)}
