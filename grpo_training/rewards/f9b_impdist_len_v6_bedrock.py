"""F9-B v6 with the DISTANCE judge on AWS Bedrock instead of a self-hosted vLLM node.

THE REWARD IS NOT REDEFINED HERE. This module imports f9b_impdist_len_v6 and
re-exports its `compute_score` unchanged; the ONLY difference is WHERE gpt-oss-120b
is called. Scores from this file and from v6 are directly comparable because they
are computed by the same code -- not by a copy that could drift.

WHY: the distance judge has needed its own 8-GPU allocation (serve_judge.sbatch).
Bedrock serves the same open weights on demand as `openai.gpt-oss-120b-1:0`, which
frees that node and removes the judge-server pool-wedge failure mode (run 26070:
357k APITimeoutError against idle judges). SciJudge-30B is NOT on Bedrock (no
OpenMOSS provider), so IMPACT is untouched and still hits the self-hosted endpoint
-- see train/launch_grpo_v6_onenode_h200.sbatch, which co-locates it with training
so the whole run fits one node.

HOW: v6 resolves its distance client through the module-global `_idea_client()` and
its model through the module-global `IDEA_JUDGE_MODEL` (rewards/f9b_impdist_len_v6.py
line ~547 reads both). We rebind exactly those two globals on the imported module.
Rebinding rather than editing v6 keeps v6 byte-identical, and rather than copying
v6 keeps the reward math single-sourced.

WHAT STAYS THE SAME: the rubric, the parser, the 0.1/0.3/0.6 field weights, the
K-gate, the curriculum, and the CARD-ONLY length penalty all come from v6. The
distance prompt is unchanged, and common.novelty.extract_json already tolerates
Bedrock's response shape (see below), so nothing downstream needed adapting --
verified end-to-end: a near-duplicate pair scored 0.0 and an unrelated pair 0.975.

TWO BEDROCK RESPONSE-SHAPE FACTS, both measured, neither requiring a code change:
  * Reasoning is INLINE in `content` as <reasoning>...</reasoning> before the JSON
    (vLLM instead populates a separate `reasoning_content`, which Bedrock leaves
    empty). extract_json takes the LAST balanced {...} that parses, so the answer
    wins. Do NOT add tag-stripping -- it would be dead code.
  * Reasoning tokens bill against max_tokens. F9B_DISTANCE_MAX_TOKENS=1024 (v6's
    default) is comfortable at a measured ~225-330 tokens for reasoning+JSON, but
    do not cut it for this backend.

Throughput measured 2026-07-26 (us-west-2): ~0.8-1.0s single call; 40 concurrent
through one client in 3.4s with 0 failures; 64 concurrent at 24 req/s, 0 throttles
-- covering the reward's offered depth (~80 distance calls across 8 workers). These
were short bursts; sustained multi-hour TPM quotas are NOT characterized, so v6's
timeout / DISTANCE_DEFAULT fallback path is deliberately left intact.

USE IT:
    REWARD_FN=.../rewards/f9b_impdist_len_v6_bedrock.py
Everything else is v6's env surface (F9B_LEN_COEF, F9B_LEN_TARGET, ...). The
Bedrock-specific knobs are:
    F9B_BEDROCK_MODEL        default openai.gpt-oss-120b-1:0
    F9B_BEDROCK_REGION       default $BEDROCK_REGION / $AWS_REGION / us-west-2
    F9B_BEDROCK_MAX_RETRIES  default 4 (SDK-level 429/5xx backoff)
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

_GRPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_GRPO_ROOT.parent), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import bedrock  # noqa: E402
from rewards import f9b_impdist_len_v6 as _v6  # noqa: E402

# Bedrock's id for gpt-oss-120b. NOTE the dots: v6's default is the SELF-HOSTED
# name "openai/gpt-oss-120b" (slash), which is a different string and would 400 here.
BEDROCK_JUDGE_MODEL = os.environ.get("F9B_BEDROCK_MODEL", bedrock.GPT_OSS_120B).strip()
BEDROCK_REGION = os.environ.get("F9B_BEDROCK_REGION", "").strip() or None
# Higher than v6's local _JUDGE_MAX_RETRIES=1: against a self-hosted server a retry
# just deepens an already-slow queue, but a Bedrock 429 is genuine throttling that
# backs off and succeeds.
BEDROCK_MAX_RETRIES = int(os.environ.get("F9B_BEDROCK_MAX_RETRIES", "4"))


@lru_cache(maxsize=1)
def _bedrock_idea_client():
    """The distance-judge client, SigV4-signed at Bedrock.

    lru_cache mirrors v6's own client caching: ONE client per worker process, so
    its httpx connection pool binds to that worker's single long-lived event loop.
    Building a client per call would defeat pooling and re-introduce the
    cross-loop binding bug that wedged run 26070.

    Pool/timeout sizing is taken from v6's resolved constants so this backend
    inherits the same hardening (pool-acquire bound + short keepalive so a
    half-open socket is re-dialed instead of awaited)."""
    return bedrock.build_async_client(
        region=BEDROCK_REGION,
        timeout=_v6.REQUEST_TIMEOUT,
        max_connections=_v6.MAX_CONCURRENT_IDEA + 8,
        keepalive_expiry=_v6.KEEPALIVE_EXPIRY,
        max_retries=BEDROCK_MAX_RETRIES,
    )


# --- the actual swap: rebind the two globals v6's _distance() reads ---------- #
# _distance does `_judge_chat(_idea_client(), IDEA_JUDGE_MODEL, ...)`, resolving
# both from v6's module dict at call time, so rebinding them here redirects every
# distance call. The impact path (_impact_client / IMPACT_MODEL) is deliberately
# NOT touched: SciJudge stays self-hosted.
_v6._idea_client = _bedrock_idea_client
_v6.IDEA_JUDGE_MODEL = BEDROCK_JUDGE_MODEL

# v6's compute_score, unmodified. verl's reward manager detects the coroutine and
# awaits it on its own loop; re-exporting the function object (not a wrapper) keeps
# inspect.iscoroutinefunction true and adds no frame.
compute_score = _v6.compute_score

# Re-export the knobs a launcher or test may want to assert on.
K = _v6.K
LEN_COEF = _v6.LEN_COEF
LEN_TARGET = _v6.LEN_TARGET
IMPACT_MODEL = _v6.IMPACT_MODEL

__all__ = ["compute_score", "BEDROCK_JUDGE_MODEL", "IMPACT_MODEL",
           "K", "LEN_COEF", "LEN_TARGET"]
