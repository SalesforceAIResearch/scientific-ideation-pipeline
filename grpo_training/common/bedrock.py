"""AWS Bedrock as an OpenAI-compatible backend (SigV4-signed), for gpt-oss-120b.

WHY THIS EXISTS: the F9-B distance judge (gpt-oss-120b) has been self-hosted on
its own vLLM node (serve_judge.sbatch, 8 GPUs). Bedrock serves the SAME weights
on demand -- `openai.gpt-oss-120b-1:0` -- which frees that whole node and removes
the judge-server failure mode that tanked run 26070 (wedged connection pool ->
timeout collapse). SciJudge is NOT on Bedrock (no OpenMOSS provider), so the
impact judge stays self-hosted; see train/launch_grpo_v6_onenode_h200.sbatch for
the single-node layout that co-locates it with training.

HOW: Bedrock exposes an OpenAI-compatible route at
``https://bedrock-runtime.<region>.amazonaws.com/openai/v1``, so we do NOT need
boto3's `converse` API or a bespoke client -- the normal `AsyncOpenAI` SDK works,
and every existing judge call site is unchanged. The one thing the SDK cannot do
is AWS SigV4 request signing, so `SigV4Auth` below is an ``httpx.Auth`` that
signs each request as it goes out. Credentials come from the standard boto3 chain
(on this cluster: the SageMaker execution role), so there is no API key to manage.

Signing is per-request and MUST re-read credentials each time: the instance role
hands out temporary credentials that rotate, so caching a frozen set would start
401ing mid-run. botocore caches internally, so this is cheap.

TWO RESPONSE-SHAPE DIFFERENCES from self-hosted vLLM gpt-oss, both verified:

  1. Reasoning is INLINE in `content` as ``<reasoning>...</reasoning>`` followed
     by the answer. vLLM instead exposes a separate ``reasoning_content`` field
     (which Bedrock does NOT populate). The rubric parsers survive this untouched
     because common.novelty.extract_json takes the LAST balanced {...} that
     parses, and the reasoning always precedes the answer. Do not "fix" this by
     stripping the tag -- nothing downstream needs it.
  2. Reasoning tokens are billed against `max_tokens`, so a budget that is fine
     on vLLM can truncate the answer here (a 64-token cap returned only "AL").
     Keep judge calls at >=1000 max_tokens, or pass reasoning_effort="low".

`reasoning_effort` is accepted as a FLAT top-level field
(``{"reasoning_effort": "low"}``); the nested ``{"reasoning": {"effort": ...}}``
form is silently ignored -- no error, no effect.

Measured 2026-07-26, us-west-2: single call ~0.8-1.0s; 40 concurrent through one
client 3.4s with 0 failures; 64 concurrent 24 req/s, 0 throttles. That covers the
reward's offered depth (~80 distance calls across 8 workers), but these were
short bursts -- sustained multi-hour TPM quotas are NOT characterized, so keep
the reward's existing timeout/imputation path intact.

NOTHING IMPORTS THIS BY DEFAULT. The existing reward and common/clients.py are
untouched -- no bedrock branch, no new backend name. You opt in by selecting a
different REWARD FILE: rewards/f9b_impdist_len_v6_bedrock.py is a thin shim that
imports v6 and rebinds only its distance client to the client built here. Verify
reachability first with extras/check_bedrock.py.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx

# Region for the Bedrock endpoint. gpt-oss-120b is ACTIVE in both us-west-2 and
# us-east-1; AWS_REGION is already exported on this cluster (us-west-2).
DEFAULT_REGION = (os.environ.get("BEDROCK_REGION", "").strip()
                  or os.environ.get("AWS_REGION", "").strip()
                  or "us-west-2")

# Bedrock's model id for gpt-oss-120b. NOT the same string as the self-hosted
# judge's "openai/gpt-oss-120b" -- Bedrock uses dots and a version suffix.
GPT_OSS_120B = "openai.gpt-oss-120b-1:0"
GPT_OSS_20B = "openai.gpt-oss-20b-1:0"

# The service name SigV4 signs for is "bedrock" even on the bedrock-runtime host.
_SIGV4_SERVICE = "bedrock"


def endpoint_url(region: Optional[str] = None) -> str:
    """OpenAI-compatible base URL for Bedrock in `region`."""
    region = region or DEFAULT_REGION
    return f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"


def is_bedrock_model(model: Optional[str]) -> bool:
    """True for a Bedrock model id (provider-prefixed, e.g. ``openai.gpt-oss-120b-1:0``).

    Deliberately checks for the DOTTED provider prefix so it cannot collide with
    the self-hosted judge name "openai/gpt-oss-120b" (slash), which must keep
    routing to vLLM."""
    name = (model or "").strip().lower()
    return name.startswith(("openai.", "qwen.", "deepseek.", "meta.", "mistral.",
                            "anthropic.", "amazon.", "google.", "nvidia.",
                            "moonshot.", "moonshotai.", "zai.", "minimax.",
                            "writer.", "cohere.", "ai21."))


class SigV4Auth(httpx.Auth):
    """Sign outgoing httpx requests with AWS SigV4 so AsyncOpenAI can talk to Bedrock.

    `requires_request_body = True` is essential: SigV4 hashes the payload, so the
    body must be materialized before the signature is computed."""

    requires_request_body = True

    def __init__(self, region: Optional[str] = None, service: str = _SIGV4_SERVICE):
        import boto3  # local import: only Bedrock users need boto3 installed

        self.region = region or DEFAULT_REGION
        self.service = service
        self._session = boto3.session.Session()

    def auth_flow(self, request: httpx.Request):
        from botocore.auth import SigV4Auth as _BotoSigV4
        from botocore.awsrequest import AWSRequest

        creds = self._session.get_credentials()
        if creds is None:
            raise RuntimeError(
                "No AWS credentials for Bedrock. On this cluster they come from the "
                "SageMaker execution role via the boto3 chain -- note that a stale "
                "AWS_PROFILE pointing at a missing profile breaks the CLI (and can "
                "break boto3); unset it if so."
            )
        # Re-freeze per request: instance-role credentials rotate mid-run.
        frozen = creds.get_frozen_credentials()
        aws_req = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content or b"",
            headers={"Content-Type": "application/json"},
        )
        _BotoSigV4(frozen, self.service, self.region).add_auth(aws_req)
        for key, value in aws_req.headers.items():
            request.headers[key] = value
        yield request


def build_async_client(
    *,
    region: Optional[str] = None,
    timeout: Optional[float] = None,
    max_connections: Optional[int] = None,
    keepalive_expiry: Optional[float] = None,
    max_retries: int = 4,
):
    """An AsyncOpenAI client pointed at Bedrock, SigV4-signed.

    Mirrors the hardened-transport choices the rewards use for vLLM (structured
    timeout with a pool bound, short keepalive so a half-open socket is re-dialed
    rather than awaited) because the failure mode being defended against -- a
    wedged pool silently stalling the reward -- is transport-level, not
    server-specific.

    `max_retries` is the SDK's own retry of 429/5xx. Bedrock throttling is a 429,
    and unlike a self-hosted server (where retries just deepen an already-slow
    queue) a Bedrock 429 IS worth backing off on, so the default is higher than
    the reward's F9B_JUDGE_MAX_RETRIES=1.
    """
    from openai import AsyncOpenAI

    t = timeout if timeout is not None else 600.0
    limits_kwargs = {}
    if max_connections is not None:
        limits_kwargs["max_connections"] = max_connections
        limits_kwargs["max_keepalive_connections"] = max_connections
    if keepalive_expiry is not None:
        limits_kwargs["keepalive_expiry"] = keepalive_expiry

    http_client = httpx.AsyncClient(
        auth=SigV4Auth(region),
        timeout=httpx.Timeout(t, connect=min(10.0, t), write=min(30.0, t),
                              pool=min(10.0, t)),
        limits=httpx.Limits(**limits_kwargs) if limits_kwargs else httpx.Limits(),
    )
    # api_key is unused (SigV4 supplies auth) but the SDK requires a non-empty value.
    return AsyncOpenAI(base_url=endpoint_url(region), api_key="EMPTY",
                       http_client=http_client, max_retries=max_retries)


def list_text_models(region: Optional[str] = None, active_only: bool = True) -> list:
    """Model ids on Bedrock in `region` that emit TEXT. Utility for checking what
    is available (see aws_avail_models.txt for a captured snapshot)."""
    import boto3

    client = boto3.client("bedrock", region_name=region or DEFAULT_REGION)
    out = []
    for m in client.list_foundation_models()["modelSummaries"]:
        if "TEXT" not in m.get("outputModalities", []):
            continue
        if active_only and m.get("modelLifecycle", {}).get("status") != "ACTIVE":
            continue
        out.append(m["modelId"])
    return sorted(set(out))
