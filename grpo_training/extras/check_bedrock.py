#!/usr/bin/env python3
"""Preflight: can this node actually reach gpt-oss-120b on Bedrock, right now?

Run this BEFORE burning an allocation. The one-node launcher calls it because
`val_before_train=True` fires distance judge calls within the first minutes, so a
credential or region problem would otherwise surface ~20 min in as a wall of judge
errors -- and the reward IMPUTES a default distance on failure, which means a
broken judge does not crash the run, it silently trains on a degenerate reward.
That is the failure this script exists to make loud and early.

It checks the three things that actually break, in the order they break:
  1. CREDENTIALS resolve through the boto3 chain. On this cluster they come from
     the SageMaker execution role -- there is no key to configure. The known
     gotcha is a stale AWS_PROFILE naming a profile with no ~/.aws/config: the
     resolver then fails even though the role is right there. Pass --unset-profile
     (or unset it in the shell) to fall through to the role.
  2. The model is ACTIVE and ON_DEMAND in this region. gpt-oss-120b is in both
     us-west-2 and us-east-1, but "the model exists" and "you may invoke it
     on demand" are different facts, and access can be revoked per-account.
  3. A REAL round-trip through the exact transport the reward uses -- AsyncOpenAI
     over the OpenAI-compatible route with SigV4 signing, via common/bedrock.py.
     Steps 1-2 are boto3 control-plane calls and would still pass if the signing
     or the /openai/v1 route were broken, so this step is the one that matters.

Exit code is 0 only if all requested checks pass, so it drops straight into
`... || exit 1` in an sbatch preflight.

    python3 extras/check_bedrock.py                        # full check
    python3 extras/check_bedrock.py --model openai.gpt-oss-120b-1:0
    python3 extras/check_bedrock.py --skip-invoke          # control plane only
    python3 extras/check_bedrock.py --concurrency 16       # also load-probe
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

_GRPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_GRPO_ROOT.parent), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _say(ok: bool, msg: str) -> bool:
    print(f"  [{'OK ' if ok else 'FAIL'}] {msg}", flush=True)
    return ok


def check_credentials(region: str) -> bool:
    """Credentials resolve, and report WHERE from (the method name is the useful
    part when debugging: 'iam-role'/'container-role' = the execution role,
    'shared-credentials-file'/'sso' = something local you may not have meant)."""
    try:
        import boto3
    except ImportError:
        return _say(False, "boto3 not installed in this env (pip install boto3)")
    try:
        creds = boto3.session.Session().get_credentials()
    except Exception as exc:  # noqa: BLE001 -- profile errors raise many types
        return _say(False, f"credential resolution raised {type(exc).__name__}: {exc}")
    if creds is None:
        _say(False, "no AWS credentials found via the boto3 chain")
        if os.environ.get("AWS_PROFILE"):
            print(f"         AWS_PROFILE={os.environ['AWS_PROFILE']} is set; if "
                  f"~/.aws/config does not define it, unset it (--unset-profile) "
                  f"so the instance role is used.")
        return False
    _say(True, f"credentials via {getattr(creds, 'method', '?')} (region {region})")
    try:
        ident = boto3.client("sts", region_name=region).get_caller_identity()
        print(f"         account={ident.get('Account')} arn={ident.get('Arn')}")
    except Exception as exc:  # noqa: BLE001 -- sts may be denied; not fatal
        print(f"         (sts:GetCallerIdentity unavailable: {type(exc).__name__})")
    return True


def check_model_available(model: str, region: str) -> bool:
    """Model is ACTIVE with ON_DEMAND inference in this region.

    ON_DEMAND is the one to assert: a model listed only as PROVISIONED or
    INFERENCE_PROFILE cannot be invoked by id the way the reward invokes it."""
    import boto3

    try:
        summaries = boto3.client("bedrock", region_name=region) \
            .list_foundation_models()["modelSummaries"]
    except Exception as exc:  # noqa: BLE001 -- perms/region/network all land here
        return _say(False, f"bedrock:ListFoundationModels failed: "
                           f"{type(exc).__name__}: {exc}")

    hit = next((m for m in summaries if m.get("modelId") == model), None)
    if hit is None:
        _say(False, f"{model} not offered in {region}")
        # Near-misses are almost always a version-suffix typo, so show them.
        stem = model.split(":")[0].rsplit("-", 1)[0]
        near = [m["modelId"] for m in summaries if m.get("modelId", "").startswith(stem)]
        if near:
            print(f"         did you mean: {', '.join(sorted(near)[:5])}")
        return False

    status = hit.get("modelLifecycle", {}).get("status")
    types = hit.get("inferenceTypesSupported", [])
    ok = status == "ACTIVE" and "ON_DEMAND" in types
    return _say(ok, f"{model}: status={status} inference={','.join(types) or 'none'}")


async def _invoke(model: str, region: str, max_tokens: int, concurrency: int) -> bool:
    """Round-trip through the reward's real transport: common/bedrock.py's
    AsyncOpenAI + SigV4 client. One shared client for all requests, exactly as the
    reward does -- a per-call client would hide pool problems."""
    from common import bedrock

    client = bedrock.build_async_client(region=region, timeout=60.0,
                                       max_connections=max(concurrency, 8),
                                       keepalive_expiry=5.0, max_retries=2)

    async def one(i: int):
        # max_tokens must be generous: Bedrock bills REASONING tokens against it,
        # so a tight cap truncates before any answer text appears (a 64-token cap
        # returned literally "AL"). This is why the check is not `== "OK"`.
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Reply with the word OK. (#{i})"}],
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "")

    t0 = time.time()
    try:
        if concurrency <= 1:
            texts = [await one(0)]
        else:
            texts = await asyncio.gather(*(one(i) for i in range(concurrency)))
    except Exception as exc:  # noqa: BLE001 -- auth/route/throttle all land here
        _say(False, f"invoke failed: {type(exc).__name__}: {exc}")
        name = type(exc).__name__
        if "AuthenticationError" in name or "PermissionDenied" in name:
            print("         SigV4 signed but was rejected: check the role's "
                  "bedrock:InvokeModel permission for this model id.")
        elif "NotFound" in name:
            print("         route/model id problem, not credentials.")
        return False
    finally:
        await client.close()

    dt = time.time() - t0
    nonempty = sum(1 for t in texts if t.strip())
    ok = nonempty == len(texts)
    _say(ok, f"{nonempty}/{len(texts)} non-empty replies in {dt:.2f}s "
             f"({len(texts) / dt:.1f} req/s)")
    print(f"         sample: {texts[0].strip()[:120]!r}")
    return ok


def main() -> int:
    from common import bedrock

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("F9B_BEDROCK_MODEL",
                                                      bedrock.GPT_OSS_120B),
                    help="Bedrock model id (dotted form). Default: %(default)s")
    ap.add_argument("--region", default=None,
                    help="default: $BEDROCK_REGION / $AWS_REGION / us-west-2")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="must cover reasoning+answer; >=400 (default: %(default)s)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel probes; >1 also load-tests the shared pool")
    ap.add_argument("--skip-invoke", action="store_true",
                    help="control-plane checks only (no tokens spent)")
    ap.add_argument("--unset-profile", action="store_true",
                    help="drop AWS_PROFILE first so the instance role is used")
    args = ap.parse_args()

    if args.unset_profile and os.environ.pop("AWS_PROFILE", None):
        print("[bedrock] AWS_PROFILE unset for this check")

    region = args.region or bedrock.DEFAULT_REGION
    print(f"Bedrock preflight: model={args.model} region={region}")

    # Short-circuit: every later check needs credentials, and a creds failure makes
    # the rest report misleading errors.
    if not check_credentials(region):
        return 1
    ok = check_model_available(args.model, region)
    if not args.skip_invoke:
        ok = asyncio.run(_invoke(args.model, region, max(args.max_tokens, 400),
                                 args.concurrency)) and ok

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
