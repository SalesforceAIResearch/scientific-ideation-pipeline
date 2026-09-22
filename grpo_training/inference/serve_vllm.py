"""Launch vLLM's OpenAI server with a small compatibility shim.

Why this exists
---------------
The `verl_grpo` env pairs a very new Starlette (1.x) with
`prometheus-fastapi-instrumentator==8.0.0` (the latest). Starlette 1.x introduced
an `_IncludedRouter` route type that the instrumentator's route-name resolver
doesn't know about: it unconditionally reads `route.path`, which `_IncludedRouter`
lacks, so the metrics middleware raises

    AttributeError: '_IncludedRouter' object has no attribute 'path'

on EVERY request (including /health and /v1/chat/completions), making the served
model unusable. There is no newer instrumentator to upgrade to, and downgrading
Starlette would disturb the proven FastAPI/vLLM/OLMo-3 stack.

This launcher monkeypatches the instrumentator's `_get_route_name` to be
route-type tolerant (read `.path` via getattr, recurse into any route that has
sub-`routes`), then delegates to the normal `vllm` CLI. Use it exactly like the
`vllm` command, e.g.:

    python serve_vllm.py serve <model> --served-model-name foo --port 8003 ...
"""

from __future__ import annotations

import sys
from typing import List, Optional


def _install_instrumentator_patch() -> None:
    try:
        import prometheus_fastapi_instrumentator.routing as routing
        from starlette.routing import Match
        from starlette.types import Scope
    except Exception as exc:  # noqa: BLE001
        # If the instrumentator isn't present, there's nothing to patch.
        print(f"[serve_vllm] instrumentator patch skipped: {exc}", file=sys.stderr)
        return

    def _safe_get_route_name(scope: "Scope", routes: List, route_name: Optional[str] = None) -> Optional[str]:
        for route in routes:
            match, child_scope = route.matches(scope)
            if match == Match.FULL:
                name = getattr(route, "path", None)
                child_scope = {**scope, **child_scope}
                # Starlette 1.x mounts/included-routers expose sub-routes; recurse
                # into ANY route that has them (not just Mount) so we never read a
                # missing `.path` on _IncludedRouter.
                sub_routes = getattr(route, "routes", None)
                if sub_routes:
                    child = _safe_get_route_name(child_scope, sub_routes, name)
                    if child is None:
                        return None
                    return (name + child) if name else child
                return name
            elif match == Match.PARTIAL and route_name is None:
                route_name = getattr(route, "path", None)
        return route_name

    routing._get_route_name = _safe_get_route_name
    print("[serve_vllm] patched prometheus_fastapi_instrumentator routing for Starlette 1.x", file=sys.stderr)


def main() -> None:
    _install_instrumentator_patch()
    from vllm.entrypoints.cli.main import main as vllm_main

    # Pass our CLI args straight through to the vllm CLI (argv[0] -> "vllm").
    sys.argv = ["vllm"] + sys.argv[1:]
    vllm_main()


if __name__ == "__main__":
    main()
