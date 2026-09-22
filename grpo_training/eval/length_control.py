#!/usr/bin/env python3
"""Length-controlled impact win rate (Length-Controlled AlpacaEval, arXiv 2404.04475).

The pairwise impact judge scores candidate-vs-gold and the candidate is almost
always LONGER than gold (our trained models never truncate), so "longer" and
"the model we are scoring" are confounded: a high raw win rate could be the
judge rewarding length rather than quality. This module debiases the win rate
post-hoc, with NO re-judging, following Dubois, Galambosi, Liang & Hashimoto,
"Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators".

TWO STEPS
---------
1) FIT (offline, over a POOL of models so length straddles 0). Per judge, fit a
   logistic GLM on every scored idea:

       logit P(cand strictly wins) = theta_model  +  phi * tanh(dlen / std(dlen))

   - dlen        = tok(candidate) - tok(gold), from meta.impact.prompt (cl100k PROXY)
   - theta_model = one intercept per model  (its quality signal, length removed)
   - phi         = the judge's LENGTH BIAS: log-odds of a win per +1 std of extra
                   length. phi > 0 means the judge rewards longer candidates.
   Target is BINARY: win (impact==1.0) -> 1, tie/loss -> 0. (Ties are kept in the
   win/same/loss breakdown the leaderboard reports; they are just not "wins" here.)
   We persist only the tiny, model-independent constants {phi, dlen_std} per judge.

2) APPLY (per file, in the leaderboard). For a NEW file the model is not in the
   fitted pool, so we FREEZE phi & dlen_std for that judge and solve the single
   intercept theta that best fits this file's (win, dlen) pairs (1-D logistic
   regression, phi held fixed). Then:

       lc_win = logistic(theta)          # the "same length as gold" counterfactual

   This is the win rate the model would get if its outputs were the same length
   as gold, using the judge's own measured length appetite.

Judge is auto-detected per idea from meta.impact.model. cl100k is a PROXY for the
judges' real tokenizers; phi magnitudes are approximate but the ORDERING across
judges (and the direction of the correction) is robust.

CLI
---
  # fit over a pool and write params
  python grpo_training/eval/length_control.py fit \
      inference/eval/jul13_scijudge/*_with_evals.jsonl \
      inference/eval/jul13_gemini/*_with_evals.jsonl \
      inference/eval/jul_13_claude/*_with_evals.jsonl \
      --out inference/eval/lc_params.json

  # show raw win/same/loss + lc_win for one file
  python grpo_training/eval/length_control.py apply \
      inference/eval/jul13_scijudge/SOME_with_evals.jsonl \
      --params inference/eval/lc_params.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# tiktoken / numpy are only needed for FIT and APPLY math. Import lazily so a
# stdlib-only consumer (serve.py) can import this module and degrade gracefully.
#
# tiktoken.get_encoding downloads the cl100k_base BPE file on first use and, with
# TIKTOKEN_CACHE_DIR unset, caches it in an OS tempdir that gets wiped -- so it
# re-fetches over the network each fresh boot and, if that fetch fails (offline /
# flaky net), HAVE_DEPS silently flips False and lc_win is skipped. Pin the cache to
# a persistent per-user dir BEFORE get_encoding so it downloads at most once, ever.
import os as _os

_os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR",
    str(Path(_os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tiktoken"),
)
try:
    import numpy as np
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    HAVE_DEPS = True
except Exception:  # pragma: no cover - optional deps
    np = None
    _ENC = None
    HAVE_DEPS = False


# --------------------------------------------------------------------------- #
# Reading impact verdicts out of *_with_evals.jsonl
# --------------------------------------------------------------------------- #
def _split_ideas(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Split an impact prompt into (candidate=Idea A, gold=Idea B) text."""
    ia = prompt.find("Idea A:")
    ib = prompt.find("Idea B:")
    if ia < 0 or ib < 0:
        return (None, None)
    return (prompt[ia + 7:ib].strip(), prompt[ib + 7:].strip())


def iter_impacts(path: str):
    """Yield (judge_model, instance_id, dlen, is_win) for every scored idea in a file.

    instance_id is the record's target_arxiv_id (fallback: index) -- the shared
    test example the idea was generated for, used to fit per-instance intercepts.
    dlen is tok(cand)-tok(gold); is_win is 1.0 iff impact==1.0 else 0.0.
    Requires deps (tokenizer). Skips ideas with no impact / unparseable prompt.
    """
    if not HAVE_DEPS:
        raise RuntimeError("length_control: numpy/tiktoken required for tokenization")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            inst = str(d.get("target_arxiv_id") or d.get("index"))
            for evl in (d.get("evals") or []):
                if not isinstance(evl, list):
                    continue
                for e in evl:
                    imp = e.get("impact")
                    if not imp or imp.get("impact") is None:
                        continue
                    meta_imp = ((e.get("meta") or {}).get("impact") or {})
                    judge = meta_imp.get("model")
                    a, b = _split_ideas(meta_imp.get("prompt") or "")
                    if a is None:
                        continue
                    dlen = len(_ENC.encode(a)) - len(_ENC.encode(b))
                    is_win = 1.0 if imp["impact"] == 1.0 else 0.0
                    yield judge, inst, float(dlen), is_win


# --------------------------------------------------------------------------- #
# FIT: per-judge phi over a pool of models
# --------------------------------------------------------------------------- #
def fit_from_rows(
    by_judge: Dict[str, List[Tuple[str, str, float, float]]],
    per_instance: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Same GLM as ``fit``, from already-extracted (model, inst, dlen, is_win) rows.

    Use this when verdicts are not in ``*_with_evals.jsonl`` (e.g. GiantsBench
    claim-only pair files). ``dlen`` must already be tok(cand) - tok(gold).
    """
    from sklearn.linear_model import LogisticRegression

    if not HAVE_DEPS:
        raise RuntimeError("length_control: numpy/sklearn/tiktoken required to fit")

    params: Dict[str, Dict[str, float]] = {}
    for judge, rows in by_judge.items():
        models = sorted({r[0] for r in rows})
        mi = {m: i for i, m in enumerate(models)}
        M = len(models)
        insts = sorted({r[1] for r in rows})
        ii = {s: i for i, s in enumerate(insts)}
        I = len(insts) if per_instance else 0
        dlen = np.array([r[2] for r in rows], dtype=float)
        y = np.array([r[3] for r in rows], dtype=float)
        std = float(dlen.std()) or 1.0
        lenfeat = np.tanh(dlen / std)
        # design: model dummies [+ instance dummies] + length feature, no shared
        # intercept. Model + instance dummies are individually collinear with a
        # global intercept, but there is none here and L2 (C=1.0) keeps the
        # remaining model<->instance rank deficiency well-posed.
        X = np.zeros((len(rows), M + I + 1))
        X[np.arange(len(rows)), [mi[r[0]] for r in rows]] = 1.0
        if per_instance:
            X[np.arange(len(rows)), [M + ii[r[1]] for r in rows]] = 1.0
        X[:, M + I] = lenfeat
        clf = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000)
        clf.fit(X, y)
        coef = clf.coef_[0]
        phi = float(coef[M + I])
        rec = {
            "phi": phi,
            "dlen_std": std,
            "n_models": M,
            "n_ideas": len(rows),
            "per_instance": bool(per_instance),
        }
        # per-model gold-length win rate straight off the joint fit:
        # mean over instances of logistic(theta_m + beta_i) (length term = 0).
        theta = coef[:M]
        beta = coef[M:M + I] if per_instance else np.zeros(1)
        lc_fit: Dict[str, float] = {}
        for m, idx in mi.items():
            z = theta[idx] + (beta if per_instance else 0.0)
            lc_fit[m] = float(np.mean(1.0 / (1.0 + np.exp(-z))))
        rec["lc_win_fit"] = lc_fit
        rec["n_instances"] = I if per_instance else len(insts)
        params[judge] = rec
    return params


def fit(files: List[str], per_instance: bool = False) -> Dict[str, Dict[str, float]]:
    """Fit {judge: {phi, dlen_std, n_models, n_ideas, ...}} over a pool of files.

    Each file is one 'model' (its own intercept). Files are grouped by the judge
    auto-detected from meta.impact.model, and one GLM is fit per judge:

        logit P(win) = theta_model  [+ beta_instance if per_instance]  + phi*lenfeat

    We only persist the model-independent {phi, dlen_std}; those are what APPLY
    freezes. When `per_instance` is set, the design also carries one dummy per
    shared test instance (beta_instance), following LC-AlpacaEval, so instance
    difficulty is controlled out of phi instead of leaking into it. Because the
    fit pool sees every model on the SAME instances, per-model lc_win can be read
    straight off the joint fit -- the counterfactual "gold-length" win rate,
    averaged over instances: mean_i logistic(theta_m + beta_i). We stash those
    under params[judge]["lc_win_fit"][model] so the CLI can show them without an
    apply pass (APPLY's 1-D re-solve is only for models NOT in the pool).
    """
    by_judge: Dict[str, List[Tuple[str, str, float, float]]] = {}
    for fp in files:
        model_id = _model_name(fp)
        for judge, inst, dlen, win in iter_impacts(fp):
            if not judge:
                continue
            by_judge.setdefault(judge, []).append((model_id, inst, dlen, win))
    return fit_from_rows(by_judge, per_instance=per_instance)


# --------------------------------------------------------------------------- #
# APPLY: freeze phi & std, solve the single intercept for one file
# --------------------------------------------------------------------------- #
def _solve_theta(lenfeat, y, phi: float) -> float:
    """1-D Newton solve for theta in logistic(theta + phi*lenfeat) vs binary y.

    phi is FROZEN; we fit only the intercept. Returns theta (clipped for sanity).
    """
    theta = 0.0
    offset = phi * lenfeat
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(theta + offset)))
        grad = float(np.sum(p - y))
        hess = float(np.sum(p * (1.0 - p))) + 1e-9
        step = grad / hess
        theta -= step
        if abs(step) < 1e-9:
            break
    return float(np.clip(theta, -20.0, 20.0))


def lc_win_rate(path: str, params: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Length-controlled strict-win rate for one file, or None if not computable.

    Auto-detects the judge; freezes that judge's phi & dlen_std; solves the
    single intercept theta for this file; returns logistic(theta).
    """
    lc, _judge = lc_win_rate_and_judge(path, params)
    return lc


def lc_win_rate_and_judge(path: str, params: Dict[str, Dict[str, float]]) -> Tuple[Optional[float], Optional[str]]:
    """Like lc_win_rate but also returns the auto-detected judge model id."""
    if not HAVE_DEPS:
        return (None, None)
    rows = list(iter_impacts(path))
    if not rows:
        return (None, None)
    judge = rows[0][0]
    jp = params.get(judge)
    if not jp:
        return (None, judge)
    dlen = np.array([r[2] for r in rows], dtype=float)
    y = np.array([r[3] for r in rows], dtype=float)
    lenfeat = np.tanh(dlen / (jp["dlen_std"] or 1.0))
    theta = _solve_theta(lenfeat, y, jp["phi"])
    return (float(1.0 / (1.0 + math.exp(-theta))), judge)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def _model_name(path: str) -> str:
    """Infer a model name from a *_with_evals.jsonl filename."""
    stem = Path(path).name
    for suf in ("_with_evals.jsonl", ".jsonl"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    # strip common eval-pipeline tails so the same model lines up across judges
    for tail in ("_test_data_2026_jun23__16k", "_lang_phys", "_imp_sci", "_imp_scijudge",
                 "_imp_gemini", "_imp_claude", "_with_target_eval"):
        stem = stem.replace(tail, "")
    return stem or Path(path).stem


def load_params(path: str) -> Dict[str, Dict[str, float]]:
    with open(path) as f:
        return json.load(f)


def _win_same_loss(path: str) -> Tuple[float, float, float, int]:
    """Raw win/same/loss rates (no length control), reading swap-averaged impact."""
    w = s = l = n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            for evl in (d.get("evals") or []):
                if not isinstance(evl, list):
                    continue
                for e in evl:
                    imp = e.get("impact")
                    if not imp or imp.get("impact") is None:
                        continue
                    v = imp["impact"]
                    n += 1
                    if v >= 1:
                        w += 1
                    elif v <= 0:
                        l += 1
                    else:
                        s += 1
    if not n:
        return (0.0, 0.0, 0.0, 0)
    return (w / n, s / n, l / n, n)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="fit phi per judge over a pool of files")
    pf.add_argument("files", nargs="+")
    pf.add_argument("--out", default="inference/eval/lc_params.json")
    pf.add_argument("--per-instance", action="store_true",
                    help="add one dummy per shared test instance (LC-AlpacaEval "
                         "beta_i) so instance difficulty is controlled out of phi")

    pa = sub.add_parser("apply", help="show raw win/same/loss + lc_win for files")
    pa.add_argument("files", nargs="+")
    pa.add_argument("--params", default="inference/eval/lc_params.json")

    ps = sub.add_parser("stamp", help="write a folder-local lc_params.json (only the "
                                      "judges present) into each DIR, so serve can "
                                      "length-correct files it loads from that folder")
    ps.add_argument("dirs", nargs="+")
    ps.add_argument("--params", default="inference/eval/lc_params.json")

    args = ap.parse_args()

    if args.cmd == "fit":
        if not HAVE_DEPS:
            raise SystemExit("fit requires numpy, scikit-learn, tiktoken")
        params = fit(args.files, per_instance=args.per_instance)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(params, f, indent=2, sort_keys=True)
        print(f"wrote {args.out}   (per_instance={args.per_instance})")
        print(f"{'judge':<28} {'phi':>8} {'dlen_std':>10} {'n_models':>9} {'n_inst':>7} {'n_ideas':>8}")
        for j, p in sorted(params.items()):
            print(f"{j:<28} {p['phi']:>8.3f} {p['dlen_std']:>10.1f} {p['n_models']:>9} "
                  f"{p.get('n_instances', '-'):>7} {p['n_ideas']:>8}")
        # per-model gold-length win rate read straight off the joint fit
        for j, p in sorted(params.items()):
            lcf = p.get("lc_win_fit") or {}
            if not lcf:
                continue
            print(f"\n[{j}] per-model lc_win (gold-length counterfactual off the joint fit):")
            for m, v in sorted(lcf.items(), key=lambda kv: -kv[1]):
                print(f"    {m:<48} {v*100:6.1f}%")
        print("\nphi = log-odds of a strict win per +1 std of extra candidate length "
              "(tanh-saturated). phi>0 => judge rewards length.")
    elif args.cmd == "stamp":
        if not HAVE_DEPS:
            raise SystemExit("stamp requires numpy/tiktoken (it reads each folder to "
                             "learn which judge lives there)")
        params = load_params(args.params)
        for d in args.dirs:
            dp = Path(d)
            # which judges actually appear in this folder's files?
            judges = set()
            for fp in sorted(dp.glob("*_with_evals.jsonl")):
                for judge, _inst, _dlen, _win in iter_impacts(str(fp)):
                    if judge:
                        judges.add(judge)
                    break  # first scored idea tells us the judge for this file
            local = {j: params[j] for j in judges if j in params}
            if not local:
                print(f"{d}: no known judge found, skipped")
                continue
            out = dp / "lc_params.json"
            with open(out, "w") as f:
                json.dump(local, f, indent=2, sort_keys=True)
            print(f"wrote {out}  judges={sorted(local)}")
    else:
        params = load_params(args.params)
        SHORT = {"OpenMOSS-Team/SciJudge-30B": "scijudge", "gemini-2.5-flash": "gemini",
                 "claude-opus-4-8": "claude"}
        rows = []
        for fp in args.files:
            w, s, l, n = _win_same_loss(fp)
            if n == 0:
                continue
            lc, judge = lc_win_rate_and_judge(fp, params) if HAVE_DEPS else (None, None)
            rows.append((SHORT.get(judge, judge or "?"), Path(fp).name, w, s, l, lc, n))
        rows.sort(key=lambda r: (r[0], r[1]))
        print(f"{'judge':<9} {'file':<70} {'win':>7} {'same':>7} {'loss':>7} {'lc_win':>8} {'n':>6}")
        for judge, name, w, s, l, lc, n in rows:
            lcs = f"{lc:.3f}" if lc is not None else "  n/a"
            print(f"{judge:<9} {name:<70} {w:>7.3f} {s:>7.3f} {l:>7.3f} {lcs:>8} {n:>6}")


if __name__ == "__main__":
    _main()
