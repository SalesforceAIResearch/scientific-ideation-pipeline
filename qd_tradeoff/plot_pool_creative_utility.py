#!/usr/bin/env python3
"""Creative-utility (CU) + QD curves over the ARM-B pooled set, GREEDY-ordered.

Arm B draws k ideas x 7 independent samples and pools them (~35-40 ideas per
instance). Unlike arm A there is no single meaningful generation order across the
7 fresh draws, so instead of walking the emit-order we walk a GREEDY order over
the pool -- the natural "if you could pick your ideas one at a time to maximize
novelty*quality, in what order would you take them, and where does the marginal
run dry?" This is the arm-B analog of the arm-A creative-utility plot.

DATA (per model, joined by instance):
  * strengths -> <tag>_summary_with_evals.jsonl : per-idea impact.impact in {0,.5,1}
    (--strength novelty uses novelty.novelty_score). Pooled draw-then-idea order.
  * distances -> pool_diversity.jsonl (built by score_pool_diversity.py): the FULL
    pooled C(n,2) F9-B rubric distance matrix, INCLUDING cross-draw pairs, so a
    pooled greedy walk sees the true redundancy between draws (the per-draw
    diversity[] blocks miss it entirely). Same index space as the strengths.

GREEDY ORDER (farthest-point / MMR-style, the CU-native pick order):
  * step 0 : the highest-strength idea (best single idea you'd take first).
  * step i : among the not-yet-picked, take the one maximizing its MARGINAL
             value  min_dist_to_ALREADY_PICKED(idea) * strength(idea).
    i.e. greedily grab the idea that adds the most NEW quality-weighted coverage.
  The marginal is then patience-discounted (marg_i *= p**(i-1)) exactly like the
  arm-A CU walk, and we report both the cumulative sum and the patience-weighted
  running mean (normalized CU).

Two panels:
  LEFT  cumulative CU vs pick k'  -- how much quality-weighted-distinct value the
        first k' greedily-chosen ideas add. Flattening = the pool is exhausted.
  RIGHT normalized CU (patience-weighted running mean). By construction the greedy
        marginals are non-increasing (min-distance-to-picked can only shrink as the
        picked set grows), so this curve NEVER rises -- it decays toward a floor.
        The discriminator is the DECAY RATE and FLOOR HEIGHT, read comparatively:
        gentle decay to a high floor = sustained quality-diversity (each pick keeps
        paying a comparable distance*strength marginal deep into the pool); steep
        decay to a low floor = a few strong ideas up front, then redundancy.

All curves macro over instances: per-instance sequence first, then mean +/- SEM
across the (usually 100) instances at each step.

Usage:
  synthesis_env/bin/python qd_tradeoff/plot_pool_creative_utility.py \
      "gpt-5.5=qd_tradeoff/evals/b_sample_sweep/gpt-5.5-2026-04-23" \
      "claude-opus=qd_tradeoff/evals/b_sample_sweep/claude-opus-4-8" \
      "full-rl=qd_tradeoff/evals/b_sample_sweep/impdist-full-dist-h200-400" \
      "empty-think=qd_tradeoff/evals/b_sample_sweep/impdist-empty-think-sft-rl-400" \
      --patience 0.9 --strength impact \
      --out qd_tradeoff/evals/b_sample_sweep/pool_creative_utility.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_QD = Path(__file__).resolve().parent
_REPO = _QD.parent
_GRPO = _REPO / "grpo_training"
for _p in (str(_REPO), str(_GRPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import ideas as I  # noqa: E402

PATIENCES = [0.5, 0.7, 0.9, 1.0]

# per-model summary file stems (empty-think used n32, the rest n7)
_SUMMARY_CANDIDATES = ("n7_summary_with_evals.jsonl", "n32_summary_with_evals.jsonl")

# jewel-tone palette: saturated lines + translucent fills. Semantic grouping:
# API models -> warm (orange / crimson), trained models -> cool (blue / green).
_MODEL_COLORS = {
    "gpt-5.5": "#E08E3C",      # orange   (API)
    "claude-opus": "#C9425A",  # crimson  (API)
    "full-rl": "#5B8FD4",      # blue     (trained)
    "empty-think": "#5AAE61",  # green    (trained)
}
_FALLBACK = ["#5B8FD4", "#E08E3C", "#5AAE61", "#C9425A",
             "#9E6FC0", "#4CB5AE", "#D6B843", "#8C8C8C"]


def _color_for(label, idx):
    return _MODEL_COLORS.get(label, _FALLBACK[idx % len(_FALLBACK)])


def _iter(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _resolve_pair(spec: Path) -> tuple[Path, Path]:
    """Resolve (summary_with_evals, pool_diversity) for one input `spec`.

    Two accepted layouts:
      DIR    `spec` is a directory (arm-B): look for nX_summary_with_evals.jsonl
             + pool_diversity.jsonl inside it.
      PREFIX `spec` is a file PREFIX (e2e_evals flat layout): use
             <prefix>.evals.jsonl + <prefix>.pool_diversity.jsonl.
    The prefix layout lets report.py pass results/<run>/<model> straight through."""
    if spec.is_dir():
        summ = None
        for name in _SUMMARY_CANDIDATES:
            if (spec / name).exists():
                summ = spec / name
                break
        if summ is None:
            raise SystemExit(f"no *_summary_with_evals.jsonl in {spec}")
        return summ, spec / "pool_diversity.jsonl"
    # prefix layout
    summ = spec.with_name(spec.name + ".evals.jsonl")
    pool = spec.with_name(spec.name + ".pool_diversity.jsonl")
    if not summ.exists():
        raise SystemExit(f"no evals file for prefix {spec} (looked for {summ})")
    return summ, pool


# CU strength modes this walk understands. impact/novelty/impact_x_novelty are the
# original arm-B set; soundness/grounding/impact_x_grounding read the per-idea
# fields the e2e_evals grounding + pointwise-soundness phases write into evals[].
STRENGTH_MODES = ("impact", "novelty", "impact_x_novelty",
                  "soundness", "grounding", "impact_x_grounding")


def _pooled_strengths(rec, mode, strict_impact=False):
    """Per-idea strength in pooled draw-then-idea order.
      impact             -> impact.impact in {0,.5,1}
      novelty            -> novelty.novelty_score in [0,1]
      soundness          -> soundness_pointwise.reward in [0,1]
      grounding          -> grounding.g in [0,1] (load-bearing multiplier)
      impact_x_novelty   -> product impact.impact * novelty.novelty_score
      impact_x_grounding -> product impact.impact * grounding.g
    Missing signal -> 0.0 (for a product, a missing factor zeroes the idea).
    strict_impact: binarize the impact factor to a strict win -- 1.0 stays 1,
    everything else (0.5 tie or 0.0 loss) -> 0 -- so only outright wins carry
    strength (applies to any mode using impact; no effect on the others)."""
    out = []
    for sample in rec.get("evals") or []:
        for idea in (sample if isinstance(sample, list) else [sample]):
            imp = (idea.get("impact") or {}).get("impact")
            nov = (idea.get("novelty") or {}).get("novelty_score")
            snd = (idea.get("soundness_pointwise") or {}).get("reward")
            grd = (idea.get("grounding") or {}).get("g")
            imp = 0.0 if imp is None else float(imp)
            nov = 0.0 if nov is None else float(nov)
            snd = 0.0 if snd is None else float(snd)
            grd = 0.0 if grd is None else float(grd)
            if strict_impact:
                imp = 1.0 if imp == 1.0 else 0.0
            if mode == "impact":
                out.append(imp)
            elif mode == "novelty":
                out.append(nov)
            elif mode == "soundness":
                out.append(snd)
            elif mode == "grounding":
                out.append(grd)
            elif mode == "impact_x_grounding":
                out.append(imp * grd)
            else:  # impact_x_novelty
                out.append(imp * nov)
    return out


def _dist_matrix(pool_rec, n):
    """Rebuild the n x n pooled rubric distance matrix from pool_diversity pairs.
    Missing/unscored pairs -> 1.0 (treated as fully distinct: rare, and this way a
    dropped judge call can't spuriously suppress a marginal)."""
    dm = np.ones((n, n), dtype=float)
    np.fill_diagonal(dm, 0.0)
    for p in pool_rec.get("pairs") or []:
        i, j, d = p.get("i"), p.get("j"), p.get("distance")
        if i is None or j is None or i >= n or j >= n:
            continue
        d = 1.0 if d is None else float(d)
        dm[i, j] = dm[j, i] = d
    return dm


def greedy_order(dm, strengths):
    """Farthest-point/MMR greedy pick order + the per-step MARGINAL along it.

    step 0 : argmax strength.
    step i : argmax over unpicked of  min_dist_to_picked(idea) * strength(idea).
    Returns (order, marginals) where marginals[0] = strength(order[0]) and
    marginals[i>0] = min_dist_to_earlier(order[i]) * strength(order[i]).
    This is the same per-step score the arm-A CU walk uses, just along the greedy
    order instead of the emit order."""
    n = len(strengths)
    if n == 0:
        return [], []
    s = np.asarray(strengths, dtype=float)
    first = int(np.argmax(s))
    order = [first]
    marg = [float(s[first])]
    picked = {first}
    # running min distance from each candidate to the picked set
    min_to_picked = dm[first].copy()
    while len(order) < n:
        best, best_val = -1, -1.0
        for c in range(n):
            if c in picked:
                continue
            val = float(min_to_picked[c]) * float(s[c])
            if val > best_val:
                best_val, best = val, c
        order.append(best)
        marg.append(best_val)
        picked.add(best)
        min_to_picked = np.minimum(min_to_picked, dm[best])
    return order, marg


def marg_seq(marg, p):
    return np.array([u * (p ** (i - 1)) if i else u for i, u in enumerate(marg)])


def weight_seq(n, p):
    return np.array([(p ** (i - 1)) if i else 1.0 for i in range(n)])


def normed_seq(marg, p):
    m = marg_seq(marg, p)
    w = weight_seq(len(marg), p)
    return np.cumsum(m) / np.cumsum(w)


def load_sequences(model_dir: Path, mode, strict_impact=False, domains=None,
                   with_ids=False):
    """Join summary strengths + pooled distances by instance; return per-instance
    greedy marginal sequences. `model_dir` may be a directory (arm-B layout) or a
    file prefix (e2e_evals flat layout) -- see _resolve_pair.

    `domains` (case-insensitive set of coarse-domain / arXiv-primary labels), when
    given, restricts the walk to matching instances. Filtering here -- on the
    summary side, which carries `domain`/`primary` -- scopes CU to the target
    subset even when the pool file was scored over every domain.

    `with_ids=True` returns [(instance_id, marginal_seq)] instead of [marginal_seq], so a
    caller can PAIR two runs on instances. Needed because the curves below macro over
    instances and keep only mean/SEM, which supports an unpaired delta at best -- and when
    two arms are two think budgets on the SAME 162 instances, instance difficulty is the
    dominant variance component and pairing removes it."""
    summ, pool = _resolve_pair(model_dir)
    if not pool.exists():
        raise SystemExit(f"missing {pool} -- run score_pool_diversity.py first")
    poolmap = {r["instance_id"]: r for r in _iter(pool)}
    wanted = {d.strip().lower() for d in domains} if domains else None
    seqs = []
    inst_ids = []
    skipped = 0
    for rec in _iter(summ):
        if wanted is not None and not (
                {str(rec.get("domain") or "").lower(),
                 str(rec.get("primary") or "").lower()} & wanted):
            continue
        inst = str(rec.get("target_arxiv_id") or rec.get("index"))
        pr = poolmap.get(inst)
        if pr is None:
            skipped += 1
            continue
        st = _pooled_strengths(rec, mode, strict_impact=strict_impact)
        n = min(len(st), pr["n_ideas"])
        if n < 1:
            skipped += 1
            continue
        st = st[:n]
        dm = _dist_matrix(pr, n)
        _, marg = greedy_order(dm, st)
        seqs.append(marg)
        inst_ids.append(inst)
    return (list(zip(inst_ids, seqs)), skipped) if with_ids else (seqs, skipped)


def curves_over(seqs, p):
    """(cum mean/sem, normed mean/sem, counts) across instances at each step."""
    maxlen = max(len(s) for s in seqs)
    steps = np.arange(1, maxlen + 1)
    cum = [np.cumsum(marg_seq(s, p)) for s in seqs]
    nrm = [normed_seq(s, p) for s in seqs]
    cm, cs, nm, ns, cnt = [], [], [], [], []
    for i in steps:
        cv = np.array([c[i - 1] for c in cum if len(c) >= i])
        nv = np.array([r[i - 1] for r in nrm if len(r) >= i])
        cm.append(cv.mean()); cs.append(cv.std() / len(cv) ** .5)
        nm.append(nv.mean()); ns.append(nv.std() / len(nv) ** .5)
        cnt.append(len(cv))
    return (steps, np.array(cm), np.array(cs), np.array(nm), np.array(ns),
            np.array(cnt))


def main():
    ap = argparse.ArgumentParser(description="pooled greedy creative-utility (arm B)")
    ap.add_argument("inputs", nargs="+", help="LABEL=model_dir pairs")
    ap.add_argument("--patience", type=float, default=0.9)
    ap.add_argument("--strength", choices=list(STRENGTH_MODES), default="impact")
    ap.add_argument("--json", dest="json_out", type=Path, default=None,
                    help="also dump headline CU numbers (cumCU + normCU per step, "
                         "per model, at --patience) to this JSON path so report.py "
                         "reads structured values instead of scraping stdout.")
    ap.add_argument("--strict-impact", action="store_true",
                    help="binarize the impact factor: strict win (1.0)->1, tie/loss->0 "
                         "(applies to impact & impact_x_novelty; no effect on novelty)")
    ap.add_argument("--xmax", type=int, default=35,
                    help="cap the rendered x range (default 35 = the 5x7 pool)")
    ap.add_argument("--normalized-only", action="store_true",
                    help="render ONLY the normalized-CU panel (single axes)")
    ap.add_argument("--domain", action="append", default=[], metavar="LABEL",
                    help="keep only instances whose coarse `domain` OR arXiv "
                         "`primary` matches, case-insensitive. Repeatable. Mirrors "
                         "run_all_evals.py / score_pool_diversity.py --domain.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = args.patience
    series = []
    per_inst = {}   # label -> (instance_ids, marginal seqs); dumped for PAIRED deltas
    for item in args.inputs:
        if "=" not in item:
            ap.error(f"expected LABEL=model_dir, got {item!r}")
        label, d = item.split("=", 1)
        pairs, skipped = load_sequences(Path(d), args.strength,
                                        strict_impact=args.strict_impact,
                                        domains=args.domain or None,
                                        with_ids=True)
        if not pairs:
            print(f"!! {label}: no sequences -- skipping")
            continue
        ids = [i for i, _ in pairs]
        seqs = [s for _, s in pairs]
        steps, cm, cs, nm, ns, cnt = curves_over(seqs, g)
        series.append((label, steps, cm, cs, nm, ns, cnt))
        per_inst[label] = (ids, seqs)
        # table
        ck = [k for k in (1, 2, 4, 8, 16, 24, 32) if k <= len(steps)]
        print(f"\n=== {label}  (n_instances={len(seqs)}, skipped={skipped}, "
              f"strength={args.strength}, p={g}) ===")
        print(f'{"k":>4}{"n":>6}{"cumCU":>10}{"normCU":>10}')
        for k in ck:
            print(f'{k:>4}{int(cnt[k-1]):>6}{cm[k-1]:>10.3f}{nm[k-1]:>10.4f}')

    if not series:
        ap.error("no plottable series")

    # ---- optional structured dump of the headline CU numbers ----
    # report.py reads this instead of scraping the stdout table. One block per
    # model: cumCU + normCU per greedy pick step at --patience, plus the headline
    # values at the deepest step (the pool-exhausted CU) and a small checkpoint
    # table. `steps` is 1-indexed pick depth.
    if args.json_out is not None:
        dump = {"patience": g, "strength": args.strength,
                "strict_impact": bool(args.strict_impact), "models": {}}
        for label, steps, cm, cs, nm, ns, cnt in series:
            last = len(steps) - 1
            dump["models"][label] = {
                "n_steps": int(len(steps)),
                "n_instances_at_step1": int(cnt[0]) if len(cnt) else 0,
                "steps": [int(s) for s in steps],
                "cum_cu": [float(x) for x in cm],
                "cum_cu_sem": [float(x) for x in cs],
                "norm_cu": [float(x) for x in nm],
                "norm_cu_sem": [float(x) for x in ns],
                "counts": [int(x) for x in cnt],
                # headline: final (deepest) cumulative CU + the normalized-CU floor
                "cum_cu_final": float(cm[last]),
                "norm_cu_final": float(nm[last]),
            }
            # Per-instance curves, so a consumer can compute a PAIRED delta between two
            # arms (same instances = same difficulty) instead of differencing two macro
            # means. cum_cu is the running sum of greedy marginals; norm_cu divides by the
            # depth reached, both 1-indexed by pick depth like `steps`.
            ids, seqs = per_inst.get(label, ([], []))
            dump["models"][label]["per_instance"] = {
                str(i): {"cum_cu": [float(x) for x in np.cumsum(marg_seq(s, g))],
                         "norm_cu": [float(x) for x in normed_seq(s, g)]}
                for i, s in zip(ids, seqs)
            }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(dump, indent=2))
        print(f"wrote {args.json_out}")

    if args.normalized_only:
        fig, axn = plt.subplots(figsize=(8, 6))
        axc = None
    else:
        fig, (axc, axn) = plt.subplots(1, 2, figsize=(15, 6))
    for idx, (label, steps, cm, cs, nm, ns, cnt) in enumerate(series):
        c = _color_for(label, idx)
        xm = min(args.xmax, len(steps))
        x = steps[:xm]
        if axc is not None:
            axc.plot(x, cm[:xm], lw=2.3, color=c, label=label)
            axc.fill_between(x, (cm - cs)[:xm], (cm + cs)[:xm], color=c, alpha=0.22)
        axn.plot(x, nm[:xm], lw=2.3, color=c, label=label)
        axn.fill_between(x, (nm - ns)[:xm], (nm + ns)[:xm], color=c, alpha=0.22)
        # mark where <50% of instances still reach this depth (pool thins out)
        full = cnt[0] if len(cnt) else 0
        thin = x[cnt[:xm] < full / 2]
        if len(thin) and axc is not None:
            axc.axvline(thin[0], ls=":", color=c, alpha=0.4)

    strlab = {"impact": "impact", "novelty": "novelty",
              "impact_x_novelty": "impact x novelty"}[args.strength]
    if args.strict_impact and args.strength != "novelty":
        strlab = strlab.replace("impact", "strict-win")
    if axc is not None:
        axc.set_xlabel("greedy pick k'  (ideas taken best-marginal-first)")
        axc.set_ylabel(f"cumulative creative utility  ({strlab}-weighted distinct)")
        axc.set_title(f"Pooled cumulative CU vs greedy pick k'  (p={g})\n"
                      "flattening = quality-weighted-distinct pool exhausted")
        axc.grid(alpha=0.3); axc.legend(fontsize=9)

    axn.set_xlabel("greedy pick k'  (ideas taken best-marginal-first)")
    axn.set_ylabel("normalized CU  (patience-weighted running mean)")
    axn.set_title("Normalized CU\ngentle decay to a high floor = sustained quality-diversity; "
                  "steep decay to a low floor = a few strong ideas, then redundancy")
    axn.grid(alpha=0.3); axn.legend(fontsize=9)

    strmode = args.strength + ("  (strict-win impact)" if args.strict_impact
                               and args.strength != "novelty" else "")
    fig.suptitle(f"Arm B pooled greedy creative utility  |  strength = per-idea {strmode}  "
                 f"|  F9-B rubric distance (incl. cross-draw)  |  gamma={g}", y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
