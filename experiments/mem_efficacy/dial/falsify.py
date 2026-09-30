"""Falsification suite.  Run BEFORE trusting anything in this package.

DIAL rests on four empirical claims.  Each is testable on the archive already on disk,
with no GPU and no new rollouts.  If a claim fails, the layer built on it is wrong and
should be deleted rather than tuned:

  T1  The ATTEMPT predicts credit better than the obligation or the sentence.
      -> if false, `strategy.py`'s cell space is decoration.

  T2  Gate tightness is NOT the cross-arm explanation (the naive version fails), but it
      IS the within-arm explanation.  Reported honestly as: cross-arm comparison is
      confounded, within-arm version ladders are not.
      -> if the within-arm ladder did not move score, there is no gate story at all and
         the no-BLOCK action space has no motivation.

  T3  The enacted->credited lag is long compared with the stagnation horizon the arms
      act on.
      -> if false, the scorer keeps up, `UNSYNCED` is a phantom cause, `RESYNC` is dead
         weight.

  T4  Episode score is BIMODAL, with the mass concentrated at exactly 0.0 and at
      "all stages credited".  The task is therefore "escape 0.0", not "improve the mean".
      -> if false, the objective is the mean and the absorbing-state argument is weaker.

Usage:  python -m dial.falsify            (writes dial/falsify_report.json)
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from .attribution import extract_all, iter_run_dirs
from .types import classify_verb

RESULTS = Path(__file__).resolve().parent.parent / "results"
TASKS = [5, 8, 19, 22]

# ---------------------------------------------------------------------------------------
# Canonical runs.  EXPLICIT, because the archive contains partial reruns and version
# ladders and picking "the latest tag by string sort" silently mixes 3-trial and 10-trial
# runs and reports them as if comparable.  That mistake is why an earlier version of this
# file reported a different GPM mean than the benchmark table.  Selection is now stated.
# ---------------------------------------------------------------------------------------
CANONICAL = {
    # GPM is the `evmem_gpm_*` tag, NOT the plain `evmem_*` tag.  Getting this wrong
    # (an earlier version read `evmem_*`) reported a GPM mean of 32.08 instead of 41.67 and
    # made the cross-arm gate ordering look non-monotone, which would have withdrawn a
    # claim that is in fact supported.  Measured GPM: t5 20.0, t8 46.68, t19 26.67, t22 73.33.
    "evmem": {5: "evmem_gpm_t5_evmem_h0_1x10", 8: "evmem_gpm_t8_evmem_h0_1x10",
              19: "evmem_gpm_t19_evmem_h0_1x10", 22: "evmem_gpm_t22_evmem_h0_1x10"},
    "aom": {5: "aom_t5_evmem_aom_h0_1x10_v2", 8: "aom_t8_evmem_aom_h0_1x10_v6",
            19: "aom_t19_evmem_aom_h0_1x10_v2", 22: "aom_t22_evmem_aom_h0_1x10_v6"},
    "bolt": {5: "bolt_t5_bolt_h0_1x10_v1", 8: "bolt_t8_bolt_h0_1x10_v9",
             19: "bolt_t19_bolt_h0_1x10_v1", 22: "bolt_t22_bolt_h0_1x10_v1"},
}

# The beat-targets the runners enforce, for reference.
GPM_TARGET = {5: 20.0, 8: 46.7, 19: 26.7, 22: 73.3}

# Within-arm version ladders: same arm, same task, only the gate/interface changed.  This
# is the valid natural experiment for "does gate tightness matter"; comparing arm A to arm
# B confounds gate count with the entire architecture.
BOLT_T8_LADDER = [
    ("v1", "bolt_t8_bolt_h0_1x10_v1", "solvability closure at its strictest: unreachable deps deadlock (0.0 x3)"),
    ("v5", "bolt_t8_bolt_h0_1x10_v5", "dependency semantics relaxed -> prep work becomes reachable"),
    ("v6", "bolt_t8_bolt_h0_1x10_v6", "stage gate tightened without lift requirement"),
    ("v7", "bolt_t8_bolt_h0_1x10_v7", "served rungs lose explicit lift verbs (0.0 x3)"),
    ("v8", "bolt_t8_bolt_h0_1x10_v8", "identifier leak: board prints actionable phase names, planner copies them"),
    ("v9", "bolt_t8_bolt_h0_1x10_v9", "board uses non-imperative stage tags + runtime I5 rewrite"),
]


def _trial_scores(tag: str, root: Path = RESULTS) -> list[float]:
    """Per-trial stage_score_pct for a run tag, in trial order."""
    for run in sorted((root / tag / "h0").glob("*_s*")):
        p = run / "prompt_trace.tsv"
        if not p.is_file():
            continue
        lines = p.read_text(errors="replace").splitlines()
        if len(lines) < 2:
            continue
        hdr = lines[0].split("\t")
        if "stage_score_pct" not in hdr:
            continue
        i = hdr.index("stage_score_pct")
        out = []
        for ln in lines[1:]:
            parts = ln.split("\t")
            if len(parts) > i and parts[i]:
                try:
                    out.append(float(parts[i]))
                except ValueError:
                    pass
        if out:
            return out
    return []


def canonical_scores() -> dict:
    """{arm: {task: (mean, n, zeros, trials)}} using the explicit CANONICAL map."""
    out: dict = {}
    for arm, per_task in CANONICAL.items():
        for task, tag in per_task.items():
            v = _trial_scores(tag)
            if not v:
                continue
            out.setdefault(arm, {})[task] = {
                "mean": sum(v) / len(v), "n": len(v),
                "zeros": sum(1 for x in v if x == 0.0), "tag": tag, "trials": v,
            }
    return out


# ---------------------------------------------------------------------------------------
# T1 — does the attempt predict credit?
# ---------------------------------------------------------------------------------------
def _cell(r) -> tuple:
    return (r.family, classify_verb(r.text)) + tuple(r.strategy_key[1:])


def test_attempt_predicts_credit(records) -> dict:
    """Within-family, does the attempt cell carry information about credit?

    Two statistics, because either alone is weak here:
      * a chi-square of (cell x credited) with a WITHIN-FAMILY permutation null, which
        handles the unbalanced and tiny cells the asymptotic table gets wrong;
      * a held-out log-likelihood of family-only vs family+cell, split by EPISODE so no
        attempt from a test episode leaks into training.
    """
    usable = [r for r in records if r.outcome in ("credited", "stalled")]
    by_fam: dict[str, list] = defaultdict(list)
    for r in usable:
        by_fam[r.family].append(r)

    def chi2(pairs) -> float:
        tab: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
        for r, lb in pairs:
            tab[_cell(r)][0 if lb == "stalled" else 1] += 1
        tot = sum(sum(v) for v in tab.values()) or 1
        g = sum(v[1] for v in tab.values()) / tot
        s = 0.0
        for fail, cred in tab.values():
            n = fail + cred
            if n == 0:
                continue
            if n * g > 0:
                s += (cred - n * g) ** 2 / (n * g)
            if n * (1 - g) > 0:
                s += (fail - n * (1 - g)) ** 2 / (n * (1 - g))
        return s

    obs = chi2([(r, r.outcome) for r in usable])
    rng = random.Random(0)
    null = []
    for _ in range(2000):
        pairs = []
        for rs in by_fam.values():
            labels = [r.outcome for r in rs]
            rng.shuffle(labels)
            pairs.extend(zip(rs, labels))
        null.append(chi2(pairs))
    p = (1 + sum(1 for x in null if x >= obs)) / (1 + len(null))

    rng2 = random.Random(1)
    eps = sorted({(r.arm, r.task, r.seed, r.episode) for r in usable})
    rng2.shuffle(eps)
    cut = int(len(eps) * 0.7)
    train_ids, test_ids = set(eps[:cut]), set(eps[cut:])
    train = [r for r in usable if (r.arm, r.task, r.seed, r.episode) in train_ids]
    test = [r for r in usable if (r.arm, r.task, r.seed, r.episode) in test_ids]

    fam_n: dict[str, int] = defaultdict(int)
    fam_c: dict[str, int] = defaultdict(int)
    for r in train:
        fam_n[r.family] += 1
        fam_c[r.family] += 1 if r.outcome == "credited" else 0
    g = (sum(fam_c.values()) + 1) / (sum(fam_n.values()) + 2)
    fam_rate = {f: (fam_c[f] + 4 * g) / (fam_n[f] + 4) for f in fam_n}

    cell_n: dict[tuple, int] = defaultdict(int)
    cell_c: dict[tuple, int] = defaultdict(int)
    for r in train:
        cell_n[_cell(r)] += 1
        cell_c[_cell(r)] += 1 if r.outcome == "credited" else 0
    cell_rate = {k: (cell_c[k] + 8 * fam_rate.get(k[0], g)) / (cell_n[k] + 8) for k in cell_n}

    def ll(rs, fn) -> float:
        s = 0.0
        for r in rs:
            q = min(max(fn(r), 1e-6), 1 - 1e-6)
            s += math.log(q) if r.outcome == "credited" else math.log(1 - q)
        return s / max(1, len(rs))

    ll_fam = ll(test, lambda r: fam_rate.get(r.family, g))
    ll_cell = ll(test, lambda r: cell_rate.get(_cell(r), fam_rate.get(r.family, g)))

    # the headline within-family contrast, reported so the claim is inspectable
    contrast = {}
    for fam in ("lift", "pour", "place", "open"):
        tab = defaultdict(lambda: [0, 0])
        for r in usable:
            if r.family == fam:
                tab[classify_verb(r.text)][0 if r.outcome == "stalled" else 1] += 1
        rows = {k: {"fail": v[0], "cred": v[1],
                    "rate": round(v[1] / max(1, v[0] + v[1]), 3)}
                for k, v in sorted(tab.items(), key=lambda kv: -sum(kv[1])) if sum(v) >= 8}
        if len(rows) >= 2:
            contrast[fam] = rows
    return {
        "n_usable": len(usable),
        "n_credited": sum(1 for r in usable if r.outcome == "credited"),
        "n_cells": len(cell_n), "chi2_obs": round(obs, 2), "p_permutation": p,
        "loglik_family_only": round(ll_fam, 5),
        "loglik_family_plus_cell": round(ll_cell, 5),
        "cell_gain_nats": round(ll_cell - ll_fam, 5),
        "cell_helps": ll_cell > ll_fam,
        "within_family_contrast": contrast,
        "passes": p < 0.01 and ll_cell > ll_fam,
    }


# ---------------------------------------------------------------------------------------
# T2 — gate tightness: the cross-arm version fails, the within-arm version holds
# ---------------------------------------------------------------------------------------
GATE_CONDITIONS = {
    "evmem": (4, "memexp_evmem.control_reject_reason: dep / attractor / unsafe-primitive / repeat"),
    "aom": (7, "memexp_aom: 3-mode obligation + dep closure + proof + ordinal + verbatim"),
    "bolt": (11, "bolt/graph.py I1-I5 + arbiter stage-gate + verbatim + ordinal + closure"),
}


def test_gate_tightness() -> dict:
    """Report BOTH the cross-arm view and the within-arm view, and say what each supports.

    With the correct GPM runs the cross-arm ordering IS monotone (GPM 4 conditions 41.67 >
    AOM 7 conditions 37.92 > BOLT 11 conditions 29.27).  That is reported as observed
    evidence, with its confound stated rather than hidden: the three arms differ in their
    whole architecture, not only in gate count, so the ordering cannot by itself isolate
    the gate.  The unconfounded evidence is the within-arm ladder, where one task is run
    with the architecture fixed and only the gate/interface changed.
    """
    cross = []
    for arm in CANONICAL:
        v = [x for t in CANONICAL[arm] for x in _trial_scores(CANONICAL[arm][t])]
        if not v:
            continue
        cross.append({"arm": arm, "gate_conditions": GATE_CONDITIONS[arm][0],
                      "mean": round(sum(v) / len(v), 2), "n": len(v),
                      "zero_frac": round(sum(1 for x in v if x == 0.0) / len(v), 3),
                      "gates_where": GATE_CONDITIONS[arm][1]})
    cross.sort(key=lambda r: r["gate_conditions"])
    n = len(cross)
    # Signed, unambiguous statistic: Pearson correlation between gate count and mean score.
    # A rank/Spearman encoding was ambiguous here — with the sample sorted by gate count, a
    # rank statistic of +1.0 means "more gates, lower score", which reads as the opposite.
    pearson = float("nan")
    if n >= 3:
        xs = [r["gate_conditions"] for r in cross]
        ys = [r["mean"] for r in cross]
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        if dx > 0 and dy > 0:
            pearson = num / (dx * dy)
    mono = all(cross[i]["mean"] >= cross[i + 1]["mean"] for i in range(len(cross) - 1))

    ladder = []
    for name, tag, note in BOLT_T8_LADDER:
        v = _trial_scores(tag)
        if not v:
            continue
        ladder.append({"version": name, "tag": tag, "n": len(v),
                       "mean": round(sum(v) / len(v), 2),
                       "zero_frac": round(sum(1 for x in v if x == 0.0) / len(v), 3),
                       "change": note, "trials": v})
    means = [r["mean"] for r in ladder]
    spread = round(max(means) - min(means), 2) if means else 0.0
    return {
        "cross_arm": {"rows": cross,
                      "pearson_gates_vs_mean": (round(pearson, 3) if pearson == pearson else None),
                      "strictly_monotone_decreasing": mono,
                      "ordering_supported": bool(mono),
                      "confound": "arms differ in architecture, not only in gate count, so "
                                  "the arm-level ordering is consistent with the gate story "
                                  "but does not isolate it"},
        "within_arm": {"task": 8, "ladder": ladder, "mean_spread": spread,
                       "unconfounded": True},
        "passes": bool(spread >= 20.0 and mono),
        "honest_verdict": (
            f"cross-arm gate ordering is monotone decreasing "
            f"(pearson(gates,mean)={None if pearson != pearson else round(pearson, 3)}), "
            f"consistent with the gate story but confounded by architecture; the "
            f"within-arm ladder on one task moves the mean by {spread} points with the "
            "architecture fixed, which is the unconfounded evidence the no-BLOCK action "
            "space rests on."
        ),
    }


# ---------------------------------------------------------------------------------------
# T3 — is the credit lag long compared with the stagnation horizon?
# ---------------------------------------------------------------------------------------
STAGNATION_HORIZON = {"evmem": 8, "aom": 12, "bolt": 80}


def test_lag_vs_horizon(records) -> dict:
    lags = sorted(r.lag_steps for r in records if r.lag_steps is not None and r.lag_steps >= 0)
    if not lags:
        return {"passes": False, "reason": "no measured lags"}
    median = lags[len(lags) // 2]
    out: dict = {"n": len(lags), "mean": round(sum(lags) / len(lags), 2), "median": median,
                 "p90": lags[int(0.9 * (len(lags) - 1))], "max": max(lags),
                 "horizons": STAGNATION_HORIZON}
    for arm, h in STAGNATION_HORIZON.items():
        out[f"frac_lag_gt_horizon_{arm}"] = round(sum(1 for x in lags if x > h) / len(lags), 3)
    out["median_exceeds_all_horizons"] = median > max(STAGNATION_HORIZON.values())
    out["passes"] = (out["median_exceeds_all_horizons"]
                     or out["frac_lag_gt_horizon_aom"] > 0.5)
    return out


# ---------------------------------------------------------------------------------------
# T4 — is the score bimodal, i.e. is the task "escape 0.0"?
# ---------------------------------------------------------------------------------------
def test_bimodality() -> dict:
    """Score mass at exactly 0.0 and at the top, versus the interior.

    This is the design driver, not a curiosity: if most of the variance lives in a point
    mass at 0.0, then every increment of mean score must come from converting zeros, and
    the correct KPI is P(escape 0.0) rather than mean score.
    """
    allv: list[float] = []
    per_run = []
    for arm in CANONICAL:
        for t, tag in CANONICAL[arm].items():
            v = _trial_scores(tag)
            if not v:
                continue
            allv.extend(v)
            per_run.append({"arm": arm, "task": t, "tag": tag, "n": len(v),
                            "zeros": sum(1 for x in v if x == 0.0),
                            "tops": sum(1 for x in v if x >= 66.6),
                            "interior": sum(1 for x in v if 0.0 < x < 66.6)})
    if not allv:
        return {"passes": False, "reason": "no scores"}
    zeros = sum(1 for x in allv if x == 0.0)
    interior = sum(1 for x in allv if 0.0 < x < 66.6)
    tops = sum(1 for x in allv if x >= 66.6)
    n = len(allv)
    # gap statistic: the largest empty band between adjacent order statistics, normalised
    s = sorted(set(allv))
    gaps = [(s[i + 1] - s[i], s[i], s[i + 1]) for i in range(len(s) - 1)]
    big = max(gaps) if gaps else (0.0, 0, 0)
    return {
        "n": n, "zero_frac": round(zeros / n, 3), "top_frac": round(tops / n, 3),
        "interior_frac": round(interior / n, 3),
        "distinct_values": [round(x, 2) for x in s],
        "largest_gap": {"size": round(big[0], 2), "between": [round(big[1], 2), round(big[2], 2)]},
        "per_run": per_run,
        "passes": (zeros / n) >= 0.25 and big[0] >= 15.0,
        "implication": ("mean score is a poor objective; the KPI is P(escape 0.0) and the "
                        "scheduler should be judged on how many seeds leave the zero mass"),
    }


def main() -> int:
    print("=" * 78)
    print("DIAL falsification suite")
    print("=" * 78)
    recs = extract_all(RESULTS, tasks=TASKS)
    print(f"\narchive: {len(recs)} attributed attempts "
          f"({sum(1 for r in recs if r.outcome == 'credited')} credited)")

    sc = canonical_scores()
    print("\ncanonical archived runs (explicit selection, no string-sort inference):")
    for arm, per_task in sorted(sc.items()):
        vals = [v["mean"] for v in per_task.values()]
        tot = [x for v in per_task.values() for x in v["trials"]]
        print(f"  {arm:6s} mean={sum(vals)/len(vals):6.2f}  zeros={sum(1 for x in tot if x==0.0)}/{len(tot)}"
              + "".join(f"  t{t}={v['mean']:5.1f}(z{v['zeros']}/{v['n']})"
                        for t, v in sorted(per_task.items())))

    r1 = test_attempt_predicts_credit(recs)
    r2 = test_gate_tightness()
    r3 = test_lag_vs_horizon(recs)
    r4 = test_bimodality()

    print("\n" + "-" * 78)
    print(f"T1 attempt predicts credit ............ {'PASS' if r1['passes'] else 'FAIL'}")
    print(f"    chi2={r1['chi2_obs']}  permutation p={r1['p_permutation']:.4g}  cells={r1['n_cells']}")
    print(f"    held-out loglik/ep: family={r1['loglik_family_only']:.5f}  "
          f"+cell={r1['loglik_family_plus_cell']:.5f}  gain={r1['cell_gain_nats']:+.5f} nats")
    for fam, rows in r1["within_family_contrast"].items():
        best = max(rows.items(), key=lambda kv: kv[1]["rate"])
        worst = min(rows.items(), key=lambda kv: kv[1]["rate"])
        print(f"      {fam:6s} best={best[0]}({best[1]['rate']}) vs worst={worst[0]}({worst[1]['rate']})")

    print(f"\nT2 gate tightness matters ............. {'PASS' if r2['passes'] else 'FAIL'}")
    print("    cross-arm (consistent, but confounded by architecture):")
    for row in r2["cross_arm"]["rows"]:
        print(f"      {row['arm']:6s} gates={row['gate_conditions']:2d} mean={row['mean']:6.2f} "
              f"zeros={row['zero_frac']:.2f}")
    print(f"      pearson(gates,mean)={r2['cross_arm']['pearson_gates_vs_mean']} "
          f"monotone_decreasing={r2['cross_arm']['strictly_monotone_decreasing']}")
    print(f"      confound: {r2['cross_arm']['confound']}")
    print("    within-arm ladder, task 8, architecture fixed:")
    for row in r2["within_arm"]["ladder"]:
        print(f"      {row['version']:3s} n={row['n']:2d} mean={row['mean']:6.2f} "
              f"zeros={row['zero_frac']:.2f}  {row['change'][:52]}")
    print(f"      mean spread across ladder = {r2['within_arm']['mean_spread']} points")

    print(f"\nT3 credit lag vs horizon .............. {'PASS' if r3['passes'] else 'FAIL'}")
    print(f"    n={r3.get('n')} mean={r3.get('mean')} median={r3.get('median')} "
          f"p90={r3.get('p90')} max={r3.get('max')}")
    for arm, h in STAGNATION_HORIZON.items():
        print(f"    horizon {arm:6s}={h:3d}: {r3.get(f'frac_lag_gt_horizon_{arm}',0)*100:5.1f}% of "
              f"successful attempts still in flight -> would be cancelled")

    print(f"\nT4 score is bimodal (task is 'escape 0.0') {'PASS' if r4['passes'] else 'FAIL'}")
    print(f"    n={r4['n']} zero={r4['zero_frac']:.2f} top(>=66.6)={r4['top_frac']:.2f} "
          f"interior={r4['interior_frac']:.2f}")
    print(f"    distinct values: {r4['distinct_values']}")
    print(f"    largest empty gap: {r4['largest_gap']['size']} between "
          f"{r4['largest_gap']['between']}")

    report = {"n_records": len(recs), "canonical": {
        a: {str(t): {k: v for k, v in d.items()} for t, d in pt.items()}
        for a, pt in sc.items()}, "T1": r1, "T2": r2, "T3": r3, "T4": r4}
    out = Path(__file__).resolve().parent / "falsify_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out}")

    verdicts = {"T1": r1["passes"], "T2": r2["passes"], "T3": r3["passes"], "T4": r4["passes"]}
    print("=" * 78)
    for k, ok in verdicts.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
