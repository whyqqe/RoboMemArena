#!/usr/bin/env bash
# Aggregate dual-track experiment: memory vs harness potentials.
# Usage: aggregate_split_tracks.sh OUT_BASE
set -euo pipefail

OUT_BASE="${1:?usage: aggregate_split_tracks.sh OUT_BASE}"

python3 - "${OUT_BASE}" <<'PY'
import json
import math
import sys
from pathlib import Path

out_base = Path(sys.argv[1])


def agg(p: Path) -> dict:
    f = p / "aggregate.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def find_runs(prefix: str) -> list[tuple[int, Path]]:
    runs = []
    for p in sorted(out_base.iterdir()):
        if not p.is_dir() or not p.name.startswith(prefix):
            continue
        try:
            seed = int(p.name.split("_s", 1)[1])
        except (IndexError, ValueError):
            continue
        runs.append((seed, p))
    return sorted(runs)


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


base = find_runs("baseline_s")
if not base:
    print("[ERR] no baseline_s* runs")
    raise SystemExit(1)
seeds = [s for s, _ in base]
base_csr = {s: agg(p).get("macro_goal_success_rate", 0) for s, p in base}
base_tsr = {s: agg(p).get("macro_stage_success_rate", 0) for s, p in base}

tracks = {
    "memory_kf": "memory_kf_s",
    "memory_ctx": "memory_ctx_s",
    "memory_plus": "memory_plus_s",
    "harness_exec": "harness_exec_s",
    "harness_v21": "harness_v21_s",
}

print("# Dual-Track Experiment Summary")
print(f"\nSeeds: {seeds}")
bm, bs = mean_std(list(base_csr.values()))
btm, bts = mean_std(list(base_tsr.values()))
print(f"\nBaseline: CSR {bm:.3f}±{bs:.3f}  TSR {btm:.3f}±{bts:.3f}")
print("\n## Memory track (no retry)")
print("| Variant | CSR mean±std | ΔCSR | TSR mean±std |")
print("|---------|--------------|------|--------------|")

memory_results = {}
for name, prefix in list(tracks.items())[:3]:
    runs = find_runs(prefix)
    csrs = [agg(p).get("macro_goal_success_rate", 0) for _, p in runs]
    tsrs = [agg(p).get("macro_stage_success_rate", 0) for _, p in runs]
    deltas = []
    for s, (_, p) in zip([x for x, _ in runs], runs):
        if s in base_csr:
            deltas.append(agg(p).get("macro_goal_success_rate", 0) - base_csr[s])
    if not csrs:
        continue
    m, st = mean_std(csrs)
    dm, _ = mean_std(deltas) if deltas else (0.0, 0.0)
    tm, _ = mean_std(tsrs)
    print(f"| {name} | {m:.3f}±{st:.3f} | {dm:+.3f} | {tm:.3f} |")
    memory_results[name] = {"csr_mean": m, "csr_std": st, "delta_csr": dm, "tsr_mean": tm}

print("\n## Harness track (execution layer)")
print("| Variant | CSR mean±std | ΔCSR | TSR mean±std |")
print("|---------|--------------|------|--------------|")

harness_results = {}
for name, prefix in list(tracks.items())[3:]:
    runs = find_runs(prefix)
    csrs = [agg(p).get("macro_goal_success_rate", 0) for _, p in runs]
    tsrs = [agg(p).get("macro_stage_success_rate", 0) for _, p in runs]
    deltas = []
    for s, (_, p) in zip([x for x, _ in runs], runs):
        if s in base_csr:
            deltas.append(agg(p).get("macro_goal_success_rate", 0) - base_csr[s])
    if not csrs:
        continue
    m, st = mean_std(csrs)
    dm, _ = mean_std(deltas) if deltas else (0.0, 0.0)
    tm, _ = mean_std(tsrs)
    print(f"| {name} | {m:.3f}±{st:.3f} | {dm:+.3f} | {tm:.3f} |")
    harness_results[name] = {"csr_mean": m, "csr_std": st, "delta_csr": dm, "tsr_mean": tm}

summary = {
    "seeds": seeds,
    "baseline": {"csr_mean": bm, "csr_std": bs, "tsr_mean": btm, "tsr_std": bts},
    "memory_track": memory_results,
    "harness_track": harness_results,
}
(out_base / "dual_track_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\n[OK] wrote {out_base / 'dual_track_summary.json'}")
PY
