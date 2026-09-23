#!/usr/bin/env python3
"""Audit the VoLo baselines for contamination by the `stage_mapper` tie-break defect.

WHY THIS EXISTS
---------------
`expected_primitive_for_stage` (rule 2, the fuzzy path) scored a label by counting how many
of its >2-char tokens appear as substrings of the active stage name, then kept the best with a
STRICT `>` comparison. Two consequences:

  * `open middle drawer` scores 2 against stage `02_Place_Cookies_Middle_Drawer`
    ("middle" and "drawer" are substrings of the stage name; "open" is not).
  * `place cookies` also scores 2. Ties keep the EARLIER label in `primitive_order`, which is
    `open middle drawer`.

Rule 1 (index alignment) only applies when `len(primitive_labels) == len(stage_specs)`, which
holds for tasks 4/5 (9 vs 9) and NOT for the self-referential family (4 vs 6, 3 vs 6). The
microwave tasks survive the fuzzy path by luck: the stage name `02_Place_Cookies_Microwave`
shares no token with `open microwave`, so the open label scores 1 and is rejected by the
`hit >= min(2, len(tokens))` gate.

The consequence in a run is that the stall-recovery ladder proposes `open the drawer` while the
robot is stuck on a `place` stage -- i.e. it sends the robot BACK to an already-completed stage.

This script reads the frozen baseline output trees and reports, per task:
  * whether the tied fuzzy path is even reachable (the structural precondition),
  * the primitive a CORRECT matcher would propose for the stalled stage,
  * the primitives the ladder ACTUALLY proposed,
  * hence whether the baseline's recovery channel was correct or mis-directed.

It is read-only: it never writes to the baseline trees.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERBS = ("open", "close", "pick", "place", "pour", "put", "push", "pull", "reach", "grasp", "lift")


def _primitive_for_stage(labels: list[str], stage_idx: int) -> str | None:
    """Index alignment, i.e. the mapping the BDDL `primitive_order` is written for."""
    if 0 <= stage_idx < len(labels):
        return labels[stage_idx]
    return None


def _stage_index(stage_specs: list) -> dict[str, int]:
    return {str(s.name): i for i, s in enumerate(stage_specs)}


def _load_tasks(tasks_json: Path) -> dict[int, dict]:
    cfg = json.loads(tasks_json.read_text(encoding="utf-8"))
    return {int(t["task_id"]): t for t in cfg["tasks"]}


def _overrides(ep_dir: Path) -> list[str]:
    """Every `harness subtask override` value logged for one episode, in order."""
    log = ep_dir / "sync_vlm.log"
    if not log.exists():
        return []
    out = []
    for line in log.read_text(errors="ignore").splitlines():
        if "harness subtask override" in line:
            out.append(line.split("override:", 1)[1].strip())
    return out


def _attempt_records(ep_dir: Path) -> list[dict]:
    """`attempts[]` from every attempt's harness_memory.json, normalised."""
    recs = []
    for att in sorted(ep_dir.glob("attempt*")):
        hm = att / "harness_memory.json"
        if not hm.exists():
            continue
        try:
            d = json.loads(hm.read_text(encoding="utf-8"))
        except Exception:
            continue
        for a in d.get("attempts", []) or []:
            recs.append(a)
    return recs


def analyse(run_dir: Path, tasks: dict[int, dict], spec_fn, *, label: str) -> dict:
    """Episodes live at <run>/<arm>_s<seed>/task<N>/ep<M>/ (the log is one level below ep)."""
    episodes = sorted(
        ep for task in run_dir.iterdir() if task.is_dir() and task.name.startswith("task")
        for ep in sorted(task.iterdir()) if ep.is_dir() and ep.name.startswith("ep")
    )
    report = {"run": label, "tasks": {}}
    for ep in episodes:
        tid = int(ep.parent.name.replace("task", ""))
        labels = [str(p["label"]) for p in tasks[tid]["primitive_order"]]
        specs = spec_fn(tid)
        idx_of = _stage_index(specs)
        tied_path_reachable = len(labels) != len(specs)

        ov = _overrides(ep)
        recs = _attempt_records(ep)
        # `stalled_stage` per attempt, de-duplicated while preserving order.
        stalled = [r.get("stalled_stage") for r in recs if r.get("stalled_stage")]
        expected, misdirected = [], []
        for st in dict.fromkeys(stalled):
            i = idx_of.get(st)
            if i is None:
                continue
            correct = _primitive_for_stage(labels, i)
            if correct is not None and correct not in ov:
                expected.append((st, correct))

        t = report["tasks"].setdefault(tid, {
            "n_episodes": 0, "n_overrides": 0, "override_values": {},
            "tied_fuzzy_path_reachable": tied_path_reachable,
            "n_labels": len(labels), "n_stages": len(specs),
            "stalled_stages": {}, "expected_but_absent": [],
        })
        t["n_episodes"] += 1
        t["n_overrides"] += len(ov)
        for v in ov:
            t["override_values"][v] = t["override_values"].get(v, 0) + 1
        for st in stalled:
            t["stalled_stages"][st] = t["stalled_stages"].get(st, 0) + 1
        t["expected_but_absent"].extend(expected)

    # De-duplicate the "expected but absent" list per task.
    for t in report["tasks"].values():
        seen, uniq = set(), []
        for st, prim in t["expected_but_absent"]:
            if (st, prim) in seen:
                continue
            seen.add((st, prim))
            uniq.append({"stalled_stage": st, "correct_primitive": prim})
        t["expected_but_absent"] = uniq
        t["contaminated"] = bool(uniq)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks-json", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation_benchmark" / "scripts"))
    import task2_26_reference_stage as M  # noqa: N812

    tasks = _load_tasks(args.tasks_json)
    reports = [analyse(args.results_root / r / "nomem_s100", tasks, M._task_specs, label=r)
               for r in args.runs]

    print("=" * 104)
    print("  VoLo 基线污染审计 · 恢复通道是否被 stage_mapper 平局缺陷误导")
    print("=" * 104)
    print(f"  {'run':<26s} {'task':>5s} {'stages/labels':>14s} {'模糊路径可达':>13s} "
          f"{'override':>9s} {'污染':>6s}")
    print("  " + "-" * 100)
    any_contam = False
    for rep in reports:
        for tid, t in sorted(rep["tasks"].items()):
            any_contam |= t["contaminated"]
            print(f"  {rep['run']:<26s} {tid:>5d} {t['n_stages']:>6d}/{t['n_labels']:<7d} "
                  f"{'是' if t['tied_fuzzy_path_reachable'] else '否(索引对齐)':>13s} "
                  f"{t['n_overrides']:>9d} {'❌ 是' if t['contaminated'] else '✅ 否':>6s}")
    print("  " + "-" * 100)
    for rep in reports:
        bad = {tid: t for tid, t in rep["tasks"].items() if t["contaminated"]}
        if not bad:
            continue
        print(f"\n  【{rep['run']}】被污染的 task 明细：")
        for tid, t in sorted(bad.items()):
            print(f"    task{tid}: 卡住的 stage 及本应提出的恢复 primitive ——")
            for e in t["expected_but_absent"]:
                print(f"        stalled={e['stalled_stage']!r:44s} 正确恢复应是 {e['correct_primitive']!r}"
                      f"  ← 未出现")
            top = sorted(t["override_values"].items(), key=lambda kv: -kv[1])[:4]
            print(f"        实际 override: {top}")
    print()
    print("=" * 104)
    print("  结论:", "❌ 基线存在被污染的族，不能作为干净基线发布" if any_contam
          else "✅ 未发现污染")
    print("=" * 104)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"runs": reports, "any_contaminated": any_contam},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写出: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
