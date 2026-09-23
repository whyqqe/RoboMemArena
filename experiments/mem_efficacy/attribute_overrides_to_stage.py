#!/usr/bin/env python3
"""Attribute every stall-ladder override in a run to the ACTIVE STAGE, and grade it.

WHY THE FIRST AUDIT WAS NOT ENOUGH
----------------------------------
`attempts[].stalled_stage` is the stage the episode was stuck on at the END. Grading an
override against that is wrong: an override fired early, while the robot was legitimately
stuck on `01_Open_Middle_Drawer`, proposes `open middle drawer` -- which is CORRECT. Only the
overrides that fire AFTER that stage is confirmed can be regressions.

The `episodic` stream in `harness_memory.json` carries the missing piece: each `stall` event
records BOTH the step and the `stage_name` that was in force. So each override can be graded
against the stage that was actually active when it fired.

THE BUG BEING GRADED (hash 40cabe7d, the version the baselines ran)
-------------------------------------------------------------------
Rule 2 of `expected_primitive_for_stage` scored a label by how many of its >2-char tokens are
substrings of the active stage name, keeping the best with a STRICT `>`:

    stage '02_place_cookies_middle_drawer'
      'open middle drawer' -> 'middle','drawer' present          hit=2
      'place cookies'      -> 'place','cookies' present          hit=2  <- TIE
    strict '>' keeps the FIRST on a tie -> 'open middle drawer'

`open middle drawer` is precisely the primitive the robot was ALREADY executing while stalled,
so `_handle_stall`'s `candidate != subtask` guard usually suppressed it; when the subtask had
drifted to something else (e.g. `reach_to_middle_drawer_handle`) the guard passed and the robot
was sent back to a stage that was already complete.

The microwave tasks (`02_Place_Cookies_Microwave`) share no token with `open microwave`, so the
open label scores 1 and is rejected -- they are the within-run control.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

OVR = re.compile(r"\[t=(-?\d+)\]\s+harness subtask override:\s*(.+?)\s*$")
STALL_LOG = re.compile(r"\[t=(-?\d+)\]\s+harness forced VLM replan after stall")
STAGE_DONE_LOG = re.compile(r"\[t=(-?\d+)\]\s+stage done:\s*(\S+)")

VERBS = ("open", "close", "pick", "place", "pour", "put", "push", "pull", "reach", "grasp", "lift")


def _verb(text: str) -> str | None:
    for tok in str(text).lower().replace("_", " ").split():
        if tok in VERBS:
            return tok
    return None


def _correct_primitive(labels: list[str], stage_idx: int, stage_name: str) -> str | None:
    """The mapping the BDDL primitive_order is written for, with the verb-consistency rule.

    Index alignment is the ground truth when the label list is 1:1 with the stages (tasks 4/5).
    Otherwise the intended label is the one that shares the stage's verb -- which is exactly the
    tie-break the defect was missing.
    """
    stage_verb = _verb(stage_name)
    verb_match = [l for l in labels if _verb(l) == stage_verb]
    if stage_verb and verb_match:
        # Prefer the label that also shares the object token, else the first verb match.
        obj = [t for t in str(stage_name).lower().replace("_", " ").split()
               if len(t) > 2 and t not in VERBS]
        for lab in verb_match:
            if any(t in lab.lower() for t in obj):
                return lab
        return verb_match[0]
    if 0 <= stage_idx < len(labels):
        return labels[stage_idx]
    return None


def _log_events(ep_dir: Path) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    log = ep_dir / "sync_vlm.log"
    if not log.exists():
        return [], []
    ovr, done = [], []
    for line in log.read_text(errors="ignore").splitlines():
        m = OVR.search(line)
        if m:
            ovr.append((int(m.group(1)), m.group(2)))
            continue
        m = STAGE_DONE_LOG.search(line)
        if m:
            done.append((int(m.group(1)), m.group(2)))
    return ovr, done


def _stalls(ep_dir: Path) -> list[tuple[int, str]]:
    """(step, stage_name) for every `stall` event, across attempts, in file order."""
    out = []
    for att in sorted(ep_dir.glob("attempt*")):
        hm = att / "harness_memory.json"
        if not hm.exists():
            continue
        try:
            d = json.loads(hm.read_text(encoding="utf-8"))
        except Exception:
            continue
        for e in d.get("episodic", []) or []:
            if e.get("event_type") == "stall" and e.get("stage_name"):
                out.append((int(e.get("step", -1)), str(e["stage_name"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tasks-json", type=Path, required=True)
    ap.add_argument("--tol", type=int, default=8, help="step tolerance when matching override to stall")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg = json.loads(args.tasks_json.read_text(encoding="utf-8"))
    tasks = {int(t["task_id"]): t for t in cfg["tasks"]}

    print("=" * 110)
    print("  stall ladder 逐事件归因：每次 override 是否把机器人推回【已完成】的 stage")
    print("=" * 110)
    print(f"  {'task':>5s} {'override':>9s} {'可归因':>7s} {'正确':>6s} {'回退已完成 stage':>16s} "
          f"{'其它错误':>9s}  {'回归率':>7s}")
    print("  " + "-" * 106)

    totals = Counter()
    per_task = {}
    for task in sorted(p for p in args.run_dir.iterdir()
                       if p.is_dir() and p.name.startswith("task")):
        tid = int(task.name.replace("task", ""))
        labels = [str(p["label"]) for p in tasks[tid]["primitive_order"]]
        ovr_n = attributed = correct = regressed = other = 0
        examples = []

        for ep in sorted(p for p in task.iterdir() if p.is_dir() and p.name.startswith("ep")):
            ovr, done = _log_events(ep)
            stalls = _stalls(ep)
            ovr_n += len(ovr)
            for step, value in ovr:
                # The stall that produced this override: nearest stall event in step.
                near = [s for s in stalls if abs(s[0] - step) <= args.tol]
                if not near:
                    continue
                _, stage_name = min(near, key=lambda s: abs(s[0] - step))
                attributed += 1
                idx = next((i for i, s in enumerate(_spec_names(tid)) if s == stage_name), -1)
                want = _correct_primitive(labels, idx, stage_name)
                if want is None:
                    continue
                if value == want:
                    correct += 1
                else:
                    # Was this stage already confirmed before the override fired?
                    already = any(ds == stage_name and dstep < step for dstep, ds in done)
                    # Regression = proposing a primitive that belongs to an EARLIER stage.
                    earlier = [l for l in labels if _verb(l) != _verb(stage_name)]
                    if _verb(value) is not None and _verb(value) != _verb(stage_name) and earlier:
                        regressed += 1
                        if len(examples) < 2:
                            examples.append((step, stage_name, value, want, already))
                    else:
                        other += 1

        rate = f"{100.0*regressed/attributed:5.1f}%" if attributed else "  n/a"
        print(f"  {tid:>5d} {ovr_n:>9d} {attributed:>7d} {correct:>6d} {regressed:>16d} "
              f"{other:>9d}  {rate}")
        totals["ovr"] += ovr_n
        totals["attr"] += attributed
        totals["correct"] += correct
        totals["regressed"] += regressed
        totals["other"] += other
        per_task[tid] = {"n_overrides": ovr_n, "attributed": attributed, "correct": correct,
                         "regressed_to_other_stage": regressed, "other_wrong": other,
                         "examples": examples}
    print("  " + "-" * 106)
    r = f"{100.0*totals['regressed']/totals['attr']:5.1f}%" if totals["attr"] else "  n/a"
    print(f"  {'ALL':>5s} {totals['ovr']:>9d} {totals['attr']:>7d} {totals['correct']:>6d} "
          f"{totals['regressed']:>16d} {totals['other']:>9d}  {r}")

    print("\n" + "=" * 110)
    print("  典型误导向实例（velocity: 在 place 阶段被推回 open 阶段）")
    print("=" * 110)
    shown = 0
    for tid, d in per_task.items():
        for step, stage, got, want, already in d["examples"]:
            print(f"  task{tid} @t={step}: 活跃 stage={stage}")
            print(f"      ladder 给了 {got!r}   正确应为 {want!r}"
                  f"   {'(该 stage 此前已确认完成!)' if already else ''}")
            shown += 1
    if not shown:
        print("  无")

    print("\n" + "=" * 110)
    fam = {"drawer": [12, 13, 17], "micro": [20, 23], "t4t5": [4, 5]}
    for name, ids in fam.items():
        rr = sum(per_task.get(i, {}).get("regressed_to_other_stage", 0) for i in ids)
        aa = sum(per_task.get(i, {}).get("attributed", 0) for i in ids)
        cc = sum(per_task.get(i, {}).get("correct", 0) for i in ids)
        print(f"  {name:<8s}: 可归因 override {aa:>4d}  正确 {cc:>4d}  误导向 {rr:>4d}"
              f"  ({100.0*rr/aa:.1f}%)" if aa else f"  {name}: 无")
    print("=" * 110)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"per_task": per_task, "totals": dict(totals)},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写出: {args.out}")
    return 0


_SPEC_CACHE: dict[int, list[str]] = {}


def _spec_names(tid: int) -> list[str]:
    return _SPEC_CACHE.get(tid, [])


if __name__ == "__main__":
    # Populate the stage-name cache before main() uses it.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation_benchmark" / "scripts"))
    try:
        import task2_26_reference_stage as M  # noqa: N812
        for _tid in (4, 5, 12, 13, 17, 20, 23):
            _SPEC_CACHE[_tid] = [str(s.name) for s in M._task_specs(_tid)]
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not load stage specs: {exc}", file=sys.stderr)
    raise SystemExit(main())
