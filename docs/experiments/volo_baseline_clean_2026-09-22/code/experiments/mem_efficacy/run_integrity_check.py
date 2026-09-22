#!/usr/bin/env python3
"""Detect runs that are internally inconsistent because two writers shared one output dir.

WHY THIS IS NEEDED, AND WHY A SCORE CHECK CANNOT REPLACE IT
-----------------------------------------------------------
Two Slurm jobs were once submitted against the SAME output directory (596262 and 596284). Each
opened `sync_vlm.log` in append mode. Their writes interleaved, so a single episode directory
can contain:

  * TWO `Episode N seed=N` records with the SAME seed but DIFFERENT stage_score, and
  * timestamps that run BACKWARDS (a line with an earlier wall-clock time appears after a later
    one), which is the fingerprint of two append offsets in one file.

The JSON artifacts (`harness_memory.json`, `api_vlm_trace.jsonl`, the mp4s) come from whichever
writer finished last, so they belong to ONE of the two records -- not necessarily the one the
summary reports. `summary.json` ends up with a duplicated task row, and any macro mean taken
over it weights that task twice.

None of this is visible in the headline score, and a score that happens to look plausible is
exactly the failure mode this check exists to catch.

WHAT COUNTS AS A FAILURE
  1. more than one `Episode <k> seed=<s>` record for a single `<task>/ep<k>` directory
  2. non-monotonic timestamps inside one `sync_vlm.log`
  3. `summary.json` carrying a duplicate `task_id`, or a task_id with no directory

Exit code 0 = every run checked is internally consistent. 1 = at least one run is NOT.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
EP = re.compile(r"Episode (\d+) seed=(\d+) stage_score=([\d.]+)")
GET_SEED = re.compile(r"seed=[0-9]+")


def _ts_key(line: str) -> tuple[str, str] | None:
    m = TS.match(line)
    return (m.group(1), m.group(2)) if m else None


def check_episode_dir(ep_dir: Path) -> dict:
    log = ep_dir / "sync_vlm.log"
    if not log.exists():
        return {"records": [], "backwards": 0, "issue": "no sync_vlm.log"}
    episodes, last, backwards = [], None, 0
    for line in log.read_text(errors="ignore").splitlines():
        m = EP.search(line)
        if m:
            episodes.append({"ep": int(m.group(1)), "seed": int(m.group(2)),
                             "stage_score": float(m.group(3))})
        k = _ts_key(line)
        if k:
            if last is not None and k < last:
                backwards += 1
            last = k
    issue = ""
    if len(episodes) > 1:
        seeds = {e["seed"] for e in episodes}
        scores = {e["stage_score"] for e in episodes}
        issue = (f"MULTIPLE Episode records ({len(episodes)})"
                 + (f", same seed {sorted(seeds)}" if len(seeds) == 1 else "")
                 + (f", differing scores {sorted(scores)}" if len(scores) > 1 else ""))
    elif backwards:
        issue = f"timestamps run backwards {backwards}x"
    return {"records": episodes, "backwards": backwards, "issue": issue}


def check_run(run_dir: Path) -> dict:
    """run_dir is the arm dir, e.g. <base>/<run>/nomem_s100."""
    report = {"run_dir": str(run_dir), "tasks": {}, "summary_issue": "", "clean": True}
    for task in sorted(p for p in run_dir.iterdir()
                       if p.is_dir() and p.name.startswith("task")):
        t = {"eps": {}, "n_ep_dirs": 0, "bad": []}
        for ep in sorted(p for p in task.iterdir() if p.is_dir() and p.name.startswith("ep")):
            t["n_ep_dirs"] += 1
            r = check_episode_dir(ep)
            if r["issue"]:
                t["bad"].append({"ep": ep.name, **r})
        report["tasks"][int(task.name.replace("task", ""))] = t

    sj = run_dir / "summary.json"
    if sj.exists():
        try:
            rows = json.loads(sj.read_text(encoding="utf-8"))
            ids = [r["task_id"] for r in rows]
            dups = sorted({i for i in ids if ids.count(i) > 1})
            missing = sorted(set(report["tasks"]) - set(ids))
            extra = sorted(set(ids) - set(report["tasks"]))
            if dups or missing or extra:
                report["summary_issue"] = (f"summary.json rows={len(rows)} unique="
                                           f"{len(set(ids))}; duplicated={dups} "
                                           f"no_dir={extra} not_in_summary={missing}")
        except Exception as exc:  # noqa: BLE001
            report["summary_issue"] = f"unreadable: {exc}"

    report["clean"] = (not report["summary_issue"]
                       and all(not t["bad"] for t in report["tasks"].values()))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True,
                    help="arm dirs, e.g. experiments/mem_efficacy/results/<run>/nomem_s100")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    reports = [check_run(r) for r in args.runs]
    print("=" * 100)
    print("  运行完整性检查：同一输出目录是否被两个写入者共享（作业碰撞）")
    print("=" * 100)
    bad_runs = 0
    for rep in reports:
        name = "/".join(Path(rep["run_dir"]).parts[-2:])
        print(f"\n  ── {name} ──")
        if rep["summary_issue"]:
            print(f"     ❌ summary.json: {rep['summary_issue']}")
        for tid, t in sorted(rep["tasks"].items()):
            mark = "✅" if not t["bad"] else "❌"
            line = f"     {mark} task{tid}: {t['n_ep_dirs']} 个 ep 目录"
            if t["bad"]:
                line += f"，其中 {len(t['bad'])} 个异常"
            print(line)
            for b in t["bad"]:
                print(f"          {b['ep']}: {b['issue']}")
                if len(b.get("records", [])) > 1:
                    print(f"             记录: {[(r['seed'], r['stage_score']) for r in b['records']]}")
        verdict = "✅ 内部自洽" if rep["clean"] else "❌ 不自洽 —— 不可发布"
        print(f"     判定: {verdict}")
        if not rep["clean"]:
            bad_runs += 1
    print()
    print("=" * 100)
    print(f"  结论: {len(reports)-bad_runs}/{len(reports)} 个运行内部自洽")
    if bad_runs:
        print("        不自洽的运行来自「两个作业写同一输出目录」，其 episode 级 JSON/视频")
        print("        无法确定归属哪个记录，summary 也已重复计数 —— 必须重跑，不能发布。")
    print("=" * 100)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"runs": reports}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"  已写出: {args.out}")
    return 1 if bad_runs else 0


if __name__ == "__main__":
    raise SystemExit(main())
