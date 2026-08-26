#!/usr/bin/env bash
# Merge task1 + tasks2to26 eval outputs into one 26-task summary (official macro CSR/TSR).
set -euo pipefail

OUT_ROOT="${1:?usage: merge_eval_outputs.sh OUT_ROOT}"

python3 - "${OUT_ROOT}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
GROUPS = [
    ("task1", {1}),
    ("tasks2to26", set(range(2, 27))),
]


def rows_from_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rows_from_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return [
        {
            "task_id": str(r["task_id"]),
            "status": r.get("status", ""),
            "error": r.get("error", ""),
            "stage_score_pct": str(r.get("stage_score_pct", "")),
            "stage_success_rate": str(r.get("stage_success_rate", "")),
            "goal_success_rate": str(r.get("goal_success_rate", "")),
            "video_dir": r.get("video_dir", ""),
            "duration_sec": str(r.get("duration_sec", "")),
        }
        for r in data
    ]


def filter_rows(rows: list[dict], task_ids: set[int]) -> list[dict]:
    out = []
    for row in rows:
        try:
            tid = int(row["task_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if tid in task_ids:
            out.append(row)
    return out


def load_group_rows(group_dir: Path, task_ids: set[int]) -> list[dict]:
    candidates: list[list[dict]] = []
    for name in ("summary.tsv", "summary.json"):
        path = group_dir / name
        if not path.is_file():
            continue
        try:
            rows = rows_from_tsv(path) if path.suffix == ".tsv" else rows_from_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        filtered = filter_rows(rows, task_ids)
        if filtered:
            candidates.append(filtered)

    for name in ("summary.tsv", "summary.json"):
        path = out_root / name
        if not path.is_file():
            continue
        try:
            rows = rows_from_tsv(path) if path.suffix == ".tsv" else rows_from_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        filtered = filter_rows(rows, task_ids)
        if filtered:
            candidates.append(filtered)

    if candidates:
        return max(candidates, key=len)

    agg_path = group_dir / "aggregate.json"
    if agg_path.is_file() and task_ids == {1}:
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        if int(agg.get("num_tasks", 0)) == 1:
            duration = ""
            log_path = group_dir / "logs" / "eval_fullvlm26_async_vlm_vla.log"
            if log_path.is_file():
                m = re.search(r"\|\s*1/1\s*\[[^\]]*?,\s*([\d.]+)s/it\]", log_path.read_text(encoding="utf-8"))
                if m:
                    duration = m.group(1)
            return [{
                "task_id": "1",
                "status": "completed",
                "error": "",
                "stage_score_pct": str(agg.get("macro_stage_score_pct", 0.0)),
                "stage_success_rate": str(agg.get("macro_stage_success_rate", 0.0)),
                "goal_success_rate": str(agg.get("macro_goal_success_rate", 0.0)),
                "video_dir": str(out_root / "videos" / "task1"),
                "duration_sec": duration,
            }]
    return []


def load_group_traces(group_dir: Path, task_ids: set[int]) -> list[str]:
    traces: list[str] = []
    for path in (group_dir / "prompt_trace.tsv", out_root / "prompt_trace.tsv"):
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= 1:
            continue
        for line in lines[1:]:
            tid = line.split("\t", 1)[0]
            try:
                if int(tid) in task_ids:
                    traces.append(line)
            except ValueError:
                continue
    return traces


rows: list[dict] = []
traces: list[str] = []
for group_name, task_ids in GROUPS:
    group_dir = out_root / group_name
    rows.extend(load_group_rows(group_dir, task_ids))
    traces.extend(load_group_traces(group_dir, task_ids))

if not rows:
    raise SystemExit(f"[ERR] no summary rows under {out_root}")

header = (
    "task_id\tstatus\terror\tstage_score_pct\tstage_success_rate\t"
    "goal_success_rate\tvideo_dir\tduration_sec\n"
)
merged_summary = out_root / "summary.tsv"
with merged_summary.open("w", encoding="utf-8") as f:
    f.write(header)
    for r in sorted(rows, key=lambda x: int(x["task_id"])):
        f.write(
            f"{r['task_id']}\t{r['status']}\t{r['error']}\t{r['stage_score_pct']}\t"
            f"{r['stage_success_rate']}\t{r['goal_success_rate']}\t{r['video_dir']}\t{r['duration_sec']}\n"
        )

trace_header = (
    "task_id\ttrial\tseed\tvlm_ckpt\tvla_prompt_last\tstage_success\tgoal_success\t"
    "stage_score_pct\textra_pour_detected\tfailure_reason\n"
)
merged_trace = out_root / "prompt_trace.tsv"
with merged_trace.open("w", encoding="utf-8") as f:
    f.write(trace_header)
    for line in sorted(traces, key=lambda ln: int(ln.split("\t", 1)[0])):
        f.write(line + "\n")

completed = [r for r in rows if r["status"] == "completed"]
aggregate = {
    "macro_stage_score_pct": sum(float(r["stage_score_pct"]) for r in completed) / max(1, len(completed)),
    "macro_stage_success_rate": sum(float(r["stage_success_rate"]) for r in completed) / max(1, len(completed)),
    "macro_goal_success_rate": sum(float(r["goal_success_rate"]) for r in completed) / max(1, len(completed)),
    "num_tasks": len(rows),
    "num_goal_scored_tasks": len(completed),
}
(out_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
(out_root / "summary.json").write_text(
    json.dumps(
        [
            {
                "task_id": int(r["task_id"]),
                "status": r["status"],
                "error": r["error"],
                "stage_score_pct": float(r["stage_score_pct"]),
                "stage_success_rate": float(r["stage_success_rate"]),
                "goal_success_rate": float(r["goal_success_rate"]),
                "video_dir": r["video_dir"],
                "duration_sec": float(r["duration_sec"]) if r["duration_sec"] else 0.0,
            }
            for r in sorted(rows, key=lambda x: int(x["task_id"]))
        ],
        indent=2,
    ),
    encoding="utf-8",
)

print(f"[OK] merged {len(rows)} tasks -> {merged_summary}")
print(f"[OK] aggregate.json CSR={aggregate['macro_goal_success_rate']:.4f} "
      f"TSR={aggregate['macro_stage_success_rate']:.4f}")
PY
