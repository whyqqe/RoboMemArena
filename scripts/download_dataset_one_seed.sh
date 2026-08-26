#!/usr/bin/env bash
# Download RoboMemArena: 1 seed per task, subtask_data only (ModelScope selective).
# HF metadata for planning; files pulled from ModelScope (HF download is gated).
# Estimated: ~148 hdf5, ~8-12 GB. Does NOT download full_trajectory/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_common.sh"
robomem_setup_cache
# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_eval.sh"

DATA_DIR="${DATA_DIR:-${ROOT}/data/RoboMemArena}"
LOG="${ROOT}/outputs/logs/download_one_seed.log"
mkdir -p "${ROOT}/outputs/logs" "${DATA_DIR}"

python -m pip install -q modelscope "huggingface_hub>=0.26.0"

echo "[START] one-seed subtask download (ModelScope) $(date)" | tee "${LOG}"
echo "[INFO] DATA_DIR=${DATA_DIR}" | tee -a "${LOG}"

python -u - <<'PY' 2>&1 | tee -a "${LOG}"
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import list_repo_files
from modelscope.hub.file_download import dataset_file_download

REPO_HF = "RoboMemArenaBenchmark/RoboMemArena"
DATA_DIR = Path(os.environ.get("DATA_DIR", "/project/peilab/why/RoboMemArena/data/RoboMemArena"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP = {
    "Multi-Counting": ("haodong123/RoboMemArena-Multi-Object-Counting", "Multi-Object_Counting"),
    "Multi-Occlusion": ("haodong123/RoboMemArena-Multi-Object-Occlusion", "Multi-Object_Occlusion"),
    "Multi-Sequence": ("haodong123/RoboMemArena-Multi-Object-Sequence", "Multi-Object_Sequence"),
    "Multi-Transferring": ("haodong123/RoboMemArena-Multi-Object-Transferring", "Multi-Object_Transferring"),
}

print("[INFO] building plan from HF file listing...")
hdf5 = [
    f for f in list(list_repo_files(REPO_HF, repo_type="dataset"))
    if f.endswith(".hdf5") and "/subtask_data/" in f
]

by_task: dict[int, set[int]] = defaultdict(set)
for f in hdf5:
    m = re.search(r"_seed(\d+)_task(\d+)\.hdf5$", f)
    if m:
        by_task[int(m.group(2))].add(int(m.group(1)))

jobs: list[tuple[str, str]] = []
for tid in sorted(by_task):
    pick = min(by_task[tid])
    for hf_path in hdf5:
        if not hf_path.endswith(f"_seed{pick}_task{tid}.hdf5"):
            continue
        hf_cat, rest = hf_path.split("/", 1)
        ms_id, ms_root = CATEGORY_MAP[hf_cat]
        jobs.append((ms_id, f"{ms_root}/{rest}"))

print(f"[INFO] {len(jobs)} files to download (1 seed/task, subtask_data only)")

ok, skip, fail = 0, 0, 0
total_bytes = 0
for i, (ms_id, ms_path) in enumerate(jobs, 1):
    dest = DATA_DIR / ms_path
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[SKIP] {i}/{len(jobs)} {ms_path}")
        skip += 1
        total_bytes += dest.stat().st_size
        continue
    try:
        out = dataset_file_download(
            dataset_id=ms_id,
            file_path=ms_path,
            local_dir=str(DATA_DIR),
        )
        sz = Path(out).stat().st_size
        total_bytes += sz
        print(f"[OK] {i}/{len(jobs)} {ms_path} ({sz/1e6:.1f} MB)")
        ok += 1
    except Exception as exc:
        print(f"[FAIL] {i}/{len(jobs)} {ms_id}::{ms_path}: {exc}", file=sys.stderr)
        fail += 1

print(f"[DONE] ok={ok} skip={skip} fail={fail} total={len(jobs)}")
print(f"[SIZE] ~{total_bytes/1e9:.2f} GB under {DATA_DIR}")
if fail:
    sys.exit(1)
PY

echo "[DONE] $(date)" | tee -a "${LOG}"
du -sh "${DATA_DIR}" | tee -a "${LOG}"
find "${DATA_DIR}" -name '*.hdf5' | wc -l | awk '{print "hdf5 count:", $1}' | tee -a "${LOG}"
