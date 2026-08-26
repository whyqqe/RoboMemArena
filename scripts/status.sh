#!/usr/bin/env bash
# Quick status for env + downloads.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== venvs ==="
[[ -x "${ROOT}/.venv/bin/python" ]] && echo "eval:  OK (${ROOT}/.venv)" || echo "eval:  MISSING"
[[ -x "${ROOT}/third_party/openpi_minimal/.venv/bin/python" ]] && echo "vla:   OK" || echo "vla:   MISSING"
echo "=== PrediMem ==="
du -sh "${ROOT}/checkpoints/PrediMem" 2>/dev/null || echo "not downloaded"
test -f "${ROOT}/checkpoints/PrediMem/vla_alltask/params/_METADATA" && echo "vla_alltask: OK"
test -f "${ROOT}/checkpoints/PrediMem/vlm_task1/model.safetensors" && echo "vlm_task1: OK"
test -d "${ROOT}/checkpoints/PrediMem/vlm_tasks1to26_ckpt74500" && echo "vlm_tasks2-26: OK"
echo "=== dataset (ModelScope categories) ==="
for d in Multi-Object-Counting Multi-Object-Occlusion Multi-Object-Sequence Multi-Object-Transferring; do
  path="${ROOT}/data/RoboMemArena/${d}"
  if [[ -d "${path}" ]]; then
    n=$(find "${path}" -name '*.hdf5' 2>/dev/null | wc -l)
    du -sh "${path}" 2>/dev/null | awk -v n="$n" -v d="$d" '{print $1, "hdf5=" n, d}'
  else
    echo "MISSING ${d}"
  fi
done
echo "=== active downloads ==="
pgrep -af "modelscope download|hf download" 2>/dev/null | grep -v pgrep || echo "none"
