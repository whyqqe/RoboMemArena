#!/usr/bin/env bash
# Download PrediMem weights and RoboMemArena dataset into the project tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_common.sh"
robomem_setup_cache

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_eval.sh" 2>/dev/null || true
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv/bin/activate"
  fi
fi

python -m pip install -q "huggingface_hub>=0.26.0"

DATA_DIR="${DATA_DIR:-${ROOT}/data/RoboMemArena}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/PrediMem}"
PREDIMEM_REVISION="${PREDIMEM_REVISION:-e645741c2f34f27e1596bfa89e856d6f3560ed90}"

echo "[INFO] Dataset  -> ${DATA_DIR}"
echo "[INFO] PrediMem -> ${CKPT_DIR} (revision ${PREDIMEM_REVISION})"

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD=(huggingface-cli download)
else
  HF_DOWNLOAD=(python -m huggingface_hub.cli.huggingface_cli download)
fi

"${HF_DOWNLOAD[@]}" --repo-type dataset RoboMemArenaBenchmark/RoboMemArena \
  --local-dir "${DATA_DIR}" || {
  echo "[WARN] HF dataset download failed (gated or network). Falling back to ModelScope..."
  bash "${ROOT}/scripts/download_dataset_modelscope.sh"
}

"${HF_DOWNLOAD[@]}" huashuolei/PrediMem \
  --revision "${PREDIMEM_REVISION}" \
  --local-dir "${CKPT_DIR}"

cat <<EOF
[OK] Downloads finished.
  dataset: ${DATA_DIR}
  weights: ${CKPT_DIR}

For reference eval, point to the local snapshot:
  export PREDIMEM_HF_SNAPSHOT=${CKPT_DIR}
  export VLA_CKPT=${CKPT_DIR}/vla_alltask
EOF
