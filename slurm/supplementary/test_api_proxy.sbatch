#!/usr/bin/env bash
#SBATCH --job-name=api-proxy-test
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=outputs/slurm/api-proxy-test-%j.out
#SBATCH --error=outputs/slurm/api-proxy-test-%j.err

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}/evaluation_benchmark"
"${ROOT}/.venv/bin/python" - <<'PY'
import json, urllib.request
from pathlib import Path
key = Path('../api_key.txt').read_text().strip()
url = 'https://api.openai-proxy.org/v1/chat/completions'
payload = {
    'model': 'gpt-4o-mini',
    'messages': [{'role': 'user', 'content': 'Reply JSON: {"ok": true}'}],
    'temperature': 0.0,
    'max_tokens': 20,
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'},
    method='POST',
)
with urllib.request.urlopen(req, timeout=30) as resp:
    body = json.loads(resp.read())
print('HTTP OK')
print(body['choices'][0]['message']['content'][:200])
PY
