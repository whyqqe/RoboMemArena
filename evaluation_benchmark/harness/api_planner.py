from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def load_api_key(key_file: Path | None) -> str:
    path = key_file or Path(os.environ.get("HARNESS_API_KEY_FILE", "api_key.txt"))
    if not path.is_file():
        return os.environ.get("HARNESS_API_KEY", "").strip()
    return path.read_text(encoding="utf-8").strip()


def suggest_subtask_via_api(
    *,
    task_block: str,
    stage_name: str | None,
    current_subtask: str,
    primitive_labels: list[str],
    memory_context: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout_sec: float = 30.0,
) -> str | None:
    """Optional OpenAI-compatible planner for stall recovery (HarnessVLA-style agentic hint)."""
    if not api_key:
        return None

    prompt = (
        "You are a robot task planner for a dual-system VLA stack.\n"
        "Given the task, current stage, recent stall, and memory context, choose exactly ONE "
        "primitive label from the allowed list.\n"
        "Return strict JSON: {\"current_primitive\": \"...\"}\n\n"
        f"Task:\n{task_block}\n\n"
        f"Allowed primitives: {json.dumps(primitive_labels, ensure_ascii=False)}\n"
        f"Current stage: {stage_name or 'unknown'}\n"
        f"Current subtask: {current_subtask}\n"
        f"Memory context:\n{memory_context}\n"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        import logging
        logging.getLogger(__name__).warning("API planner request failed: %s", exc)
        return None

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    text = str(content).strip()
    if "```" in text:
        text = text.split("```", 2)[-1]
        text = text.replace("json", "", 1).strip()
    try:
        parsed = json.loads(text)
        primitive = str(parsed.get("current_primitive", "")).strip()
    except json.JSONDecodeError:
        return None
    if primitive and primitive in primitive_labels:
        return primitive
    return None
