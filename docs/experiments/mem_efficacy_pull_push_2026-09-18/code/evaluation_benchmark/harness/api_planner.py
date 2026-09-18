from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from harness import socks_tunnel
# The shared unwrapper. Hand-rolling the fence strip here is what broke the recovery rung --
# see the note in `suggest_subtask_via_api`.
from harness.vlm_output_parser import strip_wrapper

logger = logging.getLogger(__name__)

# Install the SOCKS5 egress before anything can dial out, and do it at import time so every call
# site (planner, stall-recovery, and the harness's own probes) is covered without each one having
# to remember. Opt-in via $PLANNER_SOCKS_PROXY; unset leaves the interpreter unpatched, so the
# H/HM baselines keep their direct path. See socks_tunnel's docstring for why HK needs this.
socks_tunnel.install()


def load_api_key(key_file: Path | None = None, line: int | None = None) -> str:
    """Load API key from file. ``line`` is 1-based; defaults to env or first non-empty line."""
    env_key = os.environ.get("PLANNER_API_KEY", os.environ.get("HARNESS_API_KEY", "")).strip()
    if env_key:
        return env_key

    line_no = line
    if line_no is None:
        raw_line = os.environ.get("PLANNER_API_KEY_LINE", os.environ.get("HARNESS_API_KEY_LINE", "")).strip()
        if raw_line:
            try:
                line_no = int(raw_line)
            except ValueError:
                line_no = None

    path = key_file
    if path is None:
        key_raw = os.environ.get("PLANNER_API_KEY_FILE", os.environ.get("HARNESS_API_KEY_FILE", "")).strip()
        if key_raw:
            path = Path(key_raw)
        else:
            default_key = Path(__file__).resolve().parents[2] / "api_key.txt"
            path = default_key if default_key.is_file() else None

    if path is None or not path.is_file():
        return ""

    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        return ""
    if line_no is not None and line_no > 0:
        idx = line_no - 1
        if idx < len(lines):
            return lines[idx]
    return lines[0]


def _pil_to_jpeg_base64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _openai_vision_messages(system_prompt: str, user_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal message parts to OpenAI-compatible chat content."""
    api_user: list[dict[str, Any]] = []
    for part in user_content:
        if part.get("type") == "text":
            api_user.append({"type": "text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image":
            img = part.get("image")
            if isinstance(img, Image.Image):
                url = f"data:image/jpeg;base64,{_pil_to_jpeg_base64(img)}"
                api_user.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": api_user},
    ]


def chat_completions(
    *,
    messages: list[dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
    timeout_sec: float = 120.0,
    max_tokens: int = 256,
) -> str | None:
    if not api_key:
        return None
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
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
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.warning("API chat completion failed: %s", exc)
        return None

    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return None


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
    timeout_sec: float | None = None,
) -> str | None:
    """Harness stall-recovery planner (text-only).

    BUDGET AND TIMEOUT ARE REASONING-AWARE (measured 2026-09-14).

    The hardcoded `max_tokens=128` was calibrated on qwen-vl-max, which emits the JSON
    directly. A REASONING planner (gemini-3.8-flash via the CloseAI relay) emits its chain
    of thought INSIDE `max_tokens` before the JSON, so 128 truncates mid-token:

        mt=128  finish=length  ctok=124   -> '{"current_primitive'   (unparseable)
        mt=256  finish=stop    ctok=147   -> 'open top drawer'       (ok)
        mt=512  finish=stop    ctok=175   -> 'open top drawer'       (ok)

    The failure is SILENT and it disables the recovery mechanism while the arm still
    reports `HARNESS_SUBTASK_OVERRIDE=1`: `chat_completions` returns the truncated
    string, `json.loads` raises, the helper returns `None`, and `_handle_stall` simply
    keeps the old candidate. That is indistinguishable from "the planner had nothing to
    say" (the t22 signature), so it would be scored as mechanism inertness rather than as
    a config error.

    The default timeout is raised for the same reason: the same call measured 33.86s at
    mt=256, i.e. ABOVE the old 30.0s ceiling, so a slow-but-correct recovery would be
    converted into a timeout and then into a `None`.

    Both are env-overridable so the arm config owns the numbers instead of this file.
    """
    if not api_key:
        return None

    if timeout_sec is None:
        try:
            timeout_sec = float(os.environ.get("HARNESS_API_TIMEOUT", "90") or 90)
        except ValueError:
            timeout_sec = 90.0
    try:
        recovery_max_tokens = max(1, int(os.environ.get("HARNESS_API_MAX_TOKENS", "512") or 512))
    except ValueError:
        recovery_max_tokens = 512

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
    messages = [
        {"role": "system", "content": "Reply with JSON only."},
        {"role": "user", "content": prompt},
    ]
    content = chat_completions(
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_sec=timeout_sec,
        max_tokens=recovery_max_tokens,
    )
    if not content:
        return None

    text = str(content).strip()
    # USE THE SHARED UNWRAPPER. The hand-rolled version that stood here was:
    #
    #     if "```" in text:
    #         text = text.split("```", 2)[-1]          # <-- takes the TRAILING chunk
    #         text = text.replace("json", "", 1).strip()
    #
    # `'```json\n{"current_primitive": "..."}\n```'.split("```", 2)` is
    # `['', 'json\n{...}\n', '']`, so `[-1]` selects the EMPTY string after the closing fence,
    # `json.loads('')` raises, and the function returns None. Measured on gemini-3.8-flash
    # (CloseAI, 2026-09-14): the model wraps its answer in a fence on 9 of 16 recovery calls and
    # emits it bare on the other 7 -- 7/16 accepted, and every one of the 9 rejects was this,
    # with a perfectly good `{"current_primitive": "open top drawer"}` inside the fence.
    #
    # It stayed hidden because qwen-vl-max rarely fences, and because a None here is SILENT:
    # `_handle_stall` keeps its old candidate, the arm still reports HARNESS_SUBTASK_OVERRIDE=1,
    # and the recovery rung reads as "the planner had nothing to say" (the t22 signature) rather
    # than as a parser defect. A mechanism that fails ~half the time it is invoked would have
    # been scored as inert.
    text = strip_wrapper(text)
    if not text.startswith("{"):
        # Some writers prefix the object with a sentence. Take the outermost braces rather than
        # give up, since None here disables a recovery rung instead of surfacing an error.
        lo, hi = text.find("{"), text.rfind("}")
        if lo != -1 and hi > lo:
            text = text[lo : hi + 1]
    try:
        parsed = json.loads(text)
        primitive = str(parsed.get("current_primitive", "")).strip()
    except json.JSONDecodeError:
        return None
    if primitive and primitive in primitive_labels:
        return primitive
    return None


def infer_primitive_via_api(
    *,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
    timeout_sec: float = 120.0,
    max_tokens: int = 256,
) -> str | None:
    """Full VLM-style planner call with optional vision content."""
    messages = _openai_vision_messages(system_prompt, user_content)
    return chat_completions(
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_sec=timeout_sec,
        max_tokens=max_tokens,
    )
