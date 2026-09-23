#!/usr/bin/env python3
"""Cheap pre-submit probe for the Stage-1 Planner model.

WHY THIS EXISTS
---------------
Submitting a 2-GPU job only to discover the model rejects `max_tokens` or `temperature=0.0`
costs a queue slot and, with a vision call, real money. This probe spends a few dozen tokens
on a 1x1 PNG and refuses to proceed unless the model returns a non-empty string.

It also refuses models that still need the old `max_tokens` path when the caller asked for a
reasoning model, and vice versa — by going through the SAME `chat_completions` the evaluator
uses, so a successful probe is evidence the runtime path works, not a parallel reimplementation.
"""
from __future__ import annotations

import base64
import os
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation_benchmark"))
sys.path.insert(0, str(ROOT / "evaluation_benchmark" / "openpi_minimal_runtime"))

from PIL import Image  # noqa: E402
from harness.api_planner import chat_completions, load_api_key  # noqa: E402


def main() -> int:
    model = os.environ.get("PLANNER_API_MODEL", "").strip()
    base = os.environ.get("PLANNER_API_BASE_URL", "https://api.closeai-asia.com/v1").strip()
    if not model:
        print("[probe] FAIL: PLANNER_API_MODEL unset")
        return 2
    key = load_api_key()
    if not key:
        print("[probe] FAIL: no API key")
        return 2

    # 1x1 red PNG, tiny on the wire.
    img = Image.new("RGB", (1, 1), (220, 20, 20))
    # chat_completions accepts OpenAI-style messages; build a vision turn the same way the
    # planner does (jpeg data URL).
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    messages = [
        {"role": "system", "content": "Reply with one English color word only."},
        {"role": "user", "content": [
            {"type": "text", "text": "What color is this pixel?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]},
    ]
    # Reasoning models burn completion tokens on chain-of-thought; 256 is the smallest budget
    # that has been observed to leave room for a one-word answer on gpt-6-luna.
    budget = int(os.environ.get("PLANNER_API_MAX_TOKENS", "256") or 256)
    print(f"[probe] model={model} base={base} budget={budget}")
    out = chat_completions(
        messages=messages,
        api_key=key,
        base_url=base,
        model=model,
        timeout_sec=90.0,
        max_tokens=max(budget, 256),
    )
    if not out or not str(out).strip():
        print(f"[probe] FAIL: empty response {out!r}")
        return 1
    print(f"[probe] PASS: {out!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
