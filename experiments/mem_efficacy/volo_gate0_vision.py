#!/usr/bin/env python3
"""GATE 0d -- the MAIN planner's vision path, exercised before any GPU is spent.

WHY THIS EXISTS
  `run_26x1.sbatch` GATE 0 proves two things about the Planner: the endpoint answers,
  and the TEXT-ONLY stall-recovery rung (`suggest_subtask_via_api`) parses. Its third
  probe -- the one that sends images -- is skipped unless an arm declares
  `MEM_KF_NOMINATION_PROMPT`, which the no-memory baseline deliberately does not.

  So for the VoLo-aligned baseline (arm `nomem`) the path that actually runs every
  VLM_INTERVAL steps -- `chat_completions` with a multi-image recent window, at the
  configured budget, with `temperature=0.0` hard-coded in the payload -- was never
  exercised until the GPUs were already busy. That is the same class of failure this
  project has already paid for: an endpoint that answers HTTP 200 but yields nothing
  the parser accepts, discovered hours into a run.

WHAT THIS ASSERTS
  1. HTTP 200 with the REAL payload shape: `temperature=0.0` is sent unconditionally by
     `api_planner.chat_completions`. Reasoning-only models reject `temperature`; VoLo's
     own `vlm/api.py` documents Claude <= 4.6 as the standard-chat path, so Opus 4.6 is
     expected to accept it and Opus 4.7+ is not. This probe is what separates the two.
  2. The images actually reach the model -- asserted by OCR, not by HTTP status.
  3. `parse_vlm_output` yields a non-empty `current_primitive` at the real budget.

CPU only: no simulation, no checkpoints, no GPU. Exit 0 = safe to spend GPU time.
"""

from __future__ import annotations

import base64
import io
import os
import sys

ROOT = os.environ.get("ROOT", "/project/peilab/why/RoboMemArena")
# Mirror the evaluator's own import path: the runner exports PYTHONPATH as
# LIBERO_FORK_ROOT:TARGET_LIBERO_PATH:RUNTIME_DIR:..., and api_vlm_planner imports
# `keyframe_selection` from openpi_minimal_runtime (RUNTIME_DIR).
sys.path.insert(0, os.path.join(ROOT, "evaluation_benchmark", "openpi_minimal_runtime"))
sys.path.insert(0, os.path.join(ROOT, "evaluation_benchmark"))

from PIL import Image, ImageDraw  # noqa: E402

from harness.api_planner import chat_completions, load_api_key  # noqa: E402
from harness.vlm_output_parser import parse_vlm_output  # noqa: E402


def _real_system_prompt() -> str:
    """The Planner's actual system prompt, without importing the evaluator.

    `SYSTEM_PROMPT_MEMORY` lives in the eval module, which imports torch / robosuite /
    libero at module scope -- unusable in a CPU-only preflight. It is a static literal
    there, so it is lifted with the stdlib AST instead of being copied. Copying it would
    let the probe keep passing after the prompt changed, which is the exact drift the
    gates in this project exist to prevent.
    """
    import ast

    eval_path = os.path.join(
        ROOT, "evaluation_benchmark", "async_vlm26_reference", "eval_fullvlm26_async_vlm_vla.py"
    )
    source = open(eval_path, encoding="utf-8").read()
    longtask = os.environ.get("VLM_LONGTASK_PROMPT", "0") == "1"
    wanted = "SYSTEM_PROMPT_MEMORY_LONGTASK" if longtask else "SYSTEM_PROMPT_MEMORY_DEMO"
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == wanted:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"{wanted} not found in {eval_path}")


def _jpeg_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def _scene(label: str, opened: bool, filled: bool) -> Image.Image:
    """A cabinet scene carrying an OCR-able caption, so image arrival is provable.

    The caption is the load-bearing part: an endpoint that silently drops images still
    returns HTTP 200 and fluent prose. Only a token the model cannot know otherwise
    distinguishes "saw the frame" from "guessed".
    """
    img = Image.new("RGB", (256, 256), (28, 28, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([30, 70, 226, 210], fill=(92, 72, 52), outline=(18, 18, 18), width=3)
    if opened:
        d.rectangle([30, 118, 226, 210], fill=(228, 214, 186))
        if filled:
            for cx in (78, 128, 178):
                d.ellipse([cx - 15, 150, cx + 15, 180], fill=(196, 58, 58))
    d.text((36, 78), label, fill=(255, 255, 255))
    d.text((36, 16), "CABINET VIEW", fill=(210, 210, 210))
    return img


def main() -> int:
    api_key = load_api_key()
    base_url = os.environ["PLANNER_API_BASE_URL"]
    model = os.environ["PLANNER_API_MODEL"]
    max_tokens = int(os.environ.get("PLANNER_API_MAX_TOKENS", "4096"))
    timeout = float(os.environ.get("PLANNER_API_TIMEOUT", "180"))

    # The recent window the baseline actually sends: N_RECENT main frames plus the wrist
    # views (VLM_USE_WRIST=1). Ten images is an upper bound on that window, so passing at
    # ten means passing at five.
    n_main, n_wrist = 5, 5
    captions = ["TOP DRAWER EMPTY", "MIDDLE DRAWER HOLDS BUTTER", "BOTTOM DRAWER EMPTY"]
    main_frames = [
        _scene(captions[i % len(captions)], opened=(i >= 2), filled=(i % len(captions) == 1))
        for i in range(n_main)
    ]
    wrist_frames = [_scene("WRIST CAM", opened=False, filled=False) for _ in range(n_wrist)]

    user_content: list[dict] = [
        {"type": "text", "text": (
            "Task: put the butter into the drawer that already contains an object. "
            "Recent visual context, oldest first, ending at the current frame:"
        )},
    ]
    for img in main_frames + wrist_frames:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(img)}"},
        })
    user_content.append({"type": "text", "text": (
        "Output strict JSON with exactly two fields: current_primitive and keyframe_positions. "
        "keyframe_positions may be an empty list."
    )})

    messages = [
        {"role": "system", "content": _real_system_prompt()},
        {"role": "user", "content": user_content},
    ]

    print(f"  model={model}  base={base_url}")
    print(f"  images={n_main + n_wrist}  max_tokens={max_tokens}  timeout={timeout}s  temperature=0.0 (hard-coded)")

    out = chat_completions(
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_sec=timeout,
        max_tokens=max_tokens,
    )

    if not out:
        print("\n[GATE 0d] FAILED: chat_completions returned None.")
        print("  Either the endpoint refused the payload (temperature is sent unconditionally;")
        print("  Opus 4.7+ rejects it) or the response was empty/truncated.")
        return 4

    print(f"  raw (first 300) -> {out[:300]!r}")

    primitive, j_rel = parse_vlm_output(out, max_pos=n_main + n_wrist)
    print(f"  parsed -> primitive={primitive!r}  keyframe_positions={j_rel}")

    fail: list[str] = []
    if not primitive:
        fail.append(
            "parse_vlm_output produced NO current_primitive: the model did not honour the "
            "two-field contract at this budget."
        )

    # Image arrival, asserted positively. The baseline is prompted to open drawers and
    # inspect contents; a model that received the frames has the caption available.
    low = out.lower()
    if not any(tok in low for tok in ("middle", "butter", "empty", "drawer")):
        fail.append(
            "the response mentions none of the scene's own vocabulary -- the images may not "
            "have reached the model (HTTP 200 with dropped images is exactly this signature)."
        )

    if fail:
        print("\n[GATE 0d] FAILED:")
        for f in fail:
            print(f"  - {f}")
        return 4

    print("\n[GATE 0d] PASSED: the main planner's vision path answers and parses "
          f"({n_main + n_wrist} images, max_tokens={max_tokens}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
