#!/usr/bin/env python3
"""Does `claude-opus-4-6` honour the keyframe-nomination policy at this endpoint?

WHY THIS EXISTS, AND WHY IT IS NOT PART OF `run_26x1.sbatch`
  The `pushmem` arm sets `MEM_KF_NOMINATION_PROMPT=1`, and GATE 0 in `run_26x1.sbatch` FAILS THE
  JOB when the Planner nominates nothing:

      "MEM_KF_NOMINATION_PROMPT is ON but the model nominated NOTHING even on a deliberately
       unambiguous open-drawer transition. ... Re-word the policy or drop the claim before
       spending GPU time."

  That gate is correct, and it is also expensive to discover: it aborts before any GPU is used,
  but only after a queue wait. The question it asks -- "does THIS model act on THIS policy?" --
  is a single API call, so it can be answered now, at zero GPU cost, and the answer decides which
  arm to submit.

  The probe in the sbatch is deliberately not reused as a library: it is embedded in a heredoc and
  bound to that job's environment. This script imports the SAME policy object (`KF_NOMINATION_POLICY`)
  and the SAME parser (`parse_vlm_output`) rather than copying their text, so it cannot certify a
  channel that the arms do not actually send. That is the same argument the RSI dump module makes
  about calling the live JPEG encoder.

WHAT IS AND IS NOT BEING TESTED
  The observation window is SYNTHETIC and unambiguous on purpose. The question is whether the
  channel ENGAGES -- is the policy understood and the field used at all -- not what the hit rate
  would be on real scenes. Real-scene rate is `census_channels.py`'s job after a run, and it FAILs
  the arm when the knob is ON and the nomination count is zero.

  So a PASS here does not mean the bank will be good. It means the arm will not be void. A FAIL
  means the arm as written is the wrong arm for this Planner, and the honest fix is to drop
  `MEM_KF_NOMINATION_PROMPT` and run the stage-anchor + spread bank alone.

WHY IT REPEATS
  This endpoint paraphrases at the lexical level (~4-12% at susceptible decision points, measured
  in `experiments/rsi/`). A field that appears once could be luck. `--trials 5` is the default
  because a channel that nominates 1 time in 5 is not a channel.

USAGE
  python experiments/mem_efficacy/volo_kf_nomination_probe.py
  python experiments/mem_efficacy/volo_kf_nomination_probe.py --trials 5 --model claude-opus-4-6
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_key(path: Path, line: int = 1) -> str:
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not lines:
        raise SystemExit(f"no key in {path}")
    return lines[line - 1] if 0 < line <= len(lines) else lines[0]


def _scene(opened: bool) -> str:
    """A cabinet face, and the same face with the drawer pulled out and lit.

    Deliberately unambiguous: this tests the CHANNEL, not the hit rate. If a model will not
    nominate the transition between these two frames, it will not nominate a real one.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 256), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 80, 216, 200], fill=(90, 70, 50), outline=(20, 20, 20), width=3)
    if opened:
        d.rectangle([40, 120, 216, 200], fill=(225, 210, 180))
        for cx in (80, 128, 176):
            d.ellipse([cx - 14, 150, cx + 14, 178], fill=(200, 60, 60))
        d.text((46, 86), "OPEN DRAWER", fill=(255, 255, 255))
    else:
        d.text((46, 86), "CLOSED DRAWER", fill=(230, 230, 230))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(policy: str) -> list[dict]:
    """The request shape the arms send, including the two-field contract and the policy block.

    The order matters and is the arm's order: contract text, then the nomination policy appended
    last, so the policy is the most recent instruction before the model answers.
    """
    return [
        {"role": "system", "content": "You are a robot task planner."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Task: put the butter in the drawer that still holds nothing."},
                {
                    "type": "text",
                    "text": "Recent visual context: 2 consecutive frames ending at the current frame:",
                },
                {"type": "image_url", "image_url": {"url": _scene(False)}},
                {"type": "image_url", "image_url": {"url": _scene(True)}},
                {
                    "type": "text",
                    "text": "Output strict JSON with exactly two fields: current_primitive and "
                    "keyframe_positions. keyframe_positions are 1-indexed keyframe positions "
                    "inside the recent visual window.",
                },
                {"type": "text", "text": policy},
            ],
        },
    ]


def post(base: str, key: str, payload: dict, timeout: float) -> tuple[int | None, str, float]:
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.time() - t0
    except Exception as e:  # noqa: BLE001 -- a transport failure is a result here, not a crash
        return None, f"{type(e).__name__}: {e}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("PLANNER_API_MODEL", "claude-opus-4-6"))
    ap.add_argument("--base-url", default=os.environ.get("PLANNER_API_BASE_URL", "https://api.closeai-asia.com/v1"))
    ap.add_argument("--key-file", default=str(ROOT / "closeai_api.txt"))
    ap.add_argument("--key-line", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    # Mirror the evaluator's own import path (as `volo_gate0_vision.py` does):
    # `api_vlm_planner` imports `keyframe_selection` from openpi_minimal_runtime, which the runner
    # supplies via PYTHONPATH. Without this the import fails before any request is sent.
    sys.path.insert(0, str(ROOT / "evaluation_benchmark" / "openpi_minimal_runtime"))
    sys.path.insert(0, str(ROOT / "evaluation_benchmark"))
    from harness.api_vlm_planner import KF_NOMINATION_POLICY
    from harness.vlm_output_parser import parse_vlm_output

    key = load_key(Path(args.key_file), args.key_line)
    msgs = build_messages(KF_NOMINATION_POLICY)

    print("=" * 92)
    print("keyframe-nomination channel probe (predicts GATE 0 for any arm with")
    print("MEM_KF_NOMINATION_PROMPT=1)")
    print("=" * 92)
    print(f"  model      = {args.model}")
    print(f"  base_url   = {args.base_url}")
    print(f"  key        = ...{key[-8:]} (line {args.key_line})")
    print(f"  max_tokens = {args.max_tokens}   trials = {args.trials}")
    print(f"  policy     = {len(KF_NOMINATION_POLICY)} chars, imported not copied")
    print()

    ok = 0
    parsed_ok = 0
    nominating = 0
    for i in range(args.trials):
        st, body, dt = post(
            args.base_url,
            key,
            {"model": args.model, "max_tokens": args.max_tokens, "messages": msgs},
            args.timeout,
        )
        if st != 200:
            print(f"  [{i+1}/{args.trials}] HTTP {st} {dt:.2f}s  {body[:200]!r}")
            continue
        ok += 1
        try:
            txt = json.loads(body)["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i+1}/{args.trials}] HTTP 200 but unparsable envelope: {exc!r}  {body[:160]!r}")
            continue
        try:
            prim, jrel = parse_vlm_output(txt, max_pos=2)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i+1}/{args.trials}] parser raised {type(exc).__name__}: {exc}; raw={txt[:160]!r}")
            continue
        if prim:
            parsed_ok += 1
        if jrel:
            nominating += 1
        flag = "NOMINATED" if jrel else "none     "
        print(f"  [{i+1}/{args.trials}] {dt:5.2f}s  {flag}  primitive={prim!r} keyframe_positions={jrel}")
        if not jrel:
            print(f"           raw: {txt[:300]!r}")

    print()
    print("  " + "-" * 88)
    print(f"  HTTP 200        : {ok}/{args.trials}")
    print(f"  primitive parsed: {parsed_ok}/{args.trials}")
    print(f"  NOMINATED       : {nominating}/{args.trials}")
    print()

    if ok == 0:
        print("  VERDICT: endpoint unusable. Nothing else here is interpretable.")
        return 2
    if parsed_ok < ok:
        print("  VERDICT: the two-field contract is BREAKING. GATE 0 would fail on the primitive.")
        print("           The policy block is displacing the JSON. Do not submit an arm with")
        print("           MEM_KF_NOMINATION_PROMPT=1.")
        return 3
    if nominating == 0:
        print("  VERDICT: NO NOMINATIONS. Any arm with MEM_KF_NOMINATION_PROMPT=1 would ABORT at")
        print("           GATE 0, and even if forced past it `census_channels.py` would FAIL the")
        print("           arm. Configure the visual-channel arm WITHOUT the nomination prompt:")
        print("           the stage-anchor + spread bank carries the channel on its own")
        print("           (validate_arm.py CHECK 7 proves that bank is non-empty with no")
        print("           nominations). Set MEM_KF_NOMINATION_PROMPT=0.")
        return 4
    if nominating < ok:
        print(f"  VERDICT: PARTIAL -- nominates on {nominating}/{ok} usable calls. The channel")
        print("           engages but is unreliable. GATE 0 passes on its single call, so a run")
        print("           would proceed; the census nomination count is what decides whether the")
        print("           bank is nomination-driven or fallback-driven. Expect a MIXED bank and")
        print("           say so when reporting.")
        return 1
    print(f"  VERDICT: the channel ENGAGES on every call ({nominating}/{ok}). An arm with")
    print("           MEM_KF_NOMINATION_PROMPT=1 is the right arm for this Planner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
