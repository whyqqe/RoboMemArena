#!/usr/bin/env python3
"""Is the container actually OPEN at the stuck decision points? (cause vs symptom)

WHY THIS EXISTS
  Mined from the 50-episode VoLo-aligned baseline, the dominant failure signature is a primitive
  lock-in: the planner emits `open_middle_drawer` / `open_microwave` (or a lexical variant of the
  same action) on 96-100% of its planning steps in the zero-score episodes, against 55% in the
  100-score ones. Pooled r(score, open-step share) = -0.601.

  That correlation is necessary but not sufficient for an EXPERIENCE-MEMORY intervention, and the
  two readings of it imply opposite experiments:

    CAUSE    the planner cannot tell that the container opened, or cannot think of anything else
             to do, so it re-emits the same primitive. -> memory is the right lever, and telling
             it to diversify is the right entry.
    SYMPTOM  the container genuinely is not open, the planner's repeated command is CORRECT, and
             the failure is motor-side (the controller cannot open it). -> an entry that says
             "stop repeating open" would be instructing the planner to do the wrong thing, and
             no planner-side memory can fix this failure.

  Reverse causality is a live hazard here and it is measurable: successful episodes END EARLY
  (task 23: 7-13 planning steps for the 100s, 19-28 for the failures), so they simply have fewer
  opportunities to repeat. A cheap correlation cannot separate "few repeats because it worked"
  from "it worked because of few repeats". The only way to tell is to look at the world: is the
  container open, or not?

WHY A SEPARATE VLM CALL AND NOT THE PLANNER'S OWN TRACE
  The planner's stated primitive is the thing under investigation, so it cannot be the evidence.
  This asks an independent, stateless, temperature-0 question of the SAME frames the planner was
  given, with a closed answer set, and it reports the raw answer next to the parsed one so a
  refusal cannot be read as a verdict.

  It deliberately does not reuse `_sdv_verify_stage` (`api_vlm_planner.py:1712`). That verifier
  asks about a stage's physical outcome across the bank; this asks a narrower question about one
  frame pair, and the two must be able to disagree.

USAGE
  python experiments/mem_efficacy/volo_container_state_probe.py
  python experiments/mem_efficacy/volo_container_state_probe.py --task 20 --ep 0 --which last
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
BANK = ROOT / "experiments/rsi/results/bank_a2"

QUESTION = (
    "You are given consecutive frames from a robot episode, oldest first. The last frame is the "
    "current observation. The robot was commanded to OPEN a container (a microwave door or a "
    "kitchen drawer).\n\n"
    "Answer with strict JSON and nothing else:\n"
    '{"container":"microwave|drawer|not_visible","state":"open|closed|partially_open|not_visible",'
    '"moved_between_frames":true|false,"evidence":"<=15 words"}\n\n'
    "- state=open: the door/drawer is clearly pulled out or swung away, its interior is visible.\n"
    "- state=closed: the door/drawer is flush, its face is closed over the opening.\n"
    "- state=partially_open: it has moved but the opening is not yet usable.\n"
    "- state=not_visible: the container is not in frame, or the view is occluded.\n"
    "- moved_between_frames: whether the door/drawer position changed across these frames."
)


def load_key(path: Path, line: int = 1) -> str:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    if not lines:
        raise SystemExit(f"no key in {path}")
    return lines[line - 1] if 0 < line <= len(lines) else lines[0]


def recent_frames(bundle: Path) -> tuple[list[str], dict]:
    b = json.loads((bundle / "replay.json").read_text(encoding="utf-8"))
    parts = [p for p in b["parts"] if p.get("kind") == "image" and p.get("section") == "recent"]
    urls = []
    for p in parts:
        raw = (bundle / p["file"]).read_bytes()
        urls.append("data:image/jpeg;base64," + base64.b64encode(raw).decode())
    return urls, b


def post(base: str, key: str, payload: dict, timeout: float) -> tuple[int | None, str, float]:
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.time() - t0
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("PLANNER_API_MODEL", "claude-opus-4-6"))
    ap.add_argument("--base-url", default=os.environ.get("PLANNER_API_BASE_URL",
                                                         "https://api.closeai-asia.com/v1"))
    ap.add_argument("--key-file", default=str(ROOT / "closeai_api.txt"))
    ap.add_argument("--key-line", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--which", choices=("last", "all"), default="last",
                    help="'last' = the final logged decision of each attempt; 'all' = every bundle")
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--ep", type=int, default=None)
    args = ap.parse_args()

    key = load_key(Path(args.key_file), args.key_line)

    targets: list[tuple[int, int, str, Path]] = []
    for tdir in sorted(BANK.glob("nomem_s100/task*"), key=lambda p: int(p.name[4:])):
        tid = int(tdir.name[4:])
        if args.task is not None and tid != args.task:
            continue
        for epd in sorted(tdir.glob("ep*"), key=lambda p: int(p.name[2:])):
            eid = int(epd.name[2:])
            if args.ep is not None and eid != args.ep:
                continue
            for att in sorted(epd.glob("attempt*")):
                vd = att / "vlm_inputs"
                bundles = sorted(vd.glob("t*"))
                if not bundles:
                    continue
                chosen = bundles[-1:] if args.which == "last" else bundles
                for bd in chosen:
                    targets.append((tid, eid, att.name, bd))

    if not targets:
        print("  no bundles matched")
        return 2

    print("=" * 100)
    print("container-state probe: is the container OPEN at the stuck decision points?")
    print("=" * 100)
    print(f"  model={args.model}  bundles to judge={len(targets)}  mode={args.which}")
    print()

    tally: dict[str, int] = {}
    for tid, eid, att, bd in targets:
        urls, bundle = recent_frames(bd)
        content = [{"type": "text", "text": QUESTION}]
        for u in urls:
            content.append({"type": "image_url", "image_url": {"url": u}})
        st, body, dt = post(args.base_url, key, {
            "model": args.model, "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }, args.timeout)
        raw = ""
        if st == 200:
            try:
                raw = json.loads(body)["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                raw = f"<unparsable envelope: {exc!r}>"
        verdict = "?"
        if st == 200:
            try:
                s = raw.strip()
                if s.startswith("```"):
                    s = s.split("```")[1]
                    s = s[4:] if s.lower().startswith("json") else s
                v = json.loads(s.strip())
                verdict = str(v.get("state", "?"))
            except Exception:  # noqa: BLE001
                verdict = "unparsed"
        tally[verdict] = tally.get(verdict, 0) + 1
        print(f"  task{tid:<3} {att} {bd.name}  ({bundle['step_idx']:>5})  HTTP {st} {dt:5.2f}s")
        print(f"      last_primitive = {bundle.get('logged_out_text', '')!r}")
        print(f"      verdict        = {verdict}")
        print(f"      raw            = {raw[:220]!r}")
        print()

    print("  " + "-" * 96)
    print(f"  container state across {len(targets)} judgments: {tally}")
    print()
    opened = tally.get("open", 0) + tally.get("partially_open", 0)
    closed = tally.get("closed", 0)
    if opened == 0 and closed > 0:
        print("  VERDICT: SYMPTOM. The container is NOT open at these decision points while the")
        print("           planner keeps commanding the open primitive. The repeated command is")
        print("           CORRECT, and the failure is motor-side. An experience-memory entry that")
        print("           tells the planner to stop repeating open would be instructing it to do")
        print("           the wrong thing, and no planner-side memory can repair this failure.")
        return 4
    if opened > 0 and closed == 0:
        print("  VERDICT: CAUSE. The container IS open while the planner keeps commanding open.")
        print("           The planner cannot use what it is being shown, which is exactly the gap")
        print("           a disclosure-form memory entry can close.")
        return 0
    print("  VERDICT: MIXED. Both states occur. The lock-in is a symptom in some episodes and a")
    print("           cause in others, so a single memory entry cannot be justified for the whole")
    print("           family -- the entry has to be conditioned on the state, and the census must")
    print("           report the split rather than an average.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
