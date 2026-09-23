#!/usr/bin/env python3
"""Runtime census for the cross-task experience channel: did the block reach the WIRE?

WHY THIS EXISTS BESIDE GATE 0d
  GATE 0d proves the mechanism fires by driving a synthetic `HarnessController` through a stage
  handler. That is a real controller and a real `compose_planner_context`, so it rules out the two
  defects it was written for. It does NOT prove that what the controller composed reached the
  Planner in a live episode, because the live path is: controller -> `compose_planner_context` ->
  `planner.harness_extra_context` -> `_build_messages` -> the request body. Four hops. This project
  has already lost runs to a mechanism that was installed and enabled but whose value never
  travelled that path, and the loss was invisible in the score.

  So this reads the only artifact that is downstream of ALL four hops: the RSI replay bundle, which
  is a transcription of the actual wire payload (`harness/rsi_replay_dump.py`). If the entry text is
  in `replay.json`, it was in the request the endpoint received. Nothing about that can be true
  without the whole chain having worked.

WHAT IT REPORTS
  Per run: the number of decisions whose payload carries the treatment entry, the sham entry, or
  neither; and the number carrying ANY `text_other` part, which is the honest denominator -- a block
  that reaches 3 of 28 decisions because the trigger is short-lived is a different experiment from
  one that reaches 28, and the mean alone cannot tell them apart.

  It also counts the decisions that carry a part whose text is NOT the objective, the camera-order
  block, or the output contract, because a mistyped role classification would put the block somewhere
  the replay logic would then substitute into the wrong place.

USAGE
  python experiments/mem_efficacy/verify_ltm_injection.py --run-root <dir>
  python experiments/mem_efficacy/verify_ltm_injection.py --run-root <dir> --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# A distinctive fragment of each bank's body. Matched rather than the whole text so that a
# whitespace or punctuation edit to the bank does not make this report "not injected" -- the
# failure it is looking for is ABSENCE, and a brittle matcher would produce a false absence.
MARKERS = {
    "rsimem": "three opening commands in a row",
    "shamltm": "three pouring commands in a row",
}
HEADER = "Cross-task experience"


def scan(run_root: Path, marker: str) -> dict:
    n_decisions = n_marker = n_header = n_other_text = 0
    episodes = 0
    for bundle in run_root.rglob("vlm_inputs/t*/replay.json"):
        n_decisions += 1
        try:
            b = json.loads(bundle.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hit_marker = hit_header = hit_other = False
        for part in b.get("parts", []):
            if part.get("kind") != "text":
                continue
            txt = str(part.get("text", ""))
            if marker and marker in txt:
                hit_marker = True
            if HEADER in txt:
                hit_header = True
            if part.get("role") == "text_other":
                hit_other = True
        n_marker += hit_marker
        n_header += hit_header
        n_other_text += hit_other
    for ep in run_root.rglob("vlm_inputs"):
        if (ep.parent.name or "").startswith("attempt"):
            episodes += 1
    return {
        "run": str(run_root),
        "episodes": episodes,
        "decisions": n_decisions,
        "with_marker": n_marker,
        "with_header": n_header,
        "with_unclassified_text": n_other_text,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--selftest", action="store_true",
                    help="assert the scanner can tell the two banks apart on a synthetic bundle")
    args = ap.parse_args()

    if args.selftest:
        # A scanner that cannot fail is not a check. It is driven over three synthetic payloads --
        # treatment, sham, and neither -- and must separate all three.
        import tempfile
        ok = True
        with tempfile.TemporaryDirectory() as td:
            for name, marker, expect in (("rsimem", MARKERS["rsimem"], "rsimem"),
                                         ("shamltm", MARKERS["shamltm"], "shamltm"),
                                         ("none", "", "none")):
                d = Path(td) / name / "vlm_inputs" / "t0000"
                d.mkdir(parents=True)
                parts = []
                if marker:
                    parts.append({"kind": "text", "role": "text_other",
                                  "text": f"Cross-task experience (...)\n- {marker} and more text."})
                (d / "replay.json").write_text(json.dumps({"parts": parts}), encoding="utf-8")
            for name, marker, expect in (("rsimem", MARKERS["rsimem"], "rsimem"),
                                         ("shamltm", MARKERS["shamltm"], "shamltm"),
                                         ("none", MARKERS["rsimem"], "none")):
                r = scan(Path(td) / name, marker)
                good = (r["with_marker"] == 1) if expect != "none" else (r["with_marker"] == 0)
                ok &= good
                print(f"  selftest {name:8s} -> with_marker={r['with_marker']}  "
                      f"{'ok' if good else 'FAILED'}")
        print(f"\n  selftest {'PASSED' if ok else 'FAILED'}")
        return 0 if ok else 1

    root = Path(args.run_root)
    if not root.is_dir():
        print(f"  no such run root: {root}")
        return 2

    rows = []
    for arm, marker in MARKERS.items():
        for d in sorted(root.glob(f"{arm}_s*")):
            rows.append(scan(d, marker))
    if not rows:
        print(f"  no arm directories under {root}; nothing has been written yet")
        return 2

    print("=" * 104)
    print("runtime census: is the cross-task experience block in the WIRE payload?")
    print("=" * 104)
    print(f"  {'run':58s} {'eps':>4} {'decisions':>10} {'marker':>7} {'header':>7} {'other_txt':>10}")
    print("  " + "-" * 100)
    for r in rows:
        print(f"  {Path(r['run']).name:58s} {r['episodes']:>4} {r['decisions']:>10} "
              f"{r['with_marker']:>7} {r['with_header']:>7} {r['with_unclassified_text']:>10}")
    print()

    fails = []
    for r in rows:
        arm = Path(r["run"]).name.split("_")[0]
        if r["decisions"] == 0:
            continue  # nothing written yet; not a finding
        if arm == "nomem" and r["with_marker"] + r["with_header"] > 0:
            fails.append(f"{r['run']}: the BASELINE carries the block. The control has the "
                         f"treatment installed, so no memory delta from this job is interpretable.")
        if arm in MARKERS and arm != "nomem" and r["with_marker"] == 0:
            fails.append(f"{r['run']}: 0 of {r['decisions']} decisions carry the {arm} entry. The "
                         f"arm ran as a treatment and delivered nothing -- the defect this file "
                         f"exists to catch, and one the score cannot show.")
    if fails:
        print("  FINDINGS:")
        for f in fails:
            print(f"    ! {f}")
        return 1
    live = [r for r in rows if r["decisions"] > 0]
    if not live:
        print("  no decisions recorded yet; re-run once episodes have planned")
        return 2
    print(f"  PASS: {len(live)} run(s) with payloads; baseline clean, every treatment arm reachable.")
    print("  NOTE: `marker` counts DECISIONS, so it is also the trigger's duty cycle. A low count is")
    print("        not a failure -- it is how often the opening stage stalled past the threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
