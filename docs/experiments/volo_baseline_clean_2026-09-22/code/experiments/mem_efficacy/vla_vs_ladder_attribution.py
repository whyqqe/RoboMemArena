#!/usr/bin/env python3
"""Is the VoLo baseline's drawer-family failure the VLA's fault, or the ladder's?

THE QUESTION
------------
The baselines show the drawer family (tasks 12/13/17) stalling on `02_Place_*` / `03_Place_*`
stages while the stall ladder kept proposing `open middle drawer`. Before blaming the executor
VLA -- or proposing to swap in a different VLA -- we must separate two very different worlds:

  (A) The VLA was asked to `pick`/`place` and failed.        -> executor capability problem.
      Swapping the VLA could plausibly help.
  (B) The VLA was never asked to `pick`/`place` at all.     -> the request never left the
      harness; the executor is being blamed for work it was never given. Swapping the VLA
      CANNOT help, because the fault is upstream of the VLA.

The discriminator is simply: which primitive strings actually reached the VLA, and did the
corresponding stages ever get confirmed?

EVIDENCE SOURCES (both are run artifacts, read-only)
  * `sync_vlm.log`     -- `VLA chunk prompt=<primitive>` records every primitive handed to the
                          executor, and `harness subtask override:` records ladder injections.
  * `harness_memory.json` -- `attempts[].completed_stages` records which stages were actually
                          confirmed by the completion checker, i.e. what the executor achieved.

CONTROL GROUP
  The microwave family (tasks 20/23) shares the same VLA, the same seeds and the same harness,
  but its stage names (`02_Place_Cookies_Microwave`) share no token with `open microwave`, so
  the ladder's fuzzy matcher never tied and emitted proper `place*` overrides. It therefore
  provides a within-run control for "what this VLA does when it IS asked to place".
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VLA_PROMPT = re.compile(r"VLA chunk prompt=(.+?)\s*$")
PLACE_LIKE = ("pick", "place", "pour", "put")


def _episodes(run_dir: Path):
    for task in sorted(run_dir.iterdir()):
        if not (task.is_dir() and task.name.startswith("task")):
            continue
        tid = int(task.name.replace("task", ""))
        for ep in sorted(task.iterdir()):
            if ep.is_dir() and ep.name.startswith("ep"):
                yield tid, ep


def _vla_prompts(ep_dir: Path) -> Counter:
    log = ep_dir / "sync_vlm.log"
    if not log.exists():
        return Counter()
    c: Counter = Counter()
    for line in log.read_text(errors="ignore").splitlines():
        m = VLA_PROMPT.search(line)
        if m:
            c[m.group(1).strip()] += 1
    return c


def _confirmed_stages(ep_dir: Path) -> Counter:
    """Union of `completed_stages` across attempts, weighted by how often each was reached."""
    c: Counter = Counter()
    for att in sorted(ep_dir.glob("attempt*")):
        hm = att / "harness_memory.json"
        if not hm.exists():
            continue
        try:
            d = json.loads(hm.read_text(encoding="utf-8"))
        except Exception:
            continue
        for a in d.get("attempts", []) or []:
            for st in a.get("completed_stages", []) or []:
                c[str(st)] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    prompts: dict[int, Counter] = defaultdict(Counter)
    confirmed: dict[int, Counter] = defaultdict(Counter)
    n_ep: Counter = Counter()

    for tid, ep in _episodes(args.run_dir):
        n_ep[tid] += 1
        prompts[tid] += _vla_prompts(ep)
        confirmed[tid] += _confirmed_stages(ep)

    DRAWER, MICRO = {12, 13, 17}, {20, 23}

    print("=" * 108)
    print("  VLA 到底被要求做了什么？—— 分族对照（同一 VLA、同一 harness、同一 seed）")
    print("=" * 108)
    print(f"  {'task':>5s} {'族':<7s} {'ep':>3s} | {'送达 VLA 的 place 类 primitive':<42s} | 确认过的 place stage")
    print("  " + "-" * 104)
    fam_stats = {"drawer": {"asked_place": 0, "prompt_calls": 0, "confirmed_place": 0},
                 "micro": {"asked_place": 0, "prompt_calls": 0, "confirmed_place": 0}}
    detail = {}
    for tid in sorted(set(prompts) | set(confirmed)):
        fam = "抽屉" if tid in DRAWER else ("微波" if tid in MICRO else "其他")
        key = "drawer" if tid in DRAWER else ("micro" if tid in MICRO else None)
        pl = {k: v for k, v in prompts[tid].items() if k.lower().startswith(PLACE_LIKE)}
        cp = {k: v for k, v in confirmed[tid].items()
              if "place" in k.lower() or "put" in k.lower()}
        total = sum(prompts[tid].values())
        asked = sum(pl.values())
        if key:
            fam_stats[key]["asked_place"] += asked
            fam_stats[key]["prompt_calls"] += total
            fam_stats[key]["confirmed_place"] += sum(cp.values())
        detail[tid] = {"family": fam, "place_prompts": pl, "confirmed_place_stages": cp,
                       "all_prompts": dict(prompts[tid]), "total_prompt_calls": total,
                       "n_episodes": n_ep[tid]}
        print(f"  {tid:>5d} {fam:<7s} {n_ep[tid]:>3d} | {str(dict(pl)):<42s} | {dict(cp) if cp else '无'}")
    print("  " + "-" * 104)

    print("\n" + "=" * 108)
    print("  分族汇总：VLA 被要求执行 place 类动作的频率")
    print("=" * 108)
    for key, label in (("drawer", "抽屉族 12/13/17"), ("micro", "微波族 20/23")):
        s = fam_stats[key]
        pct = 100.0 * s["asked_place"] / s["prompt_calls"] if s["prompt_calls"] else 0.0
        print(f"  {label:<18s} 送达 VLA 的 primitive 总数 {s['prompt_calls']:>6d}"
              f"  其中 place 类 {s['asked_place']:>5d} ({pct:5.1f}%)"
              f"   被确认的 place stage 次数 {s['confirmed_place']}")

    print("\n" + "=" * 108)
    print("  判读")
    print("=" * 108)
    d, m = fam_stats["drawer"], fam_stats["micro"]
    d_pct = 100.0 * d["asked_place"] / d["prompt_calls"] if d["prompt_calls"] else 0.0
    m_pct = 100.0 * m["asked_place"] / m["prompt_calls"] if m["prompt_calls"] else 0.0
    print(f"  抽屉族: place 类指令占全部 VLA 调用的 {d_pct:.1f}%")
    print(f"  微波族: place 类指令占全部 VLA 调用的 {m_pct:.1f}%")
    if d_pct < 5 and m_pct > 20:
        print("  → 抽屉族的 VLA 几乎从未被要求做 place 动作，而微波族被大量要求。")
        print("  → 结论: 抽屉族的失败发生在【请求生成阶段】(ladder/planner)，不是执行阶段。")
        print("  → 换 VLA 无法修复一个从未被发出的请求。")
    elif d["confirmed_place"] == 0 and m["confirmed_place"] > 0:
        print("  → 抽屉族从未确认过任何 place stage，微波族确认过 → 需进一步区分请求缺失与能力缺失。")
    else:
        print("  → 数据不支持单一归因，需人工复核。")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"per_task": detail, "family_summary": fam_stats},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  已写出: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
