#!/usr/bin/env python3
"""Prove the `stage_mapper` tie-break defect is a NO-OP for the tasks being published.

WHY THIS GATE EXISTS
--------------------
We are publishing the VoLo baseline for tasks 4/5/20/23 but NOT 12/13/17. That split is only
honest if we can show the four published tasks' behaviour is identical under the defective
mapper (which actually ran, sha256 40cabe7d...) and under the fixed mapper. If they differed,
these baselines would be "a run of a known-buggy file" and the split would be arbitrary.

THE ARGUMENT PER TASK, MECHANICAL NOT RHETORICAL
  * tasks 4/5 : |primitive_order| == |stage_specs| (9 == 9), so rule (1) index alignment returns
                BEFORE the fuzzy block is ever reached. The defect is unreachable.
  * tasks 20/23: rule (1) does not apply (3 specs vs 6 labels), so the fuzzy block IS reached --
                but no tie occurs, because the only competing label, `open microwave`, scores
                hit=1 (only 'microwave' is a substring of the stage name) which is below the
                `hit >= min(2, len(tokens))` = 2 gate. With no tie, both versions agree.

This script establishes that mechanically by sweeping every (stage_idx) and a set of realistic
`current_subtask` values through BOTH implementations and asserting equality.

Exit code 0 = identical everywhere (publishable). 1 = a divergence exists (do not publish).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE_SPECS_DIR = ROOT / "evaluation_benchmark" / "scripts"
sys.path.insert(0, str(ROOT / "evaluation_benchmark"))
sys.path.insert(0, str(STAGE_SPECS_DIR))

import task2_26_reference_stage as M  # noqa: E402

# The exact file the published runs used (see each run's code_provenance.json).
DEFECTIVE = ROOT / "code_archive" / "api_pmh_md__587007" / "src" / "harness" / "stage_mapper.py"
DEFECTIVE_SHA = "40cabe7d76fd21ba145201a8aca18b27a4b699612cef240ce0ffd71a8d33a3c2"

PUBLISH = {4: "索引对齐路径（缺陷不可达）", 5: "索引对齐路径（缺陷不可达）",
           20: "模糊路径可达但无平局", 23: "模糊路径可达但无平局"}
WITHHELD = {12: "模糊路径 + 平局 → 缺陷命中", 13: "模糊路径 + 平局 → 缺陷命中",
            17: "模糊路径 + 平局 → 缺陷命中"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import hashlib

    got = hashlib.sha256(DEFECTIVE.read_bytes()).hexdigest()
    print("=" * 104)
    print("  stage_mapper 等价性检验：已发布任务的输出在缺陷版与修复版下是否完全一致")
    print("=" * 104)
    print(f"  缺陷版来源: {DEFECTIVE.relative_to(ROOT)}")
    print(f"    sha256 = {got}")
    if got != DEFECTIVE_SHA:
        print(f"  ✗ 该文件哈希与运行记录不符（期望 {DEFECTIVE_SHA}）")
        return 1
    print("    ✓ 与两个基线的 code_provenance.json 记录一致（即当时真正运行的版本）")

    buggy_mod = _load(DEFECTIVE, "buggy_stage_mapper")
    from harness.stage_mapper import expected_primitive_for_stage as fixed_fn
    buggy_fn = buggy_mod.expected_primitive_for_stage

    cfg = json.loads((ROOT / "evaluation_benchmark" / "async_vlm26_reference"
                      / "fullvlm_v2_26_memory_tasks.json").read_text(encoding="utf-8"))
    tasks = {int(t["task_id"]): t for t in cfg["tasks"]}

    print()
    print(f"  {'task':>5s} {'stages':>7s} {'labels':>7s} {'路径':<26s} {'比对次数':>9s} {'不一致':>7s}  判定")
    print("  " + "-" * 100)
    bad = 0
    for tid in list(PUBLISH) + list(WITHHELD):
        labels = [str(p["label"]) for p in tasks[tid]["primitive_order"]]
        specs = M._task_specs(tid)
        n_cmp = n_diff = 0
        # Sweep every stage, and every subtask the run could plausibly have in force.
        for idx in range(len(specs) + 1):
            for cur in [""] + labels + ["reach_to_middle_drawer_handle", "open the drawer",
                                        "reach_middle_drawer_handle", "close_the_middle_drawer"]:
                for ndone in range(len(specs) + 1):
                    stage_done = {s.name: (i < ndone) for i, s in enumerate(specs)}
                    kw = dict(primitive_labels=labels, stage_idx=idx, stage_specs=specs,
                              stage_done=stage_done, current_subtask=cur)
                    try:
                        a = buggy_fn(**kw)
                    except Exception as e:  # noqa: BLE001
                        a = f"<{type(e).__name__}>"
                    try:
                        b = fixed_fn(**kw)
                    except Exception as e:  # noqa: BLE001
                        b = f"<{type(e).__name__}>"
                    n_cmp += 1
                    if a != b:
                        n_diff += 1
                        if n_diff <= 2:
                            print(f"        ! 不一致 idx={idx} cur={cur!r} ndone={ndone}"
                                  f" 缺陷={a!r} 修复={b!r}")
        verdict = "✅ 一致 → 可发布" if n_diff == 0 else "❌ 有差异 → 不可发布"
        if tid in PUBLISH and n_diff:
            bad += 1
        tag = PUBLISH[tid] if tid in PUBLISH else WITHHELD[tid]
        mark = "" if tid in PUBLISH else "  [不发布]"
        print(f"  {tid:>5d} {len(specs):>7d} {len(labels):>7d} {tag:<26s} {n_cmp:>9d} {n_diff:>7d}  {verdict}{mark}")
    print("  " + "-" * 100)
    print()
    if bad == 0:
        print("  结论: 拟发布的 4 个任务在缺陷版与修复版下【逐输入完全一致】。")
        print("        缺陷对它们是 NO-OP，因此这些基线既非「碰巧跑对」，也无需重跑。")
    else:
        print(f"  结论: ✗ 有 {bad} 个拟发布任务存在差异，不能按「未被污染」发布。")
    print("=" * 104)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
