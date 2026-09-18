# 代码快照与改动清单

本目录 `code/` 保存了**产出本报告全部结果的那一组代码**（快照，不是工作区引用）。
对 job `595262` 而言，快照的 29 个文件与运行时的 SHA-256 **逐一相符（29/29）**，因此 `code/` 中的内容就是该作业的精确输入。

状态相对本仓库 `main` 基线 `eb86819` 而言。

## 文件清单

| 状态 | 文件 |
|---|---|
| 已修改 | `evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py` |
| 已修改 | `evaluation_benchmark/async_vlm26_reference/run_fullvlm26_async_vlm_vla_csr_tsr.sh` |
| 已修改 | `evaluation_benchmark/harness/api_planner.py` |
| 已修改 | `evaluation_benchmark/harness/config.py` |
| 已修改 | `evaluation_benchmark/harness/controller.py` |
| 已修改 | `evaluation_benchmark/harness/stage_mapper.py` |
| 已修改 | `evaluation_benchmark/memory_system/config.py` |
| 新增 | `evaluation_benchmark/harness/api_vlm_planner.py` |
| 新增 | `experiments/mem_efficacy/arms/_memexp_pysite.sh` |
| 新增 | `experiments/mem_efficacy/arms/nomem.sh` |
| 新增 | `experiments/mem_efficacy/arms/official_protocol.sh` |
| 新增 | `experiments/mem_efficacy/arms/pullmem.sh` |
| 新增 | `experiments/mem_efficacy/arms/pushmem.sh` |
| 新增 | `experiments/mem_efficacy/census_channels.py` |
| 新增 | `experiments/mem_efficacy/memexp_bind.py` |
| 新增 | `experiments/mem_efficacy/memexp_memfix.py` |
| 新增 | `experiments/mem_efficacy/memexp_substrate.py` |
| 新增 | `experiments/mem_efficacy/memexp_tools.py` |
| 新增 | `experiments/mem_efficacy/memexp_tools_search.py` |
| 新增 | `experiments/mem_efficacy/pysite/sitecustomize.py` |
| 新增 | `experiments/mem_efficacy/run_26x1.sbatch` |
| 新增 | `experiments/mem_efficacy/selftest_memfix.py` |
| 新增 | `experiments/mem_efficacy/selftest_pull.py` |
| 新增 | `experiments/mem_efficacy/validate_arm.py` |

其余 5 个文件与基线一致（未修改），一并纳入快照以保证自足：

`evaluation_benchmark/harness/external_memory.py`、`evaluation_benchmark/harness/memory_reason.py`、
`evaluation_benchmark/memory_system/keyframe_bank.py`、`evaluation_benchmark/openpi_minimal_runtime/keyframe_selection.py`、
`evaluation_benchmark/scripts/task2_26_reference_stage.py`

## 已修改文件的 diffstat

```
  1503 +     95 -  evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py
   213 +     38 -  evaluation_benchmark/async_vlm26_reference/run_fullvlm26_async_vlm_vla_csr_tsr.sh
   215 +     36 -  evaluation_benchmark/harness/api_planner.py
    53 +      1 -  evaluation_benchmark/harness/config.py
   578 +      6 -  evaluation_benchmark/harness/controller.py
    28 +      3 -  evaluation_benchmark/harness/stage_mapper.py
    36 +      0 -  evaluation_benchmark/memory_system/config.py
```

> 这 7 个文件里包含本实验之外的既有改动（`controller.py` / `stage_mapper.py` / `config.py` 等），
> 它们不属于本次记忆机制修复。完整的逐行 diff 可通过本快照与基线 `eb86819` 对比得到。

---

## 本报告结论直接依赖的 5 处改动

### D1 —— 把"帧记录"与"planner 调用"解耦

`evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py`（新增，默认关闭）

```diff
     def clone_recent_frames() -> list[tuple[np.ndarray, np.ndarray | None]]:
         return [(m.copy(), w.copy() if w is not None else None) for m, w in recent_vlm_frames]
 
-    def submit_vlm_job(step_idx: int) -> None:
+    kf_store_interval = int(os.environ.get("MEM_KF_STORE_INTERVAL", "0") or 0)
+    kf_store_count = {"n": 0, "skipped": 0}
+
+    def store_kf_dense(step_idx: int) -> None:
+        """Store this step's frame in the planner's keyframe store, on a fixed interval."""
+        if kf_store_interval <= 0 or step_idx < 0 or not recent_vlm_frames:
+            return
+        if step_idx % kf_store_interval != 0:
+            return
+        store = getattr(planner, "frame_store_main", None)
+        if store is None:
+            kf_store_count["skipped"] += 1
+            return
+        main_np, wrist_np = recent_vlm_frames[-1]
+        try:
+            from PIL import Image as _Image
+
+            store[int(step_idx)] = _Image.fromarray(main_np.astype("uint8"))
+            wstore = getattr(planner, "frame_store_wrist", None)
+            if wrist_np is not None and wstore is not None:
+                wstore[int(step_idx)] = _Image.fromarray(wrist_np.astype("uint8"))
+            kf_store_count["n"] += 1
+        except Exception:
+            kf_store_count["skipped"] += 1
+
+    def submit_vlm_job(step_idx: int, *, force: bool = False) -> None:
```

调用点（紧邻取帧处，因此由 env 循环驱动，而非由队列是否驱逐决定）：

```diff
                 obs, _, done, _ = env.step(action.tolist())
                 recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
+                # Dense keyframe store. Placed HERE, next to the frame extraction, so the record is
+                # driven by the env loop rather than by whether a planner call survived the queue.
+                store_kf_dense(t - args.num_steps_wait)
```

### D2 —— `_kf_spread_union` 端点吸附

`evaluation_benchmark/harness/api_vlm_planner.py`（新增文件）

```python
n = min(int(want), len(older))
step = len(older) / float(n)
picks = sorted(
    {
        older[min(len(older) - 1, max(0, int((k + 0.5) * step)))]
        for k in range(n)
    }
)
```

原为 `older[min(len(older) - 1, int(k * step))]`，其 `k=0` 项恒等于 `older[0]`。

### D3 / D4 —— 工具说明不再招揽枚举工具，且写明真实预算

`experiments/mem_efficacy/memexp_tools_search.py`（新增文件）

```python
def _max_rounds() -> int:
    """The loop's own fuse, read from the same variable the fuse reads."""
    return max(1, int(os.environ.get("MEMEXP_PULL_MAX_ROUNDS", "3")))
```

```python
f"You may call up to {_max_rounds()} tool(s) within one planning step; a further call "
"is refused and you will be told to finish the step. There is no penalty for calling a "
"tool and no reward for avoiding one.",
```

`list_unbound` 与 `list_facts` 的推荐文案已从 `spec_text()` 移除（两者仍注册、被调用仍应答）。

### D5 —— `search_memory` 的 miss 回退

`experiments/mem_efficacy/memexp_tools_search.py`（新增文件）

```python
def _recent_readable(self, k: int = SEARCH_MAX) -> tuple[list, str]:
    """The most recent records that can actually show an observation right now.

    The fallback arm of `search_memory`, used when no record shares a token with the query.
    Ranked by recency rather than by a score, because with zero overlap there is nothing to
    score; and FILTERED on `available_frames`, because a record whose observation window has
    not closed returns no pixels -- selecting on recency alone would hand back an empty frame
    list and reproduce exactly the dead end this fallback exists to remove.
    """
    cands = [f for f in self.substrate.facts if self.substrate.available_frames(f, set())]
    cands.sort(key=lambda f: -int(getattr(f, "mint_step", 0) or 0))
    return cands[:k], "recency_fallback"
```

### 臂声明的对应改动

`experiments/mem_efficacy/arms/pullmem.sh`：

```diff
-export MEMEXP_PULL_MAX_ROUNDS="${MEMEXP_PULL_MAX_ROUNDS:-3}"
+export MEMEXP_PULL_MAX_ROUNDS="${MEMEXP_PULL_MAX_ROUNDS:-2}"
```

---

## 新增的闸门（用于锁住上述改动）

| 闸门 | 位置 | 作用 |
|---|---|---|
| `CHECK 17: the spread picks NO frame from the episode's opening` | `validate_arm.py` | D2。已**实测证伪**：退回旧写法立即 `[FAIL] early picks = [0]` |
| `SPEC: the offer does not solicit a listing the push already rendered` | `memexp_bind.py` | D3。已实测证伪 |
| `SPEC: the budget the spec states is the budget the loop enforces` | `memexp_bind.py` | D4。已实测证伪 |
| `CHECK 19`：稠密 store 与调用窗口 store 的对比 | `validate_arm.py` | D1。要求稠密 store 比调用窗口 store 大一个数量级 |

说明与 fuse 的比对由 `memexp_bind.stated_budget()` 从**说明文本本身**解析（而非读两遍环境变量），
因此两者是相互独立的产物，才能检出不一致。

## 分母/分子口径

`census_channels.py` 新增两项直接刻画"检索工具收益"的输出：

- `frames per tool call` —— 观测帧数 / 工具调用数（595132：0.20；595262：0.88）
- `enumeration share of calls` —— 枚举类调用占比（595132：49%；595262：0%）

两者都以比值形式打印，便于与历史 run 直接比较。

## 密钥

结果目录不含密钥。`api_key.txt` 已被仓库 `.gitignore` 排除；本快照与两个结果目录均经过
明文密钥匹配与通用密钥模式（`sk-…`、`Bearer …`）扫描，结果为空。
