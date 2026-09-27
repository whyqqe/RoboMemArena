# 代码快照与架构清单

本目录 `code/` 保存了**产出本报告 GPM 结果的那一组代码**（快照，不是工作区引用）。
打包时相对仓库提交 `8af37b4`；运行时指纹见各结果目录 `code_provenance.json`，文件级 SHA-256 见 `SHA256SUMS.txt`。

`nomem` 基线结果来自既有 hard3 / task5 控制实验；对照臂脚本一并纳入以便协议复现。

## 文件清单

| 状态 | 包内路径 | 源路径 |
|---|---|---|
| 新增 | `code/memexp_evmem.py` | `experiments/mem_efficacy/memexp_evmem.py` |
| 新增 | `code/memexp_evmem_bind.py` | `experiments/mem_efficacy/memexp_evmem_bind.py` |
| 新增 | `code/selftest_evmem.py` | `experiments/mem_efficacy/selftest_evmem.py` |
| 新增 | `code/arms/evmem.sh` | `experiments/mem_efficacy/arms/evmem.sh` |
| 既有（对照） | `code/arms/nomem.sh` | `experiments/mem_efficacy/arms/nomem.sh` |
| 既有（对照） | `code/arms/official_protocol.sh` | `experiments/mem_efficacy/arms/official_protocol.sh` |
| 既有 | `code/pysite/sitecustomize.py` | `experiments/mem_efficacy/pysite/sitecustomize.py` |
| 新增 | `code/run_evmem_gpm_t8_1x10.sh` | `experiments/mem_efficacy/run_evmem_gpm_t8_1x10.sh` |
| 新增 | `code/run_evmem_gpm_remaining_1x10.sh` | `experiments/mem_efficacy/run_evmem_gpm_remaining_1x10.sh` |

## 架构要点（G / E / C）

实现入口：`memexp_evmem.py`（策略）+ `memexp_evmem_bind.py`（挂到 API planner）+ `arms/evmem.sh`（在 `nomem.sh` 上打开开关）。

| 开关 / 常量 | 默认 | 含义 |
|---|---|---|
| `MEMEXP_EVMEM` | 1（evmem 臂） | 启用 GPM |
| `MEMEXP_EVMEM_LOOKAHEAD` | 4 | 谓词脚手架前瞻窗 |
| `MEMEXP_EVMEM_MAX_ROUNDS` | 2 | 单步工具轮次上限 |
| `MEMEXP_EVMEM_EVIDENCE_CAP` | 6 | 证据帧上限 |
| `MEMEXP_EVMEM_GATE_ATTEMPTS` | 3 | attempts 门控阈值 |
| `MEMEXP_EVMEM_GATE_STALL` | 3 | stall 门控阈值 |
| `MEMEXP_EVMEM_ATTRACTOR_REPEAT` | 2 | 吸引子重复触发证据 |
| `PMH_REDACT_STAGE` | 1 | 与 redact-safe 脚手架一致 |

相对 `nomem` 的预期差异声明（`MEMEXP_EXPECTED_DIFF`）：`PYTHONPATH`、`PMH_REDACT_STAGE`。KF bank / pull-push 通道保持关闭。

## 自测

```bash
python code/selftest_evmem.py
```

## 提交脚本要点

- `run_evmem_gpm_t8_1x10.sh`：仅 t8；futility 在 6 trials 后 all-zero 或 mean ≤ 5%。
- `run_evmem_gpm_remaining_1x10.sh`：t5 / t19 / t22；t22 的 futility mean floor 为 25（避免过早掐断高基线任务）。
- 两者均设 `STAGE1_ALLOW_RERUN=1`、`EARLY_STOP_ON_QUOTA=1`。
