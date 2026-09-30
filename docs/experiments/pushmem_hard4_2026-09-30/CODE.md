# PushMem hard4 code map

Everything listed here is snapshotted under `code/` and also lives in
`experiments/mem_efficacy/` in the working tree.

| path | role |
|---|---|
| `arms/pushmem.sh` | the arm: sources `nomem.sh` then turns on channel B and the content correction |
| `arms/nomem.sh` | baseline it builds on; also performs the `PMH_/MEMEXP_/AIM_/SRH_/ALH_` purge that keeps arms order-independent |
| `arms/_memexp_pysite.sh` | adds `MEMEXP_DIR/pysite` to `PYTHONPATH` and sets `MEMEXP_MEMFIX_ENABLE=1` |
| `memexp_memfix.py` | shared memory-content correction installed in the evaluator process |
| `pysite/sitecustomize.py` | import hook that installs the bound memory modules |
| `run_pushmem_hard4_1x10.sh` | submitter: one job per task, futility OFF, seed 100, 1x10 |

## Channel B switches (the arm's declared difference)

| variable | value | effect |
|---|---|---|
| `VLM_USE_KEYFRAME_MEMORY` | `1` | enables the pushed keyframe bank |
| `MEM_STAGE_ANCHOR` | `1` | bank candidates from stage boundaries + subtask changes (not planner nominations) |
| `MEM_KF_NOMINATION_PROMPT` | `1` | instructs the API planner to nominate keyframes |
| `MEM_KF_SPREAD` | `1` | breadth floor so the bank is informative even if nominations are empty |
| `MEM_KF_STORE_INTERVAL` | `5` | one frame per replan window from the env loop |
| `N_RECENT / K_MAX / D_MERGE` | `7 / 8 / 4` | bank hyperparameters |

`HARNESS_VLM_CONTEXT` stays `0`, so channel A (textual memory context) is a controlled constant.
