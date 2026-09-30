# PushMem baseline — hard4 (tasks 5, 8, 19, 22), h0, seed 100, 1×10, no early stop

Date: 2026-09-30.
Companion: [`PROVENANCE.txt`](PROVENANCE.txt), [`CODE.md`](CODE.md), [`results/seed100_summary.tsv`](results/seed100_summary.tsv).

This package archives the **`pushmem`** arm on the four hard tasks, measured with every trial
scored (`FUTILITY_STOP_TRIALS=0`). It exists to give the post-AOM architectures (BOLT / DIAL /
ECHO, and the AOM/GPM series before them) a **complete, un-truncated control** on the same h0
profile and seed block.

---

## 1. Results

h0 / seed 100 / 1×10, all four jobs `COMPLETED` with exit code `0:0`.

| task | nomem (s100) | GPM (s100) | **pushmem (s100)** | pushmem series | Δ vs nomem | Δ vs GPM |
|---|---:|---:|---:|---|---:|---:|
| 5  | 13.75 | 20.00 | **23.75** | 25,50,0,12.5,50,0,0,0,0,100 | +10.00 | +3.75 |
| 8  | 26.67 | 46.68 | **33.35** | 0,66.7,0,66.7,66.7,0,66.7,0,0,66.7 | +6.68 | −13.33 |
| 19 | 13.32 | 26.67 | **23.33** | 0,33.3,66.7,0,66.7,0,33.3,0,33.3,0 | +10.01 | −3.34 |
| 22 | 73.34 | 73.33 | **70.00** | 66.7,100,100,33.3,66.7,0,100,100,100,33.3 | −3.34 | −3.33 |

Machine-readable: [`results/seed100_summary.tsv`](results/seed100_summary.tsv).
Raw traces: `results/pushmem_t{5,8,19,22}_s100/prompt_trace.tsv`.
Baselines: `baselines/t*_nomem_s100_*` and `baselines/t*_evmem_gpm_s100_*`.

**Read with the noise floor in mind.** Per-trial `stage_score_pct` SD is ≈32 pp on unsaturated
tasks, so the 10-trial mean SE is ≈14 pp. Every delta above is inside 1 SE. The directional
statement supported by this run is only:

- pushmem is **at or above nomem** on three of four tasks (+6.7 to +10.0 pp),
- pushmem is **at or below GPM** on all four,
- on task 22 all three arms sit at ~70–73 (the task is close to saturation for this planner).

None of these is a significant win; they are the numbers needed to build a complete comparison
grid, not evidence of a mechanism effect.

---

## 2. What was measured (arm definition)

`pushmem` is **channel B**: the harness pushes a visual keyframe bank into the planner prompt
without the planner asking. It is assembled entirely from RMA's own code paths
(`api_vlm_planner._build_messages` appends the historical-keyframe block) — no new memory module.

Key switches (declared diff): `VLM_USE_KEYFRAME_MEMORY`, `MEM_STAGE_ANCHOR=1`,
`MEM_KF_NOMINATION_PROMPT=1`, `MEM_KF_SPREAD=1`, `MEM_KF_STORE_INTERVAL=5`,
`N_RECENT=7 K_MAX=8 D_MERGE=4`, plus `PYTHONPATH` for the shared content correction.
Channel A (`HARNESS_VLM_CONTEXT`) stays OFF, so the two channels are not conflated.

Why `MEM_STAGE_ANCHOR=1` is load-bearing: with a general API planner, the model does not emit
`keyframe_positions`, so `J_hist` is a list of empty lists and the official bank is structurally
empty. Stage anchors (stage boundaries + subtask changes) are deterministic signals the planner
cannot suppress, so the bank becomes genuinely push-based.

Verified in each job's Slurm output:

- `[GATE 1] arm integrity` — pushmem resolved diff equals declared diff;
- `[CHECK 6] channel B resolves as designed` — `use_keyframe_memory=True`;
- `stage-anchored bank (MEM_STAGE_ANCHOR=1) is NON-EMPTY without nominations`;
- `census PASSED`.

One honest caveat recorded by the runner itself: the H0 read-path whitelist check was
**skipped** because only one arm was in scope. It asserts nothing cross-arm and is not claimed here.

---

## 3. Why no early stop

The user asked for the **baseline** measured without truncation. An early-stopped cell is not a
control: it changes the denominator and biases the mean. `run_pushmem_hard4_1x10.sh` therefore
sets `FUTILITY_STOP_TRIALS=0`, `FUTILITY_MEAN_FLOOR_PCT=0`, `FUTILITY_CONSEC_ZERO_M=0`, which is
the same completeness rule already used for the `nomem` control in `run_hard4_4x10.sh`.

All four jobs ran the full 10 trials; the t22 job finishing in 11:39 is a property of the task
(short episodes), not of a gate.

---

## 4. Reproduce

```bash
cd experiments/mem_efficacy
bash run_pushmem_hard4_1x10.sh --dry-run   # print the four sbatch commands
bash run_pushmem_hard4_1x10.sh             # submit; TAG=<name> for a new artifact root
```

Output roots: `results/pushmem_t{5,8,19,22}_pushmem_h0_1x10_v1/`.
