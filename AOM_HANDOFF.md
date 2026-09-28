# AOM handoff: Arbitrated Obligation Memory — status, results, and what to do next

Audience: a fresh agent picking up the AOM line with no prior context.
Read this file first, then `README.md`, then
[`docs/experiments/aom_2026-09-28/REPORT.md`](docs/experiments/aom_2026-09-28/REPORT.md)
(the full results package: traces, baselines, code snapshot, checksums).

Date: 2026-09-28. Base commit: `8e95f41`.

---

## 1. Where AOM sits in this repo

This repo evaluates whether **memory improves control** on RMA-style long-horizon manipulation
tasks. The baseline (`nomem`) is the official protocol; the treatments add a channel by which
past observations reach the asynchronous VLM planner.

**AOM (Arbitrated Obligation Memory)** is one such channel. It is not a bag of retrieved text:
it maintains an explicit set of **obligations** (one per task stage), renders a compact
`obligation board` into the planner prompt, and **gates** the planner's emitted primitive against
the obligation graph. The architectural claim is that legality is graph reachability rather than
GPM's hand-written regex table.

Obligation modes: `ACTUAL` (moved by physical action, settles on the harness's own predicate),
`PERCEPTUAL` (moved by inspection), `DERIVED` (computed, e.g. `count:place`). Plus an orthogonal
`needs_look` flag. The arbitration law returns one of `advance / retrieve / act / derive /
stagnant` in a fixed branch order.

Code: `experiments/mem_efficacy/memexp_aom.py` (architecture),
`memexp_aom_bind.py` (harness hook), `selftest_aom.py`, `arms/evmem_aom.sh`.

---

## 2. What changed in this session (the important part)

### 2.1 The defect: AOM seeded its graph from the SCORING partition

Task 19 has two decompositions of different length:

```
scoring   _task_specs(19)     = 3 steps   Place_Tomato_Sauce / Place_Milk / Place_Orange_Juice
execution primitive_order(19) = 6 steps   pick_TS, place_TS, pick_milk, place_milk, ...
```

AOM read the **scoring** table, so the graph had three `place` obligations and **no grasp
obligation at all**. The board's own rule ("serve only the ACTIVE obligation") then commanded a
`place` — an action that presupposes a held object — for the whole episode. The commanded action
was physically unsatisfiable.

Measured: AOM v2 emitted **15 grasps and 615 places** (nomem: 344 / 151), and its first command
was `Place the object into the open container.` while nomem's was
`Reach toward and grasp the tomato sauce from cabinet1.`

**Generalisation (this is the transferable finding):** an execution phase the scoring partition
omits is benign when it lies **downstream** of every scored phase and fatal when it is a
**precondition**. That is why AOM wins big on task 5 (partitions coincide: 45.0 vs nomem 13.8),
is neutral on task 8 (the omitted `place` is the epilogue), and was below control on task 19 (the
omitted `pick` precedes every scored `place`). **15 other tasks in the 26-task suite carry the
same latent precondition**, so do not treat this as a task-19 quirk.

### 2.2 The fix

`Ledger.seed_execution(...)` seeds from `task_info.primitive_labels` (the BDDL execution
partition) and binds each execution step to the scored stage that verifies it using the harness's
**own** bridge, `harness.stage_mapper.expected_primitive_for_stage` — which already existed for
the stall ladder and the ledger guard. **AOM had been bypassing it.** Grasp steps settle
**derivationally** (when the place they feed settles), so no new sensor is needed.

Gated by `should_seed_execution(scored, exec)`: ON only when the scoring partition omits a grasp
that precedes a scored phase. Over all 26 tasks: `t5 OFF, t8 OFF, t19 ON, t22 OFF`. Tasks 5 and 8
therefore keep their previously measured graphs bit-for-bit, and their archived results stay valid
controls.

Also: object naming is now ON for **transfer** tasks (`memory_type` without `O`). The
"without naming drawer identity" clause was inherited from the **occlusion** arm, where object
identity is the withheld answer; on a transfer task it *is* the instruction, and redacting it
produced commands with no object named at all.

### 2.3 The bug the fix introduced, and its fix (read this if you touch the graph)

v5 emitted **719 grasps and 0 places**: `place#1` required `pick#1` to be `SETTLED`, but `pick#1`
settles only *with* `place#1` — a cycle. Fixed by `Ledger::_dep_ready`: a grasp→place edge is
satisfied when the grasp has been **attempted** (`status != OPEN` or `tau_act > 0`), not settled.
Ordinary sequencing edges keep the hard `SETTLED` requirement. `set_active_stage` honours the
harness's scored-stage hint only when reachable, and `active()` self-heals off a settled pointer.

**Any change to the graph must keep `selftest_aom.py` green; T14/T15 exist specifically to catch
a recurrence of these two failures.**

---

## 3. Results (seed 100, like-for-like, 1x10)

| task | nomem | pullmem_er | EvMem | **EvMem-GPM** | AOM v2 | v3s | v4 | v5 | **v6** |
|---|---|---|---|---|---|---|---|---|---|
| **5**  | 13.75 | – | – | 20.00 | **45.00** | – | – | – | 36.25 |
| **8**  | 26.67 | – | – | 46.68 | 43.35 | – | – | – | **50.02** |
| **19** | 13.32 | 16.65 | 19.99 | **26.67** | 23.32 | 16.66 | 14.80* | 23.33 | 19.99 |
| **22** | 73.34 | – | – | **73.33** | – | – | – | – | 33.33* |

`*` = truncated by the early-stop gate, **not** a 10-trial mean.

**v6 vs GPM on the same seed: t5 +16.25 WIN · t8 +3.34 WIN · t19 −6.68 lose · t22 −40 (truncated).**

Slurm: 617727 (t5), 617728 (t8), 617508 (t19), 617729 (t22, early-stopped at k=9).

### 3.1 Two reporting errors in the prior record — do not propagate them

1. **The like-for-like baseline was wrong.** Single-seed arms were compared against multi-seed
   *macro averages* (nomem 23.3/25.6). The correct seed-100 control on t19 is **13.32**, so
   "AOM was below nomem" is false; v2 (23.32) and v5 (23.33) are above it.
2. **AOM t19 v4's "14.8" is a truncation, not a measurement.** It was killed by
   `FUTILITY_MEAN_FLOOR_PCT=15`, and the same-seed control passes through mean **14.8 at k=9**
   (finishing 13.32) — the gate fired on the control's own trajectory. The series therefore does
   **not** support "23.3 → 16.7 → 14.8, monotonically worse".

---

## 4. What the data falsified, and what is still open

**Falsified — do not re-litigate without new evidence:**

- *"The primitive mix drives the score."* On t19 the strongest arm (GPM 26.67) has amongst the
  **fewest** grasps (22, essentially v2's 15), while the arm with by far the most (v5, 719) scores
  23.33. Fixing the graph changed 15/615 → 122/397 and did not move the score.
- *"Fixing the graph will lift t19."* It is mechanically correct now — `exec_labels` is the 6-step
  ladder, `name_objects` true, `gate_rejects`/`dep_rejects`/`offgraph_rejects` all 0, first command
  is a grasp — and t19 sits at 19.99 vs GPM 26.67.

**Still open:**

- **t19 is underpowered.** Every arm reaches 33.3 and none reaches 100; all separation comes from
  how many episodes hit 66.7. GPM vs its control is 3/10 vs 0/10 — Fisher exact **p ≈ 0.21**.
  The whole t19 table is inside the noise band. Move the score only with more seeds.
- **The bottleneck is physical execution of the second transfer**, not the text channel. No
  prompt-structure change will move it; that is the single most useful thing to accept from this
  session.
- **No single archived tag is both defect-free and the best scorer.** v2 scores highest on t19 and
  carries the self-lock (`dep_rejects=47`), the wrong-partition graph and the object redaction.
  v6 is mechanically clean and scores lower. Say which one you mean in any claim.

---

## 5. Suggested next steps, in the order that respects the evidence

1. **Do not tune the prompt further on t19.** The remaining gap is execution-side. If you want a
   score, spend it on seeds, not on wording.
2. **Raise n before any new t19 claim.** 3 seeds (100/110/120) on the v6 arm and its control is
   the cheapest thing that would make the task-19 table interpretable. The archived controls for
   all three seeds already exist (`hard3_t19_nomem_h0_3x10`).
3. **Diagnose the second transfer offline.** Read the replay/state streams of the failing t19
   episodes for collision / drop / misalignment rather than adding memory. The v6 cell for t19 is
   the honest artefact to start from.
4. **If you want a defensible win, use tasks 5 and 8.** Both now beat GPM for v6 with the gate
   OFF, i.e. unchanged graphs; that is a clean statement.
5. **Consider a weak-gate variant** (gate as guardrail, minimal forced rewriting): v2's high t19
   score came with the planner writing freely. That is a hypothesis, not a result — measure it.

---

## 6. Working conventions and constraints

- **Both selftests must pass before any submit.** Every runner enforces this and refuses to
  submit otherwise:
  ```bash
  cd experiments/mem_efficacy
  PYTHONPATH=/project/peilab/why/RoboMemArena/evaluation_benchmark:. python selftest_aom.py
  FUTILITY_WATCH_UNDER_TEST=3 FUTILITY_FLOOR_UNDER_TEST=<floor> FUTILITY_CONSEC_UNDER_TEST=<m> \
    FUTILITY_TASK_UNDER_TEST=<task> FUTILITY_SEED_UNDER_TEST=100 \
    PYTHONPATH=/project/peilab/why/RoboMemArena/evaluation_benchmark:. python selftest_futility.py
  ```
- **Never compare a single-seed cell against a macro average.** Match the seed.
- **Never set an early-stop floor at or above a control's own early floor.** The floor must sit
  below the control's minimum running mean at every prefix the gate can see, or it kills the
  control. `selftest_futility.py` T10 automates this check for the runner's own parameters.
- **Early stop is the only mechanism that kills a live run on the basis of a score**, and a
  truncated cell is indistinguishable from a failing one in the results. Always record whether a
  cell was stopped (`futility_stop.json`) and mark it wherever you quote it.
- **Runners to use:** `run_aom_t19_1x10.sh` (task 19) and `run_aom_v6_hard3_1x10.sh`
  (tasks 5, 8, 22). The older `run_aom_t8_1x10.sh` / `run_aom_remaining_1x10.sh` are superseded.
- **Do not touch `/home`.** (Standing instruction from the project owner.)
- Raw result artefacts are gitignored (`experiments/**/results/`); archival copies live under
  `docs/experiments/<name>_<date>/`, which is the established packaging convention here.
- The working tree currently carries uncommitted modifications to tracked harness files
  (`controller.py`, `stage_mapper.py`, `config.py`, ...) and ~889 untracked files. Leave them
  alone unless your task requires them.

---

## 7. Where everything lives

| what | path |
|---|---|
| Full results package (report, traces, baselines, code, checksums) | `docs/experiments/aom_2026-09-28/` |
| Architecture + invariants | `experiments/mem_efficacy/memexp_aom.py`, `selftest_aom.py` |
| Harness hook | `experiments/mem_efficacy/memexp_aom_bind.py` |
| Arm definition | `experiments/mem_efficacy/arms/evmem_aom.sh` |
| Runners | `experiments/mem_efficacy/run_aom_t19_1x10.sh`, `run_aom_v6_hard3_1x10.sh` |
| Early-stop rule + its calibration tests | `experiments/mem_efficacy/run_26x1.sbatch`, `selftest_futility.py` |
| Sibling memory designs for comparison | `CGMH.md`, `HERMES.md`, `PMH*.md`, `KAIROS.md` (repo root) |
| GPM baseline package | `docs/experiments/evmem_gpm_hard4_2026-09-27/` |
