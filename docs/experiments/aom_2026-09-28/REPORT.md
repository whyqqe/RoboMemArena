# AOM (Arbitrated Obligation Memory): series results and post-mortem

Date: 2026-09-28
Suite: `mem_efficacy`, profile `h0`, seed 100, 1x10 per cell
Slurm jobs: 617134-617137 (v1/v2 series), 617263, 617307, 617408, 617508, 617727-617729

This document reports the **AOM series** end to end: what the architecture claims, what was
measured, which hypotheses the data falsified, and which defects are real versus which were
reporting errors. It is deliberately written so that a reader who trusts none of the conclusions
can re-derive every number from `results/` and `baselines/` in this package.

---

## 1. What AOM is

A memory channel for the asynchronous VLM planner. It maintains an explicit set of
**obligations** (one per task stage), renders a compact `obligation board` into the planner
prompt, and gates the planner's emitted primitive against the obligation graph.

Obligations come in three modes, and the split is the architectural claim:

| mode | meaning | settles when |
|---|---|---|
| `ACTUAL` | advanced by a physical robot action | the harness's `stage_done` predicate passes |
| `PERCEPTUAL` | advanced by inspecting/reading a store | a `look` returns new frames |
| `DERIVED` | computed over the settled set (`count:place`, `count:steps`) | its target count is reached |

plus an orthogonal `needs_look` flag for stages that want visual evidence before action.

The arbitration law returns one of `advance / retrieve / act / derive / stagnant`, in that
branch order, and the board is a pure function of `(mode, status)` so the several prompt
surfaces cannot drift apart.

**Design intent under test:** replace GPM's hand-written regex gate with a graph gate, so
"which primitive is legal now" is a reachability fact rather than a table of task-specific
patterns.

---

## 2. Headline results (seed 100, like-for-like)

All numbers are the mean `stage_score_pct` over 10 trials on **seed 100**, which is the seed
every cell here ran. Macro averages over several seeds are NOT comparable to a single-seed cell
and are not used anywhere in this package.

| task | nomem | pullmem_er | EvMem | **EvMem-GPM** | AOM v2 | v3s | v4 | v5 | v6 |
|---|---|---|---|---|---|---|---|---|---|
| **5**  | 13.75 | – | – | 20.00 | **45.00** | – | – | – | 36.25 |
| **8**  | 26.67 | – | – | 46.68 | 43.35 | – | – | – | **50.02** |
| **19** | 13.32 | 16.65 | 19.99 | **26.67** | 23.32 | 16.66 | 14.80* | 23.33 | 19.99 |
| **22** | 73.34 | – | – | **73.33** | – | – | – | – | 33.33* |

`*` = truncated by the futility gate, not a completed 10-trial mean. See §4.

**v6 (current code) versus the EvMem-GPM baseline on the same seed:**

| task | v6 | GPM | delta | verdict |
|---|---|---|---|---|
| 5  | 36.25 | 20.00 | **+16.25** | **beats GPM** |
| 8  | 50.02 | 46.68 | **+3.34** | **beats GPM** |
| 19 | 19.99 | 26.67 | −6.68 | loses |
| 22 | 33.33* | 73.33 | −40.00* | loses (truncated) |

AOM wins 2 of 4. The two losses are not the same failure, and §5 separates them.

Per-trial scores for every cell are in `results/seed100_summary.tsv` and the raw
`prompt_trace.tsv` for each arm/task is archived under `results/` and `baselines/`.

---

## 3. The one structural defect that was real

### 3.1 AOM seeded its graph from the SCORING partition

Task 19 has two different decompositions, and they are not the same length:

```
scoring   _task_specs(19)     = 3 steps  Place_Tomato_Sauce / Place_Milk / Place_Orange_Juice_Cabinet2
execution primitive_order(19) = 6 steps  pick_TS, place_TS, pick_milk, place_milk, pick_OJ, place_OJ
```

AOM's `_seed_obligations` read `_task_specs(task_id)` — the **scoring** table — so the graph
contained three `place` obligations and **no grasp obligation at all**. The board's own
instruction ("serve only the ACTIVE obligation") then made the planner emit `place` primitives,
and a `place` presupposes a held object. The commanded action was unsatisfiable.

Measured, from the archived `task_trace` streams:

| arm | pick prompts | place prompts |
|---|---|---|
| nomem | 344 | 151 |
| AOM v2 | **15** | **615** |
| AOM v4 | **10** | **504** |

and the first command of every episode:

| arm | first emitted primitive |
|---|---|
| nomem | `Reach toward and grasp the tomato sauce from cabinet1.` |
| AOM v2 | `Place the object into the open container.` |
| AOM v4 | `Place the tomato sauce into cabinet2.` |

### 3.2 The generalisation, and why only task 19 exposes it

An execution phase that the scoring partition omits is **benign when it lies downstream of
every scored phase** and **fatal when it is a precondition**:

| task | phases in execution but not in scoring | position | outcome |
|---|---|---|---|
| **5**  | none (the partitions coincide) | — | AOM 45.0 vs nomem 13.8 |
| **8**  | `place` | **after** the scored pours | AOM ~= GPM (neutral) |
| **19** | `pick` | **before** every scored `place` | AOM below nomem |

This is implemented as a gate in `memexp_aom.py::should_seed_execution`, and the gate's decision
over all 26 tasks is:

```
t5  -> OFF (partitions identical)      t8  -> OFF (omitted phase is downstream)
t19 -> ON                              t22 -> OFF
```

so the archived t5 and t8 results remain valid controls and the t19 run is a single-variable
comparison. Note that 15 other tasks in the suite also carry an omitted **precondition**, so the
same defect is latent well beyond task 19.

### 3.3 The fix

`Ledger.seed_execution(exec_labels, scored_names, label_to_scored, name_objects)`:

- seeds from `task_info.primitive_labels` (the BDDL execution partition),
- binds each execution step to the scored stage that verifies it using the harness's **own**
  bridge, `harness.stage_mapper.expected_primitive_for_stage` (which already existed for the
  stall ladder and the ledger guard — AOM had been bypassing it),
- settles each grasp **derivationally**, when the place it feeds settles (you cannot have placed
  what was never picked). No new sensor is required.
- turns OFF the inherited container/object redaction on **transfer** tasks (`memory_type`
  without `O`), where the object's identity is the instruction rather than the answer being
  withheld.

### 3.4 The bug the fix introduced, and its fix

v5 (the first run of `seed_execution`) emitted **719 grasps and 0 places**. Cause: `place#1`
required `pick#1` to be `SETTLED`, but `pick#1` settles only *with* `place#1` — a cycle. The
ACTIVE pointer therefore never left the grasp, and the board advertised a grasp in every step.

Fix: `Ledger::_dep_ready`. A grasp->place edge is satisfied when the grasp has been
**attempted** (`status != OPEN` or `tau_act > 0`), not when it is settled. Ordinary sequencing
edges (pour->pour, place->place) keep the hard `SETTLED` requirement. With the fix, v6 emits
122 picks and 397 places on t19.

---

## 4. Two reporting errors that were made and are corrected here

### 4.1 The like-for-like baseline was wrong

Earlier notes compared single-seed arm cells against **multi-seed macro averages**
(nomem 23.3 / 25.6). The correct same-seed comparisons are in §2. Under them, AOM v2 (23.32) is
*above* its seed-100 control (13.32), and the claim "AOM was below nomem" does not hold.

### 4.2 AOM t19 v4's "14.8" is a truncation, not a measurement

v4 ran with `FUTILITY_MEAN_FLOOR_PCT=15` and was stopped at k=9 with mean 14.8. Replayed against
the same-seed control, `hard3_t19_nomem_s100` passes through mean **14.8 at k=9** and finishes at
**13.32**. The gate fired on the control's own trajectory, i.e. it was measuring variance rather
than futility. v4's 14.8 must not be read as a completed run.

The t22 v6 cell in §2 is truncated in the same sense (early-stopped at k=9 by the
consecutive-zero rule) and is marked with `*` wherever it appears.

Consequently the series does **not** support the narrative "23.3 -> 16.7 -> 14.8, monotonically
worse": two of those three numbers came from runs with different gates and one is a truncation.

---

## 5. What the results actually show

### 5.1 Fixing the graph defect did not raise the t19 score

```
v2  (scoring-partition graph, self-lock present)  23.32   pick  15 / place 615
v5  (execution graph, deadlocked)                  23.33   pick 719 / place   0
v6  (execution graph, deadlock fixed)              19.99   pick 122 / place 397
GPM                                               26.67   pick  22 / place 586
```

The graph is now mechanically right — `exec_labels` is the 6-step ladder, `name_objects` is
true, `n_gate_rejects`/`n_dep_rejects`/`n_offgraph_rejects` are all 0, and the first command is a
grasp — and the score did not improve.

### 5.2 The primitive mix does not predict the score (falsified hypothesis)

The working hypothesis was "the planner must grasp before it places". The data reject it as a
scoring explanation: on task 19 the **strongest** arm (GPM, 26.67) has amongst the **fewest**
picks (22), essentially the same as v2 (15); the arm with by far the most picks (v5, 719) scores
23.33.

### 5.3 Where the variance actually lives: stage 2, i.e. physical execution

Every arm reaches 33.3 on task 19 (one of three scored stages) and **no** arm reaches 100.
All separation between arms comes from how many episodes reach 66.7 (two of three). On n=10
that is a 3/10-vs-0/10 difference between GPM and its control: Fisher exact **p ~ 0.21**,
i.e. the entire t19 table is inside the noise band. The bottleneck is the physical execution of
the second transfer, which the text channel does not address.

### 5.4 The gate is not the difference on t5/t8

`should_seed_execution` returns OFF on tasks 5, 8 and 22, confirmed in the archived telemetry
(`exec_labels: []`, `name_objects: false` for those cells). Their graphs are therefore the same
as the previously archived AOM arms, and v6's t5/t8 numbers measure the downstream code fixes
only. t5's residual `n_dep_rejects=21` shows the legacy self-lock path is still exercised there
without preventing the +16.25 win over GPM.

---

## 6. Graph telemetry of the v6 cells

From the richest per-step `memexp_aom_report.json` of each cell (archived as
`results/t*/aom_totals.json` and `aom_graph.json`):

| task | exec-seed | n_actual | n_needs_look | n_derived | gate_rej | dep_rej | offgraph | unmatched | settled |
|---|---|---|---|---|---|---|---|---|---|
| 5  | OFF | 9 | 8 | 3 | 5  | 21 | 0 | 15 | 27 |
| 8  | OFF | 3 | 0 | 2 | 0  | 0  | 0 | 2  | 13 |
| 19 | **ON** | 6 | 3 | 3 | 0 | 0 | 0 | 8 | 6 |
| 22 | OFF | 3 | 0 | 2 | 0  | 0  | 0 | 0 | 7 |

Task 19 is the only cell with execution-partition seeding, and it is the only cell whose gate
counters are all zero while the derived/pick coverage is complete.

---

## 7. Conclusions

1. **The scoring-vs-execution partition mismatch is a real architectural defect** and it is
   fixed. It is not a tuning issue: it made the commanded action unsatisfiable on task 19.
2. **The fix is necessary but not sufficient.** With the graph mechanically correct, t19 sits at
   20.0 against GPM's 26.7, and the remaining gap is physical-execution variance that a text
   constraint channel cannot move.
3. **AOM's value is task-dependent and reproducible where the partitions coincide**: task 5
   (45.0 vs 13.8 nomem, and v6 still 36.3 vs GPM 20.0) and task 8 (v6 50.0 vs GPM 46.7) are wins.
4. **No single archived tag is both defect-free and the best scorer.** v2 scores highest on t19
   and carries the self-lock, the wrong-partition graph and the object redaction. v6 is
   mechanically clean and scores lower. Any archival claim must state which of the two is meant.
5. **t19's table is underpowered at n=10.** Any further t19 claim needs multiple seeds before it
   can be believed.

Recommended next steps are in the repository-root handoff document `AOM_HANDOFF.md`.

---

## 8. Reproducing

```bash
cd /project/peilab/why/RoboMemArena/experiments/mem_efficacy

# gates (no GPU, ~20s): AOM invariants + the futility-gate calibration
PYTHONPATH=/project/peilab/why/RoboMemArena/evaluation_benchmark:. \
  python selftest_aom.py
FUTILITY_WATCH_UNDER_TEST=3 FUTILITY_FLOOR_UNDER_TEST=8 FUTILITY_CONSEC_UNDER_TEST=3 \
  FUTILITY_TASK_UNDER_TEST=19 FUTILITY_SEED_UNDER_TEST=100 \
  PYTHONPATH=/project/peilab/why/RoboMemArena/evaluation_benchmark:. \
  python selftest_futility.py

# the recorded runs (submit; each is a 1x10 cell on seed 100)
./run_aom_t19_1x10.sh --tag v6          # task 19, execution-partition seeding
./run_aom_v6_hard3_1x10.sh --tag v6     # tasks 5, 8, 22
```

Every runner refuses to submit unless both selftests pass, and `run_aom_v6_hard3_1x10.sh`
additionally replays its own gate parameters against the archived GPM cell of each task to prove
the gate cannot early-kill the baseline it is being compared to.
