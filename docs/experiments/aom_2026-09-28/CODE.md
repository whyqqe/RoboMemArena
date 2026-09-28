# AOM code map

Everything listed here is snapshotted under `code/` in this package, and also lives (or will live)
in the working tree at `experiments/mem_efficacy/`.

| file | role |
|---|---|
| `memexp_aom.py` | the whole architecture: `Obligation`, `Ledger`, the arbitration law, the graph gate, the board renderer, `should_seed_execution`, `naming_need`/`naming_templates` |
| `memexp_aom_bind.py` | the harness hook: seeds the ledger, tracks the harness's scored stage, rewrites rejected primitives, writes the per-episode report |
| `selftest_aom.py` | the invariants (T1..T15), no GPU, no API quota |
| `selftest_futility.py` | the early-stop rule extracted FROM the runner's own awk, replayed against archived controls (T1..T10) |
| `memexp_evmem.py` | shared scaffolding AOM imports as `G` (intent classification, templates, `admissible_templates`, `action_stem_safe`, ...) |
| `arms/evmem_aom.sh` | the arm: sources `nomem.sh`, sets `MEMEXP_AOM=1` and the knobs |
| `arms/evmem_aom_noderived.sh` | the DERIVED ablation arm |
| `run_aom_t19_1x10.sh` | task 19, 1x10, seed 100, recalibrated futility gate |
| `run_aom_v6_hard3_1x10.sh` | tasks {5,8,22}, 1x10, seed 100, per-task gate calibration |
| `run_aom_t8_1x10.sh`, `run_aom_remaining_1x10.sh` | earlier runners, superseded by the two above |
| `pysite/sitecustomize.py` | import hook that installs the bind layer into the planner process |

---

## 1. Core types (`memexp_aom.py`)

```python
@dataclass
class Obligation:
    oid, kind, mode, predicate, label, status
    stage_name      # the SCORED stage that settles it (empty for a grasp step)
    exec_label      # the EXECUTION step it is (BDDL primitive_order)
    settles_with    # a grasp settles when THIS scored stage settles (derivation)
    needs_look, deps, seq, tau_act, tau_ret, n_new_evidence, ...
```

`Ledger` holds the obligations plus the single arbitration law.

### 1.1 The law, in branch order

```python
if active is None or settled :  advance        # and MOVE the pointer
if force_retrieve            :  retrieve       # attractor-repeat signal
if needs_look and no evidence:  retrieve       # bounded single look
if not DERIVED and not sat_act: act
if DERIVED                   :  derive
if not sat_ret               :  retrieve
else                         :  stagnant
```

`stagnant` is decided **after** the retrieve branch on purpose (floor F): an obligation
saturated on both channels still retrieves, so declaring stagnation never costs the evidence
channel a frame. `STAG_ACT_REARM` re-arms `act` after N stagnant hits so "change strategy" is
actionable.

### 1.2 Seeding: two entry points

| method | source | used for |
|---|---|---|
| `seed(stage_names)` | `_task_specs(task_id)` — the SCORING partition | legacy path; gate OFF |
| `seed_execution(exec_labels, scored_names, label_to_scored, name_objects)` | `task_info.primitive_labels` — the EXECUTION partition | gate ON |

`seed_execution` is what fixes the task-19 defect: it mints the `pick` steps the scoring table
omits, gives each `place` a `deps` edge onto its own `pick`, and gives each `pick` a
`settles_with` pointing at the scored stage of its `place` (derivational settlement).

### 1.3 The gate: `should_seed_execution(scored, exec)`

Returns True **only** when the scoring partition omits a **grasp** phase that **precedes** a
scored phase. Rationale and the 26-task decision table are in `REPORT.md` §3.2. The purpose of
the narrowness is that tasks 5 and 8 keep their previously measured graphs bit-for-bit, so an
AOM run on them stays a single-variable comparison against the archive.

### 1.4 `_dep_ready(dep, dependent)` — the deadlock fix

```
dep SETTLED                                  -> ready        (ordinary sequencing)
grasp -> place and dep attempted             -> ready        (the fix)
otherwise                                    -> not ready
```

Without this, `place` requires `pick SETTLED` while `pick` settles only *with* `place`: a cycle
that pinned the ACTIVE pointer to the grasp for a whole episode (measured: 719 grasps, 0 places).

### 1.5 `set_active_stage` and `active()` self-healing

`set_active_stage` honours the harness's scored-stage hint **only when that stage is reachable**
(`_dep_ready`); otherwise it keeps the earliest open execution step, which is always the true
next action. `active()` re-points itself if the pointer's obligation is settled, so a stale
pointer can never render a NEED for a finished stage.

### 1.6 Object naming

`naming_need(label)` / `naming_templates(label)` are used when `name_objects` is on, i.e. on
transfer tasks (`memory_type` without `O`). Rationale: the redaction clause
("without naming drawer identity") was inherited from the **occlusion** arm, where the object's
identity is the answer being withheld; on a transfer task it *is* the instruction, and redacting
it produced commands like `Place the object into the open container.` with no object named.
When naming is on, `templates_for` uses the execution label rather than
`G.admissible_templates`, which hardcodes "the sauce bottle" and would name the wrong object.

---

## 2. The bind layer (`memexp_aom_bind.py`)

- `_seed_obligations(planner, led)` resolves the scored specs and the BDDL labels, calls
  `A.should_seed_execution`, and either `seed_execution` (mapping resolved through
  `harness.stage_mapper.expected_primitive_for_stage`) or falls back to `seed`. If the mapper
  binds nothing it degrades to `seed` rather than shipping a graph whose stages can never settle.
- the stage hook records the harness's scored `active_stage` and `verified_stages`, then drives
  `led.set_active_stage(...)` / `led.note_verified(...)`.
- `apply_control` rewrites a rejected primitive to the first admissible template of the ACTIVE
  obligation, so a gate rejection is a corrective command (a named grasp) rather than a no-op.
- the report is written as `memexp_aom_report.json.<pid>` with a per-step audit next to it.

Environment knobs (all read in `memexp_aom.py`, defaults in brackets):
`MEMEXP_AOM`, `MEMEXP_AOM_EXEC_SEED[auto]`, `MEMEXP_AOM_DERIVED[1]`, `MEMEXP_AOM_GRAPH_GATE[1]`,
`MEMEXP_AOM_FLOOR[1]`, `MEMEXP_AOM_GATE_ATTEMPTS[3]`, `MEMEXP_AOM_GATE_STALL[3]`,
`MEMEXP_AOM_RET_MAX[6]`, `MEMEXP_AOM_STAG_ACT_REARM[3]`.

---

## 3. Tests

`selftest_aom.py` — every check is a claim the architecture makes, not a smoke test:

| id | claim |
|---|---|
| T1  | ACTUAL obligations form a ladder; PERCEPTUAL ones deliberately do not |
| T2  | the law returns all five verdicts, and the branch ORDER is asserted |
| T3  | floor F: `stagnant` still retrieves |
| T4  | the gate is DERIVED (graph reachability), not pattern-matched, and is conservative |
| T5/T6 | DERIVED mints, recomputes on settle, and the ablation really removes it |
| T7  | the board's sections are a function of `(mode, status)` |
| T8  | floor F as a cadence claim |
| T9  | the module is inert unless the arm asks for it |
| T10 | the report is actually written |
| T11 | **place self-lock invariant** (the bug that produced 47 dep_rejects) |
| T12 | **stagnant is reachable**; T12b it re-arms `act` |
| T13 | open/close ACTUAL no longer starves `act` |
| T14 | **execution-partition seeding**: the graph contains the pick the scoring table omits; the harness hint cannot skip an unmet prerequisite; one scored settle closes both the grasp and the place |
| T14b| object naming ON for transfer, OFF for occlusion |
| T15 | **no deadlock**: place is rejected before any grasp attempt and admissible after one |

`selftest_futility.py` extracts the futility awk programs **from `run_26x1.sbatch` itself** (so a
copy cannot drift) and replays them against archived cells. T7 asserts no archived nomem control
trips the gate at any prefix; T8b asserts the consecutive-zero rule cannot trip the task-8 GPM
baseline while `cz=2` would; T9 reports the full stop list rather than hiding it; T10 replays the
**runner's own parameters** against the same-seed control, which is the test that makes the
mis-set `FLOOR=15` that truncated AOM t19 v4 impossible to submit again.
