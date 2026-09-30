# DIAL — Diagnosis-Informed Action Law

One scheduler for **Planner + VLA strategy + Proactive Memory + Harness**, driven by a
posterior over *why credit is not moving*. The action space contains **no BLOCK**.

```
                 ┌──────────────────────────────────────────────┐
                 │  β_t ∈ Δ⁵   bottleneck posterior             │
                 │  UNREACHABLE / UNKNOWN / UNGROUNDED          │
                 │  UNSYNCED / UNSAFE                           │
                 └───────────────────┬──────────────────────────┘
                                     │  one law   a* = argmax V(a)
      ┌──────────────┬───────────────┼───────────────┬──────────────┐
      ▼              ▼               ▼               ▼              ▼
  RESAMPLE        QUERY          RESYNC          PERSIST       (no BLOCK)
  different       consult        do nothing      keep the       structural
  attempt         memory         (wait λ̂)         attempt        invariant
```

---

## 1. What the archive says is broken

All numbers are measured from the runs already on disk under
`experiments/mem_efficacy/results/`. Reproduce with
`python -m dial.falsify` → `dial/falsify_report.json`.

| arm | t5 | t8 | t19 | t22 | mean | zeros |
|---|---|---|---|---|---|---|
| **GPM** (`evmem_gpm_*`) | 20.0 | 46.7 | 26.7 | 73.3 | **41.67** | 15/40 |
| AOM | 45.0 | 50.0 | 23.3 | 33.3 | **37.92** | 14/39 |
| BOLT (v9 on t8) | 13.8 | 40.0 | 23.3 | 40.0 | **29.27** | 17/40 |

Three findings drive the design.

### F1 — the attempt predicts credit; the sentence does not

`chi2 = 581`, permutation `p < 0.001`, held-out log-likelihood `+0.0043 nats/episode` over
the family-only model. Within one obligation, only the physical attempt changes:

| family | best attempt cell | worst attempt cell |
|---|---|---|
| lift | `pick_up/plain` **0.25** | `grasp/plain` **0.099** |
| pour | `grasp_lift/plain` 0.072 | `pour/plain` 0.037 |
| place | `reach/plain` 0.20 | `pick_up/plain` **0.00** |
| open | `open/plain` 0.032 | `close/plain` **0.00** |

Same obligation, same object, **2.5× difference in credit probability** — and GPM, AOM and
BOLT all vary only the wording. `strategy.py` is what acts on this.

### F2 — gate tightness costs up to 40 points on one task

Within-arm, task 8, architecture fixed (`BOLT_T8_LADDER`):

| version | n | mean | zero frac | what changed |
|---|---|---|---|---|
| v1 | 3 | **0.00** | 1.00 | solvability closure strictest → unreachable deps deadlock |
| v5 | 10 | 30.00 | 0.50 | dependency semantics relaxed |
| v6 | 10 | 26.67 | 0.60 | stage gate tightened without a lift requirement |
| v7 | 3 | **0.00** | 1.00 | served rungs lose explicit lift verbs |
| v8 | 5 | 13.34 | 0.80 | identifier leak: board prints actionable phase names |
| v9 | 10 | **40.01** | 0.40 | non-imperative stage tags + runtime I5 rewrite |

Spread: **40.01 points**, on one task, with the architecture fixed. Across arms the ordering
is also monotone (GPM 4 gate conditions 41.67 > AOM 7 → 37.92 > BOLT 11 → 29.27) but that
comparison is confounded by the whole architecture and is reported as consistent, not as
proof.

**The mechanism.** A gate is a rewrite map `R: P(o) → P(o)` onto the *same* obligation. If
no proposal in `P(o*)` is physically executable, `R` has a closed absorbing class that emits
no credit, and the only exit is the arm's own stall counter. Measured signatures: BOLT t8 v7
`0.0, 0.0, 0.0` and v8 `0.0, 0.0, 0.0, 66.7, 0.0`; AOM t22 v6 has **five 0.0 in nine**
episodes. DIAL's action space removes the rewrite, so the class is unreachable.

### F3 — the scorer lags longer than every arm's patience

Measured enacted→credited lag over the archived credited attempts: **median 25, mean 51.0,
p90 130, max 685 steps**. The horizons the arms act on:

| arm | stagnation horizon | fraction of *successful* attempts that would be cancelled in flight |
|---|---|---|
| GPM | 8 | **79.8%** |
| AOM | 12 | **66.0%** |
| BOLT | 80 | 17.0% |

Treating the scorer as a pure delay and the arm as a discrete controller with sampling
period `h` and lag `ℓ ∼ F`, throughput ∝ `F(h)`. Every arm with `h < median(ℓ)` discards
more than half its own work. **No archived arm has an action for "the scorer is simply not
due yet."** That action is `RESYNC`.

### F4 — the objective is escape, not the mean

Episode score is bimodal: **39% exactly 0.0**, 35% ≥ 66.6, and the largest empty band is 25
points wide. Score ≈ (fraction of scored stages credited), and the 0.0 mass is the first
scored stage (`01_Lift_*`) never being credited. The KPI is therefore `P(escape 0.0)`, and
every design choice below is judged on converting zeros.

---

## 2. The architecture

### `types.py` — vocabulary
`Strategy = (verb_form, approach, hold_steps, text)`; the key is deliberately coarse because
110 archived episodes cannot support a finer partition. `RewardRecord` is one attributed
attempt. `has_lift_word()` is the load-bearing predicate for tasks 8/22: an attempt without
a lift word cannot be credited on `01_Lift_*` no matter how good the grasp is.

### `attribution.py` — the reward channel that did not exist
The harness already logs everything credit assignment needs and nothing ever read it back:

```
harness_memory.json -> episode_evidence:  {step, action, instruction, outcome,
                                           notes: "01_Lift_Tomato_Sauce:80"}
harness_memory.json -> attempts[]:        {completed_stages: [...], stalled_stage: "..."}
```

Attribution is **within-stage**: a stage that stalls ten times and is then credited yields
nine labelled failures and one labelled success for the same obligation — the contrast a
strategy posterior needs and that no per-episode score can express. Yield:
**9 262 labelled attempts, 465 credited** (up from 6 336 before the `evmem_gpm_*` run
discovery was fixed).

### `bottleneck.py` — β_t
A Bayesian filter over five causes, from coarse booleans the harness already produces.
Two details are load-bearing and were found by selftest, not by design:

* **`overdue_no_credit` is a separate observable from `enacted_no_credit`.** The latter
  means "acted, scorer not due yet" and supports `UNSYNCED`; past the hold horizon the same
  silence means the opposite. Feeding one observable forever pins the posterior on
  `UNSYNCED` and the scheduler can never move — an absorbing state introduced by the
  estimator.
* **`leak = 0.05`.** A pure multiplicative filter drives losing causes to exactly 0, after
  which contradictory evidence cannot restore them. Also an absorbing state, inside the
  estimator. The leak keeps it recoverable.

`lambda_hat` = median of observed lags (robust to the 685-step tail); `hold_horizon` =
nearest-rank p90 (the window during which an attempt may still be in flight);
`escape_horizon` = strictly above `hold_horizon`. Using `k * lambda_hat` for the hold window
would flag every above-median attempt as overdue before its credit arrived — the F3 failure,
reproduced inside the fix.

### `strategy.py` — attempt space + bandit
Hierarchical prior `θ_task ∼ N(θ_family, σ)`, `θ_family ∼ N(θ_global, σ)` with partial
pooling, fitted from the archive. Selection is stratified Thompson sampling with the single
exclusion `s ≠ s_{t-1}` — and that exclusion is the only gate in the architecture. It
removes nothing from the environment; it only changes which attempt goes next.

Invariant asserted by selftest A12: every template must classify as the cell it declares,
because `warm()` reads the archive through the *text* while live selection indexes the
*declared tuple*. A mismatch silently voids the cross-task transfer — and it caught three
pour templates that differed only by an adverb and collapsed into one cell.

### `policy.py` — the law
`a* = argmax_a  E[Δ credited | a] + κ·H(β_t | a) − c(a)`, with `E[Δ credited | a]` taking
its value from the posterior because **each cause has exactly one addressing action**. That
is why one law can replace four schedulers without a negotiation protocol: the posterior is
the consensus.

Reward bookkeeping is per **family**, not per single attempt: a task steps
`01_Lift → 02_Pour`, so credit for `01` arrives while the obligation has already advanced.
A single `last_key` mis-attributes it.

### `bind.py` — the integration, constrained by three measured defects
1. **Never rewrite the VLA prompt.** `HarnessController.override_vla_prompt` documents job
   582641: rewriting pinned the VLA on the whole-task sentence for 99% of chunks and took
   t22 from 100 to 0 (−11..−22 pp). DIAL wraps it as an **observer** and returns it
   byte-identical; `selftest D1` asserts equality even when DIAL's own code throws.
2. **Never wrap `infer_primitive_via_api`.** BOLT's own bind records that its arbiter hook
   was **inert for every archived episode** (pure-kwargs call, no planner handle), and the
   function is shared with the memory decision, the PMH decision and the stage verifier.
   DIAL does not touch it — there is no rewrite path at all.
3. **Never leave per-episode state in a module global.** BOLT t8 v4 produced
   `0, 66.7, 0, 0, 0` because `reset_episode` left `verified_stages` behind. All of DIAL's
   per-episode state lives in a thread-local `_Ctx`.

Injection is into the **planner channel** (`_build_messages`), and it prescribes the attempt
as a *class* (`verb=… approach=… hold=…`) with "do not copy this line". BOLT v8 showed what
happens otherwise: a board printing `[n3] grasp tomato sauce [OPEN]` was echoed verbatim, the
robot grasped without lifting, `01_Lift` never fired, three seeds scored 0.0.

---

## 3. Predictions, and what would refute them

1. t8 ≥ GPM's 46.7 **and** t22 ≥ 73.3.
2. **The signature is the minimum, not the maximum.** On the seeds where the other arms hit
   exactly 0.0, DIAL must be > 0. If the min does not move, the absorbing class was not
   actually removed and the architectural claim is wrong. *This is the test that matters.*
3. `RESYNC` fires on a non-trivial fraction of steps and `lambda_hat` converges near 25.
4. The `lift` family drifts toward `pick_up/plain` (`pick_up` 0.25 vs `grasp` 0.099) as
   trials accumulate.

**Honest ceiling.** The VLA is frozen; DIAL cannot make the robot more capable. It can only
recover the two things the archive has already priced: rejection waste (up to 40 points,
F2) and in-flight cancellation (up to 80% of successful attempts, F3). If a scene is one
where no approach completes the lift at all, no scheduler helps, and DIAL's only contribution
is not spending the budget on it.

---

## 4. Files and how to run

| file | role |
|---|---|
| `types.py` | `Strategy`, `RewardRecord`, cause constants, verb/approach classifiers |
| `attribution.py` | archived runs → labelled reward stream (9 262 / 465) |
| `bottleneck.py` | β_t filter, `lambda_hat` / `hold_horizon` / `escape_horizon` |
| `strategy.py` | attempt-cell space + hierarchical prior + Thompson sampling |
| `policy.py` | the single scheduling law |
| `bind.py` | planner-channel injection, observer-only VLA hook, thread-local state |
| `falsify.py` | T1..T4 falsification suite → `falsify_report.json` |
| `selftest.py` | 78 headless checks (no GPU, no network) |

```bash
cd experiments/mem_efficacy

# evidence (no GPU)
python -m dial.falsify      # T1..T4, writes dial/falsify_report.json
python -m dial.selftest     # 78 checks, exit 0 = clean

# the arm is activated exactly like its siblings, via pysite/sitecustomize.py
export MEMEXP_DIAL=1

# submit (dry-run first; the runner refuses to submit unless every gate passes)
./run_dial_hard3_1x10.sh --dry-run
./run_dial_hard3_1x10.sh --tag v1
```

The runner's pre-submit gates are: AST compile of the evaluator + every `dial/*.py`;
`dial.selftest`; `dial.falsify`; and a futility replay proving the early-stop parameters do
not kill archived GPM on the same seed.

---

## 5. Bugs the selftest found (recorded so they are not reintroduced)

| # | bug | consequence if shipped |
|---|---|---|
| 1 | `advance()` fed `enacted_no_credit` forever | posterior pins on `UNSYNCED`; scheduler can never explore |
| 2 | `_emit(A_PERSIST)` wrote `last_key = ("","",0)` | exclusion set destroyed; next decision can re-pick the failed attempt |
| 3 | `decide()` fallback path did not exclude the last attempt | same-obligation absorbing loop, the exact thing the architecture forbids |
| 4 | escape bound `2 * lambda_hat` | fires at age ~8 against a real lag of 25–130 → cancels correct in-flight work |
| 5 | no `leak` in the filter | losing causes saturate at 0 and can never be recovered |
| 6 | `warm()` key space ≠ selection key space (pour) | cross-task transfer silently void |
| 7 | `_default_specs` had one candidate | "next attempt must differ" unsatisfiable → guaranteed absorption |
| 8 | `_RUN_RX` skipped `evmem_gpm_*` | GPM — the key comparison — absent from the corpus; wrong arm mean reported |
| 9 | template "same **place**" classified as the `place` family | wrong cell; prior misalignment |
| 10 | the `hold_horizon` rewiring silently did not apply to `policy.py` | the RESYNC veto and the `within` observable still used the **median** lag, so half of all in-flight attempts were declared overdue; the scheduler cancelled a correct attempt at step 26 of a 30-step lag (selftest C3). A bulk string patch that matches nothing does not error — verify patches by reading the file back. |
| 11 | `test_sim_resync...` depended on a random Thompson draw | flaky test; rewritten deterministically by issuing the good attempt explicitly, with a separate control arm reproducing GPM's 8-step cancellation |
| 12 | `install()` did `from harness import controller` eagerly, swallowed `ModuleNotFoundError`, and set `installed_at` so it never retried | **cost a real submission.** `arms/dial.sh` puts only `pysite` ahead of the submitting shell's PYTHONPATH, and the evaluator adds `evaluation_benchmark` to `sys.path` only at run time. DIAL bound NOTHING; the arm ran unbounded and scored `0,0,0`, indistinguishable from an architectural failure. Fixed by binding lazily through a meta-path finder (the mechanism `memexp_evmem_bind.py` already used), asserted by selftest D9 under the exact failing PYTHONPATH. |
| 13 | the install had no receipt | `_write_report` is only reachable from `_build_messages`, so a broken install archived nothing at all — the `0,0,0` run could not say why. A receipt is now written at install time and on every bind (selftest D10). |
| 14 | credit was read from `override_vla_prompt`'s `stage_done` argument, which in this harness is **always all-False** | it is a placeholder for the stages *before* the current index (`{spec.name: False for spec in stage_specs[:stage_idx]}`, `controller.py:272` and `:634`). A live run reported `n_stage_obs=375` with `n_credit_events=0`: the scheduler received **no reward at all** for the whole run, so it could never learn that an attempt was good and resampled forever. Fixed by taking credit from `HarnessController.on_episode_end`, which carries the true `stage_done` mapping; guarded by selftest D11. |
| 15 | the RESAMPLE directive ordered the planner to vary | *"Render a DIFFERENT physical attempt … Do not reuse the previous wording"*. Everything measured says the opposite: the lag to credit has median 25 / p90 130 steps, and **every credited attempt in the archive was earned by repeating one primitive** until the stage fired. Ordering an LLM planner to keep varying destroys exactly that mechanism. The directive is now HOLD-first ("repeat it"), and RESAMPLE names a class without forbidding repetition; guarded by D12. |
| 16 | changing an attempt was a *score* term, so a confident posterior could outvote patience | T3 is now a **rule** (`policy.py` step 2c): inside `hold_horizon` the scheduler may not resample, whatever the belief says. The only way out is the bounded `escape_horizon`, so non-absorption is preserved while the in-flight window is protected; guarded by D13. |
| 17 | `issued_at` was refreshed only when a **new** attempt class was picked, and `note_enacted` was called only on RESAMPLE | so the one clock measured *steps since the aim last changed*, and the scheduler treated its own holds as staleness. A live run reported `now=1931` on a ~1000-step episode with `forced_escape=7` — it kept switching aim on exactly the episodes the baseline wins by repeating. Fixed with **two clocks**: a refreshable patience clock (`age`, refreshed by any re-enactment, including a plain repeat) and an absolute escape clock (`age_total`, measured from when the attempt class was first chosen and never refreshed). Patience is now bounded by construction instead of by a clock that punishes holding; guarded by D14. |

### Lesson recorded

Defects 10, 12, 13, 14 and 15 share one root cause: **a failure that produces no error.** A bulk
patch that matches nothing; an `except` that swallows an import error after marking itself done; a
diagnostic write that is unreachable when the thing it reports on is broken; a credit channel read
from an argument that is always all-False; a directive that silently inverts the behaviour the data
calls for. None of these raise. All were caught by tests that assert on *observable consequences* —
a bound hook, a written file, a credited reward, a non-varied instruction, a non-replaced attempt —
rather than on return values. Keep writing that kind of test.

Detection that worked, and should be reused: **compare against a control arm on the same seed.** DIAL
scored `0,0,0` on t8 seeds 100/101/102 while GPM, BOLT and AOM all scored 66.7 on seed 101. A single
arm's mean cannot distinguish "hard task" from "broken arm"; a same-seed cross-arm comparison can.
