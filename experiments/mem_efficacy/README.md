# mem_efficacy — does a memory mechanism improve control?

An experiment on the RoboMemArena dataset. The claim under test: **giving the Planner access
to past observations improves task performance, holding everything else fixed.**

Everything else means:

| Component | Choice | Frozen? |
|---|---|---|
| Evaluation protocol | RMA official, 26 tasks, seed 100 | yes (upstream) |
| Harness | RMA's HarnessVLA-style recovery layer | yes |
| Planner | `gemini-3.8-flash` over an external API | yes (fixed triple) |
| Low-level VLA | `pi05_robomemarena` from `checkpoints/PrediMem` | yes (never fine-tuned) |
| Memory | **the experimental variable** | — |

---

## 1. Directory layout

```
experiments/mem_efficacy/
├── README.md               this file
├── arms/
│   ├── official_protocol.sh   the RMA official protocol, single source of truth
│   ├── _memexp_pysite.sh      shared helper (NOT an arm, and never read as one)
│   ├── nomem.sh               BASELINE — no memory
│   ├── pushmem.sh             TREATMENT — RMA-official-style push memory
│   └── pullmem.sh             TREATMENT — small push + agent-callable memory tools
  ├── memexp_substrate.py     the PULL arm's fact bank (deterministic, model-free writes)
  ├── memexp_tools.py         tool registry A — address-based (query_world / list_unbound / list_facts)
  ├── memexp_tools_search.py  tool registry B — A plus search_memory(query); SELECTED BY `pullmem`
  ├── memexp_bind.py          installs the PULL arm's three hooks; strict no-op when disabled
├── memexp_memfix.py        corrects the harness memory TEXT (D1/D2/D3); no-op when disabled
├── pysite/
│   └── sitecustomize.py    activates the hooks from PYTHONPATH, with no shared-code edit
├── selftest_pull.py        pre-run gate: drives the tool loop with a scripted model
├── selftest_memfix.py      pre-run gate: reproduces the memory-text defects, then shows them fixed
├── diagnosis_590799/       archived evidence for the defects above (fixtures the gates use)
├── run_26x1.sbatch         submit one or more arms
├── validate_arm.py         preflight (before spending GPUs)
├── census_channels.py      post-run census (from artifacts)
└── results/                per-job outputs
```

Each arm is **one self-contained file**. `pushmem.sh` and `pullmem.sh` both source `nomem.sh`
and then change only the keys they declare in `MEMEXP_EXPECTED_DIFF`. Nothing an arm does can
reach into another arm: the runner sources one arm per evaluation and `nomem.sh` begins by
purging every `PMH_*` / `PACT_*` / `SEAM*` / `MEM_*` variable that a previously exported shell
— or a previously sourced arm — may have left behind.

The two memory arms go further and touch shared code paths, but do so **without editing any shared
file**: `pysite/sitecustomize.py` installs `meta_path` hooks that wrap functions on
`harness.api_vlm_planner` (`memexp_bind`) and on `harness.memory_reason` / `harness.controller`
(`memexp_memfix`) *after* those modules are imported. Why a hook and not a plain import is not a
style choice — see the note at the end of §2. Each hook is installed only when its own arm flag is
set, so for `nomem` the modules are not merely disabled — they are never imported. That is asserted
directly, not assumed (see §4).

---

## 2. The arms

Three arms. `nomem` and `pushmem` differ in one thing; `pullmem` asks a second, separate
question, and the relationship between all three is spelled out at the end of this section.

| Arm | `HARNESS_VLM_CONTEXT` | `VLM_USE_KEYFRAME_MEMORY` | `MEM_STAGE_ANCHOR` | `MEMEXP_PULL_ENABLE` | Memory the Planner receives |
|---|---|---|---|---|---|
| `nomem` | 0 | 0 | 0 | — | none |
| `pushmem` | 1 | 1 | 1 | — | text evidence **and** historical frames, pushed every step |
| `pullmem` | 1 | 0 | 0 | 1 | a **small** push, plus memory tools the Planner may call itself |

### `nomem` — the baseline

The official protocol, the harness, the Planner and the frozen VLA, minus memory. The whole
mechanism is two switches going off:

* `HARNESS_VLM_CONTEXT=0` → `HarnessController._refresh_vlm_context()` returns early with
  `vlm_context = ""`, so `compose_planner_context()` returns `""` and the
  *"Harness memory context"* prompt block is never emitted.
* `VLM_USE_KEYFRAME_MEMORY=0` → `J_hist` stops accumulating and the *"Historical keyframes
  from moments before the current step"* prompt block is never emitted.

`N_RECENT` / `K_MAX` / `D_MERGE` keep their **official** values, so both arms see an identical
*recent* window — that is the current observation, not memory.

### `pushmem` — the treatment, push-style

Everything in `nomem`, plus the memory that RMA **pushes** into the prompt without the Planner
asking. No tool call, no retrieval query, no decision: the Python layer decides, every step.

This is not an invented mechanism. It is assembled from RMA's own code paths. The two prompt
blocks already exist in `api_vlm_planner._build_messages`:

```
"Harness memory context (read-time evidence; use with historical keyframes):"   # :2754
"Historical keyframes from moments before the current step ... (N timesteps):"  # :2779
```

Neither block is gated on anything the Planner emitted, and the official README describes the
official mechanism the same way: *"unlimited historical keyframes (`K_MAX=0`)"* and
*"task-conditioned VLM prompting with historical keyframes and recent visual context"*.

**Why `MEM_STAGE_ANCHOR=1` is required.** The official protocol leaves this knob unset, so it
defaults to 0, and in that mode the image bank is

```python
K_indices_abs = build_visual_memory(J_hist, ...)     # api_vlm_planner.py:4770
self.J_hist.append(j_abs)                            # api_vlm_planner.py:4714
```

`J_hist` holds the Planner's **own** `keyframe_positions` nominations. The official local
PrediMem VLM is trained to emit that field; a general API Planner is not. So `J_hist` stays a
list of empty lists, the bank is empty at every step, and the image block is never emitted. The
code already records this state (`api_vlm_planner.py:1489-1491`, and `kf_n=0` from job 586700).

`MEM_STAGE_ANCHOR=1` switches the builder to RMA's `merge_keyframe_bank`, still official code,
which takes its candidates from two signals the Planner cannot suppress: **stage boundaries**
appended by the eval loop (`eval_fullvlm26_async_vlm_vla.py:1745-1746`) and **subtask changes**
(`:1226-1227`). With `bank_max=0` no cap applies, and Planner nominations are still unioned in
when they exist — so this is a strict **superset** of the official behaviour, not a replacement.
`validate_arm.py` CHECK 7 proves all three of those properties.

  ### `pullmem` — the treatment, agent-retrieval
  
  `nomem` plus a **small** always-on push and a set of tools the Planner may call **while planning**.
  
  #### Which tool registry it runs, and why that is not a detail
  
  `pullmem` selects its registry explicitly, via `MEMEXP_TOOLS_MODULE`:
  
  | registry | tools | selected by |
  |---|---|---|
  | `memexp_tools` | `query_world` (address) / `list_unbound` / `list_facts` | nothing by default — kept for the ablation |
  | `memexp_tools_search` | the above **plus** `search_memory(query)` — free text | `pullmem` (the default) |
  
  Address-only retrieval is not a neutral default. `query_world` takes an **address**, an exact
  string the model has to copy out of the pushed block, which inverts the purpose of retrieval: a
  model that can name `contents(middle drawer)` can usually already see what it refers to, and a
  model that cannot name it has no way to reach the record at all. The archived run measured exactly
  that shape — 5 calls, every one of them on an address that had just been advertised. The address
  registry is kept, unchanged, so the ablation is still available; it is simply not what `pullmem`
  means any more.
  
  `search_memory` is deterministic token-overlap scoring (no embeddings, no sampling) over every
  record's address **and** value, with ties broken on recency. Indexing the value as well as the
  address matters: an action mints `contents(middle drawer)` with an *empty* value (the answer needs
  looking at) while a stage record mints `address="stage"` with the stage *name* as its value —
  indexing addresses alone would make every stage unfindable. The result is capped
  (`MEMEXP_SEARCH_MAX`, default 4), because a search that can return the whole bank is a push
  wearing a tool's clothes.
  
  The design point is that `pushmem` has to decide in advance what will matter, and it can only do
  that through a proxy — stage boundaries, subtask changes. That fails whenever relevance is
determined *later*. The Occlusion tasks are exactly that case: three drawers are each observed
once, and *which* one is the target depends on which turned out to be non-empty, a fact that does
not exist until all three have been looked at. No a-priori salience rule can rank the second
observation above the first, because at that moment neither is more important.

So this arm splits the two roles `pushmem` conflates:

* **getting** evidence happens on demand, chosen by the Planner, who is the only party that knows
  which address matters right now;
* **keeping** it available is still the harness's job, so the push remains.

That is the hybrid: the push makes the pull *discoverable* — it lists the addresses that exist —
while the choice of what to open stays with the Planner.

**What is pushed** (capped on both axes, never an answer):

* the last few actions the Planner itself emitted, with their step numbers;
* the addresses that have been observed and are **readable now**, with their frame counts;
* the addresses that have been observed but are **not readable yet**, named explicitly as such.

Pushing the address list is what makes the pull usable. It converts *"notice that you are missing
something"* — a judgement PMH delegated to the model and got `none` on 85% of the time — into
*"read a list of open addresses"*, which is a structural fact.

**What is not pushed:** the keyframe bank. `VLM_USE_KEYFRAME_MEMORY=0`, so the only images the
Planner sees beyond the live window are the ones **it asked for**. That is what makes the census's
`n_frames_served_off_context` interpretable as gain rather than as volume.

**How the Planner asks.** It emits strict JSON instead of a primitive:

```json
{"tool": "query_world", "address": "contents(middle drawer)"}
{"tool": "list_unbound"}
{"tool": "list_facts"}
```

`query_world` returns the record **and** the frames to look at, in one call. It deliberately does
not answer the question: the Planner is a VLM, so it resolves the address by reading the images.
That keeps the write path model-free and puts the perceptual judgement where it belongs.

**Why one call and not two.** PMH's `search_memory` returns text only and never loads images, so
a Planner that needed to *see* something had to call `search_memory`, read a segment id out of
the text, then call `retrieve_visual(segment_id=...)`. That is two model decisions per fact, in a
regime where the model picks `none` most of the time — so the probability of finishing the
sequence is the product of two small numbers. Here it is one decision and one round trip.

**How it attaches without editing shared code.** `experiments/mem_efficacy/pysite/sitecustomize.py`
installs a `meta_path` hook from `PYTHONPATH`, which post-processes `harness.api_vlm_planner` when
it is imported and wraps exactly two things:

| Hook | Why there |
|---|---|
| `ApiMemoryPlanner._build_messages` | the once-per-step prompt construction: it observes the previous primitive and appends the push + tool spec |
| `harness.api_vlm_planner.infer_primitive_via_api` | the narrowest point every plan step must pass; wrapping it leaves the rest of `infer_sync` byte-identical |

The wrapping is scoped by **object identity** on the message list, because
`infer_primitive_via_api` has four call sites and only the main planning call consumes the list
`_build_messages` just produced. Without that scoping the loop would also wrap the stage-boundary
call and the PMH/SEAM decision calls, injecting a plan step's context into prompts that have no
live window.

When `MEMEXP_PULL_ENABLE` is unset — i.e. for `nomem` and `pushmem` — the hook is not even
installed: `memexp_bind` never enters `sys.modules` and the planner module is untouched. This is
verified directly rather than assumed; see *Gates* below.

**The single-variable control.** `MEMEXP_PULL_TOOLS=0` keeps the small push and disables the tool
loop, so `pullmem` vs `pullmem`-with-tools-off isolates *agent-chosen dereferencing* from
*the pushed content*. It is an override rather than a fourth arm so the two conditions cannot
drift apart:

```bash
MEMEXP_PULL_TOOLS=0 ARM_OVERRIDE=pullmem sbatch experiments/mem_efficacy/run_26x1.sbatch
```

### How the three arms relate

`pullmem` vs `nomem` measures memory **with agent retrieval**. `pullmem` vs `pushmem` is **not** a
single-variable contrast, and must not be read as one: the two arms differ in *what is stored*
(a step-addressed fact bank vs. RMA's evidence text plus a keyframe bank) as well as in *who
selects*. The clean one-variable comparisons available are:

| Question | Comparison |
|---|---|
| does any memory help? | `nomem` → `pullmem` |
| does agent retrieval help, given identical pushed content? | `pullmem` with `MEMEXP_PULL_TOOLS=1` → `0` |
| does push-style memory help? | `nomem` → `pushmem` |

**A note on the arm that is not here.** The flag combination
`HARNESS_VLM_CONTEXT=1 VLM_USE_KEYFRAME_MEMORY=1 MEM_STAGE_ANCHOR=0` is exactly the prior HM
baseline's memory configuration, i.e. the "official flags only" variant. It is deliberately not
given an arm of its own: under an API Planner its image channel is empty at every step, so it
duplicates `pushmem` on the text channel and contributes nothing on the image channel. Its
measured behaviour is recorded instead, and the census still reports **VOID** — not "no effect" —
if any future arm reverts to that combination.

---

### The memory-content correction — `memexp_memfix.py` (shared by both memory arms)

Job 590799 measured a memory arm that scored **worse** than its own no-memory baseline (−16.68 pp
over the 8 tasks both finished). That was not "memory does not help": it was three defects in the
*content* of the memory text, all of which made the text assert things nobody had observed.

| | Defect | Evidence from 590799 |
|---|---|---|
| **D1** | The harness's own recovery **guesses** were printed as top-ranked *evidence*. The query was `f"stall {stage} {subtask}"` and the literal token `stall` matched every failure record, because the harness writes its own diagnostics into `notes` (`stall_recovery:<stage>`). A `subtask_override` — a guess, stored `outcome="planned"` — scored 5/5 and appeared under **"Recent episode evidence:"**. | On task 8 the four retrieved rows were *stall, subtask_override, subtask_update, subtask_update*; three carried the same instruction and it was the guess. The Planner then emitted that guessed primitive **eight consecutive times**; the task went 66.7 → 0.0 and its Planner call count 2 → 10. Across the 8 common tasks the call-count ratio tracks the score change (≈1 → no change; 3.5–5.0 → −25 to −66.7 pp). |
| **D2** | `_refresh_vlm_context` **accumulated** instead of refreshing (`if self.vlm_context: ctx = self.vlm_context + "\n\n" + ctx`, upstream `eb86819`). | 15 calls → **3594 characters**, the same three static rules repeated 15×. `nomem` is immune by construction (it returns early when the channel is off), so memory alone was penalised for a fault that only appears when memory is on. |
| **D3** | Static `global_rules` were 80 % of a fresh block. | *Subsumed by D2*, not independently fixed: the harm was repetition. The fix only moves them to the end of the block. |

**A second feedback loop, in the recovery rung.** `candidate = expected_primitive_for_stage(...)`
is a stage-name→primitive mapping — a guess — yet it was rendered as
*"Suggested primitive for recovery: 'X'. **Prefer this** …"*. Obeying that instruction is what
wrote an override back into memory, where D1 then promoted it to "evidence" at the next stall. The
loop has two stages, so fixing retrieval alone would have left the directive half intact; the
renderer now labels it a **hypothesis with an explicit falsifier**.

**The principle is one rule:** the memory text must distinguish what was *observed* from what was
*guessed*. The fix does not add, remove or re-rank content toward a better score — the `planned`
rows are still shown, the `active` rows are still shown, the global rules are still shown. What
changed is that they are no longer presented as the same kind of thing.

**Confirmed by measurement, not assertion** (`selftest_memfix.py`, GATE 0c):

| | original | corrected |
|---|---|---|
| context after 12 refresh calls | 3594 → 3810 chars (monotonic) | 285–341 chars (bounded, no trend) |
| duplicate rows in the retrieved evidence | 1 | 0 (collapsed) |
| guess presented under "episode evidence" | yes | no — separate, marked unverified |

**Both memory arms share this correction**, so `pushmem` and `pullmem` differ only in their declared
channels rather than in which harness code they run. `nomem` does not load the module at all — CHECK
12 asserts it is not even present in `sys.modules`, not merely inert.

**Cost of this being wrong in either direction.** The first version of the module imported the
harness eagerly from `sitecustomize`, where the harness directory is not yet on `sys.path`. It
raised `ModuleNotFoundError`, `sitecustomize` swallowed it onto a stderr nothing reads, and the
correction was configured, documented, switched on and **completely inert** — while every other
check passed. CHECK 12 (installed *and live*) and GATE 0c (behaves correctly, and reproduces each
defect on the original code first) exist so that a correction which never engages cannot be
mistaken for one that does not help.

---

## 3. Official alignment

`arms/official_protocol.sh` holds every protocol knob, copied from
`slurm/benchmark/reproduce_all26_1seed.sbatch` — the artifact the repo README names as
*"PrediMem on all 26 tasks (official protocol)"*. It is the only place the protocol is written
down, and the runner does not restate it.

This matters because the official runner resolves each knob as `${VAR:-official_default}`: an
arm that exports a knob **silently overrides** upstream. That already happened here. The harness
arm `api_qwen_harness_v21_redact` — the HM baseline every past memory delta was measured against
— exports `N_RECENT=7 / K_MAX=8 / D_MERGE=4`, while the official protocol is `5 / 0 / 6`. Those
HM numbers are therefore *not* official-protocol numbers. `validate_arm.py` CHECK 3 parses the
official sbatch and fails on any such divergence, so upstream changing cannot be absorbed
silently.

**Two deliberate deviations from the official protocol:**

1. **The Planner is an external API model** (`gemini-3.8-flash` via CloseAI) rather than the
   local PrediMem VLM. This is the point of the experiment: memory is the only thing that
   varies, and the Planner stays a strong fixed generalist. One consequence is the official
   image channel having nothing to key off, which is what `MEM_STAGE_ANCHOR` addresses above.
2. **The three memory keys** that separate `nomem` from `pushmem`, and the two that separate it
   from `pullmem` (`HARNESS_VLM_CONTEXT`, plus `PYTHONPATH` — the latter being plumbing for the
   binding, declared rather than silently exempted so CHECK 1 keeps its force). CHECK 1 asserts
   that these are the *entire* deviation set between each arm and the baseline.

---

## 4. Running

```bash
cd /project/peilab/why/RoboMemArena
python3 experiments/mem_efficacy/validate_arm.py          # preflight, cheap, do this first
ARM_OVERRIDE="nomem pushmem pullmem" sbatch experiments/mem_efficacy/run_26x1.sbatch
```

`validate_arm.py` runs first inside the job too (GATE 1), after a planner-liveness gate (GATE 0).
When `pullmem` is in the arm list, a third gate runs (GATE 0b) that drives the tool loop with a
scripted model. When either memory arm is in the list, a fourth (GATE 0c) executes the memory
content correction against an archived episode. The job aborts before touching a GPU if any gate
fails.

Afterwards:

```bash
python3 experiments/mem_efficacy/census_channels.py \
  experiments/mem_efficacy/results/mem_efficacy_<jobid>
```

### Gates

* **GATE 0 — planner liveness.** A 200 response is not enough. `gemini-3.8-flash` is a reasoning
  model whose chain of thought is spent *inside* `max_tokens` before the JSON, so too small a
  budget yields `finish_reason=length` and a string `json.loads` rejects; the caller then gets
  `None` and the stall-recovery rung becomes a silent no-op while the arm still reports
  `HARNESS_SUBTASK_OVERRIDE=1`. So the gate exercises the **real** recovery-rung code path at the
  real budget. `PLANNER_API_MAX_TOKENS` is 4096 because 2048 is a measured failure, not a guess.
* **GATE 1 — arm integrity.** Official alignment, no memory leak in the baseline, and the
  functional probes. CHECK 12 additionally asserts the memory-content correction is **live** in the
  arms that declare it and **absent** (not merely inert) everywhere else.
* **GATE 0c — the memory-content correction.** CPU only, no API calls. Runs `selftest_memfix.py`,
  which reproduces each of the three content defects on the **original** code and asserts they are
  gone on the corrected path. GATE 1 can only show that the correction installed; this shows it
  does what it claims. A gate that asserts only the fixed output cannot distinguish *fixed* from
  *the defect never reproduced here*, which is why the positive control runs first.

### Reading a result

A score is only interpretable next to the census. Four outcomes are distinguished:

| Census says | Meaning |
|---|---|
| PASS | the channel was live as designed; the score is about memory |
| FAIL | the baseline leaked, or a channel meant to be live was dead; the run is invalid |
| **VOID** | the channel was on but empty on every step; the run says **nothing** about it |
| **WARN** | the measurement is scoped: read the number, but not as the thing it looks like |

VOID is not a negative result. Reporting it as one is the specific error this census exists to
prevent.

### Channel C, and why the PULL arm needs its own census

Channels A and B are read from the arms' configuration and the planner's trace. Channel C — the
agent-called tools — is read from a report the binding writes **from inside the evaluation
process**, and it answers a question neither of the others can: `MEMEXP_PULL_ENABLE=1` proves the
loop was *installed*, GATE 0b proves it *works*, but only the report says whether the Planner ever
*called* it.

That distinction is the whole reason channel C exists. A run with the loop installed and zero
tool calls produces a score indistinguishable from *"memory does not help"* — and that is exactly
what happened to PMH, whose read path ended with zero searches in it while every flag reported
enabled. So:

* **0 plan steps → FAIL, and it is not a finding about the Planner.** The wrapped
  `_build_messages` was never called in the evaluating process, so nothing was pushed and no tool
  was ever offered. The arm did not run as configured; read its score as a broken run.
* **0 tool calls with steps > 0 → WARN, not FAIL.** The tools were offered on every one of those
  steps, so this is a finding about the Planner's propensity, not a broken mechanism. But the
  arm's number then measures the **small push only**, and nothing may be attributed to agent
  retrieval.
* **tools called, 0 frames served → FAIL.** The pull moved no pixels, so the arm is equivalent to
  the push-only control. The census separates the two possible causes: *every* dereference coming
  back *"not observed yet"* is a **timing** problem (the Planner is asking before the frames
  exist), not a filter problem, and the fix is different.
* **`cap_unresolved` > 0 → FAIL.** The model would not stop calling tools even after being told
  to. Either `MEMEXP_PULL_MAX_ROUNDS` is too small or a tool result is not answering.

### The two ways these reports lied, and what now stops it

Both were found by reading job 591479's artifacts after the run, and both had the same signature:
**the artifact was confidently wrong, and every gate passed.**

1. **`memexp_pull_report.json` was an install-time snapshot.** Its only writer was `_bind()`, so
   every runtime counter in it — `steps`, `tool_calls`, `n_query`, frames served — was zero *by
   construction*. The census duly reported *"the Planner NEVER called a memory tool"*. The file
   even carried `"substrate": null, "tools": null`, which is only ever true before a context
   exists. It is now flushed per planning step, per tool call, and at interpreter exit.
2. **A single shared path, written by whoever ran last.** An arm runs several interpreters against
   one output directory: `task1` and `tasks2to26` are separate processes, the gates run before the
   evaluation, and `merge_eval_outputs.sh` runs a `python3` *after* it. Each truncated the
   previous one. Measured on 591479: the evaluator rendered the corrected memory thousands of
   times, and the surviving `memexp_memfix_report.json` read `n_render: 0` — because that
   end-of-arm `python3` loads the module through `PYTHONPATH` and installs it without rendering
   anything. Both reports are now written per PID and merged by the census.

`validate_arm.py` CHECK 14 asserts the repairs, and it fails if either is reverted: it checks the
call sites, the per-process path, that a counter written by a probe reaches disk, and that the
reports are absent from the arms that did not request them. The lesson worth keeping is that
*"is it installed?"* is not the question — **"does activity reach the artifact?"** is.

### The four ways the code was quietly measuring something other than the mechanism

Found by reading the archived artifacts of jobs 591479 and 592860 (`pullmem`) against the source,
then reproduced with CPU probes. All four passed every gate that existed at the time, and all four
are the same kind of defect: **a number that is confidently wrong rather than missing.**

| # | defect | how it was measured | what now stops it |
|---|---|---|---|
| P0-1 | The tool spec and the forced-finish clamp taught the model to end a step with `{"primitive": ...}`. `parse_vlm_output` reads `current_primitive`, and `json.loads` succeeds, so nothing raises, the prose fallback never fires, and the step yields **no primitive at all** — a compliant model produces no action. | probe on the real parser: `{"primitive": "open middle drawer"}` → `primitive=''` | `FINISH_FORM` / `FORCE_FINAL` name the planner's own two keys; `selftest_pull` G0b-7 re-reads the planner's source and asserts the taught form **parses to a non-empty primitive**, for every registry an arm can select |
| P0-2 | `by_tool` was read live from the per-episode object, so it described whichever episode was live at the last write. Job 592860: `tool_calls = 5` correct beside `by_tool = {}`. The last episode of a 26-task run makes no calls. | 3-episode probe in one process: `by_tool = {'search_memory': 5}` cumulative, while the live `tools.by_tool` is `{}` | `_absorb()` folds each retired context's **delta** into process-wide totals; the census reads `by_tool` / `*_cumulative` and labels older reports as undercounting |
| P0-3 | A misspelled `MEMEXP_TOOLS_MODULE` made `_build_messages`' blanket handler swallow the import error, so the arm **silently ran as the push-only control** while every flag read as enabled. The census then reported "the Planner never called a tool" — a claim about the model, drawn from a typo. | probe with a bad module name | `load_registry()` validates at install time *and* on first use, records `tools_module_error`, and the census returns **FAIL** rather than WARN |
| P1-1 | The reset hooks never fired. `_LOCAL` is thread-local and the hooks run on the evaluator's **main** thread while the planning context lives in the VLM worker, so the lookup found `None` and the body was skipped. Measured: `resets` is absent from every archived report while `reset_hooks` lists both installed. | `resets` missing from job 592860's reports | `_reset_ctx` retires contexts through a module-level registry, so the thread is irrelevant |
| **P1-4** | **The bank was never replaced across tasks.** The evaluator builds ONE planner and reuses it for all 26, so freshness keyed on planner *identity* never triggered. Task 26 would be answered partly out of tasks 1–25 — a leak that makes memory look **helpful**, in the direction that gets published, and leaves every per-task score meaningless. The 42 contexts per run that made identity *look* sufficient came from the VLM worker being a fresh thread per episode, which is an implementation detail of the evaluator's thread pool, not an episode boundary. | `selftest_pull` G0b-9, with the planner reused and the reset hooks called from the main thread. Reverting the fix reproduces the leak exactly: `task1=['contents(middle drawer)']` then `task2=['contents(middle drawer)', 'contents(bottom drawer)']` | `_get_ctx` rebuilds when the live context is **closed** as well as when the planner identity changes |

Two notes on what this table is *not* claiming. P1-4 was not in the earlier list of defects and was
found while fixing P1-1 — the reset hook that had never run was also the only mechanism that would
have replaced the bank. And none of these five changes the arms' *design*: they change whether the
design was actually running, and whether the report about it is about the run or about a single
episode inside it.

---

## 5. Known confounds

* **`subtask_override` is present in every arm.** `HARNESS_SUBTASK_OVERRIDE=1` lets the harness
  rewrite the VLA prompt after a stall, and this flows through `consume_subtask_override()` into
  the evaluator loop independently of `HARNESS_VLM_CONTEXT`. It is a *control* action, not
  evidence handed to the Planner, so it is not a memory channel — but it does mean the harness
  can act on its own memory state even in `nomem`. Deliberately left identical across arms.
* **`pullmem` changes two things at once relative to `pushmem`.** It stores a different thing (a
  step-addressed fact bank instead of RMA's evidence text plus keyframe bank) *and* it lets the
  Planner select. `pullmem` vs `pushmem` is therefore not a single-variable contrast. The clean
  comparisons are listed in §2; use `MEMEXP_PULL_TOOLS=0` for the selection question.
* **The substrate's write path sees only primitives.** Addresses are minted from the primitive
  text the Planner emits, so a Planner that never says `open <container>` never creates an
  address. This is deliberate — it keeps the write path model-free — but it means the arm is
  measuring retrieval and not extraction.
* **`query_world` falls back to token overlap when the exact address is absent.** So asking for
  `contents(middle drawer)` in an episode that only ever minted `contents(bottom drawer)` returns
  the bottom-drawer row as a **hit** (the shared word is "drawer"). This is intentional — the
  addresses the push advertises are machine-generated and `search_memory` is the intended route
  for ordinary phrasing — but it means a *call rate* is not by itself evidence that retrieval was
  *accurate*. Read it together with `n_query_miss` and `n_frames_served_off_context`; the latter
  being non-zero is the property that actually matters, because it counts pixels the Planner did
  not already have.
* **Archived `pullmem` numbers are not reproducible with this code, and should not be compared
  against it.** P0-1 changed what the arm teaches the model to output, P1-4 changed the bank's
  lifetime, and the arm now selects the searchable registry. Job 592860's `pullmem` ran with a
  self-contradicting prompt, a bank that accumulated across all 26 tasks, and a `by_tool` that
  described one episode. Its MACRO number is therefore a measurement of something other than
  agent retrieval and is kept only as the record of why these repairs were made.
* **Single seed.** `NUM_TRIALS=1, SEED=100` is the official single-seed protocol. With
  `ASYNC_VLM=1, VLM_QUEUE_SIZE=1` the benchmark carries large cross-run noise, so one run per arm
  is enough to *reject* a large effect and not enough to *establish* a small one.
  `HARNESS_SKIP_RETRY_TASKS='[18,22]'` is a hardcoded task subset inherited from HM; it is
  identical across arms and so does not affect the contrast, but it is not part of the official
  protocol and is a candidate for removal.
* **The correction is an intervention, so these arms are not the HM configuration.** Both memory
  arms now run corrected memory text, while the historical HM baseline (`70.2`) ran the defective
  text. `pushmem` vs the old HM number is therefore **not** a clean before/after of the same
  mechanism, and any comparison against it must say which text it used. The baseline for these
  arms is `nomem`, measured in the same job, on the same node, with the same protocol.
* **`pushmem` still bundles two channels.** Text evidence and historical keyframes are switched on
  together, so a `nomem → pushmem` difference cannot be attributed to either one alone. The
  available decomposition is `nomem` (neither) → `pullmem` (corrected text, no keyframes, plus
  tools) → `pushmem` (corrected text **and** keyframes), read alongside the census's per-channel
  evidence rather than as a factorial.
* **D3 is not independently verified.** It is reported as subsumed by D2 (the harm was the
  repetition, not the rules themselves) rather than as a separate defect that was fixed. Anyone
  expecting a three-way decomposition of the correction should treat it as two-and-a-half.
