#!/usr/bin/env bash
#SBATCH --job-name=rma-mem-eff
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/RoboMemArena/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/RoboMemArena/outputs/slurm/%x-%j.err

# =========================================================================================
# mem_efficacy: does a memory mechanism improve control?
#
# Runs the RMA OFFICIAL protocol (26 tasks, seed 100) for one or more arms of this
# experiment. The baseline arm has no memory channel; the treatment has the same harness,
# the same Planner and the same frozen VLA plus two memory channels.
#
# OFFICIAL ALIGNMENT
#   Evaluation path, knobs, checkpoints, GPU split and the task1 / tasks2-26 split are taken
#   from `slurm/benchmark/reproduce_all26_1seed.sbatch`, which the repository README names as
#   the official protocol. All knobs live in `arms/official_protocol.sh`; this script does not
#   restate them, so there is exactly one place where the protocol is written down.
#   `validate_arm.py` parses the upstream sbatch and fails if the two ever diverge.
#
# WHY THE PLANNER IS GEMINI-3.8-FLASH ON CloseAI
#   The official protocol ships a local PrediMem VLM as the Planner. This experiment replaces
#   the Planner with an external API model, which is a DELIBERATE deviation: the question is
#   whether memory helps CONTROL, holding the Planner fixed at a strong general VLM.
#   Three measured wiring facts (2026-09-14) constrain the endpoint:
#     1. api.openai-proxy.org TLS-resets from this cluster's egress; api.closeai-asia.com works.
#     2. CloseAI does not serve qwen-vl-max (HTTP 400), so no literal fallback is permitted
#        here: the model is pinned and re-asserted after sourcing the arm.
#     3. gemini-3.8-flash is a reasoning model: the chain of thought is spent INSIDE
#        max_tokens before the JSON, so small budgets truncate mid-token and the caller gets
#        an unparsable string. `api_vlm_planner.py:362` defaults PLANNER_API_MAX_TOKENS to
#        only 256, which truncates this model every time, so the budget is set explicitly.
#        HARNESS_API_MAX_TOKENS is what the stall-recovery rung reads (its own default is
#        512, in api_planner.py:180); PLANNER_API_MAX_TOKENS is what the main planner reads.
#        The 4096 below is the value used by the last known-good gemini run (587161), and
#        2048 was NOT enough -- that is a measured value, not a preference.
#
# SUBMIT
#   cd /project/peilab/why/RoboMemArena
#   sbatch experiments/mem_efficacy/run_26x1.sbatch                     # baseline only, 26x1
#   ARM_OVERRIDE="nomem pushmem" sbatch experiments/mem_efficacy/run_26x1.sbatch
#   ARM_OVERRIDE="nomem pushmem pullmem" sbatch experiments/mem_efficacy/run_26x1.sbatch
#   ARM_OVERRIDE=pushmem sbatch experiments/mem_efficacy/run_26x1.sbatch # treatment only
#
#   Subset / multi-seed (a DIFFERENT experiment; see the scope block below for why the two
#   cannot be reported as the same number):
#   ARM_OVERRIDE="nomem pushmem pullmem" \
#     TASKS_JSON_SCOPE='[1,4,6,8,12,22,25,26]' \
#     SEED_LIST="100 101 102" \
#     OUT_BASE_OVERRIDE="${PWD}/experiments/mem_efficacy/results/mem_8x3" \
#     sbatch experiments/mem_efficacy/run_26x1.sbatch
#
# ARMS (each is an independent file under arms/; see README.md)
#   nomem    baseline  -- official protocol + harness, every memory channel off
#   pushmem  treatment -- nomem plus the memory that RMA pushes into the prompt unprompted,
#                         including a keyframe bank sourced from the harness's own stage
#                         boundaries and subtask changes (MEM_STAGE_ANCHOR=1)
#   pullmem  treatment -- nomem plus a SMALL push and memory tools the Planner may call itself.
#                         Installed by `pysite/sitecustomize.py` via PYTHONPATH; it edits no
#                         shared file. Order-independent: `nomem.sh` strips both the MEMEXP_*
#                         flags and this experiment's PYTHONPATH entry, so whichever arm runs
#                         first, each arm sees only its own configuration (validate_arm.py
#                         CHECK 9 asserts this over every ordered pair).
# =========================================================================================

set -euo pipefail

ROOT="/project/peilab/why/RoboMemArena"
export ROOT
cd "${ROOT}"

EXP_DIR="${ROOT}/experiments/mem_efficacy"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_BASE="${OUT_BASE_OVERRIDE:-${EXP_DIR}/results/mem_efficacy_${JOB_TAG}}"
ARMS_TO_RUN="${ARM_OVERRIDE:-nomem}"

# --- scope: which tasks, which seeds --------------------------------------------------------
# DEFAULTS REPRODUCE THE PREVIOUS BEHAVIOUR EXACTLY: SEED_LIST=100 and an empty
# TASKS_JSON_SCOPE, i.e. "whatever arms/official_protocol.sh declares", which is all 26 tasks.
# Both overrides exist so that a SUBSET / MULTI-SEED run does not require forking this file.
# Forking is what this file was written to prevent: it would put the gate chain, the env
# resolution and the census in two places, and this project has already lost runs to a second
# copy of a protocol drifting from the first.
#
#   SEED_LIST         space-separated seeds. One output dir per seed (`<arm>_s<seed>/`), which
#                     is exactly the layout census_channels.py already scans for.
#   TASKS_JSON_SCOPE  JSON task list. Empty keeps the arm's official 1..26 scope.
#
# A subset run is a DIFFERENT EXPERIMENT than the official 26x1 one, and the reporting has to
# say so, because stage_score_pct is a macro mean over the tasks that were RUN. Selecting tasks
# by their nomem score and then reporting a macro mean over them is a post-hoc selection: the
# number is a property of this task set and is NOT comparable to a 26-task macro mean. Every
# run of this script prints the scope in the banner and freezes it into resolved_env.txt, so
# the task set travels with the score.
#
# Applied AFTER the arm is sourced (see run_arm): official_protocol.sh sets TASKS_JSON and
# SEED=100 itself, so a pre-source override would be silently clobbered and the banner would
# agree with the intent while the evaluator disagreed. validate_arm.py CHECK 4 still validates
# the ARM's declared scope (26 tasks / seed 100); it runs before any arm is sourced, so it tests
# the arm definition, and this narrowing is recorded separately. That split is deliberate: the
# arm keeps its official definition, the run narrows it on the record.
SEED_LIST="${SEED_LIST:-100}"
TASKS_JSON_SCOPE="${TASKS_JSON_SCOPE:-}"

# Trials PER (task, seed) cell. Empty keeps the arm's official NUM_TRIALS=1.
#
# This is the lever that buys statistical power, and it is cheaper than a long SEED_LIST by a large
# factor. `eval_common.run_task` computes `current_seed = seed + ep`, so NUM_TRIALS=N covers seeds
# SEED .. SEED+N-1 while running ONE evaluation pass -- one VLA server start, one task loop. A
# SEED_LIST of N entries restarts the server and the whole sweep N times for the same coverage.
# The upstream convention agrees: the official external-eval wrapper
# (`evaluation_benchmark/scripts/run_all_tasks1_26.py`) defaults to `--num-trials-per-task 51`.
#
# It is needed at all because this benchmark's noise is large: measured over two independent runs
# of the SAME arm at the SAME seed, the per-task score difference has SD ~= 32 pp (36.8 pp on
# non-saturated tasks). A three-seed macro mean has a 95% CI half-width of ~26 pp, which cannot
# resolve the effect size this experiment is looking for. More trials per cell is what narrows it.
NUM_TRIALS_SCOPE="${NUM_TRIALS_SCOPE:-}"

# Guard against an inherited ARM_OVERRIDE. `sbatch` forwards the exporting shell's
# environment, so a leftover `export ARM_OVERRIDE=...` from an earlier interactive session
# silently becomes this job's arm list. Run 590445 hit exactly this: the banner reported
# `api_seam_m9`, and had GATE 0 passed, `source arms/api_seam_m9.sh` would have aborted the
# job on a missing file. Fail loudly here instead, and say which value was inherited.
echo "[INFO] ARM_OVERRIDE from the submitting environment: ${ARM_OVERRIDE:-<unset>}"
# Same hazard for the scope: `sbatch` forwards the exporting shell's environment, so a leftover
# SEED_LIST / TASKS_JSON_SCOPE from an interactive session becomes this job's scope. That cannot
# abort the job the way a bad arm name can -- any seed is a legal seed -- which is exactly why
# it has to be PRINTED rather than detected: a silent scope change produces a score that looks
# fine and answers a different question.
echo "[INFO] scope from the submitting environment: SEED_LIST=${SEED_LIST}  TASKS_JSON_SCOPE=${TASKS_JSON_SCOPE:-<arm default>}  NUM_TRIALS_SCOPE=${NUM_TRIALS_SCOPE:-<arm default>}"
for _arm in ${ARMS_TO_RUN}; do
  if [[ ! -f "${EXP_DIR}/arms/${_arm}.sh" ]]; then
    echo "[ERROR] no such arm: '${_arm}' -> ${EXP_DIR}/arms/${_arm}.sh" >&2
    echo "        available: $(cd "${EXP_DIR}/arms" && ls *.sh 2>/dev/null | sed 's/\.sh$//' | tr '\n' ' ')" >&2
    echo "        If this came from the environment, run: unset ARM_OVERRIDE" >&2
    exit 2
  fi
done
unset _arm

# ---------------------------------------------------------------------------------------
# Planner triple. Pinned, with no literal fallback to a model this relay does not serve.
# ---------------------------------------------------------------------------------------
export PLANNER_API_KEY_FILE="${PLANNER_API_KEY_FILE:-${ROOT}/closeai_api.txt}"
export PLANNER_API_BASE_URL="${PLANNER_API_BASE_URL:-https://api.closeai-asia.com/v1}"
export PLANNER_API_MODEL="${PLANNER_API_MODEL:-gemini-3.8-flash}"
export PLANNER_API_MAX_TOKENS="${PLANNER_API_MAX_TOKENS:-4096}"
export PLANNER_API_TIMEOUT="${PLANNER_API_TIMEOUT:-180}"

# ---------------------------------------------------------------------------------------
# RSI replay bundles. DEFAULT OFF, AND THAT DEFAULT IS LOAD-BEARING.
#
# `api_vlm_planner.py` calls `dump_decision` on every planner step; the guard inside it is
# `RSI_DUMP_VLM_INPUTS`. With the flag unset the function returns before touching the
# filesystem, so every existing arm issues byte-identical requests to before and no log line,
# no prompt, and no planner state changes. The `nomem` baseline is the control that every
# memory delta is measured against, so a default-on dump would invalidate the whole comparison
# the moment it perturbed a single decision.
#
# Turning it on is what makes Phase A2 (`experiments/rsi/replay_and_check.py`) possible: the
# bundle is a transcript of the exact wire payload, so the loop can screen a candidate memory
# by re-issuing logged decisions instead of rolling out episodes.
#
#   RSI_DUMP_VLM_INPUTS=1                 record `<run_dir>/vlm_inputs/t*/replay.json` + JPEGs
#   RSI_DUMP_MAX_DECISIONS=<n>            cap per episode, 0 = unlimited
#   RSI_DUMP_MAX_MB=<n>                   cap per run, 0 means the writer's own default of 2048
#
# Cost is a few hundred KB per decision (~10 frames at q85), i.e. ~10-30 MB per episode.
# ---------------------------------------------------------------------------------------
export RSI_DUMP_VLM_INPUTS="${RSI_DUMP_VLM_INPUTS:-0}"
export RSI_DUMP_MAX_DECISIONS="${RSI_DUMP_MAX_DECISIONS:-0}"
export RSI_DUMP_MAX_MB="${RSI_DUMP_MAX_MB:-2048}"

# The stall-recovery rung reads HARNESS_API_* (NOT PLANNER_*): harness/config.py is explicit.
# Leaving these unset would point the recovery rung at DashScope, producing a 401-ing rung and
# a report that reads "the harness never fired" when in fact the config was wrong.
export HARNESS_API_KEY_FILE="${HARNESS_API_KEY_FILE:-${PLANNER_API_KEY_FILE}}"
export HARNESS_API_KEY_LINE="${HARNESS_API_KEY_LINE:-${PLANNER_API_KEY_LINE:-1}}"
export HARNESS_API_BASE_URL="${HARNESS_API_BASE_URL:-${PLANNER_API_BASE_URL}}"
export HARNESS_API_MODEL="${HARNESS_API_MODEL:-${PLANNER_API_MODEL}}"
export HARNESS_API_MAX_TOKENS="${HARNESS_API_MAX_TOKENS:-512}"
export HARNESS_API_TIMEOUT="${HARNESS_API_TIMEOUT:-90}"

# Recorded so the banner and the report cannot disagree with what actually ran.
export PLANNER_API_MODEL_REQUESTED="${PLANNER_API_MODEL}"
export HARNESS_API_MODEL_REQUESTED="${HARNESS_API_MODEL}"

mkdir -p "${ROOT}/outputs/slurm" "${OUT_BASE}"
module load slurm "nvhpc-hpcx-cuda12/23.11" 2>/dev/null || true

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
for _glu_dir in /usr/lib/x86_64-linux-gnu /usr/lib64 /cm/shared/apps/mesa/*/lib; do
  if [[ -f "${_glu_dir}/libGLU.so.1" || -f "${_glu_dir}/libGLU.so" ]]; then
    export LD_LIBRARY_PATH="${_glu_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    break
  fi
done
unset _glu_dir

# shellcheck disable=SC1091
source "${ROOT}/scripts/env_common.sh"
robomem_setup_modules
robomem_setup_cache

export OPENPI_ROOT="${ROOT}/third_party/openpi_minimal"
export OPENPI_INFERENCE_ROOT="${ROOT}"
export LIBERO_FORK_ROOT="${ROOT}/evaluation_benchmark/libero_fork"
export TARGET_LIBERO_PATH="${LIBERO_FORK_ROOT}/libero"
export PREDIMEM_HF_SNAPSHOT="${ROOT}/checkpoints/PrediMem"
export SERVER_PY="${OPENPI_ROOT}/.venv/bin/python"
export EVAL_PY="${ROOT}/.venv/bin/python"

if command -v gcc >/dev/null 2>&1; then export CC="$(command -v gcc)"; fi
if command -v g++ >/dev/null 2>&1; then export CXX="$(command -v g++)"; fi

NGPU=$("${EVAL_PY}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${NGPU}" -lt 2 ]]; then
  echo "[ERROR] need >=2 GPUs on one node (frozen VLA server + evaluator); got ${NGPU}" >&2
  exit 2
fi

echo "=============================================================================="
echo "mem_efficacy  --  RMA official protocol, scope as declared below"
echo "  job            = ${JOB_TAG}"
echo "  arms           = ${ARMS_TO_RUN}"
echo "  seeds          = ${SEED_LIST}"
if [[ -n "${TASKS_JSON_SCOPE}" ]]; then
  echo "  tasks          = ${TASKS_JSON_SCOPE}   <-- SUBSET: scores are NOT comparable to a 26-task macro mean"
else
  echo "  tasks          = <from arms/official_protocol.sh: 1..26>"
fi
if [[ -n "${NUM_TRIALS_SCOPE}" ]]; then
  echo "  trials/cell    = ${NUM_TRIALS_SCOPE}   <-- covers seeds ${SEED_LIST%% *}..$(( ${SEED_LIST%% *} + NUM_TRIALS_SCOPE - 1 ))"
else
  echo "  trials/cell    = <from arms/official_protocol.sh: 1>"
fi
echo "  out base       = ${OUT_BASE}"
echo "  planner model  = ${PLANNER_API_MODEL}  (pinned; official uses a local VLM)"
echo "  planner base   = ${PLANNER_API_BASE_URL}"
echo "  planner tokens = ${PLANNER_API_MAX_TOKENS}  timeout=${PLANNER_API_TIMEOUT}s"
echo "  recovery model = ${HARNESS_API_MODEL}  tokens=${HARNESS_API_MAX_TOKENS}  timeout=${HARNESS_API_TIMEOUT}s"
if [[ "${RSI_DUMP_VLM_INPUTS}" != "0" && "${RSI_DUMP_VLM_INPUTS}" != "false" ]]; then
  echo "  RSI bundles    = ON  (max_decisions=${RSI_DUMP_MAX_DECISIONS} max_mb=${RSI_DUMP_MAX_MB})"
else
  echo "  RSI bundles    = off"
fi
echo "  gpu split      = per-arm (from arms/official_protocol.sh; printed again per arm below)"
echo "=============================================================================="

# ---------------------------------------------------------------------------------------
# GATE 0 -- the Planner must be alive AND the recovery rung must PARSE, before a GPU is used.
#
# A 200 response is not enough. gemini-3.8-flash spends its chain of thought inside
# max_tokens, so a budget that is too small yields `finish_reason=length` and a string that
# json.loads rejects; the caller then receives None and the recovery rung becomes a silent
# no-op while the arm still reports HARNESS_SUBTASK_OVERRIDE=1. The cheap failure to detect
# here is that silent one, so the gate exercises the REAL code path at the REAL budget.
# ---------------------------------------------------------------------------------------
echo
echo "[GATE 0] planner liveness + recovery-rung parseability"

# Does any arm IN SCOPE teach the nomination policy? GATE 0 runs BEFORE the arms are sourced, so
# the knob is not in the environment yet, and reading it directly would silently skip the probe --
# a gate that reports SKIPPED while looking like it ran. Derived from the arm FILES instead.
#
# Passed to the probe as its own variable rather than by exporting `MEM_KF_NOMINATION_PROMPT`: a
# memory knob exported here would be inherited by every arm, and the baseline is the one arm that
# must not have it (CHECK 1 asserts the baseline carries no memory key, and an extra prompt block
# in the control would contaminate every delta measured against it).
_GATE0_NOM=0
for _ga in ${ARMS_TO_RUN}; do
  if [[ -f "${EXP_DIR}/arms/${_ga}.sh" ]] && \
     grep -qE '^[[:space:]]*export[[:space:]]+MEM_KF_NOMINATION_PROMPT=(1|true|yes|on)' \
       "${EXP_DIR}/arms/${_ga}.sh"; then
    _GATE0_NOM=1
  fi
done
export GATE0_NOM_PROBE="${GATE0_NOM_PROBE:-${_GATE0_NOM}}"
echo "  nomination-policy probe: ${GATE0_NOM_PROBE} (derived from arms: ${ARMS_TO_RUN})"
unset _ga

"${EVAL_PY}" - <<'PY'
import base64, json, os, sys, time
from io import BytesIO

ROOT = os.environ["ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation_benchmark"))
sys.path.insert(0, os.path.join(ROOT, "evaluation_benchmark", "openpi_minimal_runtime"))

from harness.api_planner import chat_completions, load_api_key  # noqa: E402

base = os.environ["PLANNER_API_BASE_URL"].rstrip("/")
model = os.environ["PLANNER_API_MODEL"]
key = load_api_key()
if not key:
    print("  key: <missing>")
    raise SystemExit(4)
print(f"  key: ...{key[-8:]}")
print(f"  gate0 uses harness.api_planner.chat_completions (model-aware token/temperature keys)")

fail = []
# Reachability via the SAME client the evaluator uses. Raw urllib with hard-coded
# `max_tokens` rejects gpt-6-luna / gpt-5* (HTTP 400) and used to burn 4x10s retries
# on a permanent parameter error — that is wasted money and a false "endpoint down".
out = None
for attempt in range(4):
    out = chat_completions(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        api_key=key,
        base_url=base,
        model=model,
        timeout_sec=60.0,
        max_tokens=64,
    )
    if out and str(out).strip():
        break
    print(f"     probe attempt {attempt + 1}/4 -> empty; backing off 5s")
    time.sleep(5)
print(f"  1) reachability  -> {out!r}")
if not out or not str(out).strip():
    fail.append("planner endpoint not usable after 4 chat_completions attempts (empty reply)")

try:
    from harness.api_planner import suggest_subtask_via_api
    got = suggest_subtask_via_api(
        task_block="put the butter in the empty drawer",
        stage_name="open top drawer",
        current_subtask="open top drawer",
        primitive_labels=["open top drawer", "check contents", "close drawer", "place butter"],
        memory_context="the middle drawer is empty",
        api_key=key,
        base_url=os.environ["HARNESS_API_BASE_URL"],
        model=os.environ["HARNESS_API_MODEL"],
    )
    print(f"  2) recovery rung -> {got!r}")
    if not got:
        fail.append("suggest_subtask_via_api returned None (truncation/timeout): the recovery "
                    "rung would be a SILENT no-op while the arm still reports override=1")
except Exception as exc:
    fail.append(f"recovery rung raised: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------------------
# 3) The PrediMem nomination channel: does a REAL API model act on the nomination policy?
#
# This is the only place the question can be answered before spending GPU time. Everything else
# about channel B can be proven offline (the bank builders are pure functions, `validate_arm.py`
# CHECK 7 covers them), but "the model emits keyframe_positions when taught the policy" depends on
# the model, and the entire PrediMem comparison rests on it: without nominations the arm's bank is
# the anchor+stride fallback, not a nomination bank.
#
# The observation window is SYNTHETIC and deliberately unambiguous -- two consecutive frames where
# the second plainly shows a container that has just opened with contents visible -- because the
# purpose here is to test the CHANNEL (is the policy understood and the field used at all?), not
# the hit rate on real scenes. The real-scene rate is measured after the run by
# `census_channels.py`, which FAILs the arm when the policy is declared ON and the nomination count
# is zero across the whole run. A model that will not nominate even here will certainly not
# nominate there, and the arm would be mislabelled.
#
# The policy text is IMPORTED, not copied, so this gate cannot pass while testing a different
# prompt from the one the arms send.
if str(os.environ.get("GATE0_NOM_PROBE", "0")).strip().lower() in {"1","true","yes","on","y","t"}:
    try:
        import base64, io
        from PIL import Image, ImageDraw

        def _scene(opened):
            img = Image.new("RGB", (256, 256), (30, 30, 30))
            d = ImageDraw.Draw(img)
            # A dark cabinet face; when `opened`, the drawer is pulled out and lit, with items.
            d.rectangle([40, 80, 216, 200], fill=(90, 70, 50), outline=(20, 20, 20), width=3)
            if opened:
                d.rectangle([40, 120, 216, 200], fill=(225, 210, 180))
                for cx in (80, 128, 176):
                    d.ellipse([cx-14, 150, cx+14, 178], fill=(200, 60, 60))
                d.text((46, 86), "OPEN DRAWER", fill=(255, 255, 255))
            else:
                d.text((46, 86), "CLOSED DRAWER", fill=(230, 230, 230))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        from harness.api_vlm_planner import KF_NOMINATION_POLICY

        msgs = [
            {"role": "system", "content": "You are a robot task planner."},
            {"role": "user", "content": [
                {"type": "text", "text": "Task: put the butter in the drawer that still holds nothing."},
                {"type": "text", "text": "Recent visual context: 2 consecutive frames ending at the current frame:"},
                {"type": "image_url", "image_url": {"url": _scene(False)}},
                {"type": "image_url", "image_url": {"url": _scene(True)}},
                {"type": "text", "text":
                    "Output strict JSON with exactly two fields: current_primitive and keyframe_positions. "
                    "keyframe_positions are 1-indexed keyframe positions inside the recent visual window."},
                {"type": "text", "text": KF_NOMINATION_POLICY},
            ]},
        ]
        budget = int(os.environ.get("PLANNER_API_MAX_TOKENS", "4096") or 4096)
        t0 = time.time()
        txt = chat_completions(
            messages=msgs,
            api_key=key,
            base_url=base,
            model=model,
            timeout_sec=180.0,
            max_tokens=budget,
        ) or ""
        dt = time.time() - t0
        print(f"  3) nomination channel via chat_completions {dt:.2f}s")
        print(f"     raw -> {txt[:300]!r}")
        if not str(txt).strip():
            fail.append("nomination probe not usable: empty reply from chat_completions")
        else:
            try:
                from harness.vlm_output_parser import parse_vlm_output
                prim, jrel = parse_vlm_output(txt, max_pos=2)
                print(f"     parsed -> primitive={prim!r} keyframe_positions={jrel}")
                if not prim:
                    fail.append("nomination probe produced NO current_primitive: the two-field "
                                "contract is broken by the extra policy block (a silent no-action "
                                "step is the failure mode this project has already paid for)")
                if not jrel:
                    fail.append(
                        "MEM_KF_NOMINATION_PROMPT is ON but the model nominated NOTHING even on a "
                        "deliberately unambiguous open-drawer transition. The nomination channel "
                        "would be inert for the whole run, so the arm's bank would be the "
                        "anchor+stride fallback while the arm reports itself as PrediMem-style. "
                        "Re-word the policy or drop the claim before spending GPU time.")
                else:
                    print(f"     OK: the policy is understood and the field is used ({jrel})")
            except Exception as exc:
                fail.append(f"nomination probe unparsable: {type(exc).__name__}: {exc}; raw={txt[:200]!r}")
    except Exception as exc:
        fail.append(f"nomination probe raised: {type(exc).__name__}: {exc}")
else:
    print("  3) nomination channel SKIPPED (no arm in scope declares MEM_KF_NOMINATION_PROMPT)")

# ---------------------------------------------------------------------------------------
# 4) RSI replay bundles. If this run is meant to produce them, prove the WRITER and the
#    REBUILD agree before the GPUs, because a broken bundle set is otherwise only discovered by
#    the fidelity sweep -- which costs API calls for the verbatim replay and GPU time for the
#    substituted one. The check is pure CPU: dump a synthetic decision, require the rebuilt
#    request to hash to the captured one, and require a memory substitution to land in the same
#    two slots (`_build_messages`) a memory-enabled run would have used.
#
#    Running the run's own script rather than a copy is deliberate: the gate and the sweep must
#    fail together, or the gate is testing something else.
if str(os.environ.get("RSI_DUMP_VLM_INPUTS", "0")).strip().lower() not in {"", "0", "false", "no", "off"}:
    import subprocess
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "experiments", "rsi", "replay_and_check.py"), "--selftest"],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(f"  4) RSI replay bundle self-test -> exit {proc.returncode}")
    for _line in (proc.stdout or "").strip().splitlines():
        print(f"     {_line}")
    if proc.returncode != 0:
        for _line in (proc.stderr or "").strip().splitlines()[-6:]:
            print(f"     ! {_line}")
        fail.append(
            "RSI_DUMP_VLM_INPUTS is ON but the replay-bundle self-test FAILED, so this run would "
            "emit bundles that cannot be replayed and the whole fidelity sweep would be void. "
            "Fix the writer/rebuild pair before spending GPU time.")
else:
    print("  4) RSI replay bundles OFF (RSI_DUMP_VLM_INPUTS unset)")

# -------------------------------------------------------------------------------------------
# 5) Dead-planner watchdog. THIS GATE IS THE DIRECT RESPONSE TO JOBS 598082/598083/598084.
#
# Those three jobs ran against a planner that stopped answering at 12:02:13 and never came back.
# `chat_completions` returns None, `infer_sync` fell through to the current subtask, and the run
# kept writing episodes for another ~100 minutes: 46 of 50 episodes per arm were scored from a
# silent planner. The scores were not merely noisy, they were HIGHER than the healthy baseline,
# so nothing about the result table looked wrong.
#
# GATE 0 step 1 (reachability) cannot catch this: it is a single call at 11:45 and it PASSED on
# all three of those jobs. Liveness at t=0 says nothing about t=+75min. So the guard lives in the
# planner and this step proves the guard itself works, before any GPU time is spent relying on it.
import subprocess
proc = subprocess.run(
    [sys.executable, os.path.join(ROOT, "experiments", "mem_efficacy",
                                  "planner_watchdog_selftest.py")],
    capture_output=True, text=True, cwd=ROOT,
)
print(f"  5) dead-planner watchdog self-test -> exit {proc.returncode}")
for _line in (proc.stdout or "").strip().splitlines():
    print(f"     {_line}")
if proc.returncode != 0:
    for _line in (proc.stderr or "").strip().splitlines()[-6:]:
        print(f"     ! {_line}")
    fail.append(
        "the dead-planner watchdog self-test FAILED. Without it a silent endpoint produces a full "
        "grid of plausible numbers that measure nothing -- which is exactly what jobs "
        "598082/598083/598084 wasted. Fix the guard before starting a run.")
elif not os.environ.get("PLANNER_MAX_CONSECUTIVE_FAILURES"):
    # Printed rather than assumed: the guard's threshold is the difference between "abort in 20
    # seconds" and "abort after the episode", and a reader of the log should not have to guess it.
    print("     (PLANNER_MAX_CONSECUTIVE_FAILURES unset -> planner default of 8 takes effect)")

if fail:
    print("\n[GATE 0] FAILED:")
    for f in fail:
        print(f"  - {f}")
    raise SystemExit(4)
print("[GATE 0] PASSED: planner reachable and the recovery rung parses")
PY

# ---------------------------------------------------------------------------------------
# GATE 1 -- arm integrity. Proves official alignment and channel liveness before the GPUs.
# ---------------------------------------------------------------------------------------
echo
echo "[GATE 1] arm integrity (official alignment + memory channels)"
ROOT="${ROOT}" "${EVAL_PY}" "${EXP_DIR}/validate_arm.py"

# ---------------------------------------------------------------------------------------
# GATE 0b -- the PULL arm's loop, driven by a scripted model, with no API quota spent.
#
#   Separate from GATE 1 on purpose. GATE 1 proves the binding is INSTALLED and the substrate
#   behaves; it cannot prove the loop runs, because the loop only runs when a model answers with
#   a tool call. The distance between those two claims is exactly where this project has lost
#   runs -- a memory feature enabled, its own counter at zero, and a score that looks like
#   "memory does not help". So the loop is executed here against a model scripted to emit the
#   outputs it must handle.
# ---------------------------------------------------------------------------------------
# Membership test on a SPACE-separated list. The padded comparison is the whole point: the
# first version of this line used `",${ARMS_TO_RUN}," == *",pullmem,"*`, which is the
# comma-separated idiom. `ARMS_TO_RUN` is space-separated, so that pattern matched only a list
# that ended with a literal comma before the name -- i.e. never. The gate reported
# `SKIPPED (pullmem not in ARMS_TO_RUN=nomem pushmem pullmem)`, a message that contradicts
# itself, and job 590795 ran without the tool loop ever being exercised in the job. Padding
# both ends with spaces makes the match exact for any position, including first and last.
if [[ " ${ARMS_TO_RUN} " == *" pullmem "* ]]; then
  echo
  echo "[GATE 0b] PULL arm loop (scripted model, no API calls)"
  if ! ROOT="${ROOT}" "${EVAL_PY}" "${EXP_DIR}/selftest_pull.py"; then
    echo "[GATE 0b] FAILED: the PULL arm's tool loop is not working; refusing to spend GPU time" >&2
    exit 5
  fi
  echo "[GATE 0b] PASSED"
else
  echo
  echo "[GATE 0b] SKIPPED (pullmem not in ARMS_TO_RUN='${ARMS_TO_RUN}')"
fi

# ---------------------------------------------------------------------------------------
# GATE 0c -- the memory-content correction, executed rather than assumed.
#
#   GATE 1 (validate_arm.py CHECK 12) proves the correction is INSTALLED in the memory arms. That
#   is a weaker claim than it looks: a module can install successfully and still return the wrong
#   text, or patch only one of the two Planner call sites, or omit the defect it was written to
#   remove. This gate CALLS the corrected functions on an episode archived from job 590799 and
#   asserts that each defect reproduces on the original code and is absent from the corrected
#   path. CPU only -- no API calls, no GPUs.
#
#   It sets MEMEXP_MEMFIX_ENABLE itself rather than relying on an arm file, because the gates run
#   before any arm is sourced. The wiring question ("does the arm switch it on, and does it reach
#   the evaluator") is CHECK 12's job; this gate's job is the module's behaviour.
# ---------------------------------------------------------------------------------------
if [[ " ${ARMS_TO_RUN} " == *" pushmem "* || " ${ARMS_TO_RUN} " == *" pullmem "* ]]; then
  echo
  echo "[GATE 0c] memory-content correction (reproduces the defects, then shows them fixed)"
  if ! ROOT="${ROOT}" MEMEXP_MEMFIX_ENABLE=1 "${EVAL_PY}" "${EXP_DIR}/selftest_memfix.py"; then
    echo "[GATE 0c] FAILED: the memory correction is not working; refusing to spend GPU time" >&2
    exit 6
  fi
  echo "[GATE 0c] PASSED"
else
  echo
  echo "[GATE 0c] SKIPPED (neither pushmem nor pullmem in ARMS_TO_RUN='${ARMS_TO_RUN}')"
fi

# ---------------------------------------------------------------------------------------
# GATE 0d -- the cross-task experience memory (RSI) FIRES, and is INERT in the baseline.
#
#   Same separation of concerns as GATE 0b/0c: GATE 1 (`validate_arm.py` CHECK 5) proves the
#   baseline's composed Planner context is EMPTY and that every arm agrees on channel A. That is
#   the "is it silent" half. It cannot prove the "does it fire" half, and for a TREATMENT the
#   second half is the one that decides whether the arm means anything: job 586700 ran a memory
#   arm whose own counter was all-empty and still reported a score, which reads as "memory does
#   not help" when the truth is "memory never ran".
#
#   So this gate drives a REAL `HarnessController` through `on_stage_progress` and reads the
#   context back through `compose_planner_context` -- the same function the Planner prompt is
#   built from -- for all three arms at once, and asserts four things: the baseline stays empty
#   at every probe; the treatment fires on a stalled opening stage; it stays SILENT below the step
#   threshold and off the opening stage; and the block does not contain the stage-disclosure
#   phrasing that the channel-A regression (-13.5 pp, job 593253) traces to.
#
#   It reads the arms from their own files rather than from the environment, for the same reason
#   GATE 0's nomination probe does: this runs before any arm is sourced, so reading a knob from
#   the environment would report SKIPPED while looking like it had run.
if [[ " ${ARMS_TO_RUN} " == *" rsimem "* || " ${ARMS_TO_RUN} " == *" shamltm "* ]]; then
  echo
  echo "[GATE 0d] cross-task experience memory (fires where intended, silent where it must be)"
  if ! ROOT="${ROOT}" "${EVAL_PY}" "${EXP_DIR}/volo_ltm_gate0.py"; then
    echo "[GATE 0d] FAILED: the RSI treatment is not working; refusing to spend GPU time" >&2
    exit 7
  fi
  echo "[GATE 0d] PASSED"
else
  echo
  echo "[GATE 0d] SKIPPED (neither rsimem nor shamltm in ARMS_TO_RUN='${ARMS_TO_RUN}')"
fi

# ---------------------------------------------------------------------------------------
# Run one arm
# ---------------------------------------------------------------------------------------
run_arm() {
  local arm="$1" seed="$2"
  local out_root="${OUT_BASE}/${arm}_s${seed}"
  mkdir -p "${out_root}/logs"

  # -------------------------------------------------------------------------------------
  # COLLISION LOCK. Jobs 596262 and 596284 once wrote the SAME out_root concurrently; the
  # resulting sync_vlm.log held two Episode records per ep directory with timestamps that
  # ran backwards, and the run was permanently unpublishable. Refuse to start if another
  # job already claimed this directory. The lock is a file, not a flock, so it also
  # survives across a `--requeue` on a different node.
  # -------------------------------------------------------------------------------------
  local lock="${out_root}/.run.lock"
  if [[ -f "${lock}" ]]; then
    local prev
    prev="$(cat "${lock}" 2>/dev/null || echo unknown)"
    if [[ "${prev}" != "${SLURM_JOB_ID:-manual}"* ]]; then
      echo "[ERROR] OUT_ROOT already claimed by another job: ${out_root}" >&2
      echo "        lock=${prev}" >&2
      echo "        refusing to overwrite; pick a fresh OUT_BASE_OVERRIDE" >&2
      return 9
    fi
  fi
  echo "${SLURM_JOB_ID:-manual} $(date -Iseconds) arm=${arm} seed=${seed}" > "${lock}"

  # Clear the arm's in-process counters BEFORE anything can write them. This job is submitted with
  # `--requeue`, and a requeued attempt re-runs the same arm into the same directory without
  # clearing it. The counters are now written per process and SUMMED by the census (they used to be
  # a single file that each process truncated, which made this impossible), so leftover files from
  # a previous attempt would be added to this one -- and the tool-call count is the number the
  # whole PULL arm is judged by, so an inflated value would be worse than a missing one.
  rm -f "${out_root}"/memexp_pull_report.json* "${out_root}"/memexp_memfix_report.json*

  # OUT_ROOT must be exported BEFORE the arm is sourced: an arm may derive an artifact path from
  # it (pullmem writes its binding report there), and a path resolved from an unset OUT_ROOT
  # would send that artifact to /tmp where the census would never look for it.
  export OUT_ROOT="${out_root}"

  # Source the arm. `set -a` is deliberately NOT used: the arm file exports explicitly, and
  # auto-exporting would drag local shell state into the evaluator's environment.
  # shellcheck disable=SC1091
  source "${EXP_DIR}/arms/${arm}.sh"

  # Optional Stage-1 profile overlay (Hlegacy / H0). Applied AFTER the arm so the profile can
  # turn recovery OFF without the arm re-asserting it. `profile_preflight.py` asserts the
  # resulting env matches the profile contract before any GPU is spent.
  if [[ -n "${MEMPROF:-}" ]]; then
    if [[ ! -f "${EXP_DIR}/profiles/${MEMPROF}.sh" ]]; then
      echo "[ERROR] unknown MEMPROF='${MEMPROF}' (expected hlegacy|h0)" >&2
      return 2
    fi
    # shellcheck disable=SC1091
    source "${EXP_DIR}/profiles/${MEMPROF}.sh"
    echo "[INFO] MEMPROF=${MEMPROF} applied (HARNESS_MAX_RETRIES=${HARNESS_MAX_RETRIES} SUBTASK_OVERRIDE=${HARNESS_SUBTASK_OVERRIDE} STAGE_CHECKPOINT=${HARNESS_STAGE_CHECKPOINT} MEM_STAGE_ANCHOR=${MEM_STAGE_ANCHOR})"
  fi

  # Re-assert the pinned model AFTER sourcing. The official runner resolves
  # ${PLANNER_API_MODEL:-qwen-vl-max}, so without this a stale default could substitute a
  # model this relay does not carry.
  export PLANNER_API_MODEL="${PLANNER_API_MODEL_REQUESTED}"
  export HARNESS_API_MODEL="${HARNESS_API_MODEL_REQUESTED}"

  # Scope narrowing, applied only now. official_protocol.sh has already run and set
  # TASKS_JSON=1..26 and SEED=100, so this is the first point at which an override survives.
  # The arm files remain the single source of truth for the OFFICIAL protocol; this is the
  # documented, recorded narrowing of that scope for a subset / multi-seed run.
  #
  # SEED is exported unconditionally from the loop variable rather than defaulted: passing it
  # as an argument but letting the environment decide would let the two disagree, and the
  # output DIRECTORY name is derived from the same variable a few lines above, so a
  # disagreement would file the results under one seed and run another.
  if [[ -n "${TASKS_JSON_SCOPE}" ]]; then
    export TASKS_JSON="${TASKS_JSON_SCOPE}"
  fi
  # Same ordering rule as above, and load-bearing for the same reason: `official_protocol.sh` sets
  # NUM_TRIALS=1, so a pre-source override would be silently discarded and the run would cover one
  # seed per task while the banner claimed many.
  if [[ -n "${NUM_TRIALS_SCOPE}" ]]; then
    export NUM_TRIALS="${NUM_TRIALS_SCOPE}"
  fi
  export SEED="${seed}"
  # OUT_ROOT was exported before sourcing; re-asserting it here is harmless and keeps the value
  # correct even for an arm that overwrites it.

  echo
  echo "[INFO] === arm=${arm} seed=${seed} ==="
  echo "[INFO] PLANNER_BACKEND=${PLANNER_BACKEND} PLANNER_API_MODEL=${PLANNER_API_MODEL} PLANNER_API_KEY_LINE=${PLANNER_API_KEY_LINE}"
  echo "[INFO] PROACTIVE_MODE=${PROACTIVE_MODE:-unset} HARNESS_VLM_CONTEXT=${HARNESS_VLM_CONTEXT} VLM_USE_KEYFRAME_MEMORY=${VLM_USE_KEYFRAME_MEMORY} MEM_STAGE_ANCHOR=${MEM_STAGE_ANCHOR:-unset}"
  echo "[INFO] N_RECENT=${N_RECENT} K_MAX=${K_MAX} D_MERGE=${D_MERGE} SEED=${SEED} NUM_TRIALS=${NUM_TRIALS}"
  echo "[INFO] OUT_ROOT=${OUT_ROOT}"
  echo "[INFO] MEMEXP_PULL_ENABLE=${MEMEXP_PULL_ENABLE:-0} MEMEXP_PULL_TOOLS=${MEMEXP_PULL_TOOLS:-n/a} MEMEXP_PULL_REPORT=${MEMEXP_PULL_REPORT:-n/a}"
  echo "[INFO] gpu split server:${SERVER_CUDA_VISIBLE_DEVICES:-?} eval:${EVAL_CUDA_VISIBLE_DEVICES:-?} PORT=${PORT:-?}"

  # Freeze the environment that ACTUALLY ran, as an artifact rather than a recollection.
  {
    echo "# resolved environment of the arm actually run"
    echo "# job=${JOB_TAG} arm=${arm} tasks=${TASKS_JSON} seed=${seed}"
    # The trailing `[A-Z0-9_]*=` is load-bearing. With the `=` directly after the group as
    # `^(HARNESS_|...|PORT)=' it matches only a variable named EXACTLY `PORT`, so
    # `HARNESS_VLM_CONTEXT=1` and `PLANNER_API_MODEL=...` were silently dropped and this file --
    # the one artifact that records what an arm was configured with -- captured almost nothing.
    # A mismatch between "what the arm asked for" and "what actually ran" would then have been
    # invisible. `validate_arm.py` CHECK 11 replays representative keys through this pattern.
    #
    # PYTHONPATH is included deliberately. It is not a memory knob, but it is what decides whether
    # the memory-content correction installs at all, so a reader of this file must be able to see
    # it without re-deriving it from the arm scripts.
    env | grep -E '^(HARNESS_|PLANNER_|PROACTIVE_MODE|PMH_|PACT_|SEAM|ACE_|PCAM_|HPM_|SCEC_|CGMH_|META_|BCM_|MUSCLE_|MEM_|MEMEXP_|MEMPROF_|VLM_|VLA_|K_MAX|D_MERGE|N_RECENT|MAX_STEPS|POST_|FAIL_ON|ASYNC_|REPLAN_|NUM_TRIALS|SEED|TASKS_JSON|RESIZE_SIZE|READ_K_MAX|SERVER_CUDA|EVAL_CUDA|PORT|OUT_ROOT|PYTHONPATH)[A-Z0-9_]*=' | sort
  } > "${out_root}/resolved_env.txt"
  echo "[INFO] wrote ${out_root}/resolved_env.txt ($(wc -l < "${out_root}/resolved_env.txt") lines)"

  # WHAT CODE RAN, as opposed to what it was configured with. Both memory arms REPLACE parts of
  # the harness at runtime, so a score is a property of this experiment's files x the upstream
  # revision -- and the second factor is not recorded anywhere else. Without this, an upstream edit
  # to a patched file would change the meaning of an archived score with nothing to show for it.
  ROOT="${ROOT}" OUT_ROOT="${out_root}" MEMEXP_ARM_NAME="${arm}" \
    "${EVAL_PY}" "${EXP_DIR}/record_provenance.py" || true

  # The official evaluation path, verbatim.
  # EARLY STOP ON QUOTA. `api_planner.chat_completions` logs `API_QUOTA_EXHAUSTED` the moment
  # the relay returns insufficient_balance. Without a watcher, the evaluator keeps opening
  # episodes and the planner watchdog only kills them after N empty replies — each of which
  # is another paid 403. The watcher below tails the live log and SIGTERMs the evaluator as
  # soon as the marker appears; set EARLY_STOP_ON_QUOTA=0 to disable.
  #
  # Important: do NOT background `cmd | tee` — in bash $! is tee's PID, and killing tee
  # leaves the evaluator alive and still billing. Redirect to the log file, then mirror
  # with `tail --pid` so SLURM stdout stays readable.
  cd "${ROOT}/evaluation_benchmark/async_vlm26_reference"
  set +e
  if [[ "${EARLY_STOP_ON_QUOTA:-1}" == "1" ]]; then
    bash run_fullvlm26_async_vlm_vla_csr_tsr.sh >"${out_root}/run.log" 2>&1 &
    local eval_pid=$!
    tail -n +1 -F "${out_root}/run.log" --pid="${eval_pid}" 2>/dev/null &
    local tail_pid=$!
    local tee_ok=0
    while kill -0 "${eval_pid}" 2>/dev/null; do
      if grep -qE "API_QUOTA_EXHAUSTED|insufficient_balance|余额不足" "${out_root}/run.log" 2>/dev/null; then
        echo "[EARLY-STOP] API quota exhausted — killing evaluator pid=${eval_pid}" >&2
        kill -TERM "${eval_pid}" 2>/dev/null || true
        sleep 5
        kill -KILL "${eval_pid}" 2>/dev/null || true
        tee_ok=77
        break
      fi
      sleep 15
    done
    wait "${eval_pid}" 2>/dev/null
    local rc=$?
    kill "${tail_pid}" 2>/dev/null || true
    wait "${tail_pid}" 2>/dev/null || true
    if [[ "${tee_ok}" -eq 77 ]]; then
      rc=77
    fi
  else
    bash run_fullvlm26_async_vlm_vla_csr_tsr.sh 2>&1 | tee "${out_root}/run.log"
    local rc=${PIPESTATUS[0]}
  fi
  set -e
  cd "${ROOT}"
  bash "${ROOT}/scripts/merge_eval_outputs.sh" "${out_root}" || true
  return ${rc}
}

RC=0
# arm-outer, seed-inner. `RC` keeps only the LAST non-zero status, which is enough to fail the
# job loudly at the end; a per-arm failure that is later followed by a success still surfaces in
# that arm's run.log and in the census, and the census FAILs on missing reports regardless.
for arm in ${ARMS_TO_RUN}; do
  for seed in ${SEED_LIST}; do
    run_arm "${arm}" "${seed}" || RC=$?
  done
done

# ---------------------------------------------------------------------------------------
# Census. Two things are reported and they answer different questions:
#   * resolved_env.txt  -- what the process was configured with
#   * census_channels.py -- what the artifacts show actually happened
# A score is only interpretable alongside both.
# ---------------------------------------------------------------------------------------
echo
echo "=============================================================================="
echo "CENSUS"
echo "=============================================================================="
for arm in ${ARMS_TO_RUN}; do
  for seed in ${SEED_LIST}; do
    echo
    echo "--- ${arm} seed=${seed} ---"
    sed -n '1,4p' "${OUT_BASE}/${arm}_s${seed}/resolved_env.txt" 2>/dev/null || true
    echo "  memory channels requested:"
    grep -E '^(HARNESS_VLM_CONTEXT|VLM_USE_KEYFRAME_MEMORY|MEM_STAGE_ANCHOR|PROACTIVE_MODE)=' \
      "${OUT_BASE}/${arm}_s${seed}/resolved_env.txt" 2>/dev/null | sed 's/^/    /' || true
    echo "  scope actually run (from the frozen env, not from this script's intent):"
    grep -E '^(TASKS_JSON|SEED|NUM_TRIALS)=' \
      "${OUT_BASE}/${arm}_s${seed}/resolved_env.txt" 2>/dev/null | sed 's/^/    /' || true
  done
done

ROOT="${ROOT}" "${EVAL_PY}" "${EXP_DIR}/census_channels.py" "${OUT_BASE}" || RC=$?

echo
echo "[INFO] RC=${RC}  OUT_BASE=${OUT_BASE}  seeds=${SEED_LIST}  tasks=${TASKS_JSON_SCOPE:-<arm default 1..26>}"
if [[ "${RC}" -ne 0 ]]; then
  echo "[FAIL] rc=${RC}; inspect ${OUT_BASE}/*/run.log and the census above" >&2
  exit "${RC}"
fi
echo "[OK] mem_efficacy run complete."
