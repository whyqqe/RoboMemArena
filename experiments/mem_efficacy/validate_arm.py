#!/usr/bin/env python3
"""mem_efficacy preflight: prove the arms are officially aligned and their channels behave.

WHY THIS FILE EXISTS
--------------------
Every conclusion of this experiment rests on two claims that are easy to get wrong and
impossible to see in a score:

  1. "The baseline has no memory."  Configuration can lie about this. Several arms in
     `scripts/run_harness_variant.sh` export PMH_* / PACT_* / SEAM* / MEM_* variables, and a
     bash `export` survives into this arm unless it is explicitly purged. `HarnessConfig` reads
     `HARNESS_VLM_CONTEXT` with a default that is the opposite of what one expects. And whether
     a memory bank exists is unrelated to the resulting score.

  2. "This experiment follows the RMA official protocol."  The official runner resolves each
     knob as `${VAR:-official_default}`, so an arm that exports a knob silently overrides
     upstream. This already happened in this repository: the prior harness arm exported
     N_RECENT=7 / K_MAX=8 / D_MERGE=4 while the official protocol is 5 / 0 / 6.

So this script does not grep config files. It (a) diffs the arm files against the official
sbatch, and (b) runs REAL harness code inside each arm's own environment and reads back what a
Planner would actually receive.

ARMS
  Each arm is an independent file under `arms/`. The baseline is `nomem`; every other arm
  (except the shared `official_protocol.sh`) is a treatment, and each must declare the exact
  set of keys it is allowed to change in `MEMEXP_EXPECTED_DIFF`. Adding a new arm therefore
  cannot perturb the others: this script validates each one against its own declaration.

CHECKS
  1  Every treatment differs from the baseline in exactly the keys it declares.
  2  The baseline holds no non-allowlisted memory-layer key, and every allowlisted one is off.
  3  Every non-experimental official knob in the arms equals the value parsed out of
     `slurm/benchmark/reproduce_all26_1seed.sbatch`. Fails if upstream changes.
  4  Scope is 26 tasks / seed 100 / NUM_TRIALS=1 in every arm.
  5  Functional probe on a real HarnessController: the baseline's `get_vlm_context()` is empty
     and `compose_planner_context()` returns ""; each treatment's is non-empty (positive
     control). This is the proof that the payload channel is genuinely alive.
  6  Channel B resolves as designed under the same expression the evaluator uses.
  7  BANK-BUILDER PROBE: with a Planner that never nominates a keyframe, a nomination-only bank
     is empty while a stage-anchored bank is non-empty. This is what makes the image channel a
     property of the arm rather than of the Planner's compliance, and it is the failure that
     silently voided HM's image channel (see README "NULL comparison").
 14  IN-PROCESS REPORTS ARE LIVE: the counters behind the post-run evidence are flushed during
     the run (per planning step and per tool call), written to a per-process file, and merged by
     the census. Guards the failure where an artifact reports a confident zero because nothing
     ever updated it -- job 591479's `steps: 0, tool_calls: 0` was this, not a finding.

Exit code 0 = all passed. Non-zero = do not spend GPUs on these arms.

USAGE
  python3 validate_arm.py
  (run with the repository venv: <ROOT>/.venv/bin/python)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT") or "/project/peilab/why/RoboMemArena")
HERE = Path(__file__).resolve().parent
ARMS_DIR = HERE / "arms"

UPSTREAM_SBATCH = ROOT / "slurm" / "benchmark" / "reproduce_all26_1seed.sbatch"

BASELINE = "nomem"
SHARED_ARM = "official_protocol"  # sourced by nomem.sh; not an arm of its own

# Variables that exist in any shell and carry no experimental meaning.
SHELL_NOISE = {
    "_", "SHLVL", "PWD", "OLDPWD", "PATH", "LANG", "LANGUAGE", "LC_ALL",
    "TERM", "SHELL", "HOSTNAME", "HOME", "USER", "LOGNAME", "TMPDIR",
    "EUID", "UID", "PPID", "RANDOM", "SECONDS", "LINENO", "FUNCNAME",
    "GROUPS", "DIRSTACK", "PIPESTATUS", "IFS", "PS1", "PS2", "PS3", "PS4",
    "BASH", "BASHOPTS", "BASH_ALIASES", "BASH_ARGC", "BASH_ARGV", "BASH_COMMAND",
    "BASH_LINENO", "BASH_SOURCE", "BASH_SUBSHELL", "BASH_VERSINFO", "BASH_VERSION",
    "COMP_WORDBREAKS", "MACHTYPE", "OPTERR", "OSTYPE", "SHELLOPTS",
    "ROOT",  # injected by this script, not part of any arm
}

MEMORY_PREFIXES = ("PMH_", "PACT_", "SEAM", "KAIROS_", "ACE_", "PCAM_", "HPM_",
                   "SCEC_", "CGMH_", "MUSCLE_", "MEM_")

# Memory-layer keys the baseline is ALLOWED to contain, provided they are off. Writing `=0` is
# an explicit disable, not a leak; a key outside this set (e.g. PMH_ENABLE=1) is a real leak.
# Conflating the two produces false failures that hide the true problem.
ALLOWED_MEMORY_KEYS = {
    "META_ENABLE", "BCM_ENABLE", "SCEC_ENABLE", "ACE_ENABLE", "PCAM_ENABLE",
    "HPM_ENABLE", "CGMH_GATE", "PROACTIVE_MODE", "MEM_STAGE_ANCHOR",
}
# Mechanism redaction is not a memory layer; it is an optional answer-withholding ablation.
MEMORY_KEYS_THAT_MAY_BE_ON = {"PMH_REDACT_STAGE"}

PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)


def _is_off(value: str) -> bool:
    return str(value).strip().lower() in {"0", "false", "no", "off", ""}


# ---------------------------------------------------------------------------------------
# Official protocol parsing
# ---------------------------------------------------------------------------------------
_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*?)\s*$")
_UNSET_RE = re.compile(r"^\s*unset\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")


def parse_official_sbatch(path: Path) -> tuple[dict[str, str], set[str]]:
    """Parse `export KEY=VALUE` / `unset KEY` out of the official runner.

    Returns (exports, unsets). Shell quoting is stripped so `'[1,2]'` and `[1,2]` compare
    equal, which is what matters for an equality check against another shell file.
    """
    if not path.exists():
        raise FileNotFoundError(f"official sbatch not found: {path}")
    exports: dict[str, str] = {}
    unsets: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0] if raw.lstrip().startswith("#") else raw
        m = _EXPORT_RE.match(line)
        if m:
            key, val = m.group(1), m.group(2)
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            exports[key] = val
            continue
        m = _UNSET_RE.match(line)
        if m:
            unsets.add(m.group(1))
    return exports, unsets


def discover_arms() -> list[str]:
    """Treatment arms = every arm file except the baseline and the shared protocol file.

    Files whose stem starts with `_` are shared helpers, not arms. Without this rule the helper
    that installs the memory correction would be discovered as a fourth arm, and `validate_arm.py`
    would try to source it as an experiment and assert an `ARM_NAME` it does not set.
    """
    arms = sorted(
        p.stem for p in ARMS_DIR.glob("*.sh")
        if p.stem != SHARED_ARM and not p.stem.startswith("_")
    )
    return [a for a in arms if a != BASELINE], arms


def dump_arm_env(arm: str, after: list[str] | None = None) -> dict[str, str]:
    """Source an arm in a clean shell and read back everything it exported.

    `after` optionally sources those arms FIRST, in order, in the same shell. That is how the
    sequence defect is reproduced: the real runner sources one arm per evaluation IN THE SAME
    SHELL (`run_arm` in `run_26x1.sbatch`), so an arm's effective environment depends on what
    ran before it.

    Built as one prompt, not by exporting a hand-picked set of variables between calls, because
    the whole class of bug being tested for is "a variable nobody remembered to name". A test
    that passes only the variables it knows about cannot see the one it forgot.
    """
    parts = []
    for a in (after or []):
        parts.append(f'source "{ARMS_DIR / (a + ".sh")}"')
    parts.append(f'source "{ARMS_DIR / (arm + ".sh")}"')
    script = "set -a; " + "; ".join(parts) + "; env -0"
    proc = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "ROOT": str(ROOT)},
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sourcing arms/{arm}.sh failed:\n{proc.stderr[-800:]}")
    out: dict[str, str] = {}
    for chunk in proc.stdout.split("\0"):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        if k not in SHELL_NOISE:
            out[k] = v
    return out


# ---------------------------------------------------------------------------------------
# Functional probes: run real harness code inside the arm's environment
# ---------------------------------------------------------------------------------------
PROBE_SRC = r'''
import json, os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark"))
res = {}
try:
    from harness.config import load_harness_config
    from harness.controller import HarnessController
    from harness.belief_contract_memory import compose_planner_context

    ti = SimpleNamespace(
        task_id=1, task_block="blk",
        primitive_labels=["open top drawer", "check contents"],
        task_name="t1", scene_description="sd", brief_description="bd",
    )
    cfg = load_harness_config()
    ctrl = HarnessController.create(1, ti, cfg, memory_root=None)
    ctrl.begin_attempt(0)
    spec = SimpleNamespace(name="01_Open_Drawer")
    # The real read path: what a stage advance hands to the Planner.
    ctrl._refresh_vlm_context(0, [spec], "open drawer", 10)
    ctx = ctrl.get_vlm_context()
    res["inject_vlm_context"] = bool(cfg.inject_vlm_context)
    res["vlm_context_len"] = len(ctx)
    res["compose_len"] = len(compose_planner_context(harness=ctrl))
    res["vlm_context_head"] = (ctx.splitlines() or [""])[0][:60]
    # Channel B, parsed exactly as the evaluator does
    # (async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py:1939).
    res["keyframe_memory"] = os.environ.get("VLM_USE_KEYFRAME_MEMORY", "1") in {"1", "true", "yes"}
    # Scope, parsed exactly as the runner does.
    res["n_recent"] = int(os.environ.get("N_RECENT", "5"))
    res["k_max"] = int(os.environ.get("K_MAX", "0"))
    res["d_merge"] = int(os.environ.get("D_MERGE", "6"))
except Exception as exc:
    import traceback
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()[-1200:]
print("@@PROBE@@" + json.dumps(res))
'''

# CHECK 7: build a keyframe bank twice, once the way the official protocol does (nomination
# only) and once the way a stage-anchored arm does. The Planner is simulated as emitting NO
# `keyframe_positions`, which is what an API Planner actually does, while the eval loop's own
# deterministic signals are supplied.
BANK_PROBE_SRC = r'''
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark"))
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark", "openpi_minimal_runtime"))
res = {"error": None}
try:
    from keyframe_selection import build_visual_memory
    from memory_system.keyframe_bank import merge_keyframe_bank

    # A 600-step episode; the recent window is the last 5 steps; a stage advanced at step 200
    # and the Planner changed subtask at step 380. Both are real signals the eval loop emits.
    t, recent_window, d = 600, 5, 6
    pinned, salient = [200], [380]
    # An API Planner emits `keyframe_positions: []` on every step, so J_hist is 600 empty lists.
    j_hist_empty = [[] for _ in range(t)]

    res["nomination_only"] = build_visual_memory(j_hist_empty, t=t, N=recent_window, d=d)
    res["stage_anchored"] = merge_keyframe_bank(
        j_hist=j_hist_empty, t=t, recent_window=recent_window, cluster_distance=d,
        pinned_steps=pinned, salient_steps=salient, bank_max=0,
    )
    # And when the Planner DOES nominate, the anchored bank must still contain them: the
    # anchored builder is a superset, so this arm can never lose the official behaviour.
    j_hist_nom = [[] for _ in range(t)]
    j_hist_nom[300] = [300]
    res["stage_anchored_with_nomination"] = merge_keyframe_bank(
        j_hist=j_hist_nom, t=t, recent_window=recent_window, cluster_distance=d,
        pinned_steps=pinned, salient_steps=salient, bank_max=0,
    )
except Exception as exc:
    import traceback
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()[-1200:]
print("@@BANK@@" + json.dumps(res))
'''


# CHECK 8: the PULL arm's binding, proven by running it. Executed in the arm's own environment
# with PYTHONPATH pointing at pysite/, which is how the arm activates without editing shared code.
PULL_PROBE_SRC = r'''
import json, os, sys
# Both paths, mirroring what the evaluator sets up before it imports the planner:
# `evaluation_benchmark` for `harness.*` and `evaluation_benchmark/openpi_minimal_runtime` for
# `keyframe_selection` (which `api_vlm_planner` imports at module scope). Without the second one
# the probe dies on the import and would report a harness problem that does not exist.
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark"))
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark", "openpi_minimal_runtime"))
res = {"error": None}
try:
    # Importing the planner is what triggers the post-exec hook installed by sitecustomize.
    import harness.api_vlm_planner  # noqa: F401
    import memexp_bind
    v = memexp_bind.verify_binding()
    res["verify"] = v
    res["live"] = v.get("live", {})
    res["problems"] = v.get("problems", [])
    res["ok"] = bool(v.get("ok"))
except Exception as exc:
    import traceback
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()[-1500:]
print("@@PULL@@" + json.dumps(res))
'''


def _run_probe(src: str, marker: str, arm_env: dict[str, str]) -> dict:
    proc = subprocess.run(
        [str(PY), "-c", src],
        env={**arm_env, "PATH": os.environ.get("PATH", ""), "ROOT": str(ROOT)},
        capture_output=True, text=True, timeout=180,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(
        f"probe produced no output (rc={proc.returncode})\n"
        f"stdout={proc.stdout[-600:]}\nstderr={proc.stderr[-1200:]}"
    )


def run_probe(arm_env: dict[str, str]) -> dict:
    return _run_probe(PROBE_SRC, "@@PROBE@@", arm_env)


# CHECK 16 / 17: the two repairs made after job 593817. Both are asserted by RUNNING them, because
# each one's failure mode is "the flag is set, the code is present, and nothing happens" -- the
# shape this project has now paid for five times.
SDV_STAGE_PROBE_SRC = r'''
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark"))
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark", "openpi_minimal_runtime"))
res = {"error": None}
try:
    import harness.api_vlm_planner  # triggers the post-exec hook
    import memexp_bind
    from harness.controller import HarnessController

    res["hook_installed"] = bool(memexp_bind._STATE.get("stage_hook_installed"))
    res["hook_error"] = memexp_bind._STATE.get("stage_hook_error")

    # Stage objects only need `.name`; the driver reads nothing else. Built locally so the probe
    # does not depend on a task id or on `task2_26_reference_stage` importing cleanly.
    class S:
        def __init__(self, n): self.name = n
    specs = [S("01_Alpha"), S("02_Beta"), S("03_Gamma")]

    # Constructed WITHOUT `__init__` on purpose: the real constructor needs (config, task_id,
    # task_info, memory), none of which this observation channel touches, and building four real
    # objects would make the check depend on subsystems that have nothing to do with the stage
    # driver. The wrapper records the stage BEFORE delegating, so what the delegate does is
    # irrelevant here -- and with `actuator_guard` off it is the identity branch anyway.
    h = object.__new__(HarnessController)
    h.config = type("Cfg", (), {"actuator_guard": False})()
    h.n_actuator_emitted_vlm = 0
    h.n_actuator_gate_closed_respec = 0
    h._active_respec = ""
    h.override_vla_prompt("base prompt", stage_idx=1, stage_specs=specs,
                          stage_done={}, step=42)

    # THE DEFECT, asserted directly: the driver must return the stage for a planner whose
    # `episodic_store` is None, because that is the state this arm actually runs in
    # (PROACTIVE_MODE=off -> create_store_if_needed returns None). Reading the store was the bug.
    class P:
        episodic_store = None
    res["stage_without_store"] = memexp_bind._current_stage(P())
    res["active_stage_state"] = memexp_bind._STATE.get("active_stage")
    res["stage_hook_calls"] = memexp_bind._STATE.get("stage_hook_calls")
    res["stage_names_seen"] = memexp_bind._STATE.get("stage_names_seen")

    # And the old source, for contrast: a store that exists still works, so this is a fallback
    # rather than a replacement.
    class Store:
        current_stage_name = "09_From_Store"
    class P2:
        episodic_store = Store()
    memexp_bind._STATE["active_stage"] = ""
    res["stage_from_store"] = memexp_bind._current_stage(P2())
    res["empty_when_nothing"] = memexp_bind._current_stage(P())
except Exception as exc:
    import traceback
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()[-1500:]
print("@@SDVSTAGE@@" + json.dumps(res))
'''

KF_SPREAD_PROBE_SRC = r'''
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark"))
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark", "openpi_minimal_runtime"))
res = {"error": None}
try:
    import harness.api_vlm_planner as m
    from memory_system.config import load_memory_system_config

    res["declared_spread"] = os.environ.get("MEM_KF_SPREAD")
    res["cfg_spread"] = load_memory_system_config().kf_spread

    # A planner that never nominates: J_hist is a list of empty lists, `kf_n = 0`. This is the API
    # Planner's real state and the reason the bank needed a model-free candidate source.
    p = object.__new__(m.ApiMemoryPlanner)
    p.frame_store_main = {i: object() for i in range(600)}
    anchors = [80, 168]
    recent_start = 500
    res["anchors_only"] = len(anchors)
    out = p._kf_spread_union(anchors, recent_start, 8)
    res["union_len"] = len(out)
    res["union"] = out
    # NO ENDPOINT CLINGING, and this is the assertion that would have caught the defect. The
    # stride used to be `older[int(k * step)]`, whose k=0 term is unconditionally `older[0]` --
    # the OLDEST frame in the pool, i.e. the episode's opening, before the robot has touched
    # anything. The official builder ALSO anchors frame 0 (the step-0 subtask change is salient),
    # so the pick landed on frame 1: a near-duplicate, present on 70.1% of real bank-bearing steps
    # in pushmem and 44.8% in pullmem (job 594675), against 2-5% for every other low frame. One
    # of the eight bank slots was spent on the opening on almost every step.
    res["early_picks"] = [i for i in out if i <= 8]
    # Additivity: nothing the official builder selected may be dropped.
    res["kept_anchors"] = all(a in out for a in anchors)
    # The recent window is the Planner's own context; re-injecting it moves no information.
    res["all_before_recent"] = all(i < recent_start for i in out)
    # A spread, not a clump: the FIRST and LAST picked frames must be far apart, which is the
    # property a naive `[-want:]` would fail.
    res["span"] = (max(out) - min(out)) if out else 0
    # No duplicates. This is the invariant that matters, NOT idempotence: the call site re-runs the
    # official builder every step and passes the ANCHOR set in, never a previously-unioned set, so
    # "unioning its own output is a no-op" is not a property anything depends on -- and asserting it
    # was wrong, because given budget it correctly adds DIFFERENT frames.
    res["unique"] = len(out) == len(set(out))
    # The real second-call invariant, and the one the call site hits constantly: a zero budget (a
    # full bank) must return the bank unchanged. The first version reached this by dividing by zero
    # and catching its own exception on every plan step.
    res["zero_budget_identity"] = sorted(p._kf_spread_union(out, recent_start, 0)) == sorted(out)
    # And with no history at all it must return the input unchanged rather than raising.
    p.frame_store_main = {}
    res["empty_store_ok"] = p._kf_spread_union(anchors, recent_start, 8) == anchors
except Exception as exc:
    import traceback
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()[-1500:]
print("@@KFSPREAD@@" + json.dumps(res))
'''


def run_sdv_stage_probe(arm_env: dict[str, str]) -> dict:
    return _run_probe(SDV_STAGE_PROBE_SRC, "@@SDVSTAGE@@", arm_env)


def run_kf_spread_probe(arm_env: dict[str, str]) -> dict:
    return _run_probe(KF_SPREAD_PROBE_SRC, "@@KFSPREAD@@", arm_env)


# -------------------------------------------------------------------------------------------
# CHECK 19 -- the keyframe channel's CANDIDATE POOL is a real record of the episode.
#
# This is the falsifier for the defect that made channel B inert while every liveness counter said
# it was healthy. `_kf_spread_union` strides evenly over `frame_store_main`, so the bank can only
# be as good as the store. The store used to be filled ONLY from the context windows of planner
# calls that reached the model, and with the official `VLM_INTERVAL=5` / `VLM_QUEUE_SIZE=1` those
# submissions evict each other: measured on job 594338, an episode of ~2470 env steps produced 3-9
# calls, i.e. ~35 stored frames (1.4% of the episode, weighted to the start). The resulting bank
# was 54-62% frames <= 8 with frame 0 on 100% of plan steps -- the channel was live and empty.
#
# The probe below runs the SAME union over the two stores: the call-window-only store, and the
# dense one that `MEM_KF_STORE_INTERVAL` produces. The assertion is a CONTRAST, not a threshold on
# one number, because a single number cannot tell "the spread works" from "the pool was already
# spread" -- and it was the second case that was being read as the first.
KF_STORE_PROBE_SRC = r'''
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark"))
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark", "openpi_minimal_runtime"))
res = {"error": None}
try:
    import harness.api_vlm_planner as m

    res["declared"] = os.environ.get("MEM_KF_STORE_INTERVAL")
    res["interval"] = None
    try:
        res["interval"] = int(str(res["declared"]))
    except Exception:
        pass

    EPISODE = 2470
    RECENT_START = EPISODE - 8

    # (a) What the store USED to contain: only the 7-frame window of each planner call that
    #     survived async queue eviction. Call steps are the measured ones from job 594338.
    call_steps = [0, 405, 1005, 1585, 1890]
    sparse = {}
    for c in call_steps:
        for off in range(7):
            sparse[c + off] = object()

    # (b) What MEM_KF_STORE_INTERVAL=5 produces: one frame per replan window, across the episode.
    dense = {i: object() for i in range(0, EPISODE + 1, 5)}

    def measure(store, label):
        p = object.__new__(m.ApiMemoryPlanner)
        p.frame_store_main = dict(store)
        out = p._kf_spread_union([], RECENT_START, 8)
        early = [i for i in out if i <= 8]
        return {
            "label": label,
            "n_store": len(store),
            "picked": out,
            "n_picked": len(out),
            "span": (max(out) - min(out)) if out else 0,
            "n_early": len(early),
            "early_share": (len(early) / len(out)) if out else 0.0,
            "has_frame0": 0 in out,
        }

    res["sparse"] = measure(sparse, "call-window-only store")
    res["dense"] = measure(dense, "dense store (MEM_KF_STORE_INTERVAL)")
except Exception as exc:
    import traceback
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()[-1500:]
print("@@KFSTORE@@" + json.dumps(res))
'''


def run_kf_store_probe(arm_env: dict[str, str]) -> dict:
    return _run_probe(KF_STORE_PROBE_SRC, "@@KFSTORE@@", arm_env)


def run_bank_probe(arm_env: dict[str, str]) -> dict:
    return _run_probe(BANK_PROBE_SRC, "@@BANK@@", arm_env)


# Reads the nomination switch back inside a real arm environment. Two roots, the same two the
# spread probe uses -- `keyframe_selection` (needed by `memory_system.keyframe_bank`) lives under
# `openpi_minimal_runtime`.
NOM_CFG_PROBE_SRC = r'''
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark"))
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark", "openpi_minimal_runtime"))
res = {"error": None}
try:
    from memory_system.config import MemorySystemConfig, load_memory_system_config

    res["declared_env"] = os.environ.get("MEM_KF_NOMINATION_PROMPT")
    # The dataclass default, i.e. what an arm that says nothing gets.
    res["default"] = bool(MemorySystemConfig().nomination_prompt)
    # What the arm's environment actually resolves to, defaults applied.
    res["from_env"] = bool(load_memory_system_config().nomination_prompt)
except Exception as exc:
    res["error"] = f"{type(exc).__name__}: {exc}"
print("@@NOMCFG@@ " + json.dumps(res))
'''


def main() -> int:
    failures: list[str] = []
    notes: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if detail:
            for ln in detail.splitlines():
                print(f"         {ln}")
        if not ok:
            failures.append(label)

    treatments, all_arms = discover_arms()

    print("=" * 78)
    print("mem_efficacy preflight")
    print(f"  ROOT       = {ROOT}")
    print(f"  PY         = {PY}")
    print(f"  baseline   = {BASELINE}")
    print(f"  treatments = {treatments or '<none>'}")
    print("=" * 78)

    env_base = dump_arm_env(BASELINE)
    env_by_arm = {BASELINE: env_base}
    for arm in treatments:
        env_by_arm[arm] = dump_arm_env(arm)

    def structural(d: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in d.items() if not k.startswith("MEMEXP_")}

    sb = structural(env_base)

    # ---- check 1: each treatment changes exactly what it declares ------------------------
    print("\n[CHECK 1] each treatment differs from the baseline only in its declared keys")
    declared: dict[str, set[str]] = {}
    all_expected: set[str] = set()
    for arm in treatments:
        raw = env_by_arm[arm].get("MEMEXP_EXPECTED_DIFF", "")
        want = {k for k in raw.split() if k}
        declared[arm] = want
        all_expected |= want
        st = structural(env_by_arm[arm])
        diff = {k for k in set(sb) | set(st) if sb.get(k) != st.get(k)}
        check(
            want and diff == want,
            f"{arm}: diff = {sorted(diff)}",
            f"declared {sorted(want)}" if diff == want
            else f"declared {sorted(want)}; unexpected {sorted(diff ^ want)}",
        )
    notes.append(f"{len(treatments)} treatment arm(s); shared knobs {len(sb)}")

    # ---- check 2: no memory-layer leak in the baseline -----------------------------------
    print("\n[CHECK 2] baseline holds no non-allowlisted memory key; allowlisted ones are off")
    leaked = sorted(
        k for k in env_base
        if k.startswith(MEMORY_PREFIXES)
        and k not in ALLOWED_MEMORY_KEYS
        and k not in MEMORY_KEYS_THAT_MAY_BE_ON
    )
    check(not leaked, f"non-allowlisted memory keys = {len(leaked)}",
          f"leaked: {[f'{k}={env_base[k]}' for k in leaked]}" if leaked else "")

    still_on = sorted(
        f"{k}={env_base[k]}" for k in ALLOWED_MEMORY_KEYS
        if k in env_base and not _is_off(env_base[k])
    )
    check(not still_on, f"all {len(ALLOWED_MEMORY_KEYS)} allowlisted switches are off",
          f"still on: {still_on}" if still_on else "")

    present = sorted(k for k in ALLOWED_MEMORY_KEYS if k in env_base)
    notes.append(f"explicitly disabled memory switches: {len(present)}/{len(ALLOWED_MEMORY_KEYS)}")
    if len(present) < len(ALLOWED_MEMORY_KEYS):
        notes.append(
            "not mentioned (rely on defaults, covered by the CHECK 5 probe): "
            f"{sorted(ALLOWED_MEMORY_KEYS - set(present))}"
        )

    # ---- check 3: official-protocol alignment --------------------------------------------
    print("\n[CHECK 3] official-protocol knobs match upstream")
    print(f"  source of truth: {UPSTREAM_SBATCH}")
    try:
        official, official_unsets = parse_official_sbatch(UPSTREAM_SBATCH)
    except Exception as exc:
        check(False, "parse official sbatch", f"{type(exc).__name__}: {exc}")
        official, official_unsets = {}, set()

    if official:
        # A knob that the arms set but upstream pins differently is either (a) an intended
        # experimental variable, or (b) an unintended drift. Only (a) is allowed, and CHECK 1
        # independently proves that the deviation set is exactly the declared keys. So this
        # exempts that set and demands exact equality for every other official knob -- which is
        # what makes a future upstream change fail loudly instead of being absorbed.
        mismatched: list[str] = []
        exempted: list[str] = []
        checked = 0
        for k in sorted(official):
            if k not in sb:
                continue
            if k in all_expected:
                exempted.append(f"{k}: baseline={sb[k]!r} official={official[k]!r}")
                continue
            checked += 1
            if sb[k] != official[k]:
                mismatched.append(f"{k}: baseline={sb[k]!r} official={official[k]!r}")
        check(not mismatched,
              f"{checked} non-experimental official knobs match upstream exactly",
              "\n".join(mismatched) if mismatched else "")
        if exempted:
            notes.append("intentional deviations from the official protocol (asserted by CHECK 1): "
                         + "; ".join(exempted))

        # `unset` loop variables in the sbatch (e.g. `unset _glu_dir`) are not protocol
        # directives; only names that look like real config keys are meaningful here.
        real_unsets = sorted(k for k in official_unsets if not k.startswith("_"))
        unsets_ok = sorted(k for k in real_unsets if k in sb)
        check(not unsets_ok,
              f"official `unset` directives honoured ({real_unsets or 'none'})",
              f"baseline still sets: {unsets_ok}" if unsets_ok else "")
        notes.append(f"official sbatch pins {len(official)} exports and unsets {len(real_unsets)}")

    # ---- check 4: scope ------------------------------------------------------------------
    print("\n[CHECK 4] scope = 26 tasks, NUM_TRIALS=1, SEED=100")
    for arm in all_arms:
        env = env_by_arm[arm]
        try:
            tasks = json.loads(env.get("TASKS_JSON", "[]"))
        except Exception:
            tasks = []
        check(
            tasks == list(range(1, 27))
            and env.get("NUM_TRIALS") == "1"
            and env.get("SEED") == "100",
            f"{arm}: TASKS_JSON=1..26 ({len(tasks)}), NUM_TRIALS={env.get('NUM_TRIALS')}, "
            f"SEED={env.get('SEED')}",
        )

    # ---- checks 5 and 6: functional probes -----------------------------------------------
    print("\n[CHECK 5] functional probe on a real HarnessController (channel A)")
    probes: dict[str, dict] = {}
    crashed = False
    for arm in all_arms:
        p = run_probe(env_by_arm[arm])
        probes[arm] = p
        if "error" in p:
            check(False, f"{arm} probe crashed", p.get("traceback", p["error"]))
            crashed = True
    if crashed:
        print("\npreflight cannot continue.")
        return _summary(failures, notes)

    check(probes[BASELINE]["inject_vlm_context"] is False, "baseline inject_vlm_context = False")
    check(probes[BASELINE]["vlm_context_len"] == 0,
          "baseline get_vlm_context() is empty (channel A is dead)",
          f"len={probes[BASELINE]['vlm_context_len']}")
    check(probes[BASELINE]["compose_len"] == 0,
          "baseline compose_planner_context() is empty (never enters the Planner prompt)",
          f"len={probes[BASELINE]['compose_len']}")
    # `HARNESS_VLM_CONTEXT` used to be the experimental variable, and this loop used to assert that
    # every treatment arm had it ON. It is now a controlled CONSTANT: the treatment arms turn it
    # OFF along with the baseline, because our own 5-seed paired study attributes the gain to the
    # VISUAL channel and measures the text channel as a net loss (memory_kf +8.2 pp / 5-of-5 seeds
    # versus memory_ctx +4.0 pp / 3-of-5). See `arms/pushmem.sh` for the full attribution, and job
    # 593253 for the local measurement of the same effect.
    #
    # The assertion therefore splits in two, and BOTH halves are needed:
    #   1. every arm agrees on channel A. If one arm silently flipped it back on, that arm's
    #      difference from the baseline would include the text channel, and push-vs-pull would no
    #      longer be a comparison of who selects the images;
    #   2. the channel is OFF but still FUNCTIONAL, proven by forcing it ON in a probe. Without
    #      this, switching the channel off everywhere would make the whole check vacuous -- an
    #      empty context would be indistinguishable from a broken renderer.
    declared_a = {arm: str(env_by_arm[arm].get("HARNESS_VLM_CONTEXT", "")).strip() for arm in all_arms}
    check(len(set(declared_a.values())) == 1,
          "every arm declares the SAME channel-A setting (a controlled constant, not a variable)",
          f"{declared_a}")
    a_off = declared_a.get(BASELINE, "") in {"0", "false", "no", "off"}
    check(a_off,
          "channel A is OFF in every arm, so the arms differ in the visual channel only",
          f"{declared_a}")
    for arm in all_arms:
        p = probes[arm]
        if declared_a.get(arm, "") in {"1", "true", "yes", "on"}:
            check(p["vlm_context_len"] > 0,
                  f"{arm} declares channel A ON and get_vlm_context() is non-empty",
                  f"len={p['vlm_context_len']}, head={p['vlm_context_head']!r}")
        else:
            check(p["vlm_context_len"] == 0,
                  f"{arm} declares channel A OFF and get_vlm_context() is empty",
                  f"len={p['vlm_context_len']}, head={p['vlm_context_head']!r}")
    # Positive control for the renderer itself, independent of what any arm chooses.
    _forced = dict(env_by_arm[BASELINE])
    _forced["HARNESS_VLM_CONTEXT"] = "1"
    _fc = run_probe(_forced)
    check(_fc.get("vlm_context_len", 0) > 0,
          "positive control: with HARNESS_VLM_CONTEXT=1 forced ON the context IS non-empty "
          "(so OFF is a choice, not a broken renderer)",
          f"len={_fc.get('vlm_context_len')}, error={_fc.get('error')!r}")

    print("\n[CHECK 6] channel B resolves as designed")
    check(probes[BASELINE]["keyframe_memory"] is False, "baseline use_keyframe_memory = False")
    for arm in treatments:
        want = arm in declared and "VLM_USE_KEYFRAME_MEMORY" in declared[arm]
        got = probes[arm]["keyframe_memory"]
        check(got is want, f"{arm} use_keyframe_memory = {got} (declared change: {want})")
    for arm in all_arms:
        p = probes[arm]
        print(f"         {arm}: N_RECENT={p['n_recent']} K_MAX={p['k_max']} D_MERGE={p['d_merge']}")

    # ---- check 7: bank-builder probe ------------------------------------------------------
    print("\n[CHECK 7] bank builders under a Planner that never nominates a keyframe")
    try:
        bank = run_bank_probe(env_base)
    except Exception as exc:
        bank = {"error": f"{type(exc).__name__}: {exc}"}
    if bank.get("error"):
        check(False, "bank-builder probe crashed", bank.get("traceback", str(bank["error"])))
    else:
        nom_only = bank["nomination_only"]
        anchored = bank["stage_anchored"]
        superset = bank["stage_anchored_with_nomination"]
        check(nom_only == [],
              "nomination-only bank (official MEM_STAGE_ANCHOR=0) is EMPTY without nominations",
              f"build_visual_memory -> {nom_only}")
        check(len(anchored) > 0,
              "stage-anchored bank (MEM_STAGE_ANCHOR=1) is NON-EMPTY without nominations",
              f"merge_keyframe_bank -> {anchored}")
        check(set(anchored) <= set(superset),
              "stage-anchored bank is a SUPERSET of the nomination-only bank",
              f"with a nomination at step 300 -> {superset}")
        # The consequence for each arm, stated as a property of the arm.
        for arm in treatments:
            anchor_on = env_by_arm[arm].get("MEM_STAGE_ANCHOR", "0") not in {"0", ""}
            kf_on = probes[arm]["keyframe_memory"]
            if kf_on and not anchor_on:
                notes.append(
                    f"{arm}: MEM_STAGE_ANCHOR=0 with the keyframe channel ON. Under an API "
                    "Planner this bank is empty at EVERY step, because the Planner never emits "
                    "`keyframe_positions`. Expect the census to report NULL for channel B -- "
                    "that is a void comparison, not a negative result."
                )
            elif kf_on and anchor_on:
                notes.append(
                    f"{arm}: MEM_STAGE_ANCHOR=1, so the keyframe bank is sourced from the "
                    "harness's own stage boundaries and the Planner's subtask changes. The "
                    "image channel is genuinely live without depending on nominations."
                )

    # ---- check 8: the pull arm's binding ---------------------------------------------------
    pull_arms = [
        a for a in treatments
        if str(env_by_arm[a].get("MEMEXP_PULL_ENABLE", "")).strip() in {"1", "true", "yes", "on"}
    ]
    if not pull_arms:
        notes.append("no PULL arm present; CHECK 8 skipped")
    else:
        print("\n[CHECK 8] the PULL arm's binding, run in its own environment")
        for arm in pull_arms:
            try:
                p = _run_probe(PULL_PROBE_SRC, "@@PULL@@", {
                    **env_by_arm[arm],
                    "PATH": os.environ.get("PATH", ""),
                    "ROOT": str(ROOT),
                })
            except Exception as exc:
                check(False, f"{arm}: pull probe crashed", f"{type(exc).__name__}: {exc}")
                continue
            if p.get("error"):
                check(False, f"{arm}: pull probe raised", p.get("traceback", p["error"]))
                continue
            live = p.get("live", {})
            check(bool(p.get("ok")), f"{arm}: memexp_bind.verify_binding() = ok",
                  "\n".join(p.get("problems", [])[:8]))
            # The three properties that decide whether the arm can measure anything at all.
            for key, label in (
                ("api_call_wrapped",
                 "the tool loop is installed (infer_primitive_via_api wrapped)"),
                ("build_messages_wrapped", "the push hook is installed (_build_messages wrapped)"),
                ("arity_two_addresses_distinct",
                 "ARITY: two containers are two facts, not one overwritten slot"),
                ("gain_filter_excludes_on_context",
                 "GAIN: a served frame is never one already on context"),
                ("gain_first_query_serves_frames",
                 "GAIN-POSITIVE: a cold address DOES serve frames (the filter above cannot pass "
                 "by serving nothing)"),
                ("unobserved_frames_not_served",
                 "TIMING: a frame that does not exist yet is NOT served"),
                ("frames_appear_when_they_exist",
                 "TIMING: those same frames ARE served once they exist"),
                ("early_query_says_not_yet",
                 "MESSAGE: an early query says 'not observed yet', so it cannot be read as "
                 "'empty'"),
                ("late_query_says_already_have",
                 "MESSAGE: an all-on-context query says the frames are already held"),
                ("serve_is_bounded",
                 f"a dereference attaches at most SERVE_MAX={live.get('serve_cap')} frames"),
                ("query_returns_observation_not_answer",
                 "the tool returns the observation to look at, not a fabricated answer"),
                ("miss_is_reported", "an unmatched address FAILS rather than silently succeeding"),
                ("primitive_not_read_as_tool",
                 "a primitive is not misread as a tool call (the loop cannot swallow it)"),
                ("repeat_is_cached", "a repeated query in one step is cached, not re-served"),
                ("spec_is_neutral", "the tool spec contains no discouraging default"),
                ("spec_does_not_solicit_enumeration",
                 "SPEC: the offer does not solicit a listing the push already rendered"),
                ("spec_budget_matches_fuse",
                 "SPEC: the budget the spec states is the budget the loop enforces"),
                ("scope_engages_on_main_call", "SCOPING: the loop engages on the planning call"),
                ("scope_skips_other_call",
                 "SCOPING: the loop does NOT wrap the stage-boundary/decision calls"),
                ("scope_skips_without_context", "SCOPING: no loop without a live context"),
            ):
                check(bool(live.get(key)), f"{arm}: {label}")
            notes.append(
                f"{arm}: frames served on a cold address = "
                f"{live.get('gain_probe_first_query_frames')}; re-serving an on-context address = "
                f"{live.get('gain_off_context_frames')} (must be 0)"
            )

    # ---- check 9: arm-order independence ---------------------------------------------------
    # The runner sources one arm per evaluation IN THE SAME SHELL, so each arm inherits whatever
    # the previous one exported. A memory flag that survives into the baseline turns the control
    # into a treatment, and the resulting number looks like a positive memory effect. This is
    # not hypothetical: `MEMEXP_PULL_ENABLE` has no underscore after `MEM`, so the baseline's
    # `^MEM_` purge did not remove it, and `pullmem` followed by `nomem` left the baseline
    # Planner HOOKED. It is checked as a sequence rather than per arm because no single-arm
    # check can see it.
    print("\n[CHECK 9] arm-order independence (arms run in one shell, in sequence)")
    standalone = {a: env_by_arm[a] for a in all_arms}
    order_problems = []
    for first in all_arms:
        for second in all_arms:
            if first == second:
                continue
            try:
                seq = dump_arm_env(second, after=[first])
            except Exception as exc:
                order_problems.append(f"{first}->{second}: source failed: {exc!r}")
                continue
            diff = {
                k for k in set(seq) | set(standalone[second])
                if seq.get(k, "<absent>") != standalone[second].get(k, "<absent>")
            }
            # Only keys the arms themselves manage matter; a shell's own bookkeeping does not.
            diff = {k for k in diff if not k.startswith("_")}
            if diff:
                order_problems.append(
                    f"{first}->{second}: differs in {sorted(diff)} "
                    f"(e.g. {second} alone {standalone[second].get(sorted(diff)[0], '<absent>')!r} "
                    f"vs after {first} {seq.get(sorted(diff)[0], '<absent>')!r})"
                )
    check(not order_problems,
          f"every arm is unaffected by which arm ran before it "
          f"({len(all_arms) * (len(all_arms) - 1)} ordered pairs)",
          "\n".join(order_problems[:6]))
    if order_problems:
        failures.append("CHECK 9 arm order")

    # The specific, load-bearing instance of the above: the baseline must expose NO memory
    # channel and must not leave this experiment's hook reachable, whatever ran before it.
    #
    # Scoped to the keys that ACTIVATE the hook, not to every `MEMEXP_*` key. The baseline sets
    # `MEMEXP_ARM_NAME` / `MEMEXP_ARM_CLASS` itself, so "any MEMEXP_ key present" would flag the
    # baseline for correctly identifying itself, and a check that fires on the right answer is
    # one that gets deleted rather than fixed. The generic order-independence test above already
    # covers value drift for every key; this one asserts the semantic claim that no hook can fire.
    ACTIVATING = ("MEMEXP_PULL_ENABLE", "MEMEXP_PULL_TOOLS")
    for first in all_arms:
        if first == BASELINE:
            continue
        try:
            seq = dump_arm_env(BASELINE, after=[first])
        except Exception:
            continue
        armed = {k: seq[k] for k in ACTIVATING
                 if str(seq.get(k, "")).strip().lower() in {"1", "true", "yes", "on", "y", "t"}}
        pysite = any("mem_efficacy" in p for p in str(seq.get("PYTHONPATH", "")).split(":"))
        self_id = seq.get("MEMEXP_ARM_NAME", "<absent>")
        check(not armed and not pysite and self_id == standalone[BASELINE].get("MEMEXP_ARM_NAME"),
              f"baseline stays memory-free after {first} "
              f"(hook disarmed, sitecustomize off PYTHONPATH, self-ID intact)",
              f"armed={armed} PYTHONPATH={seq.get('PYTHONPATH', '<absent>')!r} "
                        f"ARM_NAME={self_id!r} (expected "
              f"{standalone[BASELINE].get('MEMEXP_ARM_NAME')!r})")

    # ---- check 10: the runner's own gates cannot silently skip -----------------------------
    # A gate that reports SKIPPED is a gate that did not run, and this experiment has already
    # been bitten by one: the GATE 0b membership test used the comma-separated idiom
    # (`",${ARMS_TO_RUN}," == *",pullmem,"*`) against a SPACE-separated list, so it matched only
    # when `pullmem` happened to be first. Job 590795 ran all three arms with the tool loop
    # never exercised in the job, and the skip message contradicted itself. Since the failure is
    # silent by construction, it is checked by pattern rather than trusted.
    print("\n[CHECK 10] the runner's own gates cannot silently skip")
    sbatch = (HERE / "run_26x1.sbatch").read_text(encoding="utf-8")
    # Comment lines are excluded: this file documents the broken idiom on purpose, and a lint
    # that flags its own explanation is a lint that gets deleted.
    code_lines = [ln.strip() for ln in sbatch.splitlines()
                  if not ln.strip().startswith("#")]
    bad = [ln for ln in code_lines if "ARMS_TO_RUN" in ln and "," in ln and "==" in ln]
    check(not bad, "no comma-separated membership test against the space-separated ARM list",
          "\n".join(bad))
    check('" ${ARMS_TO_RUN} " == *" pullmem "*' in sbatch,
          "the pullmem gate uses the space-padded membership form")
    for gate in ("[GATE 0]", "[GATE 0b]", "[GATE 0c]", "[GATE 1]"):
        check(gate in sbatch, f"the runner still contains {gate}")
    check(sbatch.count("[GATE 0b] SKIPPED") == 1,
          "there is exactly one GATE 0b skip path")
    # GATE 0c must exist AND be reachable, for the same reason GATE 0b had to be: it is the only
    # thing that executes the memory correction rather than merely noting that it installed. A
    # gate whose membership test never matches is a gate that reports success forever.
    check(sbatch.count("[GATE 0c] SKIPPED") == 1,
          "there is exactly one GATE 0c skip path")
    check('" ${ARMS_TO_RUN} " == *" pushmem "* || " ${ARMS_TO_RUN} " == *" pullmem "*' in sbatch,
          "the GATE 0c membership test uses the space-padded form for both corrected arms",
          "GATE 0c would be skipped for an arm that needs it")
    check("selftest_memfix.py" in sbatch,
          "GATE 0c actually invokes selftest_memfix.py")

    # ---- check 11: the frozen-env artifact cannot silently drop keys ------------------------
    # `resolved_env.txt` is described by its own writer as the environment that actually ran,
    # "as an artifact rather than a recollection". It is built as
    #     env | grep -E '^(HARNESS_|PLANNER_|...|OUT_ROOT)=' | sort
    # and the `=` sits OUTSIDE the alternation group, so the pattern means "a name that is
    # EXACTLY `HARNESS_` followed by `=`" -- which no real variable is. Measured on job 590799:
    # of 6 synthetic lines only `PROACTIVE_MODE=off` and `D_MERGE=6` survived; every prefixed key
    # (`HARNESS_VLM_CONTEXT`, `VLM_USE_KEYFRAME_MEMORY`, `MEM_STAGE_ANCHOR`, `PLANNER_API_MODEL`)
    # was dropped. The artifact therefore under-reports precisely the memory switches it exists to
    # record, which is the same class of failure as the GATE 0b bug: a check that reports success
    # while having looked at nothing.
    #
    # Checked by execution, not by inspection: the pattern is extracted from the sbatch and run
    # against synthetic lines, including the keys each arm declares as its own diff.
    print("\n[CHECK 11] the frozen-env pattern captures prefixed keys")
    pat = None
    for ln in code_lines:
        if "env | grep -E" in ln:
            a, _, rest = ln.partition("'")
            pat, _, _ = rest.partition("'")
            break
    if pat is None:
        check(False, "the frozen-env dump line is present in the runner")
    else:
        probes = ["HARNESS_VLM_CONTEXT=1", "VLM_USE_KEYFRAME_MEMORY=1", "MEM_STAGE_ANCHOR=1",
                  "PLANNER_API_MODEL=gemini-3.8-flash", "PORT=8026", "SEED=100",
                  "MEMEXP_PULL_ENABLE=1", "OUT_ROOT=/tmp/x", "D_MERGE=6"]
        for a in all_arms:
            for k in str(standalone[a].get("MEMEXP_EXPECTED_DIFF", "")).split():
                if k and f"{k}=" not in probes:
                    probes.append(f"{k}=1")
        missed = [p for p in probes if not re.search(pat, p)]
        check(not missed,
              f"the frozen-env pattern matches all {len(probes)} representative keys, "
              f"including every key the arms declare as their diff",
              "missed: " + ", ".join(missed))

    # ---- check 12: the memory correction is LIVE in the arms that need it, and absent elsewhere
    #
    # This is the check that would have caught the first version of `memexp_memfix`. That version
    # imported the harness eagerly, which cannot work from sitecustomize (the harness directory is
    # not on sys.path yet), so it raised ModuleNotFoundError, the exception was swallowed, and the
    # correction was configured, documented, switched on -- and completely inert. Every other
    # check passed. A fix that never engages is indistinguishable from a fix that does not help,
    # so "was it installed" has to be asserted at runtime and not inferred from the arm's flags.
    print("\n[CHECK 12] the memory correction is live where declared, and provably absent elsewhere")
    MEMFIX_PROBE = r'''
import os, sys, json
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark"))
sys.path.insert(0, os.path.join(os.environ["ROOT"], "evaluation_benchmark", "openpi_minimal_runtime"))
res = {}
try:
    import harness.controller as c
    import harness.memory_reason as mr
    res["renderer"] = mr.memory_reason_for_planner.__name__
    res["call_site"] = c.memory_reason_for_planner.__name__
    res["refresher_patched"] = bool(
        getattr(c.HarnessController._refresh_vlm_context, "_memexp_memfix", False))
    res["loaded"] = "memexp_memfix" in sys.modules
    res["error"] = None
except Exception as exc:
    import traceback
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()[-800:]
print("@@MEMFIX@@" + json.dumps(res))
'''
    probe_res = _run_probe_in_every_arm(MEMFIX_PROBE, "@@MEMFIX@@")
    want_memfix = {
        a for a in all_arms
        if str(standalone[a].get("MEMEXP_MEMFIX_ENABLE", "")).strip() in {"1", "true", "yes", "on"}
    }
    if not want_memfix:
        check(False, "no arm declares MEMEXP_MEMFIX_ENABLE, so the correction would never run")
    for a in all_arms:
        r = probe_res.get(a) or {}
        if r.get("error"):
            check(False, f"{a}: memfix probe failed", str(r["error"]))
            continue
        if a in want_memfix:
            check(
                r.get("loaded") is True
                and r.get("refresher_patched") is True
                and r.get("renderer") == "render_memory"
                and r.get("call_site") == "render_memory",
                f"{a}: correction LIVE (renderer={r.get('renderer')}, "
                f"call_site={r.get('call_site')}, refresher_patched={r.get('refresher_patched')})",
                f"probe reported: {r}",
            )
        else:
            # The baseline and any memory-free treatment must be untouched, not merely "probably
            # unaffected": the module must not even be in the interpreter.
            check(
                r.get("loaded") is False
                and r.get("refresher_patched") is False
                and r.get("renderer") == "memory_reason_for_planner",
                f"{a}: correction absent (renderer={r.get('renderer')}, loaded={r.get('loaded')})",
                f"probe reported: {r}",
            )
    notes.append(f"memory correction live in {len(want_memfix)}/{len(all_arms)} arm(s): "
                 f"{sorted(want_memfix) or 'none'}")

    # ---- check 13: the run can be reproduced from its own artifacts -------------------------
    #
    # The arms REPLACE parts of the harness at runtime, so a score is a property of (this
    # experiment's files) x (the upstream revision), and the second factor is recorded nowhere
    # else. This check asserts that the recorder runs and that it hashes the files the arms
    # actually patch: if, say, `harness/memory_reason.py` were dropped from its list, an upstream
    # edit to the very file this experiment overwrites would change an archived score's meaning
    # with nothing to show for it.
    #
    # The third assertion is the one that earns its place immediately: it runs the collector and
    # requires every listed path to EXIST. Writing the list by hand is error-prone -- the first
    # version named `harness/keyframe_selection.py`, which does not exist, and a silently missing
    # entry is exactly the kind of gap this file is supposed to close.
    print("\n[CHECK 13] the run records enough to be reproduced")
    sbatch13 = (HERE / "run_26x1.sbatch").read_text(encoding="utf-8")
    check("record_provenance.py" in sbatch13,
          "the runner invokes record_provenance.py")
    prov_src = (HERE / "record_provenance.py").read_text(encoding="utf-8")
    must_hash = {
        "evaluation_benchmark/harness/memory_reason.py",   # the renderer the correction replaces
        "evaluation_benchmark/harness/controller.py",      # the refresher the correction replaces
        "evaluation_benchmark/harness/api_vlm_planner.py",  # the loop the PULL arm wraps
        "experiments/mem_efficacy/memexp_memfix.py",       # the correction itself
        "experiments/mem_efficacy/memexp_bind.py",         # the PULL binding
    }
    absent = sorted(k for k in must_hash if k not in prov_src)
    check(not absent,
          f"the recorder hashes all {len(must_hash)} runtime-patched files",
          "not hashed: " + ", ".join(absent))
    try:
        sys.path.insert(0, str(HERE))
        import record_provenance as _rp  # noqa: PLC0415
        collected = _rp.collect(str(ROOT))
        missing = collected.get("missing_files") or []
        check(not missing,
              f"every one of the {len(collected.get('file_sha256') or {})} listed files exists, "
              "so the fingerprint covers what it claims",
              "missing (a silently absent entry leaves a file unfingerprinted): "
              + ", ".join(missing))
    except Exception as exc:  # noqa: BLE001
        check(False, "the provenance collector runs", f"{type(exc).__name__}: {exc}")

    # ---- check 14: the in-process reports are LIVE, not frozen at install ----------------
    # This check is the reason job 591479's numbers could not be read. Its artifacts contained a
    # `memexp_pull_report.json` reading `steps: 0, tool_calls: 0, substrate: null, tools: null`,
    # and the census correctly reported "the Planner NEVER called a memory tool". Every gate
    # passed, because every gate asked "are the hooks installed?" and they were. The file was
    # written once inside `_bind()` and never again, so every runtime counter in it was zero by
    # construction. An artifact that cannot report turns "unknown" into a confident "zero", which
    # is worse than no artifact at all -- so the property to test is not "installed" but
    # "does activity reach disk".
    print("\n[CHECK 14] the in-process counters are updated DURING a run, not frozen at install")
    bind_src = (HERE / "memexp_bind.py").read_text(encoding="utf-8")
    memfix_src = (HERE / "memexp_memfix.py").read_text(encoding="utf-8")
    census_src = (HERE / "census_channels.py").read_text(encoding="utf-8")

    def write_fn_src(src: str) -> str:
        m = re.search(r"\ndef _write_report\(.*?(?=\n\ndef |\n\nclass )", src, re.S)
        return m.group(0) if m else ""

    sites = len(re.findall(r"^\s*_write_report\(\)\s*$", bind_src, re.M))
    check(
        sites >= 3,
        "memexp_bind flushes the report from every planning step and from the tool loop",
        f"call sites = {sites} (need >= 3: one in _bind, one per step, one per tool call)\n"
        "A single call site means the file is an install-time snapshot with every runtime\n"
        "counter pinned at zero -- the exact defect that made 591479 unreadable.",
    )
    for name, src in (("memexp_bind", bind_src), ("memexp_memfix", memfix_src)):
        body = write_fn_src(src)
        check(
            bool(body) and "os.getpid()" in body,
            f"{name} writes a per-process report file",
            "Several interpreters share one output directory (task1 and tasks2to26 are separate\n"
            "processes, and merge_eval_outputs.sh runs a python3 after the evaluation). With one\n"
            "shared path each truncates the previous one, so only the last writer survives.",
        )
        check(
            not re.search(r"open\(\s*path\s*,\s*[\"']w", body or "open(path,'w'"),
            f"{name} does not truncate a shared path",
            "a direct write to `path` is the clobbering bug itself",
        )
    check(
        "memexp_pull_report.json.*" in census_src and "memexp_memfix_report.json.*" in census_src,
        "the census merges the per-process reports",
        "globbing only the bare filename would find nothing once reports are per-process",
    )
    runner_src = (ROOT / "experiments" / "mem_efficacy" / "run_26x1.sbatch").read_text(encoding="utf-8")
    check(
        bool(re.search(r"rm -f .*memexp_pull_report\.json\*", runner_src))
        and bool(re.search(r"rm -f .*memexp_memfix_report\.json\*", runner_src)),
        "the runner clears stale counters before each arm",
        "This job is submitted with --requeue and a requeued attempt re-runs an arm into the same\n"
        "directory. Per-process counters are SUMMED, so a leftover file from a previous attempt\n"
        "would be added to this one, and an inflated tool-call count is worse than a missing one.",
    )

    probe_src = r"""
import glob, json, os, sys, tempfile

# Mirror the evaluator: its cwd is the eval directory, so this experiment's modules are reachable
# only through PYTHONPATH (which sitecustomize extends). Leaving cwd on sys.path would make every
# module importable from here and turn the absence checks into tautologies -- it did, on the first
# attempt, and reported three false failures.
_here = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", _here)]
out = {}
tmp = tempfile.mkdtemp()


def probe(modname, env_key, counter_key, active_key):
    path = os.path.join(tmp, modname + ".json")
    os.environ[env_key] = path
    try:
        mod = __import__(modname)
    except Exception as exc:
        return {"importable": False, "error": repr(exc)}
    state = getattr(mod, "_STATE", {}) or {}
    # "Active" is the layer deciding for itself that it should run, read from the module rather
    # than from the arm file -- the arm file is what we are trying to hold to account. It is read
    # WITHOUT calling `install()` first: calling it here would arm the layer inside a process the
    # arm never armed, and the absence check would then fail on its own side effect.
    if active_key == "__pull_enabled__":
        active = bool(mod.pull_enabled())
    else:
        active = bool(state.get("hook_installed")) or bool(state.get(active_key))
    res = {"importable": True, "active": active}
    if counter_key not in state:
        res["counter"] = False
        return res
    state[counter_key] = 12345
    try:
        mod._write_report()
    except Exception as exc:
        res.update({"counter": True, "wrote": False, "error": repr(exc)})
        return res
    files = [f for f in glob.glob(path + "*") if not f.endswith(".tmp")]
    value = False
    for f in files:
        try:
            with open(f) as fh:
                if json.load(fh).get(counter_key) == 12345:
                    value = True
        except Exception:
            pass
    res.update({"counter": True, "wrote": True, "value_round_trips": value, "n_files": len(files),
                "per_pid": any(f.endswith("." + str(os.getpid())) for f in files)})

    # A write that lands outside a plan step (ctx is None) must not ERASE the live-object
    # snapshots. They are the only source for `by_tool`, `n_query` and frames served, and every
    # process ends with such a write. Measured on job 592860: `substrate` and `tools` were null in
    # all 20 reports, so the census reported "called 5 tools and served 0 frames" out of missing
    # data rather than measurement.
    if env_key == "MEMEXP_PULL_REPORT":
        try:
            with open(os.path.join(tmp, modname + ".json." + str(os.getpid()))) as fh:
                seed_payload = json.load(fh)
            seed_payload["substrate"] = {"n_query": 7, "addresses": 3}
            seed_payload["tools"] = {"by_tool": {"query_world": 7}}
            with open(os.path.join(tmp, modname + ".json." + str(os.getpid())), "w") as fh:
                json.dump(seed_payload, fh)
            state[counter_key] = 999          # a later write, still with ctx == None
            mod._write_report()
            with open(os.path.join(tmp, modname + ".json." + str(os.getpid()))) as fh:
                after = json.load(fh)
            res["transient_survives_null_write"] = (
                (after.get("tools") or {}).get("by_tool", {}).get("query_world") == 7
                and (after.get("substrate") or {}).get("n_query") == 7
                and after.get(counter_key) == 999
            )
        except Exception as exc:
            res["transient_survives_null_write"] = False
            res["transient_error"] = repr(exc)
    return res


out["bind"] = probe("memexp_bind", "MEMEXP_PULL_REPORT", "steps", "__pull_enabled__")
out["memfix"] = probe("memexp_memfix", "MEMEXP_MEMFIX_REPORT", "n_render", "installed")
print("MEMEXP_LIVE " + json.dumps(out))
"""
    live = _run_probe_in_every_arm(probe_src, "MEMEXP_LIVE ")
    for arm in all_arms:
        res = live.get(arm) or {"error": "no result"}
        env = env_by_arm.get(arm, {})
        wants_bind = _truthy(env.get("MEMEXP_PULL_ENABLE")) and _truthy(env.get("MEMEXP_PULL_TOOLS"))
        wants_memfix = _truthy(env.get("MEMEXP_MEMFIX_ENABLE"))
        for label, key, wanted in (("bind", "bind", wants_bind), ("memfix", "memfix", wants_memfix)):
            r = res.get(key) if isinstance(res, dict) else None
            if not isinstance(r, dict):
                check(False, f"[{arm}] the {label} probe returned a result", f"probe = {r}")
                continue
            if not r.get("importable"):
                # Absence must be demonstrated, not assumed: an arm that silently loads the layer
                # it is supposed to lack is the contamination bug CHECK 9 exists for.
                check(not wanted, f"[{arm}] the {label} layer is absent, as declared",
                      f"probe = {r}")
            elif not wanted:
                check(not r.get("active"),
                      f"[{arm}] the {label} layer is importable but INACTIVE, as declared",
                      f"probe = {r}")
            elif not r.get("wrote"):
                check(False, f"[{arm}] {label} writes its report on demand", f"probe = {r}")
            else:
                check(
                    bool(r.get("per_pid")) and bool(r.get("value_round_trips")),
                    f"[{arm}] {label} counters reach disk and are per-process",
                    f"probe = {r}",
                )
                if label == "bind":
                    check(
                        bool(r.get("transient_survives_null_write")),
                        f"[{arm}] a write outside a plan step does not erase the tool/substrate stats",
                        f"probe = {r}\nThese two keys are the only source for `by_tool`, `n_query` and\n"
                        "frames served, and every process ends with such a write -- so without this\n"
                        "the pull's effect is unreadable no matter how many tool calls happened.",
                    )

    # ---- check 15: WHICH tool registry each arm runs -------------------------------------
    # CHECK 1's structural comparison drops every `MEMEXP_` key by convention, so the registry
    # selection is invisible to it. That is fine for the shared knobs but not for this one: the
    # registry decides WHICH tool design the arm tests, and an arm that silently fell back to the
    # default (or to a module that does not exist) would produce a score attributed to the wrong
    # mechanism. That is the failure the binding now refuses at install time, and this check is what
    # makes it visible BEFORE the run rather than in a report afterwards.
    print("\n[CHECK 15] exactly the pull arm selects a tool registry, and it is usable")
    try:
        reg_by_arm: dict[str, str] = {
            arm: str(env_by_arm[arm].get("MEMEXP_TOOLS_MODULE", "")).strip()
            for arm in [BASELINE, *treatments]
        }
        selected = [a for a in treatments if reg_by_arm.get(a)]
        check(len(selected) == 1 and "pullmem" in selected,
              f"only pullmem selects a registry (selected by: {selected or 'nobody'})",
              "Every arm that runs a tool loop must name its registry, and no other arm should\n"
              "have one: a baseline carrying a registry name would load a tool module it never\n"
              "calls, and the push-only control would stop being a control.")
        check(not reg_by_arm.get(BASELINE),
              f"{BASELINE} selects no registry (got {reg_by_arm.get(BASELINE)!r})",
              "the baseline must not carry this experiment's tool plumbing at all")

        for arm, name in reg_by_arm.items():
            if not name:
                continue
            probe = _run_probe(
                f"""
import json
out = {{"name": {name!r}}}
try:
    mod = __import__({name!r})
    out["importable"] = True
    out["exports_memory_tools"] = hasattr(mod, "MemoryTools")
    out["exports_parse_tool_call"] = hasattr(mod, "parse_tool_call")
    out["exports_force_final"] = bool(getattr(mod, "FORCE_FINAL", ""))
    inst = mod.MemoryTools(__import__("memexp_substrate").Substrate())
    out["contract_aware"] = bool(getattr(inst, "CONTRACT_AWARE", False))
    out["tools"] = sorted(inst.tools)
    out["has_search_tool"] = "search_memory" in inst.tools
except Exception as exc:
    out["importable"] = False
    out["error"] = repr(exc)
print("REGISTRY_PROBE" + json.dumps(out))
""",
                "REGISTRY_PROBE",
                env_by_arm[arm],
            )
            check(bool(probe.get("importable")),
                  f"[{arm}] registry {name!r} is importable in the arm's own environment",
                  f"probe = {probe}")
            if not probe.get("importable"):
                continue
            check(bool(probe.get("exports_memory_tools"))
                  and bool(probe.get("exports_parse_tool_call")),
                  f"[{arm}] registry {name!r} is a drop-in (MemoryTools + parse_tool_call)",
                  f"probe = {probe}")
            check(bool(probe.get("contract_aware")) and bool(probe.get("exports_force_final")),
                  f"[{arm}] registry {name!r} carries the output contract",
                  f"probe = {probe}\nA registry without these would teach the model a finishing\n"
                  "form the planner's parser reads as an empty primitive (no exception is raised,\n"
                  "so the step silently yields no action), or clamp it to one that does.")
            if arm == "pullmem":
                check(bool(probe.get("has_search_tool")),
                      f"[{arm}] its registry can search without the exact address",
                      f"tools = {probe.get('tools')}\n`query_world` takes an ADDRESS the model must\n"
                      "copy out of the pushed block, which inverts retrieval: a model that can name\n"
                      "the address can usually already see what it refers to. Measured on the\n"
                      "archived run: 5 calls, every one on an address that had just been advertised.")
    except Exception as exc:  # noqa: BLE001
        check(False, "the registry check runs", f"{type(exc).__name__}: {exc}")

    # ---- check 16: the stage driver, proven on a real HarnessController --------------------
    #
    # Job 593817 ran the PULL arm with a task-spec seed that minted 12 addresses per episode and
    # opened NONE of them: `n_spec_minted = 12`, `n_spec_unopened = 12`, `n_stage_active = 0`. The
    # driver polled `planner.episodic_store.current_stage_name`, and that object is None in this arm
    # (`create_store_if_needed` returns None unless proactive memory is enabled, and the arm runs
    # `PROACTIVE_MODE=off`). So the arm pushed twelve permanently unreadable addresses on every
    # step -- prompt pollution with zero retrievable benefit, which is worse than no seed at all.
    #
    # The assertion therefore targets the DEFECT rather than the repair: `_current_stage` must
    # return a stage for a planner whose `episodic_store` is None. A check that only proved "the
    # hook installed" would have passed on the broken version too, since the hook was never the
    # missing piece -- the SOURCE was.
    print("\n[CHECK 16] the active stage comes from the evaluator, not from `episodic_store`")
    try:
        if "pullmem" in all_arms:
            sp = run_sdv_stage_probe(env_by_arm["pullmem"])
            if sp.get("error"):
                check(False, "the stage-driver probe runs", sp.get("traceback", sp["error"]))
            else:
                check(bool(sp.get("hook_installed")),
                      "the harness stage hook is installed",
                      f"hook_error={sp.get('hook_error')!r}")
                check(sp.get("stage_without_store") == "02_Beta",
                      "with `episodic_store = None` the driver still returns the live stage",
                      f"_current_stage -> {sp.get('stage_without_store')!r}, expected '02_Beta'. "
                      f"The store was the OLD source and is None in this arm, which is why all 12 "
                      f"seeded addresses stayed unopened in job 593817.")
                check(str(sp.get("active_stage_state") or "") == "02_Beta"
                      and int(sp.get("stage_hook_calls") or 0) >= 1,
                      "the hook records the stage it was called with",
                      f"active_stage={sp.get('active_stage_state')!r}, "
                      f"calls={sp.get('stage_hook_calls')}, seen={sp.get('stage_names_seen')}")
                check(sp.get("stage_from_store") == "09_From_Store",
                      "a present `episodic_store` is still honoured as a fallback",
                      f"_current_stage -> {sp.get('stage_from_store')!r}")
                check(sp.get("empty_when_nothing") == "",
                      "with no source at all the stage is empty (so the census reports it "
                      "unopened rather than opening the wrong window)",
                      f"_current_stage -> {sp.get('empty_when_nothing')!r}")
        else:
            notes.append("CHECK 16 skipped: no pullmem arm in scope")
    except Exception as exc:  # noqa: BLE001
        check(False, "the stage-driver check runs", f"{type(exc).__name__}: {exc}")

    # ---- check 17: the keyframe spread is additive, bounded and a genuine spread -------------
    #
    # Job 593817 measured the pushed bank at 1.7 frames per call against a `bank_max` of 8: the
    # anchor-only builder is CANDIDATE-limited, and no hyperparameter of it can widen the candidate
    # set. `_kf_spread_union` adds a stride over the frame store, which is a function of nothing the
    # Planner wrote -- the property that makes it available to a Planner that nominates nothing.
    print("\n[CHECK 17] the keyframe spread only ADDS, respects the recent window, and spans the episode")
    try:
        if "pushmem" in all_arms:
            kp = run_kf_spread_probe(env_by_arm["pushmem"])
            if kp.get("error"):
                check(False, "the kf-spread probe runs", kp.get("traceback", kp["error"]))
            else:
                check(int(kp.get("cfg_spread") or 0) > 0
                      and str(kp.get("declared_spread")) == str(kp.get("cfg_spread")),
                      "pushmem declares MEM_KF_SPREAD and the config reads it back",
                      f"env={kp.get('declared_spread')!r}, cfg={kp.get('cfg_spread')}")
                check(bool(kp.get("kept_anchors")),
                      "every anchor the official builder selected survives the union",
                      f"anchors_only={kp.get('anchors_only')} -> union={kp.get('union')}. An "
                      f"additive source that could DISPLACE a stage anchor would make the two "
                      f"mechanisms indistinguishable in the result.")
                check(bool(kp.get("all_before_recent")),
                      "no spread frame lands inside the Planner's own recent window",
                      f"union={kp.get('union')}, recent_start excluded")
                check(int(kp.get("union_len") or 0) > int(kp.get("anchors_only") or 0),
                      "the union actually grows the bank (it is not a no-op)",
                      f"{kp.get('anchors_only')} -> {kp.get('union_len')} frames")
                check(int(kp.get("span") or 0) > 100,
                      "the added frames SPAN the episode rather than clumping at one end",
                      f"span={kp.get('span')} over union={kp.get('union')}. A naive `[-want:]` "
                      f"would take the frames nearest the Planner's context, i.e. the least new "
                      f"information.")
                check(bool(kp.get("unique")),
                      "the union contains no duplicate frames",
                      f"union={kp.get('union')}")
                check(not kp.get("early_picks"),
                      "the spread picks NO frame from the episode's opening",
                      f"early picks = {kp.get('early_picks')} out of union={kp.get('union')}. "
                      f"An endpoint-inclusive stride always takes the pool's OLDEST frame, and the "
                      f"official builder has already anchored frame 0 as a salient step-0 event -- "
                      f"so the pick is frame 1, a near-duplicate, and a bank slot is spent on the "
                      f"episode's opening on every step.")
                check(bool(kp.get("zero_budget_identity")),
                      "a zero budget (a full bank) returns the bank unchanged",
                      f"the call site passes `cap - len(anchors)`, which is 0 whenever the bank is "
                      f"full -- the common case. The first version reached this by dividing by zero "
                      f"and catching its own exception once per plan step.")
                check(bool(kp.get("empty_store_ok")),
                      "an empty frame store leaves the bank unchanged instead of raising",
                      f"union on empty store != input")
        else:
            notes.append("CHECK 17 skipped: no pushmem arm in scope")
    except Exception as exc:  # noqa: BLE001
        check(False, "the kf-spread check runs", f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------------------------------
    # CHECK 18 -- the PrediMem nomination policy is present, GATED, and declared.
    #
    # Two separate things can go wrong here and they fail in opposite directions, which is why
    # both halves are asserted:
    #   (a) the policy block is emitted UNCONDITIONALLY -> the baseline's prompt changes and every
    #       memory delta is measured against a moved control. Silent, and it would look like the
    #       memory arms got worse.
    #   (b) the block is gated but no arm declares the knob -> the channel is off everywhere and
    #       the "PrediMem-style" arm is really the anchor+stride fallback while carrying the name.
    #
    # (a) is checked on the SOURCE with `ast` rather than by building a Planner: `_build_messages`
    # needs a fully constructed `ApiMemoryPlanner` (tens of arguments plus live frame stores), and
    # a probe that had to stand one up would be testing the harness, not the gate. The structural
    # assertion is narrow on purpose -- "an `if` whose test reads `nomination_prompt` guards the
    # append of `KF_NOMINATION_POLICY`" -- so it cannot pass by accident.
    print("\n[CHECK 18] the nomination policy exists, is GATED, and is declared by an arm")
    try:
        import ast

        planner_src = (ROOT / "evaluation_benchmark/harness/api_vlm_planner.py").read_text(
            encoding="utf-8", errors="replace"
        )
        tree = ast.parse(planner_src)

        policy = ""
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "KF_NOMINATION_POLICY" for t in node.targets
            ):
                policy = ast.literal_eval(node.value)
        check(len(policy) > 200 and "keyframe_positions" in policy,
              "`KF_NOMINATION_POLICY` is a module constant that names the field to fill",
              f"len={len(policy)}; GATE 0 imports this exact string, so a copy would let the gate "
              f"pass while testing a prompt no arm sends")

        # (a) gated, and gated on the CONFIG value rather than on some other knob.
        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.unparse(node.test)
            body_src = " ".join(ast.unparse(s) for s in node.body)
            if "nomination_prompt" in test_src and "KF_NOMINATION_POLICY" in body_src:
                guarded = True
        check(guarded,
              "the policy block is emitted ONLY under `nomination_prompt`",
              "without the guard the block reaches the BASELINE, which moves the control every "
              "memory delta is measured against -- and it would read as the memory arms degrading")

        # (b) declared by an arm in scope, and absent from the baseline.
        #      Runtime, in the arms' own environments. Reading the default by importing
        #      `memory_system` into THIS process would drag in the whole harness (`keyframe_bank`
        #      -> `keyframe_selection`, which lives under `openpi_minimal_runtime`, not under
        #      `harness`) and the validator would then be asserting against a path it guessed.
        #      `_run_probe` uses the same two roots the other checks use, and measures the value
        #      the arm will actually see.
        nom_cfg = _run_probe(NOM_CFG_PROBE_SRC, "@@NOMCFG@@", dump_arm_env(BASELINE))
        check(not nom_cfg.get("error"),
              "the nomination config reads back in the baseline's environment",
              f"{nom_cfg.get('error')}")
        check(nom_cfg.get("default") is False,
              "`nomination_prompt` defaults to False",
              f"default={nom_cfg.get('default')}; a default of True would turn the channel on for "
              f"every arm that does not mention it")

        arms_all = discover_arms()[1]
        declaring = [
            a for a in arms_all if _truthy(dump_arm_env(a).get("MEM_KF_NOMINATION_PROMPT"))
        ]
        check(not _truthy(dump_arm_env(BASELINE).get("MEM_KF_NOMINATION_PROMPT"))
              and nom_cfg.get("from_env") is False,
              f"the baseline arm ({BASELINE}) does not teach the nomination policy",
              "the control's prompt must be the official one, byte for byte")
        check("pushmem" in arms_all,
              "a pushmem arm exists to carry the policy", f"arms={arms_all}")
        for _mem_arm in ("pushmem", "pullmem"):
            if _mem_arm not in arms_all:
                notes.append(f"CHECK 18: no {_mem_arm} arm in scope; declaration half skipped")
                continue
            check(_mem_arm in declaring,
                  f"the {_mem_arm} arm declares MEM_KF_NOMINATION_PROMPT",
                  f"declaring arms = {declaring}; without it the arm's bank is the anchor+stride "
                  f"fallback and the PrediMem claim is empty")
            _pm = _run_probe(NOM_CFG_PROBE_SRC, "@@NOMCFG@@", dump_arm_env(_mem_arm))
            check(_pm.get("from_env") is True,
                  f"{_mem_arm}'s environment resolves the nomination policy to ON",
                  f"from_env={_pm.get('from_env')} err={_pm.get('error')}; the export being "
                  f"present in the file is not the same as the config reading it")
            check("MEM_KF_NOMINATION_PROMPT" in str(dump_arm_env(_mem_arm).get(
                      "MEMEXP_EXPECTED_DIFF", "")),
                  f"{_mem_arm}'s declared diff lists MEM_KF_NOMINATION_PROMPT",
                  "an undeclared difference is exactly what CHECK 1 cannot see")

        # GATE 0 derives the probe from the arm FILES, so a renamed or re-quoted export would make
        # it silently skip. Assert the runner still greps for the key it now has to find.
        runner_src = (ROOT / "experiments/mem_efficacy/run_26x1.sbatch").read_text(
            encoding="utf-8", errors="replace"
        )
        check("GATE0_NOM_PROBE" in runner_src and "MEM_KF_NOMINATION_PROMPT" in runner_src,
              "GATE 0 derives the nomination probe from the arm files",
              "the probe runs before the arms are sourced, so reading the knob from the "
              "environment would report SKIPPED while looking like it had run")
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb
        check(False, "the nomination-policy check runs",
              f"{type(exc).__name__}: {exc}\n         " + _tb.format_exc().replace("\n", "\n         ")[-600:])

    # ---------------------------------------------------------------------------------------
    # CHECK 19 -- the keyframe channel's candidate POOL is a record of the episode, not of the
    # async call pattern. See `KF_STORE_PROBE_SRC` for the full rationale and the measurement.
    #
    # Three separate failure modes, asserted separately because they fail in different directions:
    #   (a) the eval hook is ABSENT or UNGATED -> either nothing is stored (the defect) or frames
    #       are stored even for the baseline (a moved control);
    #   (b) an arm that builds a keyframe bank does not declare an interval -> the bank is built
    #       over the call-window store, i.e. the arm's own mechanism is inert before it runs;
    #   (c) the spread over a call-window store looks like one over a dense store -> then this check
    #       is not measuring the thing it claims to, and must fail rather than pass.
    print("\n[CHECK 19] the keyframe bank's candidate pool is a real record of the episode")
    try:
        eval_src = (ROOT / "evaluation_benchmark" / "async_vlm26_reference"
                    / "eval_fullvlm26_async_vlm_vla.py").read_text(
                        encoding="utf-8", errors="replace")
        check("MEM_KF_STORE_INTERVAL" in eval_src and "def store_kf_dense" in eval_src,
              "the eval loop has a frame store hook driven by MEM_KF_STORE_INTERVAL",
              "without it the planner's frame store is filled only from the context windows of "
              "planner calls that survived async queue eviction -- ~1.4% of the episode")
        # The hook must be a no-op at its default. That is what keeps the baseline bit-identical.
        check('os.environ.get("MEM_KF_STORE_INTERVAL", "0")' in eval_src
              and "kf_store_interval <= 0" in eval_src,
              "the hook defaults to 0 and returns before storing anything",
              "an always-on store would move the baseline")

        kf_arms = [a for a in all_arms if a != "nomem"]
        if kf_arms:
            st = run_kf_store_probe(env_by_arm.get(kf_arms[0]) or dump_arm_env(kf_arms[0]))
            if st.get("error"):
                check(False, "the kf-store probe runs", st.get("traceback", st["error"]))
            else:
                for a in kf_arms:
                    try:
                        ae = dump_arm_env(a)
                    except Exception:  # noqa: BLE001
                        continue
                    has_kf = _truthy(ae.get("VLM_USE_KEYFRAME_MEMORY"))
                    try:
                        iv = int(str(ae.get("MEM_KF_STORE_INTERVAL", "0")))
                    except Exception:  # noqa: BLE001
                        iv = 0
                    check((not has_kf) or iv > 0,
                          f"{a} builds a keyframe bank and declares MEM_KF_STORE_INTERVAL",
                          f"VLM_USE_KEYFRAME_MEMORY={ae.get('VLM_USE_KEYFRAME_MEMORY')!r}, "
                          f"MEM_KF_STORE_INTERVAL={ae.get('MEM_KF_STORE_INTERVAL')!r}. An arm that "
                          f"builds a bank without one builds it over the call-window store, so its "
                          f"bank is the episode's opening frames and its mechanism never ran.")

                sp, dn = st.get("sparse") or {}, st.get("dense") or {}
                # (c) The contrast must be REAL. If the sparse store already produced a spread bank,
                # the defect is not where this check thinks it is and a PASS would be a lie.
                check(float(sp.get("early_share") or 0) > float(dn.get("early_share") or 0),
                      "a call-window-only store yields a START-CLUSTERED bank and a dense one does not",
                      f"sparse: {sp.get('picked')} (early share "
                      f"{100*float(sp.get('early_share') or 0):.0f}%, span {sp.get('span')}); "
                      f"dense: {dn.get('picked')} (early share "
                      f"{100*float(dn.get('early_share') or 0):.0f}%, span {dn.get('span')})")
                check(int(dn.get("span") or 0) > 1500 and float(dn.get("early_share") or 0) <= 0.15,
                      "a dense store makes the spread span the episode with no start-clustering",
                      f"dense picked={dn.get('picked')}, span={dn.get('span')}, "
                      f"early={dn.get('n_early')}/{dn.get('n_picked')}")
                check(int(dn.get("n_store") or 0) > 10 * int(sp.get("n_store") or 1),
                      "the dense store is an order of magnitude larger than the call-window one",
                      f"{sp.get('n_store')} vs {dn.get('n_store')} frames")
        else:
            notes.append("CHECK 19 skipped: no memory arm in scope")
    except Exception as exc:  # noqa: BLE001
        check(False, "the kf-store check runs", f"{type(exc).__name__}: {exc}")

    return _summary(failures, notes)
def _truthy(value: object) -> bool:
    """Same truthiness the arms and the binding use, so a probe cannot disagree with the run."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _run_probe_in_every_arm(src: str, marker: str) -> dict[str, dict]:
    """Run one probe source in EVERY arm's environment (baseline included) and parse it."""
    out: dict[str, dict] = {}
    for a in discover_arms()[1]:
        try:
            env = dump_arm_env(a)
            proc = subprocess.run(
                [sys.executable, "-c", src],
                env={**env, "PATH": os.environ.get("PATH", ""), "ROOT": str(ROOT)},
                capture_output=True, text=True, timeout=240,
            )
            line = next((l for l in proc.stdout.splitlines() if l.startswith(marker)), None)
            out[a] = json.loads(line[len(marker):]) if line else {
                "error": "no marker", "stderr": proc.stderr[-400:],
            }
        except Exception as exc:  # noqa: BLE001
            out[a] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _summary(failures: list[str], notes: list[str]) -> int:
    print("\n" + "=" * 78)
    for n in notes:
        print(f"  note: {n}")
    if failures:
        print(f"preflight FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        print("=" * 78)
        return 4
    print("preflight PASSED: the baseline is officially aligned and has no memory channel,")
    print("and every treatment differs from it only in its declared keys.")
    print("  NOTE: runtime evidence for channel B comes from census_channels.py after the run.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
