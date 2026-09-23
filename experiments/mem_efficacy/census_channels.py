#!/usr/bin/env python3
"""mem_efficacy post-run census: prove from artifacts which memory channels were live.

WHY ARTIFACTS AND NOT CONFIGURATION
-----------------------------------
`validate_arm.py` proves the switches reached a process and that the read path is gated.
Between two runs, however, more can happen that only data can settle:

  * The keyframe channel may be EMPTY on every step even though its flag is on. With
    MEM_STAGE_ANCHOR=0 the bank is `build_visual_memory(J_hist, ...)`, and `J_hist` holds the
    Planner's OWN `keyframe_positions` nominations. An API Planner does not emit that field, so
    the bank is empty and the "Historical keyframes" prompt block is never emitted. That is a
    **VOID** comparison -- the experiment says nothing about image memory -- and it must not be
    reported as "image memory did not help".
  * The baseline's `memory_indices` may be unexpectedly non-empty, which would mean some other
    path injected historical frames and the baseline is not a no-memory arm.

Channel A has no per-step trace field, so this script re-runs the same functional probe as the
preflight and attaches that process-level evidence to the report, rather than inventing a
runtime field that does not exist.

EXPECTATIONS ARE DERIVED FROM EACH ARM, NOT HARD-CODED
  Each arm declares what it changes; the census reads that declaration back and checks the
  artifact against it. Adding or editing an arm cannot make this script silently wrong about
  the others.

USAGE
  python3 census_channels.py <outputs_root> [<outputs_root> ...]
    <outputs_root> contains <ARM>_s<seed>/**/api_vlm_trace.jsonl,
    e.g. experiments/mem_efficacy/results/mem_efficacy_<jobid>
  Run with the repository venv: <ROOT>/.venv/bin/python
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = Path(os.environ.get("ROOT") or "/project/peilab/why/RoboMemArena")


def find_traces(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "**", "api_vlm_trace.jsonl"), recursive=True))


def find_summaries(root: str) -> tuple[list[dict], dict]:
    """Per-task score rows and the merged aggregate, DEDUPLICATED.

    The runner writes the same task row into THREE files: `<arm>/summary.json` (the merged view
    over every group), and one per task-GROUP (`<arm>/task1/summary.json`,
    `<arm>/tasks2to26/summary.json`). Globbing `**/summary.json` therefore returned every row twice
    -- measured on job 594675: 5 tasks scored, but `tasks = 10` reported. The MEAN was unaffected
    (a duplicated value does not move an average), which is exactly why it survived: the only
    visibly wrong number was a count, and counts are what every completeness test in this file is
    built on. `len(vals) < 26` was reading a double, so a genuinely truncated 13-task run would have
    reported itself as complete, and a 1-task scoped run as `2/26`.

    Deduplicated on the identity of the ROW rather than on the file, so a run that legitimately
    repeats a task (a retry, or one row per trial) still contributes each of its rows.
    """
    rows: list[dict] = []
    seen: set[tuple] = set()
    for sv in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
        try:
            data = json.load(open(sv, encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        for row in items:
            if not isinstance(row, dict):
                continue
            key = (
                row.get("task_id"), row.get("seed"), row.get("trial"),
                row.get("stage_score_pct"), row.get("status"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if not rows:
        # Fallback for the flat layout some older runners used, where the row lives only in a dict.
        for sv in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
            try:
                data = json.load(open(sv, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict) and "task_id" in data:
                rows.append(data)
    macros: dict = {}
    for av in sorted(glob.glob(os.path.join(root, "**", "aggregate.json"), recursive=True)):
        try:
            macros.update(json.load(open(av, encoding="utf-8")))
        except Exception:
            pass
    return rows, macros


def declared_task_scope(arm_dir: str) -> int | None:
    """How many tasks this arm DECLARED, from the environment it actually ran.

    Read from `resolved_env.txt`, which `run_26x1.sbatch` freezes out of the LIVE environment (after
    `TASKS_JSON_SCOPE` is applied), not out of the arm file. So a deliberate narrowing is visible
    here and a truncated run is not. Returns None when the file or the key is missing, which keeps
    older runs on the previous "assume 26" behaviour rather than silently excusing them.
    """
    path = os.path.join(arm_dir, "resolved_env.txt")
    if not os.path.exists(path):
        return None
    raw = None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("TASKS_JSON="):
                    raw = line.split("=", 1)[1].strip().strip("'\"")
                    break
    except Exception:
        return None
    if not raw:
        return None
    if ".." in raw:
        try:
            lo, hi = raw.split("..", 1)
            return abs(int(hi) - int(lo)) + 1
        except Exception:
            return None
    if raw.startswith("["):
        try:
            return len(json.loads(raw))
        except Exception:
            return None
    return len([x for x in raw.split(",") if x.strip()])



def run_dirs(root: str, arms: list[str]) -> list[tuple[str, str]]:
    """Find `<ARM>_s<seed>` directories under root. Returns [(arm, path)].

    Two layouts are accepted, and both are needed:

      * `<root>/<ARM>_s<seed>`   -- what `run_26x1.sbatch` produces (one job directory, one
                                    subdirectory per arm) and what the census is invoked on;
      * `<root>/<anything>/<ARM>_s<seed>` -- a nesting used by the older runners.

    Both patterns are tried rather than just the nested one. The original code only looked one
    level down, which for this experiment's own layout matched NOTHING, so every arm silently
    fell through to the "legacy / ad-hoc" branch below -- the census then reported channel B and
    a score without ever attributing them to an arm, and channel A was skipped. A census that
    quietly stops checking anything is worse than no census, because the run still looks analysed.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for arm in arms:
        candidates = (
            glob.glob(os.path.join(root, f"{arm}_s*"))
            + glob.glob(os.path.join(root, "*", f"{arm}_s*"))
        )
        if Path(root).name.startswith(f"{arm}_s"):
            candidates.append(root)
        for d in sorted(candidates):
            if d not in seen:
                seen.add(d)
                found.append((arm, d))
    return found


def census_channel_b(traces: list[str]) -> dict:
    """Runtime evidence for channel B: did historical frames reach the Planner context?"""
    n_lines = n_kf = n_mem = n_nom = 0
    max_k = 0
    k_hist: Counter = Counter()
    unparsable = 0
    # The CANDIDATE POOL, which is the channel's real constraint. A bank can only be as good as the
    # frames the store holds, and the store used to be filled only from the context windows of
    # planner calls that survived async queue eviction. `n_store`/`recent_start` make "the pool
    # cannot span the episode" measurable rather than inferred; the early share makes "the bank is
    # the opening scene" a number instead of an impression.
    store_sizes: list[int] = []
    n_bank_slots = 0
    n_bank_early = 0
    n_store_present = 0
    for tf in traces:
        try:
            with open(tf, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        unparsable += 1
                        continue
                    n_lines += 1
                    k = list(rec.get("K_indices_abs") or [])
                    m = list(rec.get("memory_indices") or [])
                    j = list(rec.get("keyframe_positions") or [])
                    n_kf += bool(k)
                    n_mem += bool(m)
                    n_nom += bool(j)
                    max_k = max(max_k, len(k))
                    k_hist[len(k)] += 1
                    if isinstance(rec.get("n_store"), int):
                        store_sizes.append(int(rec["n_store"]))
                    for i in k:
                        n_bank_slots += 1
                        if int(i) <= 8:
                            n_bank_early += 1
        except Exception:
            continue
    return {
        "trace_files": len(traces),
        "plan_steps": n_lines,
        "steps_with_K_nonempty": n_kf,
        "steps_with_memory_nonempty": n_mem,
        "steps_with_planner_nomination": n_nom,
        "max_K": max_k,
        "K_size_histogram": dict(sorted(k_hist.items())),
        "unparsable_lines": unparsable,
        "n_store_median": (sorted(store_sizes)[len(store_sizes) // 2] if store_sizes else None),
        "n_store_max": (max(store_sizes) if store_sizes else None),
        "bank_early_share": (n_bank_early / n_bank_slots) if n_bank_slots else None,
        "bank_slots": n_bank_slots,
    }


def find_pull_reports(arm_dir: str) -> list[str]:
    """Every per-process pull report under this arm.

    Reports are written as `<path>.<pid>`, so the glob must accept both that and a bare
    `memexp_pull_report.json` (which older runs wrote). Reading only the bare name would silently
    find nothing, and the census would then report "no report" for a run that produced twelve.
    """
    pats = ("memexp_pull_report.json", "memexp_pull_report.json.*")
    out: list[str] = []
    for pat in pats:
        out.extend(glob.glob(os.path.join(arm_dir, "**", pat), recursive=True))
    return sorted(p for p in out if not p.endswith(".tmp"))


def find_memfix_reports(arm_dir: str) -> list[str]:
    pats = ("memexp_memfix_report.json", "memexp_memfix_report.json.*")
    out: list[str] = []
    for pat in pats:
        out.extend(glob.glob(os.path.join(arm_dir, "**", pat), recursive=True))
    return sorted(p for p in out if not p.endswith(".tmp"))


def read_code_provenance(arm_dir: str) -> dict | None:
    """The code fingerprint recorded for this run, or None if the run predates it."""
    for path in sorted(glob.glob(os.path.join(arm_dir, "**", "code_provenance.json"), recursive=True)):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return None
    return None


def short_fingerprint(prov: dict | None) -> str:
    """One stable digest over the hashed files, for eyeballing two runs against each other.

    Hashes the (path, sha256) pairs rather than the file contents a second time, so this is cheap
    and depends on exactly the set of files `record_provenance.py` decided were load-bearing.
    """
    if not prov:
        return "<none>"
    h = hashlib.sha256()
    for k in sorted(prov.get("file_sha256") or {}):
        h.update(k.encode())
        h.update(str((prov["file_sha256"] or {})[k]).encode())
    return h.hexdigest()[:16]


def read_resolved_env(arm_dir: str) -> dict | None:
    """The run's OWN recorded environment, or None when the artifact is absent.

    Needed wherever "the arm file says X today" and "this run was configured with X" lead to
    different verdicts. `dump_arm_env` re-sources the CURRENT arm file, which is right for stating
    what an arm REQUESTS, but re-running this census on an archived job would then attribute today's
    configuration to yesterday's run: the old pushmem directory from job 590799 predates the
    correction entirely, yet the current arm file declares it, so a check keyed on the arm file
    alone reports a missing counter that could never have existed.
    """
    for pat in ("resolved_env.txt", "*/resolved_env.txt"):
        for path in sorted(glob.glob(os.path.join(arm_dir, pat))):
            out: dict[str, str] = {}
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            out[k] = v
            except Exception:
                return None
            return out
    return None


def census_memfix(reports: list[str]) -> dict:
    """Runtime evidence that the memory-content correction ENGAGED.

    Read from the arm's own in-process counters, not from the arm's flags. `MEMEXP_MEMFIX_ENABLE=1`
    records that the arm asked for the correction, CHECK 12 records that the patches installed, and
    GATE 0c records that the corrected functions behave -- none of the three answers whether the
    corrected path was reached during the actual evaluation. `n_render == 0` means the evaluator
    never rendered memory, which would make the arm's score a statement about the original code.

    `n_guess_demoted` and `n_dup_dropped` are the two counters that correspond to the defects: the
    first counts recovery GUESSES moved out of the evidence section, the second counts repeated rows
    collapsed. Both being zero would mean the corrections had nothing to act on -- plausible on a
    clean episode, which is why the report also carries `max_ctx_len`, so a reader can tell
    "nothing to correct" apart from "the correction was never called".
    """
    merged: dict[str, Any] = {
        "reports": len(reports), "n_render": 0, "n_dup_dropped": 0, "n_guess_demoted": 0,
        "max_ctx_len": 0, "max_len_history": 0, "pids": [], "errors": [],
    }
    for path in reports:
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            merged["errors"].append(f"{os.path.basename(path)}: {exc}")
            continue
        if data.get("pid") is not None:
            merged["pids"].append(data["pid"])
        # Sums for counters, `max` for maxima. Summing `max_ctx_len` would report a context
        # length that no single process ever produced -- twelve processes at 300 chars is not a
        # 3600-character context, and that number is the one used to detect accumulation
        # returning, so an inflated value would raise a false alarm.
        for k in ("n_render", "n_dup_dropped", "n_guess_demoted"):
            try:
                merged[k] += int(data.get(k) or 0)
            except (TypeError, ValueError):
                pass
        for k in ("max_ctx_len", "max_len_history"):
            try:
                merged[k] = max(merged[k], int(data.get(k) or 0))
            except (TypeError, ValueError):
                pass
        hist = data.get("len_history") or []
        if hist:
            try:
                merged["max_len_history"] = max(
                    merged["max_len_history"], max(int(h) for h in hist)
                )
            except (TypeError, ValueError):
                pass
    return merged


def census_channel_c(reports: list[str]) -> dict:
    """Runtime evidence for channel C: did the Planner actually USE the memory tools?

    Read from the arm's OWN in-process report, not from the configuration. The distinction is the
    whole reason this channel exists: `MEMEXP_PULL_ENABLE=1` proves the loop was INSTALLED, and
    GATE 0b proves it WORKS, but neither answers whether the Planner ever called it. A run with
    the loop installed and zero tool calls produces a score that is indistinguishable from
    "memory does not help" -- which is precisely what happened to PMH, whose read path ended up
    with zero searches in it while every flag said enabled.

    CUMULATIVE KEYS, NOT PER-EPISODE ONES
    -------------------------------------
    The counters are read from `*_cumulative` / `by_tool` at the top level of the report. The
    `substrate` and `tools` keys still exist but describe ONE episode -- whichever was live at the
    last write -- and reading them was a measured defect: job 592860's reports carried a correct
    `tool_calls = 5` beside `by_tool = {}`, because the last episode of a 26-task run makes no tool
    calls. `tool_calls` was right and "which tools" was empty, and the per-tool breakdown is the
    ONLY number that distinguishes the search tool from the dereference tool.

    Reports written before the fix do not have the new keys. They are still read, under the old
    semantics, and `legacy_reports` records how many so the caller can say so rather than silently
    comparing two different measurements.
    """
    merged: dict[str, Any] = {
        "reports": len(reports), "legacy_reports": 0, "steps": 0, "steps_with_push": 0,
        "tool_calls": 0, "steps_with_tool": 0, "cap_hits": 0, "cap_unresolved": 0,
        "loop_errors": 0, "api_errors": 0, "primitive_returned": 0,
        "tools_module": set(), "tools_module_error": [], "broken": 0,
        "n_action_write": 0, "n_unbound_minted": 0, "n_query": 0, "n_query_hit": 0,
        "n_query_miss": 0, "n_frames_served_off_context": 0,
        "n_serve_empty_not_yet_observed": 0,
        # Write-path-2 counters. `n_spec_minted` is the falsifier for the task-spec seed: a run
        # with it at 0 built its bank the old (action-only) way, and is not evidence about the
        # seeded write path. Kept beside `seed_error` so "the seed did not run" and "the seed ran
        # and the Planner ignored it" can never be read as the same outcome.
        "n_spec_minted": 0, "n_stage_active": 0, "n_stage_verified": 0,
        "n_spec_unopened": 0, "seed_error": [], "seeded_stages": 0,
        "n_repeat_calls": 0, "n_spec_shown": 0, "n_search": 0, "n_search_hit": 0,
        "n_search_miss": 0, "by_tool": {}, "errors": [], "spec_counter_reports": 0,
    }
    for path in reports:
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if data.get("pid") is not None:
            merged.setdefault("pids", []).append(data["pid"])
        if data.get("tools_module"):
            merged["tools_module"].add(str(data["tools_module"]))
        if data.get("tools_module_error"):
            merged["tools_module_error"].append(str(data["tools_module_error"]))
        if data.get("broken_now"):
            merged["broken"] += 1
        for k in ("steps", "steps_with_push", "tool_calls", "steps_with_tool", "cap_hits",
                  "cap_unresolved", "loop_errors", "api_errors", "primitive_returned"):
            merged[k] = int(merged[k]) + int(data.get(k) or 0)

        cum_sub = data.get("substrate_cumulative")
        cum_tools = data.get("tools_cumulative")
        by_tool = data.get("by_tool")
        if cum_sub is None and cum_tools is None:
            # A report from before the cumulative accounting. Its `substrate`/`tools` describe a
            # single episode, so these totals UNDERCOUNT and must be labelled as such.
            merged["legacy_reports"] += 1
            cum_sub = data.get("substrate") or {}
            cum_tools = data.get("tools") or {}
            by_tool = (data.get("tools") or {}).get("by_tool") or {}
        # Whether the offer COUNTER itself exists. A pre-fix report has no `n_spec_shown` at all, so
        # reading it as 0 makes "the tool block was never shown" a confident falsehood. This is the
        # same defect class as the ones this census exists to catch -- an absent measurement
        # reported as a zero -- and it fired on the first real archived directory it was run against.
        if "n_spec_shown" in (cum_tools or {}):
            merged["spec_counter_reports"] += 1
        for k in ("n_action_write", "n_unbound_minted", "n_query", "n_query_hit", "n_query_miss",
                  "n_frames_served_off_context", "n_serve_empty_not_yet_observed",
                  "n_spec_minted", "n_stage_active", "n_stage_verified", "n_spec_unopened"):
            merged[k] = int(merged[k]) + int((cum_sub or {}).get(k) or 0)
        # The seed's own failure record. Top-level, not inside `substrate_cumulative`, because it
        # describes the INSTALL path rather than a counting interval -- and it is cleared before
        # each attempt, so what is collected here describes the last episode that tried.
        if data.get("seed_error"):
            merged["seed_error"].append(str(data["seed_error"]))
        merged["seeded_stages"] = max(int(merged["seeded_stages"]),
                                      int(data.get("seeded_stages") or 0))
        for k in ("n_repeat_calls", "n_spec_shown", "n_search", "n_search_hit", "n_search_miss"):
            merged[k] = int(merged[k]) + int((cum_tools or {}).get(k) or 0)
        for name, n in (by_tool or {}).items():
            merged["by_tool"][str(name)] = int(merged["by_tool"].get(str(name), 0)) + int(n)
        merged["errors"].extend(list(data.get("errors") or []))
    merged["tools_module"] = sorted(merged["tools_module"])
    return merged


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    # Same discovery as the preflight, so the two can never disagree about what an arm is.
    from validate_arm import discover_arms, dump_arm_env, run_probe

    treatments, all_arms = discover_arms()
    baseline = "nomem"

    print("=" * 78)
    print("mem_efficacy post-run census")
    print(f"  ROOT   = {ROOT}")
    print(f"  arms   = {all_arms}")
    print("=" * 78)

    problems: list[str] = []
    voids: list[str] = []
    warnings: list[str] = []

    # Positional run roots, plus `--run-root <path>`. The previous form iterated `argv[1:]` raw, so
    # a flag was taken as a PATH: `census_channels.py --run-root <dir>` censused the literal string
    # `--run-root`, found nothing under it, and appended "nothing to census" to the PROBLEM list
    # alongside the real findings. A census that reports a failure for its own command line is
    # teaching you to ignore its problem list.
    roots: list[str] = []
    _args = argv[1:]
    _i = 0
    while _i < len(_args):
        _a = _args[_i]
        if _a == "--run-root":
            _i += 1
            if _i < len(_args):
                roots.append(_args[_i])
        elif _a.startswith("-"):
            print(f"  !! ignoring unrecognised flag {_a!r} (usage: [--run-root] <run_root> ...)")
        else:
            roots.append(_a)
        _i += 1

    for run_root in roots:
        print(f"\n########## {run_root} ##########")
        runs = run_dirs(run_root, all_arms)
        if not runs:
            # Fallback for DIAGNOSTIC use on a legacy run that predates this experiment and so
            # has no `<ARM>_s<seed>` directory. Channel A cannot be attributed to an arm here, so
            # it is skipped; channel B and the scores are still reported, because those are what
            # one wants when answering "was this old run's image channel dead too?".
            traces = find_traces(run_root)
            if not traces:
                print(f"  no <ARM>_s<seed> directory and no api_vlm_trace.jsonl under {run_root}")
                problems.append(f"{run_root}: nothing to census")
                continue
            print("  no <ARM>_s<seed> directory; treating this as a LEGACY/AD-HOC run.")
            print("  channel A cannot be attributed to an arm and is skipped.")
            rows, macros = find_summaries(run_root)
            print("\n    [channel B: historical keyframe images] api_vlm_trace.jsonl")
            b = census_channel_b(traces)
            print(f"      trace files                 = {b['trace_files']}")
            print(f"      plan steps                  = {b['plan_steps']}"
                  f"  (unparsable {b['unparsable_lines']})")
            print(f"      steps with K non-empty      = {b['steps_with_K_nonempty']}")
            print(f"      steps with memory non-empty = {b['steps_with_memory_nonempty']}")
            print(f"      steps with a nomination     = {b['steps_with_planner_nomination']}")
            print(f"      max |K| = {b['max_K']}   |K| histogram = {b['K_size_histogram']}")
            if b["steps_with_K_nonempty"] == 0 and b["steps_with_memory_nonempty"] == 0:
                print("      -> image channel was DEAD for this whole run")
                voids.append(f"{run_root}: channel B produced no frames on any step")
            print("\n    [scores]")
            if rows:
                vals = [float(r.get("stage_score_pct") or 0.0) for r in rows]
                print(f"      tasks = {len(vals)}   macro stage_score_pct = {sum(vals)/len(vals):.2f}")
                if len(vals) < 26:
                    print(f"      !! PARTIAL: {len(vals)}/26 scored; not a protocol number")
            else:
                print("      !! no summary.json")
            continue

        for arm, arm_dir in runs:
            print(f"\n  ===== arm={arm}  dir={arm_dir} =====")

            # Per-arm expectation. Prefer the run's frozen `resolved_env.txt` over re-sourcing
            # the arm file: Stage-1 MEMPROF overlays (H0) flip MEM_STAGE_ANCHOR / retry AFTER the
            # arm is sourced, and those flips are load-bearing for the census branch. Job 609345
            # failed with "nomination prompt ON but 0 nominations" because census read the arm's
            # MEM_STAGE_ANCHOR=1 while the live run had H0's MEM_STAGE_ANCHOR=0 — wrong branch.
            try:
                env = dump_arm_env(arm)
            except Exception as exc:
                print(f"    !! cannot source arms/{arm}.sh: {type(exc).__name__}: {exc}")
                problems.append(f"{run_root}/{arm}: cannot source the arm file")
                continue
            recorded = read_resolved_env(arm_dir)
            if recorded:
                env = {**env, **recorded}
                env_src = "resolved_env.txt (overlays arm file)"
            else:
                env_src = "arm file only (no resolved_env.txt)"
            kf_on = env.get("VLM_USE_KEYFRAME_MEMORY", "1") in {"1", "true", "yes"}
            anchor_on = env.get("MEM_STAGE_ANCHOR", "0") not in {"0", "", "false", "no", "off"}
            print(f"    declared: VLM_USE_KEYFRAME_MEMORY={int(kf_on)} MEM_STAGE_ANCHOR={int(anchor_on)}"
                  f"  [{env_src}]")
            if str(env.get("MEMPROF_NAME", "")).strip():
                print(f"    MEMPROF_NAME={env.get('MEMPROF_NAME')}")

            traces = find_traces(arm_dir)
            rows, macros = find_summaries(arm_dir)

            # ---- channel A: re-run the functional probe on this arm's own environment -----
            print("    [channel A: textual evidence] functional probe")
            try:
                probe = run_probe(env)
                if "error" in probe:
                    print(f"      probe crashed: {probe['error']}")
                    problems.append(f"{run_root}/{arm}: channel A probe crashed")
                else:
                    dead = probe["vlm_context_len"] == 0 and probe["compose_len"] == 0
                    print(f"      inject_vlm_context = {probe['inject_vlm_context']}")
                    print(f"      get_vlm_context() len = {probe['vlm_context_len']}"
                          f"  head={probe['vlm_context_head']!r}")
                    print(f"      compose_planner_context() len = {probe['compose_len']}")
                    # Channel A is a CONTROLLED CONSTANT in this experiment: every arm declares
                    # HARNESS_VLM_CONTEXT=0, because the dual-track measurement showed the textual
                    # channel is detrimental (see arms/pushmem.sh). So "dead" is the CONFIGURED
                    # outcome for a treatment, not a defect -- and reporting it as one contradicted
                    # the OK line this same census prints a few lines later, on the same run. The
                    # defect is a channel that DISAGREES with its declaration, in either direction.
                    declared_a = str(env.get("HARNESS_VLM_CONTEXT", "0")).strip().lower() in {
                        "1", "true", "yes", "on", "y", "t",
                    }
                    expect_dead = (arm == baseline) or (not declared_a)
                    if expect_dead and not dead:
                        problems.append(
                            f"{run_root}/{arm}: channel A is ALIVE while declared OFF "
                            f"(HARNESS_VLM_CONTEXT={env.get('HARNESS_VLM_CONTEXT')!r})"
                        )
                    if (not expect_dead) and dead:
                        problems.append(
                            f"{run_root}/{arm}: channel A is DEAD while declared ON"
                        )
                    if expect_dead == dead:
                        print(f"      -> PASS ({'dead as designed' if dead else 'live as designed'})")
            except Exception as exc:
                print(f"      probe failed: {type(exc).__name__}: {exc}")
                problems.append(f"{run_root}/{arm}: channel A probe failed")

            # ---- channel B: per-step trace -----------------------------------------------
            print("    [channel B: historical keyframe images] api_vlm_trace.jsonl")
            b = census_channel_b(traces)
            if not traces:
                print("      !! no api_vlm_trace.jsonl found -- the run produced no trace")
                problems.append(f"{run_root}/{arm}: no trace")
            else:
                print(f"      trace files                 = {b['trace_files']}")
                print(f"      plan steps                  = {b['plan_steps']}"
                      f"  (unparsable {b['unparsable_lines']})")
                print(f"      steps with K non-empty      = {b['steps_with_K_nonempty']}")
                print(f"      steps with memory non-empty = {b['steps_with_memory_nonempty']}")
                print(f"      steps with a nomination     = {b['steps_with_planner_nomination']}"
                      f"   (PrediMem channel 1; policy declared = "
                      f"{str(env.get('MEM_KF_NOMINATION_PROMPT', '0')).strip() or '0'})")
                print(f"      max |K| = {b['max_K']}   |K| histogram = {b['K_size_histogram']}")
                # The channel's CANDIDATE POOL. Without these two lines a start-clustered bank reads
                # as a healthy one: every liveness counter above is positive while the images carry
                # nothing, which is exactly how the defect survived the earlier censuses.
                if b.get("n_store_median") is not None:
                    print(f"      frame store size   = median {b['n_store_median']}, "
                          f"max {b['n_store_max']}   (MEM_KF_STORE_INTERVAL="
                          f"{str(env.get('MEM_KF_STORE_INTERVAL', '0')).strip() or '0'})")
                if b.get("bank_early_share") is not None:
                    _es = 100 * float(b["bank_early_share"])
                    print(f"      bank slots <= frame 8 = {_es:.1f}% of {b['bank_slots']}")
                    # The opening frames are the ones taken before the robot has touched anything,
                    # so a bank dominated by them cannot answer an occlusion question and cannot be
                    # evidence that the channel works. Threshold is generous (the first 100 steps of
                    # a ~2500-step episode) so that only genuine start-collapse trips it.
                    if _es > 40.0 and b["bank_slots"] >= 20:
                        _iv = str(env.get("MEM_KF_STORE_INTERVAL", "0")).strip()
                        if _iv not in {"", "0"}:
                            print("      -> FAIL: the bank is START-CLUSTERED. The store is declared")
                            print("         dense, but most of what the Planner is shown predates the")
                            print("         manipulation, so the bank cannot hold the observation the")
                            print("         task turns on. The candidate pool is not a record of the")
                            print("         episode.")
                            problems.append(
                                f"{run_root}/{arm}: bank start-clustered ({_es:.0f}% <= frame 8) "
                                f"despite a dense store")
                        else:
                            print("      -> VOID: start-clustered bank and NO dense store declared, so")
                            print("         the pool is the async call pattern. This is the defect, not")
                            print("         a measurement of the memory mechanism.")
                            voids.append(
                                f"{run_root}/{arm}: bank start-clustered, no dense store")

                live = b["steps_with_K_nonempty"] > 0 or b["steps_with_memory_nonempty"] > 0
                if not kf_on:
                    if live:
                        print("      -> FAIL: historical frames reached a channel that is declared OFF")
                        problems.append(f"{run_root}/{arm}: channel B leaked while declared off")
                    else:
                        print("      -> PASS: K and memory empty on every step; channel B is dead")
                elif not anchor_on:
                    if live:
                        print("      -> PASS: historical frames reached context without stage-anchor")
                        print("         (dense store and/or nominations; H0-compatible)")
                        _nom_declared = str(env.get("MEM_KF_NOMINATION_PROMPT", "")).strip().lower() in {
                            "1", "true", "yes", "on", "y", "t",
                        }
                        if _nom_declared and b["steps_with_planner_nomination"] == 0:
                            print("      note: MEM_KF_NOMINATION_PROMPT is ON but nominations=0;")
                            print("            frames came from dense store/spread, not PrediMem nomination.")
                            voids.append(
                                f"{run_root}/{arm}: nomination claim unproven under MEM_STAGE_ANCHOR=0"
                            )
                    else:
                        print("      -> VOID: the arm turns the channel on but MEM_STAGE_ANCHOR=0,")
                        print("         so without a live dense store the bank is a pure function of")
                        print("         the Planner's nominations. Nominations=0 => no image channel.")
                        print("         This comparison is VOID, not negative.")
                        voids.append(f"{run_root}/{arm}: channel B is VOID (nominations=0)")
                else:
                    if live:
                        print("      -> PASS: historical frames did reach the context")
                        # NOMINATION IS NOW A DECLARED CHANNEL, so "zero nominations" is no longer
                        # automatically benign. It was benign alone (the frames came from the
                        # harness's anchors), but an arm that also teaches the nomination POLICY
                        # is claiming PrediMem's actual mechanism, and a claim of zero has to be
                        # read as a failure of that claim rather than as an expected outcome. This
                        # is the whole falsifier for `MEM_KF_NOMINATION_PROMPT`: the instruction
                        # being present in the prompt is not evidence that the model acted on it,
                        # and only the trace can tell the two apart.
                        _nom_declared = str(env.get("MEM_KF_NOMINATION_PROMPT", "")).strip().lower() in {
                            "1", "true", "yes", "on", "y", "t",
                        }
                        if b["steps_with_planner_nomination"] == 0:
                            if _nom_declared:
                                print("      -> FAIL: MEM_KF_NOMINATION_PROMPT is ON but the Planner")
                                print("         nominated NOTHING on any step. The instruction was in")
                                print("         the prompt and the model ignored it, so this arm's bank")
                                print("         is NOT PrediMem-style nomination memory -- it is the")
                                print("         anchor+stride fallback, and must be reported as such.")
                                problems.append(
                                    f"{run_root}/{arm}: nomination prompt ON but 0 nominations"
                                )
                            else:
                                print("         (and with ZERO nominations -- i.e. the frames came from")
                                print("          the harness's own stage/subtask anchors, which is the")
                                print("          whole point of this arm)")
                    else:
                        print("      -> FAIL: MEM_STAGE_ANCHOR=1 should have produced a non-empty")
                        print("         bank from stage boundaries alone. Check that the eval loop")
                        print("         advanced at least one stage and that MEM_* survived to the")
                        print("         process (resolved_env.txt).")
                        problems.append(f"{run_root}/{arm}: channel B dead despite MEM_STAGE_ANCHOR=1")

            # ---- channel C: the agent-callable tool loop (PULL arm only) -------------------
            pull_on = str(env.get("MEMEXP_PULL_ENABLE", "")).strip() in {"1", "true", "yes", "on"}
            if pull_on:
                print("    [channel C: agent-called memory tools] memexp_pull_report.json")
                reports = find_pull_reports(arm_dir)
                c = census_channel_c(reports)
                if not reports:
                    print("      !! no memexp_pull_report.json -- the binding never wrote a report.")
                    print("         Either sitecustomize did not load (check PYTHONPATH in")
                    print("         resolved_env.txt) or the evaluator never imported the planner.")
                    problems.append(f"{run_root}/{arm}: PULL arm produced no in-process report")
                else:
                    print(f"      reports                     = {c['reports']}"
                          + (f"  ({c['legacy_reports']} pre-cumulative: totals UNDERCOUNT)"
                             if c["legacy_reports"] else ""))
                    print(f"      tools module loaded          = {c['tools_module'] or '<unrecorded>'}")
                    print(f"      plan steps seen             = {c['steps']}"
                          f"  (with a push: {c['steps_with_push']})")
                    print(f"      tool spec shown             = {c['n_spec_shown']}"
                          + ("" if c["spec_counter_reports"] else
                             "   <- COUNTER ABSENT: offer count UNKNOWN, not zero")
                          + ("   <- the call rate's DENOMINATOR"
                             if c["spec_counter_reports"] else ""))
                    print(f"      tool calls                  = {c['tool_calls']}"
                          f"  across {c['steps_with_tool']} step(s)   {c['by_tool'] or '{}'}")
                    print(f"      addresses minted / total    = {c['n_unbound_minted']} / "
                          f"{c['n_action_write']} actions recorded")
                    print(f"      task-spec seed              = {c['n_spec_minted']} address(es) from "
                          f"{c['seeded_stages']} stage(s); "
                          f"{c['n_stage_active']} stage activation(s), "
                          f"{c['n_spec_unopened']} still unopened")
                    print(f"      stages verified complete    = {c['n_stage_verified']}")
                    print(f"      queries hit / miss          = {c['n_query_hit']} / {c['n_query_miss']}"
                          f"   (of {c['n_query']})")
                    print(f"      free-text searches h/m      = {c['n_search_hit']} / "
                          f"{c['n_search_miss']}   (of {c['n_search']})")
                    print(f"      frames served off-context   = {c['n_frames_served_off_context']}")
                    # THE YIELD OF THE RETRIEVAL TOOL, which is what its design is judged on and
                    # what the previous surface got wrong: 95 calls bought 19 frames (5.0 calls per
                    # observation), of which 47 calls were enumerations that cannot return a frame
                    # at all. Printed as a ratio so a re-run is comparable to that number directly.
                    _tc = int(c["tool_calls"] or 0)
                    _fs = int(c["n_frames_served_off_context"] or 0)
                    print(f"      frames per tool call        = "
                          f"{(float(_fs) / _tc) if _tc else 0.0:.2f}"
                          f"   ({_fs} frame(s) / {_tc} call(s))")
                    if _tc and c["by_tool"]:
                        _enum = (int(c["by_tool"].get("list_facts", 0))
                                 + int(c["by_tool"].get("list_unbound", 0)))
                        print(f"      enumeration share of calls  = {_enum}/{_tc} = "
                              f"{100.0 * _enum / _tc:.0f}%   <- returns no observation by "
                              f"construction; the push already renders it")
                    print(f"      served-empty (not yet seen) = {c['n_serve_empty_not_yet_observed']}")
                    print(f"      primitives returned         = {c['primitive_returned']}")
                    print(f"      repeats in a step           = {c['n_repeat_calls']}")
                    print(f"      cap hits / unresolved       = {c['cap_hits']} / {c['cap_unresolved']}")
                    print(f"      loop errors / api errors    = {c['loop_errors']} / {c['api_errors']}")

                    # A registry that failed to load makes the whole channel unmeasured, and it
                    # must be FAIL rather than the WARN below. The arm continues as the push-only
                    # control while every flag reads as enabled, so a score from it says nothing
                    # about agent retrieval -- and the WARN branch would report exactly that
                    # score as "the Planner chose not to use memory", which is a claim about the
                    # model drawn from a broken configuration.
                    # The task-spec seed's own verdict. Kept separate from the tool-loop verdict
                    # because the two failures have opposite implications: a dead tool loop means
                    # the Planner declined to retrieve, while a dead seed means the bank could not
                    # have contained the answer in the first place -- and that second one is
                    # invisible in every counter this census had before, since `facts` and
                    # `addresses` both read healthy on an unanswerable bank.
                    if c["seed_error"]:
                        print("      -> FAIL: the task-spec seed failed, so this bank holds only "
                              "what the Planner happened to name.")
                        for e in c["seed_error"][:3]:
                            print(f"         {e}")
                        problems.append(f"{run_root}/{arm}: task-spec seed failed ({c['seed_error'][:1]})")
                    elif c["n_spec_minted"] == 0:
                        print("      -> FAIL: the task-spec seed minted NOTHING. The bank was built "
                              "the old action-only way, so this arm's score is not a statement "
                              "about the seeded write path.")
                        problems.append(f"{run_root}/{arm}: task-spec seed minted 0 addresses")
                    elif c["n_stage_active"] == 0:
                        print("      -> FAIL: no stage ever activated, so every seeded address stayed "
                              "unreadable and the seed could not have been used.")
                        problems.append(f"{run_root}/{arm}: task-spec seed never opened (0 activations)")

                    if c["tools_module_error"] or c["broken"]:
                        print("      -> FAIL: the tool registry did not load, so this arm ran as "
                              "the PUSH-ONLY control.")
                        for e in c["tools_module_error"][:3]:
                            print(f"         {e}")
                        print("         `MEMEXP_TOOLS_MODULE` names a module the evaluating "
                              "process could not import. Do not report this arm's score as a "
                              "measurement of agent retrieval: nothing was ever offered.")
                        problems.append(
                            f"{run_root}/{arm}: PULL arm's tool registry failed to load "
                            f"({c['tools_module_error'][:1]}) -- the arm degraded to push-only"
                        )
                    # The three conditions that make the number meaningless rather than negative.
                    elif c["steps"] == 0 and c["tool_calls"] == 0:
                        # Distinguishable from a real zero only because the counters are now
                        # flushed per step (CHECK 14 in validate_arm.py). Before that, this
                        # branch and the one below were the same branch, and a run whose evidence
                        # file was never updated was reported as a confident finding about the
                        # Planner's behaviour.
                        print("      -> FAIL: the hook never saw a SINGLE plan step.")
                        print("         This is not 'the Planner chose not to use memory'. The")
                        print("         wrapped `_build_messages` was never called in the")
                        print("         evaluating process, so nothing was pushed and no tool was")
                        print("         offered. Treat the arm's score as a broken run, not as a")
                        print("         measurement -- and check that the evaluator really imports")
                        print("         `harness.api_vlm_planner` with this PYTHONPATH.")
                        problems.append(
                            f"{run_root}/{arm}: PULL hook saw 0 plan steps -- the arm did not run "
                            f"as configured (reports merged: {c['reports']})"
                        )
                    elif c["n_spec_shown"] == 0 and c["spec_counter_reports"]:
                        # A distinct failure from "0 calls", and it must not be merged with it. The
                        # tools were never OFFERED, so no answer the Planner gave can be read as a
                        # preference. `MEMEXP_PULL_TOOLS=0` produces exactly this, and that is the
                        # push-only control -- a legitimate arm, but not the one this name means.
                        #
                        # GUARDED ON THE COUNTER EXISTING. A report written before the counter was
                        # added has no `n_spec_shown` key, and reading that absence as 0 makes this
                        # branch announce "the tool block was never shown" about a run where the
                        # model called the tools five times. It fired on the first archived
                        # directory this census was run against -- the same absent-measurement-as-
                        # zero defect it is here to catch, one layer up.
                        print("      -> FAIL: the tool block was NEVER shown to the Planner.")
                        print(f"         The hook ran for {c['steps']} plan step(s) but appended a")
                        print("         tool spec on none of them, so nothing was offered. Check")
                        print("         `MEMEXP_PULL_TOOLS` (0 = the push-only control) and the")
                        print("         `tools_enabled_now` field in the same reports. A 0 call")
                        print("         rate under these conditions is not a finding about the")
                        print("         Planner -- there was nothing to decline.")
                        problems.append(
                            f"{run_root}/{arm}: PULL arm never showed the tool block "
                            f"({c['steps']} steps, 0 specs) -- measures the push only"
                        )
                    elif c["tool_calls"] == 0 and not c["spec_counter_reports"]:
                        # Pre-fix report: the offer count does not exist, so the arm CANNOT be
                        # judged. Reported as UNMEASURED rather than as a finding, because the two
                        # would otherwise be indistinguishable in a summary table.
                        print("      -> UNMEASURED: no tool calls, and the report predates the offer")
                        print("         counter, so whether the tools were even shown is UNKNOWN.")
                        print("         Do not read this as 'the Planner declined': the same numbers")
                        print("         are produced by a model that was never offered anything.")
                        voids.append(
                            f"{run_root}/{arm}: offer count absent (pre-fix report) -- the call "
                            f"rate cannot be interpreted"
                        )
                    elif c["tool_calls"] == 0:
                        print("      -> WARN: the Planner NEVER called a memory tool.")
                        print(f"         The hook was live for {c['steps']} plan step(s) and the tool")
                        print(f"         block was actually shown on {c['n_spec_shown']} of them, so")
                        print("         this is a finding about the Planner's propensity, not a broken")
                        print("         path -- but it means this arm's score measures the SMALL PUSH")
                        print("         ONLY. Report it as such; do not attribute anything to agent")
                        print("         retrieval. The denominator is what makes the claim honest:")
                        print("         0 calls out of 3 steps and out of 300 are the same sentence")
                        print("         only if the number of offers is stated.")
                        warnings.append(
                            f"{run_root}/{arm}: PULL arm had 0 tool calls across {c['steps']} "
                            f"step(s) ({c['n_spec_shown']} spec(s) shown) -- measures the push only"
                        )
                    elif c["n_frames_served_off_context"] == 0:
                        print("      -> FAIL: tools were called but NO new frame was ever served.")
                        if c["n_serve_empty_not_yet_observed"] >= c["n_query"] > 0:
                            print("         Every one of those came back 'not observed yet' -- the")
                            print("         Planner is asking BEFORE the frames exist. That is a")
                            print("         TIMING difference, not a filter problem: the address is")
                            print("         minted when the primitive is planned, and the evidence")
                            print("         appears over the following MEMEXP_LOOKAHEAD steps.")
                            print("         Check whether the Planner ever re-asks, and whether the")
                            print("         pushed block's 'frames from step N onward' is being read.")
                        else:
                            print("         Every dereference returned text the Planner could")
                            print("         already produce. The pull moved no pixels, so this arm")
                            print("         is equivalent to the push-only control.")
                        problems.append(
                            f"{run_root}/{arm}: PULL arm called {c['tool_calls']} tools and "
                            "served 0 off-context frames"
                        )
                    else:
                        print(f"      -> PASS: the agent pulled "
                              f"{c['n_frames_served_off_context']} frame(s) it could not see")

                    if c["loop_errors"]:
                        print(f"      -> FAIL: {c['loop_errors']} loop error(s); first: "
                              f"{str(c['errors'][0])[:160] if c['errors'] else 'n/a'}")
                        problems.append(f"{run_root}/{arm}: PULL arm loop raised {c['loop_errors']}x")
                    if c["cap_unresolved"]:
                        print(f"      -> FAIL: the model kept calling tools past the cap "
                              f"{c['cap_unresolved']}x. Either MEMEXP_PULL_MAX_ROUNDS is too")
                        print("         small or a tool result is not answering.")
                        problems.append(f"{run_root}/{arm}: PULL arm cap unresolvable "
                                        f"{c['cap_unresolved']}x")
                    if c["n_query_miss"] > 0:
                        print(f"      note: {c['n_query_miss']} queried address(es) did not match "
                              "anything. A high rate means the push is naming addresses in a form")
                        print("            the Planner cannot reuse.")

            # ---- the memory-content correction (all arms) -----------------------------------
            # Reported for EVERY arm, including the baseline. For an uncorrected arm the expected
            # shape is "no report at all", and that absence is itself evidence: it is how the
            # baseline's freedom from this experiment's patches is shown at runtime rather than
            # asserted. `n_render == 0` on a corrected arm is the failure this whole section
            # exists to catch -- GATE 0c proves the corrected functions work, but only this
            # counter says whether the evaluator ever called them.
            mfix_on = str(env.get("MEMEXP_MEMFIX_ENABLE", "")).strip() in {"1", "true", "yes", "on"}
            recorded = read_resolved_env(arm_dir)
            if recorded is None:
                # No artifact to consult: fall back to what the arm file declares today, and say so.
                mfix_in_run = mfix_on
                provenance = "arm file (this run wrote no resolved_env.txt)"
            else:
                mfix_in_run = str(recorded.get("MEMEXP_MEMFIX_ENABLE", "")).strip() in {
                    "1", "true", "yes", "on",
                }
                provenance = "the run's own resolved_env.txt"
            print("    [memory-content correction] memexp_memfix_report.json")
            mfix = find_memfix_reports(arm_dir)
            m = census_memfix(mfix)
            if not mfix:
                if mfix_in_run:
                    print(f"      !! this run declared MEMEXP_MEMFIX_ENABLE ({provenance}) but no")
                    print("         counter was written. GATE 0c proved the corrected functions")
                    print("         work, so this is about REACHING them: either the patches never")
                    print("         installed or no memory was ever rendered. Check the stderr for")
                    print("         a '[memexp] memfix_install_error' line.")
                    problems.append(
                        f"{run_root}/{arm}: corrected arm wrote no memfix report "
                        "(the correction may never have been reached)"
                    )
                elif mfix_on:
                    print("      (none) -- this run predates the correction, though the arm file")
                    print("               declares it now. Not a fault in the run; re-run the job")
                    print("               to obtain a corrected result.")
                else:
                    print("      (none) -- expected: this arm injects no corrected memory")
            else:
                print(f"      reports                     = {m['reports']}")
                print(f"      memory renders              = {m['n_render']}")
                print(f"      guesses demoted from evidence = {m['n_guess_demoted']}")
                print(f"      duplicate rows collapsed    = {m['n_dup_dropped']}")
                print(f"      max context length          = {m['max_ctx_len']}"
                      f"  (last-window max: {m['max_len_history']})")
                # A zero render count has two opposite causes, and conflating them makes this
                # check fire on a CORRECTLY configured run.
                #
                # `memexp_memfix` only ever edits CHANNEL A -- the harness's textual evidence
                # block. It has no other input. So if the run declared channel A OFF
                # (`HARNESS_VLM_CONTEXT=0`), the correction has nothing to correct and rendering
                # zero times is the CONFIGURED outcome, not a silent failure. This became reachable
                # when the treatment arms stopped pushing channel A: the arms that carry the
                # correction are exactly the arms that now declare it off, so the old
                # unconditional FAIL would have marked every correctly-configured run as broken.
                #
                # The failure this check exists to catch is the OTHER case: channel A declared ON,
                # the correction installed, and no text ever went through it. Read the declaration
                # from the run's own resolved_env.txt rather than from the arm file, so an archived
                # job is judged by what it actually ran with.
                channel_a_declared = None
                if recorded is not None:
                    channel_a_declared = str(recorded.get("HARNESS_VLM_CONTEXT", "")).strip()
                channel_a_off = channel_a_declared is not None and channel_a_declared in {
                    "0", "false", "no", "off", "",
                }
                if m["n_render"] == 0 and channel_a_off:
                    print(f"      -> OK: channel A declared OFF in this run "
                          f"(HARNESS_VLM_CONTEXT={channel_a_declared!r}). The correction only edits")
                    print("         channel A text, so zero renders is the configured outcome, and")
                    print("         this arm's score is not a statement about corrected memory.")
                elif m["n_render"] == 0:
                    print("      -> FAIL: the correction installed but rendered NOTHING. This arm's")
                    print("         score is therefore a statement about the ORIGINAL memory text,")
                    print("         not the corrected one. Do not report it as a corrected arm.")
                    problems.append(f"{run_root}/{arm}: corrected arm rendered memory 0 times")
                else:
                    if m["max_ctx_len"] > 2000:
                        print("      -> WARN: a context this long suggests accumulation is back.")
                        warnings.append(
                            f"{run_root}/{arm}: corrected context reached {m['max_ctx_len']} chars"
                        )
                    if m["n_guess_demoted"] == 0 and m["n_dup_dropped"] == 0:
                        print("      note: no guesses were demoted and no duplicates collapsed. On a")
                        print("            clean episode that is legitimate; it is reported rather")
                        print("            than treated as a fault, because a silent no-op and a")
                        print("            clean episode otherwise look the same.")
                    else:
                        print("      -> PASS: the correction had work to do and did it")

            # ---- what code produced this -----------------------------------------------------
            # Printed for every arm because its purpose is COMPARISON: two runs with different
            # fingerprints are not the same experiment, whatever their arm names say. The arms
            # replace parts of the harness at runtime, so an upstream edit to a patched file would
            # otherwise change the meaning of an archived score silently.
            prov = read_code_provenance(arm_dir)
            print("    [code provenance] code_provenance.json")
            if prov is None:
                print("      (none) -- this run predates the recorder, so the revision that")
                print("               produced these numbers is UNKNOWN and the run is not")
                print("               reproducible from its own artifacts.")
                warnings.append(f"{run_root}/{arm}: no code_provenance.json (revision unknown)")
            else:
                print(f"      arm             = {prov.get('arm')}")
                print(f"      git             = {prov.get('git_describe')}")
                print(f"      files hashed    = {len(prov.get('file_sha256') or {})}")
                print(f"      fingerprint     = {short_fingerprint(prov)}")
                miss = prov.get("missing_files") or []
                if miss:
                    print(f"      -> FAIL: {len(miss)} patch target(s)/input(s) were MISSING when")
                    print("         this arm ran, so any arm that patches them silently behaved")
                    print("         as its baseline. First: " + str(miss[:3]))
                    problems.append(
                        f"{run_root}/{arm}: {len(miss)} provenance file(s) missing at run time"
                    )

            # ---- scores -------------------------------------------------------------------
            print("    [scores]")
            if rows:
                vals = [float(r.get("stage_score_pct") or 0.0) for r in rows]
                print(f"      tasks = {len(vals)}   macro stage_score_pct = {sum(vals)/len(vals):.2f}")
                for r in sorted(rows, key=lambda x: int(x.get("task_id") or 0)):
                    print(f"        task {int(r.get('task_id') or 0):>3}  "
                          f"{float(r.get('stage_score_pct') or 0.0):>6.1f}")
            if len(vals) < 26:
                print(f"      !! PARTIAL: {len(vals)}/26 scored; not a protocol number")
                # A DECLARED SUBSET IS NOT A TRUNCATED RUN, and the census used to conflate them:
                # any count below 26 went on the problem list, so this experiment's own documented
                # subset mode (`TASKS_JSON_SCOPE`) reported itself as a failure and the runner
                # exited non-zero on a run that had completed exactly what it was asked to do
                # (job 594675: RC=4 with all 50 declared episodes scored).
                #
                # The distinction that matters is "fewer than DECLARED", not "fewer than 26", and
                # only the frozen environment knows the declaration -- `resolved_env.txt` is
                # written from the live environment AFTER the scope override applies, so it is the
                # one artifact that can tell a narrowing from a truncation.
                _scope = declared_task_scope(arm_dir)
                if _scope is not None and len(vals) >= _scope:
                    print(f"         (all {_scope} task(s) this arm DECLARED were scored, so this "
                          f"is a declared subset -- the number is still NOT a protocol number)")
                else:
                    problems.append(f"{run_root}/{arm}: partial run ({len(vals)}/26)")
            else:
                print("      !! no summary.json -- the run did not finish or scored nothing")
                problems.append(f"{run_root}/{arm}: no summary.json")
            if macros:
                print(f"      aggregate.json: {json.dumps(macros, ensure_ascii=False)[:240]}")

    print("\n" + "=" * 78)
    for v in voids:
        print(f"  VOID: {v}")
    for w in warnings:
        print(f"  WARN: {w}")
    if problems:
        print(f"census found {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("=" * 78)
        return 4
    print("census PASSED: each arm's memory channels are in their designed state.")
    if voids:
        print("  (with the VOID entries above: those arms cannot speak to image memory.)")
    if warnings:
        print("  (with the WARN entries above: read those arms' numbers with the stated caveat.)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
