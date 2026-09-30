#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------
# Shared helper -- NOT an arm. `validate_arm.py` skips files whose stem starts with "_", and the
# runner only sources the names it is given, so this file is never treated as an experiment.
#
# It exists so that `pushmem` and `pullmem` install the SAME corrected memory path. Two arms that
# inject memory must differ only in their declared channels; if only one of them carried the
# correction, the difference between them would be a code difference wearing the costume of a
# push-vs-pull result.
#
# WHY PYTHONPATH RATHER THAN AN EDIT
#   `pysite/sitecustomize.py` is imported automatically at interpreter start when its directory is
#   on PYTHONPATH, so the correction installs itself without any shared file being modified. The
#   rejected alternatives are listed in that file's docstring.
#
# ORDER MATTERS
#   Every arm sources `nomem.sh` first, and `nomem.sh` deliberately STRIPS this directory from
#   PYTHONPATH so the baseline cannot inherit a hook from a previously-run arm. So this helper
#   must be sourced AFTER `nomem.sh`, and it re-adds the directory. Both arms source this file
#   after their `nomem.sh` line for that reason.
# -----------------------------------------------------------------------------------------
export MEMEXP_DIR="${ROOT}/experiments/mem_efficacy"

# Idempotent prepend: an arm may be sourced more than once in one shell (the runner sources one
# arm per evaluation in the same shell), and a duplicated entry is harmless but noisy.
_memexp_pysite="${MEMEXP_DIR}/pysite"
case ":${PYTHONPATH:-}:" in
  *":${_memexp_pysite}:"*) ;;
  *) export PYTHONPATH="${_memexp_pysite}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
unset _memexp_pysite

# --- the memory-content correction (memexp_memfix) ----------------------------------------
#   On for EVERY arm that injects harness memory. It corrects three content defects measured in
#   job 590799; see `memexp_memfix.py` for each one, the fixture that reproduces it, and the
#   falsifier. Its effect is bounded: it can only change the WORDS the Planner is shown, never
#   the task set, the Planner, the VLA, or which channels are on.
export MEMEXP_MEMFIX_ENABLE=1

# Where the correction writes its own counters, so the census can tell "the corrected path ran"
# apart from "the correction silently failed to install" -- the two look identical in a score.
export MEMEXP_MEMFIX_REPORT="${MEMEXP_MEMFIX_REPORT:-${OUT_ROOT:-/tmp}/memexp_memfix_report.json}"
