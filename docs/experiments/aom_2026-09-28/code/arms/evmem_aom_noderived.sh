#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: EVMEM-AOM-NODERIVED  (single-variable ablation of the DERIVED mode)
#
# WHAT DIFFERS FROM evmem_aom
#   Exactly one knob: MEMEXP_AOM_DERIVED=0. The obligation graph, the arbitration law, the board
#   and the graph-derived gate are all unchanged, so a delta against `evmem_aom` isolates the
#   contribution of the DERIVED mode alone.
#
# WHY THIS ABLATION EXISTS
#   DERIVED is the only genuinely new element in AOM -- ACTUAL and PERCEPTUAL are the unification
#   of two behaviours that already exist (GPM's stage predicates and ER's `contents:` addresses),
#   so on those alone AOM is expected to reproduce GPM. If AOM beats GPM anywhere, the claim is
#   that it is because the ledger can represent PROGRESS ("poured 1 of 2") as a live state
#   variable. This arm is what makes that claim falsifiable: if `evmem_aom` and
#   `evmem_aom_noderived` score the same, the DERIVED mode bought nothing and the claim is false.
#
# THE DECLARED DIFF IS THE SAME AS evmem_aom's, AND THAT IS CORRECT
#   `MEMEXP_AOM_DERIVED` is a `MEMEXP_` key, and CHECK 1's structural comparison drops that
#   prefix by convention. So the two arms are declared identical against `nomem` while differing
#   from EACH OTHER on the one knob that matters. That asymmetry is not a loophole: CHECK 1 exists
#   to catch UNDECLARED leakage out of the baseline, and the arm-to-arm contrast here is declared
#   in this file, in `MEMEXP_ARM_CLASS`, and in the arm name the census reads back.
# =========================================================================================

: "${ROOT:?arms/evmem_aom_noderived.sh requires ROOT to be exported}"

# NOTE ON ORDERING -- this file got it wrong first, and the preflight is what caught it.
# `evmem_aom.sh` sources `nomem.sh`, whose purge loop unsets EVERY `MEMEXP_` variable (it matches
# the bare prefix, so `MEMEXP_AOM_DERIVED` is covered). Exporting the ablation knob BEFORE the
# source therefore had it deleted, `evmem_aom.sh`'s `${MEMEXP_AOM_DERIVED:-1}` default then set it
# back to 1, and this arm ran as the FULL arm while calling itself an ablation -- a silent
# single-variable ablation with no variable, which is the exact class of defect the preflight
# exists for.
# shellcheck source=evmem_aom.sh
source "$(dirname "${BASH_SOURCE[0]}")/evmem_aom.sh"

# Set AFTER the source, so the parent's purge cannot reach it and its `${...:-1}` default has
# already been applied. Same for the self-identification, which the parent sets unconditionally.
export MEMEXP_AOM_DERIVED=0
export MEMEXP_ARM_NAME="evmem_aom_noderived"
export MEMEXP_ARM_CLASS="ablation_aom_without_derived_mode"
export MEMEXP_EXPECTED_DIFF="PYTHONPATH PMH_REDACT_STAGE"
