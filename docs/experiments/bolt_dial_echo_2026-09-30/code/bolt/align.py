"""Execution-partition ↔ scoring-partition alignment (BOLT-local).

BOLT's structural claim is that the obligation graph is seeded from the BDDL
EXECUTION partition (`primitive_order`) while scored stages come from the
scoring partition.  Those two partitions are equal-sized only for the
one-primitive-per-stage families (the drawer chains, e.g. task 5 and 19).
For the counting-pour family the execution partition is a strict REFINEMENT:
the primitives that merely prepare the target (pick/place the substance that
is later poured *over*) are not scored at all, while a single scored pour
stage is carried by an explicit "1st"/"2nd" primitive.

`harness.stage_mapper.expected_primitive_for_stage` cannot express that.  Its
fuzzy matcher demands 2+ token hits, which no label can achieve against
`02_Pour_One` (its token set is just {pour, one}), so it falls through to a
"first pick-shaped label" fallback and maps a POUR stage onto a PICK
primitive.  Measured on task 8 before this module existed:

    02_Pour_One -> 'pick chocolate'      (wrong: a pick, not a pour)
    03_Pour_Two -> 'pick chocolate'      (wrong: collapsed onto the same node)

BOLT then credited the wrong node and the hard closure dead-locked every node
downstream of it, which is why task 8 scored 0/0/0 and was early-stopped.

This module is deliberately BOLT-local.  Editing the shared mapper would move
every baseline (nomem / AOM / GPM all consume it) and destroy the
single-variable comparison, so the fix lives on BOLT's side of the seam.

Alignment rules, in preference order:

  A1  equal partition sizes → index alignment.  This is the only mapping that
      handles REPEAT stages correctly (`07_Open_Top_Drawer_Again` must map to
      `open top drawer again`, not to the shorter `open top drawer`).

  A2  execution refines scoring (|labels| > |specs|) → order-preserving
      subsequence alignment maximising phase + object agreement.  A scored
      stage may only be carried by a compatible primitive, so a pour stage can
      never land on a pick label.

  A3  any stage still unbridged whose phase does occur among the leftover
      labels takes the earliest such label.  This keeps every scored stage
      reachable without ever inventing a cross-phase bridge.

Anything that stays unbridged is reported as a *coverage hole* rather than
papered over: those nodes are "instrumental" (they prepare a later scored
stage) and the graph closure invariant downgrades any hard edge that depends
on them.
"""
from __future__ import annotations

from typing import Any, Sequence

from .graph import CONTAINER_RE, phase_of

# Leading stage numbering: "01_Lift_Tomato_Sauce" -> "lift tomato sauce"
_LEAD = r"^[0-9]+\s*"
_VERB = (
    r"^(pick|grasp|lift|place|put|insert|drop|pour|open|close|push|pull|"
    r"reach|move|set|tilt|fill|turn)\b"
)
# Ordinals / spatial words are never part of the object identity.
_STOP = (
    r"\b(the|a|an|into|in|onto|on|over|from|to|of|again|final|first|second|third|"
    r"one|two|three|1st|2nd|3rd|top|middle|bottom)\b"
)

_POUR_FAMILY = {"pour", "pour_one", "pour_two"}


def _norm(text: str) -> str:
    return str(text or "").replace("_", " ").strip()


def spec_phase(stage_name: str) -> str:
    """Phase of a scoring-stage name (`01_Lift_Tomato_Sauce` -> grasp)."""
    return phase_of(_norm(stage_name))


def spec_object_tokens(stage_name: str) -> set[str]:
    """Object tokens of a scoring-stage name, with stage numbering/verbs/ordinals gone."""
    import re

    t = _norm(stage_name).lower()
    t = re.sub(_LEAD, "", t)
    t = re.sub(_VERB, "", t).strip()
    t = CONTAINER_RE.sub("", t)
    t = re.sub(_STOP, " ", t)
    return {w for w in re.sub(r"\s+", " ", t).strip().split() if len(w) > 1}


def label_phase(label: str) -> str:
    return phase_of(label)


def label_object_tokens(label: str) -> set[str]:
    from .graph import object_of

    return {w for w in str(object_of(label) or "").split() if len(w) > 1}


def _phase_compatible(sp: str, lp: str) -> bool:
    if sp == lp:
        return True
    # pour_one / pour_two are ordinals of the same action family
    return sp in _POUR_FAMILY and lp in _POUR_FAMILY


def _pair_score(sp: str, s_tokens: set[str], lp: str, l_tokens: set[str]) -> float:
    """> 0 when the pair is usable; <= 0 means "refuse this pair"."""
    if not sp or not lp or not _phase_compatible(sp, lp):
        return -1.0
    base = 1.0 if sp == lp else 0.5          # exact phase beats family-only
    if not s_tokens:                          # stage names no substance (Pour_One)
        return base
    if not l_tokens:
        return base * 0.5
    inter = len(s_tokens & l_tokens)
    if not inter:
        return -1.0
    return base * (inter / len(s_tokens))


def align_exec_to_scored(
    exec_labels: Sequence[str],
    stage_specs: Sequence[Any],
) -> dict[str, str]:
    """Map execution primitive label -> scored stage name.

    Returns only the confident bridges.  Labels missing from the result are
    instrumental primitives that carry no scored stage; callers must treat
    their nodes as unverifiable (see ObligationGraph's closure invariant).
    """
    labels = [str(x).strip() for x in (exec_labels or []) if str(x).strip()]
    scored = [str(getattr(s, "name", "") or "") for s in (stage_specs or [])]
    scored = [s for s in scored if s]
    if not labels or not scored:
        return {}

    # A1 — equal partitions: 1:1 by index (the REPEAT-stage correct mapping).
    if len(labels) == len(scored):
        return {labels[i]: scored[i] for i in range(len(labels))}

    # Degenerate: BDDL enumerated fewer primitives than scored stages.  Keep the
    # historical prefix behaviour instead of inventing a bridge.
    if len(labels) < len(scored):
        return {labels[i]: scored[i] for i in range(len(labels))}

    n, m = len(scored), len(labels)
    s_phase = [spec_phase(s) for s in scored]
    s_tok = [spec_object_tokens(s) for s in scored]
    l_phase = [label_phase(x) for x in labels]
    l_tok = [label_object_tokens(x) for x in labels]

    # A2 — monotone subsequence alignment (DP over "first i stages using labels < j").
    NEG = float("-inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i][j - 1]                       # skip label j-1
            sc = _pair_score(s_phase[i - 1], s_tok[i - 1], l_phase[j - 1], l_tok[j - 1])
            if sc > 0 and dp[i - 1][j - 1] > NEG:
                cand = dp[i - 1][j - 1] + sc
                if cand > best:
                    best = cand
            dp[i][j] = best

    pairs: list[tuple[int, int]] = []
    if dp[n][m] > 0:
        i, j = n, m
        while i > 0 and j > 0:
            sc = _pair_score(s_phase[i - 1], s_tok[i - 1], l_phase[j - 1], l_tok[j - 1])
            if sc > 0 and abs(dp[i][j] - (dp[i - 1][j - 1] + sc)) < 1e-9:
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
            elif abs(dp[i][j] - dp[i][j - 1]) < 1e-9:
                j -= 1
            else:                                     # pragma: no cover - defensive
                i -= 1
        pairs.reverse()

    mapping: dict[str, str] = {labels[j]: scored[i] for i, j in pairs}
    used_labels = {j for _, j in pairs}
    matched_stages = {i for i, _ in pairs}

    # A3 — every unbridged stage whose phase exists among the leftover labels
    # takes the earliest such label.  Never crosses phases.
    for i, stage in enumerate(scored):
        if i in matched_stages:
            continue
        for j in range(m):
            if j in used_labels:
                continue
            if _pair_score(s_phase[i], s_tok[i], l_phase[j], l_tok[j]) > 0:
                mapping[labels[j]] = stage
                used_labels.add(j)
                break
    return mapping


def describe_alignment(exec_labels: Sequence[str], stage_specs: Sequence[Any]) -> dict[str, Any]:
    """Compact diagnostics for the per-episode report."""
    mapping = align_exec_to_scored(exec_labels, stage_specs)
    scored = [str(getattr(s, "name", "") or "") for s in (stage_specs or [])]
    carried = set(mapping.values())
    return {
        "rule": (
            "A1" if len(list(exec_labels or [])) == len(scored)
            else "A2/A3"
        ),
        "n_exec": len(list(exec_labels or [])),
        "n_scored": len(scored),
        "bridge": dict(mapping),
        "instrumental": [x for x in (exec_labels or []) if x not in mapping],
        "unbridged_stages": [s for s in scored if s and s not in carried],
    }
