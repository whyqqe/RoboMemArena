"""AOM: Arbitrated Obligation Memory.

WHY THIS MODULE EXISTS
======================
`memexp_evmem` (EvMem-GPM) and `memexp_er` (Evidence Retrieval) are two answers to one question:
*when should the harness touch the Planner's context, and with what?* They answer it with two
different type systems, two schedulers, and three prompt blocks that each re-decide policy:

  GPM  -- scheduler, no unified proposition type. `Claim` exists only for a stage.
          `decide_mode` gates evidence on behavioural error (`attempts`/`stall`/attractor).
  ER   -- proposition types, no scheduler. `_ADDRESS_KINDS` says HOW an address is answered, but
          *when* to ask is delegated to the model, which the file itself measures failing 85% of
          the time ("PMH ... 0 searches in it").

The design argument of this module is that those are not two mechanisms. They are two DEGENERATE
POINTS of one law, and the reason a bolt-together fails is that it doubles the ledgers, the
schedulers and the prompt surface -- the failure mode PIC-MEM hit in all four of its versions.

THE LAW
=======
Every memory demand is an OBLIGATION: a proposition that is either pending or settled.

    o = (id, kind, MODE, status, predicate, evidence, tension)

MODE answers "does the world change by a physical action", and it is orthogonal to whether the
obligation ALSO needs a look. `_ADDRESS_KINDS` in `memexp_er` and `is_open_close_stage` in
`memexp_evmem` were two incomplete cuts of the same distinction; collapsing them into MODE alone
made open/close stages (which the robot must physically perform) starve the `act` branch. AOM
keeps both axes:

    ACTUAL      its truth moves by ACTING            (lift, pour, place, open, close)
    PERCEPTUAL  its truth moves by LOOKING alone     (pure inspect / observe / contents)
    DERIVED     its truth moves by RECOMPUTING from  (progress: "poured 1 of 2")
                the obligations already SETTLED
    needs_look  orthogonal flag: establish evidence before / while acting
                (open/close and every PERCEPTUAL)

DERIVED is not a third memory. It is the one thing neither existing mode can express, and its
absence is measurable: `extra_pour_detected` is 0 on every archived task-8 arm, because
`predicate_scaffold` renders "pour count 2" as STATIC TEXT counted once from the stage graph
rather than as a live state variable. A counting task's central uncertainty -- how many pours have
actually landed -- is therefore unrepresentable. PERCEPTUAL cannot cover it either: one frame
cannot answer "how many times", since after two pours the pan looks the same.

The status lattice is likewise one lattice. GPM's `UNVERIFIED/OPEN_Q/SEEN/VERIFIED` and ER's
`UNKNOWN/PENDING/DETERMINED` both embed into:

    OPEN  --act/retrieve-->  PROBING  --evidence-->  SEEN  --predicate-->  SETTLED

`SEEN` means evidence arrived; `SETTLED` means the predicate holds. GPM's comment "SEEN ≠ done" is
a sentence in a prompt there; here it is a state in a type.

THE ARBITRATION LAW (single, replaces both schedulers)
=====================================================
With `sat_act` = consecutive effort without movement on the action channel and `sat_ret` the same
on the retrieval channel:

    if SETTLED                             -> advance
    if needs_look and no evidence yet      -> retrieve   (establish evidence once)
    if not DERIVED and not sat_act         -> act
    if DERIVED                             -> derive
    if not sat_ret                         -> retrieve
    else                                   -> stagnant

GPM's `decide_mode` is this law at {ACTUAL} x {retrieve = harness auto-look}:
`attempts >= GATE_ATTEMPTS or stall >= GATE_STALL` IS `sat_act`. `pullmem_er` is this law with the
`act` branch removed (retrieval is agent-chosen, so `sat_act` does not exist) and a full-episode
store. Neither is a separate mechanism.

THE SAFETY FLOOR (F)
====================
  Whenever GPM would auto-look, AOM also auto-looks.

`sat_act` uses GPM's own gate constants, and `stagnant` is an addition to the BOARD, not a
subtraction from the evidence channel -- STAGNANT still retrieves. This is the invariant with
teeth: it is what stops AOM from repeating PIC-MEM v3, where withholding evidence on exhaustion
cut the cadence from 8-15 frames/episode to 4 and scored 27.8 where GPM scored 66.7.

ADMISSIBILITY IS DERIVED, NOT PATTERN-MATCHED
=============================================
GPM's C-gate is a table of regexes (`_MICROWAVE_RX`, `_POUR_RX`, `_SECOND_POUR_RX`...). Its own
comment names the defect: "That is a memory wired to the tasks in front of it." In AOM a primitive
is admissible iff the obligation it would advance (a) exists in the graph, (b) is in the active
set, and (c) has all dependencies SETTLED. Each archived regex rule falls out:

  "microwave forbidden"      -> microwave is not in the graph
  "pour before lift"         -> pour#1 depends on lift, not SETTLED
  "stage label pasted"       -> instantiates no ACTUAL obligation
  "second pour before #1"    -> pour#2 depends on pour#1, not SETTLED
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The proven primitives are REUSED, not re-derived. `predicate_scaffold`, `admissible_templates`,
# `extract_primitive`, `parse_tool_call`, `merge_keyframe_positions` and `looks_like_label_primitive`
# are byte-identical behaviour shared with GPM and PIC by construction -- otherwise an AOM-vs-GPM
# difference could be an artefact of a paraphrased template rather than of the arbitration.
import memexp_evmem as G

DIGEST_CAP = int(os.environ.get("MEMEXP_AOM_DIGEST_CAP", "2400") or 2400)
LOOKAHEAD = int(os.environ.get("MEMEXP_AOM_LOOKAHEAD", "4") or 4)
MAX_ROUNDS = int(os.environ.get("MEMEXP_AOM_MAX_ROUNDS", "2") or 2)
EVIDENCE_CAP = int(os.environ.get("MEMEXP_AOM_EVIDENCE_CAP", "6") or 6)
# A* and the stall gate default to GPM's, so the two arms' evidence cadence is comparable by
# construction rather than by assertion.
GATE_ATTEMPTS = int(os.environ.get("MEMEXP_AOM_GATE_ATTEMPTS", "3") or 3)
GATE_STALL = int(os.environ.get("MEMEXP_AOM_GATE_STALL", "3") or 3)
# R*: how many fruitless retrievals on one obligation before the board declares STAGNANT. GPM has
# no equivalent counter -- it never leaves evidence mode -- so this is pure addition, and it is
# bounded by floor F: STAGNANT still retrieves.
RET_MAX = int(os.environ.get("MEMEXP_AOM_RET_MAX", "6") or 6)
# After this many consecutive stagnant verdicts on one obligation, re-arm the act channel so
# "change the strategy" is actionable. Without re-arm, sat_act stays true forever and stagnant
# absorbs the rest of the episode (v2 t19: act≈2 on failures vs v1 act≈5 on successes).
STAG_ACT_REARM = int(os.environ.get("MEMEXP_AOM_STAG_ACT_REARM", "3") or 3)
ATTRACTOR_REPEAT = int(os.environ.get("MEMEXP_AOM_ATTRACTOR_REPEAT", "2") or 2)

MODE_ACTUAL = "ACTUAL"
MODE_PERCEPTUAL = "PERCEPTUAL"
MODE_DERIVED = "DERIVED"

ST_OPEN = "OPEN"
ST_PROBING = "PROBING"
ST_SEEN = "SEEN"
ST_SETTLED = "SETTLED"

FINISH_FORM = G.FINISH_FORM
FORCE_FINAL = G.FORCE_FINAL
FORCE_NATURAL = G.FORCE_NATURAL

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}

# Intent tokens. These are NOT the gate -- they only NAME which obligation a primitive is claiming
# to advance; whether that claim is admissible is decided by the graph, in `admissibility()`.
_INTENT_LIFT = re.compile(r"\b(grasp|lift|pick up|pick|reach)\b", re.I)
_INTENT_POUR = re.compile(r"\b(pour|tilt)\b", re.I)
_INTENT_SECOND = re.compile(
    r"\b(second|2nd|twice|two times|again)\b.*\b(pour|sauce)\b|"
    r"\b(pour|sauce)\b.*\b(second|2nd|twice)\b", re.I)
_INTENT_OPEN = re.compile(r"\b(open|inspect|look inside|check inside)\b", re.I)
_INTENT_CLOSE = re.compile(r"\b(close)\b", re.I)
_INTENT_PLACE = re.compile(r"\b(put|place|insert|drop)\b", re.I)
_INTENT_MICROWAVE = re.compile(r"\bmicrowave\b", re.I)


def enabled() -> bool:
    return str(os.environ.get("MEMEXP_AOM", "")).strip().lower() in _TRUTHY


def derived_on() -> bool:
    return str(os.environ.get("MEMEXP_AOM_DERIVED", "1")).strip().lower() in _TRUTHY


def graph_gate_on() -> bool:
    """The obligation-graph gate. Off is the ablation that measures the gate's own contribution."""
    return str(os.environ.get("MEMEXP_AOM_GRAPH_GATE", "1")).strip().lower() in _TRUTHY


def floor_on() -> bool:
    """Safety floor F. Off is the ablation that reproduces PIC-MEM v3's starved cadence, so the
    defect and the fix can be measured against each other."""
    return str(os.environ.get("MEMEXP_AOM_FLOOR", "1")).strip().lower() in _TRUTHY


# --------------------------------------------------------------------------------------------
# The single type.
# --------------------------------------------------------------------------------------------
@dataclass
class Obligation:
    oid: str
    kind: str                       # short token: "lift" | "pour_one" | "place#2" | "pick#1" | ...
    mode: str                       # ACTUAL | PERCEPTUAL | DERIVED
    predicate: str                  # the scorer-aligned NEED text (or the derived statement)
    label: str                      # public ordinal, no container identity
    status: str = ST_OPEN
    stage_name: str = ""
    exec_label: str = ""            # the EXECUTION step this obligation is (BDDL primitive_order)
    settles_with: str = ""          # derive: settle when this scored stage settles (pick steps)
    address: str = ""               # PERCEPTUAL: the address a look would answer
    value: str = ""                 # DERIVED: the live computed value, e.g. "1/2"
    needs_look: bool = False        # orthogonal to MODE: establish evidence before acting
    deps: list[str] = field(default_factory=list)
    counted: list[str] = field(default_factory=list)   # DERIVED: the oids it summarises
    target: int = 0                                     # DERIVED: settle at this count
    frames: list[int] = field(default_factory=list)
    offered_ever: set[int] = field(default_factory=set)  # frames already shown for this oid
    tau_act: int = 0                # consecutive ACT attempts without movement
    tau_ret: int = 0                # consecutive retrievals without an informative status move
    n_stag_hits: int = 0            # consecutive stagnant verdicts since last act re-arm
    n_new_evidence: int = 0
    seq: int = 0
    last_step: int = 0
    settled_step: int = -1
    stagnant: bool = False
    reasks: int = 0
    _rearmed: bool = False          # one-shot board nudge after act re-arm

    def is_open(self) -> bool:
        return self.status != ST_SETTLED


@dataclass
class Event:
    step: int
    intent: str
    verdict: str
    frames: list[int]
    mode: str = ""
    oid: str = ""


class Ledger:
    """The obligation set and its single arbitration law."""

    def __init__(self) -> None:
        self.obligations: dict[str, Obligation] = {}
        self._order: list[str] = []
        self._next_id = 1
        self.active_oid: str = ""
        self.events: list[Event] = []
        self.evidence_abs: list[int] = []
        self.evidence_meta: list[dict[str, Any]] = []
        self.tool_trace: list[dict[str, Any]] = []
        self.attractor_hist: list[str] = []
        self.force_retrieve = False
        self.pending_pins: list[str] = []
        self._verified_seen: set[str] = set()
        self.last_board: str = ""
        self.last_mode: str = "fluent"
        self.last_verdict: str = "advance"
        self.stage_names: list[str] = []
        self.exec_labels: list[str] = []
        self.name_objects: bool = False
        # counters
        self.n_plan_steps = 0
        self.n_events = 0
        self.n_advance = 0
        self.n_act = 0
        self.n_retrieve = 0
        self.n_derive = 0
        self.n_stagnant = 0
        self.n_fluent = 0
        self.n_evidence_mode_steps = 0
        self.n_auto_looks = 0
        self.n_looks_new_frames = 0
        self.n_looks_zero_gain = 0
        self.n_reask = 0
        self.n_gate_rejects = 0
        self.n_label_rejects = 0
        self.n_offgraph_rejects = 0
        self.n_dep_rejects = 0
        self.n_unmatched = 0
        self.n_board_redacted = 0
        self.n_settled_events = 0
        self.n_evidence_carried = 0
        self.n_derived_recomputes = 0
        self.n_seeded = 0
        self.n_board_chars = 0

    # -- minting ------------------------------------------------------------------------------
    def _mint(self) -> str:
        oid = f"o{self._next_id}"
        self._next_id += 1
        return oid

    def seed(self, stage_names: list[str]) -> int:
        """Build the obligation graph from the task's stage ladder.

        Dependency rule (causal, not ordinal):
          * ACTUAL obligations form a SEQUENCE, EXCEPT same-family place siblings -- three
            independent objects going into the same cabinet are not causal predecessors of each
            other (task 19). Chaining them produced the self-lock `place waits on place`.
          * Pure PERCEPTUAL obligations (inspect/observe) stay INDEPENDENT -- the occlusion case
            the pull arm was built for: three drawers, each observed once, and which one matters
            is not known until all three have been looked at.
          * open/close are ACTUAL (the robot must physically open) with needs_look=True. Treating
            them as PERCEPTUAL starved the act branch (task 5: fluent ratio 1.4%).
        """
        names = [str(n).strip() for n in (stage_names or []) if str(n).strip()]
        if not names:
            return 0
        self.stage_names = names
        pour_total = sum(1 for nm in names if "pour" in nm.lower())
        pour_slots = 0
        # Pre-count families that need a discriminant kind (place#1 / open#2), so the board never
        # renders the self-referential `Not yet: 'place' waits on place.`
        base_kinds = [_base_kind(nm) for nm in names]
        fam_totals: dict[str, int] = {}
        for bk in base_kinds:
            fam = _family_of(bk)
            if fam in {"place", "open", "close"}:
                fam_totals[fam] = fam_totals.get(fam, 0) + 1
        fam_seen: dict[str, int] = {
            fam: sum(1 for o in self._ordered() if _family_of(o.kind) == fam)
            for fam in fam_totals
        }
        minted = 0
        last_actual = ""
        # Resume the ACTUAL chain from the last minted ACTUAL, so an incremental seed keeps the
        # causal edge across the discovery boundary.
        for o in reversed(self._ordered()):
            if o.mode == MODE_ACTUAL:
                last_actual = o.oid
                break
        for i, nm in enumerate(names):
            if any(o.stage_name == nm for o in self.obligations.values()):
                continue
            low = nm.lower()
            # Pure observation -> PERCEPTUAL; everything the robot must physically do -> ACTUAL.
            # needs_look is orthogonal: open/close and every PERCEPTUAL establish evidence first.
            is_pure_look = bool(re.search(r"inspect|observe|contents", low)) and "open" not in low
            mode = MODE_PERCEPTUAL if is_pure_look else MODE_ACTUAL
            needs_look = bool(is_pure_look or G.is_open_close_stage(nm))
            pidx = None
            if "pour" in low:
                pour_slots += 1
                pidx = pour_slots
            bk = base_kinds[i]
            fam = _family_of(bk)
            if fam in fam_totals and fam_totals[fam] > 1:
                fam_seen[fam] = fam_seen.get(fam, 0) + 1
                kind = f"{bk}#{fam_seen[fam]}"
            else:
                kind = bk
            deps: list[str] = []
            if mode == MODE_ACTUAL and last_actual:
                prev = self.obligations[last_actual]
                if _causal_dep(prev.kind, kind):
                    deps = [last_actual]
            oid = self._mint()
            self.obligations[oid] = Obligation(
                oid=oid,
                kind=kind,
                mode=mode,
                predicate=G.predicate_scaffold(nm, pour_index=pidx, pour_total=pour_total or None),
                label=G.public_ordinal(i, len(names)),
                stage_name=nm,
                needs_look=needs_look,
                deps=deps,
                seq=i,
            )
            self._order.append(oid)
            if mode == MODE_ACTUAL:
                last_actual = oid
            minted += 1
        minted += self._mint_derived()
        # Render the DERIVED values immediately. Without this the board's PROGRESS rows are EMPTY
        # until the first settle, which is the first stage boundary -- i.e. the one moment the
        # Planner most needs "where are we" is the one moment the field is blank.
        self.recompute_derived()
        self.n_seeded += minted
        return minted

    def seed_execution(self, exec_labels: list[str], scored_names: list[str],
                       label_to_scored: dict[str, str] | None = None,
                       name_objects: bool = False) -> int:
        """Build the obligation graph from the EXECUTION partition (BDDL `primitive_order`).

        WHY THIS IS THE ROOT FIX, AND NOT A TUNING CHANGE
        -------------------------------------------------
        `seed()` builds from `_task_specs(task_id)`, which is the SCORING partition. On task 19 the
        two partitions differ structurally:

            scoring (_task_specs)   = 3 steps   Place_Tomato_Sauce / Place_Milk / Place_Orange_Juice
            execution (primitive_order) = 6 steps  pick_TS, place_TS, pick_milk, place_milk, ...

        Seeding from the scoring partition therefore minted a graph of three `place` obligations and
        NOT ONE grasp obligation. The board's NEED line was always a place, and its own instruction
        ("Serve only the ACTIVE obligation") then made the Planner emit place primitives 504-615
        times per 10 episodes against 10-15 grasps (measured, t19 vs nomem's 332 grasps / 86
        places). A `place` presupposes a held object, so the commanded action was UNSATISFIABLE and
        no stage could ever settle except by accident. That is why every AOM version landed at or
        below nomem (23.3/16.7/14.8) while AOM WON on task 5 (45.0 vs 13.8), where the two
        partitions coincide.

        The rule this implements: an execution phase that the scoring partition omits is harmless
        when it lies DOWNSTREAM of every scored phase (task 8: the omitted `place` follows the
        pours) and fatal when it is a PRECONDITION (task 19: `pick` precedes every scored `place`).

        Settlement: a scored StageSpec settles the execution step that verifies it. A `pick` step
        has no StageSpec of its own, so it settles DERIVATIONALLY -- when the `place` it feeds
        settles (you cannot have placed what you never picked). No new sensor is required.

        `name_objects` turns OFF the container/object redaction inherited from the occlusion arm
        (`memory_type` containing 'O'). On a TRANSFER task ('T') the object identity is not the
        answer being withheld, it IS the instruction; redacting it produced t19 commands like
        "Place the object into the open container." with no object named at all.
        """
        execs = [str(x).strip() for x in (exec_labels or []) if str(x).strip()]
        scored = [str(x).strip() for x in (scored_names or []) if str(x).strip()]
        if not execs:
            return self.seed(scored)
        self.name_objects = bool(name_objects)
        self.exec_labels = list(execs)
        self.stage_names = list(scored)
        l2s = {str(k).strip(): str(v or "").strip() for k, v in (label_to_scored or {}).items()}

        base = [_base_kind(l) for l in execs]
        fam_totals: dict[str, int] = {}
        for bk in base:
            fam = _family_of(bk)
            if fam in {"place", "open", "close", "pick"}:
                fam_totals[fam] = fam_totals.get(fam, 0) + 1
        fam_seen: dict[str, int] = {}
        # Pair every pick with the place it feeds: settles_with carries the scored name so the
        # harness's own verification settles BOTH steps of the object.
        pair_scored: dict[int, str] = {}
        for i, bk in enumerate(base):
            if bk != "pick":
                continue
            for j in range(i + 1, len(base)):
                if base[j] == "place":
                    pair_scored[i] = l2s.get(execs[j], "")
                    break

        minted = 0
        last_actual = ""
        for i, lab in enumerate(execs):
            bk = base[i]
            fam = _family_of(bk)
            if fam in fam_totals and fam_totals[fam] > 1:
                fam_seen[fam] = fam_seen.get(fam, 0) + 1
                kind = f"{bk}#{fam_seen[fam]}"
            else:
                kind = bk
            scored_for_this = l2s.get(lab, "")
            deps: list[str] = []
            if last_actual:
                prev = self.obligations[last_actual]
                if _causal_dep(prev.kind, kind):
                    deps = [last_actual]
            # A pick step carries the execution label as its stage_name (it is not scored), and
            # settles with the place it feeds. A place step carries the SCORED name so the
            # harness's `verified_stages` settles it directly.
            stage_name = scored_for_this or lab
            settles_with = pair_scored.get(i, "")
            needs_look = (bk == "pick")
            predicate = (naming_need(lab) if self.name_objects else "") \
                or G.predicate_scaffold(stage_name)
            if self.name_objects:
                predicate = predicate.replace(" without naming drawer identity", "")
            oid = self._mint()
            self.obligations[oid] = Obligation(
                oid=oid,
                kind=kind,
                mode=MODE_ACTUAL,
                predicate=predicate,
                label=G.public_ordinal(i, len(execs)),
                stage_name=stage_name,
                exec_label=lab,
                settles_with=settles_with,
                needs_look=needs_look,
                deps=deps,
                seq=i,
            )
            self._order.append(oid)
            last_actual = oid
            minted += 1
        minted += self._mint_derived()
        self.recompute_derived()
        self.n_seeded += minted
        return minted

    def templates_for(self, o: Obligation | None) -> list[str]:
        """Admissible templates for an obligation.

        When object naming is ON the execution label is authoritative: `G.admissible_templates` is
        shared with the task-8 arms and hardcodes "the sauce bottle" for every lift/pick stage,
        which on task 19 would tell the Planner to grasp a sauce bottle while it is trying to pick
        up milk.
        """
        if o is None:
            return []
        if self.name_objects and o.exec_label:
            t = naming_templates(o.exec_label)
            if t:
                return t
        return G.admissible_templates(o.stage_name or o.exec_label)

    def _mint_derived(self) -> int:
        """Mint (or refresh) the DERIVED obligations: progress statements over the obligation set.

        Seed is incremental across stage discovery, so an existing count:* must have its
        counted/target rewritten whenever the ACTUAL set grows -- otherwise the board freezes at
        a stale `0/1` after later stages appear (the task-5 progress lie).
        """
        if not derived_on():
            return 0
        minted = 0
        actuals = [o for o in self._ordered() if o.mode == MODE_ACTUAL]
        families: dict[str, list[str]] = {}
        for o in actuals:
            fam = _family_of(o.kind)
            if fam:
                families.setdefault(fam, []).append(o.oid)
        for fam, oids in families.items():
            existing = next((o for o in self.obligations.values() if o.kind == f"count:{fam}"), None)
            if existing is not None:
                existing.counted = list(oids)
                existing.target = len(oids)
                continue
            if len(oids) < 2:
                continue
            oid = self._mint()
            self.obligations[oid] = Obligation(
                oid=oid,
                kind=f"count:{fam}",
                mode=MODE_DERIVED,
                predicate=f"complete every {fam} step exactly once (no repeats, none skipped)",
                label="progress",
                counted=list(oids),
                target=len(oids),
                deps=[],
                seq=10_000 + len(self._order),
            )
            self._order.append(oid)
            minted += 1
        steps = next((o for o in self.obligations.values() if o.kind == "count:steps"), None)
        if steps is not None:
            steps.counted = [o.oid for o in actuals]
            steps.target = len(actuals)
        elif actuals:
            oid = self._mint()
            self.obligations[oid] = Obligation(
                oid=oid,
                kind="count:steps",
                mode=MODE_DERIVED,
                predicate="complete the remaining steps in order, one at a time",
                label="progress",
                counted=[o.oid for o in actuals],
                target=len(actuals),
                deps=[],
                seq=20_000 + len(self._order),
            )
            self._order.append(oid)
            minted += 1
        return minted

    # -- accessors ----------------------------------------------------------------------------
    def _ordered(self) -> list[Obligation]:
        return [self.obligations[o] for o in self._order if o in self.obligations]

    def active(self) -> Obligation | None:
        """The single obligation the board speaks about.

        Self-healing on purpose: a pointer left on a SETTLED obligation would render a NEED for a
        stage that is already done -- a command the harness's verification can never satisfy, i.e.
        the same class of defect as commanding a place while nothing is held. Any caller that reads
        `active()` therefore gets the true next step, whether or not `advance_active()` ran.
        """
        cur = self.obligations.get(self.active_oid) if self.active_oid else None
        if cur is not None and cur.mode != MODE_DERIVED and cur.is_open():
            return cur
        for o in self._ordered():
            if o.mode != MODE_DERIVED and o.is_open():
                self.active_oid = o.oid
                return o
        return None

    def open_obligations(self) -> list[Obligation]:
        return [o for o in self._ordered() if o.is_open()]

    def fluent_equivalent(self) -> bool:
        """Floor F's trigger: nothing pending means nothing to say, so behaviour is `nomem`'s."""
        return not any(o.is_open() for o in self._ordered() if o.mode != MODE_DERIVED) \
            and not any(o.is_open() for o in self._ordered() if o.mode == MODE_DERIVED)

    def progress(self) -> tuple[int, int]:
        """Settled / total over every non-DERIVED obligation. Counting only ACTUAL lied on tasks
        whose open/close stages were mis-typed as PERCEPTUAL (board showed settled 0/1 while six
        obligations were already SETTLED)."""
        acts = [o for o in self._ordered() if o.mode != MODE_DERIVED]
        return sum(1 for o in acts if o.status == ST_SETTLED), len(acts)

    # -- state transitions --------------------------------------------------------------------
    def set_active_stage(self, name: str, step: int) -> str:
        """Honour the harness's scored-stage hint ONLY when that stage is reachable.

        The harness reports the SCORING partition's active stage. On task 19 that is
        `01_Place_Tomato_Sauce_Cabinet2` from step 0 -- but its execution prerequisite (the grasp)
        is not settled, so pointing ACTIVE there would re-create exactly the defect this fixes: a
        board that demands a place while nothing is held. When the hint is unreachable we keep the
        earliest open EXECUTION step (which is always the true next action).
        """
        nm = str(name or "").strip()
        if not nm:
            return ""
        for o in self._ordered():
            if o.mode != MODE_DERIVED and o.stage_name == nm:
                # `_dep_ready`, not `SETTLED`: the place the hint names is admissible as soon as its
                # grasp has been ATTEMPTED. Requiring the settle here kept ACTIVE pinned to the
                # grasp for the whole episode (job 617408: 719/719 emitted primitives were grasps,
                # because the board never advertised anything else, and the pointer could only move
                # on a settle that only a place could produce).
                if all(self._dep_ready(self.obligations[d], o)
                       for d in o.deps if d in self.obligations):
                    self.active_oid = o.oid
                    o.last_step = int(step)
                    return o.oid
                break
        return self.advance_active()

    def note_verified(self, name: str, step: int) -> None:
        """Settle an obligation. EVERY settle re-runs the DERIVED obligations, because a progress
        statement is by definition a function of the settled set and is stale the moment that set
        changes."""
        nm = str(name or "").strip()
        if not nm:
            return
        changed = False
        for o in self._ordered():
            if o.mode == MODE_DERIVED:
                continue
            # Direct settle by scored stage name, OR derived settle: a pick step settles when the
            # place it feeds settles (you cannot have placed what was never picked).
            if not (o.stage_name == nm or (o.settles_with and o.settles_with == nm)):
                continue
            if o.status != ST_SETTLED:
                changed = True
            o.status = ST_SETTLED
            o.settled_step = int(step)
            o.tau_act = 0
            o.tau_ret = 0
            o.n_stag_hits = 0
            o.stagnant = False
            if nm not in self._verified_seen:
                self._verified_seen.add(nm)
                # PERCEPTUAL settles by looking and wants its frames pinned so the Planner can
                # keep using what it saw; ACTUAL settles by the harness's own predicate.
                if o.mode == MODE_PERCEPTUAL:
                    self.pending_pins.append(o.oid)
        if changed:
            self.n_settled_events += 1
            self.recompute_derived(step)

    def recompute_derived(self, step: int = 0) -> None:
        """The DERIVED channel. Recomputation is idempotent and cannot fail; it has no evidence to
        wait for, which is why `stagnant` is unreachable for these obligations -- there is always
        a next action available (recompute)."""
        for o in self._ordered():
            if o.mode != MODE_DERIVED:
                continue
            self.n_derived_recomputes += 1
            done = sum(1 for cid in o.counted
                       if cid in self.obligations and self.obligations[cid].status == ST_SETTLED)
            o.value = f"{done}/{o.target}"
            if done >= o.target and o.status != ST_SETTLED:
                o.status = ST_SETTLED
                o.settled_step = int(step)

    def note_action(self, primitive: str, step: int, frame_lo: int, frame_hi: int) -> None:
        text = str(primitive or "").strip()
        frames = list(range(int(frame_lo), int(frame_hi) + 1)) if frame_hi >= frame_lo else []
        self.n_plan_steps += 1

        intent = intent_of(text)
        cl = G.attractor_cluster(text)
        self.attractor_hist.append(cl)
        if len(self.attractor_hist) > 8:
            self.attractor_hist = self.attractor_hist[-8:]
        # Attractor repeat is a BEHAVIOURAL error signal, exactly as in GPM: it does not need the
        # Planner to report being stuck. Kept as an ORTHOGONAL trigger so the floor holds even
        # when the stage pointer has not moved and `attempts` has been reset by a settle.
        if len(self.attractor_hist) >= ATTRACTOR_REPEAT:
            tail = self.attractor_hist[-ATTRACTOR_REPEAT:]
            if len(set(tail)) == 1 and tail[0] in {"microwave", "pour", "pour_second"}:
                if tail[0] == "microwave" or (tail[0] in {"pour", "pour_second"} and self._lift_open()):
                    self.force_retrieve = True

        o = self.active()
        if o is None:
            return
        o.last_step = int(step)
        if frames:
            have = set(o.frames)
            o.frames.extend(i for i in frames if i not in have)
        o.tau_act += 1
        if o.status == ST_OPEN:
            o.status = ST_PROBING
        self.events.append(Event(step=int(step), intent=text, verdict="", frames=frames,
                                 mode=o.mode, oid=o.oid))
        self.n_events += 1

    def _lift_open(self) -> bool:
        lifts = [o for o in self._ordered() if o.mode == MODE_ACTUAL and re.search(r"lift|grasp|pick", o.stage_name, re.I)]
        if not lifts:
            return False
        return any(o.is_open() for o in lifts)

    def note_verdict(self, checked: str, passed: bool) -> None:
        if not self.events:
            return
        self.events[-1].verdict = f"{checked} {'passed' if passed else 'failed'}"

    # -- the law -------------------------------------------------------------------------------
    def sat_act(self, o: Obligation, stall: int) -> bool:
        return int(o.tau_act) >= GATE_ATTEMPTS or int(stall) >= GATE_STALL

    def sat_ret(self, o: Obligation) -> bool:
        return int(o.tau_ret) >= RET_MAX

    def arbitrate(self, stall: int) -> tuple[str, str]:
        """THE law. Returns (verdict, oid).

        Floor F is enforced here and not in the renderer: `stagnant` is decided AFTER the retrieve
        branch, so an obligation that is saturated on both channels still retrieves. The verdict
        only changes what the BOARD SAYS, never whether the evidence channel runs.
        """
        o = self.active()
        if o is None or o.status == ST_SETTLED:
            # `advance` is not a no-op: it MOVES THE POINTER. Returning the verdict without moving
            # it would leave the board advertising a settled obligation as ACTIVE forever, which is
            # the "digest frozen" failure PIC-MEM v1 hit.
            nxt = self.advance_active()
            if nxt:
                o = self.obligations[nxt]
            else:
                d = next((x for x in self._ordered() if x.mode == MODE_DERIVED and x.is_open()), None)
                if d is not None:
                    return "derive", d.oid
                return "advance", ""
        # The attractor-repeat signal OVERRIDES the act branch, exactly as GPM's
        # `if self.pending_pin_cids or self.force_evidence: return "evidence"` does. It is the one
        # trigger that fires BEFORE saturation, and putting it after the act branch (as an
        # `and not self.force_retrieve` conjunct) made it unreachable whenever both channels were
        # already saturated -- i.e. precisely when it is most needed.
        if self.force_retrieve:
            return "retrieve", o.oid
        # needs_look: establish evidence once before the act branch is allowed to fire. Without
        # this, open/close (ACTUAL + needs_look) would skip straight to act and never look.
        if o.needs_look and o.status == ST_OPEN and int(o.n_new_evidence) == 0 \
                and int(o.tau_ret) == 0:
            # Bounded single look before acting: establishing evidence is worthwhile, but looping
            # here when no frame is available yet would burn RET_MAX steps per obligation.
            return "retrieve", o.oid
        # Act for every non-DERIVED unsaturated obligation. Restricting this to MODE_ACTUAL was
        # what starved task 5 when open/close were mis-typed as PERCEPTUAL.
        if o.mode != MODE_DERIVED and not self.sat_act(o, stall):
            return "act", o.oid
        if o.mode == MODE_DERIVED:
            return "derive", o.oid
        if not self.sat_ret(o):
            return "retrieve", o.oid
        # Both channels saturated. Stagnant is reachable (tau_ret only clears on status move) but
        # must not be absorbing: v2 t19 spent ~half its steps here with act starved (act≈2 vs
        # v1's act≈5), and the board's "change strategy" line was a lie because sat_act never
        # reset. Re-arm act every STAG_ACT_REARM stagnant hits so a new physical attempt is
        # actually scheduled; raise RET_MAX so this path is rare rather than the default.
        o.stagnant = True
        o.n_stag_hits += 1
        if STAG_ACT_REARM > 0 and o.n_stag_hits >= STAG_ACT_REARM:
            o.n_stag_hits = 0
            o.tau_act = 0
            o.tau_ret = 0  # full cycle: act -> retrieve -> stagnant -> re-arm
            o.stagnant = False
            o._rearmed = True  # one-shot board nudge
            return "act", o.oid
        return "stagnant", o.oid

    def advance_active(self) -> str:
        """Move the active pointer to the next open non-derived obligation. Idempotent."""
        cur = self.obligations.get(self.active_oid)
        if cur is not None and cur.is_open() and cur.mode != MODE_DERIVED:
            return cur.oid
        for o in self._ordered():
            if o.mode != MODE_DERIVED and o.is_open():
                self.active_oid = o.oid
                return o.oid
        self.active_oid = ""
        return ""

    def consume_pins(self) -> list[str]:
        out = list(self.pending_pins)
        self.pending_pins = []
        return out

    def clear_force_retrieve(self) -> None:
        self.force_retrieve = False

    # -- retrieval -----------------------------------------------------------------------------
    def look(self, oid: str, on_context: set[int], store: dict[int, Any], *, auto: bool = False
             ) -> tuple[str, list[int]]:
        o = self.obligations.get(str(oid or "").strip())
        if o is None:
            return f"look({oid!r}): unknown obligation.", []
        if o.status == ST_SETTLED and not auto:
            o.reasks += 1
            self.n_reask += 1
            return f"look({o.oid}) -> SETTLED. Do not re-ask a settled obligation.", []
        offered = _available(o.frames, on_context, store)
        # Display-only fallback: never counts as novel evidence and must NOT clear tau_ret.
        # The previous `cand[-2:]` path made offered non-empty whenever any frame remained in
        # store, which reset tau_ret on every look and made `stagnant` structurally unreachable.
        display = list(offered)
        if not display and auto and o.frames:
            cand = [int(i) for i in o.frames if int(i) in store]
            display = [i for i in cand if i not in on_context][-LOOKAHEAD:] or cand[-min(2, len(cand)):]
        novel = [i for i in offered if i not in o.offered_ever]
        status_before = o.status
        o.tau_ret += 1
        if offered:
            if o.status == ST_OPEN:
                o.status = ST_PROBING
            if o.status != ST_SETTLED:
                o.status = ST_SEEN
            if novel:
                o.n_new_evidence += 1
                o.offered_ever.update(novel)
                self.n_looks_new_frames += 1
                self.remember(offered, oid=o.oid)
            else:
                self.n_looks_zero_gain += 1
            # Reset tau_ret ONLY on an informative status transition (first evidence). Subsequent
            # looks while stuck at SEEN keep incrementing, so sat_ret can fire and stagnant is
            # reachable -- the single prompt-level interrupt for the lift attractor.
            if status_before != o.status:
                o.tau_ret = 0
        else:
            self.n_looks_zero_gain += 1
        if auto:
            self.n_auto_looks += 1
        shown = offered or display
        lines = [
            f"{'auto-' if auto else ''}look({o.oid}) {o.kind}  mode={o.mode}  status={o.status}",
            f"frames: {shown or 'none'}",
            "SEEN means evidence arrived. SETTLED means the predicate holds. They are different states.",
        ]
        if o.stagnant:
            lines.append("This obligation is STAGNANT: the last attempts produced no movement and "
                         "no new evidence. Change the STRATEGY, not the wording.")
        return "\n".join(lines), shown

    def remember(self, frames: list[int], *, oid: str = "") -> None:
        if not frames:
            return
        have = set(self.evidence_abs)
        for i in frames:
            ii = int(i)
            if ii in have:
                continue
            self.evidence_abs.append(ii)
            self.evidence_meta.append({"abs": ii, "oid": oid})
            have.add(ii)
            self.n_evidence_carried += 1
        while len(self.evidence_abs) > EVIDENCE_CAP:
            self.evidence_abs.pop(0)
            if self.evidence_meta:
                self.evidence_meta.pop(0)

    def take_evidence(self) -> list[tuple[int, str]]:
        meta = {int(m.get("abs", -1)): str(m.get("oid", "")) for m in self.evidence_meta}
        return [(int(a), meta.get(int(a), "")) for a in self.evidence_abs]

    def ls(self, path: str = "") -> str:
        return "obligations/\n  " + "\n  ".join(
            f"{o.oid} {o.kind} [{o.mode}/{o.status}]" for o in self._ordered()) if self._order else "(none)"

    def cat(self, path: str, on_context: set[int], store: dict[int, Any]) -> tuple[str, list[int]]:
        p = str(path or "").strip()
        if p in self.obligations or (p.startswith("o") and p[1:].isdigit()):
            return self.look(p, on_context, store)
        if p in {"board", "BOARD.md"}:
            return self.board(mode=self.last_mode), []
        if p in {"events.jsonl", "events"}:
            return "\n".join(json.dumps({"step": e.step, "intent": e.intent, "verdict": e.verdict})
                             for e in self.events) or "(no events)", []
        return f"cat: {p!r} not found", []

    def grep(self, pattern: str) -> str:
        pat = str(pattern or "").strip()
        if not pat:
            return "grep needs a pattern"
        rx = re.compile(re.escape(pat), re.IGNORECASE)
        hits = [f"step={e.step} {e.intent} {e.verdict}" for e in self.events
                if rx.search(f"{e.intent} {e.verdict}")]
        if not hits:
            return f"grep({pat!r}): no hits"
        return f"grep({pat!r}) {len(hits)} hit(s):\n" + "\n".join(hits[:20])

    def known_tools(self) -> set[str]:
        return {"look", "ls", "cat", "grep"}

    def dispatch(self, name: str, args: dict[str, Any], *, on_context: set[int],
                 store: dict[int, Any]) -> tuple[str, list[int]]:
        n = str(name or "").strip().lower()
        if n not in self.known_tools():
            return f"unknown tool {name!r}", []
        if n == "look":
            return self.look(str(args.get("id", args.get("oid", ""))).strip(), on_context, store)
        if n == "ls":
            return self.ls(str(args.get("path", "") or "")), []
        if n == "cat":
            return self.cat(str(args.get("path", args.get("id", "")) or ""), on_context, store)
        if n == "grep":
            return self.grep(str(args.get("pattern", "") or "")), []
        return f"unknown tool {name!r}", []

    # -- admissibility, DERIVED from the graph --------------------------------------------------
    def obligations_for(self, primitive: str) -> list[Obligation]:
        """Which obligation(s) a primitive claims to advance. At most ONE per call.

        Naming only -- admissibility is decided by `admissibility()`, which asks the graph.
        Returning every same-family obligation (the old place branch) made the dep check a
        universal quantifier: any place primitive was rejected because place#3's unmet dep on
        place#2, including the primitive that would have settled place#2. Single-target selection
        (first open of the family, else last) is the pour rule, applied uniformly.
        """
        p = str(primitive or "")
        intent = intent_of(p)

        def _stem(k: str) -> str:
            return k.split("#", 1)[0]

        def _match(o: Obligation) -> bool:
            if o.mode == MODE_DERIVED:
                return False
            k = _stem(o.kind)
            if intent == "lift" and k in {"lift", "grasp", "pick"}:
                return True
            if intent in {"pour", "pour_second"} and k.startswith("pour"):
                return True
            if intent == "open" and k.startswith("open"):
                return True
            if intent == "close" and k.startswith("close"):
                return True
            if intent == "place" and k.startswith("place"):
                return True
            return False

        cands = [o for o in self._ordered() if _match(o)]
        if not cands:
            return []
        if intent == "pour_second":
            return cands[-1:]
        open_ones = [x for x in cands if x.is_open()]
        return open_ones[:1] if open_ones else cands[-1:]

    def _dep_ready(self, dep: "Obligation", dependent: "Obligation") -> bool:
        """Is a dependency satisfied ENOUGH for `dependent` to be emitted?

        For an ordinary sequencing edge the answer is `dep is SETTLED` -- a pour must not start
        before the previous pour has been verified.

        For a grasp/place edge it CANNOT be, and demanding it produces a cycle rather than a
        discipline: `pick#1` settles only WITH `place#1` (that is the derivation that lets a
        six-step execution graph settle from a three-step scoring table), so "place requires
        pick SETTLED" can only ever be satisfied by a place that the rule itself refuses.
        Measured consequence of the cycle in job 617408: of 719 emitted primitives, 719 were
        grasps and NOT ONE was a place, and the board advertised a grasp in every single step.

        The observable discretisation that preserves the intent is "the grasp has been ATTEMPTED":
        either an informative look moved it past OPEN, or the act channel ran at least once. What
        the edge forbids -- a place before the robot has ever reached for the object -- is exactly
        what it still forbids.
        """
        if dep.status == ST_SETTLED:
            return True
        if _phase_class(dep.kind) == "grasp" and _phase_class(dependent.kind) == "place":
            return dep.status != ST_OPEN or int(dep.tau_act) > 0
        return False

    def admissibility(self, primitive: str) -> str | None:
        """Return a rejection reason, or None. THE gate.

        Every rule is a graph fact; none is a pattern aimed at a benchmark's vocabulary. The gate
        is CONSERVATIVE by construction: it rejects only on POSITIVE evidence of inadmissibility
        (an off-graph entity is named, or a dependency is unmet, or a completed counting family is
        repeated) and never because an intent could not be classified. A gate that rejected on
        classification failure would be a gate that punishes paraphrase, which is the opposite of
        what the admissible-templates channel is for -- and it would be indistinguishable from a
        model that simply writes differently.

        Turning the gate off (`MEMEXP_AOM_GRAPH_GATE=0`) leaves only the empty and label checks,
        which is the ablation that measures what the graph buys over GPM's regex table.
        """
        p = str(primitive or "").strip()
        if not p:
            return "empty primitive"
        if G.looks_like_label_primitive(p, None):
            return "copied stage label/ordinal"
        if not graph_gate_on():
            return None
        if _INTENT_MICROWAVE.search(p) and not any(
                "microwave" in o.stage_name.lower() for o in self._ordered()):
            self.n_offgraph_rejects += 1
            return "off-graph: no obligation involves the microwave"
        targets = self.obligations_for(p)
        if not targets:
            # Cannot tell which obligation this advances. Allowed: an unclassified paraphrase is a
            # measurement of the model's vocabulary, not of the arm's validity.
            self.n_unmatched += 1
            return None
        # Single target by construction of obligations_for. Check THAT obligation's deps only.
        for o in targets:
            for dep in o.deps:
                d = self.obligations.get(dep)
                if d is not None and not self._dep_ready(d, o):
                    self.n_dep_rejects += 1
                    return (f"dependency not met: '{o.kind}' requires '{d.kind}' to be served "
                            f"first (it is {d.status}, attempts={d.tau_act})")
            if o.status == ST_SETTLED and o.mode == MODE_ACTUAL:
                fam = _family_of(o.kind)
                ders = [x for x in self._ordered() if x.kind == f"count:{fam}"] if fam else []
                if any(x.status == ST_SETTLED for x in ders):
                    self.n_offgraph_rejects += 1
                    return (f"'{fam}' is complete: repeating it would over-serve a counted step "
                            f"(the task counts to {ders[0].target if ders else 'n'})")
        return None

    def apply_control(self, out_text: str) -> tuple[str, bool]:
        prim = G.extract_primitive(out_text)
        reason = self.admissibility(prim)
        if reason is None:
            return out_text, False
        self.n_gate_rejects += 1
        o = self.active()
        fb = (self.templates_for(o) or ["Reach for and grasp the target object."])[0]
        return json.dumps({"current_primitive": fb, "keyframe_positions": []}, ensure_ascii=False), True

    def forbidden_lines(self) -> list[str]:
        """The forbidden set, COMPUTED from the graph. GPM hardcodes four sentences; here the list
        is the set of obligations this task cannot reach yet, so a new task needs no edit."""
        lines = ["Do not copy step labels or stage ordinals."]
        if not any("microwave" in o.stage_name.lower() for o in self._ordered()):
            lines.append("No obligation involves the microwave: do not open or use it.")
        for o in self._ordered():
            if o.mode == MODE_DERIVED or not o.is_open():
                continue
            blocked = [self.obligations[d].kind for d in o.deps
                       if d in self.obligations and self.obligations[d].status != ST_SETTLED]
            if blocked:
                # Discriminant kinds (place#2) already prevent the self-same-name lie; attach a
                # short stage stem so the model can tell which object is waiting.
                stem = G.action_stem_safe(o.stage_name) or o.kind
                lines.append(f"Not yet: '{o.kind}' ({stem}) waits on {'/'.join(blocked)}.")
        lines.append("Serve only the ACTIVE obligation; ignore later task steps until it is SETTLED.")
        return lines

    # -- the unified board ---------------------------------------------------------------------
    def board(self, mode: str = "fluent", stall: int = 0, redact_fn: Any = None) -> str:
        """ONE block, generated from (mode, status). The sections are functions of the type system,
        which is what stops the three prompt surfaces of the two parent designs from drifting apart
        -- the defect `memexp_er` names at its own `_ADDRESS_KINDS` comment."""
        self.last_mode = mode
        o = self.active()
        v, n = self.progress()
        verdict = self.last_verdict
        lines = [f"=== obligation board ({mode}{'' if verdict == 'advance' else ' / ' + verdict}) ===",
                 f"settled {v}/{n}"]
        # SETTLED: closed. A settled obligation is not to be re-argued.
        settled = [x for x in self._ordered() if x.mode != MODE_DERIVED and x.status == ST_SETTLED]
        if settled:
            lines.append("SETTLED (do not redo): " + ", ".join(x.kind for x in settled))
        # ACTIVE: the one obligation to serve, with the admissible templates when acting. A settled
        # pointer is NOT rendered as active -- see `arbitrate`, which advances it instead.
        if o is not None and (o.is_open() or mode == "fluent"):
            lines.append(f"ACTIVE {o.oid} {o.kind} [{o.mode}/{o.status}]  attempts={o.tau_act} looks={o.tau_ret}")
            lines.append(f"NEED: {o.predicate}")
        lines.extend(self.forbidden_lines())
        # OPEN needs_look: the addresses, i.e. the pull surface. Rendered from the flag, so a task
        # with no needs_look obligation prints nothing here rather than an empty advert block.
        # (Previously keyed on MODE_PERCEPTUAL alone, which hid open/close after the MODE split.)
        percept = [x for x in self._ordered() if x.needs_look and x.is_open()]
        if percept:
            lines.append("OPEN (must be LOOKED at; query them before inventing answers):")
            for x in percept:
                lines.append(f"  - {x.oid} {x.kind}  ({x.predicate})")
        # OPEN DERIVED: the live progress values. This is the section that cannot exist in either
        # parent design, and the reason `extra_pour_detected` can move off 0.
        ders = [x for x in self._ordered() if x.mode == MODE_DERIVED]
        if ders and derived_on():
            lines.append("PROGRESS (computed from SETTLED obligations; trust this over recall):")
            for x in ders:
                mark = "done" if x.status == ST_SETTLED else "pending"
                lines.append(f"  - {x.kind}: {x.value or '0/' + str(x.target)}  [{mark}]")
        if mode == "evidence" and o is not None:
            lines.append("Admissible templates (paraphrase OK, same intent):")
            for t in self.templates_for(o):
                lines.append(f"  - {t}")
            lines.append(f"stall={int(stall)}  evidence_abs={self.evidence_abs or []}")
            lines.append("SEEN is not SETTLED. Emit ONE natural command for NEED only.")
        elif mode == "fluent":
            lines.append("Emit ONE natural command that serves NEED only. Do not skip ahead.")
        if verdict == "stagnant" and o is not None:
            lines.append("")
            lines.append(f"STAGNANT: {o.oid} {o.kind} — evidence is not advancing the stage. "
                         "Keep serving NEED with a concrete physical command; a fresh act attempt "
                         "will be re-armed shortly. Do not invent later steps.")
        if verdict == "act" and o is not None and getattr(o, "_rearmed", False):
            lines.append("Retry NEED with a different approach angle or grip; same object.")
            o._rearmed = False
        text = "\n".join(lines)
        if len(text) > DIGEST_CAP:
            text = text[:DIGEST_CAP]
        if redact_fn is not None:
            try:
                red = str(redact_fn(text) or text)
                if red != text:
                    self.n_board_redacted += 1
                text = red
            except Exception:
                pass
        self.n_board_chars = len(text)
        self.last_board = text
        return text

    def spec_text(self) -> str:
        return "\n".join([
            "=== obligation tools (optional) ===",
            '{"tool":"look","id":"<oN>"}',
            '{"tool":"ls","path":"obligations"}',
            '{"tool":"cat","path":"board"}',
            '{"tool":"grep","pattern":"<literal>"}',
            FINISH_FORM,
        ])

    def record_tool(self, *, step: int, tool: str, args: dict, offered: list[int],
                    text_head: str, auto: bool = False) -> None:
        self.tool_trace.append({
            "step": int(step), "tool": str(tool), "auto": bool(auto), "args": dict(args or {}),
            "offered": [int(x) for x in offered], "text_head": str(text_head or "")[:240],
        })

    def summary(self) -> dict[str, Any]:
        return {
            "n_obligations": len(self.obligations),
            "n_events": self.n_events,
            "n_plan_steps": self.n_plan_steps,
            "n_advance": self.n_advance,
            "n_act": self.n_act,
            "n_retrieve": self.n_retrieve,
            "n_derive": self.n_derive,
            "n_stagnant": self.n_stagnant,
            "n_fluent_steps": self.n_fluent,
            "n_evidence_mode_steps": self.n_evidence_mode_steps,
            "n_auto_looks": self.n_auto_looks,
            "n_looks_new_frames": self.n_looks_new_frames,
            "n_looks_zero_gain": self.n_looks_zero_gain,
            "n_reask": self.n_reask,
            "n_gate_rejects": self.n_gate_rejects,
            "n_offgraph_rejects": self.n_offgraph_rejects,
            "n_dep_rejects": self.n_dep_rejects,
            "n_unmatched": self.n_unmatched,
            "n_label_rejects": self.n_label_rejects,
            "n_derived_recomputes": self.n_derived_recomputes,
            "n_settled_events": self.n_settled_events,
            "n_evidence_carried": self.n_evidence_carried,
            "n_seeded": self.n_seeded,
            "fluent_equivalent": self.fluent_equivalent(),
            "modes": {o.oid: o.mode for o in self._ordered()},
            "needs_look": {o.oid: bool(o.needs_look) for o in self._ordered()},
            "statuses": {o.oid: o.status for o in self._ordered()},
            "derived_values": {o.kind: o.value for o in self._ordered() if o.mode == MODE_DERIVED},
            "coverage": {
                "n_actual": sum(1 for o in self._ordered() if o.mode == MODE_ACTUAL),
                "n_perceptual": sum(1 for o in self._ordered() if o.mode == MODE_PERCEPTUAL),
                "n_needs_look": sum(1 for o in self._ordered() if o.needs_look),
                "n_derived": sum(1 for o in self._ordered() if o.mode == MODE_DERIVED),
                "n_stagnant_steps": int(self.n_stagnant),
            },
            "exec_labels": list(self.exec_labels),
            "name_objects": bool(self.name_objects),
            "last_verdict": self.last_verdict,
            "last_board_head": (self.last_board or "")[:600],
        }


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _base_kind(stage_name: str) -> str:
    """Undiscriminated kind token for a stage name OR an execution (BDDL) label."""
    low = str(stage_name or "").lower()
    if "pour" in low:
        return "pour_one" if ("one" in low or "_1" in low or low.endswith("1")) else (
            "pour_two" if ("two" in low or "_2" in low) else "pour")
    if re.search(r"\bpick\b", low):
        return "pick"
    if re.search(r"lift|grasp|pick", low):
        return "lift"
    if "open" in low:
        return "open"
    if "close" in low:
        return "close"
    if re.search(r"put|place|insert", low):
        return "place"
    if re.search(r"inspect|observe", low):
        return "inspect"
    return re.sub(r"[^a-z0-9]+", "_", G.action_stem_safe(stage_name).lower()).strip("_") or "step"


def _kind_of(stage_name: str) -> str:
    """Backward-compatible alias; new minting uses `_base_kind` + family discriminant."""
    return _base_kind(stage_name)


def _family_of(kind: str) -> str:
    """The counting family an obligation belongs to. Only repeated actions have one."""
    base = str(kind or "").split("#", 1)[0]
    if base.startswith("pour"):
        return "pour"
    if base in {"place", "put"}:
        return "place"
    if base == "pick":
        return "pick"
    if base in {"open", "close"}:
        return base
    return ""


_CONTAINER_RX = re.compile(r"\s+(cabinet\d*|drawer\d*|container\d*|basket\d*|bin\d*)\s*$", re.I)

# Phase classes that are GRASPS. `lift` / `grasp` / `pick` are one class on purpose: the scoring
# partition of task 8 writes `Lift_Tomato_Sauce` while the execution partition writes
# `pick_tomato_sauce`, and both verify the same physical phase.
_GRASP_KINDS = {"lift", "grasp", "pick"}


def _phase_class(kind: str) -> str:
    k = str(kind or "").split("#", 1)[0]
    return "grasp" if k in _GRASP_KINDS else k


def should_seed_execution(scored_names: list[str], exec_labels: list[str]) -> tuple[bool, str]:
    """Whether to build the graph from the EXECUTION partition instead of the scoring one.

    The gate is narrow ON PURPOSE, and it encodes the whole diagnosis:

      * A phase the scoring partition omits is BENIGN when it lies downstream of every scored
        phase (task 8: the scoring table is lift/pour/pour and the omitted `place` is the epilogue)
        -- `seed()` is kept, so the measured task-8 behaviour cannot move.
      * It is FATAL when it is a GRASP that PRECEDES a scored phase (task 19), because a grasp is
        the one phase whose absence makes a scored phase physically unsatisfiable: a `place`
        presupposes a held object, and nothing in the scoring table ever mints the hold.

    Measured on this repo: task 19 -> True, task 5 -> False (partitions identical), task 8 -> False.
    Tasks 5 and 8 therefore keep their previous graphs bit-for-bit, which is what makes a t19 run a
    single-variable comparison against the archived AOM results rather than a whole-suite change.
    """
    scored_cls = {_phase_class(_base_kind(n)) for n in (scored_names or []) if str(n).strip()}
    exec_cls = [_phase_class(_base_kind(l)) for l in (exec_labels or []) if str(l).strip()]
    if not exec_cls or not scored_cls:
        return False, "empty partition"
    omitted = set(exec_cls) - scored_cls
    if "grasp" not in omitted:
        return False, f"no grasp phase omitted (omitted={sorted(omitted)})"
    for i, cl in enumerate(exec_cls):
        if cl == "grasp" and any(c in scored_cls for c in exec_cls[i + 1:]):
            return True, "scoring partition omits a grasp that precedes a scored phase"
    return False, "omitted grasp is downstream of every scored phase"


def _exec_obj(label: str) -> str:
    """The object phase of an execution label: 'place tomato sauce cabinet2' -> 'tomato sauce'."""
    t = re.sub(r"[_]+", " ", str(label or "").lower()).strip()
    t = re.sub(r"^(pick|grasp|lift|place|put|insert|drop)\s+", "", t).strip()
    return _CONTAINER_RX.sub("", t).strip()


def naming_need(label: str) -> str:
    """Scorer-aligned NEED that NAMES the object. Empty when the label is not recognisable, so
    callers can fall back to the shared scaffold rather than emit a malformed predicate."""
    low = str(label or "").lower()
    obj = _exec_obj(label)
    if not obj:
        return ""
    if re.search(r"\bpick\b|\bgrasp\b|\blift\b", low):
        return f"grasp the {obj} and lift it clear of the source container"
    if re.search(r"\bplace\b|\bput\b|\binsert\b", low):
        return f"place the {obj} into the target container"
    if re.search(r"\bopen\b", low):
        return f"open the {obj} and observe its contents"
    return ""


def naming_templates(label: str) -> list[str]:
    """Two fluent, object-naming primitives for an execution step. This is the counterpart of
    `naming_need`: the Planner imitates templates, and it cannot imitate an object it is not told."""
    low = str(label or "").lower()
    obj = _exec_obj(label)
    if not obj:
        return []
    if re.search(r"\bpick\b|\bgrasp\b|\blift\b", low):
        return [f"Pick up the {obj} from the source container.",
                f"Reach for and grasp the {obj}."]
    if re.search(r"\bplace\b|\bput\b|\binsert\b", low):
        return [f"Place the {obj} into the target container.",
                f"Put the {obj} inside the target container."]
    if re.search(r"\bopen\b", low):
        return [f"Open the {obj} and look inside."]
    return []


def _causal_dep(prev_kind: str, curr_kind: str) -> bool:
    """True iff curr causally requires prev to be SETTLED. Place-family siblings do not."""
    return not (_family_of(prev_kind) == "place" and _family_of(curr_kind) == "place")


def intent_of(primitive: str) -> str:
    """Name the action a primitive claims. Deliberately coarse: this classifies INTENT so the graph
    can be asked whether that intent is reachable, and it never decides admissibility itself.

    Place is checked BEFORE open/close: admissible templates say e.g. "Place the object into the
    open container", and classifying that as `open` (because of the noun phrase) was the reason
    `obligations_for` returned nothing for a legal place command on task 19.
    """
    p = str(primitive or "")
    if _INTENT_MICROWAVE.search(p):
        return "microwave"
    if _INTENT_POUR.search(p) and _INTENT_SECOND.search(p):
        return "pour_second"
    if _INTENT_POUR.search(p):
        return "pour"
    if _INTENT_LIFT.search(p):
        return "lift"
    if _INTENT_PLACE.search(p):
        return "place"
    if _INTENT_OPEN.search(p):
        return "open"
    if _INTENT_CLOSE.search(p):
        return "close"
    return "other"


def _available(candidates: list[int], on_context: set[int], store: dict[int, Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for i in candidates:
        ii = int(i)
        if ii in seen or ii in on_context or ii not in store:
            continue
        seen.add(ii)
        out.append(ii)
        if len(out) >= LOOKAHEAD:
            break
    return out


def parse_tool_call(text: str, known: set[str]) -> dict[str, Any] | None:
    return G.parse_tool_call(text, known)


def merge_keyframe_positions(out_text: str, abs_frames: list[int], *, recent_start: int,
                             n_context: int) -> str:
    return G.merge_keyframe_positions(out_text, abs_frames,
                                      recent_start=recent_start, n_context=n_context)


def report_path() -> Path:
    explicit = str(os.environ.get("MEMEXP_AOM_REPORT", "")).strip()
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("MEMEXP_RUN_DIR") or os.environ.get("OUT_ROOT") or ".")
    return root / "memexp_aom_report.json"


def write_report(ledgers: list["Ledger"]) -> Path | None:
    path = report_path()

    def total(attr: str) -> int:
        return sum(int(getattr(x, attr, 0)) for x in ledgers)

    totals = {
        "n_ledgers": len(ledgers),
        "n_obligations": sum(len(x.obligations) for x in ledgers),
        "n_events": total("n_events"),
        "n_plan_steps": total("n_plan_steps"),
        "n_act": total("n_act"),
        "n_retrieve": total("n_retrieve"),
        "n_derive": total("n_derive"),
        "n_advance": total("n_advance"),
        "n_stagnant": total("n_stagnant"),
        "n_fluent_steps": total("n_fluent"),
        "n_evidence_mode_steps": total("n_evidence_mode_steps"),
        "n_auto_looks": total("n_auto_looks"),
        "n_looks_new_frames": total("n_looks_new_frames"),
        "n_gate_rejects": total("n_gate_rejects"),
        "n_offgraph_rejects": total("n_offgraph_rejects"),
        "n_dep_rejects": total("n_dep_rejects"),
        "n_unmatched": total("n_unmatched"),
        "n_derived_recomputes": total("n_derived_recomputes"),
        "n_settled_events": total("n_settled_events"),
        "n_evidence_carried": total("n_evidence_carried"),
    }
    payload = {
        "arm": os.environ.get("MEMEXP_ARM_NAME", "evmem_aom"),
        "pid": os.getpid(),
        "version": "aom_v3",
        "knobs": {
            "gate_attempts": GATE_ATTEMPTS,
            "gate_stall": GATE_STALL,
            "ret_max": RET_MAX,
            "stag_act_rearm": STAG_ACT_REARM,
            "derived": derived_on(),
            "graph_gate": graph_gate_on(),
            "floor": floor_on(),
        },
        "totals": totals,
        "ledgers": [x.summary() for x in ledgers],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        own = Path(f"{path}.{os.getpid()}")
        tmp = Path(f"{own}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(own)
        audit = Path(f"{own}.board.jsonl")
        with audit.open("w", encoding="utf-8") as fh:
            for led in ledgers:
                if not led.tool_trace and not led.last_board:
                    continue
                fh.write(json.dumps({
                    "mode": led.last_mode,
                    "verdict": led.last_verdict,
                    "evidence_abs": led.evidence_abs,
                    "last_board": led.last_board,
                    "tool_trace": led.tool_trace,
                    "derived_values": {o.kind: o.value for o in led._ordered()
                                       if o.mode == MODE_DERIVED},
                }, ensure_ascii=False) + "\n")
        return own
    except OSError:
        return None
