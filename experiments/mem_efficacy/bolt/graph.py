"""ObligationGraph: always built from the BDDL EXECUTION partition.

BOLT never seeds from the scoring partition. Scoring stages exist only as
`verifies_stage` mappings (via harness.stage_mapper.expected_primitive_for_stage).

Dependency hardness:
  soft  — grasp → place  (attempted is enough; avoids AOM v5 deadlock)
  hard  — pour_1 → pour_2, place_i → place_{i+1}, open → place, ...

Graph potential:
  Φ(G) = Σ w_i (1 - 1[node_i=SETTLED]) + λ Σ 1[node_i=FAILED]
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from . import serve as S
from .types import (
    DEP_HARD,
    DEP_SOFT,
    NS_AMB,
    NS_ATTEMPTED,
    NS_FAILED,
    NS_OPEN,
    NS_SETTLED,
    RISK_HIGH,
    RISK_LOW,
    RISK_MID,
)

# Optional GPM helpers (templates / scaffolds). Import lazily so unit tests without
# the full mem_efficacy path still work for pure-graph logic.
try:
    import memexp_evmem as G  # type: ignore
except Exception:  # noqa: BLE001
    G = None  # type: ignore

_GRASP = re.compile(r"\b(pick|grasp|lift)\b", re.I)
_PLACE = re.compile(r"\b(place|put|insert|drop)\b", re.I)
_POUR = re.compile(r"\b(pour|tilt)\b", re.I)
_OPEN = re.compile(r"\bopen\b", re.I)
_CLOSE = re.compile(r"\bclose\b", re.I)
_CONTAINER = re.compile(
    r"\b(drawer|cabinet|microwave|basket|bowl|plate|frypan|drainer|mug|container|"
    r"fridge|oven|sink|tray|shelf|bin|box|pot|cup|rack)[0-9]*\b",
    re.I,
)
# Public alias so bolt.align can reuse the exact same container vocabulary.
CONTAINER_RE = _CONTAINER


def phase_of(label: str) -> str:
    low = str(label or "").lower()
    if _POUR.search(low):
        if re.search(r"\b(2nd|second|twice|two)\b", low):
            return "pour_two"
        if re.search(r"\b(1st|first|one)\b", low):
            return "pour_one"
        return "pour"
    if _GRASP.search(low):
        return "grasp"
    if _PLACE.search(low):
        return "place"
    if _OPEN.search(low):
        return "open"
    if _CLOSE.search(low):
        return "close"
    return "other"


def object_of(label: str) -> str:
    t = re.sub(r"[_]+", " ", str(label or "").lower()).strip()
    # "pour tomato sauce over chocolate 1st" names the SUBSTANCE before the
    # destination; everything from "over"/"into"/"onto" describes where it goes and
    # must not become part of the object identity (it produced "tomato sauce
    # chocolate bottle" as a served noun phrase).
    t = re.split(r"\b(over|into|onto|in)\b", t)[0].strip()
    t = re.sub(r"^(pick|grasp|lift|place|put|insert|drop|pour|open|close)\s+", "", t).strip()
    t = _CONTAINER.sub("", t).strip()
    t = re.sub(
        r"\b(into|in|on|over|from|to|the|a|an|1st|2nd|first|second|again|final)\b",
        " ",
        t,
    )
    return re.sub(r"\s+", " ", t).strip()


def target_of(label: str) -> str:
    """Destination phrase for a label, e.g. 'chocolate in the frypan' — or ''.

    The control arm pours with "over the chocolate in the frypan" while BOLT v1..v7
    poured with "over the target container": the BDDL label literally names the
    destination ("pour tomato sauce over chocolate 1st") and it was being discarded,
    so the VLA was asked to pour at nothing in particular.  Naming the destination is
    free grounding and is worth the few lines.
    """
    low = re.sub(r"[_]+", " ", str(label or "").lower())
    m = _CONTAINER.search(low)
    cont = m.group(0) if m else ""
    # text introduced by a preposition: "pour X over Y", "place X into Y"
    dest = ""
    prep = ""
    m2 = re.search(r"\b(over|onto|into|in|on)\s+([a-z0-9 ]+?)\s*$", low)
    if m2:
        prep = m2.group(1)
        dest = _CONTAINER.sub("", m2.group(2))
        dest = re.sub(r"\b(1st|2nd|first|second|again|final|the|a|an)\b", " ", dest)
        dest = re.sub(r"\s+", " ", dest).strip()
    if not dest:
        return cont
    if not cont or dest == cont:
        return dest
    # "in / into / on" makes `dest` a qualifier of the container ("middle drawer");
    # "over / onto" makes it a separate object sitting at the container, which is how
    # the pour family reads ("the chocolate in the frypan").
    if prep in {"in", "into", "on"}:
        return f"{dest} {cont}"
    return f"{dest} in the {cont}"


def risk_of(phase: str) -> str:
    if phase in {"place", "pour", "pour_one", "pour_two"}:
        return RISK_HIGH
    if phase in {"grasp", "open", "close"}:
        return RISK_MID
    return RISK_LOW


@dataclass
class ObligationNode:
    nid: str
    primitive: str                  # execution label (BDDL)
    phase: str
    object: str = ""
    target: str = ""
    status: str = NS_OPEN
    deps: list[tuple[str, str]] = field(default_factory=list)  # (dep_nid, hardness)
    settles_with: str = ""          # scored stage that settles a grasp derivationally
    verifies_stage: str = ""        # scored stage this node verifies
    success_observation: str = ""
    recovery_edges: list[str] = field(default_factory=list)
    risk_level: str = RISK_MID
    weight: float = 1.0
    tau_act: int = 0
    n_fail: int = 0
    n_amb: int = 0
    settled_step: int = -1
    seq: int = 0
    needs_look: bool = False
    predicate: str = ""
    templates: list[str] = field(default_factory=list)

    def is_open(self) -> bool:
        return self.status not in {NS_SETTLED}

    def to_dict(self) -> dict[str, Any]:
        return {
            "nid": self.nid,
            "primitive": self.primitive,
            "phase": self.phase,
            "object": self.object,
            "target": self.target,
            "status": self.status,
            "deps": [{"nid": d, "hardness": h} for d, h in self.deps],
            "settles_with": self.settles_with,
            "verifies_stage": self.verifies_stage,
            "risk_level": self.risk_level,
            "weight": self.weight,
            "tau_act": self.tau_act,
            "n_fail": self.n_fail,
            "n_amb": self.n_amb,
            "seq": self.seq,
            "needs_look": self.needs_look,
            "predicate": self.predicate,
        }


class ObligationGraph:
    """`G_t` — always an EXECUTION graph."""

    def __init__(self) -> None:
        self.nodes: dict[str, ObligationNode] = {}
        self._order: list[str] = []
        self.active_nid: str = ""
        self.exec_labels: list[str] = []
        self.scored_names: list[str] = []
        self.name_objects: bool = True
        self.lambda_fail: float = 0.5
        self.n_seeded = 0
        self.n_soft_deps = 0
        self.n_hard_deps = 0
        # closure invariant bookkeeping
        self.n_hard_downgraded = 0
        self.unreachable_stages: list[str] = []
        # progress guard: a node the planner is repeatedly directed at while Φ stays
        # put is demoted out of ACTIVE selection (per-node counts, see note_serve).
        self.stall_limit = int(os.environ.get("MEMEXP_BOLT_STALL_LIMIT", "2") or 0)
        self._serve_counts: dict[str, int] = {}
        self._last_served: dict[str, str] = {}
        self._prev_served: dict[str, str] = {}
        self._serve_phi = float("inf")
        # substances that a pour obligation pours (drive vessel naming + prep rules)
        self._pour_objects: list[str] = []
        self.n_prep_deps_dropped = 0

    # -- construction ----------------------------------------------------------------------
    def seed_execution(
        self,
        exec_labels: list[str],
        scored_names: list[str],
        label_to_scored: dict[str, str] | None = None,
        *,
        name_objects: bool = True,
    ) -> int:
        labels = [str(x).strip() for x in (exec_labels or []) if str(x).strip()]
        if not labels:
            return 0
        self.exec_labels = labels
        self.scored_names = [str(x) for x in (scored_names or [])]
        self.name_objects = bool(name_objects)
        mapping = dict(label_to_scored or {})

        # mint nodes
        for i, lab in enumerate(labels):
            phase = phase_of(lab)
            nid = f"n{i + 1}"
            obj = object_of(lab)
            tgt = target_of(lab)
            scored = mapping.get(lab, "")
            # grasp settles with the place that follows for the same object, if any
            settles_with = ""
            verifies = scored
            if phase == "grasp":
                for j in range(i + 1, len(labels)):
                    if phase_of(labels[j]) == "place" and object_of(labels[j]) == obj:
                        # settles_with must name a *scored* stage: the env can only
                        # report scored stages.  Falling back to the raw label made
                        # an unverifiable grasp look solvable and kept it as ACTIVE.
                        settles_with = mapping.get(labels[j], "")
                        break
            pred, tmpls = self._need_and_templates(lab, phase, obj, tgt)
            node = ObligationNode(
                nid=nid,
                primitive=lab,
                phase=phase,
                object=obj,
                target=tgt,
                verifies_stage=verifies,
                settles_with=settles_with,
                success_observation=self._success_obs(phase, obj, tgt),
                risk_level=risk_of(phase),
                weight=1.0 + (0.5 if phase.startswith("pour") else 0.0),
                seq=i,
                needs_look=(phase in {"open"} or "inspect" in lab.lower()),
                predicate=pred,
                templates=tmpls,
                recovery_edges=self._default_recovery(phase),
            )
            self.nodes[nid] = node
            self._order.append(nid)
        self._inherit_pour_objects()
        self._refresh_templates()
        self._wire_deps()
        self._dedup_deps()
        self._enforce_solvability_closure()
        self._assert_invariants()
        self.n_seeded = len(self._order)
        # Prefer the first *solvable* ready obligation, not BDDL index 0.  On the
        # counting-pour family BDDL starts with instrumental prep (pick chocolate);
        # selecting that first burned the episode stall budget before Lift was even
        # attempted (measured: every v3 trial stalled on 01_Lift at step 80 while
        # the board was still directing chocolate place).
        nxt = self.earliest_open()
        self.active_nid = nxt.nid if nxt else (self._order[0] if self._order else "")
        return self.n_seeded

    def _refresh_templates(self) -> None:
        """Rebuild every command ladder now that objects and pour registry are known.

        Templates are first built while minting, before `_pour_objects` exists.  The
        vessel naming ("tomato sauce bottle") depends on knowing which substances get
        poured, so the ladders are rebuilt once that is resolved rather than guessed
        during minting.
        """
        self._pour_objects = [n.object for n in self.ordered()
                              if n.phase.startswith("pour") and n.object]
        for n in self.ordered():
            n.predicate, n.templates = self._need_and_templates(
                n.primitive, n.phase, n.object, n.target
            )

    def _assert_invariants(self) -> None:
        """Fail loudly in the report if an invariant did not survive the repair pass."""
        self.invariant_violations: list[str] = []
        solve = {n.nid for n in self.ordered() if self.solvable(n)}
        for n in self.ordered():
            for d, h in n.deps:
                dep = self.nodes.get(d)
                if dep is None:
                    self.invariant_violations.append(f"I0 missing dep {d} on {n.nid}")
                    continue
                if self.solvable(n) and dep.nid not in solve:
                    self.invariant_violations.append(
                        f"I2 {n.nid}(verifiable) gated by {dep.nid}(unverifiable,{h})"
                    )
        pours = {S.pour_ordinal(n.phase): n for n in self.ordered()
                 if n.phase.startswith("pour")}
        p1, p2 = pours.get(1), pours.get(2)
        if p1 and p2 and not any(d == p1.nid and h == DEP_HARD for d, h in p2.deps):
            self.invariant_violations.append(
                f"I4 {p2.nid}(pour_two) lacks hard dep on {p1.nid}(pour_one)"
            )
        # I5 — STAGE SATISFIABILITY: every rung we may serve must be able to satisfy
        # the stage the node is verified by.  v7 shipped grasp rungs that only asked
        # for a grasp while the stage rewarded a lift: the planner complied, nothing
        # was ever credited, and three whole trials produced 0.0.  Checked, not trusted.
        for n in self.ordered():
            bad = S.violating_rungs(n.phase, self.serve_ladder(n))
            if bad:
                self.invariant_violations.append(
                    f"I5 {n.nid}({n.phase}) rungs cannot satisfy its stage: {bad}"
                )
        if self.unreachable_stages:
            self.invariant_violations.append(
                f"I3 unreachable scored stages {self.unreachable_stages}"
            )

    def _inherit_pour_objects(self) -> None:
        """Pour labels in the counting family do not name their substance.

        `pour first` / `pour tomato sauce over chocolate 1st` leave the object
        field empty or polluted, so the served command degenerates into a bare
        "pour" with no noun and the VLA has nothing to ground.  Inherit the
        substance from the nearest preceding grasp/place node: that is the
        bottle that is about to be poured.
        """
        for n in list(self.ordered()):
            if not n.phase.startswith("pour"):
                continue
            if n.object and n.object not in {"first", "second", "one", "two"}:
                continue
            for prev in reversed(self.ordered()[: n.seq]):
                if prev.object and prev.phase in {"grasp", "place"}:
                    n.object = prev.object
                    break
            n.predicate, n.templates = self._need_and_templates(
                n.primitive, n.phase, n.object, n.target
            )

    def _dedup_deps(self) -> None:
        """Collapse duplicate edges, soft winning: the weakest gate is the safe one."""
        for n in self.ordered():
            seen: dict[str, str] = {}
            for d, h in n.deps:
                if d not in seen or h == DEP_SOFT:
                    seen[d] = h
            n.deps = list(seen.items())

    def _enforce_solvability_closure(self) -> None:
        """Invariants that must hold for every graph, checked and repaired here.

        BOLT's architecture is only as good as its invariants.  Each rule below was
        added after a measured failure, and each is repaired in place so an illegal
        graph can never reach the Arbiter:

        I1  CLOSURE — no HARD edge may point at a node that can never SETTLE.
            A node settles only when the environment reports a scored stage.  A hard
            gate onto an unverifiable node can never open (task 8 v1: total stall).

        I2  NO-GATE — a *verifiable* node must not be gated by an unverifiable one.
            I1 handles `hard`; this adds `soft`.  The pour-over-chocolate prep was
            wired as a soft dep on the pour, so Lift could settle and ACTIVE would
            still be unable to advance to Pour_One — it fell into the instrumental
            chocolate nodes instead and spent the episode there (task 8 v5: Lift
            finished, then `grasp chocolate` appeared ahead of `pour`).

        I3  REACHABILITY — every scored stage must be carried by some node, and that
            node must be reachable through the dependency graph alone.  When a stage
            is unreachable BOLT can never score it, so this is reported rather than
            silently tolerated.

        I4  ORDINALITY — pour_two hard-depends on pour_one.  Stated here rather than
            left to wiring heuristics because "pour a second time" is only meaningful
            after a first pour, and the VLA will happily emit it out of order.
        """
        solve = {n.nid for n in self.ordered() if self.solvable(n)}
        carried = {n.verifies_stage for n in self.ordered() if n.verifies_stage}
        carried |= {n.settles_with for n in self.ordered() if n.settles_with}
        self.unreachable_stages = [s for s in self.scored_names if s and s not in carried]

        self.n_hard_downgraded = 0
        self.n_prep_deps_dropped = 0
        for n in self.ordered():
            new: list[tuple[str, str]] = []
            for d, h in n.deps:
                dep = self.nodes.get(d)
                if dep is None:
                    continue
                if self.solvable(n) and dep.nid not in solve:
                    # I2 (and I1 as the hard case): a scored obligation may never be
                    # gated by an obligation whose completion cannot be observed.
                    self.n_hard_downgraded += int(h == DEP_HARD)
                    self.n_prep_deps_dropped += 1
                    continue
                if h == DEP_HARD and dep.nid not in solve:
                    new.append((d, DEP_SOFT))          # I1
                    self.n_hard_downgraded += 1
                else:
                    new.append((d, h))
            n.deps = new

        # I4 — allow only one pour ordinal in the active chain
        self._enforce_pour_ordinality()

        self.n_soft_deps = sum(1 for n in self.ordered() for _, h in n.deps if h == DEP_SOFT)
        self.n_hard_deps = sum(1 for n in self.ordered() for _, h in n.deps if h == DEP_HARD)

    def _enforce_pour_ordinality(self) -> None:
        """I4: pour_two requires SETTLED pour_one; pour_one requires nothing poured."""
        pours = [n for n in self.ordered() if n.phase.startswith("pour")]
        for i, n in enumerate(pours):
            if S.pour_ordinal(n.phase) < 2:
                continue
            priors = [p for p in pours[:i] if S.pour_ordinal(p.phase) < 2]
            if not priors:
                continue
            prev = priors[-1]
            n.deps = [
                (d, h) for d, h in n.deps
                if self.nodes[d].phase != "pour_one"
            ]
            n.deps.append((prev.nid, DEP_HARD))
        # pour_one must not depend on a later ordinal (defensive: wiring heuristics)
        for n in pours:
            if S.pour_ordinal(n.phase) != 1:
                continue
            n.deps = [
                (d, h) for d, h in n.deps
                if not self.nodes[d].phase.startswith("pour")
            ]

    def solvable(self, node: ObligationNode) -> bool:
        """True when some environment-reported scored stage can SETTLE this node."""
        return bool(node.verifies_stage or node.settles_with)

    def coverage_report(self) -> dict[str, Any]:
        return {
            "n_exec": len(self._order),
            "n_solvable": sum(1 for n in self.ordered() if self.solvable(n)),
            "n_instrumental": sum(1 for n in self.ordered() if not self.solvable(n)),
            "n_hard_downgraded": int(getattr(self, "n_hard_downgraded", 0)),
            "unreachable_stages": list(getattr(self, "unreachable_stages", [])),
            "stall_limit": int(self.stall_limit),
            "n_stalled": sum(1 for n in self.ordered() if self.stalled(n.nid)),
            "n_verbatim_blocks": int(getattr(self, "n_verbatim_blocks", 0)),
            "invariant_violations": list(getattr(self, "invariant_violations", [])),
            "prep_deps_dropped": int(getattr(self, "n_prep_deps_dropped", 0)),
            "serve_counts": {k: v for k, v in self._serve_counts.items()},
            "active_serve": (
                self.serve_text(self.nodes[self.active_nid])
                if self.active_nid in self.nodes else ""
            ),
        }

    def _need_and_templates(self, lab: str, phase: str, obj: str, tgt: str) -> tuple[str, list[str]]:
        """Internal reasoning view (predicate) + the served command ladder.

        The predicate is a graph artefact — it is never shown to the VLA.  The
        ladder is delegated wholesale to the serve-policy layer so there is exactly
        one place that decides what the robot is told.
        """
        tmpls = S.command_ladder(
            phase,
            obj,
            tgt,
            is_pour_substance=self._is_pour_substance(obj),
            pour_ordinal=S.pour_ordinal(phase),
        )
        if self.name_objects and obj:
            if phase == "grasp":
                return (
                    f"grasp the {obj} and lift it clear of the source container",
                    tmpls,
                )
            if phase == "place":
                where = f" into the {tgt}" if tgt else " into the target container"
                return (f"place the {obj}{where}", tmpls)
            if S.is_pour_phase(phase):
                ordinal = " a second time" if phase == "pour_two" else ""
                return (f"pour from the {obj}{ordinal}", tmpls)
        if S.is_pour_phase(phase):
            return ("pour the contents over the target container", tmpls)
        # fall back to GPM scaffolds when available
        if G is not None:
            try:
                return G.predicate_scaffold(lab), list(G.admissible_templates(lab) or []) or tmpls
            except Exception:
                pass
        return f"advance: {lab}", tmpls

    def _is_pour_substance(self, obj: str) -> bool:
        """True when `obj` is the substance that some pour obligation pours."""
        o = str(obj or "").strip().lower()
        if not o:
            return False
        return any(
            o == str(p or "").strip().lower() or o in str(p or "").lower()
            for p in self._pour_objects
        )

    def _success_obs(self, phase: str, obj: str, tgt: str) -> str:
        if phase == "grasp":
            return f"held({obj})" if obj else "object_in_gripper"
        if phase == "place":
            if obj and tgt:
                return f"inside({obj}, {tgt})"
            return "object_no_longer_in_gripper_and_inside_target"
        if phase.startswith("pour"):
            return "pour_completed"
        if phase == "open":
            return f"open({tgt or obj or 'container'})"
        if phase == "close":
            return f"closed({tgt or obj or 'container'})"
        return "stage_advanced"

    def _default_recovery(self, phase: str) -> list[str]:
        if phase == "place":
            return ["regrasp", "reopen_target", "inspect_last_segment"]
        if phase == "grasp":
            return ["reapproach", "inspect_object"]
        if phase.startswith("pour"):
            return ["regrasp_bottle", "retry_pour"]
        return ["inspect_last_segment", "retry"]

    def _recovery_templates(self, node: ObligationNode) -> list[str]:
        """Deprecated shim — the ladder now lives in `bolt.serve`."""
        return S.command_ladder(
            node.phase,
            node.object,
            node.target,
            is_pour_substance=self._is_pour_substance(node.object),
            pour_ordinal=S.pour_ordinal(node.phase),
        )

    def serve_ladder(self, node: ObligationNode) -> list[str]:
        """Ordered commands for one obligation: primary templates, then recovery.

        The raw BDDL label is deliberately NOT a rung.  It is a graph identifier, not
        a robot command: v5 shipped `pick tomato sauce.` as the terminal rung and the
        planner echoed it back for the rest of the episode (67.5% of all outputs on
        task 8).  Every rung here is a complete imperative with a grounded noun phrase.
        """
        out: list[str] = []
        seen: set[str] = set()
        for raw in list(node.templates) + self._recovery_templates(node):
            t = str(raw or "").strip()
            if not t:
                continue
            if not t.endswith("."):
                t = t + "."
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out or [f"Retry the current step on the {node.object or 'target object'}."]

    def _pick_serve(self, node: ObligationNode, k: int) -> str:
        """Serve-ladder entry `k`, cycling and never repeating consecutively.

        Saturation is the bug this replaces: the old code clamped at the last rung,
        so a stalled obligation was directed with one identical sentence for the
        rest of the episode (measured: 14/17, 15/17, 16/18, 17/19 identical outputs
        in the four task-8 trials that never lifted).  Cycling keeps the *intent*
        identical — same obligation, same node — while varying the phrasing every
        step, which is precisely the behaviour the control arm uses to recover.
        """
        ladder = self.serve_ladder(node)
        if not ladder:
            return node.primitive
        k = int(k) % len(ladder)
        pick = ladder[k]
        prev = str(self._last_served.get(node.nid, "")).strip().lower()
        if len(ladder) > 1 and pick.strip().lower() == prev:
            pick = ladder[(k + 1) % len(ladder)]
        return pick

    def serve_text(self, node: ObligationNode) -> str:
        """Command currently served for `node`, rotating on repeated serves."""
        # First serve of an obligation uses ladder[0]; note_serve() advances it.
        k = max(0, int(self._serve_counts.get(node.nid, 1)) - 1)
        return self._pick_serve(node, k)

    def serve_counts(self, nid: str) -> int:
        return int(self._serve_counts.get(nid, 0))

    def prev_served(self, nid: str) -> str:
        """The sentence issued for `nid` on the preceding decision ('' if none)."""
        return str(self._prev_served.get(nid, ""))

    def served_texts(self, nid: str) -> list[str]:
        """Distinct sentences issued so far — the diversity measure used in reporting."""
        return [t for t in (self._prev_served.get(nid), self._last_served.get(nid)) if t]

    def _wire_deps(self) -> None:
        """Causal edges, expressed so that the invariants hold by construction.

        Ordering rule: an obligation may only be gated by an obligation whose
        completion the environment can observe.  Instrumental prep (place the
        chocolate, open the microwave) is therefore never a gate on a scored stage;
        the invariant pass enforces that, and this wiring avoids emitting those edges
        in the first place so the graph is clean rather than repaired.

        Edges created:
          grasp_i → place_i                SOFT   (attempted is enough — AOM v5 fix)
          grasp_bottle → pour(first)       SOFT   (the bottle must be in hand)
          pour_n → pour_{n+1}              HARD   (ordinality, see I4)
          open → place into that container HARD
          scored_i → scored_{i+1}          HARD   (generic, verifiable predecessors only)
        """
        last_by_obj: dict[str, str] = {}
        last_pour = ""
        last_solvable = ""
        for nid in self._order:
            n = self.nodes[nid]
            # grasp → following place (same object): SOFT
            if n.phase == "place" and n.object and n.object in last_by_obj:
                dep = last_by_obj[n.object]
                if self.nodes[dep].phase == "grasp":
                    n.deps.append((dep, DEP_SOFT))
                    self.n_soft_deps += 1
            # pour chain: HARD on previous pour (I4 restates this defensively)
            if n.phase.startswith("pour") and last_pour:
                n.deps.append((last_pour, DEP_HARD))
                self.n_hard_deps += 1
            # bottle in hand before the first pour
            if n.phase.startswith("pour") and not last_pour and n.object:
                for prev in reversed(self.ordered()[: n.seq]):
                    if prev.phase == "grasp" and (
                        prev.object == n.object
                        or prev.object in n.object
                        or n.object in prev.object
                    ):
                        n.deps.append((prev.nid, DEP_SOFT))
                        self.n_soft_deps += 1
                        break
            # open → place into that container: HARD
            if n.phase == "place" and n.target:
                for prev_id in reversed(self._order[: n.seq]):
                    p = self.nodes[prev_id]
                    if p.phase == "open" and (
                        p.target == n.target or p.object == n.target or n.target in p.primitive
                    ):
                        n.deps.append((prev_id, DEP_HARD))
                        self.n_hard_deps += 1
                        break
            # generic sequencing, but only through a verifiable predecessor
            if last_solvable and n.phase not in {"place"} and not n.deps:
                if not (self.nodes[last_solvable].phase == "place" and n.phase == "place"):
                    n.deps.append((last_solvable, DEP_HARD))
                    self.n_hard_deps += 1

            if n.phase == "grasp" and n.object:
                last_by_obj[n.object] = nid
            if n.phase.startswith("pour"):
                last_pour = nid
            if n.phase not in {"other"} and self.solvable(n):
                last_solvable = nid

    # -- queries ---------------------------------------------------------------------------
    def ordered(self) -> list[ObligationNode]:
        return [self.nodes[i] for i in self._order if i in self.nodes]

    def active(self) -> ObligationNode | None:
        """Current obligation, re-evaluated whenever the cached one is unusable.

        The original version returned the cached `active_nid` for as long as it was
        not SETTLED.  Combined with a node that can never be verified, that pinned
        ACTIVE forever: `earliest_open()` (and therefore every tiering rule) was
        never consulted, so the Arbiter rewrote every proposal back to the same
        dead primitive for the rest of the episode.

        Returns None once every verifiable obligation is closed.  Acting further
        could only disturb a scene that already scores, and the Arbiter treats
        "no active node" as a pass-through rather than a rewrite target.
        """
        if not self._any_solvable_open():
            self.active_nid = ""
            return None
        if self.active_nid and self.active_nid in self.nodes:
            n = self.nodes[self.active_nid]
            if n.status != NS_SETTLED and self.deps_ready(n) and not self._tier_suppressed(n):
                return n
        nxt = self.earliest_open()
        self.active_nid = nxt.nid if nxt else ""
        return nxt

    def _any_solvable_open(self) -> bool:
        return any(n.is_open() and self.solvable(n) for n in self.ordered())

    def _tier_suppressed(self, node: ObligationNode) -> bool:
        """Instrumental nodes are demoted once the stall budget is spent."""
        return not self.solvable(node) and self.stalled(node.nid)

    def earliest_open(self) -> ObligationNode | None:
        """Select the next obligation, preferring nodes that can actually close.

        Tiering matters because a node with no verification path can be
        *attempted* but never *settled*.  Selecting it purely by graph order
        made it the ACTIVE obligation forever: the Arbiter then rewrote every
        proposal back to that one primitive (measured: 62% of task 5's and 96%
        of task 22's planner outputs collapsed onto a single node).

        tier 0 — ready AND can be verified  → real progress
        tier 1 — ready, instrumental, never attempted → do it once to unblock
        tier 2 — ready, instrumental, already attempted → fallback
        tier 3 — stalled / demoted → last resort

        When nothing above tier 3 is available the tie is broken by how many
        verifiable obligations the node gates: repeating a dead primitive that
        leads nowhere is strictly worse than repeating one that unblocks a
        scored stage (this is what kept the ladder oscillating between two
        instrumental nodes and never reaching the first pour).
        """
        ready = [n for n in self.ordered() if n.is_open() and self.deps_ready(n)]
        for tier in (0, 1, 2):
            for n in ready:
                if self._tier(n) == tier:
                    return n
        if ready:
            return max(ready, key=lambda n: (self.unblock_value(n.nid), -n.seq))
        for n in self.ordered():
            if n.is_open():
                return n
        return None

    def unblock_value(self, nid: str) -> int:
        """How many verifiable obligations become reachable once `nid` is done."""
        seen: set[str] = set()
        stack = [nid]
        value = 0
        while stack:
            cur = stack.pop()
            for n in self.ordered():
                if n.nid in seen:
                    continue
                if any(d == cur for d, _ in n.deps):
                    seen.add(n.nid)
                    if self.solvable(n):
                        value += 1
                    stack.append(n.nid)
        return value

    def _tier(self, node: ObligationNode) -> int:
        # Solvable nodes always keep top priority: closing them is what scores.
        # Only instrumental (never-verifiable) nodes are demoted, because budget
        # spent on them can never be confirmed by the environment.
        if self.solvable(node):
            return 0
        if self.stalled(node.nid):
            return 3
        return 1 if int(node.tau_act) <= 0 else 2

    def stalled(self, nid: str) -> bool:
        return self.stall_limit > 0 and int(self._serve_counts.get(nid, 0)) >= self.stall_limit

    def note_serve(self, nid: str) -> None:
        """One planning decision is about to be directed at `nid`.

        Advances the serve ladder and records the exact sentence that goes out, so
        `_pick_serve` can guarantee the next step does not repeat it.  Φ moving
        forward clears the per-node counters: progress re-opens the question.
        """
        if not nid:
            return
        phi = self.phi()
        if abs(phi - self._serve_phi) > 1e-9:
            self._serve_counts.clear()
            self._last_served.clear()
            self._prev_served.clear()
        self._serve_counts[nid] = int(self._serve_counts.get(nid, 0)) + 1
        node = self.nodes.get(nid)
        if node is not None:
            self._prev_served[nid] = self._last_served.get(nid, "")
            self._last_served[nid] = self._pick_serve(node, self._serve_counts[nid] - 1)
        self._serve_phi = phi

    def deps_ready(self, node: ObligationNode) -> bool:
        for dep_id, hardness in node.deps:
            d = self.nodes.get(dep_id)
            if d is None:
                continue
            if hardness == DEP_SOFT:
                # attempted is enough (status != OPEN or tau_act > 0) — AOM v5 fix.
                # "Served as the active obligation" also counts: BOLT directed the
                # robot at it, and this dep must never be able to deadlock the graph
                # on a planner that simply did not emit the expected primitive.
                served = int(self._serve_counts.get(d.nid, 0)) > 0
                if d.status == NS_OPEN and int(d.tau_act) <= 0 and not served:
                    return False
            else:
                if d.status != NS_SETTLED:
                    return False
        return True

    def phi(self) -> float:
        """Graph potential Φ(G)."""
        open_cost = sum(n.weight for n in self.ordered() if n.status != NS_SETTLED)
        fail_cost = self.lambda_fail * sum(1 for n in self.ordered() if n.status == NS_FAILED)
        return float(open_cost + fail_cost)

    def coverage(self) -> dict[str, Any]:
        by = {NS_OPEN: 0, NS_ATTEMPTED: 0, NS_AMB: 0, NS_FAILED: 0, NS_SETTLED: 0}
        for n in self.ordered():
            by[n.status] = by.get(n.status, 0) + 1
        return {
            "n_nodes": len(self._order),
            "n_soft_deps": self.n_soft_deps,
            "n_hard_deps": self.n_hard_deps,
            **{f"n_{k.lower()}": v for k, v in by.items()},
            "phi": round(self.phi(), 3),
            **self.coverage_report(),
        }

    # -- mutations -------------------------------------------------------------------------
    def note_attempt(self, nid: str) -> None:
        n = self.nodes.get(nid)
        if n is None:
            return
        n.tau_act += 1
        if n.status == NS_OPEN:
            n.status = NS_ATTEMPTED

    def note_verified_stage(self, stage_name: str, *, step: int = -1) -> list[str]:
        """Settle every node whose verifies_stage / settles_with matches."""
        settled: list[str] = []
        sn = str(stage_name or "")
        if not sn:
            return settled
        for n in self.ordered():
            if n.status == NS_SETTLED:
                continue
            if n.verifies_stage == sn or n.settles_with == sn:
                n.status = NS_SETTLED
                n.settled_step = int(step)
                settled.append(n.nid)
        # advance active pointer
        if settled:
            # progress resets the guard: every node deserves a fresh look
            self._serve_counts.clear()
        if self.active_nid in settled:
            nxt = self.earliest_open()
            self.active_nid = nxt.nid if nxt else ""
        return settled

    def note_failure(self, nid: str, *, reason: str = "") -> None:
        n = self.nodes.get(nid)
        if n is None:
            return
        n.status = NS_FAILED
        n.n_fail += 1

    def note_ambiguous(self, nid: str) -> None:
        n = self.nodes.get(nid)
        if n is None:
            return
        n.status = NS_AMB
        n.n_amb += 1

    def set_active(self, nid: str) -> None:
        if nid in self.nodes and self.deps_ready(self.nodes[nid]):
            self.active_nid = nid

    @staticmethod
    def _norm_cmd(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower()).strip()

    def recognize(self, text: str) -> str:
        """Bind free text to a node by recognising a command BOLT itself serves.

        This is tier 1 of matching and it must come first.  The planner's dominant
        behaviour is to echo the ACTIVE line of the board, so recognising our own
        ladder strings binds the proposal to the node we intended — instead of
        re-parsing its words.  Without this, the recovery rungs ("Re-localize the
        tomato sauce bottle, then close the gripper around it") re-derived a nonsense
        phase (`close`) and object, failed to bind, and fell through to the off-graph
        pass-through — escaping the dependency and ordinal gates entirely.  That is a
        correctness hole, not a cosmetic one: it is how an impatient `pour a second
        time` could reach the environment while the first pour was still open.
        """
        key = self._norm_cmd(text)
        if not key:
            return ""
        idx: dict[str, str] = {}
        # open nodes first so a shared phrasing resolves to the live obligation
        for n in sorted(self.ordered(), key=lambda x: (not x.is_open(), x.seq)):
            for t in self.serve_ladder(n):
                idx.setdefault(self._norm_cmd(t), n.nid)
        return idx.get(key, "")

    def match_primitive(self, text: str) -> list[ObligationNode]:
        """Find open nodes the planner's emitted text can be bound to.

        Tier 1: exact recognition of a served command (see `recognize`) — this is the
        path the planner actually takes.
        Tier 2: re-derive phase/object tokens, for genuinely novel phrasings.
        Tier 3: ordinal safety net — a later-pour request that failed to bind is still
        bound to the pour node of that ordinal so the gate can refuse it.
        """
        low = str(text or "").lower()
        nid = self.recognize(text)
        if nid and nid in self.nodes:
            return [self.nodes[nid]]

        phase = phase_of(low)
        cands = [n for n in self.ordered() if n.is_open() and n.phase == phase]
        if not cands and phase == "grasp":
            cands = [n for n in self.ordered() if n.is_open() and n.phase == "grasp"]
        # prefer object-named match
        if cands:
            named = [n for n in cands if n.object and n.object in low]
            if named:
                return named[:1]
        if cands:
            return cands[:1]

        ord_ = S.pour_ordinal(phase) or self._ordinal_from_text(low)
        if ord_ >= 2:
            later = [n for n in self.ordered() if n.is_open() and n.phase == "pour_two"]
            if later:
                return later[:1]
        if phase in {"pour", "other"} and self._mentions_pour(low):
            first = [n for n in self.ordered() if n.is_open() and n.phase == "pour_one"]
            if first and self._ordinal_from_text(low) < 2:
                return first[:1]
        return []

    @staticmethod
    def _mentions_pour(low: str) -> bool:
        return any(w in low for w in ("pour", "tilt", "tip ", "empty ", "decant"))

    @staticmethod
    def _ordinal_from_text(low: str) -> int:
        if any(w in low for w in ("second", "2nd", "again", "once more", "one more",
                                  "twice", "another")):
            return 2
        if any(w in low for w in ("first", "1st")):
            return 1
        return 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_nid": self.active_nid,
            "exec_labels": list(self.exec_labels),
            "scored_names": list(self.scored_names),
            "name_objects": self.name_objects,
            "coverage": self.coverage(),
            "nodes": [n.to_dict() for n in self.ordered()],
        }
