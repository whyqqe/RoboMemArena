"""The single scheduling law.

Every previous arm has at least one scheduler per component: GPM has `decide_mode` plus
`apply_control`; AOM has `arbitrate` plus a separate proof-obligation retry; BOLT has the
graph's `earliest_open` tie-break, the router's evidence floor and the clock's
`allow_commit`.  They disagree, and the archive shows the disagreement is priced.

DIAL has one law, and two properties that are structural rather than tuned:

    (1) The action space contains no BLOCK.
        a in { RESAMPLE, QUERY, RESYNC, PERSIST }
        RESAMPLE always names an attempt DIFFERENT from the one just run, so
        "same-obligation absorbing loop" is not a state this scheduler can enter.
        GPM, AOM and BOLT all reach it, because their rejection path rewrites *onto* the
        active obligation and an obligation that is not physically executable then has no
        exit.  Measured signature: BOLT t8 v7 `0.0,0.0,0.0` and v8 `0.0,0.0,0.0,66.7,0.0`;
        AOM t22 v6 has five 0.0 in nine episodes.

    (2) PERSIST is bounded.
        Waiting is correct while the scorer is plausibly not due yet (T3: the median
        enacted->credited lag is 25 steps against GPM's 8-step and AOM's 12-step
        stagnation horizons, so 72.9% / 62.7% of *successfully credited* attempts would
        have been cancelled in flight).  But patience that never expires is just a slower
        absorbing loop, so past `belief.escape_horizon` an attempt is declared dead and
        RESAMPLE is forced.  The horizon clears the UPPER end of the measured lag
        distribution (median 25, p90 145, max 685), because a bound of the form
        `k * lambda_hat` fires at age ~8 under an optimistic lambda_hat and would cancel
        exactly the attempts that were about to succeed.

The law is a rational choice over (expected credit, information, cost):

    a* = argmax  E[d credited | a]  +  kappa * H(beta | a)  -  cost(a)

The expected-credit column is where the bottleneck does the work: each cause has exactly
one action that addresses it, so the posterior's argmax is also the argmax of expected
credit.  That is the formal reason a single scheduler can serve planner, VLA strategy,
memory and harness without a negotiation protocol — the posterior IS the consensus.

Reward bookkeeping is per FAMILY, not per single attempt.  A task steps through
01_Lift, 02_Pour_One, 03_Pour_Two; when 02 is credited the attempt that earned it was
issued while the obligation was 01.  A single `last_key` would mis-attribute that credit
to whatever was issued last, so each family keeps its own outstanding attempt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .bottleneck import BottleneckBelief, observables
from .strategy import StrategyBank
from .types import (
    CAUSE_UNGROUNDED,
    CAUSE_UNKNOWN,
    CAUSE_UNREACHABLE,
    CAUSE_UNSAFE,
    CAUSE_UNSYNCED,
    Strategy,
    family_of,
)

A_RESAMPLE = "resample"
A_QUERY = "query"
A_RESYNC = "resync"
A_PERSIST = "persist"
ACTIONS = (A_RESAMPLE, A_QUERY, A_RESYNC, A_PERSIST)
# Structural invariant, asserted by selftest: there is no rejecting action.
FORBIDDEN_ACTIONS = ("block", "reject", "rewrite")

_COST = {A_RESAMPLE: 1.0, A_QUERY: 1.0, A_RESYNC: 0.0, A_PERSIST: 0.0}

Key = tuple[str, str, int]


@dataclass
class DIALDecision:
    action: str
    strategy: Strategy | None
    reason: str
    bottleneck: str
    beta: dict[str, float] = field(default_factory=dict)
    expected_credit: float = 0.0
    information: float = 0.0
    value: float = 0.0

    @property
    def text(self) -> str:
        return self.strategy.text if self.strategy else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action, "bottleneck": self.bottleneck, "text": self.text,
            "reason": self.reason, "expected_credit": round(self.expected_credit, 4),
            "information": round(self.information, 4), "value": round(self.value, 4),
            "beta": {k: round(v, 3) for k, v in self.beta.items()},
        }


@dataclass
class DIALPolicy:
    """One scheduler for planner + VLA strategy + memory + harness."""
    bank: StrategyBank
    belief: BottleneckBelief = field(default_factory=BottleneckBelief)
    info_weight: float = 0.30
    persist_threshold: float = 0.05
    resample_cost: float = 0.20

    # current obligation
    obligation: str = ""
    family: str = "other"
    obj: str = "the target object"
    now: int = 0

    # per-family outstanding attempt: family -> (key, step_issued, text)
    pending: dict[str, tuple[Key, int, str]] = field(default_factory=dict)
    # family -> the step at which the CURRENT ATTEMPT CLASS was first chosen.  Unlike the
    # `issued_at` inside `pending`, this is never refreshed by a re-enactment, and it is what
    # the escape horizon is measured against (defect 17).
    first_issued: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    last_key: dict[str, Key] = field(default_factory=dict)
    n_credited: int = 0
    n_reenact: int = 0

    # counters
    n_resample: int = 0
    n_query: int = 0
    n_resync: int = 0
    n_persist: int = 0
    n_forced_escape: int = 0
    n_self_check_fail: int = 0

    # -- lifecycle -------------------------------------------------------------------------
    def set_obligation(self, obligation: str, family: str, obj: str) -> None:
        if obligation != self.obligation:
            # the evidence in the posterior was about the previous obligation
            self.belief.reset(keep_lambda=True)
        self.obligation = obligation
        self.family = family
        self.obj = obj

    def age(self, family: str | None = None) -> int:
        """Steps since the outstanding attempt was last (re-)enacted; -1 if none.

        DEFECT 17: this used to be the ONLY clock, and `issued_at` was refreshed only when a
        *new* attempt class was picked.  A live run then reported `now=1931` on a ~1000-step
        episode with `forced_escape=7`, i.e. the scheduler was timing "steps since the aim
        last changed" and treating its own holds as staleness — so patience and the escape
        horizon were measured against the wrong origin, and the arm kept switching aim on
        precisely the episodes the baseline wins by repeating.  This clock now measures
        "steps since the robot last pursued this attempt", which is what `hold_horizon`
        was fitted to.  Non-absorption is preserved by `age_total` below, not by this one.
        """
        fam = family if family is not None else self.family
        pend = self.pending.get(fam)
        return -1 if pend is None else max(0, self.now - pend[1])

    def age_total(self, family: str | None = None) -> int:
        """Steps since the current ATTEMPT CLASS was first chosen; -1 if none.

        Never refreshed by a re-enactment.  This is the absolute cap that guarantees the
        scheduler cannot absorb: however patiently an attempt is repeated, past
        `escape_horizon` the scheduler must resample.  Patience is bounded by construction
        rather than by a score penalty that a confident posterior could outvote.
        """
        fam = family if family is not None else self.family
        if fam not in self.pending:
            return -1
        first = self.first_issued.get(fam)
        if first is None:
            return -1
        return max(0, self.now - first)

    def note_enacted(self, key: Key, text: str, *, family: str | None = None,
                     step: int | None = None) -> None:
        """Record that `key` is the attempt now being pursued for `family`.

        Re-enacting the SAME key is a continuation, not a new attempt: it refreshes the
        patience clock (`issued_at`) but not `first_issued`, so the escape horizon still
        bounds how long one class may be held.
        """
        fam = family if family is not None else self.family
        st = self.now if step is None else int(step)
        prev = self.pending.get(fam)
        self.pending[fam] = (key, st, text)
        if prev is not None and prev[0] == key:
            # continuation of the same attempt class
            self.n_reenact += 1
            self.last_key[fam] = key
            return
        self.first_issued[fam] = st
        self.attempts[fam] = int(self.attempts.get(fam, 0)) + 1
        self.last_key[fam] = key

    def advance(self, step: int, *, zero_displacement: bool = False,
                retract_or_slip: bool = False, no_observation: bool = False,
                memory_hit: bool = False) -> None:
        """One planner step with no credit yet: fold the observables into the posterior."""
        self.now = int(step)
        age = self.age()
        if age < 0:
            return
        # `within` is the window during which the attempt may still be in flight.  It is
        # `hold_horizon` (a high quantile), NOT `lambda_hat` (the median): using the median
        # flags half of all attempts as overdue before their credit arrives.  A bulk patch
        # had silently failed to apply this, leaving the median here, and the scheduler
        # cancelled a correct attempt at step 26 of a 30-step lag — caught by selftest C3.
        within = age <= self.belief.hold_horizon and self.age_total() <= self.belief.escape_horizon
        self.belief.update(observables(
            # `enacted_no_credit` is only evidence for UNSYNCED while the attempt may still
            # be in flight.  Past that, the same silence is evidence that the attempt failed,
            # and must be fed as such or the posterior pins on UNSYNCED and never moves.
            enacted_no_credit=within,
            overdue_no_credit=not within,
            zero_displacement=zero_displacement,
            retract_or_slip=retract_or_slip,
            no_observation=no_observation,
            repeat_no_move=int(self.attempts.get(self.family, 0)) >= 2,
            lag_within_lambda=within,
            memory_hit=memory_hit,
        ))

    def retire(self, *, family: str | None = None) -> None:
        """Declare the outstanding attempt dead and record it as a negative example.

        Without this the strategy posterior only ever sees successes, so it cannot tell a
        cell that never works from one that has not been tried — and the bandit would keep
        re-selecting a hopeless cell on the strength of its optimistic prior.
        """
        fam = family if family is not None else self.family
        pend = self.pending.pop(fam, None)
        self.first_issued.pop(fam, None)
        if pend is None:
            return
        self.bank.observe(fam, pend[0], False)

    def on_credit(self, stage: str, *, step: int | None = None) -> bool:
        """Credit arrived for `stage`.  Attribute it to the family that earned it."""
        fam = family_of(stage)
        self.now = self.now if step is None else int(step)
        pend = self.pending.pop(fam, None)
        self.first_issued.pop(fam, None)
        self.n_credited += 1
        self.belief.update(observables(credited=True), strength=1.0)
        if pend is None:
            return False
        key, issued_at, _text = pend
        self.belief.observe_lag(max(0, self.now - issued_at))
        self.bank.observe(fam, key, True)
        return True

    # -- the law ---------------------------------------------------------------------------
    def decide(self) -> DIALDecision:
        b = self.belief
        ent = b.entropy()
        fam = self.family
        age = self.age()

        # --- (2) RESYNC is a veto, not a competitor -------------------------------------
        # If the attempt may still be in flight, changing the instruction destroys work that
        # cannot be recovered.  The window is `hold_horizon`, a high quantile of the measured
        # lag distribution, so that attempts with an above-median delay are still protected.
        if b.p(CAUSE_UNSYNCED) >= 0.35 and 0 <= age <= b.hold_horizon:
            return self._emit(A_RESYNC,
                              f"attempt may still be in flight (age={age} <= hold_horizon="
                              f"{b.hold_horizon:.0f}); hold the current attempt",
                              expected=b.p(CAUSE_UNSYNCED), information=0.0)

        # --- (2b) patience is bounded: past the escape horizon the attempt is dead ------
        # Measured against `age_total` (since the attempt CLASS was first chosen), never
        # against the refreshable patience clock: that is what keeps a repeatedly-re-enacted
        # attempt from being held forever, so the scheduler still cannot absorb.
        age_total = self.age_total()
        forced = age_total >= 0 and age_total > b.escape_horizon
        if forced:
            self.retire()                      # record the dead attempt as a negative example
            strat = self._pick(exclude_current=True)
            if strat is not None:
                self.n_forced_escape += 1
                return self._emit(A_RESAMPLE,
                                  f"attempt is dead (age_total={age_total} > escape_horizon="
                                  f"{b.escape_horizon:.0f}); forced escape to {strat.sid}",
                                  strat=strat, expected=1.0, information=self.info_weight * ent)

        # --- (2c) HOLD-FIRST: while the attempt may still be in flight, do not change it --
        # This is the T3 constraint stated as a rule rather than as a score term.  Measured:
        # the enacted->credited lag has median 25 and p90 130 steps, and 79.8% / 66.0% of
        # *successfully credited* attempts would still have been in flight at GPM's 8-step
        # and AOM's 12-step horizons.  Changing the attempt inside that window forfeits work
        # that the environment was about to confirm, and the archived successes are all
        # built on repeating one primitive until the stage fires.  A scoring term could be
        # outvoted by a confident posterior; a rule cannot, and the only way to leave this
        # branch is the bounded `escape_horizon` above, so the scheduler still cannot absorb.
        if 0 <= age <= b.hold_horizon:
            if b.p(CAUSE_UNSYNCED) >= 0.35:
                return self._emit(A_RESYNC,
                                  f"attempt may still be in flight (age={age} <= "
                                  f"hold_horizon={b.hold_horizon:.0f}); hold the current attempt",
                                  expected=b.p(CAUSE_UNSYNCED), information=0.0)
            return self._emit(A_PERSIST,
                              f"attempt may still be in flight (age={age} <= "
                              f"hold_horizon={b.hold_horizon:.0f}); repeat it",
                              strat=self._current_strategy(), expected=0.5, information=0.0)

        # --- (1) expected-credit column -------------------------------------------------
        # Each cause has exactly one addressing action, so the posterior argmax is the
        # expected-credit argmax.  That is what lets one law replace four schedulers.
        v: dict[str, float] = {}
        v[A_RESAMPLE] = (b.p(CAUSE_UNREACHABLE)
                         + 0.8 * b.p(CAUSE_UNGROUNDED)
                         + 0.9 * b.p(CAUSE_UNSAFE))
        v[A_QUERY] = b.p(CAUSE_UNKNOWN) + 0.5 * b.p(CAUSE_UNGROUNDED)
        v[A_RESYNC] = (0.0 if age < 0 else
                       b.p(CAUSE_UNSYNCED) * (1.0 if age <= b.hold_horizon else 0.0))
        v[A_PERSIST] = self.persist_threshold

        for a in ACTIONS:
            v[a] += self.info_weight * ent * (0.5 if a in (A_RESAMPLE, A_QUERY) else 0.1)
            v[a] -= _COST[a] * self.resample_cost

        order = sorted(ACTIONS, key=lambda a: -v[a])
        for a in order:
            if a == A_RESYNC:
                continue
            if a == A_RESAMPLE:
                strat = self._pick(exclude_current=True)
                if strat is None:
                    continue
                return self._emit(A_RESAMPLE,
                                  f"credit stalled; different attempt {strat.sid} for the "
                                  f"same obligation", strat=strat,
                                  expected=v[A_RESAMPLE], information=self.info_weight * ent)
            if a == A_QUERY:
                return self._emit(A_QUERY,
                                  "referent not observed; consult memory before naming it",
                                  expected=v[A_QUERY], information=self.info_weight * ent)
            if a == A_PERSIST:
                return self._emit(A_PERSIST, "current attempt is still the best available",
                                  strat=self._current_strategy(), expected=v[A_PERSIST],
                                  information=0.0)

        # Unreachable in practice: PERSIST is always in `order` and never `continue`s.
        # Guarded so this function can never return None, which would silently stop scheduling.
        return self._emit(A_PERSIST, "fallback: hold", strat=None, expected=0.0)

    def _current_strategy(self) -> Strategy | None:
        """Reconstruct the outstanding attempt as a `Strategy` (for a PERSIST directive)."""
        fam = self.family
        pend = self.pending.get(fam)
        if pend is None:
            return None
        key, _issued, text = pend
        return Strategy(obligation=fam, verb_form=key[0], approach=key[1],
                        hold_steps=key[2], text=text)

    def _pick(self, *, exclude_current: bool) -> Strategy | None:
        """Choose a genuinely different attempt, or None when none exists."""
        fam = self.family
        cur = self.last_key.get(fam)
        exclude = {cur} if (exclude_current and cur is not None) else set()
        strat = self.bank.select(fam, self.obj, attempt=int(self.attempts.get(fam, 0)),
                                exclude=exclude)
        if cur is not None and strat.key() == cur:
            # The bank could not offer a different cell.  Report failure rather than
            # silently re-issuing the attempt that did not work: re-issuing is exactly the
            # absorbing behaviour this architecture exists to remove.
            self.n_self_check_fail += 1
            return None
        return strat

    def _emit(self, action: str, reason: str, *, strat: Strategy | None = None,
              expected: float = 0.0, information: float = 0.0) -> DIALDecision:
        if action == A_RESAMPLE:
            self.n_resample += 1
            # Only a RESAMPLE defines the next attempt; PERSIST/QUERY/RESYNC must not
            # overwrite `last_key`, or the exclusion set would lose the failed attempt and
            # the next decision could pick it again.
            if strat is not None:
                self.last_key[self.family] = strat.key()
        elif action == A_QUERY:
            self.n_query += 1
        elif action == A_RESYNC:
            self.n_resync += 1
        else:
            self.n_persist += 1
        return DIALDecision(action=action, strategy=strat, reason=reason,
                            bottleneck=self.belief.argmax(), beta=dict(self.belief.beta),
                            expected_credit=expected, information=information,
                            value=expected + information)

    # -- reads -----------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "obligation": self.obligation, "family": self.family, "now": self.now,
            "attempts": dict(self.attempts), "age": self.age(),
            "age_total": self.age_total(), "n_reenact": self.n_reenact,
            "n_credited": self.n_credited,
            "counts": {"resample": self.n_resample, "query": self.n_query,
                       "resync": self.n_resync, "persist": self.n_persist,
                       "forced_escape": self.n_forced_escape,
                       "self_check_fail": self.n_self_check_fail},
            "belief": self.belief.snapshot(),
        }
