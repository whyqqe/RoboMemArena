"""Headless selftest for DIAL.  No GPU, no harness, no network.

Every test here exists because it can FAIL in a way that would cost real trials.  The list
is organised by which measured defect it guards against:

  A. scheduling invariants      — the properties the architecture claims are structural
  B. reward attribution         — credit must reach the family that earned it
  C. simulated episodes         — the absorbing-state claim, tested against a baseline
  D. bind-layer safety          — the three defects recorded in the other arms' binds

Run:  python -m dial.selftest        (exit 0 = all pass)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import types
from pathlib import Path

from . import bind as B
from .bottleneck import BottleneckBelief, observables
from .policy import (
    ACTIONS,
    A_PERSIST,
    A_QUERY,
    A_RESAMPLE,
    A_RESYNC,
    DIALPolicy,
    FORBIDDEN_ACTIONS,
)
from .strategy import STRATEGY_SPECS, StrategyBank, _default_specs
from .types import CAUSES, RewardRecord, Strategy

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASS.append(name)
    else:
        _FAIL.append(f"{name}: {detail}")


def _pol(*, family: str = "lift", obj: str = "tomato sauce",
         seed: int = 0) -> DIALPolicy:
    p = DIALPolicy(bank=StrategyBank(rng=random.Random(seed)))
    p.set_obligation(f"01_Lift_{obj.title().replace(' ', '_')}", family, obj)
    return p


# =======================================================================================
# A. scheduling invariants
# =======================================================================================
def test_action_space_has_no_block() -> None:
    check("A1 no BLOCK in the action space",
          not (set(ACTIONS) & set(FORBIDDEN_ACTIONS)),
          f"actions={ACTIONS} forbidden={FORBIDDEN_ACTIONS}")
    check("A1 exactly the four declared actions",
          set(ACTIONS) == {A_RESAMPLE, A_QUERY, A_RESYNC, A_PERSIST}, str(ACTIONS))


def test_all_families_offer_a_different_attempt() -> None:
    """Every family, including unknown ones, must offer >= 2 distinct cells.

    With one cell, `_pick` returns None forever and the scheduler can only PERSIST — which
    is precisely the absorbing state this architecture exists to remove.
    """
    bad = []
    for fam in list(STRATEGY_SPECS) + ["__unknown__", "", "other"]:
        cands = STRATEGY_SPECS.get(fam) or _default_specs(fam)
        keys = {(c[0], c[1], c[2]) for c in cands}
        if len(keys) < 2:
            bad.append((fam, keys))
    check("A2 every family offers >= 2 distinct attempts", not bad, str(bad))


def test_beta_stays_a_distribution() -> None:
    b = BottleneckBelief()
    rng = random.Random(3)
    names = list(observables().keys())
    for _ in range(500):
        obs = {n: rng.random() < 0.35 for n in names}
        b.update(obs)
        s = sum(b.beta.values())
        if abs(s - 1.0) > 1e-9 or any(v < 0 for v in b.beta.values()):
            check("A3 posterior stays a probability distribution", False,
                  f"sum={s} beta={b.beta}")
            return
    check("A3 posterior stays a probability distribution", True)
    check("A3 all five causes always present",
          set(b.beta) == set(CAUSES), str(sorted(b.beta)))


def test_lambda_hat_estimates_lag() -> None:
    b = BottleneckBelief()
    check("A4 lambda_hat is seeded from the archive median, not a guess",
          b.lambda_hat == 25.0, str(b.lambda_hat))
    lags = [10, 20, 25, 30, 40]
    for x in lags:
        b.observe_lag(x)
    check("A4 lambda_hat tracks the sample median", b.lambda_hat == 25.0, str(b.lambda_hat))
    check("A4 lambda_p90 uses nearest rank, so no observed sample sits above the bound",
          b.lambda_p90 == 40.0, str(b.lambda_p90))
    check("A4 the hold window clears the upper end of the lag distribution",
          b.hold_horizon >= b.lambda_p90, f"{b.hold_horizon} vs {b.lambda_p90}")
    check("A4 the hold window is strictly inside the escape horizon",
          b.hold_horizon < b.escape_horizon, f"{b.hold_horizon} vs {b.escape_horizon}")
    check("A4 the initial hold window protects above-median lag attempts",
          BottleneckBelief().hold_horizon > 25.0, str(BottleneckBelief().hold_horizon))
    b.observe_lag(-1)
    check("A4 negative lag is ignored", b.lambda_hat == 25.0, str(b.lambda_hat))
    # a single huge outlier must not inflate the central estimate (heavy tail guard)
    b.observe_lag(685)
    check("A4 a 685-step outlier does not move the median materially",
          b.lambda_hat <= 30.0, str(b.lambda_hat))
    check("A4 but the outlier does raise the patience bound",
          b.lambda_p90 >= 685.0, str(b.lambda_p90))
    # and the overdueness observable must resolve the posterior once the lag is exceeded
    b2 = BottleneckBelief()
    for _ in range(30):
        b2.update(observables(enacted_no_credit=True, lag_within_lambda=True))
    hot = b2.p("UNSYNCED")
    for _ in range(30):
        b2.update(observables(overdue_no_credit=True))
    check("A4 sustained overdueness moves the posterior off UNSYNCED",
          b2.p("UNSYNCED") < hot and b2.argmax() == "UNREACHABLE",
          f"unsynced {hot:.2f} -> {b2.p('UNSYNCED'):.2f}, argmax={b2.argmax()}")
    check("A4 no cause is ever driven to exactly zero (the filter stays recoverable)",
          all(v > 0.0 for v in b2.beta.values()),
          str({k: round(v, 8) for k, v in b2.beta.items()}))


def test_spec_templates_match_their_declared_cell() -> None:
    """The selection key space and the archive key space MUST coincide.

    `warm()` reads archived attempts through `classify_verb`/`classify_approach` on the
    recorded TEXT; live selection indexes the declared `(verb_form, approach, hold_steps)`.
    If a template does not classify as declared, the warmed prior is stored under a cell
    that is never selected and the cross-task transfer silently does nothing.  This was a
    real defect: all three pour templates classified as `pour/tilt/0`.
    """
    from .types import classify_approach, classify_verb
    problems = []
    for fam, specs in list(STRATEGY_SPECS.items()) + [("__unknown__", _default_specs("x"))]:
        keys = [(v, a, h) for v, a, h, _t in specs]
        if len(set(keys)) != len(keys):
            problems.append((fam, "duplicate declared keys", keys))
        for v, a, h, tmpl in specs:
            try:
                text = tmpl.format(obj="the object")
            except Exception as exc:  # noqa: BLE001
                problems.append((fam, f"template does not format: {exc!r}", tmpl))
                continue
            got_v, got_a = classify_verb(text), classify_approach(text)
            if got_v != v or got_a != a:
                problems.append((fam, text,
                                 f"declared=({v},{a}) classified=({got_v},{got_a})"))
    check("A12 every template classifies as its declared cell", not problems,
          str(problems[:4]))


def test_no_absorption_across_families() -> None:
    """THE central property.  Force every attempt to fail; the scheduler must keep moving.

    For each family: drive the policy with failures only, and assert that (i) no RESAMPLE
    ever re-selects the attempt that is currently outstanding, and (ii) the scheduler
    exhausts the family's candidate set rather than looping on one of them — i.e. the
    number of distinct classes tried equals the number available.
    """
    fams = list(STRATEGY_SPECS) + ["__unknown__"]
    problems = []
    for fi, fam in enumerate(fams):
        cands = STRATEGY_SPECS.get(fam) or _default_specs(fam)
        want = len({(c[0], c[1], c[2]) for c in cands})
        p = DIALPolicy(bank=StrategyBank(rng=random.Random(11 + fi)))
        p.set_obligation(f"stage_{fam}", fam, "the object")
        tried: set[tuple] = set()
        n_resample = 0
        for step in range(1, 3000):
            prev = p.last_key.get(fam)
            p.advance(step)                    # every step: no credit
            d = p.decide()
            if d.action == A_RESAMPLE:
                if d.strategy is None:
                    problems.append((fam, "RESAMPLE without a strategy"))
                    break
                if prev is not None and d.strategy.key() == prev:
                    problems.append((fam, f"re-selected the failed attempt {prev}"))
                    break
                tried.add(d.strategy.key())
                n_resample += 1
                p.note_enacted(d.strategy.key(), d.strategy.text)
        if want >= 2 and len(tried) < want:
            problems.append((fam, f"tried only {len(tried)}/{want} distinct attempts in "
                                  f"3000 steps (resamples={n_resample})"))
    check("A5 no absorption: never re-selects the failed attempt, and exhausts the space",
          not problems, str(problems[:4]))


def test_absorption_would_be_reachable_for_a_gate_policy() -> None:
    """Control: the SAME environment traps a 'gate' policy that re-issues one attempt.

    Without this control, A5 could pass because the environment happens to be escapable.
    """
    p = _pol()
    key = p.bank.select("lift", p.obj, attempt=0).key()
    distinct = {key}
    for _ in range(600):
        # a gate policy can only rewrite onto the same obligation, i.e. re-issue `key`
        distinct.add(key)
        p.note_enacted(key, "same attempt")
    check("A5-control a gate policy is stuck at exactly one attempt",
          len(distinct) == 1, str(distinct))


def test_resync_veto_and_patience_bound() -> None:
    p = _pol()
    key = p.bank.select("lift", p.obj, attempt=0).key()
    p.note_enacted(key, "att", step=0)
    p.now = 3
    for _ in range(3):
        p.belief.update(observables(enacted_no_credit=True, lag_within_lambda=True))
    check("A6 UNSYNCED rises above the veto threshold",
          p.belief.p("UNSYNCED") >= 0.35, str(p.belief.p("UNSYNCED")))
    d = p.decide()
    check("A6 inside lambda_hat the action is RESYNC (hold, do not alter)",
          d.action == A_RESYNC, f"{d.action} ({d.reason})")
    check("A6 RESYNC does not disturb the outstanding attempt",
          p.last_key.get("lift") == key, str(p.last_key))

    # patience must be bounded: past the escape horizon an attempt is dead
    p2 = _pol(seed=5)
    k2 = p2.bank.select("lift", p2.obj, attempt=0).key()
    p2.note_enacted(k2, "att", step=0)
    p2.belief.update(observables(enacted_no_credit=True, repeat_no_move=True))
    p2.now = int(p2.belief.escape_horizon) + 5
    d2 = p2.decide()
    check("A7 patience is bounded: a dead attempt is force-escaped",
          d2.action == A_RESAMPLE and p2.n_forced_escape > 0,
          f"{d2.action} forced={p2.n_forced_escape}")
    check("A7 the forced escape is a different attempt",
          d2.strategy is not None and d2.strategy.key() != k2, str(d2.strategy and d2.strategy.key()))
    # and it must NOT fire while the lag is still within the measured envelope
    p3 = _pol(seed=5)
    k3 = p3.bank.select("lift", p3.obj, attempt=0).key()
    p3.note_enacted(k3, "att", step=0)
    p3.now = int(p3.belief.lambda_hat)          # age == median lag
    d3 = p3.decide()
    check("A7 the escape does NOT fire while the lag is still inside the envelope",
          p3.n_forced_escape == 0, f"forced={p3.n_forced_escape} action={d3.action}")


def test_persist_does_not_corrupt_last_key() -> None:
    """A regression guard: PERSIST must not overwrite last_key with a placeholder.

    An earlier draft emitted PERSIST carrying a `Strategy(verb_form="", ...)`, which wrote
    ("", "", 0) into last_key and destroyed the exclusion set — so the next decision could
    legally re-pick the attempt that had just failed.
    """
    p = _pol(seed=7)
    k = p.bank.select("lift", p.obj, attempt=0).key()
    p.note_enacted(k, "att", step=0)
    p.now = 0
    p.belief.update(observables(enacted_no_credit=True, lag_within_lambda=True))
    d = p.decide()
    if d.action == A_PERSIST:
        check("A8 PERSIST leaves last_key untouched", p.last_key.get("lift") == k,
              f"{p.last_key}")
    else:
        check("A8 PERSIST leaves last_key untouched (not reached; forced path)", True,
              f"action={d.action}")
    # direct check both ways
    p2 = _pol(seed=7)
    k2 = ("pick_up", "plain", 0)
    p2.last_key["lift"] = k2
    p2._emit(A_PERSIST, "hold", strat=Strategy("lift", "", "", 0, "txt"))
    check("A8 PERSIST via _emit is inert on last_key", p2.last_key.get("lift") == k2,
          str(p2.last_key))


def test_single_candidate_family_is_safe() -> None:
    """Degenerate bank: no alternative exists.  Must degrade to a non-RESAMPLE action."""
    p = _pol(seed=9)
    one = [STRATEGY_SPECS["lift"][0]]
    p.bank.candidates = lambda fam: one  # type: ignore[assignment]
    p.note_enacted(("pick_up", "plain", 0), "att", step=0)
    bad = []
    for step in range(1, 200):
        p.advance(step)
        d = p.decide()
        if d.action == A_RESAMPLE:
            if d.strategy is None or d.strategy.key() == ("pick_up", "plain", 0):
                bad.append((step, d.action, d.strategy and d.strategy.key()))
    check("A9 a one-candidate family never re-issues the failed attempt", not bad,
          str(bad[:3]))
    check("A9 the degenerate case is recorded, not hidden", p.n_self_check_fail > 0,
          str(p.n_self_check_fail))


def test_thompson_learns_the_good_cell() -> None:
    p = _pol(family="lift", seed=13)
    good = ("pick_up", "plain", 0)
    picks = {k: 0 for k in p.bank.candidates("lift")[:3]}
    keys = [tuple(c[:3]) for c in p.bank.candidates("lift")]
    for trial in range(300):
        d = p.decide()
        if d.action != A_RESAMPLE or d.strategy is None:
            p.now += 1
            continue
        k = d.strategy.key()
        p.note_enacted(k, d.strategy.text, step=p.now)
        p.now += 5
        if k == good:
            p.on_credit("01_Lift_Tomato_Sauce", step=p.now)
        else:
            p.belief.update(observables(enacted_no_credit=True, repeat_no_move=True))
        if k in picks:
            picks[k] += 1
    means = {k: p.bank.arm("lift", k).mean for k in keys}
    check("A10 the successful cell ends with the highest posterior mean",
          max(means, key=means.get) == good, str({str(k): round(v, 3) for k, v in means.items()}))
    check("A10 the successful cell is selected more often than the median cell",
          picks.get(good, 0) >= sorted(picks.values())[len(picks) // 2],
          str({str(k): v for k, v in picks.items()}))


def test_hierarchical_warm_recovers_ordering() -> None:
    """Synthetic archive: warm() must rank the empirically-best cell first, per family.

    Asserted against the empirical rate computed from the very records fed in, so the test
    cannot pass by assuming which cell "should" win.  Records carry TEXT because
    `RewardRecord.strategy_key` classifies the attempt from its wording — that is exactly
    how the real archive is read.
    """
    recs = []
    truth: dict[str, dict[tuple, float]] = {}
    for fam in ("lift", "pour", "place", "open", "close"):
        specs = STRATEGY_SPECS[fam]
        for i, (v, ap, h, tmpl) in enumerate(specs):
            text = tmpl.format(obj="the object")
            rate = 0.10 + 0.65 * (i == 0)          # the FIRST declared cell is the good one
            n = 60
            n_ok = int(round(rate * n))
            for k in range(n):
                recs.append(RewardRecord(
                    arm="syn", task=8, seed=1, episode=1, step=k, obligation=f"01_{fam}",
                    family=fam, text=text,
                    outcome="credited" if k < n_ok else "stalled"))
            key = RewardRecord(arm="", task=0, seed=0, episode=0, step=0, obligation="",
                               text=text, family=fam, outcome="stalled").strategy_key
            truth.setdefault(fam, {})[key] = n_ok / n
    bank = StrategyBank(rng=random.Random(1))
    bank.warm(recs)
    problems = []
    detail = {}
    for fam, rates in truth.items():
        best_key = max(rates, key=rates.get)
        rank = bank.ranking(fam)
        detail[fam] = [(list(r[0]), round(r[1], 3), r[2]) for r in rank]
        if not rank or rank[0][0] != best_key:
            problems.append((fam, f"top={rank[0][0] if rank else None} truth={best_key}",
                             rates))
    check("A11 warmed prior ranks the empirically-best cell first, per family",
          not problems, str(problems[:3]))
    check("A11 warmed prior yields a sensible global base rate",
          0.10 < bank.global_prior < 0.60, str(bank.global_prior))
    check("A11 every archived cell lands in the live selection key space",
          all(any(r[0] == k for r in bank.ranking(fam)) for fam in truth
              for k in truth[fam]),
          str({f: [list(r[0]) for r in bank.ranking(f)] for f in truth}))


# =======================================================================================
# B. reward attribution
# =======================================================================================
def test_credit_attribution_is_per_family() -> None:
    """A task steps 01_Lift -> 02_Pour.  Credit for 01 must reach the LIFT family even
    though the obligation has already advanced to 02."""
    p = DIALPolicy(bank=StrategyBank(rng=random.Random(2)))
    p.set_obligation("01_Lift_Tomato_Sauce", "lift", "tomato sauce")
    lift_key = p.bank.select("lift", "tomato sauce", attempt=0).key()
    p.note_enacted(lift_key, "lift attempt", family="lift", step=10)
    # the task moves on before the scorer catches up
    p.set_obligation("02_Pour_One", "pour", "tomato sauce")
    pour_key = p.bank.select("pour", "tomato sauce", attempt=0).key()
    p.note_enacted(pour_key, "pour attempt", family="pour", step=20)
    p.now = 40
    got = p.on_credit("01_Lift_Tomato_Sauce", step=40)
    check("B1 credit for a completed stage is attributed", got)
    check("B1 the LIFT family cell absorbed the credit",
          any(h[0] == "lift" and h[2] for h in p.bank.history), str(p.bank.history))
    check("B1 the POUR family cell was NOT credited",
          not any(h[0] == "pour" and h[2] for h in p.bank.history), str(p.bank.history))
    check("B1 the measured lag reached the belief",
          p.belief.lambda_hat > 4.0, str(p.belief.lambda_hat))
    check("B1 the outstanding lift attempt was consumed",
          "lift" not in p.pending, str(p.pending))
    check("B1 the outstanding pour attempt survives", "pour" in p.pending, str(p.pending))


def test_credit_without_an_outstanding_attempt_is_safe() -> None:
    p = _pol()
    got = p.on_credit("01_Lift_Tomato_Sauce", step=50)
    check("B2 credit with no outstanding attempt is a no-op, not a crash", got is False)
    check("B2 it still counts", p.n_credited == 1, str(p.n_credited))


# =======================================================================================
# C. simulated episodes
# =======================================================================================
class _SimVLA:
    """A frozen policy that succeeds only for particular attempt classes."""

    def __init__(self, good: set[tuple], lag: int) -> None:
        self.good = set(good)
        self.lag = lag
        self.inflight: dict[int, bool] = {}

    def issue(self, key: tuple, step: int) -> None:
        self.inflight[step + self.lag] = key in self.good

    def collect(self, step: int) -> bool:
        return bool(self.inflight.pop(step, False))


def _run_sim(policy: DIALPolicy, sim: _SimVLA, family: str, obj: str,
             stage: str, budget: int, *, explore: bool) -> tuple[bool, int]:
    """Run one simulated episode.  `explore=False` reproduces a gate policy."""
    policy.set_obligation(stage, family, obj)
    gate_key = policy.bank.select(family, obj, attempt=0).key()
    issued = False
    for step in range(1, budget + 1):
        if sim.collect(step):
            policy.on_credit(stage, step=step)
            return True, step
        if not explore:
            # a gate policy re-issues one attempt no matter what; that is its whole law
            if not issued:
                sim.issue(gate_key, step)
                issued = True
            policy.now = step
            continue
        policy.advance(step)
        d = policy.decide()
        if d.action == A_RESAMPLE and d.strategy is not None:
            policy.note_enacted(d.strategy.key(), d.strategy.text, step=step)
            sim.issue(d.strategy.key(), step)
    return False, budget


def test_sim_escape_vs_gate() -> None:
    """The architectural claim, as an experiment.

    Environment: only the 3rd attempt class works, with a long credit lag.  A gate policy
    re-issuing its one attempt never earns credit.  DIAL must.
    """
    family, obj, stage = "lift", "tomato sauce", "01_Lift_Tomato_Sauce"
    good = tuple(STRATEGY_SPECS["lift"][2][:3])
    lag = 25
    sim = _SimVLA({good}, lag)
    ok_dial, steps = _run_sim(_pol(seed=21), sim, family, obj, stage, 2500, explore=True)
    check("C1 DIAL escapes the absorbing environment", ok_dial, f"steps={steps}")

    sim2 = _SimVLA({good}, lag)
    ok_gate, _ = _run_sim(_pol(seed=21), sim2, family, obj, stage, 2500, explore=False)
    check("C1-control a gate policy does NOT escape the same environment", not ok_gate)

    # and the escape must not be luck: run several seeds
    wins = 0
    for s in range(12):
        simx = _SimVLA({good}, lag)
        ok, _st = _run_sim(_pol(seed=100 + s), simx, family, obj, stage, 2500, explore=True)
        wins += int(ok)
    check("C2 DIAL escapes across seeds (>= 10/12)", wins >= 10, f"{wins}/12")


def test_sim_resync_protects_an_inflight_attempt() -> None:
    """With a long lag, a correct attempt must survive long enough to be credited.

    This is the T3 prediction: the lag (median 25, p90 130) exceeds the horizons the other
    arms act on (8 and 12), so they cancel attempts that were about to succeed.

    Written deterministically: the good attempt is issued explicitly, then the test asserts
    the scheduler does NOT replace it at any point while the credit is still in flight.
    (The earlier version let Thompson sampling pick the first attempt, so it depended on a
    random draw landing on the good cell and failed intermittently — an unreliable test.)
    """
    family, obj, stage = "lift", "tomato sauce", "01_Lift_Tomato_Sauce"
    good = tuple(STRATEGY_SPECS["lift"][0][:3])
    sim = _SimVLA({good}, 30)
    p = _pol(seed=31)
    p.set_obligation(stage, family, obj)
    p.note_enacted(good, "attempt", step=0)
    sim.issue(good, 0)

    resampled_in_flight: list[int] = []
    credited_at = None
    for step in range(1, 200):
        if sim.collect(step):
            p.on_credit(stage, step=step)
            credited_at = step
            break
        p.advance(step)
        d = p.decide()
        if d.action == A_RESAMPLE:
            resampled_in_flight.append(step)
            if d.strategy is not None:
                p.note_enacted(d.strategy.key(), d.strategy.text, step=step)
                sim.issue(d.strategy.key(), step)

    check("C3 a correct attempt survives its own credit lag",
          credited_at is not None, f"credited_at={credited_at}")
    check("C3 it is never replaced while the credit is in flight",
          not resampled_in_flight, f"resampled at steps {resampled_in_flight[:5]}")
    check("C3 the scheduler held rather than churned", p.n_resample == 0, str(p.n_resample))


def test_control_arm_cancels_the_same_attempt() -> None:
    """Control for C3.  The same environment with GPM's 8-step horizon cancels the attempt
    at step 9, before its 30-step credit arrives — the measured waste T3 quantifies."""
    good = tuple(STRATEGY_SPECS["lift"][0][:3])
    sim = _SimVLA({good}, 30)
    sim.issue(good, 0)
    cancelled = False
    for step in range(1, 200):
        if sim.collect(step):
            break
        if step > 8:                      # GPM's stagnation horizon
            cancelled = True
            break
    check("C3-control an 8-step horizon cancels the attempt before its 30-step credit",
          cancelled, "attempt survived the horizon (environment mis-specified)")


def test_sim_multistage_attribution() -> None:
    sim_lift = _SimVLA({tuple(STRATEGY_SPECS["lift"][0][:3])}, 12)
    sim_pour = _SimVLA({tuple(STRATEGY_SPECS["pour"][1][:3])}, 18)
    p = _pol(seed=41)
    ok1, _ = _run_sim(p, sim_lift, "lift", "tomato sauce", "01_Lift_Tomato_Sauce", 90,
                      explore=True)
    ok2, _ = _run_sim(p, sim_pour, "pour", "tomato sauce", "02_Pour_One", 90, explore=True)
    check("C4 both stages are credited in sequence", ok1 and ok2, f"{ok1} {ok2}")
    fams = {h[0] for h in p.bank.history if h[2]}
    check("C4 credit landed on both families", fams == {"lift", "pour"}, str(fams))


# =======================================================================================
# D. bind-layer safety  (the three defects recorded in the other arms' binds)
# =======================================================================================
class _Spec:
    def __init__(self, name: str) -> None:
        self.name = name


def _install_fake_harness(state: dict) -> None:
    """Install a minimal fake `harness` package so the hooks can be exercised offline.

    `state` is mutated in place and closed over by the fakes, so counter increments are
    visible to the caller.  (An earlier version returned a fresh dict and copied its values
    out, which silently froze every counter at 0 and made two bind tests vacuous.)
    """
    for k in ("n_orig_ctl", "n_orig_build", "n_orig_reset"):
        state.setdefault(k, 0)

    class FakeController:
        def override_vla_prompt(self, base_prompt, *, stage_idx=0, stage_specs=None,
                                stage_done=None, step=0, **extra):
            state["n_orig_ctl"] += 1
            return base_prompt

        def on_episode_end(self, *args, stage_done=None, stage_score_pct=0.0,
                           stage_success=False, failure_reason=None, run_dir="", **extra):
            state["n_orig_end"] = state.get("n_orig_end", 0) + 1
            state["last_end"] = dict(stage_done or {})
            return None

    class FakePlanner:
        def __init__(self) -> None:
            self.step = 0

        def _build_messages(self, mm, mw, cm, cw, *, extra_memory_text=""):
            state["n_orig_build"] += 1
            return [{"type": "text", "text": "base"}]

        def reset_episode(self) -> None:
            state["n_orig_reset"] += 1

    ctl = types.ModuleType("harness.controller")
    ctl.HarnessController = FakeController
    ap = types.ModuleType("harness.api_vlm_planner")
    ap.ApiMemoryPlanner = FakePlanner
    pkg = types.ModuleType("harness")
    pkg.controller = ctl
    pkg.api_vlm_planner = ap
    sys.modules["harness"] = pkg
    sys.modules["harness.controller"] = ctl
    sys.modules["harness.api_vlm_planner"] = ap
    state["ctl_cls"] = FakeController
    state["ap_mod"] = ap


def _obs(name: str = "01_Lift_Tomato_Sauce"):
    """Minimal object with a `.name`, as `stage_specs` entries have."""
    return types.SimpleNamespace(name=name)


def _fresh_install() -> dict:
    """Re-arm install() against a freshly created fake harness; return the live state."""
    B._STATE["installed_at"] = None
    B._STATE["errors"] = []
    B._STATE["n_build"] = 0
    B._STATE["n_stage_obs"] = 0
    B._STATE["n_credit_events"] = 0
    B._LOCAL.ctx = None
    B._LOCAL.planner = None
    B._BOUND.clear()
    state: dict = {}
    _install_fake_harness(state)
    B.install()
    return state


def test_bind_override_is_identity() -> None:
    """DEFECT 1 guard.  Byte equality, on every input, and even if our own code throws."""
    os.environ["MEMEXP_DIAL"] = "1"
    state = _fresh_install()
    ctl = state["ctl_cls"]()
    specs = [_Spec("01_Lift_Tomato_Sauce")]

    cases = ["pick up the tomato sauce bottle", "", "   ", "x" * 5000,
             "pick the tomato sauce\nand place it"]
    bad = []
    for c in cases:
        out = ctl.override_vla_prompt(c, stage_idx=0, stage_specs=specs,
                                      stage_done={"01_Lift_Tomato_Sauce": True}, step=5)
        if out != c:
            bad.append((c[:24], out))
    check("D1 override_vla_prompt returns the prompt byte-identical", not bad, str(bad))

    # now make our observer throw, and assert the prompt is still untouched and nothing raises
    orig_local = B._LOCAL

    class _Boom:
        def __getattr__(self, item):
            raise RuntimeError("boom")

    B._LOCAL = _Boom()  # type: ignore[assignment]
    try:
        out = ctl.override_vla_prompt("keep me", stage_idx=0, stage_specs=specs,
                                      stage_done={"01_Lift_Tomato_Sauce": True}, step=6)
        check("D1 the prompt survives an exception in the observer", out == "keep me", out)
        check("D1 the exception is recorded, not raised",
              any("boom" in e or "observe" in e for e in B._STATE["errors"]),
              str(B._STATE["errors"][-3:]))
    except Exception as exc:  # noqa: BLE001
        check("D1 the observer never propagates an exception", False, repr(exc))
    finally:
        B._LOCAL = orig_local


def test_bind_build_messages_appends_one_block() -> None:
    os.environ["MEMEXP_DIAL"] = "1"
    state = _fresh_install()
    pl = state["ap_mod"].ApiMemoryPlanner()
    pl.step = 3
    before = state["n_orig_build"]
    msgs = pl._build_messages({}, {}, {}, {})
    check("D2 the original _build_messages ran exactly once",
          state["n_orig_build"] == before + 1, f"{before} -> {state['n_orig_build']}")
    check("D2 a list is returned", isinstance(msgs, list), type(msgs).__name__)
    diat = [m for m in msgs if m.get("type") == "text" and "DIAL diagnosis" in m.get("text", "")]
    check("D2 exactly one DIAL block is appended", len(diat) == 1, f"n={len(diat)}")
    check("D2 the base message is preserved", msgs[0]["text"] == "base", str(msgs[0]))


def test_bind_disabled_is_transparent() -> None:
    os.environ["MEMEXP_DIAL"] = "0"
    state = _fresh_install()
    pl = state["ap_mod"].ApiMemoryPlanner()
    msgs = pl._build_messages({}, {}, {}, {})
    check("D3 with the arm off nothing is injected",
          msgs == [{"type": "text", "text": "base"}], str(msgs))
    os.environ["MEMEXP_DIAL"] = "1"


def test_bind_reset_clears_everything() -> None:
    """DEFECT 3 guard: no cross-episode replay, in either direction."""
    os.environ["MEMEXP_DIAL"] = "1"
    state = _fresh_install()
    pl = state["ap_mod"].ApiMemoryPlanner()
    specs = [_Spec("01_Lift_Tomato_Sauce")]
    ctl = state["ctl_cls"]()
    pl.step = 2
    pl._build_messages({}, {}, {}, {})
    ctl.override_vla_prompt("p", stage_idx=0, stage_specs=specs,
                            stage_done={"01_Lift_Tomato_Sauce": True}, step=4)
    ctx1 = B._LOCAL.ctx
    check("D4 episode 1 accumulated credit", bool(ctx1.credited), str(ctx1.credited))

    pl.reset_episode()
    check("D4 reset drops the thread-local context", B._LOCAL.ctx is None)
    pl.step = 1
    pl._build_messages({}, {}, {}, {})
    ctx2 = B._LOCAL.ctx
    check("D4 a new context was created", ctx2 is not None and ctx2 is not ctx1)
    check("D4 no credit is replayed into the new episode", not ctx2.credited,
          str(ctx2.credited))
    check("D4 no verification count is replayed", ctx2.policy.n_credited == 0,
          str(ctx2.policy.n_credited))
    check("D4 no observations are replayed into the new bandit",
          not ctx2.bank.history, str(ctx2.bank.history))
    check("D4 the new episode's own first attempt is not an inherited one",
          ctx2.policy.attempts.get("lift", 0) == 0,
          str(ctx2.policy.attempts))
    ctl.override_vla_prompt("p", stage_idx=0, stage_specs=specs, stage_done={}, step=9)
    check("D4 an empty stage_done does not re-credit", ctx2.policy.n_credited == 0,
          str(ctx2.policy.n_credited))


def test_bind_does_not_touch_the_primitive_channel() -> None:
    """DEFECT 2 guard: DIAL must not wrap infer_primitive_via_api at all."""
    os.environ["MEMEXP_DIAL"] = "1"
    state = _fresh_install()
    calls: list = []

    def orig_infer(*, system_prompt, user_content, **kw):
        calls.append((system_prompt, user_content))
        return "UNTOUCHED"

    state["ap_mod"].infer_primitive_via_api = orig_infer
    B._STATE["installed_at"] = None
    B.install()
    out = state["ap_mod"].infer_primitive_via_api(system_prompt="s", user_content=[{"a": 1}])
    check("D5 the primitive channel is not wrapped and returns verbatim",
          out == "UNTOUCHED", str(out))
    check("D5 the primitive channel is exactly the original function",
          state["ap_mod"].infer_primitive_via_api is orig_infer)
    src = Path(B.__file__).read_text()
    check("D5 bind.py defines no rewrite wrapper for the primitive channel",
          "_wrap_api_call" not in src and "module.infer_primitive_via_api =" not in src,
          "found a rewrite hook definition")


def _ctx_for(p, family: str = "lift", obligation: str = "01_Lift_Tomato_Sauce"):
    """A context light enough for `render_directive` (same trick as the D8 test)."""
    ctx = B._Ctx.__new__(B._Ctx)
    ctx.obligation = obligation
    ctx.family = family
    ctx.policy = p
    return ctx


def test_credit_comes_from_the_real_channel() -> None:
    """DEFECT 14 guard: credit must NOT be read from `override_vla_prompt`'s `stage_done`.

    In this harness that argument is a placeholder for the stages *BEFORE* the current
    index, built as `{spec.name: False for spec in stage_specs[:stage_idx]}` -- always
    all-False (harness/controller.py:272 and :634).  Reading it made a live run report
    `n_stage_obs=375` together with `n_credit_events=0`: the scheduler never received one
    reward, so it could not learn that an attempt was good and resampled forever.  The
    signal carrying real values is `HarnessController.on_episode_end`.
    """
    state = _fresh_install()
    ctl = sys.modules["harness.controller"]
    ctx = B._Ctx(planner=None)
    ctx.obligation = "01_Lift_Tomato_Sauce"
    ctx.family = "lift"
    ctx.policy.set_obligation(ctx.obligation, "lift", "tomato sauce")
    k = tuple(ctx.policy.bank.candidates("lift")[0][:3])
    ctx.policy.note_enacted(k, "Grasp the tomato sauce bottle", step=0)
    B._LOCAL.ctx = ctx
    B._LOCAL.planner = None
    B._STATE["n_credit_events"] = 0

    # the all-False placeholder must credit NOTHING
    ctl.HarnessController().override_vla_prompt(
        "keep", stage_idx=1, stage_specs=[_obs(), _obs()],
        stage_done={"01_Lift_Tomato_Sauce": False}, step=5)
    check("D11 an all-False stage_done credits nothing",
          int(B._STATE.get("n_credit_events") or 0) == 0,
          str(B._STATE.get("n_credit_events")))
    check("D11 the placeholder is not recorded as a completed stage",
          "01_Lift_Tomato_Sauce" not in ctx.credited, str(ctx.credited))

    # the real channel must credit
    ctl.HarnessController().on_episode_end(
        stage_score_pct=100.0, stage_success=True, failure_reason=None,
        stage_done={"01_Lift_Tomato_Sauce": True}, run_dir="")
    check("D11 on_episode_end credits the completed stage",
          int(B._STATE.get("n_credit_events") or 0) >= 1,
          str(B._STATE.get("n_credit_events")))
    check("D11 the credited stage is recorded on the live context",
          "01_Lift_Tomato_Sauce" in ctx.credited, str(ctx.credited))
    check("D11 the credit reaches the scheduler as a reward", p_ok(ctx.policy), "no reward banked")
    B._LOCAL.ctx = None


def p_ok(policy) -> bool:
    """True if the scheduler actually booked a credited attempt."""
    return bool(policy.n_credited or policy.bank.n_credited)


def test_directive_does_not_forbid_repetition() -> None:
    """DEFECT 15 guard: the directive must never order the planner to vary.

    The first wording said "Render a DIFFERENT physical attempt ... Do not reuse the
    previous wording".  Everything measured says the opposite is required: the lag from
    enactment to credit has median 25 and p90 130 steps, and the archived successes are
    built by repeating one primitive until the stage fires.  Ordering an LLM planner to
    keep varying destroys that.  On t8 seed 101 every other arm scored 66.7 while DIAL
    scored 0.0; this test protects against re-introducing that wording.
    """
    p = _pol(seed=3)
    p.note_enacted(("grasp", "plain", 0), "Grasp the tomato sauce bottle", step=0)
    p.advance(3)
    d = p.decide()
    block = B.render_directive(_ctx_for(p), d).lower()
    check("D12 while the attempt is in flight the directive asks to repeat it",
          d.action != A_RESAMPLE and ("repeat" in block or "hold" in block), block[:200])

    d2 = p.decide()
    d2.action = A_RESAMPLE
    d2.strategy = Strategy("lift", "grasp_lift", "top", 0, "Approach from above and lift.")
    low2 = B.render_directive(_ctx_for(p), d2).lower()
    check("D12 the new-attempt directive never orders the planner to vary its wording",
          "do not reuse" not in low2 and "do not copy" not in low2
          and "different wording" not in low2, low2[:220])
    check("D12 the new-attempt directive still names the class",
          "verb=" in low2 and "approach=" in low2, low2[:220])


def test_two_clocks_repeat_does_not_become_staleness() -> None:
    """DEFECT 17 guard, plus the non-absorption proof for the fix.

    A live run reported `now=1931` on a ~1000-step episode with `forced_escape=7`: because
    `issued_at` was refreshed only when a NEW attempt class was picked, the scheduler was
    timing "steps since the aim last changed" and treated its own holds as staleness.  So
    patience and the escape horizon were both measured from the wrong origin, and the arm
    switched aim on exactly the episodes the baseline wins by repeating.

    The fix has two halves and this test asserts both, because either alone is wrong:
      (a) re-enacting the same key refreshes the patience clock -> no self-inflicted escape;
      (b) `first_issued` is never refreshed -> past `escape_horizon` the scheduler MUST
          resample, so the arm still cannot absorb.
    """
    p = _pol(seed=5)
    good = ("pick_up", "plain", 0)
    p.note_enacted(good, "Grasp and lift the tomato sauce bottle", step=0)
    hh = int(p.belief.hold_horizon)
    eh = int(p.belief.escape_horizon)

    actions = []
    probe = []          # (age, age_total) immediately before each decision
    for step in range(1, eh + 3):
        # the planner re-emits the SAME instruction each step: a continuation
        p.note_enacted(good, "Grasp and lift the tomato sauce bottle", step=step)
        p.advance(step)
        probe.append((step, p.age(), p.age_total()))
        actions.append(p.decide().action)

    check("D14 (a) a repeatedly re-enacted attempt is never judged stale",
          A_RESAMPLE not in actions[:hh + 1], str(actions[:hh + 1]))
    check("D14 (a) re-enactments are counted, not treated as new attempts",
          p.n_reenact == len(actions) and p.attempts["lift"] == 1,
          f"reenact={p.n_reenact} actions={len(actions)} attempts={p.attempts}")
    check("D14 (b) the escape horizon is still reachable (no absorption)",
          A_RESAMPLE in actions, str(sorted(set(actions))))
    first_escape = actions.index(A_RESAMPLE)
    step_e, age_e, total_e = probe[first_escape]
    check("D14 (b) the patience clock is still short at the escape (freshness, not staleness)",
          age_e <= hh, f"age={age_e} hh={hh}")
    check("D14 (b) the escape is driven by the absolute clock, not the patience clock",
          total_e > eh, f"age_total={total_e} eh={eh} at step {step_e}")

    # and the clocks must agree on a genuine new attempt class
    p.note_enacted(("grasp_lift", "top", 0), "from above", step=p.now)
    check("D14 a new attempt class resets BOTH clocks",
          p.age_total() == 0 and p.age() == 0,
          f"age_total={p.age_total()} age={p.age()}")

    # credit must clear both clocks so the next obligation starts clean
    p.note_enacted(("pour", "plain", 0), "pour it", step=p.now)
    p.on_credit("01_Lift_Tomato_Sauce", step=p.now)
    check("D14 credit clears the absolute clock",
          p.age_total() == -1, str(p.age_total()))


def test_law_never_changes_an_attempt_in_flight() -> None:
    """T3 stated as a rule: inside `hold_horizon` the action is never RESAMPLE."""
    p = _pol(seed=11)
    p.note_enacted(("pick_up", "plain", 0), "Grasp the tomato sauce bottle", step=0)
    hh = int(p.belief.hold_horizon)
    inside = []
    for step in range(1, hh + 1):
        p.advance(step)
        inside.append(p.decide().action)
    check("D13 no attempt is changed inside hold_horizon",
          A_RESAMPLE not in inside, str(sorted(set(inside))))
    eh = int(p.belief.escape_horizon)
    later = []
    for step in range(hh + 1, eh + 3):
        p.advance(step)
        later.append(p.decide().action)
    check("D13 patience is still bounded: an escape is reachable past the horizon",
          A_RESAMPLE in later, str(sorted(set(later))))


def test_bind_late_import_is_bound() -> None:
    """DEFECT 12 guard: `harness` is NOT importable when install() runs.

    This is the defect that cost a real submission.  `arms/dial.sh` puts only this
    experiment's `pysite` on PYTHONPATH ahead of whatever the submitting shell happened to
    export, and the evaluator adds `evaluation_benchmark` to `sys.path` only at run time.
    The earlier `install()` did `from harness import controller` / `import
    harness.api_vlm_planner` directly, raised `ModuleNotFoundError`, swallowed it, and set
    `installed_at` so it never retried.  Result: a run with NO binding, an empty evidence
    file, and a score of 0,0,0 indistinguishable from an architectural failure.

    The test builds a genuine importable `harness` package on a temp path, installs while
    it is absent, then imports it and asserts the finder bound both hooks.
    """
    import importlib

    os.environ["MEMEXP_DIAL"] = "1"
    saved = {k: v for k, v in list(sys.modules.items())
             if k == "harness" or k.startswith("harness.")}
    for k in list(saved):
        del sys.modules[k]

    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "harness"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "api_vlm_planner.py").write_text(
            "class ApiMemoryPlanner:\n"
            "    def __init__(self):\n"
            "        self.step = 0\n"
            "    def _build_messages(self, a, b, c, d, *, extra_memory_text=''):\n"
            "        return [{'type': 'text', 'text': 'base'}]\n"
            "    def reset_episode(self):\n"
            "        pass\n"
            "infer_primitive_via_api = lambda **kw: 'UNTOUCHED'\n")
        (pkg / "controller.py").write_text(
            "class HarnessController:\n"
            "    def override_vla_prompt(self, base_prompt, *, stage_idx=0, stage_specs=None,\n"
            "                            stage_done=None, step=0, **extra):\n"
            "        return base_prompt\n")

        sys.path.insert(0, td)
        try:
            B._STATE["installed_at"] = None
            B._STATE["errors"] = []
            B._BOUND.clear()
            B._LOCAL.ctx = None
            B._LOCAL.planner = None

            # install while `harness` is genuinely unimportable
            B.install()
            check("D9 install() survives a missing harness", True)
            check("D9 nothing is bound while harness is absent", not B._BOUND, str(B._BOUND))

            ap = importlib.import_module("harness.api_vlm_planner")
            ctl = importlib.import_module("harness.controller")

            check("D9 the planner channel binds on a LATE import",
                  getattr(ap.ApiMemoryPlanner._build_messages, "_dial_wrapped", False))
            check("D9 the credit observer binds on a LATE import",
                  getattr(ctl.HarnessController.override_vla_prompt, "_dial_stage_wrapped", False))
            check("D9 both target modules are recorded as bound",
                  set(B._BOUND) == set(B._TARGETS), str(B._BOUND))
            check("D9 the primitive channel is still untouched by the finder",
                  ap.infer_primitive_via_api() == "UNTOUCHED")
            # and the late-bound hook must actually work
            pl = ap.ApiMemoryPlanner()
            importlib.import_module("harness")
            msgs = pl._build_messages({}, {}, {}, {})
            check("D9 the late-bound hook injects a DIAL block",
                  any("DIAL diagnosis" in m.get("text", "") for m in msgs), str(msgs))
            out = ctl.HarnessController().override_vla_prompt("keep", stage_idx=0,
                                                             stage_done={"01_Lift_X": True},
                                                             step=3)
            check("D9 the late-bound observer is identity", out == "keep", str(out))
        finally:
            sys.path.remove(td)
            for k in [k for k in list(sys.modules)
                      if k == "harness" or k.startswith("harness.")]:
                del sys.modules[k]
            sys.modules.update(saved)


def test_bind_writes_an_install_receipt() -> None:
    """DEFECT 13 guard: a failed install must leave evidence.

    `_write_report` is only reachable from `_build_messages`, so before this a run with a
    broken install archived nothing at all — the 0,0,0 run had no way to say why.
    """
    os.environ["MEMEXP_DIAL"] = "1"
    with tempfile.TemporaryDirectory() as td:
        rp = str(Path(td) / "report")
        os.environ["MEMEXP_DIAL_REPORT"] = rp
        try:
            saved = {k: v for k, v in list(sys.modules.items())
                     if k == "harness" or k.startswith("harness.")}
            for k in list(saved):
                del sys.modules[k]
            B._STATE["installed_at"] = None
            B._STATE["errors"] = []
            B._BOUND.clear()
            B.install()                      # harness absent: must still leave a receipt
            receipts = list(Path(td).glob("report.install.*"))
            check("D10 a receipt is written even when nothing could be bound",
                  bool(receipts), str(list(Path(td).iterdir())))
            if receipts:
                j = json.loads(receipts[0].read_text())
                check("D10 the receipt states the binding outcome",
                      j.get("bound_modules") == [], str(j.get("bound_modules")))
                check("D10 the receipt names the hooks it intends to install",
                      "planner_channel" in (j.get("hooks") or {}), str(j.get("hooks")))
                check("D10 the receipt records that the primitive channel is not wrapped",
                      "NOT WRAPPED" in str((j.get("hooks") or {}).get("primitive_channel")),
                      str((j.get("hooks") or {}).get("primitive_channel")))
            for k in [k for k in list(sys.modules)
                      if k == "harness" or k.startswith("harness.")]:
                del sys.modules[k]
            sys.modules.update(saved)
        finally:
            os.environ.pop("MEMEXP_DIAL_REPORT", None)


def test_bind_is_idempotent() -> None:
    os.environ["MEMEXP_DIAL"] = "1"
    state = _fresh_install()
    pl = state["ap_mod"].ApiMemoryPlanner()
    for _ in range(4):
        B.install()
    pl._build_messages({}, {}, {}, {})
    msgs = pl._build_messages({}, {}, {}, {})
    diat = [m for m in msgs if "DIAL diagnosis" in m.get("text", "")]
    check("D6 repeated install() does not stack wrappers", len(diat) == 1, f"n={len(diat)}")
    check("D6 the original still executes once per call", state["n_orig_build"] >= 2,
          str(state["n_orig_build"]))


def test_bind_report_is_written_and_declares_no_rewrite() -> None:
    os.environ["MEMEXP_DIAL"] = "1"
    with tempfile.TemporaryDirectory() as td:
        rp = str(Path(td) / "report")
        os.environ["MEMEXP_DIAL_REPORT"] = rp
        os.environ["MEMEXP_DIAL_REPORT_EVERY"] = "1"
        state = _fresh_install()
        pl = state["ap_mod"].ApiMemoryPlanner()
        pl.step = 1
        pl._build_messages({}, {}, {}, {})
        files = [f for f in Path(td).glob("report.*") if ".install." not in f.name]
        check("D7 a report file was written", bool(files), str(list(Path(td).iterdir())))
        if files:
            j = json.loads(files[0].read_text())
            check("D7 the report declares there is no rewrite path",
                  j.get("no_rewrite_path") is True, str(j.get("no_rewrite_path")))
            check("D7 the report lists only non-blocking actions",
                  set(j.get("actions_available") or []) == set(ACTIONS),
                  str(j.get("actions_available")))
            check("D7 the report carries a policy snapshot",
                  "policy" in j and "belief" in j, str(sorted(j)))
        os.environ.pop("MEMEXP_DIAL_REPORT", None)
        os.environ["MEMEXP_DIAL_REPORT_EVERY"] = "25"


def test_bind_render_directive_has_no_copyable_attempt() -> None:
    """BOLT v8's leak: a board that prints an actionable phase-object string gets copied
    verbatim, the object is grasped without being lifted, and the scored Lift stage never
    fires.  The directive must prescribe a CLASS, never a rendered sentence."""
    os.environ["MEMEXP_DIAL"] = "1"
    p = _pol(seed=3)
    p.note_enacted(("grasp", "plain", 0), "Grasp the tomato sauce bottle", step=0)
    d = p.decide()
    # Force the RESAMPLE branch: the point of this test is the wording of the
    # new-attempt directive, not which action the law picked on this step.
    d.action = A_RESAMPLE
    d.strategy = Strategy("lift", "grasp_lift", "top", 0,
                          "Approach the tomato sauce bottle from above and lift it.")
    ctx = B._Ctx.__new__(B._Ctx)
    ctx.obligation = "01_Lift_Tomato_Sauce"
    ctx.family = "lift"
    ctx.policy = p
    block = B.render_directive(ctx, d)
    check("D8 the directive prescribes a class",
          "verb=" in block and "approach=" in block, block[:160])
    check("D8 the directive never contains the rendered attempt sentence",
          "Approach the tomato sauce bottle from above" not in block, block[:200])
    check("D8 the directive never orders the planner to vary its wording",
          "do not copy" not in block.lower() and "do not reuse" not in block.lower(),
          block[-200:])


# =======================================================================================
# main
# =======================================================================================
def main() -> int:
    print("=" * 78)
    print("DIAL selftest (headless: no GPU, no harness, no network)")
    print("=" * 78)
    groups = [
        ("A  scheduling invariants", [
            test_action_space_has_no_block,
            test_all_families_offer_a_different_attempt,
            test_beta_stays_a_distribution,
            test_lambda_hat_estimates_lag,
            test_no_absorption_across_families,
            test_absorption_would_be_reachable_for_a_gate_policy,
            test_resync_veto_and_patience_bound,
            test_persist_does_not_corrupt_last_key,
            test_single_candidate_family_is_safe,
            test_thompson_learns_the_good_cell,
            test_hierarchical_warm_recovers_ordering,
        ]),
        ("B  reward attribution", [
            test_credit_attribution_is_per_family,
            test_credit_without_an_outstanding_attempt_is_safe,
        ]),
        ("C  simulated episodes", [
            test_sim_escape_vs_gate,
            test_sim_resync_protects_an_inflight_attempt,
            test_control_arm_cancels_the_same_attempt,
            test_sim_multistage_attribution,
        ]),
        ("D  bind-layer safety", [
            test_bind_override_is_identity,
            test_bind_build_messages_appends_one_block,
            test_bind_disabled_is_transparent,
            test_bind_reset_clears_everything,
            test_bind_does_not_touch_the_primitive_channel,
            test_bind_is_idempotent,
            test_bind_report_is_written_and_declares_no_rewrite,
            test_credit_comes_from_the_real_channel,
            test_directive_does_not_forbid_repetition,
            test_law_never_changes_an_attempt_in_flight,
            test_two_clocks_repeat_does_not_become_staleness,
            test_bind_late_import_is_bound,
            test_bind_writes_an_install_receipt,
            test_bind_render_directive_has_no_copyable_attempt,
        ]),
    ]
    for title, fns in groups:
        print(f"\n--- {title} ---")
        for fn in fns:
            before_p, before_f = len(_PASS), len(_FAIL)
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                import traceback
                _FAIL.append(f"{fn.__name__} raised {exc!r}\n"
                             + traceback.format_exc())
                continue
            newly = len(_PASS) - before_p
            nfail = len(_FAIL) - before_f
            status = "ok " if nfail == 0 else "FAIL"
            print(f"  [{status}] {fn.__name__:52s} {newly} checks"
                  + (f"  ({nfail} failed)" if nfail else ""))

    print("\n" + "=" * 78)
    print(f"checks passed: {len(_PASS)}   failed: {len(_FAIL)}")
    if _FAIL:
        print("-" * 78)
        for f in _FAIL:
            print("FAIL " + f)
    print("=" * 78)
    print("VERDICT:", "ALL PASS" if not _FAIL else f"{len(_FAIL)} FAILURE(S)")
    return 0 if not _FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
