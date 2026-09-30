# BOLT / DIAL / ECHO — post-AOM architectures and seed-100 results

Date: 2026-09-30.
Audience: anyone picking up the mem_efficacy line after AOM.
Companion: [`PROVENANCE.txt`](PROVENANCE.txt), [`CODE.md`](CODE.md), repo-root [`BOLT.md`](../../../BOLT.md), [`MEMORY_LINEAGE.md`](../../../MEMORY_LINEAGE.md), [`AOM_HANDOFF.md`](../../../AOM_HANDOFF.md).

This package archives three architectures built **after** AOM under the same paradigm:

> **Planner (API VLM) + frozen VLA + Harness + Proactive Memory**

All numbers below are **h0 / seed 100 / 1×10** unless a futility gate truncated the run. SE ≈ 14 pp on unsaturated tasks; treat score deltas as weak evidence and mechanism traces as strong evidence.

---

## 1. Scoreboard (seed 100)

| arm | task | tag | n | mean `stage_score_pct` | series | note |
|---|---:|---|---:|---:|---|---|
| **GPM** | 8 | archived | 10 | **46.7** | 66.7,100,66.7,0,66.7,100,0,0,66.7,0 | control |
| BOLT | 8 | v9 | 10 | 40.0 | 0,66.7,0,33.3,0,100,66.7,66.7,0,66.7 | best BOLT t8 |
| BOLT | 5 | v1 | 10 | 13.8 | 0,25,0,37.5,25,0,25,12.5,0,12.5 | |
| BOLT | 19 | v1 | 10 | 23.3 | 0,33.3,66.7,0,33.3,0,33.3,33.3,33.3,0 | |
| BOLT | 22 | v1 | 10 | 40.0 | 66.7,100,0,100,33.3,0,0,100,0,0 | |
| DIAL | 8 | v1 | 3 | 0.0 | 0,0,0 | early-stop (futility) |
| ECHO | 8 | v1 | 10 | 33.3 | 0,100,33.3,0,66.7,66.7,0,0,66.7,0 | identity-observer |
| ECHO | 8 | v2_strict40 | 3 | 22.2 | 66.7,0,0 | early-stop mean≤40 |

Traces: [`results/seed100_summary.tsv`](results/seed100_summary.tsv) and per-run `prompt_trace.tsv`.
GPM baseline: [`baselines/t8_evmem_gpm_s100_prompt_trace.tsv`](baselines/t8_evmem_gpm_s100_prompt_trace.tsv).

**None of BOLT / DIAL / ECHO beat GPM on task 8 under this protocol.** The useful output of the series is the failure modes below.

---

## 2. BOLT — Belief–Obligation–Ledger–Telemetry + Clock

### Claim

Separate three layers that earlier arms mixed:

1. **Invariant graph** (`graph.py`) — what may gate what; hard edges only among settleable nodes.
2. **Serve ladder** (`serve.py`) — pure wording for the active obligation; never gates.
3. **Arbiter** (`arbiter.py`) — rejects illegal proposals; never invents a new goal.

Plus dual cognition/action clocks and a verified ledger. Full design: [`BOLT.md`](../../../BOLT.md).

### What the runs showed

- Early t8 tags (v1–v8) repeatedly hit **invariant violations**: hard gates on unsettleable prep nodes, serve text that asked for grasp while the scored stage required **lift**, chocolate prep blocking sauce progress.
- **v9** fixed the worst graph bugs and reached **40.0** on t8 (still below GPM 46.7) and completed hard3/hard4 cells without seismic collapse.
- Cross-task means remain below GPM; BOLT's value is the **auditable graph+clock discipline**, not a score win.

Live code: `experiments/mem_efficacy/bolt/` + `memexp_bolt_bind.py` + `arms/bolt.sh`.

---

## 3. DIAL — Diagnosis-Informed Action Law

### Claim

One scheduler over Planner + VLA strategy + memory + harness, driven by a posterior over *why credit is not moving* (`UNREACHABLE / UNKNOWN / UNGROUNDED / UNSYNCED / UNSAFE`). Action space has **no BLOCK**; resampling is allowed when the attempt cell is the bottleneck.

### What the runs showed

- Offline falsification against archived GPM/AOM/BOLT traces motivated the design (`dial/falsify.py`).
- First live t8 1×10 (**v1**) early-stopped **0,0,0**. Root causes in that series: over-eager resampling / clock-reset bugs and bind/credit wiring failures (see also failed tags `*_bindfail`, `*_creditfail`, `*_clockfail` under `experiments/mem_efficacy/results/`).
- Lesson: a rich diagnosis law is not safe to ship until the **bind seam and credit clock** are proven non-destructive under the same futility gate used for GPM.

Live code: `experiments/mem_efficacy/dial/` + `memexp_dial_bind.py` + `arms/dial.sh`.

---

## 4. ECHO — Evidence-conditioned Commitment and Horizon-aware Orchestration

### Claim

Keep VLA prompts **byte-identical** to the underlying harness (zero rewrite), write prospective evidence from the planner frame store, and inject descriptive evidence into the **Planner** channel only when needed. Commitments distinguish `DELIVERED` vs `VERIFIED`; online scorer is not an online credit source.

### v1 (job 626108) — 33.3 on t8

- Escaped absorbing 0,0,0 collapses (unlike DIAL v1).
- Paired against GPM: losses concentrated on trials **0 / 2 / 5**.
- Trace cause: planner **chocolate attractor** on trial 0 (15/15 chocolate-only primitives) while GPM repeatedly issued sauce lift. Passive observation + stall-gated recall did not constrain the first decision.

### v2_strict40 (job 626132) — early-stop 22.2

Changes: always inject a short current-obligation constraint; soft retry + sanitize against premature pour / off-stage chocolate; shared episode state for telemetry; futility floor raised to **mean ≤ 40 after 3 trials** (GPM same-seed prefixes still survive).

| trial | ECHO v2 | GPM | mechanism |
|---:|---:|---:|---|
| 0 | 66.7 | 66.7 | chocolate attractor fixed; starts on sauce lift |
| 1 | 0 | 100 | stuck on lift (~15×); over-anchored obligation |
| 2 | 0 | 66.7 | same lift lock |

**Tradeoff:** fixing the attractor with a permanent lift obligation destroyed recovery diversity when grasp failed. Early-stop at floor 40 is expected under that failure mode.

Live code: `experiments/mem_efficacy/echo/` + `memexp_echo_bind.py` + `arms/echo.sh` + `run_echo_t8_1x10.sh`.

---

## 5. Cross-cutting lessons

1. **Non-destructive bind first.** Identity observing (ECHO v1) beats a clever law that zeroes the episode (DIAL v1).
2. **GPM's edge on t8 is active order control**, not retrieval volume: sauce-lift first, block premature pour language, suppress attractors.
3. **Hard obligations without independent progress evidence freeze the agent** (ECHO v2 trials 1–2).
4. **Graph invariants are testable offline** (BOLT selftest); ship them before GPU jobs.
5. **Stricter futility floors must be replayed on archived GPM** before submit; this package's ECHO v2 gate does that in `run_echo_t8_1x10.sh`.

---

## 6. How to reproduce

```bash
cd experiments/mem_efficacy
# BOLT hard3 / t19
bash run_bolt_hard3_1x10.sh
bash run_bolt_t19_1x10.sh
# DIAL
bash run_dial_hard3_1x10.sh
# ECHO t8 (default tag v2_strict40; floor 40)
bash run_echo_t8_1x10.sh v2_strict40
# Headless
python -m echo.selftest
python selftest_bolt.py
```

Do not compare truncated early-stop means to full 10-trial GPM means without marking truncation.
