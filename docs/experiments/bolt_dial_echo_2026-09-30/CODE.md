# BOLT / DIAL / ECHO code map

Everything listed here is snapshotted under `code/` in this package and also lives in
`experiments/mem_efficacy/` in the working tree.

## BOLT

| path | role |
|---|---|
| `bolt/graph.py` | obligation graph + invariant assertions (I1–I5) |
| `bolt/serve.py` | pure serve ladder (wording only) |
| `bolt/arbiter.py` | rejects illegal proposals |
| `bolt/clock.py` / `ledger.py` / `board.py` / `harness.py` | dual clock, verified ledger, board, controller |
| `bolt/bind.py` | planner hooks |
| `memexp_bolt_bind.py` | sitecustomize entry |
| `arms/bolt.sh` | arm |
| `selftest_bolt.py` | headless T1–T24 |
| `run_bolt_hard3_1x10.sh`, `run_bolt_t19_1x10.sh` | submitters |

## DIAL

| path | role |
|---|---|
| `dial/policy.py` | action law over bottleneck posterior |
| `dial/strategy.py` | attempt-cell library |
| `dial/bind.py` | runtime bind |
| `dial/falsify.py` | offline archive falsification |
| `memexp_dial_bind.py` | sitecustomize entry |
| `arms/dial.sh` | arm |
| `run_dial_hard3_1x10.sh` | submitter |

## ECHO

| path | role |
|---|---|
| `echo/core.py` | evidence, commitment, phase constraint, sanitize |
| `echo/bind.py` | planner message inject + API soft retry + VLA identity observe |
| `echo/selftest.py` | late-import, identity, gate, shared-state tests |
| `memexp_echo_bind.py` | sitecustomize entry |
| `arms/echo.sh` | arm |
| `run_echo_t8_1x10.sh` | t8 submitter; replays GPM against futility floor before sbatch |

## Shared

| path | role |
|---|---|
| `pysite/sitecustomize.py` | installs bolt/dial/echo when `MEMEXP_{BOLT,DIAL,ECHO}=1` |

VLA prompts remain under harness control for ECHO (observer returns the underlying result).
BOLT/DIAL may rewrite planner primitives via their arbiters; see each README.
