"""mem_efficacy / zero-touch activation of this experiment's runtime patches.

WHY sitecustomize
-----------------
Both patches have to reach into `harness.*`, and the constraint is that no shared file may be
edited. Every other option costs more than it looks:

  * editing `harness/*.py` or the evaluator          -- forbidden, and it would move these arms'
                                                        behaviour into code all arms share
  * a `-c` prelude that imports the evaluator        -- the evaluator is launched by
                                                        `run_fullvlm26_async_vlm_vla_csr_tsr.sh`,
                                                        which is upstream and must not be forked
  * a new arm in `scripts/run_harness_variant.sh`    -- puts this experiment's state into the
                                                        file that every other arm reads

`sitecustomize` is the mechanism Python provides for exactly this: it is imported automatically
at interpreter start when its directory is on PYTHONPATH. The arm file sets PYTHONPATH, so the
activation lives entirely in this experiment's own directory and the shared tree is untouched.

TWO INDEPENDENT PATCHES
-----------------------
They are activated separately because they answer different questions and are used by different
arms:

  MEMEXP_PULL_ENABLE=1    the PULL arm's memory tools (memexp_bind). Off for `pushmem`.
  MEMEXP_MEMFIX_ENABLE=1  the memory-CONTENT correction (memexp_memfix). On for every arm that
                          injects memory, so that `pushmem` and `pullmem` run the SAME corrected
                          harness and differ only in their declared channels. If only one of them
                          were corrected, the two memory arms would not be comparable, and that
                          difference would be attributed to push-vs-pull.
  MEMEXP_ER=1             Evidence-Retrieval overlay on pull (memexp_er): OpenEvidence gap ads +
                          serve→commit ledger. Requires MEMEXP_PULL_ENABLE; no-op alone.
  MEMEXP_EVMEM=1          EvMem-GPM on nomem: NEED predicates + C-gate + gated evidence
                          (fluent default; no notes; KF bank off).
  MEMEXP_EVMEM_SR=1       EvMem-SR/GPM-PushPull v2: sparse AutoPush + stall
                          force-pull≤1 + carried evidence. Exclusive with MEMEXP_EVMEM.
  MEMEXP_EVMEM_PIC=1      PIC-MEM: typed invariants + VoI threshold +
                          PIN/PUSH/WITHHOLD/SWITCH/TRIAGE over GPM floor.
                          Exclusive with MEMEXP_EVMEM / MEMEXP_EVMEM_SR.
  MEMEXP_AOM=1            AOM: Arbitrated Obligation Memory. One obligation type with
                          modes {ACTUAL, PERCEPTUAL, DERIVED}, one arbitration law, one
                          board, and admissibility DERIVED from the obligation graph.
                          Unifies EvMem-GPM (the scheduler) and memexp_er (the types).
                          Exclusive with MEMEXP_EVMEM / *_SR / *_PIC, which all bind the
                          same planner object and would each install their own ledger.
  MEMEXP_BOLT=1           BOLT-Sync: Belief–Obligation–Ledger–Telemetry with Clock
                          Synchronization. Execution-graph obligations + Claim belief +
                          dual-clock commitment–evidence window + unified Arbiter.
                          Exclusive with MEMEXP_EVMEM / *_SR / *_PIC / MEMEXP_AOM.
  MEMEXP_DIAL=1           DIAL: Diagnosis-Informed Action Law. One scheduler for planner +
                          VLA strategy + proactive memory + harness, driven by a posterior
                          over why credit is not moving. No rejecting action exists
                          ({RESAMPLE, QUERY, RESYNC, PERSIST}), so the same-obligation
                          absorbing loop is structurally unreachable. Observes the VLA
                          prompt and never rewrites it, and does not wrap
                          infer_primitive_via_api.
                          Exclusive with all of the above.

SAFETY
------
`sitecustomize` runs in EVERY interpreter that inherits PYTHONPATH -- including the VLA server and
the small torch probes the runner runs before the evaluation starts. So this file is required to
be a strict do-nothing unless an arm's own flag is set, and it must never raise: an exception here
is printed by `site` and then ignored, which would turn a memory-layer bug into a puzzling warning
that no census would attribute to this module.

The no-memory baseline never sets either flag, so it never imports either module. That is not a
convention but a checkable property: `validate_arm.py` asserts that `nomem` leaves PYTHONPATH free
of this directory under every arm ordering.
"""
from __future__ import annotations

import os
import sys

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY


def _pkg_dir() -> str:
    """This file sits in <experiments>/mem_efficacy/pysite/, and its siblings live one level up."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _activate() -> None:
    want_pull = _flag("MEMEXP_PULL_ENABLE")
    want_memfix = _flag("MEMEXP_MEMFIX_ENABLE")
    want_er = _flag("MEMEXP_ER")
    want_evmem = _flag("MEMEXP_EVMEM")
    want_evmem_sr = _flag("MEMEXP_EVMEM_SR")
    want_evmem_pic = _flag("MEMEXP_EVMEM_PIC")
    want_aom = _flag("MEMEXP_AOM")
    want_bolt = _flag("MEMEXP_BOLT")
    want_dial = _flag("MEMEXP_DIAL")
    want_echo = _flag("MEMEXP_ECHO")
    if not want_pull and not want_memfix and not want_er and not want_evmem \
            and not want_evmem_sr and not want_evmem_pic and not want_aom \
            and not want_bolt and not want_dial and not want_echo:
        return

    pkg = _pkg_dir()
    if pkg not in sys.path:
        sys.path.insert(0, pkg)

    # Each import is guarded on its own so that a fault in one cannot silently disable the other.
    # A silently-disabled correction would look exactly like "the correction did not help".
    if want_memfix:
        try:
            import memexp_memfix  # noqa: PLC0415 - deliberately late, only when requested

            memexp_memfix.install()
        except Exception as exc:  # noqa: BLE001
            _record("memfix_install_error", exc)
    if want_pull:
        try:
            import memexp_bind  # noqa: PLC0415

            memexp_bind.install()
        except Exception as exc:  # noqa: BLE001
            _record("pull_install_error", exc)
        # ER wraps Substrate/MemoryTools after the pull binding owns them.
        if want_er:
            try:
                import memexp_er  # noqa: PLC0415

                memexp_er.install()
            except Exception as exc:  # noqa: BLE001
                _record("er_install_error", exc)
    if want_evmem:
        try:
            import memexp_evmem_bind  # noqa: PLC0415

            memexp_evmem_bind.install()
        except Exception as exc:  # noqa: BLE001
            _record("evmem_install_error", exc)
    if want_evmem_sr:
        try:
            import memexp_evmem_sr_bind  # noqa: PLC0415

            memexp_evmem_sr_bind.install()
        except Exception as exc:  # noqa: BLE001
            _record("evmem_sr_install_error", exc)
    if want_evmem_pic:
        try:
            import memexp_evmem_pic_bind  # noqa: PLC0415

            memexp_evmem_pic_bind.install()
        except Exception as exc:  # noqa: BLE001
            _record("evmem_pic_install_error", exc)
    if want_aom:
        try:
            import memexp_aom_bind  # noqa: PLC0415

            memexp_aom_bind.install()
        except Exception as exc:  # noqa: BLE001
            _record("aom_install_error", exc)
    if want_bolt:
        try:
            import memexp_bolt_bind  # noqa: PLC0415

            memexp_bolt_bind.install()
        except Exception as exc:  # noqa: BLE001
            _record("bolt_install_error", exc)
    if want_dial:
        try:
            import memexp_dial_bind  # noqa: PLC0415

            memexp_dial_bind.install()
        except Exception as exc:  # noqa: BLE001
            _record("dial_install_error", exc)
    if want_echo:
        try:
            import memexp_echo_bind  # noqa: PLC0415

            memexp_echo_bind.install()
        except Exception as exc:  # noqa: BLE001
            _record("echo_install_error", exc)


def _record(key: str, exc: BaseException) -> None:
    """Record where the census can find it rather than only on stderr, which a child process
    makes easy to lose."""
    try:
        path = str(os.environ.get("MEMEXP_MEMFIX_REPORT", "")).strip()
        if not path:
            path = str(os.environ.get("MEMEXP_PULL_REPORT", "")).strip()
        if path:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f'{{"{key}": {exc!r}}}\n')
    except Exception:
        pass
    sys.stderr.write(f"[memexp] {key}: {exc!r}\n")


try:
    _activate()
except Exception as exc:  # noqa: BLE001
    _record("sitecustomize_error", exc)
