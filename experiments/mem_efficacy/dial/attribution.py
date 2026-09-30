"""Turn archived runs into a reward stream.

The harness already logs everything a credit assignment needs, it was simply never
read back:

    harness_memory.json -> episode_evidence:
        {"step": 80, "action": "stall",
         "instruction": "Reach to the tomato sauce bottle, grasp it, and lift it clear.",
         "outcome": "stalled", "notes": "01_Lift_Tomato_Sauce:80"}

That is (attempt text, outcome, stage) — a labelled example of "this phrasing of this
obligation did / did not move credit".  Across the four arms there are ~110 episodes
of it.  Every architecture on this line trains on nothing; this is the corpus DIAL
fits its priors on, and it costs one offline pass.

Attribution rules (conservative, and stated so they can be argued with):

  * a `stall` row credits the stage named in `notes` with the instruction that was
    active — the attempt that failed to move it;
  * a `subtask_update` row is the attempt that was *issued*; it is `active` until a
    later row for the same stage resolves it;
  * a stage appearing in `verified_stages` (or in the trial's `stage_done`) is
    credited, and the most recent outstanding attempt for it is marked `credited`,
    which also yields the enacted->credited lag in steps.  That lag is the quantity
    that tells `UNSYNCED` apart from `UNREACHABLE`, and no arm in the archive models it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from .types import RewardRecord, family_of, parse_stage_note

# ---------------------------------------------------------------------------------------
# Run discovery.  Layout (from the runners): results/<arm>_t<task>_<...>_<tag>/h0/<arm>_s<seed>/
#
# The arm prefix may itself contain underscores (`evmem_gpm`, `evmem_sr`, `evmem_pic`), so a
# plain `[a-z]+` prefix silently skipped every `evmem_gpm_t*_*` run — the GPM control, i.e. the
# single most important comparison in the archive.  That omission is what produced an
# inconsistent arm mean in an earlier revision of falsify.py.
# ---------------------------------------------------------------------------------------
_RUN_RX = re.compile(r"^(?P<arm>[a-z_]+)_t(?P<task>\d+)_(?P<rest>.+)$")


def iter_run_dirs(results_root: Path) -> Iterator[tuple[str, int, str, Path]]:
    """Yield (arm, task, tag, run_dir) for every archived run under `results_root`."""
    for d in sorted(Path(results_root).glob("*_t*_*")):
        if not d.is_dir():
            continue
        m = _RUN_RX.match(d.name)
        if not m:
            continue
        arm = m.group("arm")
        task = int(m.group("task"))
        tag = d.name
        for prof in sorted(d.glob("h0")):
            for run in sorted(prof.glob("*_s*")):
                if run.is_dir():
                    yield arm, task, tag, run


def _read_trial_scores(run_dir: Path) -> dict[int, float]:
    """trial index -> stage_score_pct, from prompt_trace.tsv (the scorer's own record)."""
    out: dict[int, float] = {}
    p = Path(run_dir) / "prompt_trace.tsv"
    if not p.is_file():
        return out
    lines = p.read_text(errors="replace").splitlines()
    if not lines:
        return out
    hdr = lines[0].split("\t")
    try:
        i_trial, i_score = hdr.index("trial"), hdr.index("stage_score_pct")
    except ValueError:
        return out
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) <= i_score or not parts[i_score]:
            continue
        try:
            out[int(parts[i_trial])] = float(parts[i_score])
        except ValueError:
            continue
    return out


def _read_stage_done(run_dir: Path, task: int, ep: int) -> list[str]:
    """Scored stages the episode is recorded as having completed, if available."""
    for cand in (run_dir / f"task{task}" / f"ep{ep}" / "attempt0" / "stage_report.json",
                 run_dir / f"task{task}" / f"ep{ep}" / "stage_report.json"):
        if cand.is_file():
            try:
                j = json.loads(cand.read_text())
            except Exception:  # noqa: BLE001
                continue
            for key in ("stage_done", "stages_done", "done", "completed_stages"):
                v = j.get(key)
                if isinstance(v, list):
                    return [str(x) for x in v]
    return []


def _read_stage_outcomes(run_dir: Path, task: int, ep: int) -> tuple[list[str], list[str]]:
    """(credited_stages, stalled_stages) for one episode, from the harness's own bookkeeping.

    `harness_memory.json -> attempts[]` carries, per attempt,
        {"completed_stages": ["01_Lift_..","02_Pour_One"], "stalled_stage": "03_Pour_Two"}
    which is the credit ledger the harness already maintains and no arm reads back.
    """
    mem = run_dir / f"task{task}" / f"ep{ep}" / "attempt0" / "harness_memory.json"
    if not mem.is_file():
        return [], []
    try:
        j = json.loads(mem.read_text(errors="replace"))
    except Exception:  # noqa: BLE001
        return [], []
    credited: list[str] = []
    stalled: list[str] = []
    for a in (j.get("attempts") or []):
        if not isinstance(a, dict):
            continue
        for s in (a.get("completed_stages") or []):
            if str(s) not in credited:
                credited.append(str(s))
        st = str(a.get("stalled_stage") or "")
        if st and st not in stalled:
            stalled.append(st)
    bp = j.get("best_partial") or {}
    if isinstance(bp, dict):
        for s in (bp.get("completed_stages") or []):
            if str(s) not in credited:
                credited.append(str(s))
        st = str(bp.get("stalled_stage") or "")
        if st and st not in stalled:
            stalled.append(st)
    for s in _read_stage_done(run_dir, task, ep):
        if s not in credited:
            credited.append(s)
    return credited, stalled


def _ordinal(stage: str) -> int:
    m = re.match(r"(\d+)", str(stage or ""))
    return int(m.group(1)) if m else 99


def _attempt_events(j: dict) -> list[tuple[int, str]]:
    """(step, text) for every attempt the episode issued, in issue order."""
    ev = j.get("episode_evidence") or []
    out: list[tuple[int, str]] = []
    for row in ev:
        if not isinstance(row, dict):
            continue
        if str(row.get("action") or "") != "subtask_update":
            continue
        text = str(row.get("instruction") or "").strip()
        if not text:
            continue
        try:
            step = int(row.get("step", -1))
        except (TypeError, ValueError):
            step = -1
        out.append((step, text))
    if not out:
        for row in (j.get("working") or []):
            try:
                out.append((int(row[0]), str(row[1]).strip()))
            except Exception:  # noqa: BLE001
                continue
    out.sort(key=lambda x: x[0])
    return out


def extract_episode(run_dir: Path, task: int, ep: int, *, arm: str, seed: int,
                    trial_score: float) -> list[RewardRecord]:
    """One episode -> attributed reward records.

    The attribution is *within-stage*, and that is the whole value of doing it here
    rather than at the episode level: a stage that stalls ten times and is then
    credited contributes nine labelled failures and one labelled success for the same
    obligation, which is exactly the contrast a strategy posterior needs and which no
    per-episode score can express.

        stalled stage S  -> the attempt active when S stalled is a failure for S
        credited stage S -> within S's window (up to the next stall of a later
                            ordinal), the last attempt of S's family is the success
                            and every earlier attempt of that family is a failure
    """
    mem = run_dir / f"task{task}" / f"ep{ep}" / "attempt0" / "harness_memory.json"
    if not mem.is_file():
        return []
    try:
        j = json.loads(mem.read_text(errors="replace"))
    except Exception:  # noqa: BLE001
        return []

    attempts = _attempt_events(j)
    if not attempts:
        return []
    credited, stalled = _read_stage_outcomes(run_dir, task, ep)
    if not credited and not stalled:
        return []

    ev = j.get("episode_evidence") or []
    stall_marks: list[tuple[int, str]] = []
    for row in ev:
        if not isinstance(row, dict) or str(row.get("action") or "") != "stall":
            continue
        stage, step = parse_stage_note(str(row.get("notes") or ""))
        if stage:
            stall_marks.append((step, stage))
    stall_marks.sort(key=lambda x: x[0])

    def _mk(step: int, obligation: str, text: str, outcome: str, lag: int = -1) -> RewardRecord:
        return RewardRecord(
            arm=arm, task=task, seed=seed, episode=ep, step=step,
            obligation=obligation, family=family_of(obligation), text=text,
            outcome=outcome, trial_score=trial_score,
            credited_delta=1.0 if outcome == "credited" else 0.0, lag_steps=lag,
        )

    records: list[RewardRecord] = []
    claimed: set[int] = set()          # indices of attempts already attributed

    # -- failures: the attempt active at each stall -----------------------------------------
    for step, stage in stall_marks:
        prior = [i for i, (s, _t) in enumerate(attempts) if s <= step and i not in claimed]
        if not prior:
            continue
        i = prior[-1]
        claimed.add(i)
        records.append(_mk(attempts[i][0], stage, attempts[i][1], "stalled"))

    # -- successes: last matching attempt inside each credited stage's window ----------------
    for stage in credited:
        o = _ordinal(stage)
        fam = family_of(stage)
        boundary = min([s for s, st in stall_marks if _ordinal(st) > o], default=10 ** 9)
        window = [i for i, (s, t) in enumerate(attempts)
                  if s < boundary and family_of_stage_guess(t) == fam]
        if not window:
            window = [i for i, (s, _t) in enumerate(attempts) if s < boundary]
        if not window:
            continue
        fresh = [i for i in window if i not in claimed]
        if not fresh:
            # all attempts in the window were already marked stalled: the credit came from
            # the last one anyway, so relabel it rather than inventing an attempt.
            i = window[-1]
            for r_i, r in enumerate(records):
                if r.step == attempts[i][0] and r.obligation == stage:
                    records[r_i] = _mk(r.step, stage, r.text, "credited")
                    break
            continue
        for i in fresh[:-1]:
            claimed.add(i)
            records.append(_mk(attempts[i][0], stage, attempts[i][1], "stalled"))
        i = fresh[-1]
        claimed.add(i)
        prev = [s for s, _t in attempts[:i]]
        lag = (attempts[i][0] - prev[-1]) if prev else -1
        records.append(_mk(attempts[i][0], stage, attempts[i][1], "credited",
                           lag=max(0, lag) if lag >= 0 else -1))

    # -- middle attempts of the frontier stage that neither stalled nor got credited ---------
    if stalled:
        o = max(_ordinal(s) for s in stalled)
        fam = family_of([s for s in stalled if _ordinal(s) == o][0])
        for i, (s, t) in enumerate(attempts):
            if i in claimed or family_of_stage_guess(t) != fam:
                continue
            claimed.add(i)
            records.append(_mk(s, [x for x in stalled if _ordinal(x) == o][0], t, "stalled"))
    return records


def family_of_stage_guess(text: str) -> str:
    """Family implied by an attempt's own wording (used only to align attempts to stages)."""
    from .types import classify_verb
    v = classify_verb(text)
    if v in ("pick_up", "grasp_lift", "lift", "grasp", "reach"):
        return "lift"
    if v == "pour":
        return "pour"
    if v == "place":
        return "place"
    if v in ("open", "close"):
        return v
    return "other"


def extract_all(results_root: Path, *, arms: Iterable[str] | None = None,
                tasks: Iterable[int] | None = None) -> list[RewardRecord]:
    """Every archived episode -> reward records.  `arms`/`tasks` filter for speed."""
    want_arms = set(arms) if arms else None
    want_tasks = set(tasks) if tasks else None
    out: list[RewardRecord] = []
    for arm, task, _tag, run in iter_run_dirs(results_root):
        if want_arms and arm not in want_arms:
            continue
        if want_tasks and task not in want_tasks:
            continue
        scores = _read_trial_scores(run)
        if not scores:
            continue
        m = re.search(r"_s(\d+)$", run.name)
        seed = int(m.group(1)) if m else -1
        for ep in sorted(scores):
            out.extend(extract_episode(run, task, ep, arm=arm, seed=seed,
                                       trial_score=scores[ep]))
    return out


def to_jsonl(records: Iterable[RewardRecord], path: Path) -> int:
    n = 0
    with Path(path).open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
