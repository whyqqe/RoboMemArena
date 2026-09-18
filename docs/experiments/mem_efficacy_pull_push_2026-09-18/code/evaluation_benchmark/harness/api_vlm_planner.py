from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from harness.api_planner import infer_primitive_via_api, load_api_key
from harness import redact
from harness.proactive_memory import (
    EpisodicStore,
    ProactiveMemoryConfig,
    apply_anti_collapse,
    create_store_if_needed,
    memory_decision_system_prompt,
    parse_memory_decision,
)
from harness.pmh_memory import (
    apply_pmh_anti_collapse,
    citation_ci_overlap,
    collect_pact_extra,
    commit_segment,
    create_pmh_store_if_needed,
    dump_pmh_episode_stats,
    parse_pmh_decision,
    parse_provenance,
    pick_representative_indices,
    merge_verified_fact,
    render_verified_ledger,
    sdv_verified_key_rejected,
    pmh_decision_system_prompt,
    pmh_enabled,
    resolve_slot_for_question,
    select_by_marginal_gain,
)
from harness.evidence_compiler import AceController
from harness.seam_memory import looks_non_canonical as _seam_looks_non_canonical
from harness.pcam_access import PcamController
from harness.scec_memory import ScecController
from harness.vlm_output_parser import parse_vlm_output
from harness.kairos_wm import compute_residual, kairos_enabled
from keyframe_selection import build_visual_memory, get_frames_from_indices
from memory_system.config import load_memory_system_config

# The PrediMem nomination POLICY, as a module constant rather than an inline string.
#
# WHY A CONSTANT: GATE 0 in `run_26x1.sbatch` has to prove that a real API model ACTS on this text
# before any GPU time is spent, and the only honest way to do that is to send the exact string the
# planner sends. A second copy inside the gate would drift from this one, and the gate would then
# be testing a prompt that no arm uses -- passing while the arm's channel stays dead. That is the
# "two parsers disagree" shape this repository has already paid for once (job 581594).
#
# WHAT IT IS FOR. The line that specifies the output format ("...two fields: current_primitive and
# keyframe_positions...") states the FIELD but never the POLICY, so a general model with no policy
# returns the empty list and the official bank builder is structurally dead -- `J_hist` all-empty,
# `build_visual_memory` -> [], measured as `kf_n = 0` on job 586700. PrediMem gets this behaviour
# from TRAINING; this text is the instruction-time substitute.
KF_NOMINATION_POLICY = (
    "How to choose keyframe_positions. Each frame you nominate will be shown "
    "back to you again later in this task, AFTER it has left the recent window "
    "above -- so nomination is the only way a visual state you have already "
    "seen can be recalled. Nominate the position of a frame that records a "
    "STATE CHANGE whose value you will still need later and which will no "
    "longer be visible once this window moves on. That means, specifically: a "
    "container (drawer, microwave, cabinet, box, basket) being opened or "
    "closed; an object being placed into, taken out of, or poured into its "
    "target; the moment a numbered subtask completes; or a frame that clearly "
    "shows which container is empty and which still holds something. Nominate "
    "the ONE frame in which the change first becomes visible, at most 3 "
    "positions per step. Do NOT nominate frames that only show the arm in "
    "transit, ordinary motion, or a view the recent window already makes "
    "obvious -- a nominating-everything habit makes the memory useless. "
    "Return an empty list when no frame in the window shows such a change."
)

logger = logging.getLogger(__name__)
# Unshadowable alias. `ApiMemoryPlanner.__init__` takes a parameter literally named `logger`
# (default None, and the real driver always passes None), so a bare `logger.info(...)` inside
# __init__ resolves to that None parameter and raises
#   AttributeError: 'NoneType' object has no attribute 'info'
# which is exactly how job 568049 died 65s in, right after the VLA server came up, before a
# single step ran. The file's existing convention is `if self.logger:` guards; this module-level
# alias is used instead wherever a message MUST reach the log regardless of that parameter, so
# the sbatch mechanism census can actually see it.
_LOG = logging.getLogger(__name__)

# Common words ignored when matching a past segment's text to the current subtask.
# Keep location/object nouns (drawer, cabinet, basket, ...) — those are the discriminative
# tokens for "which container did I use". Only drop function words and repeated verbs.
_PMH_STOPWORDS = {
    "the", "and", "with", "from", "into", "onto", "then", "next", "first", "second",
    "third", "after", "before", "object", "objects", "task", "stage", "subtask",
    "place", "placing", "put", "picking", "pick", "move", "moving", "robot", "arm",
    "using", "use", "grasp", "grasping", "lift", "hand", "your", "now", "has", "have",
    "been", "already", "previous", "currently", "about", "left", "right",
}
_REDACT_CONTAINER_NOUNS = redact.CONTAINER_NOUNS
_REDACT_QUALIFIERS = redact.QUALIFIERS

# v0.16 Gap Gate: phrases that mark a G3 gap - the CONTAINER is named but the position
# INSIDE it is given only by reference to an earlier placement ("place chocolate at the
# location where the cookies were placed"). Same gate, same evidence channel as G1; only the
# contract field differs (placement_frame vs observed_at_frame).
GAP_G3_LOCATION_PHRASES = (
    "location where",
    "where the",
    "placement location",
    "historical placement",
)

# Stage-name action verbs that either observe container contents or move an object. Used to
# pick the evidence frames for an open gap: the Open_*/Lift_* observations are exactly the
# frames in which drawer contents were visible, i.e. the frames that can answer "which
# drawer was non-empty?".
GAP_EVIDENCE_ACTIONS = {"open", "opening", "lift", "lifting", "pick", "picking", "inspect", "inspection"}


def _redact_container_phrases(text: str) -> set[str]:
    """Extract QUALIFIER+CONTAINER phrases ("top drawer", "middle drawer") from free text.

    Thin alias for `redact.container_phrases`. The vocabulary MOVED to `harness/redact.py` so
    that this planner and the controller's α-guard cannot disagree about what an answer key is:
    they used to hold separate copies, and the guard re-introduced an answer HM masks (see that
    module's docstring). The private name is kept because ~4 call sites reference it.

    A bare container noun is deliberately excluded: the memory question in this benchmark is
    always "WHICH of several containers", so only the qualified form carries the answer
    (task4 withholds "which drawer", never the word "drawer"). Keeping bare nouns out also
    guarantees object words can never be registered - the earlier version derived
    `tokens[2:]`, so task22's `01_Lift_Tomato_Sauce` registered "sauce" and masked the very
    object the planner had to reason about.
    """
    return redact.container_phrases(text)


def _pmh_stall_mode() -> str:
    """How the controller interprets "no progress". `legacy` reproduces every past run.

    `delta` is the v2.0 reading: the only evidence of being stuck is that the state stopped
    changing. Counting identical subtask labels conflates "the arm is executing a long
    manipulation" with "the agent is going in circles", and on t22 that conflation is worth
    12.5pp of the benchmark on its own.
    """
    v = os.environ.get("PMH_STALL_MODE", "legacy").strip().lower()
    return v if v in {"legacy", "delta"} else "legacy"


def _pmh_demand_mode() -> bool:
    """PMH-O: complementary read is gated by the address-demand queue, not by stall.

    When ON, the legacy `stall_now -> retrieve last` force path is suppressed (physical stalls
    with no missing address must not spend a `full_visual`), and the CE branch uses
    `_pmh_unaddressed_demand` instead. Default OFF so every pre-PMH-O arm reproduces byte for
    byte; `api_pmh_ce` turns it on.
    """
    return os.environ.get("PMH_DEMAND_READ", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _pmh_graph_enabled() -> bool:
    return os.environ.get("PMH_GRAPH", "0").strip().lower() not in {"0", "false", "no", "off"}


def _pmh_gain_enabled() -> bool:
    """v3.0 marginal-gain selection (PMH.md Sec 6.4: no graph traversal needed).

    Defaults ON, unlike the retired graph selector, because it is a *filter on frames the
    planner was already going to receive* rather than a new evidence channel: with it off the
    pack is the old recency order, with it on the same frames are considered in gain order and
    redundant ones are dropped. It therefore cannot make evidence unavailable that was
    available before, which is the property the graph selector lacked.
    """
    return os.environ.get("PMH_GAIN", "1").strip().lower() not in {"0", "false", "no", "off"}


def _pmh_ladder_targets(search_text: str) -> list[str]:
    """Ordered, de-duplicated segment ids the text rung pointed at.

    Module-level and pure so the preflight can test it directly. The ordering rule it encodes is
    a fix, not a nicety: `search_memory` returns hits in relevance order and the top hit is the
    most recent relevant segment, which is precisely the segment the resident KF-spine bank
    already carries (both draw from the same keyframe spine). Opening only the first hit therefore
    re-opens bank-covered frames by construction, which is what job 570778's t5 recorded as 22
    identical rounds ending in `n_added=0` with `n_gain_picked=0`.

    Duplicates are removed because `search_memory` can list the same segment in more than one
    line, and re-opening it would spend a rung to re-measure a known-empty result. The `["last"]`
    fallback preserves the pre-existing behaviour when the text rung names no segment at all.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for sid in re.findall(r"seg_\d+", search_text or ""):
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    return ordered or ["last"]


def _stage_scene_facts(stage_name: str) -> list[tuple[str, str, str]]:
    """D1 FIX: derive SCENE cells from a grader-confirmed stage name, with NO env gate.

    The previous arm's scene half received nothing, and the reason was structural rather than a
    bug in the mirror: `_pmh_note_placement_from_stage` returns early unless
    `PMH_CONTRACT_LEDGER` or `PMH_PLACEMENT_NOTES` is set, and the PMH.md recipe sets neither. So
    both `store.contracts` and `store.placement_notes` stayed empty, the mirror loops never ran,
    and every scene counter -- including `rejected` -- read 0. A silent zero is indistinguishable
    from "never called", which is why it survived a full 26-task run.

    The extraction below is the env-independent half of that function, generalised from
    placements to the three stage shapes this benchmark actually uses, and it is deliberately a
    PURE function so it can be unit tested without an API key:

        Place_X_Y / Put_X_Y   -> (X, location, Y)      where X now is
        Open_Y / Close_Y      -> (Y, state, open|closed)   what state the world is in
        Pick_X / Lift_X       -> (X, status, picked)   what the robot has hold of

    Only a CONFIRMED stage reaches here, so every fact is grader-derived rather than asserted.
    Entities and values stay in the internal key form; `surface()` converts them at render time,
    which is the one place the conversion is allowed to happen (see D2).
    """
    s = re.sub(r"^\d+_", "", str(stage_name or "").strip())
    toks = [t for t in s.split("_") if t]
    if len(toks) < 2:
        return []
    action = toks[0].lower()
    if action in {"place", "put", "placing"} and len(toks) >= 3:
        obj = toks[1].lower()
        container = "_".join(t.lower() for t in toks[2:]).strip("_")
        if len(obj) >= 2 and len(container) >= 2:
            return [(obj, "location", container)]
        return []
    if action in {"open", "close"}:
        target = "_".join(t.lower() for t in toks[1:]).strip("_")
        if len(target) >= 2:
            return [(target, "state", "open" if action == "open" else "closed")]
        return []
    if action in {"pick", "lift"}:
        obj = "_".join(t.lower() for t in toks[1:]).strip("_")
        if len(obj) >= 2:
            return [(obj, "status", "picked")]
        return []
    return []


def _pmh_stage_write_candidates(
    *,
    context_idx: list[int],
    anchors: list[int],
    step_idx: int,
    frame_store: dict[int, Any],
    fallback: list[int],
) -> tuple[list[int], list[int]]:
    """Choose which frames a stage-indexed write may archive.

    Extracted so the CHOICE can be unit-checked without an API key, because getting it wrong was
    invisible in production. Job 586567 measured 30/39 stage writes archiving `frames=[0, 6]` --
    the episode's opening frames -- for stages firing at t=88..1288. The write succeeded, logged
    its frames, and was counted as healthy; the archive then answered nearly every retrieval with
    pixels the planner already held (`n_new=0` on 19/21 calls), which we had been reading as "the
    planner does not want memory".

    The cause was the candidate pool. It came from the rolling `active_frame_indices` accumulator,
    which had not advanced to the stage's own step, so the *oldest* frames in the episode were the
    only ones ever offered. The pool must be anchored on the frames the triggering `plan_step` was
    actually SHOWN, and `step_idx` itself must be reachable: a stage at t must be able to archive
    t.

    Returns `(candidates, dropped)`. `dropped` is the primary pool that was discarded for naming no
    frame we hold - non-empty means the caller fell back, which is worth logging rather than
    silently degrading (the v0.8 failure mode this whole path already suffered from once).
    """
    primary: list[int] = []
    for i in list(context_idx or []) + list(anchors or []) + [int(step_idx)]:
        i = int(i)
        if i >= 0 and i not in primary:
            primary.append(i)
    held = [i for i in primary if i in frame_store]
    if held:
        return held, []
    # No plan_step-anchored frames survive (stage raised between plans, or the frame store was
    # trimmed): fall back to the accumulator, then to the stage step's own neighbourhood.
    acc = [int(i) for i in (fallback or []) if int(i) >= 0 and int(i) in frame_store]
    if acc:
        return acc, primary
    return (
        [
            i
            for i in range(max(0, int(step_idx) - 8), int(step_idx) + 1)
            if i in frame_store
        ],
        primary,
    )


def _pmh_bank_annotation(memory_indices: list[int], *, kf_spine: bool) -> str:
    """The bank annotation appended to the decide prompt's memory text.

    Extracted from `plan_step` so the SUFFIX can be unit-checked. That matters because the
    suffix is the whole mechanism behind `PMH_READ_OPEN`: the same prompt offers the planner
    `search_memory` and, in the legacy wording, then tells it "prefer none" and "avoid habitual
    search_memory". Job 570639 measured the result over all 8 tasks - `query search=0` - and
    every earlier PMH arm agrees, which is why raising the search budget (0 -> 1) never changed
    anything: the binding constraint was the instruction, not the budget.

    PMH_READ_OPEN=1 (default OFF) removes ONLY the discouragement. The bank is still described
    as the on-context evidence it is, so "do not re-request what you already hold" survives -
    that guard was correct, and removing it is what caused the n_new=0 retrieval loop that cost
    t22 in job 568062. What is removed is the instruction to prefer the bank OVER the archive,
    which is what turned an agent-controlled read path into a fixed-context passive one
    (PMH.md Sec 8 names that shape PACE/ACE and explicitly distinguishes it from this work).
    """
    read_open = _truthy_env("PMH_READ_OPEN")
    if kf_spine:
        head = (
            f"\nkf_spine_visual_bank: {memory_indices} "
            f"(|K|={len(memory_indices)}; dense KF Write + stage/event index; "
        )
        tail = (
            "on-context evidence you already hold)\n"
            if read_open
            else "same pixel spine as harness_v21; prefer none)\n"
        )
        return head + tail
    head = (
        f"\nstage_indexed_visual_bank: {memory_indices} "
        f"(Active+Stage+Archive; |K|={len(memory_indices)}; "
    )
    tail = (
        "on-context evidence you already hold)\n"
        if read_open
        else "prefer these; avoid habitual search_memory)\n"
    )
    return head + tail


def _truthy_env(key: str, default: bool = False) -> bool:
    """Local boolean-env reader.

    `pmh_memory._truthy` is the canonical one but it is not exported here, and the v3.0 read
    path needs it in three places. Defined as a module function rather than an inline
    `os.environ.get(...) not in {...}` so the accepted spellings cannot drift between call
    sites - the codebase already contains both the `{"0","false","no","off"}` form and the
    `{"1","true","yes","y","on"}` form, and mixing them is how a flag ends up meaning the
    opposite of what its comment says.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class ApiMemoryPlanner:
    """
    Cloud API planner (OpenAI-compatible multimodal chat).
    Replaces local PrediMem VLM weights; keeps keyframe memory + optional proactive/SCEC access.
    """

    def __init__(
        self,
        *,
        task_info: Any,
        system_prompt: str,
        n_recent: int = 5,
        d_merge: int = 6,
        k_max: int = 0,
        use_keyframe_memory: bool = True,
        use_wrist: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.n_recent = n_recent
        self.d_merge = d_merge
        self.k_max = k_max
        self.use_keyframe_memory = use_keyframe_memory
        self.use_wrist = use_wrist
        self.logger = logger

        self.api_key = load_api_key()
        self.api_base_url = os.environ.get(
            "PLANNER_API_BASE_URL",
            os.environ.get("HARNESS_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ).strip()
        self.api_model = os.environ.get("PLANNER_API_MODEL", os.environ.get("HARNESS_API_MODEL", "qwen-vl-max")).strip()
        self.api_timeout = float(os.environ.get("PLANNER_API_TIMEOUT", "120"))
        self.api_max_tokens = int(os.environ.get("PLANNER_API_MAX_TOKENS", "256"))
        self.read_k_max = int(os.environ.get("READ_K_MAX", "0"))
        self.read_stall_repeats = int(os.environ.get("READ_STALL_REPEATS", "3"))
        self.read_stall_extra_k = int(os.environ.get("READ_STALL_EXTRA_K", "2"))
        self.semantic_recent_n = int(os.environ.get("SEMANTIC_RECENT_N", "3"))
        self._last_planned_subtask = ""
        self._consecutive_same_subtask = 0
        # v3.0: the staleness reset must fire on an ARCHIVED segment, not on an attempted one.
        # When the completeness verdict withholds a commit (PMH.md Sec 4.2) the frames remain
        # buffered, so treating the attempt as progress would hide exactly the stall this
        # counter exists to detect. Seeded with 0 rather than a sentinel like -1, because a
        # sentinel makes the very first withheld commit look like progress.
        self._pmh_commits_at_last_mark = 0
        # HERMES: Supervisor arms at most one Inquiry Δ per episode.
        self.hermes_inquiry_pending = False

        if not self.api_key:
            raise RuntimeError("PLANNER API key missing. Set PLANNER_API_KEY or api_key.txt line PLANNER_API_KEY_LINE.")

        self.harness_extra_context: str = ""
        self.pinned_keyframe_steps: list[int] = []
        self.salient_keyframe_steps: list[int] = []
        self.memory_system_config = load_memory_system_config()
        self.proactive_cfg = ProactiveMemoryConfig.from_env()
        self.pcam: PcamController | None = PcamController.create()
        self.ace: AceController | None = AceController.create() if self.pcam is None else None
        self.scec: ScecController | None = (
            ScecController.create() if self.pcam is None and self.ace is None else None
        )
        # Prefer SCEC store when enabled; otherwise optional proactive store.
        if self.scec is not None:
            self.episodic_store: EpisodicStore | None = self.scec.store
        else:
            self.episodic_store = create_store_if_needed(self.proactive_cfg)
        self.pmh_store = create_pmh_store_if_needed()
        self._pmh_attach_seam_redactor()
        self.pmh_max_tool_rounds = int(os.environ.get("PMH_MAX_TOOL_ROUNDS", "2"))
        self.pmh_active_visual_k = int(os.environ.get("PMH_ACTIVE_VISUAL_K", "3"))
        self.pmh_decide_recent = int(os.environ.get("PMH_DECIDE_RECENT", "2"))
        self._pmh_has_archive_visual = False
        self._pmh_exec_stall = False
        self._pmh_episode_task_id = int(getattr(task_info, "task_id", 0) or 0)
        self._pmh_enter_stage = False
        self._pmh_new_segment = False
        self._pmh_last_boundary_deepen = -10_000
        self._pmh_last_retrieve_novel = False
        self._pmh_rel_cands = 0
        # v0.14: repeated-stall "change approach" nudge (second chance within an attempt).
        self._pmh_last_alt_stall = -10_000
        self._pmh_n_alt_inject = 0
        # v0.15: active-stage + occlusion-gate state (CGM-OB).
        self._pmh_current_stage = ""
        # P1 (job 586567): the absolute frame indices the planner actually SAW on the last
        # plan_step. Stage-indexed writes must archive frames from the step that triggered the
        # stage; deriving them from `active_frame_indices` let a stale window win, so stages
        # firing at t=1220 archived `[0, 6]` -- the episode's opening frames.
        self._pmh_last_context_idx: list[int] = []
        self._pmh_contract_hinted: set[str] = set()
        self._pmh_last_contract_hint = -10_000
        self._pmh_gap_hinted: set[str] = set()
        # v0.15b ablation: stage-name redaction (see _redact_planner_text).
        self._redact_terms: set[str] = set()
        self._redact_n_masked = 0
        # ---- v2.0 L0: retrieval rights + non-productive-retrieval guard ------------------
        # WHY. Measured over job 566614, `retrieve_visual` answered `n_new=0` (handed back
        # frames that were already on context) on 65-83% of calls; on t22 it was 83%, and on
        # t22 attempts 1-3 the planner chose a retrieval on EVERY decision while the episode
        # made no physical progress and scored 0.0 - the plain Harness scored 100.0 on the
        # same task in 13 seconds. The cause is structural: `by_salience` resolves to
        # `segments[-1]`, the segment just committed, i.e. the frames the planner already
        # holds. Nothing bounded the loop, so evidence acquisition crowded out action.
        # These counters give retrieval a per-stage RIGHT that can be exhausted, and refuse a
        # request that provably cannot add evidence - without ever blocking a stage that
        # genuinely has something new to show. (v1.0's mistake was blocking on gap-existence
        # and still consuming the plan step, which cost 50pp on t4.)
        self._pmh_retrievals_in_stage = 0
        self._pmh_nonproductive_streak = 0
        self._pmh_retrieval_refused = 0
        self._pmh_steps_since_change = 0
        self._pmh_graph_forest = None
        self._pmh_graph_last_meta: dict[str, Any] = {}
        # CE / KAIROS (ported from FullVlm26MemoryPlanner): once-per-plateau complementary
        # read + residual gate. Without these the API path could not measure the two arms.
        self._pmh_stall_committed = False
        self._pmh_stall_read_done = False
        self._pmh_paper_read_count = 0
        # PMH-O: addresses whose tier-1 question has already been put to the planner. Per
        # episode (not per plateau) because the question is a property of the DEMAND, and a
        # demand that survives a subtask change was already asked once.
        self._pmh_demand_asked: set[str] = set()
        # A selector that is enabled but non-functional would waste an entire run, so it is
        # exercised once here and any failure is stated at ERROR level.
        if _pmh_graph_enabled():
            try:
                from harness import pmh_graph as _pg
            except Exception:
                try:
                    import pmh_graph as _pg  # type: ignore
                except Exception as _exc:
                    _LOG.error("[pmh] PMH_GRAPH=1 but pmh_graph cannot be imported: %s", _exc)
                    _pg = None
            if _pg is not None:
                _ok, _why = _pg.runtime_selftest()
                if _ok:
                    _LOG.info("[pmh] graph selector self-test PASS: %s", _why)
                else:
                    _LOG.error("[pmh] graph selector self-test FAIL: %s", _why)

        self.frame_store_main: dict[int, Image.Image] = {}
        self.frame_store_wrist: dict[int, Image.Image | None] = {}
        self.K_indices_abs: list[int] = []
        self.K_main_frames: list[Image.Image] = []
        self.K_wrist_frames: list[Image.Image | None] = []
        self.J_hist: list[list[int]] = []
        self.step = 0
        self.run_dir: Path | None = None
        self._trace_fh = None
        self._current_subtask = ""
        self.set_task_info(task_info)

        if self.logger:
            self.logger.info(
                "ApiMemoryPlanner: model=%s base_url=%s keyframe=%s k_max=%s proactive=%s scec=%s ace=%s pcam=%s",
                self.api_model,
                self.api_base_url,
                self.use_keyframe_memory,
                self.k_max,
                self.proactive_cfg.mode,
                bool(self.scec),
                bool(self.ace),
                bool(self.pcam),
            )

    def set_task_info(self, task_info: Any) -> None:
        self.task_info = task_info
        self.default_subtask_prompt = task_info.brief_description.strip()
        self._current_subtask = self.default_subtask_prompt
        # v0.15c: register the answer key here so redaction does not depend on pmh_store
        # (the Harness ablation arm has none, which silently disabled redaction before).
        # Kept last so this method's original ordering is unchanged.
        self.register_redact_answer_key()

    def _pmh_ce_enabled(self) -> bool:
        return os.environ.get("PMH_CE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _pmh_ce_filter_search(self, search_text: str, state_text: str) -> str:
        """PMH-O invariant I4: redundancy is an ADDRESS test, never a WORD-OVERLAP test.

        WHAT THE OLD BODY DID AND WHY IT IS DELETED. It split each retrieved line into tokens and
        dropped the line when `>= max(2, (len(toks)+1)//2)` of its tokens appeared anywhere in
        the Task State text. Three consequences, all adverse:

        * It measured sentence similarity, which has no defined relationship to whether a piece
          of KNOWLEDGE is already held. A line stating a NEW fact about a container already named
          in the ledger ("the middle drawer is empty" vs a ledger that mentions `middle_drawer`)
          was dropped as redundant.
        * Its input was the ledger's own keys. D4 wrote those keys, so D4's writes MASKED the
          retrieved lines that could have supplied the fact D4 failed to write. The write side
          was suppressing the read side, in the same mechanism, by construction.
        * The suppression was invisible: it produced a shorter string, not a counter. Nothing in
          the census could distinguish "the archive had nothing" from "the filter deleted it",
          which is the same defect class as `CIO=1.0` for three distinct failures (job 570856).
          That is why the address version COUNT its suppressions in
          `n_false_suppress_by_token` - a word-based suppression now trips that counter, so if the
          old behaviour ever returns it shows up as a number rather than as silence.

        An address is decidable: `top_drawer.contents` is redundant iff the ledger holds a value
        for `top_drawer.contents`. Anything else is kept, because the cost of keeping a line is
        tokens and the cost of dropping a needed one is the whole mechanism.
        """
        store = getattr(self, "pmh_store", None)
        if store is None:
            return search_text
        kept: list[str] = []
        dropped_addrs: list[str] = []
        for line in (search_text or "").splitlines():
            addrs = re.findall(r"(?<![\w#.])([a-z][a-z0-9_]*(?:#\w+)?\.(?:location|position|contents|count|progress))", line.lower())
            if addrs:
                redundant = all(
                    str(store.ts_facts.get(a, "")).strip().lower() not in {"", "unknown"}
                    for a in addrs
                )
                if redundant:
                    dropped_addrs.extend(addrs)
                    continue
            kept.append(line)
        if dropped_addrs:
            logger.debug(
                "[pmh] demand_filter dropped_by_address=%s", sorted(set(dropped_addrs))
            )
        return "\n".join(kept)

    def note_pmh_exec_stall(self) -> None:
        """Harness execution stall → next decide forces Visual Archive retrieve."""
        self._pmh_exec_stall = True

    def snapshot_pmh_memory(self) -> dict[str, Any] | None:
        """Preserve PMH store + archived frames across Harness retries."""
        if self.pmh_store is None:
            return None
        need: set[int] = set()
        for seg in self.pmh_store.segments:
            need.update(int(i) for i in (seg.keyframe_indices or []))
            need.update(int(i) for i in (seg.visual_indices or [])[-12:])
        need.update(int(i) for i in self.pmh_store.active_frame_indices)
        for ref in self.pmh_store.stage_visuals:
            need.update(int(i) for i in (ref.frame_indices or []))
        mains = {i: self.frame_store_main[i] for i in need if i in self.frame_store_main}
        wrists = {i: self.frame_store_wrist[i] for i in need if i in self.frame_store_wrist}
        return {
            "store": self.pmh_store,
            "mains": mains,
            "wrists": wrists,
            "K": list(self.K_indices_abs),
            "pinned": list(self.pinned_keyframe_steps),
            "salient": list(self.salient_keyframe_steps),
        }

    def restore_pmh_memory(self, snap: dict[str, Any] | None) -> None:
        if not snap or snap.get("store") is None:
            return
        self.pmh_store = snap["store"]
        for i, img in (snap.get("mains") or {}).items():
            self.frame_store_main[int(i)] = img
        for i, img in (snap.get("wrists") or {}).items():
            self.frame_store_wrist[int(i)] = img
        # Restore KF spine across Harness retries (v0.9).
        k_restored = [int(i) for i in (snap.get("K") or []) if int(i) in self.frame_store_main]
        if k_restored:
            self.K_indices_abs = list(k_restored)
            self.K_main_frames = get_frames_from_indices(k_restored, self.frame_store_main)
            self.K_wrist_frames = [self.frame_store_wrist.get(idx) for idx in k_restored]
        if snap.get("pinned"):
            self.pinned_keyframe_steps = list(snap["pinned"])
        if snap.get("salient"):
            self.salient_keyframe_steps = list(snap["salient"])
        logger.info(
            "[pmh] restored memory across retry: segs=%s frames=%s kf_n=%s",
            len(self.pmh_store.segments),
            len(snap.get("mains") or {}),
            len(k_restored),
        )

    def _dump_pmh_stats(self, *, final: bool = False) -> None:
        """Durable PMH counters (survives video/trace cleanup).

        `final=True` is the episode-end call and additionally folds every still-open address
        demand into `n_bindings_expired_unresolved` before the snapshot is written. It has to be
        distinct from the mid-episode calls: expiring at every dump would count the same open
        demand once per decide and turn the failure signature into a measure of episode length.
        """
        if self.pmh_store is None or self.run_dir is None:
            return
        if final:
            try:
                _n = self.pmh_store.expire_bindings()
                logger.info(
                    "[pmh] demand_expire n=%s opened=%s resolved_read=%s resolved_plan=%s "
                    "unresolved=%s",
                    _n,
                    self.pmh_store.n_bindings_opened,
                    self.pmh_store.n_bindings_resolved_by_read,
                    self.pmh_store.n_bindings_resolved_by_plan,
                    self.pmh_store.n_bindings_expired_unresolved,
                )
            except Exception:
                pass
        dump_pmh_episode_stats(
            self.pmh_store,
            self.run_dir / "pmh_episode_stats.json",
            # Use episode-start task id: set_task_info() may already point at the next task.
            task_id=int(getattr(self, "_pmh_episode_task_id", 0) or 0),
            extra=self._pmh_pact_extra(),
        )

    def _pmh_pact_extra(self) -> dict[str, Any]:
        """PACT + α-guard + A5 counters for the durable census.

        Thin wrapper over the shared `collect_pact_extra`: this logic used to be duplicated per
        backend, and the local backend's copy was missing entirely (see `collect_pact_extra`).
        """
        return collect_pact_extra(self)

    def _pmh_commit_active(
        self,
        step_idx: int,
        *,
        prev_subtask: str,
        new_subtask: str,
        reason: str,
        force_close: bool = False,
    ) -> None:
        """Archive the unfinished active segment into episodic evidence.

        `force_close` distinguishes the two kinds of closure, which v3.0 must not conflate
        (PMH.md Sec 4.2 vs Sec 4.1):
          * a CANDIDATE EVENT BOUNDARY (subtask_boundary) is a question - the consolidator's
            completeness verdict may withhold the commit and leave the frames in the buffer;
          * a FRAME-BUDGET closure (pre_decide_flush) is a capacity limit, and Sec 4.1 forbids
            losing the current event, so it commits unconditionally.
        Defaulting to False keeps the verdict in play for any caller that does not opt out,
        because a silent default of True would recreate the "always commit" behaviour this
        change exists to remove.
        """
        if self.pmh_store is None:
            return
        if len(self.pmh_store.active_frame_indices) < 1:
            return
        commit_idxs = pick_representative_indices(
            list(self.pmh_store.active_frame_indices),
            max_k=int(os.environ.get("PMH_CONSOLIDATE_IMAGE_K", "4")),
        )
        commit_images = [
            self.frame_store_main[i] for i in commit_idxs if i in self.frame_store_main
        ]
        logger.info(
            "[pmh] commit reason=%s frames=%s stall=%s",
            reason,
            len(self.pmh_store.active_frame_indices),
            self._consecutive_same_subtask,
        )
        card = commit_segment(
            self.pmh_store,
            step_idx=step_idx,
            prev_subtask=prev_subtask or new_subtask or self.pmh_store.active_subtask,
            new_subtask=new_subtask or prev_subtask or self.pmh_store.active_subtask,
            api_key=self.api_key,
            base_url=self.api_base_url,
            model=self.api_model,
            images=commit_images,
            force_close=force_close,
        )
        # v3.0: the verdict may have withheld the commit (PMH.md Sec 4.2 - the frames stay in
        # the buffer). Everything below writes card-attached state and pointer bookkeeping, so
        # it must not run for a card that was never archived.
        if card is None:
            return
        # Stamp the stage on the card. `SegmentCard.stage_tag` is the field the graph selector
        # reads to give archived frames their schema identity, and it was declared but NEVER
        # assigned anywhere in the repo, so every segment-derived node arrived with an empty
        # stage. The stage machine's current label is the controller's own ground truth for
        # "where in the task this observation happened"; it is used only for frame-to-frame
        # relations and never enters planner-visible text.
        try:
            stage_now = str(
                getattr(self, "_pmh_current_stage", "") or getattr(self.pmh_store, "current_stage_name", "") or ""
            )
            if stage_now:
                card.stage_tag = stage_now
        except Exception:
            pass
        # PMH-O: a commit is an OBSERVATION entering the ledger, so it is the moment an open
        # address demand may have been answered without anyone spending a read. Reconciling here
        # (rather than only at read time) is what makes `n_bindings_resolved_by_plan` a real
        # category instead of a permanent zero: without it, every demand that a consolidator
        # observation satisfied would sit in the queue until episode end and be counted as
        # `n_bindings_expired_unresolved` - inflating the memory-failure signature with demands
        # that the memory had in fact met, which is the exact "one counter, two meanings" defect
        # the closure census exists to avoid.
        try:
            _closed_by_plan = self.pmh_store.reconcile_bindings_from_ledger()
            if _closed_by_plan:
                logger.info(
                    "[pmh] demand_closed_by_plan n=%s opened=%s resolved_plan=%s open_now=%s t=%s",
                    _closed_by_plan,
                    self.pmh_store.n_bindings_opened,
                    self.pmh_store.n_bindings_resolved_by_plan,
                    len(self.pmh_store.open_bindings),
                    step_idx,
                )
        except Exception:
            pass
        kf_spine = os.environ.get("PMH_KF_SPINE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        # v0.8: card visual pointers snap onto KF spine (pixels stay KF-sourced).
        if kf_spine and self.pmh_store.segments:
            card = self.pmh_store.segments[-1]
            snapped = self._pmh_snap_to_kf(
                list(card.keyframe_indices or commit_idxs),
                max_k=int(os.environ.get("PMH_STAGE_WRITE_K", "2")),
            )
            if snapped:
                card.keyframe_indices = list(snapped)
                card.visual_indices = list(snapped)
        # Event index write: KF pointers only under KF-spine (or legacy event pixels).
        event_write = os.environ.get("PMH_EVENT_VISUAL_WRITE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if reason == "subtask_boundary":
            # v0.10 note + v0.11 fix: only a REAL subtask transition arms the boundary
            # deepen. soft_stall / max_active_frames / pre_decide_flush happen repeatedly
            # inside a stall loop and must not each trigger a grounded deepen.
            self._pmh_new_segment = True
        if event_write and reason in {"subtask_boundary", "soft_stall", "max_active_frames"}:
            tag = f"event:{reason}:{(prev_subtask or new_subtask or 'seg')[:48]}"
            cands = list(commit_idxs) or list(self.pmh_store.active_frame_indices)
            picked_cands = self._pmh_snap_to_kf(cands, max_k=int(os.environ.get("PMH_STAGE_WRITE_K", "2")))
            if picked_cands:
                picked = self.pmh_store.note_stage_visual(tag, picked_cands, int(step_idx))
                logger.info(
                    "[pmh] event_visual_write reason=%s frames=%s t=%s kf_spine=%s",
                    reason,
                    picked,
                    step_idx,
                    int(kf_spine),
                )
        self._dump_pmh_stats()

    def _pmh_maybe_flush_before_decide(self, step_idx: int) -> None:
        """
        Flush oversized active short-term into the archive BEFORE tool decide.

        v0.4: only flush on frame-budget overflow (event-like capacity), not on every
        mild stall — noisy soft flushes hurt segment quality (KEMO/PrediMem lesson).
        """
        if self.pmh_store is None:
            return
        if os.environ.get("PMH_PRE_DECIDE_FLUSH", "1").strip().lower() in {"0", "false", "no", "off"}:
            return
        max_active = int(os.environ.get("PMH_MAX_ACTIVE_FRAMES", "14"))
        n = len(self.pmh_store.active_frame_indices)
        if n < max(1, max_active):
            return
        prev = self.pmh_store.active_subtask or self._current_subtask or "active_event"
        self._pmh_commit_active(
            step_idx,
            prev_subtask=prev,
            new_subtask=prev,
            reason="pre_decide_flush",
            # Frame-budget closure, NOT a semantic boundary: PMH.md Sec 4.1 forbids losing the
            # current event, so the completeness verdict must not be able to refuse this one.
            force_close=True,
        )

    def reset_episode(self, instruction: str | None = None, run_dir=None, logger=None) -> None:
        if self._trace_fh is not None:
            self._trace_fh.close()
            self._trace_fh = None
        if logger is not None:
            self.logger = logger
        # Persist previous episode counters before wipe (correct task id, not the next task).
        # `final=True`: this is an episode boundary, so outstanding demands are genuinely
        # unresolved for the episode that raised them - that is the memory-failure signature.
        self._dump_pmh_stats(final=True)
        self.frame_store_main = {}
        self.frame_store_wrist = {}
        self.K_indices_abs = []
        self.K_main_frames = []
        self.K_wrist_frames = []
        self.J_hist = []
        self.step = 0
        self.harness_extra_context = ""
        self.pinned_keyframe_steps = []
        self._pmh_last_context_idx = []
        self.salient_keyframe_steps = []
        self.memory_system_config = load_memory_system_config()
        self.proactive_cfg = ProactiveMemoryConfig.from_env()
        self.pcam = PcamController.create()
        self.ace = None
        self.scec = None
        tid = int(getattr(self.task_info, "task_id", 0) or 0)
        self._pmh_episode_task_id = tid
        if self.pcam is not None:
            self.pcam.reset()
            if self.episodic_store is None:
                self.episodic_store = create_store_if_needed(self.proactive_cfg)
            elif self.episodic_store is not None:
                self.episodic_store.reset()
        else:
            self.ace = AceController.create()
            if self.ace is not None:
                self.ace.reset()
                if self.episodic_store is None:
                    self.episodic_store = create_store_if_needed(self.proactive_cfg)
                elif self.episodic_store is not None:
                    self.episodic_store.reset()
            else:
                self.scec = ScecController.create()
                if self.scec is not None:
                    self.scec.reset(task_id=tid)
                    self.episodic_store = self.scec.store
                else:
                    if self.episodic_store is None:
                        self.episodic_store = create_store_if_needed(self.proactive_cfg)
                    elif self.episodic_store is not None:
                        self.episodic_store.reset()
        if pmh_enabled():
            if self.pmh_store is None:
                self.pmh_store = create_pmh_store_if_needed()
            elif self.pmh_store is not None:
                self.pmh_store.reset()
        else:
            self.pmh_store = None
        self._pmh_attach_seam_redactor()
        self._current_subtask = self.default_subtask_prompt
        self.read_stall_repeats = int(os.environ.get("READ_STALL_REPEATS", "3"))
        self.read_stall_extra_k = int(os.environ.get("READ_STALL_EXTRA_K", "2"))
        self.semantic_recent_n = int(os.environ.get("SEMANTIC_RECENT_N", "3"))
        self.pmh_max_tool_rounds = int(os.environ.get("PMH_MAX_TOOL_ROUNDS", "2"))
        self.pmh_active_visual_k = int(os.environ.get("PMH_ACTIVE_VISUAL_K", "3"))
        self.pmh_decide_recent = int(os.environ.get("PMH_DECIDE_RECENT", "2"))
        self._last_planned_subtask = ""
        self._consecutive_same_subtask = 0
        self._pmh_has_archive_visual = False
        self._pmh_exec_stall = False
        self._pmh_enter_stage = False
        self._pmh_new_segment = False
        self._pmh_stall_committed = False
        self._pmh_stall_read_done = False
        self._pmh_paper_read_count = 0
        self._pmh_last_boundary_deepen = -10_000
        self._pmh_last_retrieve_novel = False
        # v3.0: seeded per episode, next to the other per-episode PMH flags. The store is reset
        # at episode start, so the archived count it is compared against restarts at 0 too -
        # carrying a value across episodes would make the first commit of a new episode look
        # like a stall (or hide one).
        self._pmh_commits_at_last_mark = 0
        self._pmh_rel_cands = 0
        # v0.14: repeated-stall "change approach" nudge (second chance within an attempt).
        self._pmh_last_alt_stall = -10_000
        self._pmh_n_alt_inject = 0
        # v0.15: active-stage + occlusion-gate state (CGM-OB).
        self._pmh_current_stage = ""
        # P1 (job 586567): the absolute frame indices the planner actually SAW on the last
        # plan_step. Stage-indexed writes must archive frames from the step that triggered the
        # stage; deriving them from `active_frame_indices` let a stale window win, so stages
        # firing at t=1220 archived `[0, 6]` -- the episode's opening frames.
        self._pmh_last_context_idx: list[int] = []
        self._pmh_contract_hinted: set[str] = set()
        self._pmh_last_contract_hint = -10_000
        self._pmh_gap_hinted: set[str] = set()
        # PMH-O: tier-1 "already asked this demand" set, per episode. Cleared with the store
        # reset so a resumed episode does not inherit another episode's asked-set while its own
        # demand queue is empty (the counter-symmetry rule from job 570856).
        self._pmh_demand_asked = set()
        # v0.15b ablation: stage-name redaction (see _redact_planner_text). Terms are
        # re-registered from the answer key because this method clears them.
        self._redact_terms: set[str] = set()
        self._redact_n_masked = 0
        self.register_redact_answer_key()
        # v2.0 L0: per-attempt reset of the retrieval rights.
        self._pmh_retrievals_in_stage = 0
        self._pmh_nonproductive_streak = 0
        self._pmh_retrieval_refused = 0
        self._pmh_graph_forest = None
        self._pmh_graph_last_meta = {}
        self.hermes_inquiry_pending = False

        if run_dir is not None:
            self.run_dir = Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            save_trace = os.environ.get("SAVE_API_TRACE", "1").strip().lower() not in {"0", "false", "no", "off"}
            if save_trace:
                self._trace_fh = open(self.run_dir / "api_vlm_trace.jsonl", "w", encoding="utf-8")
            else:
                self._trace_fh = None
        else:
            self.run_dir = None
            self._trace_fh = None

    def pin_keyframe(self, step: int) -> None:
        if step >= 0 and step not in self.pinned_keyframe_steps:
            self.pinned_keyframe_steps.append(step)

    def _redact_enabled(self) -> bool:
        return os.environ.get("PMH_REDACT_STAGE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _pmh_attach_seam_redactor(self) -> None:
        """Hand SEAM the SAME redactor every other controller-synthesised string goes through.

        D2 was not only a formatting bug: the projection rendered `ob.raw`, the raw stage name,
        which names the destination container -- the answer key. That is the leak class job 576660
        documented for the address-demand queue, repeated in a new channel. Redaction has to be
        the same object, not a second implementation, or the two drift the way the alpha-guard's
        copy drifted from the planner's in job 581642.
        """
        store = getattr(self, "pmh_store", None)
        seam = getattr(store, "seam", None) if store is not None else None
        if seam is None:
            return
        try:
            seam.redact_fn = self._pmh_fair_inject
            logger.info("[seam] redactor attached (projection uses the planner's own mask)")
        except Exception:
            logger.exception("[seam] could not attach the redactor; projection stays unmasked")

    def _register_redact_terms(self, text: str) -> int:
        """v0.15b: collect the DESTINATION phrases that spell out a memory answer.

        Why this ablation exists: the benchmark's stage specs are PRESCRIPTIVE - the stage
        name encodes the object AND its destination container (task4's memory question is
        "which drawer already has an object?" - its task_block deliberately withholds it,
        "put butter into THAT non-empty drawer" - yet its stage 08 is literally
        `08_Put_Butter_Top_Drawer`, and the primitive label is "place butter into top
        drawer"). Harness injects those strings into the planner prompt every plan step, so
        the planner is handed the answer before any memory lookup could matter. Measured on
        job 565610: 0/8 episodes died on a memory-sensitive stage, perfect memory was worth
        +0.00pp, physical failures +36.01pp.

        v0.15c FIXES vs the first attempt (which produced `grasp_tomato_[withheld]_bottle`):
          * Only mask phrases that are QUALIFIER + CONTAINER-NOUN ("top drawer",
            "middle drawer"). A bare container noun ("microwave") is NOT a memory answer -
            there is no choice to remember - and masking it caused collateral damage on
            non-memory reasoning. Requiring a qualifier also means object words can never be
            registered: the old code took `tokens[2:]` blindly, so task22's stage
            `01_Lift_Tomato_Sauce` registered "sauce" and masked the object the task is about.
          * Do NOT mask whole primitive labels: the planner must be able to emit a valid
            primitive, and partial masking made labels unusable.

        Returns the number of newly registered phrases.
        """
        terms = getattr(self, "_redact_terms", None)
        if terms is None:  # tolerate callers that bypass __init__
            terms = self._redact_terms = set()
        before = len(terms)
        for phrase in _redact_container_phrases(text):
            terms.add(phrase)
            terms.add(phrase.replace(" ", "_"))
        return len(terms) - before

    def _redact_log(self, msg: str, *args: Any) -> None:
        """Emit a redaction log line, tolerating `logger=None`.

        `ApiMemoryPlanner(..., logger=None)` is the normal production path (the eval driver
        passes None and `reset_episode` later installs the per-episode logger), and the
        existing code always guards with `if self.logger:`. Registering the answer key happens
        during __init__, i.e. before any episode logger exists - calling self.logger.info
        there crashed job 566068 with `AttributeError: 'NoneType' object has no attribute
        'info'`. Falling back to the module logger keeps the message in the eval log, which
        the sbatch census relies on to prove redaction was not inert.
        """
        target = self.logger or logger
        target.info(msg, *args)

    def register_redact_answer_key(self) -> None:
        """Register the full answer key (task4: top/middle/bottom drawer) at construction.

        CRITICAL v0.15c FIX: registration used to happen only inside `pmh_note_stage_start`
        / `pmh_note_stage_done`, which the eval loop gates on `planner.pmh_store is not None`.
        The Harness ablation arm has no pmh_store, so it registered NOTHING - the census
        showed `masked_occurrences=0` for api_qwen_harness_v21_redact while the PMH arm got
        137. The "answer withheld" control was therefore NOT redacted at all, invalidating
        the whole A-vs-B contrast. `task_info.primitive_labels` carries the answer key for
        both arms, so registering from it here makes redaction arm-independent.
        """
        if not self._redact_enabled():
            return
        n = self._register_redact_terms(
            " ".join(str(x) for x in (getattr(self.task_info, "primitive_labels", None) or []))
        )
        n += self._register_redact_terms(getattr(self.task_info, "task_name", "") or "")
        self._redact_log(
            "[redact] enabled task=%s registered_phrases=%d terms=%s",
            getattr(self.task_info, "task_id", "?"),
            n,
            sorted(self._redact_terms),
        )

    def _redact_planner_text(self, text: str) -> str:
        """Mask every registered destination phrase in planner-visible text.

        Applied to the harness memory context and the PMH tool-decision stage hint. The
        task_block / scene description are deliberately NOT passed through here: they are the
        legitimate task specification, and t4's block is exactly what makes the memory
        question real (it names no drawer).
        """
        if not text or not self._redact_enabled():
            return text
        out = text
        for term in sorted(self._redact_terms, key=len, reverse=True):
            if len(term) < 4:
                continue
            out = re.sub(re.escape(term), "[withheld]", out, flags=re.IGNORECASE)
        if out != text:
            # getattr: tolerate partially-constructed planners (e.g. the preflight harness).
            self._redact_n_masked = getattr(self, "_redact_n_masked", 0) + 1
            # Log the first application per episode: the sbatch census greps this, so an
            # inert redaction (the job-565808 failure) fails the job instead of silently
            # invalidating the control arm for five hours.
            if self._redact_n_masked == 1:
                self._redact_log(
                    "[redact] applied task=%s terms=%s",
                    getattr(self.task_info, "task_id", "?"),
                    sorted(self._redact_terms),
                )
        return out

    def _pmh_fair_inject(self, text: str) -> str:
        """Put every CONTROLLER-SYNTHESISED injection through the same redaction as the hint.

        MEASURED DEFECT (job 576660, found while auditing that run's numbers). The address-demand
        queue derives its demands from the stage NAME:

            _pmh_note_binding_from_stage("01_Open_Top_Drawer") -> demand top_drawer.contents

        and `eval_fullvlm26_async_vlm_vla.py` hands that function `spec.name`, which is the
        UNREDACTED stage name. Tier 1 then injected

            address: top_drawer.contents
            question: what is inside the top drawer?

        straight into `extra_memory_text`, which was never passed through `_redact_planner_text`.
        So under `PMH_REDACT_STAGE=1` the arm withheld `top drawer` from the harness context and
        then handed the same phrase back through the demand queue.

        That does not merely leak the container - it leaks EXACTLY the token `HM` was denied, so
        the HM->PG contrast measured at -1.1 (vs -22.5 before the REDACT fix) is an OPTIMISTIC
        bound, not a clean one. The comment above the old injection claimed the question "carries
        no ANSWER"; that was true about contents and false about identity, and identity is what
        the withheld key encodes.

        The frame pointers are deliberately NOT redacted - they are the planner's own prior
        observations, which is the legitimate channel the demand queue exists to exploit. Only
        the phrases derived from the answer key are masked.

        Gated so the ablation stays available: OFF reproduces job 576660 byte for byte.
        """
        if not _truthy_env("PMH_FAIR_INJECT", default=True):
            return text
        return self._redact_planner_text(text)

    def pmh_note_stage_start(self, stage_name: str, step_idx: int) -> None:
        """v0.15: record the ACTIVE stage (what the robot is trying to achieve now).

        The occlusion gate needs the active stage's target object; `current_stage_name`
        only ever holds the last *completed* stage (set in note_stage_visual).
        """
        self._pmh_current_stage = str(stage_name or "").strip()
        self._register_redact_terms(stage_name)
        # v2.0 L0: the evidence budget belongs to the stage, so a new stage restores it.
        # Without this the right would be spent once at the first stage of the episode and
        # every later stage would run blind - the worst possible failure direction.
        self._pmh_retrievals_in_stage = 0
        self._pmh_nonproductive_streak = 0
        logger.info("[pmh] stage_start stage=%s t=%s", self._pmh_current_stage, step_idx)

    # ---- v2.0 L0: retrieval rights ---------------------------------------------------------
    # The planner may WANT evidence without limit; the controller decides whether the system
    # can AFFORD to answer. A refusal must never consume the plan step, otherwise the move
    # "block retrieval" silently becomes "waste the step", which is how v1.0 turned a memory
    # win on t4 into a 50pp loss (v1.0 anchored on gap-existence, not on a countable right).

    def _pmh_retrieve_right(self) -> tuple[bool, str]:
        """Can this plan step spend a retrieval? Default: unlimited (cap 0 => off)."""
        cap = int(os.environ.get("PMH_RETRIEVE_PER_STAGE", "0") or 0)
        np_max = int(os.environ.get("PMH_NONPRODUCTIVE_MAX", "0") or 0)
        if cap > 0 and self._pmh_retrievals_in_stage >= cap:
            return False, (
                f"retrieve_visual: the evidence budget for this stage is spent "
                f"({self._pmh_retrievals_in_stage}/{cap}). Further retrieval cannot add new "
                f"evidence. ACT NOW on the plan and evidence already available."
            )
        if np_max > 0 and self._pmh_nonproductive_streak >= np_max:
            return False, (
                f"retrieve_visual: the last {self._pmh_nonproductive_streak} retrievals returned "
                f"frames already on context (no new pixels). Memory is exhausted for this "
                f"question. ACT NOW; if the answer is still unknown, try a different approach "
                f"rather than asking memory again."
            )
        return True, ""

    def _pmh_note_retrieve_outcome(self, n_added: int) -> None:
        """Account a retrieval that actually ran. `n_added == 0` means no new pixels."""
        self._pmh_retrievals_in_stage += 1
        if int(n_added) > 0:
            self._pmh_nonproductive_streak = 0
        else:
            self._pmh_nonproductive_streak += 1

    def _pmh_cite_retrieve(
        self,
        question: str,
        *,
        on_context: set[int],
        bank_cap: int,
        step_idx: int,
    ) -> tuple[list[str], list[int], str, float]:
        """PMH-C: retrieve by the PROVENANCE of a belief, not by similarity to the question.

        Returns (notes, admitted_frames, outcome, cio) with outcome in
        {"hit", "miss", "blocked", "no_provenance", "fallback"}.

        WHY THIS REPLACES THE SIMILARITY SEARCH. `search_memory` returns Segment Cards in
        relevance order, and the top hit is the most recent relevant segment - which is precisely
        what the resident KF-spine bank already contains, because both are built from the same
        keyframe spine. Job 570810 measured the result frame by frame: t5 admitted 1 of 102
        retrieved frames (CIO 0.99), t14 4 of 39 (0.90), t4 3 of 41 (0.93). The candidate pool was
        redundant BEFORE `select_by_marginal_gain` ever ran, which is why raising the retrieval
        budget never changed anything (0 -> 1 -> 2 all measured as no-ops).

        The ledger already records which segment and step established each fact (`ts_src`), so the
        question can be answered by opening the segment that is ABOUT the fact instead of the
        segment that merely resembles the question. That segment is normally older than the recency
        window, so the two channels stop overlapping by construction. This is the "recover
        low-similarity complementary evidence" goal of GraphMemix (arXiv:2608.26983) realised with
        an EXACT pointer the system already maintains, rather than a learned relation verifier.

        ROUTING DEPTH IS ONE, AND THAT IS DELIBERATE. The retired three-rung ladder walked
        text -> keyframes -> full_visual across multiple candidate segments. The controlled study
        in arXiv:2607.17598 finds a second routing level "never reproduces this gain and sometimes
        breaks accuracy", and this project's own ladder agreed: 90/99 runs admitted nothing, and at
        rung 3 - the most expensive rung - `n_added` was still 0 on every measured t5 round. So the
        target is resolved ONCE (the provenance of the slot the question is about) and never
        re-routed. The two DETAIL levels of PMH.md Sec 6.3 are kept, because the spec requires them
        and they are a detail choice within one target, not a second routing decision.

        CIO GATING. If both detail levels admit nothing, the slot is recorded as unrecoverable and
        refused from then on without spending a retrieval. That makes the read path's cost bounded
        by one probe per slot, which is the property that lets this architecture be strictly
        no-worse than a closed read path in expectation - the thing that makes it shippable
        alongside a measured-best passive configuration.
        """
        store = self.pmh_store
        notes: list[str] = []
        if store is None:
            return notes, [], "miss", 1.0

        slot = resolve_slot_for_question(question, store.ts_facts)
        if not slot:
            # No ledger key matches the question, so there is nothing to cite. Falling back to
            # similarity is honest (a question about something the ledger never modelled) but it is
            # counted separately, because "the generator was never switched on this question" and
            # "the archive could not answer" are different failures. This path was 59% of job
            # 570856's citation attempts and had NO counter, so the largest single signal in that
            # run was invisible to the census that was built to measure it.
            store.n_cite_fallback += 1
            store.events.append(
                {"op": "cite", "query": question, "slot": "", "outcome": "fallback",
                 "cio": 1.0, "n_added": 0}
            )
            logger.info(
                "[pmh] cite FALLBACK query=%r: no ledger slot matches (keys=%s)",
                question[:70],
                sorted(store.ts_facts),
            )
            return notes, [], "fallback", 1.0

        if slot in store.ts_unrecoverable:
            store.n_cite_blocked += 1
            notes.append(
                f"[cite] `{slot}` was already fetched to exhaustion and added no new evidence. "
                "The archive cannot answer this. ACT NOW with what is on context."
            )
            return notes, [], "blocked", 1.0

        seg = parse_provenance(store.ts_src.get(slot, ""))
        if not seg:
            store.n_cite_no_provenance += 1
            notes.append(
                f"[cite] the belief `{slot}` has no segment provenance (seeded or rule-written), "
                "so there is no archive segment to open. ACT NOW."
            )
            return notes, [], "no_provenance", 1.0

        store.n_cite_retrieve += 1
        admitted: list[int] = []
        cio = 1.0
        # Detail level 1 (PMH.md Sec 6.3 `keyframes`), then detail level 2 (`full_visual`), on the
        # SAME resolved target. Escalation is one-way and one-step: no re-routing, no third level.
        #
        # PER-LEVEL CENSUS. `n_cands` is what the archive actually offered; nothing about the 0-hit
        # run of job 570856 could be attributed without it, because a level that offers 12 frames
        # and loses them all to the on-context filter and a level that offers 0 frames are the same
        # `n_added=0` after the fact.
        levels: list[dict[str, Any]] = []
        for detail in ("keyframes", "full_visual"):
            txt, idxs = store.retrieve_visual(segment_id=seg, detail=detail, mode="by_id")
            picked, skipped = self._pmh_admit_by_gain(
                idxs, on_context=on_context, cap=bank_cap
            )
            store.n_gain_skipped += skipped
            cio = citation_ci_overlap(len(picked), skipped)
            levels.append(
                {"detail": detail, "n_cands": len(idxs), "picked": len(picked), "skipped": skipped}
            )
            if picked:
                notes.append(
                    f"[cite] `{slot}` <- {store.ts_src.get(slot, '')} ({detail}): "
                    f"{txt.splitlines()[0] if txt else ''}"
                )
                admitted.extend(picked)
                store.n_cite_hit += 1
                store.events.append(
                    {"op": "cite", "query": question, "slot": slot, "seg": seg,
                     "outcome": "hit", "cio": round(float(cio), 3),
                     "n_added": len(picked), "levels": levels}
                )
                return notes, admitted, "hit", cio
        # Both detail levels produced nothing: the archive genuinely has no new evidence for this
        # slot. Record it so the same question is not paid for twice, and tell the planner to act -
        # this note is the give-up condition, and without it the loop has no exit.
        store.ts_unrecoverable.add(slot)
        store.n_cite_miss += 1
        # Which of the three causes it was, stated in the log so the census can classify misses
        # instead of lumping them. `nocands` means the archive could not even offer frames (they
        # are not in `frame_store_main`); `on_ctx` means it offered only frames the planner already
        # holds (the coupling between the kf-spine floor and provenance citation); `dup` means
        # near-duplicate pruning on `sim_floor` rejected them.
        _nc = sum(lv["n_cands"] for lv in levels)
        _ndup = sum(lv["skipped"] for lv in levels)
        _cause = "nocands" if _nc == 0 or all(lv["picked"] + lv["skipped"] == 0 for lv in levels) else (
            "dup" if _ndup else "on_ctx"
        )
        store.events.append(
            {"op": "cite", "query": question, "slot": slot, "seg": seg, "outcome": "miss",
             "cio": round(float(cio), 3), "n_added": 0, "levels": levels, "cause": _cause}
        )
        notes.append(
            f"[cite] opened the source of `{slot}` ({store.ts_src.get(slot, '')}) at both detail "
            f"levels: every frame is already on context (CIO={cio:.2f}). This slot is now marked "
            "unresolvable; do NOT ask about it again. ACT NOW."
        )
        logger.info(
            "[pmh] cite MISS slot=%s seg=%s cio=%.2f cause=%s levels=%s (marked unrecoverable)",
            slot,
            seg,
            cio,
            _cause,
            levels,
        )
        return notes, admitted, "miss", cio

    def _pmh_ladder(
        self,
        question: str,
        *,
        on_context: set[int],
        search_text: str,
    ) -> tuple[list[str], list[int], int]:
        """v3.0 (PMH.md Sec 5.2): walk the coarse-to-fine ladder until it stops paying.

        Returns (evidence_notes, newly_admitted_frames, depth) with depth in {1,2,3}.

        THE POINT IS THE BUDGET. Spec Sec 5.2 requires depth to be decided by the decision's
        need, not by a constant: "检索深度由当前决策需要决定，而不是每轮固定读取相同数量的文本和
        图像". The old arm read a fixed `PMH_VISUAL_BANK_K=8` regardless of the question. Here the
        ladder advances only while the previous rung failed to produce evidence that was not
        already on context - invariant I4 - so a question that Step 1 answers costs one text
        lookup and zero frames, while a question only Step 3 can answer pays for all three rungs.

        Escalation is driven by `n_added` rather than by a VLM judgement of "is this enough",
        because a judgement call would need another API round-trip per rung and would make the
        budget depend on the model's mood. "Did the last rung put new pixels in front of the
        planner" is a fact the harness already knows, and it is the same fact the novelty
        guarantee and the L0 guards already use, so the ladder cannot disagree with them.

        `depth` is recorded so the census can falsify prediction P4: a distribution collapsed on
        a single value means the budget is still effectively constant.
        """
        notes: list[str] = []
        added: list[int] = []
        store = self.pmh_store
        if store is None:
            return notes, added, 0
        max_rung = int(os.environ.get("PMH_LADDER_MAX", "3"))
        bank_cap = int(os.environ.get("PMH_VISUAL_BANK_K", "8"))

        # Which segments did the text rung point at? `search_memory` emits lines prefixed with
        # `- seg_XXX`; the text rung exists to give the visual rungs somewhere to look, which is
        # what spec Sec 6.2 promises ("只返回少量相关 segment 的文本信息和 ID").
        #
        # PMH_LADDER_TARGETS: an earlier version opened `ids[0]` and nothing else. Job 570778
        # measured what that costs. t5's whole attempt is the same five steps repeated 22 times:
        #   search "which drawer is empty?" -> seg_007 keyframes -> seg_007 full_visual -> n_added=0
        # `search_memory` returns its top hits in relevance order, and the top hit is the most
        # RECENT relevant segment - which is exactly the one the resident KF-spine bank already
        # covers, because both channels draw from the same keyframe spine (`PMH_KF_SPINE=1`).
        # The ladder therefore re-opened a segment whose frames were ALL already on context, every
        # single time, and `n_gain_picked=0 / n_gain_skipped=44` confirms it frame-by-frame. A
        # deterministic first-hit target makes the ladder a no-op whenever the top hit is bank-
        # covered, which is the common case by construction, not an unlucky one.
        # So the rung walks the hits IN ORDER and stops at the first that can actually add
        # evidence. The ladder's own escalation rule is unchanged: it only moves to the expensive
        # rung when the cheap one produced nothing (invariant I4).
        targets = _pmh_ladder_targets(search_text)
        store.n_ladder_search += 1
        depth = 1

        if max_rung >= 2:
            depth = 2
            for target in targets:
                txt, idxs = store.retrieve_visual(
                    segment_id=target, detail="keyframes", mode="by_id"
                )
                picked, skipped = self._pmh_admit_by_gain(
                    idxs, on_context=on_context, cap=bank_cap
                )
                store.n_ladder_keyframes += 1
                store.n_gain_skipped += skipped
                if picked:
                    notes.append(
                        f"[ladder] step2 keyframes seg={target}: "
                        f"{txt.splitlines()[0] if txt else ''}"
                    )
                    added.extend(picked)
                    break

        # Advance only if the text rung AND the keyframe rung produced nothing new: an empty
        # rung is the signal that this question needs the expensive rung (invariant I4).
        if max_rung >= 3 and not added:
            depth = 3
            for target in targets:
                txt, idxs = store.retrieve_visual(
                    segment_id=target, detail="full_visual", mode="by_id"
                )
                picked, skipped = self._pmh_admit_by_gain(
                    idxs, on_context=on_context, cap=bank_cap
                )
                store.n_ladder_full += 1
                store.n_gain_skipped += skipped
                if picked:
                    notes.append(
                        f"[ladder] step3 full_visual seg={target}: "
                        f"{txt.splitlines()[0] if txt else ''}"
                    )
                    added.extend(picked)
                    break
        return notes, added, depth

    def _pmh_admit_by_gain(
        self, candidates: list[int] | None, *, on_context: set[int], cap: int
    ) -> tuple[list[int], int]:
        """v3.0 (PMH.md Sec 6.4): admit frames by MARGINAL gain; no graph, no forest.

        Thin wrapper so every call site shares one selection policy - the retired graph selector
        owed its existence to this idea and was retired for its machinery, so it matters that
        the idea survives in a form that cannot be inert: a frame is admitted when it is not
        already on context and not a near-duplicate of what is already admitted, and selection
        STOPS when nothing left clears that bar instead of padding up to `cap` (invariant I4).
        """
        cands = [int(i) for i in (candidates or []) if int(i) in self.frame_store_main]
        # Split the "admitted nothing" outcome into its three possible causes BEFORE the on-context
        # filter below can hide them. Job 570856's 0-hit run could not be interpreted without this:
        # 22 citation attempts, CIO=1.00 every time, and no way to tell whether the archive held
        # nothing, held only frames already on context, or held only near-duplicates.
        st = self.pmh_store
        if not cands:
            if st is not None:
                st.n_gain_nocands += 1
            return [], 0
        n_on_ctx = sum(1 for i in cands if int(i) in {int(j) for j in on_context})
        if st is not None and n_on_ctx:
            st.n_gain_on_ctx += n_on_ctx
        if not _pmh_gain_enabled():
            room = max(0, int(cap) - len(on_context))
            return cands[:room], 0
        budget = max(0, min(int(cap), int(os.environ.get("PMH_GAIN_K", str(cap)))))
        picked, skipped = select_by_marginal_gain(
            cands,
            on_context=set(on_context),
            main_frames=self.frame_store_main,
            k_max=budget,
            sim_floor=float(os.environ.get("PMH_GAIN_SIM_FLOOR", "0.98")),
        )
        if picked:
            self.pmh_store.n_gain_picked += len(picked) if self.pmh_store else 0
        return picked, skipped

    def _pmh_bank_room(
        self, memory_indices: list[int], cap: int, step_idx: int, *, protect: int = -1
    ) -> None:
        """Make room in the on-context bank WITHOUT evicting the recent control frames.

        v3.0 F4. `memory_indices` serves two masters with one FIFO policy: it is the recency
        window the decide/plan step needs in order to see the CURRENT situation, and it is where
        retrieved archive evidence lands. A plain `pop(0)` cannot tell them apart, so a long
        ladder round can evict the frames that describe what the robot is doing right now - and
        the evidence it just fetched is itself evicted by the next fetch, so a multi-rung ladder
        can end up holding neither continuity nor evidence. That is a resource conflict, not a
        tuning issue: spec Sec 3.2 separates short-term (for control) from long-term (for
        decisions), and this bank was merging them under one eviction rule.

        Fix: never evict a frame from the last `protect` steps while a non-recent frame is
        available to drop. Only if EVERY entry is recent do we fall back to dropping the oldest,
        because shrinking the bank below the budget would remove evidence more silently than an
        eviction does.
        """
        limit = max(1, int(cap))
        if len(memory_indices) < limit:
            return
        if protect < 0:
            protect = int(os.environ.get("PMH_BANK_PROTECT_RECENT", "3"))
        cutoff = int(step_idx) - max(0, protect)
        # Drop the oldest entry that is NOT part of the protected recent window.
        for pos, idx in enumerate(memory_indices):
            if int(idx) < cutoff:
                memory_indices.pop(pos)
                return
        memory_indices.pop(0)

    def _pmh_bank_add(
        self, memory_indices: list[int], idx: int, cap: int, step_idx: int
    ) -> bool:
        """Append one frame to the bank, evicting safely. Returns True if it was added."""
        idx = int(idx)
        if idx in memory_indices:
            return False
        self._pmh_bank_room(memory_indices, cap, step_idx)
        memory_indices.append(idx)
        return True

    def _kf_spread_union(
        self, selected: list[int], recent_start: int, want: int
    ) -> list[int]:
        """Union `want` temporally-spread frames into a keyframe bank. Additive only.

        THE DEFECT THIS CLOSES. `merge_keyframe_bank` sources candidates exclusively from stage
        boundaries (`pinned_steps`) and subtask changes (`salient_steps`). Both are transition
        events, so the bank is a few anchors crowded around moments of change -- measured on job
        593817 at 1.7 frames injected per plan call against a `bank_max` of 8. The bank was
        therefore CANDIDATE-limited rather than cap-limited, which is why widening
        `cluster_distance` 6 -> 4 moved the density by ~14% and not by the amount the shape of the
        problem suggested. No hyperparameter of that builder can fix this: the candidate SET is
        the constraint.

        The official protocol's own description of channel B is "unlimited historical keyframes
        (K_MAX=0)", i.e. the intended bank is a broad record of the episode, not a list of
        transitions. With the official local VLM that breadth comes for free because the model
        NOMINATES frames and `build_visual_memory` clusters its nominations; an API Planner
        nominates nothing (measured: `J_hist` all-empty, `kf_n = 0`), so the breadth has to come
        from somewhere model-free. A fixed stride over the frame store is exactly that: a pure
        function of `(recent_start, frame_store_main)`, dependent on nothing any model wrote.

        Deliberately NOT a smarter salience rule (frame difference, object tracking). A stride is
        the first-order approximation and it is cheap: frames strictly older than the recent
        window are already in memory, the stride costs one pass over an index range, and the result
        is bounded by the caller's cap. A salience rule would need per-step image work and would
        put a model-adjacent heuristic into a channel whose whole point here is to be model-free.
        If the stride measures as inert, that is the cheap experiment that says a richer salience
        rule would have to earn its cost.
        """
        try:
            excl = set(int(i) for i in (selected or []))
            # Explicit, not incidental. The call site passes `cap - len(anchors)`, which is 0
            # whenever the bank is already full, and the stride below divides by `n` -- so without
            # this the common full-bank case would be handled by the `except` clause instead of by
            # logic, once per plan step, silently. Behaviourally identical, but a control path that
            # only works because an exception happens to be caught is not a control path.
            if int(want) <= 0:
                return sorted(excl)
            older = sorted(
                int(i)
                for i in self.frame_store_main
                if 0 <= int(i) < int(recent_start) and int(i) not in excl
            )
            if not older:
                return list(selected or [])
            # Evenly subsample so the added frames span the episode instead of clumping at one
            # end -- a naive `[-want:]` would hand the Planner the last `want` pre-window frames,
            # which are the ones already closest to its own context and therefore the least new
            # information. Same reasoning as `_sdv_substrate_floor`'s even subsample.
            #
            # STRATIFIED, NOT ENDPOINT-INCLUSIVE. The first form was `older[int(k * step)]`, whose
            # `k = 0` term is unconditionally `older[0]` -- the OLDEST frame in the pool -- on every
            # single plan step. That is the same failure the endpoint comment below was written to
            # guard against ("the first frame shows the transition"), except that reasoning does
            # not hold for the first frame OF THE EPISODE: it shows the initial scene, before the
            # robot has touched anything, not a transition. Measured on job 594675, where the
            # official builder already anchors frame 0 (the step-0 subtask change is a salient
            # event, so `merge_keyframe_bank` selects it): the pick was therefore frame 1, a
            # near-duplicate of frame 0, present on 70.1% of bank-bearing steps in `pushmem` and
            # 44.8% in `pullmem`, against 2-5% for every other low frame. One of eight slots was
            # spent on the episode's OPENING on almost every step -- the exact composition this
            # whole mechanism exists to avoid.
            #
            # Centring each pick in its own equal segment keeps what the endpoint form was FOR --
            # even coverage of the pool, so the bank spans the episode -- and drops the clinging.
            # The span stays effectively full: the first pick moves from 0 to len/2n, which for a
            # 500-frame pool is ~31 frames.
            n = min(int(want), len(older))
            step = len(older) / float(n)
            picks = sorted(
                {
                    older[min(len(older) - 1, max(0, int((k + 0.5) * step)))]
                    for k in range(n)
                }
            )
            out = list(selected or [])
            for i in picks:
                if i not in excl:
                    out.append(i)
                    excl.add(i)
            return sorted(out)
        except Exception:  # noqa: BLE001 - memory must not be able to break a plan step
            return list(selected or [])

    # ---------------------------------------------------------------------------------------

    def _sdv_substrate_floor(
        self, recent_start: int, *, exclude: set[int] | None = None
    ) -> tuple[list[Image.Image], list[Image.Image | None], list[int]]:
        """SDV L0: the DETERMINISTIC memory substrate.

        Design constraint, and the reason this does not simply call `build_visual_memory`:

            `build_visual_memory` is a pure function of `J_hist`. `J_hist` is a list of the
            planner's OWN `keyframe_positions` nominations. The API planner never nominates any,
            so in this arm `J_hist == [[], [], [], ...]` -- populated, but with zero candidates.

        Job 586700 measured the consequence: `kf_n=0` on every plan_step, `K_indices_abs` empty,
        `build_visual_memory` returning nothing, and therefore L0 empty too -- while the census
        dutifully reported `substrate_injected=0 substrate_empty=0`, because an empty substrate
        with no eligible candidates is not a violation of anything. Availability was chained to the
        agent's output after all, one level up from the write path I had just removed it from.

        So L0 is taken over the FRAME STORE: a uniform stride through every frame strictly older
        than the recent window. That is a pure function of `(recent_start, frame_store_main)` and
        of nothing the planner, the memory store or the archive ever wrote. HM's candidate-derived
        bank is then UNIONed in when it has something, because it is a good salience signal -- but
        it can only ADD. That asymmetry is the whole invariant: the agent may enrich the substrate,
        never empty it.

        Exclusion of the recent window is not an optimization: those frames are already in the
        plan_step's own context, and re-injecting them is precisely the duplicate that made
        `n_new=0` uninformative in 586567.
        """
        excl = set(exclude or set())
        stride = max(1, int(os.environ.get("PMH_SDV_STRIDE", "40")))
        cap = max(1, int(os.environ.get("PMH_SDV_SUBSTRATE_K", "8")))
        older = sorted(
            i for i in self.frame_store_main if int(i) >= 0 and int(i) < int(recent_start) and int(i) not in excl
        )
        cand: list[int] = list(older[::stride])
        # Union the candidate-derived bank when it exists. Additive only.
        if self.use_keyframe_memory and self.J_hist:
            try:
                for i in build_visual_memory(
                    self.J_hist,
                    t=self.step,
                    N=max(1, int(self.step - recent_start)),
                    d=self.d_merge,
                ):
                    i = int(i)
                    if i >= 0 and i < int(recent_start) and i not in excl and i in self.frame_store_main:
                        cand.append(i)
            except Exception:  # noqa: BLE001 - a substrate must not be able to crash a plan_step
                pass
        idxs = sorted({int(i) for i in cand})
        if not idxs:
            return [], [], []
        # Even subsample down to the cap, so the substrate is spread across the episode rather
        # than a dense cluster at one end (which is what a naive `[-cap:]` would give).
        if len(idxs) > cap:
            step = len(idxs) / float(cap)
            idxs = [idxs[min(len(idxs) - 1, int(i * step))] for i in range(cap)]
            idxs = sorted(set(idxs))
        mains = get_frames_from_indices(idxs, self.frame_store_main)
        wrists = [self.frame_store_wrist.get(i) for i in idxs]
        return mains, wrists, idxs

    def _sdv_substrate_inject(
        self,
        memory_main_frames: list[Image.Image],
        memory_wrist_frames: list[Image.Image | None],
        memory_indices: list[int],
        recent_start: int,
        extra_memory_text: str,
    ) -> tuple[list[Image.Image], list[Image.Image | None], list[int], str]:
        """Attach L0 to whatever the read path produced, and instrument S0.

        The substrate is APPENDED and is never subject to the gain gate, the novelty rule, or any
        planner decision. That is the whole architectural claim: availability is orthogonal to
        choice. If the agent's demand channel is entirely silent, the planner still sees the
        deterministic floor -- which is exactly the regime 586617 was stuck in (26/21 retrievals
        with `n_new=0`).
        """
        sub_main, sub_wrist, sub_idxs = self._sdv_substrate_floor(
            recent_start, exclude=set(memory_indices)
        )
        # S0 is only meaningful as a falsifier when there EXIST candidates outside the recent
        # window. Early in an episode every candidate is inside the window, so an empty substrate
        # is then correct rather than a failure -- and a guard that fires on the correct case is
        # worthless, because it trains the operator to ignore it. Precisely: violation = "the
        # trajectory holds an eligible frame and the substrate did not surface it".
        eligible = {
            int(i) for J in (self.J_hist or []) for i in J if int(i) >= 0 and int(i) < int(recent_start)
        }
        eligible &= set(self.frame_store_main)
        if sub_idxs:
            self.pmh_store.n_substrate_injected += 1
            memory_indices = list(memory_indices) + list(sub_idxs)
            memory_main_frames = list(memory_main_frames) + list(sub_main)
            memory_wrist_frames = list(memory_wrist_frames) + list(sub_wrist)
            extra_memory_text += (
                f"\nsubstrate_frames: {sub_idxs} "
                "(deterministic keyframes from the trajectory; ALWAYS on, not retrievable away)\n"
            )
        elif eligible:
            # S0 violation. A nonzero count here means the substrate silently produced nothing
            # while eligible candidates existed -- the exact signature that cost four mechanisms
            # in 586567 and that no counter caught at the time.
            self.pmh_store.n_substrate_empty += 1
            logger.warning(
                "[sdv] S0 VIOLATION: substrate EMPTY at t=%s while %s eligible candidate(s) "
                "existed outside the recent window (recent_start=%s, J_hist=%s)",
                self.step,
                len(eligible),
                recent_start,
                len(self.J_hist),
            )
        return memory_main_frames, memory_wrist_frames, memory_indices, extra_memory_text

    def _sdv_verify_stage(
        self,
        stage_name: str,
        step_idx: int,
        picked: list[int],
        *,
        on_screen: list[int] | None = None,
    ) -> None:
        """SDV L3/H4: an INDEPENDENT verifier decides whether the stage actually happened.

        The gap this fills is the oldest one in the project and the one no counter had closed:
        `verify(action_trace, expected_transition)` was specified in PMH.md's unified model and
        never implemented, so every "did it happen" fact in the system was a claim the planner
        wrote about itself. For the counting tasks that is fatal by construction -- t8 and t22
        ("pour sauce twice", HM >=66.7, PG 0.0 in job 586567's contrast) are scored on the NUMBER
        of verified transitions, and a self-asserted count cannot be evidence for it.

        Design constraints taken from the failures already measured:
          * a SEPARATE stateless call, not the planner's own context, so it cannot simply agree
            with whatever the planner just said;
          * `temperature=0` and a closed 3-way verdict, so the answer is a reading rather than a
            sample (the API planner's sampling is a 28.85pp noise source across jobs);
          * failures are COUNTED, never silent (verdict `not_achieved` records nothing but
            increments a counter, `ambiguous` is first-class);
          * the result is written ONLY through `merge_verified_fact`, i.e. into the reserved
            namespace the planner cannot reach.
        """
        if self.pmh_store is None:
            return
        if not _truthy_env("PMH_SDV_VERIFY"):
            # Counted, not merely skipped. A verifier that is switched off and a verifier that
            # judges everything the same are indistinguishable in a success rate; in job 586700
            # the verifier produced no log line at all for 9 stage boundaries and no counter moved,
            # which is exactly the shape of the silent mechanisms this project keeps paying for.
            self.pmh_store.n_sdv_disabled += 1
            return
        # EVIDENCE SELECTION, and the defect this fixes. Job 586794 ran 26 verify calls and
        # returned `achieved` ZERO times, while t1-t3 scored 100 -- i.e. those stages provably DID
        # complete and the verifier said they had not. The cause was that the evidence came from
        # the stage write's archive (`picked`), and 10 of those 26 archives were `frames=[0, 6]`:
        # the episode's OPENING frames, from before anything had happened. Handed an opening frame
        # and asked "is this stage complete?", `not_achieved` is the CORRECT reading. The verifier
        # was not wrong; the evidence was.
        #
        # So the verifier now takes the frames that were actually ON SCREEN when the stage fired
        # (`on_screen`, the plan_step context at `step_idx`, whose last element IS `step_idx`).
        # That is the only set that can show the stage's outcome, and it is independent of the
        # archive write path whose staleness has now caused two separate failures (P1 in 586617,
        # and this). The archive is used only as a fallback, and a fallback that predates the stage
        # is COUNTED, because a verifier fed impossible evidence produces confident wrong verdicts
        # rather than an error.
        win = max(1, int(os.environ.get("PMH_SDV_EVIDENCE_WINDOW", "2")))
        usable = [int(i) for i in (on_screen or []) if int(i) in self.frame_store_main and int(i) >= 0]
        frames: list[int] = []
        if usable:
            frames = usable[-win:]
        else:
            fallback = [int(i) for i in (picked or []) if int(i) in self.frame_store_main]
            frames = fallback[-win:] if fallback else []
            if frames and max(frames) < int(step_idx):
                self.pmh_store.n_sdv_evidence_stale += 1
                logger.warning(
                    "[sdv] verify evidence STALE stage=%s t=%s frames=%s (fell back to the archive "
                    "and every frame predates the stage, so the verdict cannot be about it)",
                    stage_name,
                    step_idx,
                    frames,
                )
        if frames and max(frames) < int(step_idx) - 1:
            # Even on-screen frames can lag if the stage fires between plan_steps. Counted for the
            # same reason as above: the verdict is still usable (the stage outcome persists in the
            # scene) but a LARGE lag is a different regime, and it must not be discovered later.
            self.pmh_store.n_sdv_evidence_lagging += 1
        if not frames:
            self.pmh_store.n_sdv_ambiguous += 1
            logger.info("[sdv] verify skip stage=%s reason=no_frames t=%s", stage_name, step_idx)
            return
        max_img = max(1, int(os.environ.get("PMH_SDV_VERIFY_IMAGES", "4")))
        use = frames[-max_img:]
        imgs = get_frames_from_indices(use, self.frame_store_main)
        label = re.sub(r"^\d+_", "", str(stage_name or "")).replace("_", " ").strip()
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "These frames are the visual evidence recorded for ONE manipulation stage of a "
                    f"robot task. The stage was believed to be: '{label}'.\n\n"
                    "Judge ONLY what the images show. Do not assume the stage succeeded.\n\n"
                    "Answer with strict JSON and nothing else:\n"
                    '{"verdict":"achieved|not_achieved|ambiguous","evidence":"<=15 words"}\n\n'
                    "- achieved: the images clearly show this stage's physical outcome holding.\n"
                    "- not_achieved: the images clearly show it did NOT hold.\n"
                    "- ambiguous: the images do not decide it (occluded, wrong view, too early)."
                ),
            }
        ]
        for img in imgs:
            user_content.append({"type": "image", "image": img})
        # Entry log, BEFORE the API call. In job 586700 the verifier emitted no line at all on 9
        # stage boundaries, so "returned early", "raised and was swallowed" and "never called" were
        # all consistent with the evidence. A function that can be silent about whether it ran is a
        # function whose behaviour is unobservable, so this line is the minimum contract.
        logger.info(
            "[sdv] verify enter stage=%s t=%s frames=%s verify_env=%r",
            stage_name,
            step_idx,
            use,
            os.environ.get("PMH_SDV_VERIFY"),
        )
        try:
            # `infer_primitive_via_api`, not `chat_completions`: the latter serializes the payload
            # directly (`json.dumps`) and therefore cannot carry images at all. Using it here raised
            # `TypeError: Object of type Image is not JSON serializable` on every stage boundary of
            # job 586759 -- loud only because the entry log and the try/except were added there
            # after the same silent no-op cost a full 26-task run in 586700.
            out = infer_primitive_via_api(
                system_prompt="You are a strict visual verifier. Output JSON only.",
                user_content=user_content,
                api_key=self.api_key,
                base_url=self.api_base_url,
                model=self.api_model,
                timeout_sec=min(float(os.environ.get("PMH_SDV_TIMEOUT", "120")), self.api_timeout),
                max_tokens=int(os.environ.get("PMH_SDV_TOKENS", "256")),
            )
        except Exception as exc:  # noqa: BLE001 - a verifier fault must be LOUD, never a no-op
            self.pmh_store.n_sdv_ambiguous += 1
            logger.error("[sdv] verify RAISED stage=%s t=%s exc=%r", stage_name, step_idx, exc)
            return
        verdict, evidence = "ambiguous", ""
        if out:
            start, end = out.find("{"), out.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(out[start : end + 1])
                    verdict = str(obj.get("verdict", "ambiguous")).strip().lower()
                    evidence = str(obj.get("evidence", ""))[:200]
                except Exception:  # noqa: BLE001
                    verdict = "ambiguous"
            if verdict not in {"achieved", "not_achieved", "ambiguous"}:
                verdict = "ambiguous"
        # The stage label is the fact's ADDRESS; the value is the observed outcome. Keyed by the
        # stage so that a later contradicting verification supersedes rather than duplicates.
        key = f"stage.{label.lower().replace(' ', '_')}.done"
        merge_verified_fact(
            self.pmh_store,
            key,
            "true",
            verdict=verdict,
            evidence=evidence or f"frames={use}",
        )
        logger.info(
            "[sdv] verify stage=%s verdict=%s frames=%s evidence=%r t=%s "
            "verified_n=%s not_achieved=%s ambiguous=%s",
            stage_name,
            verdict,
            use,
            evidence,
            step_idx,
            self.pmh_store.n_sdv_verified,
            self.pmh_store.n_sdv_not_achieved,
            self.pmh_store.n_sdv_ambiguous,
        )

    def pmh_note_stage_done(self, stage_name: str, step_idx: int) -> None:
        """Stage-indexed Evidence write: prefer pointers into dense KF (v0.8)."""
        if self.pmh_store is None:
            return
        self._register_redact_terms(stage_name)
        # SEAM Epsilon: the benchmark's OWN stage confirmation is the strongest verdict source
        # available, and it is the only one that can settle a count -- `03_Pour_Two` confirming is
        # exactly one more pour than `02_Pour_One` confirming. Recorded HERE because this is the
        # one call site that fires on a real grader predicate; the planner's declared primitive
        # (recorded below in `plan_step`) deliberately cannot write a verdict (E2).
        if self.pmh_store.seam is not None:
            try:
                frame_hint = -1
                ctx0 = getattr(self, "_pmh_last_context_idx", None) or []
                if ctx0:
                    frame_hint = int(ctx0[-1])
                # Epsilon: the benchmark's OWN stage confirmation is the strongest verdict source
                # available, and the only one that can settle a count -- `03_Pour_Two` confirming
                # is exactly one more pour than `02_Pour_One` confirming, so ordinality falls out
                # of the confirmation sequence. Recorded HERE because this is the one call site
                # that fires on a real grader predicate (E2); the planner's declared primitive,
                # recorded in `plan_step`, deliberately cannot write a verdict.
                self.pmh_store.seam.note_check(stage_name, int(step_idx), frame_hint)
            except Exception:
                logger.exception("[seam] stage-done verdict write failed; memory continues")
        # The scene half is written AFTER `picked` is known, so every scene cell carries a frame
        # pointer that is genuinely held by the frame store. Writing it earlier -- as the previous
        # arm did, with `frame_hint` alone -- means the pointer can be -1, and an S1 rejection is
        # then indistinguishable from an absent source, which is the defect class D1 belongs to.
        # P1 (job 586567): a stage write must archive the frames that were on screen when the
        # stage fired. Previously the candidate pool was `active_frame_indices[-24:]` alone, which
        # is a rolling accumulator that lags or never advances, so 30/39 writes stored the
        # episode's OPENING frames (`frames=[0, 6]`) for stages firing at t=88..1288. The write
        # then looked perfectly healthy while archiving evidence that answered nothing.
        ctx = [
            int(i)
            for i in (getattr(self, "_pmh_last_context_idx", None) or [])
            if int(i) >= 0
        ]
        anchors = [int(p) for p in self.pinned_keyframe_steps[-3:] if int(p) >= 0]
        cands, dropped = _pmh_stage_write_candidates(
            context_idx=ctx[-24:],
            anchors=anchors,
            step_idx=int(step_idx),
            frame_store=self.frame_store_main,
            fallback=self.pmh_store.active_frame_indices[-24:],
        )
        if dropped:
            logger.info(
                "[pmh] stage_visual_write fell back to the frame accumulator: none of the "
                "plan_step-anchored frames %s are held (t=%s)",
                dropped,
                step_idx,
            )
        write_k = int(os.environ.get("PMH_STAGE_WRITE_K", "2"))
        kf_spine = os.environ.get("PMH_KF_SPINE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        picked_cands = self._pmh_snap_to_kf(cands, max_k=write_k) if kf_spine else cands
        picked = self.pmh_store.note_stage_visual(stage_name, picked_cands, int(step_idx), max_k=write_k)
        if picked:
            # v0.10: a stage just finished → the next plan enters a new phase where the
            # planner may need archived evidence beyond the on-context bank (boundary deepen).
            self._pmh_enter_stage = True
            # P1 falsifier: "healthy write" != "correct archive". A stage at t archives frames
            # that PREDATE it → the bank can only answer with pixels already on screen, which is
            # exactly the `retrieve n_new=0` signature. Count it instead of trusting the write.
            lag = int(step_idx) - max(int(i) for i in picked)
            lag_cap = int(os.environ.get("PMH_STAGE_STALE_LAG", "64"))
            if lag >= lag_cap:
                self.pmh_store.n_stage_write_stale += 1
                logger.warning(
                    "[pmh] stage_visual_write STALE stage=%s frames=%s t=%s lag=%s cap=%s",
                    stage_name,
                    picked,
                    step_idx,
                    lag,
                    lag_cap,
                )
        logger.info(
            "[pmh] stage_visual_write stage=%s frames=%s t=%s kf_spine=%s enter_stage=%s stale_n=%s",
            stage_name,
            picked,
            step_idx,
            int(kf_spine),
            int(bool(self._pmh_enter_stage)),
            int(getattr(self.pmh_store, "n_stage_write_stale", 0) or 0),
        )
        logger.info(
            "[sdv] census substrate_injected=%s substrate_empty=%s kf_nom_empty=%s kf_nom_nonempty=%s "
            "verified=%s not_achieved=%s ambiguous=%s ev_stale=%s ev_lag=%s "
            "verified_write_rejected=%s verified_facts=%s",
            int(getattr(self.pmh_store, "n_substrate_injected", 0) or 0),
            int(getattr(self.pmh_store, "n_substrate_empty", 0) or 0),
            int(getattr(self.pmh_store, "n_kf_nomination_empty", 0) or 0),
            int(getattr(self.pmh_store, "n_kf_nomination_nonempty", 0) or 0),
            int(getattr(self.pmh_store, "n_sdv_verified", 0) or 0),
            int(getattr(self.pmh_store, "n_sdv_not_achieved", 0) or 0),
            int(getattr(self.pmh_store, "n_sdv_ambiguous", 0) or 0),
            int(getattr(self.pmh_store, "n_sdv_evidence_stale", 0) or 0),
            int(getattr(self.pmh_store, "n_sdv_evidence_lagging", 0) or 0),
            int(getattr(self.pmh_store, "n_sdv_verified_write_rejected", 0) or 0),
            len(getattr(self.pmh_store, "verified_facts", {}) or {}),
        )
        # SEAM SCENE (D1 FIX). Unconditional: no env gate stands between a confirmed stage and the
        # scene cell it implies. The frame pointer is taken from the frames this stage actually
        # archived (`picked`), falling back to the on-screen context and only then to the raw step,
        # so the pointer names a frame the planner can really be shown.
        if self.pmh_store.seam is not None:
            try:
                ptr = -1
                for src in (list(picked or []), list(ctx or []), [int(step_idx)]):
                    src = [int(i) for i in src if int(i) >= 0]
                    if src:
                        ptr = int(src[-1])
                        break
                seam = self.pmh_store.seam
                wrote = seam.mirror_from_store(self.pmh_store, step=int(step_idx))
                for ent, attr, val in _stage_scene_facts(stage_name):
                    if seam.note_scene(
                        ent, val, attribute=attr, frame_idx=ptr, step=int(step_idx), source="check"
                    ):
                        wrote += 1
                if picked:
                    seam.note_scene_anchor(int(picked[-1]), stage_name, step=int(step_idx))
                # D1 falsifier: the store held evidence and we still wrote nothing. Any non-zero
                # reading here means the silent-zero defect is back, and the sbatch gates on it.
                has_evidence = bool(
                    getattr(self.pmh_store, "contracts", None)
                    or getattr(self.pmh_store, "segments", None)
                    or _stage_scene_facts(stage_name)
                )
                if not wrote and has_evidence:
                    seam.n_scene_sources_empty += 1
                    logger.warning(
                        "[seam] scene write produced NOTHING while evidence existed "
                        "stage=%s ptr=%s contracts=%s segments=%s",
                        stage_name,
                        ptr,
                        len(getattr(self.pmh_store, "contracts", []) or []),
                        len(getattr(self.pmh_store, "segments", []) or []),
                    )
                logger.info(
                    "[seam] scene_write stage=%s cells=%s wrote=%s ptr=%s anchors=%s empty_n=%s",
                    stage_name,
                    seam.scene_cells(),
                    wrote,
                    ptr,
                    seam.n_scene_anchor,
                    seam.n_scene_sources_empty,
                )
            except Exception:
                logger.exception("[seam] scene write failed; memory continues without it")
        self._pmh_note_placement_from_stage(stage_name, int(step_idx))
        self._pmh_note_binding_from_stage(stage_name, int(step_idx))
        # SDV L3/H4: the independent verification of this stage's outcome. Runs AFTER the archive
        # write so it can be shown the frames this stage actually recorded, and its verdict lands
        # in the reserved namespace no planner write can reach.
        self._sdv_verify_stage(
            stage_name,
            int(step_idx),
            list(picked or []) or list(cands),
            on_screen=ctx,
        )
        self._dump_pmh_stats()

    def _pmh_episode_n_stages(self) -> int:
        """Counted stages for this episode; 99 when unknown.

        RETIRED as a read gate (PMH-O). It used to back `_pmh_ce_short_episode`, a hardcoded
        `{t1: 2, t4: 9, ...}` per-task table whose stated purpose was to stop short tasks from
        burning a read. That is a per-task rule, which PMH.md Sec 11 forbids outright, and it was
        also the wrong predicate: t1/t18 should not read because their spec names no container,
        so NO address demand can exist - a fact the demand queue derives from the episode rather
        than from a table that has to be kept in sync with the stage lists by hand.

        Kept only for the diagnostic log line, where a stage count is still useful context.
        """
        tid = int(getattr(self, "_pmh_episode_task_id", 0) or getattr(self.task_info, "task_id", 0) or 0)
        # Hardcoded from task2_26_reference_stage._task_specs — avoid importing that module
        # into the API planner process (heavy deps). Keep in sync when stage lists change.
        n_by_tid = {1: 2, 4: 9, 5: 9, 11: 6, 14: 6, 16: 3, 18: 2, 22: 3}
        if tid in n_by_tid:
            return int(n_by_tid[tid])
        raw = os.environ.get("PMH_CE_STAGE_COUNT", "").strip()
        if raw.isdigit():
            return int(raw)
        return 99

    def _pmh_note_binding_from_stage(self, stage_name: str, step_idx: int = -1) -> None:
        """D4 → PMH-O: turn a check-confirmed stage name into an ADDRESS DEMAND, not a fact.

        WHAT D4 GOT WRONG, and why this function no longer writes the ledger.

        D4 read a stage name and wrote the ledger:

            01_Open_Top_Drawer        -> top_drawer.status: open
            02_Close_Top_Drawer       -> top_drawer.status: closed
            02_Place_Cookies_Top_Drawer -> top_drawer.status: occupied

        Three separate defects, each structural rather than a tuning miss:

        1. A stage name records what the robot DID, not what the robot SAW. "The drawer was
           opened" does not entail "the drawer is empty", and the empty-drawer fact is exactly
           what t4/t5 (the memory-decisive tasks) ask for. So the writer could not, in
           principle, produce the one fact the read mechanism existed to find - which is why
           `binding_write` counted 7/11/12 while the claim moved nothing.
        2. `status` carried two orthogonal properties (position: open/closed, contents:
           empty/occupied), so the last writer destroyed the other. Fixed in the ledger by
           invariant I3; this function must no longer emit the conflated key at all.
        3. It overwrote the consolidator's own `object_status` with a narrower value. The
           destructive write is now rejected by the ledger (the key is not an address under
           I2/I3), so it cannot happen - but the fix must also stop PRODUCING it.

        The salvageable and genuinely free part of D4 is that the stage name PROVES a question
        is open: a stage that names a container the planner cannot identify ("the non-empty
        drawer") raises a demand for that container's contents, and the controller may raise a
        demand without answering it. That asymmetry is the whole architecture, so it is what
        remains here.

        Gated by PMH_BINDING_WRITE (default off; the api_pmh_ce arm turns it on).
        """
        store = self.pmh_store
        if store is None:
            return
        on = os.environ.get("PMH_BINDING_WRITE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if not on:
            return
        self._pmh_raise_demands_for_phrase(stage_name, step_idx, source="stage")

    @staticmethod
    def _pmh_phrase_tokens(phrase: str) -> list[str]:
        """Normalise a stage name OR a planner primitive into comparable tokens.

        THE TWO SOURCES SPELL THE SAME EVENT DIFFERENTLY and the demand queue must accept both:
            harness-confirmed stage : 01_Open_Top_Drawer
            planner primitive       : "open top drawer" / "open the top drawer"

        Job 586421 made the asymmetry fatal rather than cosmetic. The demand queue was reachable
        ONLY through the stage-confirm path (`_pmh_note_stage_visual`), so on t4/t5/t8/t9 -- the
        memory-decisive tasks, where nothing confirms because the robot is stuck -- no demand was
        ever raised (`n_bindings_opened=0`, no `binding_demand` line in those windows), the ledger
        stayed empty, and the planner's correct response to an empty ledger was `{"tool":"none"}`.
        Meanwhile t11/t12/t13, which confirmed stages normally, did raise demands. The channel was
        wired to the one signal that failure destroys.

        The planner declares its primitive every single step regardless of whether a stage ever
        completes, so that declaration is the evidence source that survives.
        """
        s = re.sub(r"^\d+_", "", str(phrase or "").strip()).replace("-", " ").replace("_", " ")
        toks = [
            t
            for t in re.split(r"[^a-z0-9]+", s.lower())
            if t and t not in {"the", "a", "an", "again", "final", "into", "in", "to", "then"}
        ]
        return toks

    def _pmh_raise_demands_for_phrase(
        self, phrase: str, step_idx: int = -1, *, source: str = "stage"
    ) -> int:
        """Raise address demands provable from `phrase`. Returns how many were newly opened.

        Shared by both evidence sources so a fix to the derivation lands on both. `source` is
        carried only into the log, because the two callers mean different things: a confirmed
        stage additionally proves the action HAPPENED, whereas a declared primitive proves only
        that a question is open -- and it is the latter that must exist when nothing completes.
        """
        store = self.pmh_store
        if store is None:
            return 0
        toks = self._pmh_phrase_tokens(phrase)
        if len(toks) < 2:
            return 0
        action = toks[0]
        demands: list[tuple[str, str, str]] = []
        if action in {"open", "close"} and len(toks) >= 2:
            # Open_Top_Drawer / "open top drawer" -> top_drawer.contents
            container = "_".join(toks[1:]).strip("_")
            if len(container) >= 2:
                # The demand an Open stage PROVES is about contents: the stage exists in the
                # task precisely because something in that container had to be established,
                # and the planner is being shown the container but not what is in it.
                demands.append(
                    (
                        f"{container}.contents",
                        "open_stage",
                        f"what is inside the {container.replace('_', ' ')}?",
                    )
                )
        elif action in {"place", "put", "placing"} and len(toks) >= 3:
            # Only when the controller can prove a gap: the stage names a container the
            # planner-visible spec does not. Otherwise the destination is already in the task
            # text and no address is missing - opening a demand would re-spend reads on closed
            # gaps (the v0.15e over-retrieval shape on t11/t22).
            parsed = self._stage_container(str(phrase))
            if parsed is not None:
                kind, container_phrase = parsed
                text = " ".join(
                    str(getattr(self.task_info, attr, "") or "")
                    for attr in ("task_block", "brief_description", "scene_description")
                ).lower()
                if container_phrase not in text:
                    demands.append(
                        (
                            f"{kind}#ref.contents",
                            "place_by_reference",
                            f"which {kind} is the one the task refers to?",
                        )
                    )
                elif any(
                    p in text
                    for p in ("where the", "location where", "same place", "same location")
                ):
                    demands.append(
                        (
                            f"{kind}#ref.location",
                            "place_by_prior_location",
                            f"where inside the {kind} was the earlier object placed?",
                        )
                    )
        if not demands:
            return 0
        opened = 0
        for addr, why, question in demands:
            if store.open_binding(addr, why=why, stage=" ".join(toks), question=question):
                opened += 1
        store.n_binding_writes = int(getattr(store, "n_binding_writes", 0) or 0) + 1
        if source == "evidence":
            store.n_obligations_from_evidence = int(
                getattr(store, "n_obligations_from_evidence", 0) or 0
            ) + 1
        logger.info(
            "[pmh] binding_demand stage=%s opened=%s addrs=%s why=%s source=%s t=%s",
            phrase,
            opened,
            [a for a, _w, _q in demands],
            [w for _a, w, _q in demands],
            source,
            step_idx,
        )
        return opened

    def _pmh_note_obligation_from_evidence(
        self, primitive: str, step_idx: int = -1
    ) -> None:
        """Raise an address demand from the planner's OWN declared primitive.

        This is the second trigger source, and its absence was the job-586421 root cause: the
        demand queue hung off stage confirmation, which never fires on the memory-decisive tasks.
        A declared primitive is weaker evidence than a confirmed stage -- it proves the question
        is OPEN, not that the action happened -- and that is exactly the strength needed, because
        "the planner is trying to open the top drawer" already entails that `top_drawer.contents`
        must be known before it can place anything by reference.

        `open_binding` dedupes by address and refuses to ask for what the ledger already answers,
        so calling this every step cannot inflate the queue or re-spend reads.
        """
        if not _truthy_env("PMH_OBLIGATION_FROM_PRIMITIVE", default=False):
            return
        if os.environ.get("PMH_BINDING_WRITE", "0").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return
        self._pmh_raise_demands_for_phrase(primitive, step_idx, source="evidence")

    def _pmh_demand_tier1(self, step_idx: int) -> tuple[str, list[int]]:
        """Tier 1 of the read predicate: raise each open address demand as a QUESTION, once.

        Fires at most once per address per episode. Re-injecting the same question every decide
        would turn a knowledge signal into a per-step token tax, and the planner cannot learn
        anything new from being asked the same thing twice in a row - it either answered it from
        its own frames the first time or it did not.

        When tier 1 has already fired and the demand is still open, the caller's tier 2
        (`demand_read`) is the escalation, after `PMH_DEMAND_GRACE` decides. So the two tiers
        partition the work: tier 1 spends the planner's own already-resident evidence, tier 2
        spends the archive, and a demand that tier 1 closes costs zero retrieval.
        """
        if not self._pmh_ce_enabled():
            return "", []
        store = self.pmh_store
        if store is None or not getattr(store, "open_bindings", None):
            return "", []
        asked = getattr(self, "_pmh_demand_asked", None)
        if asked is None:
            asked = self._pmh_demand_asked = set()
        texts: list[str] = []
        frames: list[int] = []
        for addr in sorted(store.open_bindings):
            if addr in asked:
                continue
            txt, fr = self._pmh_demand_evidence(addr, store.open_bindings[addr])
            if not txt:
                # No observation frames yet: stay silent rather than inject a question the
                # planner cannot act on (v0.13's failure mode). The demand stays queued and will
                # be offered again once an Open_*/Lift_* stage has written frames.
                continue
            asked.add(addr)
            texts.append(txt)
            for i in fr:
                if i not in frames:
                    frames.append(i)
            store.events.append(
                {
                    "op": "demand_tier1",
                    "addr": addr,
                    "why": store.open_bindings[addr].get("why"),
                    "frames": list(fr),
                    "t": int(step_idx),
                }
            )
            logger.info(
                "[pmh] demand_tier1 addr=%s why=%s frames=%s n_open=%s t=%s",
                addr,
                store.open_bindings[addr].get("why"),
                fr,
                len(store.open_bindings),
                step_idx,
            )
        if not texts:
            return "", []
        return "\n\n".join(texts), frames

    def _pmh_unaddressed_demand(self) -> tuple[str, dict[str, Any]] | None:
        """The oldest outstanding address demand, or None. THE read predicate (I5).

        Replaces `consecutive_same_subtask >= threshold` as the reason to read. The difference
        is that a stall is a symptom EVERY failure class shares - t22's was `01_Lift_Tomato_Sauce`,
        a physical grasp failure with no missing address, and the old predicate spent its most
        expensive call (`full_visual`) on it. A demand is a claim that a specific piece of
        knowledge is absent and required, which is falsifiable: `n_bindings_opened` counts it,
        and `n_bindings_resolved_by_{read,plan}` says whether anything closed it.

        Oldest-first rather than newest: the demand raised earliest is the one that has been
        blocking the longest, and FIFO also makes the choice independent of how long the current
        stage happens to run.
        """
        store = self.pmh_store
        if store is None or not getattr(store, "open_bindings", None):
            return None
        for addr in sorted(store.open_bindings, key=lambda a: int(store.open_bindings[a].get("last_t", 0) or 0)):
            # R2 negative cache: archive already failed this address — do not re-spend tier 2.
            if store.binding_is_exhausted(addr):
                continue
            return addr, store.open_bindings[addr]
        return None

    def _pmh_demand_evidence(self, addr: str, info: dict[str, Any]) -> tuple[str, list[int]]:
        """Inject a demand as a QUESTION with the planner's own observation pointers.

        Same shape as `_pmh_gap_evidence` (v0.16), which is the one injection in this project that
        measured non-negative: a question plus frame pointers, never an answer. The controller is
        allowed to know that a fact is missing; it is NOT allowed to supply it, because the
        t4 measurement (job 566614) showed that telling the planner where the butter goes - the
        ANSWER to the drawer question - collapses that task to a zero-variance 25.0 by letting
        the planner skip the scored search stages.
        """
        store = self.pmh_store
        if store is None:
            return "", []
        acts = {"open", "opening", "lift", "lifting", "pick", "picking", "inspect", "inspection"}
        frames: list[int] = []
        obs: list[str] = []
        for ref in store.stage_visuals:
            nm = re.sub(r"^\d+_", "", ref.stage_name or "")
            tk = [t for t in nm.split("_") if t]
            if not tk or tk[0].lower() not in acts:
                continue
            idxs = [int(i) for i in ref.frame_indices if int(i) in self.frame_store_main]
            if not idxs:
                continue
            obs.append(f"  - {ref.stage_name}: frames {idxs}")
            for i in idxs:
                if i not in frames:
                    frames.append(i)
        if not frames:
            return "", []
        keep = max(1, int(os.environ.get("PMH_GAP_EVIDENCE_K", "4")))
        frames = frames[-keep:]
        line = (
            "Unresolved binding (controller-verified: the ledger cannot answer this and the "
            "answer is NOT given on purpose):\n"
            f"  address: {addr}\n"
            f"  question: {info.get('question') or addr}\n"
            "  your own earlier observations that may answer it:\n"
            + "\n".join(obs[-6:])
            + "\n  Read the attached historical keyframes and resolve it, or state which "
            "container you are treating the address as referring to."
        )
        return line, frames

    def _pmh_stall_alt_directive(self, step_idx: int) -> str:
        """v0.14: when the planner keeps repeating the SAME subtask without progress
        (consecutive_same_subtask >= threshold), give it ONE bounded textual nudge to
        change approach instead of silently replaying the identical instruction.

        Evidence: retries/stalls that merely repeat the same failing plan never recover
        (measured duplicate-stall trajectories in v0.11); short single-attempt tasks
        (t16/t18/t22) collapse to 0% when a grasp stalls for the whole attempt. This is
        a deterministic, rate-limited, text-only nudge — no extra tool/API call, no new
        memory content injected (avoids the v0.10/v0.12/v0.13 failure mode).
        """
        if self.pmh_store is None:
            return ""
        on = os.environ.get("PMH_ALT_STALL", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if not on:
            return ""
        at_thr = max(2, int(os.environ.get("PMH_ALT_STALL_AT", "4")))
        cd_steps = max(1, int(os.environ.get("PMH_ALT_STALL_CD", "8")))
        if int(self._consecutive_same_subtask) < at_thr:
            return ""
        if (int(step_idx) - int(self._pmh_last_alt_stall)) < cd_steps:
            return ""
        self._pmh_last_alt_stall = int(step_idx)
        self._pmh_n_alt_inject += 1
        if self.logger:
            self.logger.info(
                "[pmh] alt_stall inject n=%s stall=%s step=%s",
                self._pmh_n_alt_inject,
                self._consecutive_same_subtask,
                step_idx,
            )
        return (
            "\nEXECUTION STALL NOTICE: you have issued the same subtask repeatedly without "
            "any progress. The current approach is failing. State a concrete hypothesis for "
            "why (e.g. unreachable grasp pose, wrong object/container interpretation, state "
            "changed) and issue a DIFFERENT next action — change grasp approach side/height, "
            "re-target the object, or verify the container. Do NOT repeat the identical "
            "subtask wording."
        )

    def _pmh_contract_hint(self, step_idx: int) -> str:
        """v0.15 CGM-OB: Contract-Gated Memory at Occlusion Boundaries.

        The access policy that 10 rounds of evidence converged on:
          * NOT "ask the VLM every round if evidence is sufficient" (degenerate: v0.6-0.11
            tools idle, v0.10 spam);
          * NOT "always inject extra memory" (v0.10/v0.12/v0.13/v0.14 all net-negative);
          * NOT "inject what the planner already knows" (v0.13's facts were redundant).
        Instead the *controller* computes a provable information gap and injects exactly one
        evidence-linked line at that moment:

            fires iff  a check-confirmed contract exists for object O
                   and the active stage's target object is O
                   and O is absent from what the consolidator recently reported visible
                   and this (stage, O) pair has not been hinted yet (cooldown)

        Zero extra API calls, zero pixels (cannot visually mislead as v0.12 did), and the
        payload is a *verified* fact with a frame pointer, so it cannot be stale guesswork.
        """
        store = self.pmh_store
        if store is None:
            return ""
        on = os.environ.get("PMH_CONTRACT_GATE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if not on:
            return ""
        stage = re.sub(r"^\d+_", "", (self._pmh_current_stage or "").strip())
        if not stage:
            return ""
        toks = [t for t in stage.split("_") if t]
        if len(toks) < 2:
            return ""
        action = toks[0].lower()
        if action not in {"place", "put", "placing", "pick", "picking", "lift", "open", "pour"}:
            return ""
        obj = toks[1].lower()
        if len(obj) < 3:
            return ""
        contract = store.contract_for(obj)
        if contract is None:
            return ""
        # Occlusion test: is the object still being reported as visible? Matching is
        # stem-based so "cookie"/"cookies" do not masquerade as different objects; any
        # recent mention counts as visible (conservative: we would rather stay silent
        # than inject a fact the planner can already see — v0.13's failure mode).
        visible = store.recent_objects(k=int(os.environ.get("PMH_CONTRACT_VISIBLE_K", "2")))
        obj_stem = obj.rstrip("s")
        if any((w == obj or w.rstrip("s") == obj_stem) for w in visible):
            return ""
        cd = max(1, int(os.environ.get("PMH_CONTRACT_CD", "6")))
        if (int(step_idx) - int(self._pmh_last_contract_hint)) < cd:
            return ""
        key = f"{stage}|{obj}"
        if key in self._pmh_contract_hinted:
            return ""
        self._pmh_contract_hinted.add(key)
        self._pmh_last_contract_hint = int(step_idx)
        store.n_contract_hints += 1
        store.last_contract_hint = int(step_idx)
        ptr = f", frame=f{contract.frame}" if int(contract.frame) >= 0 else ""
        line = (
            f"verified memory (check-confirmed, not currently visible): "
            f"{contract.obj} was placed in {contract.container} "
            f"at stage {contract.stage} (t={contract.step}{ptr}). "
            f"If the current stage needs it, it is in {contract.container}."
        )
        # Durable, greppable evidence of the gate firing (v0.14 lesson: controller
        # interventions MUST be observable in the module log and in the store events).
        logger.info(
            "[pmh] contract_hint n=%s stage=%s obj=%s container=%s visible_k=%s t=%s",
            store.n_contract_hints,
            stage,
            obj,
            contract.container,
            int(os.environ.get("PMH_CONTRACT_VISIBLE_K", "2")),
            step_idx,
        )
        store.events.append(
            {
                "op": "contract_hint",
                "stage": stage,
                "obj": obj,
                "container": contract.container,
                "contract_stage": contract.stage,
                "contract_step": contract.step,
                "frame": contract.frame,
                "t": int(step_idx),
            }
        )
        return line

    # ------------------- v0.16 Gap Gate (PMH_OPTIMAL_ARCH.md sec 4.2/4.4) -------------------
    def _gap_gate_enabled(self) -> bool:
        return os.environ.get("PMH_GAP_GATE", "0").strip().lower() not in {
            "0", "false", "no", "off",
        }

    @staticmethod
    def _stage_container(stage_name: str) -> tuple[str, str] | None:
        """Controller-private: parse a PLACEMENT stage name into (kind, container phrase).

        The controller is allowed to know the answer; the PLANNER is not. That asymmetry is
        the whole point of v1.0: the stage name may be used to COMPUTE A QUESTION here, but
        must never be injected into planner-visible text - injecting it is precisely the leak
        measured in job 566614 (t4 best 25.0 with the name injected vs 68.8 withheld, a
        zero-variance structural ceiling, because the planner skips the scored drawer-search
        stages 01-06 and jumps straight to 08 once it is told where the butter goes).

        Returns None for non-placement stages, so Open_*/Close_* stages never open the gate.
        """
        s = re.sub(r"^\d+_", "", (stage_name or "").strip())
        toks = [t for t in s.split("_") if t]
        if len(toks) < 3:
            return None
        if toks[0].lower() not in {"place", "put", "placing"}:
            return None
        container = " ".join(t.lower() for t in toks[2:]).strip()
        if not container:
            return None
        for kind in ("drawer", "microwave", "basket", "cabinet", "plate", "drainer", "mug"):
            if kind in container:
                return (kind, container)
        return (container.split()[-1], container)

    def _pmh_gap_state(self) -> tuple[str, dict[str, Any] | None]:
        """Controller-computed gap state: "open" | "closed" | "unknown".

        Three-valued on purpose (v1.1 fix). The original returned `None` for BOTH "no gap
        exists" and "cannot tell", which made the access gate fail CLOSED whenever the stage was
        unknown. The v1.0 runtime log shows this firing in production:

            [pmh] force tool=retrieve_visual reason=exec_stall
            [pmh] gap_gate BLOCKED tool=retrieve_visual stage= n=1 (no unresolved binding)

        `stage=` is empty, so `_stage_container` returned None, so the gate treated it as
        "provably nothing to remember" and refused the retrieval - including the anti-collapse
        forced retrieve whose entire purpose is to break a stall. Absence of evidence about a
        gap is NOT evidence of its absence, so for an ACCESS decision unknown must allow.

        Returns ("open", gap) with a QUESTION and never the answer. Two gap kinds share the gate:
          G1 - the container itself is unspecified: t4's task_block says "put butter into THAT
               non-empty drawer" and names no drawer, while its stage is 08_Put_Butter_Top_Drawer.
          G3 - the container IS named but the in-container position is given by reference:
               t20 "place chocolate at the location where the cookies were placed".
        Deliberately no per-task rules (PMH.md sec 11): the only inputs are the active stage's
        container and the planner-visible spec text.
        """
        if not self._gap_gate_enabled() or self.pmh_store is None:
            return ("unknown", None)
        parsed = self._stage_container(self._pmh_current_stage)
        if parsed is None:
            # Either a non-placement stage (strictly: no gap) or an unrecognised/empty stage
            # (cannot tell). Both are returned as "unknown" because only an explicit
            # containment test against the spec text may CLOSE the gate - anything else must
            # leave visual access alone. v1.0 collapsed these into "closed" and lost t4's
            # 87.5 -> 25.0 (best) purely from refusing retrieval during observation stages.
            return ("unknown", None)
        kind, container = parsed
        text = " ".join(
            str(getattr(self.task_info, attr, "") or "")
            for attr in ("task_block", "brief_description", "scene_description")
        ).lower()
        if container not in text:
            return ("open", {
                "kind": "G1",
                "container": container,
                "kind_name": kind,
                "question": f"which {kind} is the one the task refers to?",
            })
        if any(p in text for p in GAP_G3_LOCATION_PHRASES):
            return ("open", {
                "kind": "G3",
                "container": container,
                "kind_name": kind,
                "question": f"where inside the {kind} was the earlier object placed?",
            })
        return ("closed", None)

    def _pmh_gap(self) -> dict[str, Any] | None:
        """The gap payload, present ONLY when the controller has positively confirmed one.

        Injection stays conservative (no speculation on "unknown"), while access does not - the
        asymmetry is deliberate.
        """
        state, gap = self._pmh_gap_state()
        return gap if state == "open" else None

    def _pmh_gap_evidence(self, step_idx: int) -> tuple[str, list[int]]:
        """When the gate is OPEN: push the planner's OWN earlier observations as evidence.

        The payload is a question plus frame pointers to the Open_* observations - i.e. the
        frames in which drawer contents were visible. It is not an answer, and it is not a
        stage name: the planner has to read the frames. Fires at most once per
        (kind, container, stage) so it cannot become a per-step tax.
        Returns (text, frame_indices_to_attach).
        """
        gap = self._pmh_gap()
        if gap is None:
            return "", []
        store = self.pmh_store
        key = f"{gap['kind']}|{gap['container']}|{self._pmh_current_stage}"
        hinted = getattr(self, "_pmh_gap_hinted", None)
        if hinted is None:
            hinted = self._pmh_gap_hinted = set()
        if key in hinted:
            return "", []
        hinted.add(key)

        obs: list[str] = []
        frames: list[int] = []
        for ref in store.stage_visuals:
            s = re.sub(r"^\d+_", "", ref.stage_name or "")
            toks = [t for t in s.split("_") if t]
            if not toks or toks[0].lower() not in GAP_EVIDENCE_ACTIONS:
                continue
            idxs = [int(i) for i in ref.frame_indices if int(i) in self.frame_store_main]
            if not idxs:
                continue
            obs.append(f"  - {ref.stage_name}: frames {idxs}")
            for idx in idxs:
                if idx not in frames:
                    frames.append(idx)
        if not frames:
            # Nothing observed yet: stay silent rather than inject an unusable question
            # (v0.13's failure mode was injecting facts the planner could not act on).
            return "", []
        keep = max(1, int(os.environ.get("PMH_GAP_EVIDENCE_K", "4")))
        frames = frames[-keep:]
        n_open = int(getattr(store, "n_gap_open", 0)) + 1
        store.n_gap_open = n_open
        logger.info(
            "[pmh] gap_gate OPEN kind=%s container=%s stage=%s frames=%s n=%s t=%s",
            gap["kind"], gap["container"], self._pmh_current_stage, frames, n_open, step_idx,
        )
        store.events.append({
            "op": "gap_gate_open",
            "kind": gap["kind"],
            "container": gap["container"],
            "stage": self._pmh_current_stage,
            "frames": frames,
            "t": int(step_idx),
        })
        line = (
            "Unresolved binding (controller-verified information gap; the answer is NOT "
            "given on purpose):\n"
            f"  question: {gap['question']}\n"
            "  your own earlier observations that can answer it:\n"
            + "\n".join(obs[-6:])
            + "\n  The observations above are attached below as historical keyframes. Resolve "
            "the binding by reading them and then act."
        )
        return line, frames

    def _pmh_gap_blocks_visual(self) -> bool:
        """Whether the gate may REFUSE visual access. Now OFF by default (v1.1).

        v1.0 shipped this at "on", and job 567662/567743 killed the idea with two measurements:
          * t4: retrieval allowed (v0.15e) = 87.5 best vs blocked (v1.0) = 25.0 best. Same task,
            same seed, 62.5pp apart, and the v1.0 stall moved back to 03_Open_Middle_Drawer -
            i.e. it regressed to Harness's structural ceiling of 2/8 scored stages.
          * t22: blocked = 0.0 and unblocked (v0.15e, 158 retrievals) = 0.0. Blocking bought
            nothing, because t22 is a PHYSICAL grasp failure at 01_Lift_Tomato_Sauce, not an
            over-retrieval problem. The over-retrieval diagnosis recorded in
            PMH_OPTIMAL_ARCH.md was therefore wrong and is retracted.
        Net: blocking cost 62.5pp on the task where memory matters and fixed nothing. It also
        fail-CLOSED on an unknown stage (see `_pmh_gap_state`), refusing even the forced
        stall-breaking retrieve.
        Kept as an explicit switch so the v1.0 behaviour stays reproducible for the record;
        v1.1 arms leave PMH_GAP_BLOCK unset. Only visual tools were ever affected - text
        search_memory stays available per PMH.md sec 6.5.
        """
        on = os.environ.get("PMH_GAP_BLOCK", "0").strip().lower() not in {
            "0", "false", "no", "off",
        }
        if not on:
            return False
        state, _ = self._pmh_gap_state()
        return state == "closed"

    def _pmh_note_placement_from_stage(self, stage_name: str, step_idx: int = -1) -> None:
        """v0.13: on check-confirmed stage completion, distill the durable memory fact
        from the stage name itself (zero extra API call): 'Place/ Put <object> <container>'
        → '<object> placed in <container> (check-confirmed)'. This is the occluded-object
        answer memory-heavy tasks actually need, and it lives in Task State text where it
        cannot visually contradict the current observation (v0.12's pixel seating could).
        """
        store = self.pmh_store
        if store is None:
            return
        # v0.15: contract recording and the v0.13 free-text note block are separate
        # switches. The contract ledger is the machine-usable form (evidence-linked,
        # gate-readable); the raw text block is what v0.13 showed to be redundant.
        contract_on = os.environ.get("PMH_CONTRACT_LEDGER", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        notes_on = os.environ.get("PMH_PLACEMENT_NOTES", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if not contract_on and not notes_on:
            return
        s = re.sub(r"^\d+_", "", (stage_name or "").strip())
        toks = [t for t in s.split("_") if t]
        if len(toks) < 3:
            return
        action = toks[0].lower()
        if action not in {"place", "put", "placing"}:
            return
        # Observed RoboMemArena stage vocab: single-token object, then a possibly
        # multi-token container (Basket | Cabinet2 | Top_Drawer | Middle_Drawer ...).
        obj = toks[1].lower()
        container = "_".join(t.lower() for t in toks[2:]).strip("_")
        if len(obj) < 2 or len(container) < 2:
            return
        # v0.15: record an evidence-linked contract (stage id + step + frame pointer),
        # not just free text. The frame pointer makes the claim traceable to visual
        # evidence (PMH.md Sec 4.4) and is what the occlusion gate later echoes.
        frame = -1
        try:
            recent = [int(i) for i in self.pmh_store.active_frame_indices[-6:] if int(i) >= 0]
            if recent and recent[-1] in self.frame_store_main:
                frame = int(recent[-1])
        except Exception:
            frame = -1
        if contract_on and store.add_contract(
            obj=obj, container=container, stage=stage_name, step=int(step_idx), frame=frame
        ):
            logger.info(
                "[pmh] contract_add obj=%s container=%s stage=%s t=%s frame=%s",
                obj,
                container,
                stage_name,
                step_idx,
                frame,
            )
        note = f"{obj} placed in {container}"
        if notes_on and store.add_placement_note(stage_name, note):
            logger.info("[pmh] placement_note stage=%s note=%s", stage_name, note)

    def pmh_note_retry_brief(
        self,
        *,
        attempt_idx: int,
        stalled_stage: str | None,
        completed_stages: list[str] | None,
        repeated_stall: bool = False,
    ) -> None:
        """v0.10: after a Harness retry restore, tell the planner what the previous
        attempt achieved and where it stalled so plans focus repair on the first
        incomplete stage instead of restarting the whole task from scratch.
        v0.14: when the last TWO attempts stalled at the SAME stage with identical
        progress (measured duplicate-stall pattern), add a hard divergence warning so
        the retry does not replay the exact same failing plan."""
        if self.pmh_store is None:
            return
        comp = list(completed_stages or [])
        brief = f"harness_retry #{max(1, int(attempt_idx))}:"
        if comp:
            brief += f" previous attempt completed [{', '.join(str(c) for c in comp)}]"
        else:
            brief += " previous attempt completed no stages"
        if stalled_stage:
            brief += (
                f" then stalled at stage '{stalled_stage}'. "
                "Prioritize finishing this stage first; objects placed earlier may now be "
                "occluded — check Task State and use archive frames if needed."
            )
        else:
            brief += " then stalled mid-stage (no completed stage advanced)."
        if repeated_stall:
            brief += (
                " WARNING: the last two attempts stalled at the same stage with identical "
                "progress. Repeating the same plan has already failed once. Do NOT replay "
                "the previous subtask wording — hypothesize the failure cause (grasp pose, "
                "wrong object/container interpretation, changed state) and change the "
                "approach this attempt."
            )
        self.pmh_store.retry_brief = brief
        logger.info(
            "[pmh] retry_brief attempt=%s stalled=%s completed=%s repeated=%s",
            attempt_idx,
            stalled_stage,
            comp,
            int(bool(repeated_stall)),
        )

    def mark_salient_keyframe(self, step: int) -> None:
        if step >= 0 and step not in self.salient_keyframe_steps:
            self.salient_keyframe_steps.append(step)

    def close(self) -> None:
        self._dump_pmh_stats(final=True)
        if self._trace_fh is not None:
            self._trace_fh.close()
            self._trace_fh = None

    def _camera_order_text(self, use_wrist_images: bool) -> str:
        if use_wrist_images:
            return (
                "Camera order for each timestep: first image = main (agentview), "
                "second image = wrist (eye_in_hand)."
            )
        return "Camera order: one main (agentview) image per timestep."

    def _build_messages(
        self,
        memory_main_frames: list[Image.Image],
        memory_wrist_frames: list[Image.Image | None],
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Image.Image | None],
        *,
        extra_memory_text: str = "",
    ) -> list[dict[str, Any]]:
        use_wrist_images = self.use_wrist and any(
            frame is not None for frame in (memory_wrist_frames + context_wrist_frames)
        )
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Global objective: infer the robot's current primitive action from historical keyframes "
                    "before the current step and recent visual history within the same execution.\n\n"
                    "Task objective:\n"
                    f"{self.task_info.task_block}\n\n"
                    "Scene description:\n"
                    f"{self.task_info.scene_description or self.task_info.brief_description}\n\n"
                ),
            }
        ]
        merged_extra = "\n".join(
            x.strip() for x in (self.harness_extra_context, extra_memory_text) if x and x.strip()
        )
        # v0.15b ablation: strip answer-bearing stage/primitive phrases (task_block above is
        # deliberately NOT redacted - it is the legitimate task specification).
        merged_extra = self._redact_planner_text(merged_extra)
        if merged_extra:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Harness memory context (read-time evidence; use with historical keyframes):\n"
                        f"{merged_extra}\n"
                    ),
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"{self._camera_order_text(use_wrist_images)}\n"
                    "Current observation:"
                ),
            }
        )

        def append_timestep_images(main_frames, wrist_frames) -> None:
            for idx, main_img in enumerate(main_frames):
                user_content.append({"type": "image", "image": main_img})
                if use_wrist_images:
                    wrist_img = wrist_frames[idx] if idx < len(wrist_frames) else None
                    if wrist_img is not None:
                        user_content.append({"type": "image", "image": wrist_img})

        if memory_main_frames:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Historical keyframes from moments before the current step in the same execution "
                        f"({len(memory_main_frames)} timesteps):"
                    ),
                }
            )
            append_timestep_images(memory_main_frames, memory_wrist_frames)

        user_content.append(
            {
                "type": "text",
                "text": (
                    "Recent visual context: "
                    f"{len(context_main_frames)} consecutive frames ending at the current frame:"
                ),
            }
        )
        append_timestep_images(context_main_frames, context_wrist_frames)
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Output strict JSON with exactly two fields: current_primitive and keyframe_positions. "
                    "keyframe_positions are 1-indexed keyframe positions inside the recent visual window."
                ),
            }
        )
        # The nomination POLICY, which the line above never stated. Telling the model the field
        # exists is not the same as telling it when to fill it: a general VLM asked for a list it
        # has no policy for returns the empty list, which is what made the official bank builder
        # structurally dead here (`J_hist` all-empty, `build_visual_memory` -> [], job 586700).
        #
        # Gated on `MEM_KF_NOMINATION_PROMPT` and emitted as a SEPARATE block, so an arm that does
        # not declare it produces byte-identical content to before -- the baseline must not shift.
        # See `MemorySystemConfig.nomination_prompt` for why this is a substitute for PrediMem's
        # trained nomination behaviour rather than an equivalent of it.
        if bool(getattr(self.memory_system_config, "nomination_prompt", False)):
            user_content.append({"type": "text", "text": KF_NOMINATION_POLICY})
        return user_content

    def _cap_memory_frames(
        self,
        main_frames: list[Image.Image],
        wrist_frames: list[Image.Image | None],
        indices: list[int] | None = None,
    ) -> tuple[list[Image.Image], list[Image.Image | None], list[int]]:
        idxs = list(indices or range(len(main_frames)))
        if self.read_k_max <= 0 or len(main_frames) <= self.read_k_max:
            return main_frames, wrist_frames, idxs
        return (
            main_frames[-self.read_k_max :],
            wrist_frames[-self.read_k_max :],
            idxs[-self.read_k_max :],
        )

    def _merge_extra_visual(self, base_indices: list[int], extra_k: int) -> list[int]:
        if extra_k <= 0:
            return base_indices
        bank = list(self.K_indices_abs)
        out = list(base_indices)
        for idx in reversed(bank):
            if idx not in out:
                out.append(idx)
            if len(out) >= len(base_indices) + extra_k:
                break
        cap = len(base_indices) + extra_k
        if self.read_k_max > 0:
            cap = min(cap, self.read_k_max + extra_k)
            return out[-cap:]
        return out

    def _append_trace(self, record: dict[str, Any]) -> None:
        if self._trace_fh is None:
            return
        self._trace_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._trace_fh.flush()

    def _decide_memory_access(
        self,
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Image.Image | None],
    ) -> dict[str, Any]:
        cfg = self.proactive_cfg
        store = self.episodic_store
        default = {"tool": "none", "k": cfg.visual_k, "what": "all", "parsed_ok": False}
        if store is None:
            return default
        store.n_decide_calls += 1
        use_wrist = self.use_wrist and any(w is not None for w in context_wrist_frames)
        cur_main = context_main_frames[-1] if context_main_frames else None
        cur_wrist = context_wrist_frames[-1] if context_wrist_frames else None
        stage_hint = store.current_stage_name or "(unknown)"
        n_kf = len(store.keyframe_indices)
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Task objective:\n{self.task_info.task_block}\n\n"
                    f"Current stage hint: {self._redact_planner_text(stage_hint)}\n"
                    f"Available historical keyframes in store: {n_kf}\n"
                    f"Completed stage count: {sum(1 for e in store.stage_events if e.done)}\n"
                    f"Recent primitive count: {len(store.subtask_events)}\n\n"
                    f"{self._camera_order_text(use_wrist)}\n"
                    "Current observation (latest frame only):"
                ),
            }
        ]
        if cur_main is not None:
            user_content.append({"type": "image", "image": cur_main})
            if use_wrist and cur_wrist is not None:
                user_content.append({"type": "image", "image": cur_wrist})
        user_content.append(
            {
                "type": "text",
                "text": (
                    'Output strict JSON only, e.g. '
                    + (
                        '{"tool":"recall_visual","k":4} or {"tool":"none"}.'
                        if not cfg.allow_semantic
                        else '{"tool":"recall_visual","k":4} or {"tool":"recall_semantic","what":"all"} '
                        'or {"tool":"none"}.'
                    )
                ),
            }
        )
        out_text = infer_primitive_via_api(
            system_prompt=memory_decision_system_prompt(),
            user_content=user_content,
            api_key=self.api_key,
            base_url=self.api_base_url,
            model=self.api_model,
            timeout_sec=min(60.0, self.api_timeout),
            max_tokens=cfg.decide_max_new_tokens,
        ) or ""
        decision = parse_memory_decision(
            out_text,
            default_k=cfg.visual_k,
            parse_fail_tool=cfg.parse_fail_tool,
        )
        if not cfg.allow_semantic and str(decision.get("tool", "none")) == "recall_semantic":
            decision = {
                **decision,
                "tool": "recall_visual",
                "k": int(decision.get("k", cfg.visual_k)),
                "remapped_from": "recall_semantic",
            }
        decision = apply_anti_collapse(decision, store, cfg)
        store.last_tool = str(decision.get("tool", "none"))
        store.events.append({"op": "decide", "raw": out_text.strip()[:200], **decision})
        if self.logger:
            self.logger.info(
                "[proactive-api] decide tool=%s k=%s forced=%s raw=%s",
                decision.get("tool"),
                decision.get("k"),
                decision.get("forced", False),
                out_text.strip()[:120],
            )
        return decision

    def _decide_pmh_access(
        self,
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Image.Image | None],
        *,
        evidence_so_far: str = "",
        bank_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        store = self.pmh_store
        default = {
            "tool": "none",
            "query": "",
            "segment_id": "last",
            "detail": "keyframes",
            "parsed_ok": False,
        }
        if store is None:
            return default
        store.n_decide += 1
        use_wrist = self.use_wrist and any(w is not None for w in context_wrist_frames)
        recent_n = max(1, min(self.pmh_decide_recent, len(context_main_frames)))
        recent_mains = context_main_frames[-recent_n:]
        recent_wrists = context_wrist_frames[-recent_n:] if context_wrist_frames else []
        # v0.10 grounded decide: show the on-context bank indices (and, when enabled,
        # the bank images) so "none vs retrieve" is answered against what the planner
        # already sees, not blind. Retrieve should only add evidence NOT in this bank.
        #
        # v3.0 F5: `PMH_ASK_MODE` FORCES this on. Spec Sec 5.2 Step 0 asks the agent to decide
        # whether "current information" suffices - and the on-context bank IS part of the current
        # information, because the plan step injects it. A decide step that cannot see the bank
        # is being asked a question about a set it has not been shown, so its only strategy is to
        # guess, and the predictable guess is to re-request what it already holds. That is the
        # measured defect behind `n_new=0` on 23 of 33 retrievals in job 568062 (83% of t22's):
        # the retrieval round ran, the refusal logic never fired, and the round changed nothing.
        # Coupling rather than configuring is deliberate: an ask-mode whose sufficiency judgement
        # is blind is not a weaker version of this mechanism, it is a different and broken one.
        show_bank = _truthy_env("PMH_DECIDE_SHOW_BANK") or _truthy_env("PMH_ASK_MODE")
        bank_k = max(0, int(os.environ.get("PMH_DECIDE_BANK_K", "6")))
        bank_idx: list[int] = []
        if show_bank and bank_indices:
            bank_idx = [int(i) for i in bank_indices if int(i) in self.frame_store_main]
            if bank_k > 0:
                bank_idx = bank_idx[-bank_k:]
        bank_text = (
            f"On-context visual bank indices: {list(bank_idx)}\n"
            "(Do NOT retrieve_visual for frames you already have; retrieve only archive "
            "evidence absent from the bank.)\n"
            if bank_idx
            else ""
        )
        # ---- v2.0 L0: tell the planner the truth about (a) its retrieval right and (b) what
        # "stall" actually measures. The old line fed the planner "Stall count (same subtask)",
        # which reads as "memory is not helping, ask again" and is a direct cause of the
        # retrieve-every-step loop: a long manipulation stage keeps one subtask name, so the
        # count climbs while the arm is making perfectly good progress.
        budget_text = ""
        cap = int(os.environ.get("PMH_RETRIEVE_PER_STAGE", "0") or 0)
        allowed_now, refuse_why = self._pmh_retrieve_right()
        if cap > 0 or int(os.environ.get("PMH_NONPRODUCTIVE_MAX", "0") or 0) > 0:
            if allowed_now:
                remaining = (cap - self._pmh_retrievals_in_stage) if cap > 0 else -1
                budget_text = (
                    f"Visual evidence budget for stage '{self._pmh_current_stage or '?'}': "
                    + (f"{remaining} retrieval(s) left. " if remaining >= 0 else "")
                    + "Spend one ONLY on evidence absent from the bank; if the bank already "
                    "answers the open question, answer 'none'.\n"
                )
            else:
                budget_text = (
                    f"Visual evidence budget for stage '{self._pmh_current_stage or '?'}': "
                    "EXHAUSTED - retrieve_visual is disabled for this stage. Answer 'none' and "
                    "proceed with the action.\n"
                )
        stall_text = ""
        if _pmh_stall_mode() == "delta":
            # State the quantity that actually matters: whether evidence/state changed.
            stall_text = (
                f"Steps without a new subtask or new evidence: {self._pmh_nonproductive_streak}"
                f" (a stable subtask during manipulation is normal progress, NOT a stall)\n"
            )
        else:
            stall_text = f"Stall count (same subtask): {self._consecutive_same_subtask}\n"
        graph_text = ""
        if os.environ.get("PMH_GRAPH", "0").strip().lower() not in {"0", "false", "no", "off"}:
            graph_text = str(self._pmh_graph_last_meta.get("render") or "")
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Task objective:\n{self.task_info.task_block}\n\n"
                    f"{store.render_task_state()}\n"
                    f"{bank_text}"
                    f"Archived segments: {len(store.segments)}\n"
                    f"{stall_text}"
                    f"{budget_text}"
                    f"{graph_text}"
                    f"{evidence_so_far}\n"
                    f"{self._camera_order_text(use_wrist)}\n"
                    f"Short-term observations ({len(recent_mains)} recent frames):"
                ),
            }
        ]
        bank_mains = get_frames_from_indices(bank_idx, self.frame_store_main)
        bank_wrists = [self.frame_store_wrist.get(idx) for idx in bank_idx]
        for i, main_img in enumerate(bank_mains):
            user_content.append(
                {
                    "type": "text",
                    "text": f"bank frame idx={bank_idx[i]} (main):",
                }
            )
            user_content.append({"type": "image", "image": main_img})
            if use_wrist and i < len(bank_wrists) and bank_wrists[i] is not None:
                user_content.append({"type": "image", "image": bank_wrists[i]})
        for i, main_img in enumerate(recent_mains):
            user_content.append({"type": "image", "image": main_img})
            if use_wrist and i < len(recent_wrists) and recent_wrists[i] is not None:
                user_content.append({"type": "image", "image": recent_wrists[i]})
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Output strict JSON only, e.g. "
                    '{"tool":"none"} or {"tool":"search_memory","query":"..."} or '
                    '{"tool":"inspect_segment","segment_id":"seg_000","detail":"keyframes"}.'
                ),
            }
        )
        out_text = infer_primitive_via_api(
            system_prompt=pmh_decision_system_prompt(),
            user_content=user_content,
            api_key=self.api_key,
            base_url=self.api_base_url,
            model=self.api_model,
            timeout_sec=min(float(os.environ.get("PMH_DECIDE_TIMEOUT", "180")), self.api_timeout),
            # THE DECIDE BUDGET MUST SCALE WITH THE PLANNER (measured 2026-09-15, job 585835).
            #
            # This was hardcoded to 96, calibrated on qwen-vl-max, which emits the JSON directly.
            # A REASONING planner (gemini-3.8-flash via the CloseAI relay) writes its chain of
            # thought INSIDE max_tokens first, so 96 bytes the answer off mid-token:
            #
            #     585835, all 68 decide calls:  parsed=False   raw=```json\n{"tool":"none
            #
            # The failure is SILENT and it INVERTS THE MECHANISM UNDER TEST. A truncated decision
            # parses to the default `{"tool":"none"}`, so the arm stops evaluating the agent's
            # tool CHOICE at all: every retrieval is either suppressed (55/68, forced=False) or
            # taken by anti-collapse's stall path (13/68, forced=True). We had read those forced
            # counts -- and written it into the model as the autonomy-migration law -- as "the
            # agent never generates its own queries, alpha(free) ~ 0". It was never the agent:
            # it was a token ceiling. The planner was never allowed to choose anything.
            #
            # Fall back to PLANNER_API_MAX_TOKENS so the two budgets cannot diverge again -- that
            # divergence IS the bug class (the same truncation already cost us the recovery rung
            # via HARNESS_API_MAX_TOKENS, and later the manifest parser via strip_wrapper).
            max_tokens=int(
                os.environ.get("PMH_DECIDE_TOKENS")
                or os.environ.get("PLANNER_API_MAX_TOKENS")
                or "1024"
            ),
        ) or ""
        decision = parse_pmh_decision(out_text)
        exec_stall = bool(getattr(self, "_pmh_exec_stall", False))
        # A PARSE FAILURE IS NOT A DECISION (job 585835).
        #
        # `parse_pmh_decision` maps an unparseable response onto the same object it returns for a
        # deliberate `{"tool":"none"}`, and nothing downstream could tell the two apart: no
        # counter existed, `store.consecutive_none` was incremented either way, and
        # `apply_pmh_anti_collapse` read a transport fault as "the agent has gone silent" and
        # forced a retrieval on the stall path. So when the token ceiling truncated all 68 calls
        # in 585835, the arm did not merely lose the reads it should have made -- it MANUFACTURED
        # retrievals that no agent ever asked for, and reported them as the mechanism working.
        # We then wrote that artefact into the model as the autonomy law alpha(free) ~ 0.
        #
        # The old ProactiveMemory path had this right (`parse_fail_tool` in proactive_memory.py);
        # the PMH decide path dropped the distinction. Restore it at the one place where the
        # decision is born, so every consumer sees an explicit fault rather than a silent "none":
        #   * counted, so an invalidating run is visible from the census instead of inferred;
        #   * NOT counted as silence: `consecutive_none` feeds the stall-force, and a corrupted
        #     answer is not evidence about the agent's intent. Anti-collapse may still rescue via
        #     its exec_stall branch, which keys on the controller's evidence, not on this counter.
        parsed_ok = bool(decision.get("parsed_ok", False))
        if not parsed_ok:
            self._pmh_decide_parse_failures = int(
                getattr(self, "_pmh_decide_parse_failures", 0)
            ) + 1
            decision["parse_failure"] = True
            logger.warning(
                "[pmh] decide PARSE-FAILURE n=%s budget=%s raw=%r -- this call carries NO "
                "evidence about the agent's tool choice; not counted as silence",
                self._pmh_decide_parse_failures,
                os.environ.get("PMH_DECIDE_TOKENS")
                or os.environ.get("PLANNER_API_MAX_TOKENS")
                or "1024",
                out_text.strip()[:160],
            )
        decision = apply_pmh_anti_collapse(
            decision,
            store,
            stall_count=self._consecutive_same_subtask,
            task_id=int(getattr(self.task_info, "task_id", 0) or 0),
            already_has_archive_visual=bool(
                getattr(self, "_pmh_has_archive_visual", False)
            ),
            exec_stall=exec_stall,
        )
        if exec_stall:
            self._pmh_exec_stall = False

        # v0.16 GAP GATE (PMH_OPTIMAL_ARCH.md sec 4.2/4.4): when the controller proves there
        # is no unresolved binding, refuse VISUAL access. Checked AFTER anti-collapse on
        # purpose, so the stall/exec_stall forcing rules are also gated - they were the direct
        # cause of the v0.15e t22 regression (8/8 attempts stalled on a physical grasp failure
        # at 01_Lift_Tomato_Sauce while PMH still forced ~53 visual retrievals per episode).
        blocked_tool = ""
        if self._pmh_gap_blocks_visual() and str(decision.get("tool", "none")) in {
            "retrieve_visual", "inspect_segment",
        }:
            blocked_tool = str(decision.get("tool"))
            store.n_gap_blocked = int(getattr(store, "n_gap_blocked", 0)) + 1
            store.events.append({
                "op": "gap_gate_block",
                "tool": blocked_tool,
                "stage": self._pmh_current_stage,
            })
            logger.info(
                "[pmh] gap_gate BLOCKED tool=%s stage=%s n=%s (no unresolved binding)",
                blocked_tool, self._pmh_current_stage, store.n_gap_blocked,
            )
            decision = dict(default)
            decision["blocked_by_gap_gate"] = True

        store.last_tool = str(decision.get("tool", "none"))
        store.events.append({"op": "pmh_decide", "raw": out_text.strip()[:200], **decision})
        # Always log via module logger so Slurm out keeps process evidence after cleanup.
        logger.info(
            "[pmh] decide tool=%s forced=%s parsed=%s segs=%s stall=%s raw=%s",
            decision.get("tool"),
            bool(decision.get("forced", False)),
            decision.get("parsed_ok"),
            len(store.segments),
            self._consecutive_same_subtask,
            out_text.strip()[:120],
        )
        if self.logger:
            self.logger.info(
                "[pmh] decide tool=%s forced=%s parsed=%s raw=%s",
                decision.get("tool"),
                bool(decision.get("forced", False)),
                decision.get("parsed_ok"),
                out_text.strip()[:120],
            )
        return decision

    def _pmh_short_term_visual(
        self,
        *,
        recent_start: int,
        exclude: set[int] | None = None,
    ) -> tuple[list[Image.Image], list[Image.Image | None], list[int]]:
        """Active-segment representative frames for Short-Term (not archived KF pack)."""
        store = self.pmh_store
        if store is None:
            return [], [], []
        excl = exclude or set()
        idxs = [
            i
            for i in store.active_visual_indices(self.pmh_active_visual_k, before=recent_start)
            if i not in excl and i in self.frame_store_main
        ]
        if not idxs:
            return [], [], []
        mains = get_frames_from_indices(idxs, self.frame_store_main)
        wrists = [self.frame_store_wrist.get(idx) for idx in idxs]
        return mains, wrists, idxs

    def _pmh_archived_visual_floor(
        self,
        *,
        exclude: set[int] | None = None,
    ) -> tuple[list[Image.Image], list[Image.Image | None], list[int]]:
        """Small always-on archived keyframe floor (PrediMem/KEMO-style sparse evidence)."""
        store = self.pmh_store
        if store is None:
            return [], [], []
        excl = exclude or set()
        floor_k = int(os.environ.get("PMH_ARCHIVED_VISUAL_K", "2"))
        idxs = [
            i
            for i in store.archived_visual_floor_indices(floor_k)
            if i not in excl and i in self.frame_store_main
        ]
        if not idxs:
            return [], [], []
        mains = get_frames_from_indices(idxs, self.frame_store_main)
        wrists = [self.frame_store_wrist.get(idx) for idx in idxs]
        return mains, wrists, idxs

    def _pmh_snap_to_kf(self, cands: list[int], max_k: int = 2) -> list[int]:
        """Map candidate frames onto dense KF spine (v0.8); fall back to reps if KF empty."""
        max_k = max(1, int(max_k))
        kf = [
            int(i)
            for i in self.K_indices_abs
            if int(i) in self.frame_store_main and int(i) >= 0
        ]
        raw = [int(i) for i in cands if int(i) in self.frame_store_main and int(i) >= 0]
        if not kf:
            return pick_representative_indices(raw, max_k=max_k)
        in_k = [i for i in raw if i in kf]
        if in_k:
            return pick_representative_indices(in_k, max_k=max_k)
        snapped: list[int] = []
        for c in raw or ([kf[-1]] if kf else []):
            nearest = min(kf, key=lambda k: abs(k - c))
            if nearest not in snapped:
                snapped.append(nearest)
            if len(snapped) >= max_k:
                break
        return snapped[:max_k]

    def _ensure_pmh_evidence_floor(self, recent_start: int) -> None:
        """v0.9: guarantee a dense archived floor when the J-based KF bank is empty
        (the v0.8 failure mode). Seed from committed PMH evidence so the bank is a
        real, passive, evenly-sampled past floor (harness-like dense pack) instead
        of silently degrading to per-step soft commits / an empty spine."""
        store = self.pmh_store
        if store is None or not self.use_keyframe_memory:
            return
        kf_spine = os.environ.get("PMH_KF_SPINE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        seed_on = os.environ.get("PMH_SEED_KF_FROM_EVIDENCE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if not (kf_spine and seed_on):
            return
        present = [
            int(i)
            for i in self.K_indices_abs
            if int(i) in self.frame_store_main and int(i) >= 0
        ]
        min_k = max(1, int(os.environ.get("PMH_KF_MIN_FOR_SPINE", "3")))
        if len(present) >= min_k:
            return
        cands: set[int] = set()
        for seg in store.segments:
            cands.update(int(i) for i in (seg.keyframe_indices or seg.visual_indices or []))
        for ref in store.stage_visuals:
            cands.update(int(i) for i in (ref.frame_indices or []))
        cands = sorted(
            i
            for i in cands
            if i in self.frame_store_main and (recent_start is None or i < int(recent_start))
        )
        if not cands:
            return
        cap = int(self.k_max) if (self.k_max or 0) > 0 else int(os.environ.get("K_MAX", "8"))
        cap = max(1, cap)
        floor = pick_representative_indices(cands, max_k=min(cap, len(cands)))
        merged = sorted(set(int(i) for i in list(self.K_indices_abs) + floor))
        self.K_indices_abs = merged[-cap:]
        self.K_main_frames = get_frames_from_indices(self.K_indices_abs, self.frame_store_main)
        self.K_wrist_frames = [self.frame_store_wrist.get(idx) for idx in self.K_indices_abs]
        logger.info(
            "[pmh] seeded evidence floor from committed segments: kf_n=%s cands=%s recent_start=%s",
            len(self.K_indices_abs),
            len(cands),
            recent_start,
        )

    def _pmh_novel_archive_visual(self, exclude: set[int], max_k: int = 2) -> list[int]:
        """v0.10: pick archived frames NOT already on-context (newest segments first, then
        stage visuals). Guarantees a retrieve round adds genuinely new pixels, not a copy
        of the passive bank."""
        store = self.pmh_store
        if store is None:
            return []
        max_k = max(1, int(max_k))
        out: list[int] = []
        excl = set(int(i) for i in (exclude or set()))
        for seg in reversed(store.segments):
            cands = list(seg.keyframe_indices or []) or list(seg.visual_indices or [])[:4]
            for i in cands:
                ii = int(i)
                if ii not in excl and ii in self.frame_store_main and ii not in out:
                    out.append(ii)
                if len(out) >= max_k:
                    return out
        for ref in reversed(store.stage_visuals):
            for i in ref.frame_indices:
                ii = int(i)
                if ii not in excl and ii in self.frame_store_main and ii not in out:
                    out.append(ii)
                if len(out) >= max_k:
                    return out
        return out

    def _pmh_relevant_bank_frames(
        self, kf_set: set[int], exclude: set[int], max_k: int = 2
    ) -> list[int]:
        """v0.12: seat decision-critical *archived* frames into the always-on bank.

        The dense KF pack is a temporal median sample with no notion of object relevance,
        and because archived frames were snapped to KF indices they get evicted from the
        pack as the episode advances (v0.11 bug: the selection required the archived frame
        to still be IN the current KF pack, so nothing was ever selected). Here we search
        the *persistent pixel store* (frame_store_main keeps every frame of the episode;
        only the KF *pack* is evicted), pick frames whose archived cards/stage names share
        object words with the current decision text, and return them for seating ahead of
        the low-relevance tail of the pack.

        Frames already in the current KF pack are excluded (they are seated anyway through
        the pack); this adds exactly the frames the pack has *forgotten*.

        Costs no extra API call; pixels are archived episode frames.
        """
        store = self.pmh_store
        if store is None:
            return []
        blob = " ".join(
            [
                str(self._current_subtask or ""),
                str(store.current_stage_name or ""),
                str(store.active_subtask or ""),
            ]
        ).lower()
        words = {
            w for w in re.split(r"[^a-z0-9_]+", blob) if len(w) >= 3 and w not in _PMH_STOPWORDS
        }
        if not words:
            return []
        kf = set(int(i) for i in kf_set)
        excl = set(int(i) for i in (exclude or set()))
        scored: list[tuple[int, int, int]] = []
        # Candidate archives: newest first by construction below (reverse iteration).
        for idx, seg in enumerate(store.segments):
            seg_txt = " ".join(
                [seg.caption or "", seg.action or "", seg.outcome or "", " ".join(seg.objects or [])]
            ).lower()
            hit = sum(1 for w in words if w in seg_txt)
            if hit <= 0:
                continue
            for i in list(seg.keyframe_indices or []) or list(seg.visual_indices or [])[:4]:
                ii = int(i)
                if (
                    ii >= 0
                    and ii not in kf
                    and ii not in excl
                    and ii in self.frame_store_main
                ):
                    # (hit, segment recency desc, frame index) — prefer more overlap, then
                    # the most recently archived matching segment.
                    scored.append((hit, -idx, ii))
        for seg_i, ref in enumerate(store.stage_visuals):
            for i in ref.frame_indices:
                ii = int(i)
                if (
                    ii >= 0
                    and ii not in kf
                    and ii not in excl
                    and ii in self.frame_store_main
                ):
                    # Match the stage name text too (e.g. 'Place_Cookies_Basket').
                    hit = sum(
                        1
                        for w in words
                        if w in (ref.stage_name or "").lower().replace("_", " ")
                    )
                    if hit > 0:
                        scored.append((hit, -seg_i, ii))
        self._pmh_rel_cands = len(scored)
        out: list[int] = []
        for _hit, _rec, ii in sorted(scored, key=lambda x: (-x[0], x[1], -x[2])):
            if ii not in out:
                out.append(ii)
            if len(out) >= max_k:
                break
        return out

    def _pmh_graph_pack(self, kf: list[int], meta: dict[str, int]) -> list[int] | None:
        """v2.0 L1-L5: replace the recency-ordered pack with a gap-conditioned evidence forest.

        Returns None (leaving the legacy pack untouched) whenever the graph cannot produce a
        usable selection. That direction matters: this must never be able to empty the visual
        context. Every experiment in this project that removed visual access went badly - the
        v1.0 gap gate refused retrieval and cost t4 50pp - so a failure here has to be inert.

        The node pool is every view the store already tracks (dense KF pack, stage anchors,
        short-term window, archived segments), which is what makes complementary evidence
        reachable: the legacy pack could only ever show the newest stages, so a frame proving
        what is inside a drawer opened earlier had no route into the prompt at all.
        """
        store = self.pmh_store
        if store is None or not kf:
            return None
        try:
            from harness import pmh_graph as pg
        except Exception:
            try:
                import pmh_graph as pg
            except Exception:
                logger.info("[pmh] graph pack unavailable: pmh_graph import failed")
                return None

        nodes: dict[int, pg.Node] = {}

        def _node(idx: int) -> pg.Node | None:
            ii = int(idx)
            if ii < 0 or ii not in self.frame_store_main:
                return None
            n = nodes.get(ii)
            if n is None:
                n = pg.Node(idx=ii, t=ii)
                nodes[ii] = n
            return n

        def _facets(n: pg.Node) -> None:
            """Fill objects/containers from the stage name when they are not already known.

            `stage_facets` is the ONLY reliable source here: `SegmentCard.stage_tag` is never
            assigned anywhere in the repo and `SegmentCard.objects` is usually empty, so without
            this the nodes reach `build_edges` with no stage, no objects and no containers, all
            four schema relations in `_schema_adjacent` are dead, and the only live signal is
            the segment id - which is unique per segment. That is why the live run produced a
            forest with zero edges.
            """
            if not n.stage:
                return
            objs, conts = pg.stage_facets(n.stage)
            if not n.objects and objs:
                n.objects = objs
            if not n.containers and conts:
                n.containers = conts

        for i in kf:
            n = _node(i)
            if n is not None:
                n.is_kf = True
        for ref in store.stage_visuals:
            for i in ref.frame_indices:
                n = _node(i)
                if n is not None:
                    n.is_stage_visual = True
                    if not n.stage:
                        n.stage = str(ref.stage_name or "")
                    _facets(n)
        for seg in store.segments:
            seg_stage = str(getattr(seg, "stage_tag", "") or "")
            for i in list(seg.keyframe_indices or []) + list(seg.visual_indices or [])[:2]:
                n = _node(i)
                if n is not None:
                    if not n.stage and seg_stage:
                        n.stage = seg_stage
                    n.segment_id = str(seg.segment_id or "")
                    # Merge rather than overwrite: the VLM-reported objects and the ones implied
                    # by the stage name are both true and both useful for linking frames.
                    if seg.objects:
                        n.objects = tuple(dict.fromkeys(list(n.objects) + list(seg.objects)))
                    n.text = f"{seg.caption} {seg.action} {seg.outcome}".strip()
                    _facets(n)
        for i in store.active_frame_indices:
            n = _node(i)
            if n is not None:
                n.is_active = True
        if not nodes:
            return None

        # The gap is the QUERY memory must answer. It is carried as a question, never an answer.
        gap = None
        try:
            parsed = self._pmh_gap_state()
            if parsed[0] == "open" and parsed[1]:
                g = parsed[1]
                container = str(g.get("container") or "")
                cands: list[str] = []
                kind_word = container.split("_")[-1] if container else ""
                if kind_word:
                    # The stage-spec vocabulary that lists every container of this task lives in
                    # the CONTROLLER, not in this process, so the planner cannot enumerate
                    # sibling containers from here. What it can do - and does - is expand the
                    # position qualifier, which is precisely the G1 case: "which drawer is the
                    # non-empty one" is a question about top/middle/bottom of one cabinet. No
                    # `getattr` on an attribute that is never assigned: a dangling read would
                    # silently return [] and look like it worked.
                    cands = pg.enumerate_container_candidates(container, [])
                gap = pg.Gap(
                    gap_id=f"{g.get('kind')}:{container}",
                    kind=str(g.get("kind") or "G1"),
                    question=str(g.get("question") or ""),
                    candidates=cands,
                    containers=[container.replace("_", " ")] if container else [],
                )
        except Exception:
            gap = None

        cfg = pg.GraphConfig.from_env()
        cfg.enabled = True
        # Pass the live frame store so `build_edges` can compare actual pixels when it matters.
        # Without it every node carries no descriptor, no pair can be shown to be a duplicate,
        # and "redundant" can never be claimed - the pack would be rebuilt blind.
        forest = pg.select_evidence_forest(
            list(nodes.values()), gap=gap, cfg=cfg, main_store=self.frame_store_main
        )
        picked = pg.forest_indices(forest)
        if not picked:
            return None

        # Keep the pack's size policy identical to the legacy one (same max_k), so this changes
        # WHICH frames are shown and never how many.
        max_k = max(1, int(os.environ.get("PMH_VISUAL_BANK_K", os.environ.get("K_MAX", "8"))))
        out = picked[:max_k]
        # Backfill from the legacy order so a long stage cannot leave the pack under-filled.
        for i in kf:
            if len(out) >= max_k:
                break
            if int(i) not in out:
                out.append(int(i))
        meta["graph_n"] = len(picked)
        meta["graph_pool"] = forest.meta.get("pool", 0)
        meta["graph_components"] = forest.meta.get("components", 0)
        meta["graph_budget"] = forest.meta.get("budget", 0)
        meta["graph_dup_pruned"] = forest.pruned_redundant
        self._pmh_graph_forest = forest
        self._pmh_graph_last_meta = {
            "render": pg.render_forest(forest, gap=gap),
            "pool": forest.meta.get("pool"),
            "components": forest.meta.get("components"),
            "budget": forest.meta.get("budget"),
            "gap": gap.kind if gap else None,
        }
        logger.info(
            "[pmh] graph_pack pool=%s picked=%s kept=%s components=%s budget=%s gap=%s dup_pruned=%s",
            forest.meta.get("pool"),
            len(picked),
            len(out),
            forest.meta.get("components"),
            forest.meta.get("budget"),
            gap.kind if gap else None,
            forest.pruned_redundant,
        )
        return out

    def _pmh_kf_spine_bank(
        self,
    ) -> tuple[list[Image.Image], list[Image.Image | None], list[int], dict[str, int]]:
        """
        v0.8: default Evidence visuals = dense KF pack (same Write as harness_v21),
        optionally re-ordered by stage/event pointers into KF.
        v0.11: also promote KF frames relevant to the current subtask's objects
        (PMH_RELEVANCE_RERANK=1), so the always-on pack serves the in-progress decision.
        """
        max_k = int(os.environ.get("PMH_VISUAL_BANK_K", os.environ.get("K_MAX", "8")))
        max_k = max(1, max_k)
        kf = [
            int(i)
            for i in self.K_indices_abs
            if int(i) in self.frame_store_main and int(i) >= 0
        ]
        meta = {"kf_n": len(kf), "preferred_n": 0, "from_kf": 0, "relevant_n": 0, "rel_cands": 0}
        if not kf:
            return [], [], [], meta
        preferred: list[int] = []
        store = self.pmh_store
        if store is not None:
            for ref in reversed(store.stage_visuals[-12:]):
                for i in ref.frame_indices:
                    ii = int(i)
                    if ii in kf and ii not in preferred:
                        preferred.append(ii)
        meta["preferred_n"] = len(preferred)
        relevant: list[int] = []
        rerank = os.environ.get("PMH_RELEVANCE_RERANK", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if rerank:
            relevant = self._pmh_relevant_bank_frames(
                kf_set=set(kf),
                exclude=set(preferred),
                max_k=int(os.environ.get("PMH_RELEVANT_K", "2")),
            )
        meta["relevant_n"] = len(relevant)
        meta["rel_cands"] = int(getattr(self, "_pmh_rel_cands", 0))
        out: list[int] = []
        for i in preferred + relevant + kf:
            if i not in out:
                out.append(i)
            if len(out) >= max_k:
                break
        # v2.0 L1-L5: if the evidence forest is enabled, it decides WHICH frames are shown.
        # The legacy order above stays as the fallback and as the backfill source, so the pack
        # can never come back empty or smaller than before.
        if _pmh_graph_enabled():
            try:
                graph_out = self._pmh_graph_pack(kf, meta)
                if graph_out:
                    out = graph_out[:max_k]
            except Exception as exc:  # never let the selector break a run
                logger.error("[pmh] graph_pack failed (%s: %s); falling back to legacy pack",
                             type(exc).__name__, exc)
        if relevant:
            logger.info(
                "[pmh] relevant_seat cands=%s seated=%s subtask=%r",
                meta["rel_cands"],
                relevant,
                str(self._current_subtask or "")[:60],
            )
        meta["from_kf"] = sum(1 for i in out if i in kf)
        mains = get_frames_from_indices(out, self.frame_store_main)
        wrists = [self.frame_store_wrist.get(idx) for idx in out]
        return mains, wrists, out, meta

    def _pmh_visual_bank(
        self,
        *,
        recent_start: int,
    ) -> tuple[list[Image.Image], list[Image.Image | None], list[int]]:
        """Stage-indexed visual bank; v0.8 KF-spine uses dense KF as pixel source."""
        kf_spine = os.environ.get("PMH_KF_SPINE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if kf_spine and self.use_keyframe_memory:
            self._ensure_pmh_evidence_floor(recent_start)
            mains, wrists, idxs, meta = self._pmh_kf_spine_bank()
            if idxs:
                logger.info(
                    "[pmh] kf_spine_bank mem_k=%s kf_n=%s preferred=%s relevant=%s cands=%s from_kf=%s",
                    len(idxs),
                    meta.get("kf_n"),
                    meta.get("preferred_n"),
                    meta.get("relevant_n"),
                    meta.get("rel_cands"),
                    meta.get("from_kf"),
                )
                return mains, wrists, idxs
            # Cold start before first KF: fall through to legacy assembler.
        store = self.pmh_store
        if store is None:
            return [], [], []
        idxs = [
            i
            for i in store.assemble_visual_bank_indices(before=recent_start)
            if i in self.frame_store_main
        ]
        if not idxs:
            return [], [], []
        mains = get_frames_from_indices(idxs, self.frame_store_main)
        wrists = [self.frame_store_wrist.get(idx) for idx in idxs]
        return mains, wrists, idxs

    def infer_sync(self, step_idx: int, context_frames_np: list[tuple[Any, Any | None]]) -> str:
        if not context_frames_np:
            return self._current_subtask

        # v0.11: clamp the recent window to non-negative absolute indices. Early in an
        # episode (step_idx < window) the runtime pads the context, which used to produce
        # virtual negative frame indices (e.g. frames=[-6]) that leaked into J_hist / KF /
        # stage visuals as fake "evidence".
        recent_start = max(0, step_idx - len(context_frames_np) + 1)
        # P1: remember exactly which absolute frames this plan_step is shown, so a stage write
        # raised by this step can archive *these* frames instead of a possibly-stale accumulator.
        self._pmh_last_context_idx = list(range(recent_start, recent_start + len(context_frames_np)))
        context_main_frames: list[Image.Image] = []
        context_wrist_frames: list[Image.Image | None] = []
        for offset, frame_pack in enumerate(context_frames_np):
            abs_idx = recent_start + offset
            main_np, wrist_np = frame_pack
            main_img = Image.fromarray(main_np.astype("uint8"))
            wrist_img = Image.fromarray(wrist_np.astype("uint8")) if self.use_wrist and wrist_np is not None else None
            self.frame_store_main[abs_idx] = main_img
            self.frame_store_wrist[abs_idx] = wrist_img
            context_main_frames.append(main_img)
            context_wrist_frames.append(wrist_img)
        self.step = max(self.step, step_idx + 1)

        if self.episodic_store is not None:
            self.episodic_store.sync_keyframes(list(self.K_indices_abs))
        if self.pmh_store is not None:
            # Record the whole short-term window into the active segment buffer.
            self.pmh_store.note_frames(
                list(range(recent_start, recent_start + len(context_frames_np)))
            )

        mode = self.proactive_cfg.mode if self.proactive_cfg is not None else "off"
        if self.scec is not None:
            mode = "scec"
        extra_memory_text = ""
        memory_main_frames: list[Image.Image] = []
        memory_wrist_frames: list[Image.Image | None] = []
        memory_indices: list[int] = []
        ace_trace: dict[str, Any] | None = None
        pcam_trace: dict[str, Any] | None = None

        if mode in {"pmh", "pmh_v0", "pmh_v01", "pmh_state_only"} and self.pmh_store is not None:
            # PMH: Task State always on; Short-Term = recent + active-segment reps;
            # v0.6 bank mode: stage-indexed visual bank (|K|~6–8) replaces tiny floor.
            if mode != "pmh_state_only":
                self._pmh_maybe_flush_before_decide(step_idx)
            extra_memory_text = self.pmh_store.render_task_state()
            use_visual_bank = os.environ.get("PMH_VISUAL_BANK", "0").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            floor_idxs: list[int] = []
            if use_visual_bank:
                memory_main_frames, memory_wrist_frames, memory_indices = self._pmh_visual_bank(
                    recent_start=recent_start
                )
                kf_spine = os.environ.get("PMH_KF_SPINE", "0").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                if memory_indices:
                    extra_memory_text += _pmh_bank_annotation(memory_indices, kf_spine=kf_spine)
                # Bank is "ready" once it reaches a few frames — skip stall-force spam.
                bank_ready_k = int(os.environ.get("PMH_BANK_READY_K", "4"))
                self._pmh_has_archive_visual = len(memory_indices) >= max(1, bank_ready_k)
            else:
                memory_main_frames, memory_wrist_frames, memory_indices = self._pmh_short_term_visual(
                    recent_start=recent_start
                )
                floor_main, floor_wrist, floor_idxs = self._pmh_archived_visual_floor(
                    exclude=set(memory_indices)
                )
                if floor_idxs:
                    memory_indices = list(memory_indices) + list(floor_idxs)
                    memory_main_frames = list(memory_main_frames) + list(floor_main)
                    memory_wrist_frames = list(memory_wrist_frames) + list(floor_wrist)
                    extra_memory_text += (
                        f"\narchived_visual_floor_frames: {floor_idxs} "
                        "(recent completed-event keyframes; always on)\n"
                    )
                self._pmh_has_archive_visual = bool(floor_idxs)
            # --- SDV L0: the deterministic substrate, appended to whatever the read path gave ---
            # Placed here, AFTER both branches, so it is structurally impossible for a branch to
            # skip it. 586567's lesson was that a mechanism reachable by one path (the stage
            # confirmation) was assumed reachable by all, and the two paths that never fired went
            # unnoticed because the mechanism "ran" on the third.
            if _truthy_env("PMH_SDV_SUBSTRATE"):
                memory_main_frames, memory_wrist_frames, memory_indices, extra_memory_text = (
                    self._sdv_substrate_inject(
                        memory_main_frames,
                        memory_wrist_frames,
                        memory_indices,
                        recent_start,
                        extra_memory_text,
                    )
                )
                # Log the RESOLVED gate values once per episode, not the intent. Job 586700 ran 26
                # tasks believing SDV was on while `substrate_injected` stayed 0 for the entire run,
                # and asking "is the flag set?" from outside the process is what made that possible
                # to miss: the answer is only meaningful when read from the process that acts on it.
                if not getattr(self, "_sdv_gate_logged", False):
                    self._sdv_gate_logged = True
                    logger.info(
                        "[sdv] gates ACTIVE in-process: substrate=%r verify=%r stride=%r cap=%r "
                        "kf_spine=%r j_hist_n=%s",
                        os.environ.get("PMH_SDV_SUBSTRATE"),
                        os.environ.get("PMH_SDV_VERIFY"),
                        os.environ.get("PMH_SDV_STRIDE", "40"),
                        os.environ.get("PMH_SDV_SUBSTRATE_K", "8"),
                        os.environ.get("PMH_KF_SPINE"),
                        len(self.J_hist),
                    )
            extra_memory_text += render_verified_ledger(self.pmh_store)
            did_inspect = False
            n_search_used = 0
            n_inspect_used = 0
            max_search = int(os.environ.get("PMH_MAX_SEARCH_PER_PLAN", "1"))
            max_inspect = int(os.environ.get("PMH_MAX_INSPECT_PER_PLAN", "1"))
            logger.info(
                "[pmh] plan_step mode=%s rounds=%s segs=%s stage_vis=%s bank=%s mem_k=%s floor_k=%s stall=%s kf_n=%s",
                mode,
                self.pmh_max_tool_rounds,
                len(self.pmh_store.segments),
                len(self.pmh_store.stage_visuals),
                int(use_visual_bank),
                len(memory_indices),
                len(floor_idxs),
                self._consecutive_same_subtask,
                len(self.K_indices_abs),
            )
            if mode != "pmh_state_only":
                evidence_bits: list[str] = []
                max_visual = int(os.environ.get("PMH_MAX_VISUAL_PER_PLAN", os.environ.get("PMH_MAX_INSPECT_PER_PLAN", "1")))
                sparse = os.environ.get("PMH_SPARSE_DECIDE", "0").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                stall_thr = int(os.environ.get("PMH_STALL_FORCE", "6"))
                need_decide = True
                boundary_deepen = os.environ.get("PMH_BOUNDARY_DEEPEN", "0").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                deepen_cd = max(1, int(os.environ.get("PMH_DEEPEN_COOLDOWN", "2")))
                boundary_armed = bool(self._pmh_enter_stage or self._pmh_new_segment)
                boundary_cd_ok = (
                    int(self.pmh_store.n_decide) - int(self._pmh_last_boundary_deepen)
                ) >= deepen_cd
                boundary_trigger = boundary_deepen and boundary_armed and boundary_cd_ok
                if sparse:
                    unc_high = "uncertainty: high" in (self.pmh_store.task_state or "").lower()
                    # v2.0 L0: in delta mode "stalled" means the state stopped changing, not
                    # that the subtask label repeated. The label reading is what drove t22 into
                    # a retrieve-every-decision loop while the arm was manipulating normally.
                    stall_signal = (
                        self._pmh_steps_since_change
                        if _pmh_stall_mode() == "delta"
                        else self._consecutive_same_subtask
                    )
                    need_decide = (
                        not self._pmh_has_archive_visual
                        or bool(getattr(self, "_pmh_exec_stall", False))
                        or stall_signal >= max(1, stall_thr)
                        or unc_high
                        or boundary_trigger
                    )
                tool_rounds = max(0, self.pmh_max_tool_rounds) if need_decide else 0
                if sparse and not need_decide:
                    logger.info("[pmh] sparse_decide skip (bank ready, no deficit)")
                boundary_was_used = False
                for _round in range(tool_rounds):
                    decision = self._decide_pmh_access(
                        context_main_frames,
                        context_wrist_frames,
                        evidence_so_far="\n".join(evidence_bits[-3:]),
                        bank_indices=list(memory_indices),
                    )
                    boundary_was_used = boundary_was_used or boundary_trigger
                    tool = str(decision.get("tool", "none"))
                    if tool == "none":
                        self.pmh_store.n_decide_none += 1
                        break
                    if tool == "search_memory":
                        if n_search_used >= max(0, max_search):
                            evidence_bits.append("search_segments: budget exhausted this plan step.")
                            break
                        q = str(decision.get("query", ""))
                        n_search_used += 1
                        # PMH-C: the GENERATOR is the thing being fixed. A similarity generator
                        # returns the recent relevant segment, which the resident bank already
                        # holds; a provenance generator returns the segment that ESTABLISHED the
                        # fact, which is normally older than the recency window. Counted separately
                        # from the legacy path so the census can tell the two apart.
                        if _truthy_env("PMH_CITE"):
                            bank_cap = int(os.environ.get("PMH_VISUAL_BANK_K", "8"))
                            notes, added, outcome, cio = self._pmh_cite_retrieve(
                                q,
                                on_context=set(memory_indices),
                                bank_cap=bank_cap,
                                step_idx=step_idx,
                            )
                            evidence_bits.extend(notes)
                            for idx in added:
                                self._pmh_bank_add(memory_indices, idx, bank_cap, step_idx)
                            if added:
                                memory_main_frames = get_frames_from_indices(
                                    memory_indices, self.frame_store_main
                                )
                                memory_wrist_frames = [
                                    self.frame_store_wrist.get(idx) for idx in memory_indices
                                ]
                                self._pmh_has_archive_visual = True
                            # A citation retrieval SPENDS a retrieval and must be accounted as one,
                            # or PMH_NONPRODUCTIVE_MAX can never fire (the exact defect that let
                            # the retired ladder run 352 rounds with the guard configured).
                            self._pmh_note_retrieve_outcome(len(added))
                            self.pmh_store.events.append({
                                "op": "cite",
                                "query": q,
                                "outcome": outcome,
                                "cio": round(float(cio), 3),
                                "n_added": len(added),
                            })
                            logger.info(
                                "[pmh] cite query=%r outcome=%s cio=%.2f n_added=%s mem_k=%s",
                                q[:60],
                                outcome,
                                cio,
                                len(added),
                                len(memory_indices),
                            )
                            # `blocked` / `no_provenance` / `miss` all mean the archive has said
                            # everything it can, and each carries a note telling the planner so.
                            # Continuing the tool loop would re-ask a question the harness has
                            # already proven unanswerable - the loop has no other exit.
                            if outcome in {"blocked", "no_provenance", "miss"}:
                                extra_memory_text = (
                                    self.pmh_store.render_task_state()
                                    + "\n\n"
                                    + "\n\n".join(evidence_bits)
                                )
                                break
                            extra_memory_text = (
                                self.pmh_store.render_task_state()
                                + "\n\n"
                                + "\n\n".join(evidence_bits)
                            )
                            continue
                        txt = self.pmh_store.search_memory(q)
                        if self._pmh_ce_enabled():
                            txt = self._pmh_ce_filter_search(
                                txt, self.pmh_store.render_task_state()
                            )
                        evidence_bits.append(txt)
                        # v3.0 (PMH.md Sec 5.2): the text rung is Step 1 of a ladder, not a
                        # dead end. It exists to locate a segment, and the visual rungs then
                        # open that segment until new evidence appears. Without this the arm
                        # could finally SEARCH (budget 0 -> 1) but the answer would still be
                        # text-only, which is the half of the coarse-to-fine contract that
                        # spec Sec 6.2/6.3 split across two tools.
                        if _truthy_env("PMH_LADDER"):
                            # Defined locally, NOT read from the retrieve branch below: that
                            # branch assigns `bank_cap` too, but it runs only when the agent
                            # chose retrieve_visual, so relying on it here would be a NameError
                            # exactly when the agent asks a question first - i.e. in the one
                            # code path this change adds.
                            bank_cap = int(os.environ.get("PMH_VISUAL_BANK_K", "8"))
                            # THE SIDE-DOOR FIX. `_pmh_ladder` calls `store.retrieve_visual`
                            # directly, so until this check it was the ONE retrieval path that did
                            # not pass through `_pmh_retrieve_right()` - which is where BOTH L0
                            # guards live (PMH_RETRIEVE_PER_STAGE=4, PMH_NONPRODUCTIVE_MAX=2).
                            # Job 570778's t5 is the consequence: 22 identical
                            # `search -> keyframes -> full_visual -> n_added=0` rounds in one
                            # stage, with both guards configured and neither physically able to
                            # fire, because the only code that increments the counters they read
                            # (`_pmh_note_retrieve_outcome`) sits in the retrieve_visual branch
                            # that the agent never chose. A guard one path can walk around is not
                            # a guard, and the loop it failed to stop is the one this arm exists to
                            # make possible - so this is a correctness fix to the arm's own claim,
                            # not extra machinery.
                            allowed, why = self._pmh_retrieve_right()
                            if not allowed:
                                self._pmh_retrieval_refused += 1
                                evidence_bits.append(why)
                                logger.info(
                                    "[pmh] ladder REFUSED (stage=%s in_stage=%s nonprod=%s): %s",
                                    self._pmh_current_stage,
                                    self._pmh_retrievals_in_stage,
                                    self._pmh_nonproductive_streak,
                                    why.splitlines()[0],
                                )
                                # Keep the text answer, drop the frames: the planner gets the
                                # answer `search_memory` gave AND the instruction to act, then the
                                # tool loop stops. Same rule as the retrieve branch - a refusal
                                # must cost nothing - and it is what makes the planner act instead
                                # of asking the same question a 23rd time.
                                extra_memory_text = (
                                    self.pmh_store.render_task_state()
                                    + "\n\n"
                                    + "\n\n".join(evidence_bits)
                                )
                                break
                            notes, added, depth = self._pmh_ladder(
                                q, on_context=set(memory_indices), search_text=txt
                            )
                            evidence_bits.extend(notes)
                            for idx in added:
                                self._pmh_bank_add(memory_indices, idx, bank_cap, step_idx)
                            if added:
                                memory_main_frames = get_frames_from_indices(
                                    memory_indices, self.frame_store_main
                                )
                                memory_wrist_frames = [
                                    self.frame_store_wrist.get(idx) for idx in memory_indices
                                ]
                                self._pmh_has_archive_visual = True
                            # The ladder SPENT a retrieval, so it must be accounted like one.
                            # Without this line the nonproductive streak stays 0 forever and
                            # PMH_NONPRODUCTIVE_MAX can never fire, however many empty rounds run.
                            self._pmh_note_retrieve_outcome(len(added))
                            store_ev = {
                                "op": "ladder",
                                "query": q,
                                "depth": depth,
                                "n_added": len(added),
                            }
                            self.pmh_store.events.append(store_ev)
                            logger.info(
                                "[pmh] ladder query=%r depth=%s n_added=%s mem_k=%s",
                                q[:60],
                                depth,
                                len(added),
                                len(memory_indices),
                            )
                        extra_memory_text = (
                            self.pmh_store.render_task_state() + "\n\n" + "\n\n".join(evidence_bits)
                        )
                        continue
                    if tool in {"retrieve_visual", "inspect_segment"}:
                        if n_inspect_used >= max(0, max_visual):
                            evidence_bits.append("retrieve_visual: budget exhausted this plan step.")
                            break
                        # v2.0 L0: a refusal must cost nothing. Breaking here returns control
                        # to the VLA for this step instead of burning it on an empty answer.
                        allowed, why = self._pmh_retrieve_right()
                        if not allowed:
                            self._pmh_retrieval_refused += 1
                            evidence_bits.append(why)
                            logger.info(
                                "[pmh] retrieve REFUSED (stage=%s in_stage=%s nonprod=%s): %s",
                                self._pmh_current_stage,
                                self._pmh_retrievals_in_stage,
                                self._pmh_nonproductive_streak,
                                why.splitlines()[0],
                            )
                            break
                        txt, idxs = self.pmh_store.retrieve_visual(
                            segment_id=str(decision.get("segment_id", "last")),
                            detail=str(decision.get("detail", "keyframes")),
                            mode=str(decision.get("mode", "by_id")),
                        )
                        evidence_bits.append(txt)
                        did_inspect = True
                        n_inspect_used += 1
                        n_added = 0
                        bank_cap = int(os.environ.get("PMH_VISUAL_BANK_K", "8"))
                        if idxs:
                            # Prefer KF-spine frames when available.
                            kf_set = set(self.K_indices_abs)
                            ordered = [i for i in idxs if i in kf_set] + [
                                i for i in idxs if i not in kf_set
                            ]
                            # v3.0 (PMH.md Sec 6.4): admit by MARGINAL gain rather than in
                            # recency/FIFO order. The old loop appended every candidate and
                            # evicted the oldest when full, so a retrieval whose frames were
                            # near-copies of what the planner already held consumed the whole
                            # budget and displaced frames that were carrying different
                            # information - the planner paid context for evidence it already
                            # had. Selection now stops when nothing left clears the similarity
                            # floor (invariant I4) instead of padding the pack.
                            if _pmh_gain_enabled():
                                admitted, skipped = self._pmh_admit_by_gain(
                                    ordered, on_context=set(memory_indices), cap=bank_cap
                                )
                                self.pmh_store.n_gain_skipped += skipped
                                for idx in admitted:
                                    if self._pmh_bank_add(memory_indices, idx, bank_cap, step_idx):
                                        n_added += 1
                            else:
                                for idx in ordered:
                                    if idx in self.frame_store_main and self._pmh_bank_add(
                                        memory_indices, idx, bank_cap, step_idx
                                    ):
                                        n_added += 1
                        # v0.10 novelty guarantee -- v2.1: this MUST sit outside `if idxs:`.
                        # A retrieve that duplicates the on-context bank adds zero evidence, so
                        # pull genuinely new archive frames and make the round change something.
                        # The original placement only ran when `idxs` was non-empty, but the case
                        # that actually needs a guarantee is the one where the retrieval returns
                        # NOTHING: measured in job 568062, `n_new=0` held on 23 of 33 retrievals
                        # with the bank stuck at 2-3 frames and ZERO substitutions ever made.
                        novel_k = int(os.environ.get("PMH_NOVEL_K", "0"))
                        if n_added == 0 and novel_k > 0:
                            novel = self._pmh_novel_archive_visual(
                                exclude=set(memory_indices), max_k=novel_k
                            )
                            for idx in novel:
                                if self._pmh_bank_add(memory_indices, idx, bank_cap, step_idx):
                                    n_added += 1
                            self._pmh_last_retrieve_novel = bool(novel)
                            if novel:
                                note = (
                                    f"retrieve_visual: requested frames already on-context; "
                                    f"added novel archive frames {novel} instead.\n"
                                )
                                evidence_bits.append(note)
                        if idxs or n_added:
                            memory_main_frames = get_frames_from_indices(
                                memory_indices, self.frame_store_main
                            )
                            memory_wrist_frames = [
                                self.frame_store_wrist.get(idx) for idx in memory_indices
                            ]
                            self._pmh_has_archive_visual = True
                        # v2.0 L0: account the outcome so an unproductive retrieval cannot be
                        # repeated indefinitely within the same stage.
                        self._pmh_note_retrieve_outcome(n_added)
                        logger.info(
                            "[pmh] tool retrieve n_new=%s mem_k=%s did_inspect=%s nonprod=%s in_stage=%s",
                            n_added,
                            len(memory_indices),
                            int(did_inspect),
                            self._pmh_nonproductive_streak,
                            self._pmh_retrievals_in_stage,
                        )
                        extra_memory_text = (
                            self.pmh_store.render_task_state() + "\n\n" + "\n\n".join(evidence_bits)
                        )
                        continue
                    break
                # Stall safety / CE complementary read / KAIROS residual gate.
                # Shape MUST match FullVlm26MemoryPlanner._pmh_paper_read:
                #   stall ∧ read_open ∧ segments → one search+inspect per plateau.
                # Job 574701 measured paper_read=0 because three extra gates
                # (did_inspect / bank_skip∧has_archive_visual / elif-not-ce) made the
                # CE branch unreachable once floor frames set has_archive_visual.
                bank_skip = os.environ.get("PMH_BANK_SKIP_STALL_FORCE", "1").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                kairos_gate = False
                if kairos_enabled() and self.pmh_store is not None:
                    recent_imgs = [
                        self.frame_store_main[i]
                        for i in list(self.K_indices_abs)[-6:]
                        if i in self.frame_store_main
                    ]
                    if len(recent_imgs) < 2:
                        recent_imgs = [
                            self.frame_store_main[i]
                            for i in sorted(self.frame_store_main.keys())[-6:]
                        ]
                    residual = compute_residual(
                        recent_imgs,
                        same_subtask=int(self._consecutive_same_subtask or 0),
                    )
                    kairos_gate = bool(residual.get("gate"))
                    if residual.get("ok"):
                        logger.info(
                            "[kairos] residual stuck=%.3f surprise=%.3f gate=%s same=%s",
                            float(residual.get("stuck") or 0.0),
                            float(residual.get("surprise") or 0.0),
                            int(kairos_gate),
                            int(self._consecutive_same_subtask or 0),
                        )
                stall_now = (
                    self._consecutive_same_subtask >= max(1, stall_thr)
                    or bool(getattr(self, "_pmh_exec_stall", False))
                    or kairos_gate
                )
                ce = self._pmh_ce_enabled()
                read_open = os.environ.get("PMH_READ_OPEN", "0").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                # ---- PMH-O READ PREDICATE (invariants I4/I5) ----------------------------
                #
                # TWO CRITERIA, ONE STAIRCASE - and only the second one costs pixels.
                #
                # The old predicate was `stall_now`, a SYMPTOM. Every failure class produces it,
                # so it spent reads on t22's `01_Lift_Tomato_Sauce` (a physical grasp failure
                # with no missing address, where the read escalated to `full_visual` - the most
                # expensive call available - and could not help by construction), and it needed a
                # hardcoded per-task stage table to stop firing on t1/t18, which PMH.md Sec 11
                # forbids. Meanwhile the tasks where memory IS decisive (t4/t5) got the same
                # undifferentiated treatment as the tasks where it is irrelevant.
                #
                # The predicate is now the address demand queue, in two tiers:
                #   tier 1 (cheap, no archive read): inject the QUESTION plus pointer to the
                #     planner's OWN earlier observation frames - already resident, so this costs
                #     no retrieval and cannot mislead (v0.12's failure was injecting pixels).
                #     This is `_pmh_demand_evidence`, and it is what gives the planner its first
                #     chance to answer from evidence it already has.
                #   tier 2 (costs a read): the demand is STILL open after `PMH_DEMAND_GRACE`
                #     decides, i.e. the planner had its chance and did not take it. Only then is
                #     the archive searched, and the query is the ADDRESS, not the subtask prose.
                #     `_pmh_unaddressed_demand` returns the oldest demand, so a physical stall
                #     (which opens no demand) consumes nothing.
                _demand = self._pmh_unaddressed_demand() if (ce and read_open) else None
                _gd = max(0, int(os.environ.get("PMH_DEMAND_GRACE", "3")))
                _demand_ready = False
                if _demand is not None:
                    _daddr, _dinfo = _demand
                    _now_t = int(getattr(self.pmh_store, "n_decide", 0) or 0)
                    if not int(_dinfo.get("last_t", 0) or 0):
                        # First sighting: stamp it and give tier 1 the floor. Stamping on the
                        # decide count rather than the step index keeps the grace window
                        # independent of how many steps a single decision spans.
                        _dinfo["last_t"] = _now_t
                    _demand_ready = (_now_t - int(_dinfo.get("last_t", 0) or 0)) >= _gd
                if (
                    (_demand is not None and _demand_ready)
                    and self.pmh_store is not None
                    and self.pmh_store.segments
                ):
                    _cap = int(os.environ.get("PMH_CE_MAX_PAPER_READS", "4"))
                    _count = int(getattr(self, "_pmh_paper_read_count", 0) or 0)
                    if ce and bool(getattr(self, "_pmh_stall_read_done", False)):
                        logger.info(
                            "[pmh] ce skip demand read (already read this plateau) addr=%s same=%s",
                            _demand[0],
                            self._consecutive_same_subtask,
                        )
                    elif ce and _cap > 0 and _count >= _cap:
                        logger.info(
                            "[pmh] ce skip demand read (episode cap %s reached) addr=%s same=%s",
                            _cap,
                            _demand[0],
                            self._consecutive_same_subtask,
                        )
                    elif did_inspect:
                        logger.info(
                            "[pmh] ce skip demand read (agent already inspected) addr=%s same=%s",
                            _demand[0],
                            self._consecutive_same_subtask,
                        )
                    else:
                        # I5: the query IS the address. Rendered through `binding_query` so the
                        # tokenisation actually matches what a consolidator writes -
                        # `top_drawer.contents` would otherwise require the literal substring
                        # `top_drawer` in a caption and could never match, which reads in the log
                        # as a retrieval that ran and found nothing.
                        query = self.pmh_store.binding_query(_demand[0])
                        _dq = str(_demand[1].get("question") or "").strip()
                        if _dq:
                            query = f"{query} {_dq}".strip()[:160]
                        logger.info(
                            "[pmh] demand_read addr=%s why=%s query=%r grace=%s n_open=%s same=%s",
                            _demand[0],
                            _demand[1].get("why"),
                            query,
                            _gd,
                            len(self.pmh_store.open_bindings),
                            self._consecutive_same_subtask,
                        )
                        search_text = self.pmh_store.search_memory(query, top_k=2)
                        if ce:
                            search_text = self._pmh_ce_filter_search(
                                search_text, self.pmh_store.render_task_state()
                            )
                        evidence_bits.append(search_text)
                        sids = re.findall(r"seg_\d+", search_text)
                        if not sids:
                            sids = ["last"]
                        bank_cap = int(os.environ.get("PMH_VISUAL_BANK_K", "8"))
                        n_added = 0
                        used_sid = sids[0]
                        detail_kf = os.environ.get("PMH_INSPECT_DETAIL", "keyframes")

                        def _ingest(sid: str, detail: str) -> int:
                            nonlocal n_added, used_sid
                            txt, idxs = self.pmh_store.retrieve_visual(
                                segment_id=sid, detail=detail, mode="by_id"
                            )
                            evidence_bits.append(txt)
                            used_sid = sid
                            added_here = 0
                            for idx in idxs:
                                if idx in self.frame_store_main and self._pmh_bank_add(
                                    memory_indices, idx, bank_cap, step_idx
                                ):
                                    n_added += 1
                                    added_here += 1
                            return added_here

                        for sid in sids[:2]:
                            if _ingest(sid, detail_kf):
                                break
                        if n_added == 0:
                            _ingest(used_sid, "full_visual")
                            logger.info(
                                "[pmh] ce escalate full_visual sid=%s n_added=%s",
                                used_sid,
                                n_added,
                            )
                        if n_added:
                            memory_main_frames = get_frames_from_indices(
                                memory_indices, self.frame_store_main
                            )
                            memory_wrist_frames = [
                                self.frame_store_wrist.get(idx) for idx in memory_indices
                            ]
                            self._pmh_has_archive_visual = True
                        if ce:
                            self._pmh_stall_read_done = True
                            self._pmh_paper_read_count = _count + 1
                        # ---- PMH-O EFFECT GATE (the D4 lesson, made operational) ----------
                        #
                        # A read is credited ONLY when it did something observable. The two
                        # failure modes this excludes are exactly the ones job 575817 ran on:
                        #   * `n_added == 0` - the read admitted no new frame, so nothing the
                        #     planner did next can be attributed to it. Counting it (the old
                        #     behaviour: `_pmh_paper_read_count += 1` unconditionally) makes an
                        #     inert read look productive, which is how a census reads healthy
                        #     while the claim moves nothing.
                        #   * no address closed - the retrieved text did not supply a demanded
                        #     address. Even with new frames admitted, if the demand is still open
                        #     the read did not resolve the gap it was spent on.
                        # The demand is closed ONLY on the first case; the second case
                        # (new frames, demand still open) leaves the demand queued for tier 1 to
                        # retry, because the frames may still answer it after the planner looks.
                        _closed = False
                        if n_added:
                            _closed = self.pmh_store.close_binding(_demand[0], by="read")
                        elif "NO LEXICAL MATCH" in (search_text or "") or n_added == 0:
                            # R2: one empty archive search is enough. Leave the demand open for
                            # plan/observation closure, but refuse further tier-2 spends.
                            self.pmh_store.exhaust_binding(_demand[0])
                            logger.info(
                                "[pmh] binding_exhausted addr=%s n_added=%s no_match=%s",
                                _demand[0],
                                n_added,
                                int("NO LEXICAL MATCH" in (search_text or "")),
                            )
                        self.pmh_store.events.append(
                            {
                                "op": "demand_read",
                                "addr": _demand[0],
                                "why": _demand[1].get("why"),
                                "query": query,
                                "n_added": int(n_added),
                                "closed": bool(_closed),
                                "effective": bool(n_added),
                                "exhausted": self.pmh_store.binding_is_exhausted(_demand[0]),
                                "t": int(step_idx),
                            }
                        )
                        logger.info(
                            "[pmh] demand_read_effect addr=%s n_added=%s closed=%s opened=%s "
                            "resolved_read=%s resolved_plan=%s open_now=%s",
                            _demand[0],
                            n_added,
                            int(bool(_closed)),
                            self.pmh_store.n_bindings_opened,
                            self.pmh_store.n_bindings_resolved_by_read,
                            self.pmh_store.n_bindings_resolved_by_plan,
                            len(self.pmh_store.open_bindings),
                        )
                        self._pmh_exec_stall = False
                        self.pmh_store.n_forced += 1
                        extra_memory_text = (
                            self.pmh_store.render_task_state()
                            + "\n\n"
                            + "\n\n".join(evidence_bits)
                        )
                        logger.info(
                            "[pmh] paper_read stall search+inspect query=%r n_added=%s sid=%s ce=%s segs=%s",
                            query[:80],
                            n_added,
                            used_sid,
                            int(ce),
                            len(self.pmh_store.segments),
                        )
                elif (
                    not did_inspect
                    and not self._pmh_has_archive_visual
                    and self.pmh_store is not None
                    and self.pmh_store.segments
                    and stall_now
                    and not (bank_skip and self._pmh_has_archive_visual)
                    and not _pmh_demand_mode()
                ):
                    txt, idxs = self.pmh_store.retrieve_visual(
                        segment_id="last", detail="keyframes", mode="by_salience"
                    )
                    extra_memory_text = (
                        self.pmh_store.render_task_state() + "\n\n" + txt
                    )
                    self.pmh_store.n_forced += 1
                    self.pmh_store.last_force_decide = int(self.pmh_store.n_decide)
                    self.pmh_store.events.append({"op": "force_retrieve_visual_stall_fallback"})
                    self._pmh_exec_stall = False
                    n_added = 0
                    bank_cap = int(os.environ.get("PMH_VISUAL_BANK_K", "8"))
                    if idxs:
                        for idx in idxs:
                            if idx in self.frame_store_main and self._pmh_bank_add(
                                memory_indices, idx, bank_cap, step_idx
                            ):
                                n_added += 1
                    # v2.1: hoisted out of `if idxs:` for the same reason as the decision-driven
                    # path above. There are TWO retrieve sites in this file and the first repair
                    # only fixed one; scripts/validate_pmh_v21_repairs.py caught the miss by
                    # asserting structurally that the PMH_NOVEL_K lookup is not a descendant of
                    # `if idxs:`. This one is the stall fallback, i.e. the path that runs when the
                    # planner is stuck - exactly when a retrieve that returns nothing is worst.
                    novel_k = int(os.environ.get("PMH_NOVEL_K", "0"))
                    if n_added == 0 and novel_k > 0:
                        novel = self._pmh_novel_archive_visual(
                            exclude=set(memory_indices), max_k=novel_k
                        )
                        for idx in novel:
                            if self._pmh_bank_add(memory_indices, idx, bank_cap, step_idx):
                                n_added += 1
                        if novel:
                            extra_memory_text += (
                                f"\nretrieve_visual(stall): requested frames already on-context; "
                                f"added novel archive frames {novel} instead.\n"
                            )
                    if idxs or n_added:
                        memory_main_frames = get_frames_from_indices(
                            memory_indices, self.frame_store_main
                        )
                        memory_wrist_frames = [
                            self.frame_store_wrist.get(idx) for idx in memory_indices
                        ]
                        self._pmh_has_archive_visual = True
                    logger.info(
                        "[pmh] stall fallback retrieve_visual n_frames=%s n_new=%s", len(idxs), n_added
                    )
                    if self.logger:
                        self.logger.info(
                            "[pmh] stall fallback retrieve_visual n_frames=%s n_new=%s", len(idxs), n_added
                        )
                # v0.10: consume boundary-deepen trigger (one grounded decide per boundary).
                if boundary_armed:
                    if boundary_was_used:
                        self._pmh_last_boundary_deepen = int(self.pmh_store.n_decide)
                        self._pmh_enter_stage = False
                        self._pmh_new_segment = False
                        logger.info(
                            "[pmh] boundary_deepen consumed (bank_ready=%s tool=%s)",
                            int(bool(self._pmh_has_archive_visual)),
                            str(self.pmh_store.last_tool),
                        )
                    elif boundary_trigger:
                        # Cooldown deferred (not used this step): keep armed for next plan.
                        logger.info(
                            "[pmh] boundary_deepen deferred (cooldown) enter=%s new_seg=%s",
                            int(bool(self._pmh_enter_stage)),
                            int(bool(self._pmh_new_segment)),
                        )
            # v0.14: repeated-stall "change approach" nudge (visible to THIS planner call).
            alt_txt = self._pmh_stall_alt_directive(step_idx)
            if alt_txt:
                extra_memory_text += "\n" + alt_txt.strip()
            # v0.15 CGM-OB: inject ONE evidence-linked fact, only when the active stage
            # needs an object the consolidator no longer reports as visible.
            contract_txt = self._pmh_contract_hint(step_idx)
            if contract_txt:
                extra_memory_text += "\n" + contract_txt.strip() + "\n"
            # v0.16 Gap Gate: when the controller proves an unresolved binding exists, push
            # the planner's OWN earlier container observations (the Open_* frames) and attach
            # those frames. This is the replaced-and-inverted policy: the gate OPENS here
            # instead of PMH blindly retrieving everywhere (v0.15e: t4 +12.5pp where a gap is
            # real, but t11 -20pp / t22 -50pp where the gate is provably closed).
            gap_txt, gap_frames = self._pmh_gap_evidence(step_idx)
            if gap_txt:
                extra_memory_text += "\n" + self._pmh_fair_inject(gap_txt).strip() + "\n"
                bank_cap = int(os.environ.get("PMH_VISUAL_BANK_K", "8"))
                for idx in gap_frames:
                    self._pmh_bank_add(memory_indices, idx, bank_cap, step_idx)
                memory_main_frames = get_frames_from_indices(
                    memory_indices, self.frame_store_main
                )
                memory_wrist_frames = [
                    self.frame_store_wrist.get(idx) for idx in memory_indices
                ]
            # PMH-O TIER 1: the address demand, as a QUESTION with the planner's own
            # observation pointers. Zero archive reads, zero extra API calls, and - unlike the
            # v0.12/v0.13 injections - it carries no ANSWER, so it cannot hand the planner the
            # binding the scored stages exist to test (job 566614: injecting the drawer name
            # collapsed t4 to a zero-variance 25.0 by letting the planner skip stages 01-06).
            # Tier 2 (`demand_read`) is what runs when this question survives the grace window.
            dem_txt, dem_frames = self._pmh_demand_tier1(step_idx)
            if dem_txt:
                extra_memory_text += "\n" + self._pmh_fair_inject(dem_txt).strip() + "\n"
                bank_cap = int(os.environ.get("PMH_VISUAL_BANK_K", "8"))
                for idx in dem_frames:
                    self._pmh_bank_add(memory_indices, idx, bank_cap, step_idx)
                memory_main_frames = get_frames_from_indices(
                    memory_indices, self.frame_store_main
                )
                memory_wrist_frames = [
                    self.frame_store_wrist.get(idx) for idx in memory_indices
                ]
            ace_trace = {
                "source": "pmh",
                "mode": mode,
                "stats": self.pmh_store.stats(),
                "last_tool": self.pmh_store.last_tool,
                "short_term_indices": list(memory_indices),
                "archived_floor_k": len(floor_idxs),
                "alt_stall_n": self._pmh_n_alt_inject,
                "contract_hint_n": self.pmh_store.n_contract_hints,
            }
            # Keep counters durable even if process dies mid-episode / cleanup deletes traces.
            self._dump_pmh_stats()
        elif mode == "scec" and self.scec is not None:
            tid = int(getattr(self.task_info, "task_id", 0) or 0)
            self.scec.sync_before_plan(task_id=tid, keyframe_indices=list(self.K_indices_abs))
            pack = self.scec.compile()
            if pack.visual_indices:
                memory_main_frames = get_frames_from_indices(pack.visual_indices, self.frame_store_main)
                memory_wrist_frames = [self.frame_store_wrist.get(idx) for idx in pack.visual_indices]
            extra_memory_text = pack.semantic_text or ""
        elif mode == "passive_samepool":
            memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
            memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
            memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
            memory_main_frames, memory_wrist_frames, memory_indices = self._cap_memory_frames(
                memory_main_frames, memory_wrist_frames, memory_indices
            )
            if self.episodic_store is not None:
                extra_memory_text = self.episodic_store.render_semantic()
        elif mode in {"passive_semantic", "passive_k4_sem", "passive_k4_sem_stall"}:
            memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
            memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
            memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
            memory_main_frames, memory_wrist_frames, memory_indices = self._cap_memory_frames(
                memory_main_frames, memory_wrist_frames, memory_indices
            )
            if mode != "passive_semantic" and self.episodic_store is not None:
                extra_memory_text = self.episodic_store.render_semantic(recent_n=self.semantic_recent_n)
            elif mode == "passive_semantic" and self.episodic_store is not None:
                extra_memory_text = self.episodic_store.render_semantic()
            if mode == "passive_k4_sem_stall" and self._consecutive_same_subtask >= self.read_stall_repeats:
                memory_indices = self._merge_extra_visual(memory_indices, self.read_stall_extra_k)
                memory_main_frames = get_frames_from_indices(memory_indices, self.frame_store_main)
                memory_wrist_frames = [self.frame_store_wrist.get(idx) for idx in memory_indices]
        elif mode in {
            "passive_ace",
            "passive_ace_v2",
            "passive_pace_plus",
            "passive_cog",
            "passive_hermes",
        } and self.ace is not None:
            store = self.episodic_store
            brief = self.ace.build_brief(
                task_id=int(getattr(self.task_info, "task_id", 0) or 0),
                step=step_idx,
                stall_count=self._consecutive_same_subtask,
                store=store,
            )
            manifest = self.ace.resolve_manifest(
                brief,
                api_key=self.api_key,
                base_url=self.api_base_url,
            )
            bank = list(self.K_indices_abs) if self.use_keyframe_memory else []
            memory_indices = self.ace.select_visual_indices(bank, manifest)
            memory_main_frames = get_frames_from_indices(memory_indices, self.frame_store_main)
            memory_wrist_frames = [self.frame_store_wrist.get(idx) for idx in memory_indices]
            if store is not None:
                extra_memory_text = self.ace.render_semantic(store, manifest.semantic_recent_n)
            ace_trace = manifest.to_dict()
            ace_trace["brief"] = {
                "stall": brief.stall_count,
                "bank": brief.bank_size,
                "stage_switched": brief.stage_switched,
            }
        elif mode in {"passive_pcam", "passive_pcam_act"} and self.pcam is not None:
            store = self.episodic_store
            brief = self.pcam.build_brief(
                task_id=int(getattr(self.task_info, "task_id", 0) or 0),
                step=step_idx,
                stall_count=self._consecutive_same_subtask,
                store=store,
            )
            visual_k, semantic_n, deficit_tag = self.pcam.resolve_budget(brief)
            bank = list(self.K_indices_abs) if self.use_keyframe_memory else []
            memory_indices = self.pcam.select_visual_indices(store, bank, visual_k, step_idx)
            memory_main_frames = get_frames_from_indices(memory_indices, self.frame_store_main)
            memory_wrist_frames = [self.frame_store_wrist.get(idx) for idx in memory_indices]
            if store is not None:
                extra_memory_text = store.render_semantic(recent_n=semantic_n)
            pcam_trace = {
                "visual_k": visual_k,
                "semantic_recent_n": semantic_n,
                "deficit_tag": deficit_tag,
                "source": "pcam",
                "brief": {
                    "stall": brief.stall_count,
                    "bank": brief.bank_size,
                    "stage_switched": brief.stage_switched,
                },
            }
        elif mode == "proactive":
            decision = self._decide_memory_access(context_main_frames, context_wrist_frames)
            tool = str(decision.get("tool", "none"))
            store = self.episodic_store
            if tool == "recall_semantic" and store is not None and self.proactive_cfg.allow_semantic:
                extra_memory_text = store.render_semantic(what=str(decision.get("what", "all")))
                store.n_recall_semantic += 1
            elif tool == "recall_visual" and store is not None:
                k = int(decision.get("k", self.proactive_cfg.visual_k))
                if self.read_k_max > 0:
                    k = min(k, self.read_k_max)
                idxs = store.select_visual_indices(k)
                memory_main_frames = get_frames_from_indices(idxs, self.frame_store_main)
                memory_wrist_frames = [self.frame_store_wrist.get(idx) for idx in idxs]
                store.n_recall_visual += 1
            else:
                if store is not None:
                    store.n_decide_none += 1
        elif mode == "hermes_delta":
            # Default ≡ harness_v21 KF path (never rewrite with ACE floor).
            memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
            memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
            memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
            memory_main_frames, memory_wrist_frames, memory_indices = self._cap_memory_frames(
                memory_main_frames, memory_wrist_frames, memory_indices
            )
            # One-shot agent Inquiry Δ only when Supervisor armed info_gap.
            if getattr(self, "hermes_inquiry_pending", False):
                self.hermes_inquiry_pending = False
                decision = self._decide_memory_access(context_main_frames, context_wrist_frames)
                tool = str(decision.get("tool", "none"))
                store = self.episodic_store
                hermes_delta_trace: dict[str, Any] = {
                    "tool": tool,
                    "k": decision.get("k"),
                    "what": decision.get("what"),
                    "forced": bool(decision.get("forced", False)),
                    "parsed_ok": bool(decision.get("parsed_ok", False)),
                }
                if tool == "recall_semantic" and store is not None and self.proactive_cfg.allow_semantic:
                    extra_memory_text = store.render_semantic(what=str(decision.get("what", "all")))
                    store.n_recall_semantic += 1
                elif tool == "recall_visual" and store is not None:
                    k = int(decision.get("k", self.proactive_cfg.visual_k))
                    if self.read_k_max > 0:
                        k = min(k, self.read_k_max)
                    extra = store.select_visual_indices(k)
                    # Merge Δ into Harness pack (union), do not replace Baseline.
                    merged = list(memory_indices)
                    for idx in extra:
                        if idx not in merged:
                            merged.append(idx)
                    memory_indices = merged
                    memory_main_frames = get_frames_from_indices(memory_indices, self.frame_store_main)
                    memory_wrist_frames = [self.frame_store_wrist.get(idx) for idx in memory_indices]
                    store.n_recall_visual += 1
                    hermes_delta_trace["extra_indices"] = list(extra)
                else:
                    if store is not None:
                        store.n_decide_none += 1
                ace_trace = {"source": "hermes_inquiry", **hermes_delta_trace}
                if self.logger:
                    self.logger.info(
                        "[hermes-delta] inquiry tool=%s forced=%s pack_k=%s",
                        tool,
                        hermes_delta_trace.get("forced"),
                        len(memory_indices),
                    )
        else:
            memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
            memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
            memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
            memory_main_frames, memory_wrist_frames, memory_indices = self._cap_memory_frames(
                memory_main_frames, memory_wrist_frames, memory_indices
            )

        user_content = self._build_messages(
            memory_main_frames,
            memory_wrist_frames,
            context_main_frames,
            context_wrist_frames,
            extra_memory_text=extra_memory_text,
        )

        out_text = infer_primitive_via_api(
            system_prompt=self.system_prompt,
            user_content=user_content,
            api_key=self.api_key,
            base_url=self.api_base_url,
            model=self.api_model,
            timeout_sec=self.api_timeout,
            max_tokens=self.api_max_tokens,
        )
        if not out_text:
            if self.logger:
                self.logger.warning("API planner returned empty response at step=%s", step_idx)
            return self._current_subtask or self.default_subtask_prompt

        vlm_subtask, j_rel = parse_vlm_output(out_text, max_pos=len(context_main_frames))
        j_abs = [recent_start + (p - 1) for p in j_rel]

        if self.use_keyframe_memory:
            self.J_hist.append(j_abs)
            # FALSIFIER for the planner's keyframe channel (SDV). This counter is the one whose
            # absence cost this project the most time. `J_hist.append([])` keeps `len(J_hist)`
            # growing, so every downstream reader saw a populated trajectory, `kf_n=0` was read as
            # "the bank happens to be empty right now", and the actual condition -- the planner
            # NEVER nominates a keyframe -- was invisible. Job 586700 finally exposed it: with an
            # empty `J_hist` of candidates, `build_visual_memory` returns nothing, `K_indices_abs`
            # stays empty, and L0's substrate is empty with it.
            #
            # A trajectory of empty nominations is not a small trajectory; it is a DISABLED one.
            # Counted here so the two can never be confused again.
            # GUARDED ON THE STORE, AND THIS IS NOT DEFENSIVE PADDING. `infer_sync` is called
            # UNCONDITIONALLY by the VLM worker (`eval_..._vla.py:1526`), and on an arm without
            # `PMH_ENABLE` -- `api_qwen_harness_v21_redact`, the HM baseline every memory delta is
            # measured against -- `pmh_store` is legitimately None for the whole run. Job 589531
            # crashed exactly here:
            #
            #     AttributeError: 'NoneType' object has no attribute 'n_kf_nomination_empty'
            #
            # which raised out of the worker, failed the episode, and scored 0.0 on task 12 where
            # the SAME arm had scored 100.0 on 2026-09-14. It is worth being precise about the
            # failure mode: this counter was added to make `kf_n=0` VISIBLE (an empty nomination
            # trajectory is a disabled one, not a small one), and instead it made the baseline
            # unrunnable.
            #
            # IT SAT DORMANT FOR TWO DAYS because the branch is only reached when
            # `use_keyframe_memory` is on AND the store is present: every arm run in between set
            # `PMH_ENABLE`, so every one of them executed it safely. The arm that breaks is the
            # one nobody re-runs while chasing a memory delta -- which is why the guard belongs
            # here rather than in a note.
            if self.pmh_store is not None:
                if not j_abs:
                    self.pmh_store.n_kf_nomination_empty = (
                        int(getattr(self.pmh_store, "n_kf_nomination_empty", 0) or 0) + 1
                    )
                else:
                    self.pmh_store.n_kf_nomination_nonempty = (
                        int(getattr(self.pmh_store, "n_kf_nomination_nonempty", 0) or 0) + 1
                    )
            prev_subtask = self._current_subtask
            mem_cfg = self.memory_system_config
            if mem_cfg.salience_subtask_change and vlm_subtask and vlm_subtask != prev_subtask:
                self.mark_salient_keyframe(step_idx)
            if mem_cfg.stage_anchor:
                from memory_system.keyframe_bank import merge_keyframe_bank

                self.K_indices_abs = merge_keyframe_bank(
                    j_hist=self.J_hist,
                    t=self.step,
                    recent_window=len(context_main_frames),
                    cluster_distance=mem_cfg.cluster_distance,
                    pinned_steps=self.pinned_keyframe_steps,
                    salient_steps=self.salient_keyframe_steps,
                    bank_max=mem_cfg.bank_max,
                )
            else:
                raw_k_indices = build_visual_memory(
                    self.J_hist, t=self.step, N=len(context_main_frames), d=self.d_merge
                )
                self.K_indices_abs = [idx for idx in raw_k_indices if idx < recent_start]
            # Additive temporal spread (mem_efficacy channel B). Runs AFTER the official builder
            # so it can only ADD frames -- it never removes an anchor the official path selected,
            # which is the same asymmetry `_sdv_substrate_floor` uses ("the agent may enrich the
            # substrate, never empty it") and for the same reason: a candidate source that could
            # displace a stage anchor would make the two mechanisms indistinguishable in the
            # result.
            #
            # IT IS BUDGET-AWARE, AND THAT WAS A REAL DEFECT. The first version was not: it
            # unioned `kf_spread` frames on top of the anchors and let the existing cap block trim
            # the result. That block keeps the LAST `cap` entries (`unpinned[-(cap - kept):]`), so
            # with two anchors and 8 spread frames against `K_MAX=8` the two EARLIEST frames were
            # discarded -- measured: [0,112,180,225,337,420,451,563,675,787] -> [180,...,787],
            # span 787 -> 607. The whole point of the spread is to span the episode, so the cap
            # was silently converting "spread" into "the second half". Passing the remaining
            # budget down instead makes the later cap block a no-op and the span a property of the
            # bank rather than of the truncation order.
            cap = self.k_max if self.k_max > 0 else mem_cfg.bank_max
            if int(getattr(mem_cfg, "kf_spread", 0) or 0) > 0:
                _anchors = len(self.K_indices_abs)
                _budget = int(mem_cfg.kf_spread)
                if cap > 0:
                    _budget = min(_budget, max(0, int(cap) - _anchors))
                self.K_indices_abs = self._kf_spread_union(
                    self.K_indices_abs, recent_start, _budget
                )
                self.K_main_frames = get_frames_from_indices(
                    self.K_indices_abs, self.frame_store_main
                )
                self.K_wrist_frames = [
                    self.frame_store_wrist.get(idx) for idx in self.K_indices_abs
                ]
            self.K_main_frames = get_frames_from_indices(self.K_indices_abs, self.frame_store_main)
            self.K_wrist_frames = [self.frame_store_wrist.get(idx) for idx in self.K_indices_abs]
            cap = self.k_max if self.k_max > 0 else mem_cfg.bank_max
            if cap > 0 and len(self.K_indices_abs) > cap:
                pinned = set(self.pinned_keyframe_steps)
                pinned_kept = [idx for idx in self.K_indices_abs if idx in pinned]
                unpinned = [idx for idx in self.K_indices_abs if idx not in pinned]
                if len(pinned_kept) >= cap:
                    self.K_indices_abs = pinned_kept[-cap:]
                else:
                    self.K_indices_abs = pinned_kept + unpinned[-(cap - len(pinned_kept)) :]
                self.K_main_frames = get_frames_from_indices(self.K_indices_abs, self.frame_store_main)
                self.K_wrist_frames = [self.frame_store_wrist.get(idx) for idx in self.K_indices_abs]

        if self.episodic_store is not None:
            self.episodic_store.sync_keyframes(list(self.K_indices_abs))
            if vlm_subtask:
                if self.scec is not None:
                    self.scec.note_primitive(step_idx, vlm_subtask)
                else:
                    self.episodic_store.record_subtask(step_idx, vlm_subtask)

        # Stall tracker (shared by PMH anti-collapse / soft commit). Keep independent of episodic store.
        if vlm_subtask:
            if vlm_subtask == self._last_planned_subtask:
                self._consecutive_same_subtask += 1
            else:
                self._last_planned_subtask = vlm_subtask
                self._consecutive_same_subtask = 1
                self._pmh_steps_since_change = 0
                self._pmh_stall_committed = False
                self._pmh_stall_read_done = False
            # PUSH CHANNEL + EVIDENCE-BASED OBLIGATION (job 586421).
            #
            # Both are wired HERE, at the point where the planner's own declaration is known,
            # rather than at the stage-confirm path. That placement IS the fix: stage confirmation
            # never fires on t4/t5/t8/t9 (measured: zero `stage_visual_write` in t4's window while
            # t11 confirmed normally), so anything hung off it is unreachable on exactly the tasks
            # where memory is decisive. The declaration happens every step, on every task.
            #
            # PUSH: record the primitive so `render_task_state` can always show the planner what
            # it has already been doing. Appended per STEP (not per change) so a repeated
            # primitive is visible as repetition -- the loop signature no ledger key can carry,
            # and on t4 the loop ("open top drawer" again and again) is the whole story.
            if self.pmh_store is not None and _truthy_env("PMH_PUSH_TIMELINE", default=False):
                self.pmh_store.push_timeline.append(
                    f"t={int(step_idx)} {str(vlm_subtask).strip()[:60]}"
                )
            # SEAM: the observed trace, re-typed as (verb, entity), plus Sigma's self-invalidation.
            #
            # Note what this replaces. The segment captions carry the same information as a JOINT
            # SENTENCE -- "Executed primitive 'pick tomato' then switched to 'open microwave'" --
            # and a sentence can be neither folded into a count nor compared against an obligation.
            # `note_executed` is also the S3 invalidation signal: the robot's own arm is what
            # causes occlusion, so its own action is the one invalidation signal that survives it.
            if self.pmh_store is not None and self.pmh_store.seam is not None:
                try:
                    self.pmh_store.seam.note_observed_primitive(
                        str(vlm_subtask), int(step_idx)
                    )
                    self.pmh_store.seam.note_executed(str(vlm_subtask), int(step_idx))
                except Exception:
                    logger.exception("[seam] observed-trace write failed")
            # OBLIGATION: a declared primitive proves the question is OPEN. Deduped by address
            # inside `open_binding`, so per-step calling cannot inflate the queue.
            self._pmh_note_obligation_from_evidence(str(vlm_subtask), int(step_idx))
        # v2.0 L0: a second, honest staleness signal. `_consecutive_same_subtask` counts
        # identical LABELS, which is not progress: t22 keeps one subtask name across a long
        # manipulation, so the count climbs to the trigger threshold, decide fires, retrieval
        # returns already-held frames (n_new=0 on 83% of t22 retrievals), and the episode never
        # moves - H scores 100.0 there in 13s while PMH scores 0.0. This counter only advances
        # when nothing actually changed: no new subtask AND no segment committed.
        self._pmh_steps_since_change += 1

        # PMH write path. v3.0 (PMH.md Sec 4.2): a candidate boundary PROPOSES an event; the
        # consolidator's verdict decides. The buffer itself no longer commits on a step count -
        # only a semantic candidate (subtask_boundary) or a capacity limit may close an event.
        #
        # `soft_stall` is retained as a branch but must default to 0 for v3.0 arms. It is the
        # timer that produced 69% of job 568107's commits and fired six times back to back
        # inside a single stage of t5; spec Sec 4.2 lists four boundary signals, all of them
        # semantic, and a step counter is none of them. It is left in place (rather than
        # deleted) so the value stays a documented, revertible knob instead of a code change.
        if self.pmh_store is not None and vlm_subtask:
            soft_stall = int(os.environ.get("PMH_SOFT_COMMIT_STALL", "3"))
            soft_min_frames = int(os.environ.get("PMH_SOFT_COMMIT_MIN_FRAMES", "6"))
            max_active = int(os.environ.get("PMH_MAX_ACTIVE_FRAMES", "14"))
            do_commit = self.pmh_store.maybe_boundary(step_idx, vlm_subtask)
            reason = "subtask_boundary" if do_commit else ""
            if (
                not do_commit
                and soft_stall > 0
                and self._consecutive_same_subtask >= soft_stall
                and len(self.pmh_store.active_frame_indices) >= soft_min_frames
            ):
                do_commit = True
                reason = "soft_stall"
            if not do_commit and len(self.pmh_store.active_frame_indices) >= max_active:
                do_commit = True
                reason = "max_active_frames"
            if do_commit:
                ce = self._pmh_ce_enabled()
                if reason == "soft_stall" and ce and bool(getattr(self, "_pmh_stall_committed", False)):
                    logger.info(
                        "[pmh] ce skip stall commit (already committed this plateau) same=%s",
                        self._consecutive_same_subtask,
                    )
                    do_commit = False
            if do_commit:
                prev = self.pmh_store.active_subtask or vlm_subtask
                # Only a semantic candidate goes through the completeness verdict. The other
                # two reasons are capacity/anti-stall closures whose whole purpose is that the
                # event is NOT allowed to stay buffered (PMH.md Sec 4.1), so they force-close.
                self._pmh_commit_active(
                    step_idx,
                    prev_subtask=prev,
                    new_subtask=vlm_subtask,
                    reason=reason or "commit",
                    force_close=(reason != "subtask_boundary"),
                )
                if reason == "subtask_boundary":
                    self._pmh_stall_committed = False
                    self._pmh_stall_read_done = False
                elif reason == "soft_stall" and self._pmh_ce_enabled():
                    self._pmh_stall_committed = True
                # A commit IS a state change, so staleness resets here too. A WITHHELD commit
                # is not: the frames are still on the buffer, so the next candidate must still
                # be able to see that nothing has been archived yet.
                if self.pmh_store.n_commits > getattr(self, "_pmh_commits_at_last_mark", -1):
                    self._pmh_steps_since_change = 0
                    self._pmh_commits_at_last_mark = self.pmh_store.n_commits
            else:
                if not self.pmh_store.active_subtask:
                    self.pmh_store.active_subtask = vlm_subtask.strip()
                self.pmh_store.note_frame(step_idx)

        if vlm_subtask:
            # D2 falsifier. The previous arm's projection rendered an underscore-paren token
            # (`open(middle_drawer)`) and the planner adopted it verbatim on the NEXT `plan_step`,
            # after which the stage matcher could no longer recognise the primitive and stage 01
            # never completed -- 9 tasks at 0, every one of them stalled at stage 01. This counter
            # is the measurement that says whether the canonicalisation actually held: it must be
            # 0, and a non-zero reading means a rendered line is still teaching the planner a form
            # it should not emit.
            try:
                if _seam_looks_non_canonical(vlm_subtask):
                    if self.pmh_store is not None and self.pmh_store.seam is not None:
                        self.pmh_store.seam.n_subtask_surface_drift += 1
                    logger.warning(
                        "[seam] planner emitted a NON-CANONICAL primitive: %r", str(vlm_subtask)
                    )
            except Exception:
                pass
            self._current_subtask = vlm_subtask
        subtask = self._current_subtask

        trace = {
            "t": int(step_idx),
            "task_id": int(self.task_info.task_id),
            "subtask": subtask,
            "keyframe_positions": j_rel,
            "J_abs": j_abs,
            "K_indices_abs": list(self.K_indices_abs),
            "out_text": str(out_text).strip()[:600],
            "planner_backend": "api",
            "api_model": self.api_model,
            "proactive_mode": mode,
            "read_k_max": self.read_k_max,
            "memory_indices": memory_indices,
            "stall_count": self._consecutive_same_subtask,
            # Observability for the keyframe channel's CANDIDATE POOL, which is the channel's real
            # constraint. A bank can only be as good as the frames the store holds, and the store
            # was until now populated only from the context windows of planner calls that survived
            # async queue eviction -- about 1.4% of the episode. `n_store` against `recent_start`
            # is what makes "the pool is too small to span the episode" measurable instead of
            # inferred, and `n_ctx` is the window size `recent_start` is derived from.
            "n_store": len(self.frame_store_main),
            "recent_start": int(recent_start),
            "n_ctx": len(context_main_frames),
        }
        if self.pmh_store is not None:
            trace["pmh_stats"] = self.pmh_store.stats()
            trace["pmh_last_tool"] = self.pmh_store.last_tool
        if self.scec is not None:
            trace["scec_stats"] = self.scec.stats()
            if self.scec.last_pack is not None:
                trace["scec_gated"] = self.scec.last_pack.gated
                trace["scec_reason"] = self.scec.last_pack.reason
        elif self.episodic_store is not None:
            trace["proactive_stats"] = self.episodic_store.stats()
            trace["proactive_last_tool"] = self.episodic_store.last_tool
        if self.ace is not None:
            trace["ace_stats"] = self.ace.stats.to_dict()
            if ace_trace is not None:
                trace["ace_manifest"] = ace_trace
        if self.pcam is not None:
            trace["pcam_stats"] = self.pcam.stats.to_dict()
            if pcam_trace is not None:
                trace["pcam_budget"] = pcam_trace
        if ace_trace is not None and self.pmh_store is not None:
            trace["pmh_trace"] = ace_trace
        self._append_trace(trace)
        if self.logger:
            self.logger.info(
                "API VLM @t=%s task=%s subtask=%s keyframes=%s",
                step_idx,
                self.task_info.task_id,
                subtask,
                j_rel,
            )
        return subtask
