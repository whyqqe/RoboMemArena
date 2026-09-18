from __future__ import annotations

import json
import logging

# The module has always logged through a bare module-level `logger` (25 call sites), but the name
# was never bound. That was latent and invisible while the taken code paths happened to avoid it:
# `logger.info("async VLM enabled: ...")` sits on the worker startup path and every retry path in
# `run_episode_async_stateful` uses it too, so the local planner only had to reach one of them once
# to die with `NameError: name 'logger' is not defined` - which is exactly how job 571167 lost
# task1 (via `restore_pmh_memory`, the first PMH-only path to reach it). Binding it here is the fix
# the module's own design implies: `main()` configures the ROOT logger with basicConfig, so a child
# logger propagates to that handler and nothing else has to change. A `.getLogger("__main__")`
# variant would ALSO work but only when the file is executed as a script, whereas this keeps
# `import` - which is how every one of these call sites is reached - working identically.
logger = logging.getLogger(__name__)
import copy
import os
import re
import queue
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: F401


REFERENCE_DIR = Path(__file__).resolve().parent
EVAL_BENCHMARK_DIR = REFERENCE_DIR.parent
RUNTIME_DIR = EVAL_BENCHMARK_DIR / "openpi_minimal_runtime"
SCRIPTS_DIR = EVAL_BENCHMARK_DIR / "scripts"
ROOT = Path(os.environ["OPENPI_ROOT"])
INFERENCE_ROOT = Path(os.environ.get("OPENPI_INFERENCE_ROOT", str(ROOT.parent / "openpi_inference")))
OPENPI_CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
OPENPI_SRC = ROOT / "packages" / "openpi" / "src"
LIBERO_PATH_ENV = os.environ.get("TARGET_LIBERO_PATH", "").strip()
if not LIBERO_PATH_ENV:
    _fallback_libero = ROOT / "third_party" / "libero"
    if _fallback_libero.exists():
        LIBERO_PATH_ENV = str(_fallback_libero)
LIBERO_PATHS: list[Path] = []
if LIBERO_PATH_ENV:
    _libero_path = Path(LIBERO_PATH_ENV)
    LIBERO_PATHS.extend([_libero_path, _libero_path.parent])

module_paths = [
    str(EVAL_BENCHMARK_DIR),
    str(RUNTIME_DIR),
    str(SCRIPTS_DIR),
    str(OPENPI_CLIENT_SRC),
    str(OPENPI_SRC),
]
for _lib_path in LIBERO_PATHS:
    module_paths.append(str(_lib_path))

for p in module_paths:
    if p and p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import eval_common as ec
import task2_26_reference_stage as stage_eval
from eval_task1_qwen3_async_openpi_inference_vla_cam import (
    Args as BaseArgs,
    StableWebsocketClientPolicy,
    SyncLoRAPlanner,
    _apply_vlm_input_profile,
    _extract_vlm_frame,
    _seed_everywhere,
    _write_video,
    make_episode_logger,
)
from harness.api_vlm_planner import ApiMemoryPlanner, _redact_container_phrases
from harness.kairos_wm import compute_residual, kairos_enabled
from harness.belief_contract_memory import BeliefContractMemory, compose_planner_context
from harness.config import HarnessConfig, load_harness_config
from harness.controller import HarnessController
from harness.experience_ltm import HpmController, find_latest_trace
from harness.meta_controller import MetaController
from harness.muscle_memory import MuscleMemoryRing
from harness.primitives import release_gripper
from harness.pact import PactController
from harness.evidence_compiler import AceController
from harness.proactive_memory import (
    EpisodicStore,
    ProactiveMemoryConfig,
    apply_anti_collapse,
    create_store_if_needed,
    memory_decision_system_prompt,
    parse_memory_decision,
)
# PMH-P on the LOCAL planner transport. The memory layer itself is backend-agnostic; these are
# the only entry points the local planner needs (invariant I3 deletes every selective-read
# mechanism, so there is deliberately nothing else to import here).
from harness.pmh_memory import (
    collect_pact_extra,
    commit_segment,
    create_pmh_store_if_needed,
    dump_pmh_episode_stats,
    pick_representative_indices,
)
from harness.vlm_output_parser import parse_vlm_output
from memory_system.config import load_memory_system_config
from keyframe_selection import build_visual_memory, get_frames_from_indices
from robocerebra_adapter import obs_to_pi_element


SYSTEM_PROMPT_MEMORY_DEMO = """You are an embodied-memory robot VLM planner.

You will observe two kinds of visual evidence from the same long-horizon execution:
1. Historical keyframes: moments before the current step, used to remember important past states.
2. A recent 5-frame dual-camera window ending at the current frame, used to infer the current primitive.
Temporal order: historical keyframes are ordered from earliest to latest; the recent 5-frame window is also ordered from earliest to latest, and the last timestep in that window is the current frame.

Your goal is not to narrate the full execution. Your goal is to infer the primitive the robot is currently executing, or should execute now, from these images.

Important rules:
- Historical keyframes are always earlier than the recent visual window.
- If there is no keyframe in the recent window, keyframe_positions must be an empty list.
- keyframe_positions are 1-indexed positions within the recent 5-frame window.
- Output strict JSON only, with no extra text.
- The JSON must contain exactly two fields: current_primitive and keyframe_positions."""

SYSTEM_PROMPT_MEMORY_LONGTASK = SYSTEM_PROMPT_MEMORY_DEMO

SYSTEM_PROMPT_MEMORY = (
    SYSTEM_PROMPT_MEMORY_LONGTASK
    if os.environ.get("VLM_LONGTASK_PROMPT", "0") == "1"
    else SYSTEM_PROMPT_MEMORY_DEMO
)


@dataclass(frozen=True)
class TaskInfo:
    task_id: int
    suite: str
    task_name: str
    memory_type: str
    challenge: str
    brief_description: str
    task_block: str
    scene_description: str
    primitive_labels: list[str]


def load_task_infos(path: Path) -> dict[int, TaskInfo]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, TaskInfo] = {}
    for task in raw["tasks"]:
        out[int(task["task_id"])] = TaskInfo(
            task_id=int(task["task_id"]),
            suite=str(task["suite"]),
            task_name=str(task["task_name"]),
            memory_type=str(task["memory_type"]),
            challenge=str(task["challenge"]),
            brief_description=str(task["brief_description"]),
            task_block=str(task["task_block"]),
            scene_description=str(task.get("scene_description", "")),
            primitive_labels=[str(p["label"]) for p in task["primitive_order"]],
        )
    return out


def _camera_order_text(use_wrist_images: bool) -> str:
    if use_wrist_images:
        return (
            "Camera order for every timestep: agentview_rgb, eye_in_hand_rgb. "
            "agentview_rgb is the external main-view camera, and eye_in_hand_rgb is the wrist/end-effector camera."
        )
    return "Camera: agentview_rgb. agentview_rgb is the external main-view camera."


def _parse_output_no_mapping(output_text: str, max_pos: int) -> tuple[str, list[int]]:
    return parse_vlm_output(output_text, max_pos)


class FullVlm26MemoryPlanner(SyncLoRAPlanner):
    def __init__(self, *args: Any, task_info: TaskInfo, **kwargs: Any) -> None:
        processor_model_dir = kwargs.pop("processor_model_dir", None)
        if processor_model_dir is None:
            processor_model_dir = os.environ.get("VLM_PROCESSOR_DIR", kwargs.get("base_model_dir", args[0] if args else ""))
        model_dir = Path(kwargs.get("base_model_dir", args[0] if args else ""))
        processor_dir = Path(processor_model_dir)
        if model_dir.is_dir() and processor_dir.is_dir():
            for name in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json"):
                src = processor_dir / name
                dst = model_dir / name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        super().__init__(*args, **kwargs)
        # Some DeepSpeed/Trainer checkpoints save model weights but not the image processor files.
        # Keep the trained weights from base_model_dir, but use the canonical Qwen3-VL processor.
        self.processor = AutoProcessor.from_pretrained(
            processor_model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.harness_extra_context: str = ""
        self.pinned_keyframe_steps: list[int] = []
        self.salient_keyframe_steps: list[int] = []
        self.memory_system_config = load_memory_system_config()
        self.proactive_cfg = ProactiveMemoryConfig.from_env()
        self.ace: AceController | None = AceController.create()
        self.episodic_store: EpisodicStore | None = create_store_if_needed(self.proactive_cfg)
        self.read_k_max = int(os.environ.get("READ_K_MAX", "0"))
        self.read_stall_repeats = int(os.environ.get("READ_STALL_REPEATS", "3"))
        self.read_stall_extra_k = int(os.environ.get("READ_STALL_EXTRA_K", "2"))
        self.semantic_recent_n = int(os.environ.get("SEMANTIC_RECENT_N", "3"))
        self._last_planned_subtask = ""
        self._consecutive_same_subtask = 0
        self.hermes_inquiry_pending = False
        # PMH-P: the addressable ledger is created only when PROACTIVE_MODE=pmh (or PMH_ENABLE).
        # A None store makes every hook below a no-op, so arms that do not use PMH are unaffected.
        self.pmh_store = create_pmh_store_if_needed()
        self._pmh_exec_stall = False
        self._pmh_hold_ledger = False
        self._pmh_new_segment = False
        self._pmh_current_stage = ""
        self._pmh_episode_task_id = int(getattr(task_info, "task_id", 0) or 0)
        # PMH-local stall tracking, kept separate from `_consecutive_same_subtask` (which only
        # advances when the episodic store exists) so no other arm's behaviour is touched.
        self._pmh_last_subtask = ""
        self._pmh_same_subtask = 0
        self._pmh_stall_committed = False
        # CE complementary read is also once-per-plateau. Jobs 573557/573616 aborted
        # (SIGABRT rc=134) on task1 after 10–11 identical stall paper_reads in one plateau;
        # kairos under the same contract did one read and finished. Re-injecting the same
        # archive every plan tick does not add evidence and was correlated with the crash.
        self._pmh_stall_read_done = False
        self._pmh_paper_read_count = 0
        self._redact_terms: set[str] = set()
        self._redact_n_masked = 0
        self.set_task_info(task_info)

    def set_task_info(self, task_info: TaskInfo) -> None:
        self.task_info = task_info
        self.default_subtask_prompt = task_info.brief_description.strip()
        self._current_subtask = self.default_subtask_prompt
        self._pmh_episode_task_id = int(getattr(task_info, "task_id", 0) or 0)
        self.register_redact_answer_key()

    def _redact_enabled(self) -> bool:
        return os.environ.get("PMH_REDACT_STAGE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _redact_log(self, msg: str, *args: Any) -> None:
        target = getattr(self, "logger", None) or logger
        target.info(msg, *args)

    def _register_redact_terms(self, text: str) -> int:
        terms = getattr(self, "_redact_terms", None)
        if terms is None:
            terms = self._redact_terms = set()
        before = len(terms)
        for phrase in _redact_container_phrases(text):
            terms.add(phrase)
            terms.add(phrase.replace(" ", "_"))
        return len(terms) - before

    def register_redact_answer_key(self) -> None:
        """Same answer-key registration as ApiMemoryPlanner: HM must not depend on pmh_store."""
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
        """Mask qualified container phrases in planner-visible extra. Never touch task_block."""
        if not text or not self._redact_enabled():
            return text
        out = text
        for term in sorted(self._redact_terms, key=len, reverse=True):
            if len(term) < 4:
                continue
            out = re.sub(re.escape(term), "[withheld]", out, flags=re.IGNORECASE)
        if out != text:
            self._redact_n_masked = getattr(self, "_redact_n_masked", 0) + 1
            if self._redact_n_masked == 1:
                self._redact_log(
                    "[redact] applied task=%s terms=%s",
                    getattr(self.task_info, "task_id", "?"),
                    sorted(self._redact_terms),
                )
        return out

    def _pmh_ce_enabled(self) -> bool:
        return os.environ.get("PMH_CE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _pmh_ce_filter_search(self, search_text: str, state_text: str) -> str:
        """Keep search lines that add facts the default Task State does not already hold."""
        state_l = (state_text or "").lower()
        kept: list[str] = []
        dropped = 0
        skip = {
            "seg",
            "caption",
            "action",
            "outcome",
            "objects",
            "state",
            "delta",
            "none",
            "search",
            "segments",
            "cards",
            "text",
            "only",
            "query",
            "hits",
            "note",
            "this",
            "path",
            "does",
            "load",
            "visual",
            "archive",
            "frames",
            "retrieve",
        }
        for line in (search_text or "").splitlines():
            if not line.startswith("- "):
                kept.append(line)
                continue
            toks = [t for t in re.split(r"[^a-z0-9_]+", line.lower()) if len(t) >= 4 and t not in skip]
            if toks and sum(1 for t in toks if t in state_l) >= max(2, (len(toks) + 1) // 2):
                dropped += 1
                continue
            kept.append(line)
        if dropped:
            logger.info(
                "[pmh] ce_filter dropped=%s kept=%s",
                dropped,
                sum(1 for line in kept if line.startswith("- ")),
            )
        return "\n".join(kept)

    def reset_episode(self, instruction: str | None = None, run_dir=None, logger=None):
        super().reset_episode(instruction=instruction, run_dir=run_dir, logger=logger)
        self._current_subtask = self.default_subtask_prompt
        self.harness_extra_context = ""
        self.pinned_keyframe_steps = []
        self.salient_keyframe_steps = []
        self.memory_system_config = load_memory_system_config()
        self.proactive_cfg = ProactiveMemoryConfig.from_env()
        self.ace = AceController.create()
        if self.ace is not None:
            self.ace.reset()
        if self.episodic_store is None:
            self.episodic_store = create_store_if_needed(self.proactive_cfg)
        elif self.episodic_store is not None:
            self.episodic_store.reset()
        self.read_k_max = int(os.environ.get("READ_K_MAX", "0"))
        self.read_stall_repeats = int(os.environ.get("READ_STALL_REPEATS", "3"))
        self.read_stall_extra_k = int(os.environ.get("READ_STALL_EXTRA_K", "2"))
        self.semantic_recent_n = int(os.environ.get("SEMANTIC_RECENT_N", "3"))
        self._last_planned_subtask = ""
        self._consecutive_same_subtask = 0
        self.hermes_inquiry_pending = False
        # Ledger lives at episode scope (PMH.md §4): an attempt is a reread, not a rewrite of
        # history. `reset_episode` runs at the start of EVERY attempt; wiping the store here is
        # what made job 571321 restore `segs=0` 19/19 times and dropped t22 100→66.7. Attempt 0
        # of a new task still rebuilds (`_pmh_hold_ledger` is false); retries set the flag.
        if self.pmh_store is None:
            self.pmh_store = create_pmh_store_if_needed()
        elif not getattr(self, "_pmh_hold_ledger", False):
            self.pmh_store.reset()
        self._pmh_exec_stall = False
        self._pmh_new_segment = False
        self._pmh_current_stage = ""
        self._pmh_last_subtask = ""
        self._pmh_same_subtask = 0
        self._pmh_stall_committed = False
        self._pmh_stall_read_done = False
        self._pmh_paper_read_count = 0
        self._redact_n_masked = 0
        self.register_redact_answer_key()

    def _cap_memory_frames(
        self,
        main_frames: list[Image.Image],
        wrist_frames: list[Image.Image | None],
        indices: list[int],
    ) -> tuple[list[Image.Image], list[Image.Image | None], list[int]]:
        if self.read_k_max <= 0 or len(main_frames) <= self.read_k_max:
            return main_frames, wrist_frames, indices
        return (
            main_frames[-self.read_k_max :],
            wrist_frames[-self.read_k_max :],
            indices[-self.read_k_max :],
        )

    def _merge_extra_visual(
        self,
        base_indices: list[int],
        extra_k: int,
    ) -> list[int]:
        if extra_k <= 0:
            return base_indices
        bank = list(self.K_indices_abs)
        out = list(base_indices)
        for idx in reversed(bank):
            if idx not in out:
                out.append(idx)
            if len(out) >= len(base_indices) + extra_k:
                break
        return out[-(len(base_indices) + extra_k) :] if self.read_k_max > 0 else out

    def pin_keyframe(self, step: int) -> None:
        if step >= 0 and step not in self.pinned_keyframe_steps:
            self.pinned_keyframe_steps.append(step)

    def mark_salient_keyframe(self, step: int) -> None:
        if step >= 0 and step not in self.salient_keyframe_steps:
            self.salient_keyframe_steps.append(step)

    def _build_messages(
        self,
        memory_main_frames: list[Image.Image],
        memory_wrist_frames: list[Image.Image | None],
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Image.Image | None],
        *,
        extra_memory_text: str = "",
    ):
        use_wrist_images = self.use_wrist and any(
            frame is not None for frame in (memory_wrist_frames + context_wrist_frames)
        )
        num_history_keyframes = len(memory_main_frames)
        num_history_images = num_history_keyframes * (2 if use_wrist_images else 1)
        num_context_frames = len(context_main_frames)
        num_context_images = num_context_frames * (2 if use_wrist_images else 1)

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Global objective: infer the robot's current primitive action from historical keyframes before the current step and recent visual history within the same execution.\n\n"
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
                    f"{_camera_order_text(use_wrist_images)}\n"
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
                        f"({num_history_keyframes} timesteps, {num_history_images} images):"
                    ),
                }
            )
            append_timestep_images(memory_main_frames, memory_wrist_frames)

        user_content.append(
            {
                "type": "text",
                "text": (
                    "Recent visual context: "
                    f"{num_context_frames} consecutive frames ending at the current frame "
                    f"({num_context_images} images):"
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
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def _vlm_generate(self, messages: list[dict[str, Any]], *, max_new_tokens: int | None = None) -> str:
        images = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                images.extend(c["image"] for c in content if isinstance(c, dict) and c.get("type") == "image")

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(text, list):
            text = text[0]
        inputs = self.processor(text=[text], images=images if images else None, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        gen_tokens = int(max_new_tokens if max_new_tokens is not None else self.max_new_tokens)
        with __import__("torch").inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=gen_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # ===================== PMH-P (addressable memory) hooks ==========================
    # The SAME memory layer that ApiMemoryPlanner drives, but on the local transport. The four
    # invariants are what keep this small:
    #   I1 every state fact is an address <entity>[#<ordinal>].<attribute> with write-time
    #      provenance (enforced inside merge_task_state / the consolidator prompt);
    #   I2 resolved addresses are resident in the planner's prompt, so reading needs NO tool call
    #      and planning never writes state;
    #   I3 there is NO selective read: no similarity search, no ladder, no citation retrieval,
    #      no gap gate, no tool-call loop. Evidence is reached only through segment provenance.
    #   I4 one writer (the consolidator, at segment commit) and one reader (the planner prompt).
    # Methods that are not needed do not exist here on purpose: an unused hook is a lying
    # interface. Every one of these is a no-op when the ledger is absent.

    def _pmh_generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        """Transport shim: the consolidator speaks the (system, user-parts) message form that
        this planner's own generator already consumes, so no adapter is required."""
        return self._vlm_generate(messages, max_new_tokens=max_new_tokens)

    def _pmh_active_add(self, indices: list[int]) -> None:
        """Grow the open segment's frame buffer. This is the WRITE side of the ledger: frames
        accumulate here and are only ever turned into state at commit (invariant I4)."""
        if self.pmh_store is None:
            return
        buf = self.pmh_store.active_frame_indices
        for i in indices:
            try:
                iv = int(i)
            except (TypeError, ValueError):
                continue
            if iv >= 0 and iv not in buf:
                buf.append(iv)

    def _pmh_snap_to_local_kf(self, cands: list[int], max_k: int = 2) -> list[int]:
        """Snap arbitrary commit frames onto the local keyframe spine.

        The spine (`K_indices_abs`) is the only frame set the local planner already curates and
        keeps in VRAM, so anchoring archived pointers to it keeps the evidence channel coherent
        without introducing a new frame-selection mechanism (invariant I3).
        """
        bank = [int(i) for i in self.K_indices_abs]
        if not bank:
            return []
        out: list[int] = []
        for c in cands:
            try:
                cv = int(c)
            except (TypeError, ValueError):
                continue
            nearest = min(bank, key=lambda b: abs(b - cv))
            if nearest not in out:
                out.append(nearest)
            if len(out) >= max(0, int(max_k)):
                break
        return out

    def _dump_pmh_stats(self) -> None:
        """Durable counters that survive video/trace cleanup (I1/I3 falsifiers live here)."""
        if self.pmh_store is None or self.run_dir is None:
            return
        dump_pmh_episode_stats(
            self.pmh_store,
            self.run_dir / "pmh_episode_stats.json",
            task_id=int(self._pmh_episode_task_id or 0),
            # A1/A2/A3/A4/A5 census (α-guard, PACT, recovery ladder). Missing here until now, so
            # every local run's own falsifiers read the treatment as inert.
            extra=collect_pact_extra(self),
        )

    def _pmh_commit_active(
        self,
        step_idx: int,
        *,
        prev_subtask: str,
        new_subtask: str,
        reason: str,
        force_close: bool = False,
    ) -> None:
        """Archive the open segment, gated by the consolidator's completeness verdict.

        `force_close` separates the two closure kinds that must not be conflated (PMH.md Sec
        4.2 vs 4.1): a candidate BOUNDARY is a question the verdict may answer NO to (frames
        stay buffered), whereas a frame-budget closure is a capacity limit that must never lose
        the current event and therefore commits unconditionally.
        """
        if self.pmh_store is None:
            return
        frames = list(self.pmh_store.active_frame_indices)
        if len(frames) < 1:
            return
        commit_idxs = pick_representative_indices(
            frames,
            max_k=int(os.environ.get("PMH_CONSOLIDATE_IMAGE_K", "4")),
        )
        commit_images = [self.frame_store_main[i] for i in commit_idxs if i in self.frame_store_main]
        card = commit_segment(
            self.pmh_store,
            step_idx=step_idx,
            prev_subtask=prev_subtask or new_subtask or self.pmh_store.active_subtask,
            new_subtask=new_subtask or prev_subtask or self.pmh_store.active_subtask,
            api_key="",
            base_url="",
            model="",
            images=commit_images,
            force_close=force_close,
            generate_fn=self._pmh_generate,
        )
        # The verdict may have withheld the commit (Sec 4.2); everything below writes
        # card-attached state, so it must not run for a card that was never archived.
        if card is None:
            return
        stage_now = str(self._pmh_current_stage or self.pmh_store.current_stage_name or "")
        if stage_now:
            card.stage_tag = stage_now
        kf_spine = os.environ.get("PMH_KF_SPINE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if kf_spine and self.pmh_store.segments:
            card = self.pmh_store.segments[-1]
            snapped = self._pmh_snap_to_local_kf(
                list(card.keyframe_indices or commit_idxs),
                max_k=int(os.environ.get("PMH_STAGE_WRITE_K", "2")),
            )
            if snapped:
                card.keyframe_indices = list(snapped)
                card.visual_indices = list(snapped)
        # Only a REAL subtask transition arms the boundary deepen; soft_stall /
        # max_active_frames repeat inside a stall loop and must not each trigger one.
        if reason == "subtask_boundary":
            self._pmh_new_segment = True
        logger.info(
            "[pmh] commit reason=%s frames=%s stall=%s stage=%s",
            reason,
            len(frames),
            self._consecutive_same_subtask,
            stage_now or "(none)",
        )
        self._dump_pmh_stats()

    def note_pmh_exec_stall(self) -> None:
        """Harness execution stall. Recorded, never acted on by a retrieval mechanism (I3)."""
        self._pmh_exec_stall = True

    def snapshot_pmh_memory(self) -> dict[str, Any] | None:
        """Preserve the ledger and its archived frames across Harness retries."""
        if self.pmh_store is None:
            return None
        need: set[int] = set()
        for seg in self.pmh_store.segments:
            need.update(int(i) for i in (seg.keyframe_indices or []))
            need.update(int(i) for i in (seg.visual_indices or [])[-12:])
        need.update(int(i) for i in self.pmh_store.active_frame_indices)
        for ref in self.pmh_store.stage_visuals:
            need.update(int(i) for i in (ref.frame_indices or []))
        return {
            "store": copy.deepcopy(self.pmh_store),
            "mains": {i: self.frame_store_main[i] for i in need if i in self.frame_store_main},
            "wrists": {i: self.frame_store_wrist[i] for i in need if i in self.frame_store_wrist},
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

    def pmh_note_stage_start(self, stage_name: str, step_idx: int) -> None:
        """Record the ACTIVE stage (what the robot is trying to achieve now)."""
        self._pmh_current_stage = str(stage_name or "").strip()
        self._register_redact_terms(stage_name)
        if self.pmh_store is not None:
            self.pmh_store.current_stage_name = self._pmh_current_stage
        logger.info("[pmh] stage_start stage=%s t=%s", self._pmh_current_stage, step_idx)

    def pmh_note_stage_done(self, stage_name: str, step_idx: int) -> None:
        """Stage-indexed evidence write, snapped to the local keyframe spine."""
        if self.pmh_store is None:
            return
        cands: list[int] = [int(i) for i in self.pmh_store.active_frame_indices[-24:] if int(i) >= 0]
        for p in self.pinned_keyframe_steps[-3:]:
            if int(p) >= 0 and int(p) not in cands:
                cands.append(int(p))
        if int(step_idx) not in cands and int(step_idx) in self.frame_store_main:
            cands.append(int(step_idx))
        cands = [i for i in cands if i in self.frame_store_main and int(i) >= 0] or [
            i for i in range(max(0, int(step_idx) - 8), int(step_idx) + 1) if i in self.frame_store_main
        ]
        write_k = int(os.environ.get("PMH_STAGE_WRITE_K", "2"))
        kf_spine = os.environ.get("PMH_KF_SPINE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        picked_cands = self._pmh_snap_to_local_kf(cands, max_k=write_k) if kf_spine else cands
        picked = self.pmh_store.note_stage_visual(stage_name, picked_cands, int(step_idx), max_k=write_k)
        logger.info(
            "[pmh] stage_visual_write stage=%s frames=%s t=%s kf_spine=%s",
            stage_name,
            picked,
            step_idx,
            int(kf_spine),
        )
        self._dump_pmh_stats()

    def pmh_note_retry_brief(
        self,
        *,
        attempt_idx: int,
        stalled_stage: str | None,
        completed_stages: list[str] | None,
        repeated_stall: bool = False,
    ) -> None:
        """Tell the retry what the previous attempt achieved and where it stalled.

        Pure text into the ledger's resident render (invariant I2): the retry focuses repair on
        the first incomplete stage instead of restarting, and a repeated identical stall gets a
        hard divergence warning rather than an identical replay.
        """
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
                "occluded — check Task State."
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
            repeated_stall,
        )

    def _pmh_visual_indices(self) -> list[int]:
        """I3: the evidence channel is provenance-only.

        Frames come from (a) the newest archived segment's provenance-stamped pointers and (b)
        the newest entries of the curated keyframe spine. No similarity search, no ranking, no
        query is involved — the ledger decides WHAT is archived, and this only decides how much
        of it is resident.
        """
        if self.pmh_store is None:
            return []
        max_visual = int(os.environ.get("PMH_MAX_VISUAL_PER_PLAN", "1"))
        protect_recent = int(os.environ.get("PMH_BANK_PROTECT_RECENT", "3"))
        out: list[int] = []
        for seg in reversed(self.pmh_store.segments):
            for i in list(seg.keyframe_indices or [])[:max(0, max_visual)]:
                iv = int(i)
                if iv not in out and iv in self.frame_store_main:
                    out.append(iv)
            if out:
                break
        for i in list(self.K_indices_abs)[-max(0, protect_recent):] if protect_recent > 0 else []:
            iv = int(i)
            if iv not in out and iv in self.frame_store_main:
                out.append(iv)
        return out

    def _pmh_paper_read(self) -> tuple[list[Image.Image], list[Image.Image | None], list[int], str]:
        """PMH.md §5.1 default plus §5.2/§8 stall fallback.

        Default every round: recent keyframes + Task State. Archive frames stay out of the
        default channel (that collapse is why search/cite historically returned CIO≈1). When
        the plan is repeating, walk the paper tools once: search_memory then inspect_segment
        (retrieve_visual keyframes). Local VLM tool-calling is unreliable, so this is the
        ACE fallback PMH.md §8 names, not a second architecture.
        """
        memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
        extra = self.pmh_store.render_task_state() if self.pmh_store is not None else ""
        soft_at = int(os.environ.get("PMH_SOFT_COMMIT_STALL", "3"))
        stall = bool(self._pmh_exec_stall) or (
            soft_at > 0 and int(getattr(self, "_pmh_same_subtask", 0) or 0) >= soft_at
        )
        kairos_gate = False
        if kairos_enabled():
            # Residual over recent keyframes already in the bank (no third prompt channel).
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
                same_subtask=int(getattr(self, "_pmh_same_subtask", 0) or 0),
            )
            kairos_gate = bool(residual.get("gate"))
            if residual.get("ok"):
                logger.info(
                    "[kairos] residual stuck=%.3f surprise=%.3f gate=%s same=%s",
                    float(residual.get("stuck") or 0.0),
                    float(residual.get("surprise") or 0.0),
                    int(kairos_gate),
                    int(getattr(self, "_pmh_same_subtask", 0) or 0),
                )
        stall = stall or kairos_gate
        read_open = os.environ.get("PMH_READ_OPEN", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        ce = self._pmh_ce_enabled()
        if stall and read_open and self.pmh_store is not None and self.pmh_store.segments:
            # CE: at most one archive search+inspect per stall plateau (mirrors once-per-
            # plateau commit). Further ticks keep the default Task State + KF channel only.
            # Episode hard cap (default 4) is a second fuse: jobs 573557/573616 aborted after
            # ~10–11 reads in one plateau before the plateau flag existed.
            _cap = int(os.environ.get("PMH_CE_MAX_PAPER_READS", "4"))
            _count = int(getattr(self, "_pmh_paper_read_count", 0) or 0)
            if ce and bool(getattr(self, "_pmh_stall_read_done", False)):
                logger.info(
                    "[pmh] ce skip stall paper_read (already read this plateau) same=%s",
                    int(getattr(self, "_pmh_same_subtask", 0) or 0),
                )
            elif ce and _cap > 0 and _count >= _cap:
                logger.info(
                    "[pmh] ce skip stall paper_read (episode cap %s reached) same=%s",
                    _cap,
                    int(getattr(self, "_pmh_same_subtask", 0) or 0),
                )
            else:
                query = str(self._current_subtask or self.default_subtask_prompt or "recent event")[:160]
                search_text = self.pmh_store.search_memory(query, top_k=2)
                if ce:
                    search_text = self._pmh_ce_filter_search(search_text, extra)
                extra = extra + "\n" + search_text
                resident = set(int(i) for i in memory_indices)
                sids = re.findall(r"seg_\d+", search_text)
                if not sids:
                    sids = ["last"]
                used_sid = sids[0]
                added: list[int] = []

                def _ingest(sid: str, detail: str) -> list[int]:
                    vis_text, vis_idx = self.pmh_store.retrieve_visual(segment_id=sid, detail=detail)
                    nonlocal extra
                    extra = extra + "\n" + vis_text
                    new: list[int] = []
                    for raw in vis_idx:
                        iv = int(raw)
                        if iv not in resident and iv in self.frame_store_main:
                            memory_indices.append(iv)
                            resident.add(iv)
                            new.append(iv)
                    return new

                if ce:
                    for sid in sids[:2]:
                        added = _ingest(sid, os.environ.get("PMH_INSPECT_DETAIL", "keyframes"))
                        used_sid = sid
                        if added:
                            break
                    if not added:
                        added = _ingest(used_sid, "full_visual")
                        logger.info(
                            "[pmh] ce escalate full_visual sid=%s n_added=%s resident_k=%s",
                            used_sid,
                            len(added),
                            len(self.K_indices_abs or []),
                        )
                else:
                    added = _ingest("last", os.environ.get("PMH_INSPECT_DETAIL", "keyframes"))
                    used_sid = "last"
                if ce:
                    self._pmh_stall_read_done = True
                    self._pmh_paper_read_count = _count + 1
                logger.info(
                    "[pmh] paper_read stall search+inspect query=%r added=%s n_added=%s sid=%s ce=%s segs=%s",
                    query[:80],
                    added,
                    len(added),
                    used_sid,
                    int(ce),
                    len(self.pmh_store.segments),
                )
        memory_main_frames, memory_wrist_frames, memory_indices = self._cap_memory_frames(
            get_frames_from_indices(memory_indices, self.frame_store_main),
            [self.frame_store_wrist.get(idx) for idx in memory_indices],
            memory_indices,
        )
        return memory_main_frames, memory_wrist_frames, memory_indices, extra

    def _decide_memory_access(
        self,
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Image.Image | None],
    ) -> dict[str, Any]:
        cfg = self.proactive_cfg
        store = self.episodic_store
        default = {"tool": "none", "k": cfg.visual_k, "what": "all"}
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
                    f"Current stage hint: {stage_hint}\n"
                    f"Available historical keyframes in store: {n_kf}\n"
                    f"Completed stage count: {sum(1 for e in store.stage_events if e.done)}\n\n"
                    f"{_camera_order_text(use_wrist)}\n"
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
                    '{"tool":"none"} or {"tool":"recall_semantic","what":"all"} '
                    'or {"tool":"recall_visual","k":4}.'
                ),
            }
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": memory_decision_system_prompt()}]},
            {"role": "user", "content": user_content},
        ]
        out_text = self._vlm_generate(messages, max_new_tokens=cfg.decide_max_new_tokens)
        decision = parse_memory_decision(
            out_text,
            default_k=cfg.visual_k,
            parse_fail_tool=cfg.parse_fail_tool,
        )
        decision = apply_anti_collapse(decision, store, cfg)
        store.last_tool = decision["tool"]
        store.events.append({"op": "decide", "raw": out_text.strip()[:200], **decision})
        if self.logger:
            self.logger.info("[proactive] decide tool=%s k=%s raw=%s", decision["tool"], decision["k"], out_text.strip()[:120])
        return decision

    def infer_sync(self, step_idx: int, context_frames_np: list[tuple[np.ndarray, np.ndarray | None]]) -> str:
        if not context_frames_np:
            return self._current_subtask

        recent_start = step_idx - len(context_frames_np) + 1
        context_main_frames: list[Image.Image] = []
        context_wrist_frames: list[Image.Image | None] = []
        for offset, frame_pack in enumerate(context_frames_np):
            abs_idx = recent_start + offset
            main_np, wrist_np = frame_pack
            main_img = Image.fromarray(main_np.astype(np.uint8))
            wrist_img = Image.fromarray(wrist_np.astype(np.uint8)) if self.use_wrist and wrist_np is not None else None
            self.frame_store_main[abs_idx] = main_img
            self.frame_store_wrist[abs_idx] = wrist_img
            context_main_frames.append(main_img)
            context_wrist_frames.append(wrist_img)
        self.step = max(self.step, step_idx + 1)

        if self.episodic_store is not None:
            self.episodic_store.sync_keyframes(list(self.K_indices_abs))

        mode = self.proactive_cfg.mode if self.proactive_cfg is not None else "off"
        # PMH-P: the open segment grows with every plan step. This is the frame buffer the
        # consolidator will later read; no state is written here (invariant I4).
        if self.pmh_store is not None:
            self._pmh_active_add([step_idx])
        extra_memory_text = ""
        memory_main_frames: list[Image.Image] = []
        memory_wrist_frames: list[Image.Image | None] = []
        memory_indices: list[int] = []
        ace_trace: dict[str, Any] | None = None

        if mode == "passive_samepool":
            memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
            memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
            memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
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
            # Local PrediMem screening path for ACE/PACE/Cog (rule compiler; no API tokens).
            store = self.episodic_store
            brief = self.ace.build_brief(
                task_id=int(getattr(self.task_info, "task_id", 0) or 0),
                step=step_idx,
                stall_count=self._consecutive_same_subtask,
                store=store,
            )
            manifest = self.ace.resolve_manifest(brief, api_key="", base_url="")
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
        elif mode == "proactive":
            decision = self._decide_memory_access(context_main_frames, context_wrist_frames)
            tool = decision["tool"]
            store = self.episodic_store
            if tool == "recall_semantic" and store is not None:
                extra_memory_text = store.render_semantic(what=str(decision.get("what", "all")))
                store.n_recall_semantic += 1
            elif tool == "recall_visual" and store is not None:
                idxs = store.select_visual_indices(int(decision.get("k", self.proactive_cfg.visual_k)))
                memory_indices = idxs
                memory_main_frames = get_frames_from_indices(idxs, self.frame_store_main)
                memory_wrist_frames = [self.frame_store_wrist.get(idx) for idx in idxs]
                store.n_recall_visual += 1
            else:
                if store is not None:
                    store.n_decide_none += 1
            # proactive: do not auto-inject full bank unless tool returned visual
        elif mode == "hermes_delta":
            memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
            memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
            memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
            memory_main_frames, memory_wrist_frames, memory_indices = self._cap_memory_frames(
                memory_main_frames, memory_wrist_frames, memory_indices
            )
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
                if tool == "recall_semantic" and store is not None:
                    extra_memory_text = store.render_semantic(what=str(decision.get("what", "all")))
                    store.n_recall_semantic += 1
                elif tool == "recall_visual" and store is not None:
                    extra = store.select_visual_indices(int(decision.get("k", self.proactive_cfg.visual_k)))
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
        elif mode == "pmh":
            # PMH.md §5.1 / §5.2: Task State is always on; archive frames are not. Stall
            # opens search_memory then inspect_segment once (§8 ACE fallback).
            memory_main_frames, memory_wrist_frames, memory_indices, extra_memory_text = (
                self._pmh_paper_read()
            )
            ace_trace = {
                "source": "pmh_md",
                "n_segments": len(self.pmh_store.segments) if self.pmh_store is not None else 0,
                "n_search": getattr(self.pmh_store, "n_search", 0) if self.pmh_store else 0,
                "n_inspect": getattr(self.pmh_store, "n_inspect", 0) if self.pmh_store else 0,
            }
        else:
            memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
            memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
            memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
            memory_main_frames, memory_wrist_frames, memory_indices = self._cap_memory_frames(
                memory_main_frames, memory_wrist_frames, memory_indices
            )

        messages = self._build_messages(
            memory_main_frames,
            memory_wrist_frames,
            context_main_frames,
            context_wrist_frames,
            extra_memory_text=extra_memory_text,
        )
        out_text = self._vlm_generate(messages)
        vlm_subtask, j_rel = _parse_output_no_mapping(
            out_text,
            max_pos=len(context_main_frames),
        )
        j_abs = [recent_start + (p - 1) for p in j_rel]
        prev_subtask_any = self._current_subtask

        if self.use_keyframe_memory:
            self.J_hist.append(j_abs)
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
                self.episodic_store.record_subtask(step_idx, vlm_subtask)
                if vlm_subtask == self._last_planned_subtask:
                    self._consecutive_same_subtask += 1
                else:
                    self._last_planned_subtask = vlm_subtask
                    self._consecutive_same_subtask = 1

        # ---- PMH-P commit triggers (invariant I4: the only write instant) --------------------
        # Three closures with distinct semantics (PMH.md Sec 4.1 vs 4.2):
        #   subtask_boundary  - a semantic candidate. The consolidator's verdict may answer NO
        #                       and withhold the commit, leaving the frames buffered.
        #   soft_stall        - the plan is repeating itself; offer the same candidate again.
        #   max_active_frames - a CAPACITY limit. Sec 4.1 forbids losing the current event, so
        #                       this one commits unconditionally (force_close=True).
        if self.pmh_store is not None:
            ce = self._pmh_ce_enabled()
            if vlm_subtask:
                if vlm_subtask == self._pmh_last_subtask:
                    self._pmh_same_subtask += 1
                else:
                    self._pmh_last_subtask = vlm_subtask
                    self._pmh_same_subtask = 1
                    self._pmh_stall_committed = False
                    self._pmh_stall_read_done = False
            n_active = len(self.pmh_store.active_frame_indices)
            boundary = bool(vlm_subtask) and vlm_subtask != prev_subtask_any
            max_active = int(os.environ.get("PMH_MAX_ACTIVE_FRAMES", "24"))
            soft_at = int(os.environ.get("PMH_SOFT_COMMIT_STALL", "3"))
            if boundary:
                self._pmh_commit_active(
                    step_idx,
                    prev_subtask=prev_subtask_any,
                    new_subtask=vlm_subtask,
                    reason="subtask_boundary",
                )
                self._pmh_stall_committed = False
                self._pmh_stall_read_done = False
            elif n_active > 0 and max_active > 0 and n_active > max_active:
                self._pmh_commit_active(
                    step_idx,
                    prev_subtask=prev_subtask_any,
                    new_subtask=vlm_subtask,
                    reason="max_active_frames",
                    force_close=True,
                )
            elif n_active > 0 and (
                (soft_at > 0 and self._pmh_same_subtask >= soft_at) or self._pmh_exec_stall
            ):
                if ce and self._pmh_stall_committed:
                    logger.info(
                        "[pmh] ce skip stall commit (already committed this plateau) same=%s",
                        self._pmh_same_subtask,
                    )
                else:
                    self._pmh_commit_active(
                        step_idx,
                        prev_subtask=prev_subtask_any,
                        new_subtask=vlm_subtask,
                        reason="soft_stall",
                    )
                    if ce:
                        self._pmh_stall_committed = True
            self._pmh_exec_stall = False

        self._dump_new_keyframes()
        if vlm_subtask:
            self._current_subtask = vlm_subtask
        subtask = self._current_subtask

        image_rel = None
        if self.run_dir is not None:
            image_rel = self._save_vlm_input_bundle(
                step_idx=step_idx,
                memory_main_frames=memory_main_frames,
                memory_wrist_frames=memory_wrist_frames,
                memory_indices=memory_indices,
                context_main_frames=context_main_frames,
                context_wrist_frames=context_wrist_frames,
                subtask=subtask,
            )
        trace_row = {
            "t": int(step_idx),
            "task_id": int(self.task_info.task_id),
            "subtask": subtask,
            "keyframe_positions": j_rel,
            "J_abs": j_abs,
            "K_indices_abs": list(self.K_indices_abs),
            "out_text": out_text.strip()[:600],
            "image": image_rel,
            "proactive_mode": mode,
        }
        if ace_trace is not None:
            trace_row["ace"] = ace_trace
            if self.ace is not None:
                trace_row["ace_stats"] = self.ace.stats.to_dict()
        if self.episodic_store is not None:
            trace_row["proactive_stats"] = self.episodic_store.stats()
            trace_row["proactive_last_tool"] = self.episodic_store.last_tool
        if self.pmh_store is not None:
            trace_row["pmh_stats"] = self.pmh_store.stats()
        self._append_trace(trace_row)
        if self.pmh_store is not None:
            self._dump_pmh_stats()
        if self.logger:
            self.logger.info("VLM @t=%s task=%s subtask=%s keyframes=%s", step_idx, self.task_info.task_id, subtask, j_rel)
            self.logger.info("  raw=%s", out_text.strip()[:220])
        return subtask


def _task_specs(task_id: int) -> list[stage_eval.StageSpec]:
    return stage_eval._task_specs(task_id)


def _goal_override_check(task_id: int):
    return stage_eval._goal_override_check(task_id)


def run_episode_async_stateful(
    *,
    task_id: int,
    env: Any,
    client: Any,
    planner: Any,
    args: BaseArgs,
    stage_specs: list[stage_eval.StageSpec],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override,
    vlm_camera_pose: dict | None,
    logger: logging.Logger,
    fail_on_extra_pour: bool,
    extra_pour_monitor_steps: int,
    post_goal_steps: int,
    harness: HarnessController | None = None,
    meta: MetaController | None = None,
    muscle: MuscleMemoryRing | None = None,
    bcm: BeliefContractMemory | None = None,
    hpm: HpmController | None = None,
    pact_ctrl: PactController | None = None,
) -> tuple[float, dict[str, bool], bool | None, dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    obs = env.reset()
    if harness is not None and harness.attempt_idx > 0 and harness.config.release_gripper_on_retry:
        release_gripper(env)
        logger.info("harness release_gripper after env.reset (attempt=%s)", harness.attempt_idx)
    if meta is not None:
        meta.begin_attempt()
        meta.apply_initial(planner, harness)
    if muscle is not None:
        muscle.reset()
    try:
        from harness.meta_llm import get_meta_llm_state

        _mls = get_meta_llm_state()
        if _mls is not None:
            _mls.begin_episode()
    except Exception:
        pass
    if bcm is not None:
        bcm.begin_attempt()
    if getattr(planner, "episodic_store", None) is not None:
        first_stage = stage_specs[0].name if stage_specs else ""
        if getattr(planner, "scec", None) is not None:
            planner.scec.reset(task_id=int(task_id), first_stage=first_stage)
            planner.episodic_store = planner.scec.store
        else:
            planner.episodic_store.reset()
            if stage_specs:
                planner.episodic_store.set_current_stage(stage_specs[0].name)
    if hpm is not None:
        hint = hpm.episode_start_hint(int(task_id))
        if hint:
            existing = getattr(planner, "harness_extra_context", "") or ""
            planner.harness_extra_context = "\n".join(x for x in (existing, hint) if x and x.strip())
            if logger:
                logger.info("[hpm] injected LTM hint for task=%s visit=%s", task_id, hpm.config.visit)
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    recent_vlm_frames: deque[tuple[np.ndarray, np.ndarray | None]] = deque(maxlen=args.n_recent)
    worker_error: list[str] = []
    worker_stop = threading.Event()
    vlm_job_queue: queue.Queue | None = queue.Queue(maxsize=max(1, args.vlm_queue_size)) if args.async_vlm else None
    subtask_lock = threading.Lock()
    subtask_buffer = {"value": "", "step_idx": -1}
    stage_done = {spec.name: False for spec in stage_specs}
    stage_idx = 0
    # A4: count the language-rung recoveries this episode. REWIND is the LAST rung, so it must not
    # fire until the language rung has actually been tried more than once. Job 581594 t1 fired one
    # rewind after only TWO forced replans on a task the baseline solves in ~600 steps, and finished
    # with 1 of its 2 stages (50.0 vs H/HM attempt0 = 100.0). Three is the bar: a task that needs
    # two language recoveries is still being helped by them, and rewinding there trades a real
    # repair for a lost grip.
    language_recoveries = 0
    all_stages_logged = False
    state: dict[str, Any] | None = None
    current_stage_start = 0
    current_subtask_prompt = ""
    counting_pour_task = stage_eval._is_counting_pour_task(task_id)
    drawer_task = stage_eval._is_drawer_task(task_id)
    goal_success: bool | None = None if counting_pour_task else False
    goal_reached_t: int | None = None
    extra_pour_check = stage_eval._extra_pour_check(task_id)
    extra_monitor_start_state_idx: int | None = None
    extra_monitor_deadline_t: int | None = None
    extra_pour_detected = False
    pour_1_step: int | None = None
    pour_2_step: int | None = None

    def refresh_planner_context() -> None:
        parts = [compose_planner_context(harness=harness, bcm=bcm)]
        # Soft System1: inject muscle hint only when meta allows (Normal / NOOP).
        if muscle is not None:
            allow_hint = True
            if meta is not None and getattr(meta, "config", None) is not None and meta.config.style == "cog":
                allow_hint = bool(getattr(meta, "muscle_hold_allowed", True))
            if allow_hint:
                hint = muscle.context_hint()
                if hint:
                    parts.append(hint)
        planner.harness_extra_context = "\n".join(x for x in parts if x and str(x).strip())

    def write_subtask(step_idx: int, subtask: str) -> None:
        with subtask_lock:
            subtask_buffer["value"] = subtask
            subtask_buffer["step_idx"] = step_idx

    def read_subtask() -> tuple[str, int]:
        with subtask_lock:
            return str(subtask_buffer["value"]), int(subtask_buffer["step_idx"])

    def clone_recent_frames() -> list[tuple[np.ndarray, np.ndarray | None]]:
        return [(m.copy(), w.copy() if w is not None else None) for m, w in recent_vlm_frames]

    # ---- dense keyframe store (mem_efficacy channel B) ------------------------------------
    # DISABLED BY DEFAULT: 0 leaves the call below a no-op, so the official path is unchanged.
    #
    # WHY IT EXISTS, AND IT IS THE ROOT CAUSE OF A MEASURED DEFECT. The official configuration is
    # `VLM_INTERVAL=5` with `VLM_QUEUE_SIZE=1`, and `submit_vlm_job` EVICTS the pending payload
    # whenever a new submission arrives (`queue.Full` -> `get_nowait()` -> `put_nowait`). The env
    # loop runs far faster than a reasoning planner (8-60 s per call), so most submissions never
    # reach the model: measured on job 594338, an episode of ~2470 env steps produced only 3-9
    # planner calls.
    #
    # The planner only ever receives the `n_recent`-frame window of a call it was actually given,
    # so `planner.frame_store_main` accumulated ONLY those windows -- about 35 frames, 1.4% of the
    # episode, weighted to its beginning. The keyframe bank's candidate pool was therefore a
    # byproduct of which async jobs happened to survive eviction, and `_kf_spread_union`'s "even
    # stride across the episode" degenerated into the episode's OPENING frames: 54-62% of all bank
    # slots were frames <= 8 and frame 0 was present on 100% of plan steps, on every task checked.
    # A channel whose premise is "a record of the past" had almost no past to record.
    #
    # This hook decouples the RECORD from the CALL: it stores a frame on a fixed interval whether
    # or not a planner call was submitted. Additive and gated, so no archived number moves.
    kf_store_interval = int(os.environ.get("MEM_KF_STORE_INTERVAL", "0") or 0)
    kf_store_count = {"n": 0, "skipped": 0}

    def store_kf_dense(step_idx: int) -> None:
        """Store this step's frame in the planner's keyframe store, on a fixed interval."""
        if kf_store_interval <= 0 or step_idx < 0 or not recent_vlm_frames:
            return
        if step_idx % kf_store_interval != 0:
            return
        store = getattr(planner, "frame_store_main", None)
        if store is None:
            kf_store_count["skipped"] += 1
            return
        main_np, wrist_np = recent_vlm_frames[-1]
        try:
            # Same conversion the planner itself uses for these frames, so a stored frame and a
            # context frame are the same kind of object and the bank can serve either.
            from PIL import Image as _Image

            store[int(step_idx)] = _Image.fromarray(main_np.astype("uint8"))
            wstore = getattr(planner, "frame_store_wrist", None)
            if wrist_np is not None and wstore is not None:
                wstore[int(step_idx)] = _Image.fromarray(wrist_np.astype("uint8"))
            kf_store_count["n"] += 1
        except Exception:
            kf_store_count["skipped"] += 1

    def submit_vlm_job(step_idx: int, *, force: bool = False) -> None:
        if not args.async_vlm or vlm_job_queue is None:
            return
        if step_idx < 0 or len(recent_vlm_frames) < args.n_recent:
            return
        if (not force) and args.vlm_interval > 1 and step_idx % args.vlm_interval != 0:
            return
        payload = (step_idx, clone_recent_frames(), planner.harness_extra_context)
        try:
            vlm_job_queue.put_nowait(payload)
            return
        except queue.Full:
            try:
                vlm_job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                vlm_job_queue.put_nowait(payload)
            except queue.Full:
                return

    def vlm_worker() -> None:
        assert vlm_job_queue is not None
        while not worker_stop.is_set():
            try:
                payload = vlm_job_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if payload is None:
                break
            step_idx, frames, harness_ctx = payload
            try:
                planner.harness_extra_context = harness_ctx
                subtask = planner.infer_sync(step_idx=step_idx, context_frames_np=frames)
                if subtask:
                    write_subtask(step_idx, subtask)
            except Exception as exc:
                worker_error.append(f"{type(exc).__name__}: {exc}")
                logger.error("VLM worker failed", exc_info=True)
                break

    vlm_thread = None
    if args.async_vlm:
        vlm_thread = threading.Thread(target=vlm_worker, name=f"vlm-task{planner.task_info.task_id}", daemon=True)
        vlm_thread.start()
        logger.info("async VLM enabled: single-slot subtask buffer")

    try:
        t = 0
        while t < args.max_steps + args.num_steps_wait:
            if worker_error:
                raise RuntimeError(worker_error[-1])

            if t < args.num_steps_wait:
                obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
                recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                t += 1
                submit_vlm_job(t - args.num_steps_wait)
                continue

            if state is None:
                state = stage_eval._build_initial_state(env)
                current_stage_start = state["step_idx"]

            effective_t = t - args.num_steps_wait
            if len(recent_vlm_frames) < args.n_recent:
                obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
                recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                t += 1
                submit_vlm_job(t - args.num_steps_wait)
                continue

            if args.async_vlm:
                refresh_planner_context()
                submit_vlm_job(effective_t)
                latest_subtask, latest_step = read_subtask()
            else:
                refresh_planner_context()
                latest_subtask = planner.infer_sync(effective_t, clone_recent_frames())
                latest_step = effective_t

            if harness is not None:
                override = harness.consume_subtask_override()
                if override:
                    write_subtask(latest_step if latest_step >= 0 else effective_t, override)
                    latest_subtask = override
                    logger.info("[t=%s] harness subtask override: %s", t, override)

            # System-1 muscle memory: stabilize subtask when meta allows (Normal).
            if muscle is not None and recent_vlm_frames:
                main_frame, _ = recent_vlm_frames[-1]
                muscle.update(main_frame)
                allow_hold = True
                if meta is not None and getattr(meta, "config", None) is not None and meta.config.style == "cog":
                    allow_hold = bool(getattr(meta, "muscle_hold_allowed", True))
                if allow_hold and latest_subtask:
                    held = muscle.maybe_hold_subtask(current_subtask_prompt, latest_subtask)
                    if held != latest_subtask:
                        logger.info(
                            "[t=%s] muscle hold subtask (stab=%.2f sim=%.2f)",
                            t,
                            muscle.last_stability,
                            muscle.last_similarity,
                        )
                        latest_subtask = held
                        write_subtask(latest_step if latest_step >= 0 else effective_t, held)

            if latest_subtask and latest_subtask != current_subtask_prompt:
                current_subtask_prompt = latest_subtask
                logger.info("[t=%s] VLM prompt update from step=%s: %s", t, latest_step, current_subtask_prompt)
                if harness is not None:
                    harness.on_subtask_update(latest_step, current_subtask_prompt)

            if harness is not None and harness.consume_force_replan():
                refresh_planner_context()
                if hasattr(planner, "note_pmh_exec_stall"):
                    planner.note_pmh_exec_stall()
                submit_vlm_job(effective_t, force=True)
                language_recoveries += 1
                logger.info("[t=%s] harness forced VLM replan after stall (pmh visual retrieve armed)", t)

            if harness is not None and harness.consume_evidence_gate():
                planner.hermes_inquiry_pending = True
                logger.info("[t=%s] CGMH evidence-gate armed Inquiry Δ", t)

            # PACT: the physical half of the recovery ladder. Until this hook existed, an
            # in-attempt stall could only change a STRING (measured: stall 199 ->
            # subtask_override 172, with `override_vla_prompt` returning base_prompt unchanged
            # and `release_gripper` reachable only after `env.reset()`). REWIND restores the
            # robot's qpos/qvel to this stage's start and leaves the SCENE alone, so a jammed
            # arm is traded for a clean re-approach instead of for the whole episode.
            #
            # A4: REWIND now gates on three failure predicates, not a timer. The old call
            # `pact_ctrl.tick(env, t=t, stage_idx=stage_idx)` had no failure signal, so it
            # fired on a pure clock — 18 rewinds / 6 eps = 3.0/ep, zero variance (job 581446).
            # The three gates:
            #   stage_failing    : the current stage's check_fn has NOT passed and `after` steps
            #                      have elapsed since stage start.
            #   gripper_empty    : rewinding while clamped drags the scene.
            #   language_tried   : harness has already fired force_replan + subtask_override.
            if pact_ctrl is not None and pact_ctrl.enabled and state is not None:
                _elapsed = (t - current_stage_start) if current_stage_start is not None else 0
                _stage_failing = (
                    stage_idx < len(stage_specs)
                    and not stage_done.get(stage_specs[stage_idx].name, False)
                    and _elapsed >= pact_ctrl.after
                )
                _gripper_empty = True  # default: assume not clamped; refined below
                try:
                    _gq = float(env.sim.data.qpos[env.sim.model.dof_qposadr[0]])
                    _gripper_empty = _gq < 0.02  # gripper open (unclamped) threshold
                except Exception:
                    pass
                _language_tried = (
                    harness is not None
                    and getattr(harness, "stall_triggered", False)
                    and language_recoveries >= 3
                )
                # A5: when the rolling ladder has PROVEN the language rung to be a no-op
                # (`_after_stall_recovery` saw `_handle_stall` set no override), the third
                # predicate is not "language was tried N times" but "the language rung had
                # nothing to offer". That is a real failure predicate, so it replaces the
                # `language_recoveries >= 3` counter -- which on t22 could never be reached
                # anyway, because the one-shot latch meant only ONE recovery offer existed in
                # the whole 2500-step stage.
                if harness is not None and harness.hard_recovery_armed():
                    if getattr(harness, "stall_triggered", False):
                        _language_tried = True
                    # TV (HarnessVLA's Transition Verifier): rather than REFUSING to rewind
                    # because the gripper is clamped, apply the forced action it prescribes.
                    # `rewind` already releases before restoring, so opening here is the
                    # physical half of the transition and is what makes the gate actionable
                    # instead of a silent no-op.
                    if not _gripper_empty:
                        try:
                            release_gripper(env, steps=8)
                            _gripper_empty = True
                            if harness is not None:
                                harness.note_tv_release()
                            logger.info("[t=%s] hard-recovery: forced release_gripper (TV)", t)
                        except Exception:
                            logger.exception("hard-recovery: forced release_gripper failed")
                _rewound = pact_ctrl.tick(
                    env,
                    t=t,
                    stage_idx=stage_idx,
                    stage_failing=_stage_failing,
                    gripper_empty=_gripper_empty,
                    language_tried=_language_tried,
                )
                if harness is not None:
                    harness.note_hard_recovery(bool(_rewound))
                if _rewound:
                    new_obs = pact_ctrl.refresh_obs(env)
                    if new_obs is not None:
                        obs = new_obs
                        stage_eval._update_state(obs, state)
                        recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                    # The VLA chunk about to be planned is stale - it was computed for the pose we
                    # just discarded - so ask for a fresh replan rather than replaying it.
                    if harness is not None:
                        harness.consume_force_replan()
                    refresh_planner_context()
                    submit_vlm_job(effective_t, force=True)

            prompt_for_vla = current_subtask_prompt or planner.default_subtask_prompt
            if harness is not None:
                prompt_for_vla = harness.override_vla_prompt(
                    prompt_for_vla,
                    stage_idx=stage_idx,
                    stage_specs=stage_specs,
                    stage_done=stage_done,
                    step=t,
                )
            element = obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
            out = client.infer(element)
            actions = np.asarray(out["actions"])
            logger.info("[t=%s] VLA chunk prompt=%s", t, prompt_for_vla)

            for action in actions[: args.replan_steps]:
                element_step = obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
                replay.append(element_step["observation/image"])
                wrist = element_step.get("observation/wrist_image")
                if wrist is not None:
                    replay_wrist.append(wrist)

                obs, _, done, _ = env.step(action.tolist())
                recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                # Dense keyframe store. Placed HERE, next to the frame extraction, so the record is
                # driven by the env loop rather than by whether a planner call survived the queue.
                store_kf_dense(t - args.num_steps_wait)
                if state is not None:
                    stage_eval._update_state(obs, state)
                t += 1
                submit_vlm_job(t - args.num_steps_wait)

                if state is not None and stage_idx < len(stage_specs):
                    spec = stage_specs[stage_idx]
                    if spec.check_fn(env, state, current_stage_start):
                        stage_done[spec.name] = True
                        logger.info("[t=%s] stage done: %s", t, spec.name)
                        if spec.name.endswith("_Pour_One"):
                            pour_1_step = t
                        elif spec.name.endswith("_Pour_Two"):
                            pour_2_step = t
                            if counting_pour_task and fail_on_extra_pour:
                                extra_monitor_start_state_idx = int(state["step_idx"])
                                extra_monitor_deadline_t = t + extra_pour_monitor_steps
                                logger.info(
                                    "[t=%s] extra-pour monitor started; deadline=%s",
                                    t,
                                    extra_monitor_deadline_t,
                                )
                        stage_idx += 1
                        current_stage_start = state["step_idx"]
                        if planner.memory_system_config.stage_anchor:
                            planner.pin_keyframe(int(current_stage_start))
                        if getattr(planner, "pmh_store", None) is not None and hasattr(
                            planner, "pmh_note_stage_done"
                        ):
                            planner.pmh_note_stage_done(spec.name, int(t))
                        # v0.15: tell PMH which stage is now ACTIVE (the occlusion gate keys
                        # on the active stage's target object, not the last completed one).
                        if (
                            getattr(planner, "pmh_store", None) is not None
                            and hasattr(planner, "pmh_note_stage_start")
                            and stage_idx < len(stage_specs)
                        ):
                            planner.pmh_note_stage_start(stage_specs[stage_idx].name, int(t))
                        if getattr(planner, "scec", None) is not None:
                            next_stage = stage_specs[stage_idx].name if stage_idx < len(stage_specs) else ""
                            planner.scec.on_stage_done(t, spec.name, next_stage=next_stage)
                        elif getattr(planner, "episodic_store", None) is not None:
                            planner.episodic_store.record_stage(t, spec.name, done=True)
                            if stage_idx < len(stage_specs):
                                planner.episodic_store.set_current_stage(stage_specs[stage_idx].name)

                if harness is not None and state is not None:
                    harness.on_stage_progress(
                        step=t,
                        stage_idx=stage_idx,
                        stage_specs=stage_specs,
                        subtask=current_subtask_prompt or planner.default_subtask_prompt,
                    )
                if bcm is not None and state is not None:
                    changed = bcm.step(
                        step=t,
                        stage_idx=stage_idx,
                        stage_specs=stage_specs,
                        subtask=current_subtask_prompt or planner.default_subtask_prompt,
                        planner=planner,
                    )
                    if changed:
                        refresh_planner_context()
                if meta is not None and state is not None:
                    muscle_stab = float(muscle.last_stability) if muscle is not None else None
                    meta.step(
                        step=t,
                        stage_idx=stage_idx,
                        subtask=current_subtask_prompt or planner.default_subtask_prompt,
                        planner=planner,
                        harness=harness,
                        planner_stall=int(getattr(planner, "_consecutive_same_subtask", 0) or 0),
                        progress_score=None,
                        muscle_stability=muscle_stab,
                    )

                if stage_idx >= len(stage_specs) and not all_stages_logged:
                    logger.info("[t=%s] all stages done", t)
                    all_stages_logged = True

                if (
                    counting_pour_task
                    and fail_on_extra_pour
                    and extra_pour_check is not None
                    and extra_monitor_start_state_idx is not None
                    and extra_monitor_deadline_t is not None
                    and pour_2_step is not None
                    and pour_2_step < t <= extra_monitor_deadline_t
                    and extra_pour_check(env, state, extra_monitor_start_state_idx)
                ):
                    extra_pour_detected = True
                    logger.info("[t=%s] third pour detected; episode failed", t)
                    raise StopIteration

                if not counting_pour_task and stage_eval._stage_success_from_stage_done(task_id, stage_done):
                    goal_success = True
                    logger.info("[t=%s] required stages done", t)
                    raise StopIteration

                all_stages_complete = bool(stage_done) and all(stage_done.values())
                extra_monitor_complete = (
                    not fail_on_extra_pour
                    or (
                        extra_monitor_deadline_t is not None
                        and t >= extra_monitor_deadline_t
                    )
                )
                if counting_pour_task and all_stages_complete and extra_monitor_complete:
                    raise StopIteration
                if done or t >= args.max_steps + args.num_steps_wait:
                    raise StopIteration
    except StopIteration:
        pass
    except Exception:
        logger.exception("episode failed")
    finally:
        if args.async_vlm and vlm_job_queue is not None:
            worker_stop.set()
            try:
                vlm_job_queue.put_nowait(None)
            except queue.Full:
                pass
            if vlm_thread is not None and vlm_thread.is_alive():
                vlm_thread.join(timeout=3.0)

    stage_pct = stage_eval._stage_score_pct(task_id, stage_done)
    all_stages_complete = bool(stage_done) and all(stage_done.values())
    extra_monitor_complete = (
        not fail_on_extra_pour
        or (
            extra_monitor_deadline_t is not None
            and t >= extra_monitor_deadline_t
        )
    )
    required_stages_complete = stage_eval._stage_success_from_stage_done(task_id, stage_done)
    stage_success = required_stages_complete and (
        not counting_pour_task
        or (extra_monitor_complete and not extra_pour_detected)
    )
    if extra_pour_detected:
        failure_reason = "extra_pour"
    elif not stage_success:
        failure_reason = "incomplete_stage"
    elif counting_pour_task and not extra_monitor_complete:
        failure_reason = "monitor_incomplete"
    else:
        failure_reason = None
    diagnostics = {
        "stage_success": bool(stage_success),
        "extra_pour_detected": bool(extra_pour_detected),
        "pour_1_step": pour_1_step,
        "pour_2_step": pour_2_step,
        "extra_monitor_end_step": (
            extra_monitor_deadline_t
            if extra_monitor_deadline_t is not None and t >= extra_monitor_deadline_t
            else None
        ),
        "failure_reason": failure_reason,
    }
    if harness is not None:
        diagnostics.update(harness.diagnostics)
    if meta is not None:
        diagnostics["meta"] = meta.summary()
    if bcm is not None:
        diagnostics["bcm"] = bcm.summary()
    goal_success = stage_success
    return stage_pct, stage_done, goal_success, diagnostics, replay, replay_wrist


def patch_env_resolution() -> None:
    base_env = ec._get_env_class()
    orig_init = base_env.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["camera_heights"] = 480
        kwargs["camera_widths"] = 640
        return orig_init(self, *args, **kwargs)

    base_env.__init__ = patched_init
    ec._get_env_class = lambda: base_env


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    patch_env_resolution()

    out_root = Path(os.environ["OUT_ROOT"])
    video_root = Path(os.environ["VIDEO_DIR"])
    save_video = os.environ.get("SAVE_VIDEO", "1").strip().lower() not in {"0", "false", "no", "off"}
    summary_json = Path(os.environ["SUMMARY_JSON"])
    summary_tsv = Path(os.environ["SUMMARY_TSV"])
    prompt_trace_tsv = Path(os.environ.get("PROMPT_TRACE_TSV", str(out_root / "prompt_trace.tsv")))
    task_config = Path(os.environ.get("TASK_CONFIG", str(REFERENCE_DIR / "fullvlm_v2_26_memory_tasks.json")))
    task_infos = load_task_infos(task_config)
    tasks = [int(x) for x in json.loads(os.environ.get("TASKS_JSON", json.dumps(list(range(1, 27)))))]

    args = BaseArgs()
    args.host = os.environ.get("HOST", "127.0.0.1")
    args.port = int(os.environ.get("PORT", "8026"))
    args.base_model_dir = os.environ["VLM_CKPT"]
    args.lora_path = os.environ.get("VLM_LORA_PATH", "none")
    args.vlm_device = os.environ.get("VLM_DEVICE", "cuda:1")
    args.resize_size = int(os.environ.get("RESIZE_SIZE", "256"))
    args.replan_steps = int(os.environ.get("REPLAN_STEPS", "10"))
    args.num_steps_wait = int(os.environ.get("NUM_STEPS_WAIT", "10"))
    args.max_steps = int(os.environ.get("MAX_STEPS", "2500"))
    args.seed = int(os.environ.get("SEED", "100"))
    args.num_trials_per_task = int(os.environ.get("NUM_TRIALS", "1"))
    args.vlm_input_profile = os.environ.get("VLM_INPUT_PROFILE", "fullvlm_256")
    args.vlm_match_training_jpeg_roundtrip = os.environ.get("VLM_MATCH_TRAINING_JPEG_ROUNDTRIP", "0") in {"1", "true", "yes"}
    args.vlm_training_jpeg_quality = int(os.environ.get("VLM_TRAINING_JPEG_QUALITY", "30"))
    args.async_vlm = os.environ.get("ASYNC_VLM", "1") in {"1", "true", "yes"}
    args.vlm_interval = int(os.environ.get("VLM_INTERVAL", "5"))
    args.vlm_queue_size = int(os.environ.get("VLM_QUEUE_SIZE", "1"))
    args.n_recent = int(os.environ.get("N_RECENT", "5"))
    args.k_max = int(os.environ.get("K_MAX", "0"))
    args.d_merge = int(os.environ.get("D_MERGE", "6"))
    args.vlm_use_wrist = os.environ.get("VLM_USE_WRIST", "1") in {"1", "true", "yes"}
    args.vlm_use_keyframe_memory = os.environ.get("VLM_USE_KEYFRAME_MEMORY", "1") in {"1", "true", "yes"}
    fail_on_extra_pour = os.environ.get("FAIL_ON_EXTRA_POUR", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    extra_pour_monitor_steps = int(os.environ.get("POST_STAGE_STEPS", os.environ.get("EXTRA_POUR_MONITOR_STEPS", "30")))
    post_goal_steps = int(os.environ.get("POST_GOAL_STEPS", "200"))
    harness_config = load_harness_config()
    meta_ctrl = MetaController.create()
    muscle_ctrl = MuscleMemoryRing.create()
    bcm_ctrl = BeliefContractMemory.create()
    hpm_ctrl = HpmController.create()
    proactive_cfg = ProactiveMemoryConfig.from_env()
    if hpm_ctrl is not None:
        logging.info(
            "HpmController enabled: visit=%s bank=%s csr_threshold=%.1f inject=%s",
            hpm_ctrl.config.visit,
            hpm_ctrl.config.bank_path,
            hpm_ctrl.config.csr_threshold,
            hpm_ctrl.config.inject_mode,
        )
    if meta_ctrl is not None:
        logging.info(
            "MetaController enabled: style=%s initial_steps=%s recover_until_progress=%s "
            "error_hold=%s respect_skip=%s k_max=%s stall=%s cog_llm=%s",
            meta_ctrl.config.style,
            meta_ctrl.config.initial_steps,
            meta_ctrl.config.recover_until_progress,
            meta_ctrl.config.error_hold_steps,
            meta_ctrl.config.respect_skip_override,
            meta_ctrl.config.k_max if meta_ctrl.config.style != "ladder" else meta_ctrl.config.k_max_dense,
            meta_ctrl.config.stall_steps if meta_ctrl.config.style != "ladder" else meta_ctrl.config.stall_nominal,
            meta_ctrl.config.cog_meta_llm,
        )
    if muscle_ctrl is not None:
        logging.info(
            "MuscleMemory enabled: window=%s decay=%.2f hold_thr=%.2f hard_hold=%s",
            muscle_ctrl.config.window,
            muscle_ctrl.config.decay,
            muscle_ctrl.config.hold_threshold,
            muscle_ctrl.config.hard_hold,
        )
    if meta_ctrl is not None and meta_ctrl.config.cog_meta_llm:
        try:
            from harness.meta_llm import get_meta_llm_state

            mls = get_meta_llm_state()
            if mls is not None:
                logging.info(
                    "CogMeta LLM ready: backend=%s model=%s ep_cap=%s global_cap=%s",
                    mls.backend,
                    mls.model,
                    mls.max_calls_per_episode,
                    mls.max_calls_global,
                )
            else:
                logging.warning("CogMeta LLM enabled but state unavailable; rules only")
        except Exception as exc:  # noqa: BLE001
            logging.warning("CogMeta LLM warmup failed: %s", exc)
    if bcm_ctrl is not None:
        logging.info(
            "BeliefContractMemory enabled: stall_steps=%s max_active=%s",
            bcm_ctrl.config.stall_steps,
            bcm_ctrl.config.max_active_render,
        )
    if proactive_cfg.enabled:
        logging.info(
            "ProactiveMemory mode=%s max_tools=%s visual_k=%s",
            proactive_cfg.mode,
            proactive_cfg.max_tools,
            proactive_cfg.visual_k,
        )
    if harness_config.enabled:
        default_rules = REFERENCE_DIR.parent / "harness" / "global_rules.json"
        if harness_config.global_rules_path is None and default_rules.is_file():
            harness_config = replace(harness_config, global_rules_path=default_rules)
        logging.info(
            "Harness v2: max_retries=%s stall_steps=%s vlm_context=%s vla_hints=%s api_planner=%s smart_retry=%s skip_score=%s",
            harness_config.max_episode_retries,
            harness_config.stall_step_threshold,
            harness_config.inject_vlm_context,
            harness_config.inject_vla_hints,
            harness_config.api_planner_enable,
            harness_config.smart_retry,
            harness_config.retry_skip_score_pct,
        )
    _apply_vlm_input_profile(args)

    out_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    prompt_trace_tsv.write_text(
        "task_id\ttrial\tseed\tvlm_ckpt\tvla_prompt_last\tstage_success\tgoal_success\t"
        "stage_score_pct\textra_pour_detected\tfailure_reason\n",
        encoding="utf-8",
    )
    summary_tsv.write_text(
        "task_id\tstatus\terror\tstage_score_pct\tstage_success_rate\tgoal_success_rate\t"
        "video_dir\tduration_sec\n",
        encoding="utf-8",
    )

    _seed_everywhere(args.seed)
    client = StableWebsocketClientPolicy(args.host, args.port, ping_interval=None, ping_timeout=None, close_timeout=30.0)
    first_task = task_infos[tasks[0]]
    planner_backend = os.environ.get("PLANNER_BACKEND", "local").strip().lower()
    if planner_backend in {"api", "qwen_api", "openai_api"}:
        planner = ApiMemoryPlanner(
            task_info=first_task,
            system_prompt=SYSTEM_PROMPT_MEMORY,
            n_recent=args.n_recent,
            d_merge=args.d_merge,
            k_max=args.k_max,
            use_keyframe_memory=args.vlm_use_keyframe_memory,
            use_wrist=args.vlm_use_wrist,
            logger=None,
        )
        logging.info(
            "Planner backend=api model=%s base_url=%s",
            planner.api_model,
            planner.api_base_url,
        )
    else:
        planner = FullVlm26MemoryPlanner(
            base_model_dir=args.base_model_dir,
            lora_path=args.lora_path,
            instruction="",
            system_prompt=SYSTEM_PROMPT_MEMORY,
            prompt_profile="task1_kf5",
            n_recent=args.n_recent,
            d_merge=args.d_merge,
            k_max=args.k_max,
            use_keyframe_memory=args.vlm_use_keyframe_memory,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "256")),
            device=args.vlm_device,
            logger=None,
            vlm_model_type=args.vlm_model_type,
            enable_thinking=False,
            crop_right_half=False,
            use_wrist=args.vlm_use_wrist,
            task_info=first_task,
        )

    # PACT: the physical actuator. Created once (the dof layout is per-sim, not per-episode) and
    # handed to the planner so its counters land in the same durable `pmh_episode_stats.json`
    # the sbatch census reads. `bind()` FAILS SAFE: if no joints match the robot prefix it
    # disarms and logs, rather than reporting rewinds it never performed.
    pact_ctrl = PactController.from_env()
    if pact_ctrl.enabled:
        planner.pact_controller = pact_ctrl
        logging.info("PACT enabled: after=%s max=%s prefix=%s", pact_ctrl.after, pact_ctrl.max_rewinds, pact_ctrl.prefix)
    else:
        planner.pact_controller = None
        logging.info("PACT disabled (PACT_REWIND unset)")

    results = []
    for task_id in tasks:
        task_info = task_infos[task_id]
        planner.set_task_info(task_info)
        bddl_path = ec._resolve_bddl_path(task_id)
        stage_specs = _task_specs(task_id)
        counting_pour_task = stage_eval._is_counting_pour_task(task_id)
        goal_monitor_dict = {} if counting_pour_task else ec._build_goal_monitor_dict(bddl_path)
        goal_check_override = _goal_override_check(task_id)
        task_video = video_root / f"task{task_id}"
        task_video.mkdir(parents=True, exist_ok=True)
        task_root = out_root / f"task{task_id}"
        task_root.mkdir(parents=True, exist_ok=True)
        harness_ctrl = None
        if harness_config.enabled:
            harness_ctrl = HarnessController.create(
                task_id=task_id,
                task_info=task_info,
                config=harness_config,
                memory_root=out_root / "harness_memory",
            )
            # rho (SEAM): hand the controller the SAME store instance the planner writes to, so
            # the ladder's trigger reads the evidence rather than the planner's phrasing.
            #
            # Wired here, from the planner's live store, rather than built independently, because
            # a second store would be empty and the ladder would dedupe every rung against a
            # residual that never moves -- a silent, self-consistent failure of exactly the kind
            # that has cost this project several runs. When the layer is off (`store.seam` is
            # None) the controller keeps its previous text-based accounting, so an arm without
            # SEAM behaves exactly as before.
            _seam_store = getattr(getattr(planner, "pmh_store", None), "seam", None)
            if _seam_store is not None:
                harness_ctrl.seam_store = _seam_store
                logging.info("[seam] rho attached to the harness ladder (evidence-based rungs)")
            # Wire the harness controller onto the planner so the census can read its A1/A2/A3
            # α-guard counters and its A5 recovery-ladder counters into the durable
            # `pmh_episode_stats.json`. Without this link the counters are live in the controller
            # but never reach the file the sbatch falsifiers read.
            #
            # This assignment used to sit behind `if hasattr(planner, "harness_controller")`, and
            # that guard silently disabled the whole thing on the LOCAL planner: `FullVlm26MemoryPlanner`
            # never declares a `harness_controller` attribute, so `hasattr` was always False and the
            # assignment was always skipped. Measured consequence (job 582641): its
            # `pmh_episode_stats.json` contains `n_pact_rewind=3` (attached unconditionally at
            # `planner.pact_controller = pact_ctrl` below, which is why PACT shows up) but has NO
            # `n_language_noop`, `n_recovery_escalation`, `n_tv_release`, `noop_streak` or
            # `n_actuator_*` keys at all -- so the A5 and A1 falsifiers were reading the absence of a
            # wiring as the absence of a mechanism, and would have called a firing ladder "inert".
            # The API planner declares the attribute, which is why only API runs ever reported these.
            # Assign unconditionally: setting an attribute on the planner is always legal.
            planner.harness_controller = harness_ctrl
        status = "completed"
        err = ""
        st = time.time()
        stage_sum = 0.0
        goal_cnt = 0
        stage_success_cnt = 0

        try:
            env_cls = ec._get_env_class()
            env = env_cls(
                bddl_file_name=str(bddl_path),
                camera_heights=480,
                camera_widths=640,
                ignore_done=True,
                reward_shaping=True,
                control_freq=20,
                initialization_noise=None,
            )
            # Resolve the robot's dof indices against THIS task's sim: LIBERO adds joints per
            # scene, so the slice must be computed from the live model rather than assumed.
            pact_ctrl.bind(env)
            for ep in tqdm.tqdm(range(args.num_trials_per_task), desc=f"task{task_id}"):
                seed = args.seed + ep
                _seed_everywhere(seed)
                pact_ctrl.begin_episode()
                try:
                    env.seed(seed)
                except AttributeError:
                    pass
                run_dir = task_root / f"ep{ep}"
                ep_logger = make_episode_logger(run_dir)
                ep_logger.info("task_id=%s bddl=%s vlm_ckpt=%s", task_id, bddl_path, args.base_model_dir)

                best_stage_pct = -1.0
                best_stage_done: dict[str, bool] = {spec.name: False for spec in stage_specs}
                best_goal_success: bool | None = False
                best_diagnostics: dict[str, Any] = {}
                best_replay: list[np.ndarray] = []
                best_replay_wrist: list[np.ndarray] = []
                best_attempt = 0
                # v0.14: per-attempt outcome history for duplicate-stall detection.
                attempt_history: list[dict[str, Any]] = []

                total_attempts = (
                    harness_ctrl.total_attempts_for_task()
                    if harness_ctrl is not None
                    else (harness_config.total_attempts if harness_config.enabled else 1)
                )
                for attempt in range(total_attempts):
                    if harness_ctrl is not None:
                        harness_ctrl.begin_attempt(attempt)
                    pmh_snap = None
                    keep_pmh = os.environ.get("PMH_KEEP_STORE_ON_RETRY", "1").strip().lower() not in {
                        "0",
                        "false",
                        "no",
                        "off",
                    }
                    if (
                        attempt > 0
                        and keep_pmh
                        and hasattr(planner, "snapshot_pmh_memory")
                        and getattr(planner, "pmh_store", None) is not None
                    ):
                        pmh_snap = planner.snapshot_pmh_memory()
                    if getattr(planner, "pmh_store", None) is not None or hasattr(
                        planner, "_pmh_hold_ledger"
                    ):
                        planner._pmh_hold_ledger = bool(attempt > 0 and keep_pmh)
                    planner.reset_episode(instruction="", run_dir=run_dir / f"attempt{attempt}", logger=ep_logger)
                    if pmh_snap is not None and hasattr(planner, "restore_pmh_memory"):
                        planner.restore_pmh_memory(pmh_snap)
                    if (
                        attempt > 0
                        and hasattr(planner, "pmh_note_retry_brief")
                        and getattr(planner, "pmh_store", None) is not None
                    ):
                        # v0.10: tell the planner what the previous attempt achieved/stalled
                        # so retry plans focus repair on the first incomplete stage.
                        completed = [sp.name for sp in stage_specs if best_stage_done.get(sp.name)]
                        stalled = next(
                            (sp.name for sp in stage_specs if not best_stage_done.get(sp.name)),
                            None,
                        )
                        # v0.14: detect "same stage, same score" repeated stall across the
                        # last two attempts → retry must diverge, not replay.
                        repeated_stall = False
                        diverge_on = os.environ.get("PMH_RETRY_DIVERGE", "0").strip().lower() not in {
                            "0",
                            "false",
                            "no",
                            "off",
                        }
                        if diverge_on and len(attempt_history) >= 2:
                            a2, a1 = attempt_history[-1], attempt_history[-2]
                            if (
                                a1.get("stalled") is not None
                                and a1.get("stalled") == a2.get("stalled")
                                and a1.get("stage_pct") == a2.get("stage_pct")
                            ):
                                repeated_stall = True
                        planner.pmh_note_retry_brief(
                            attempt_idx=attempt,
                            stalled_stage=stalled,
                            completed_stages=completed,
                            repeated_stall=repeated_stall,
                        )
                    stage_pct, stage_done, goal_success, diagnostics, replay, replay_wrist = run_episode_async_stateful(
                        task_id=task_id,
                        env=env,
                        client=client,
                        planner=planner,
                        args=args,
                        stage_specs=stage_specs,
                        goal_monitor_dict=goal_monitor_dict,
                        goal_check_override=goal_check_override,
                        vlm_camera_pose=None,
                        logger=ep_logger,
                        fail_on_extra_pour=fail_on_extra_pour,
                        extra_pour_monitor_steps=extra_pour_monitor_steps,
                        post_goal_steps=post_goal_steps,
                        harness=harness_ctrl,
                        meta=meta_ctrl,
                        muscle=muscle_ctrl,
                        bcm=bcm_ctrl,
                        hpm=hpm_ctrl,
                        pact_ctrl=pact_ctrl,
                    )
                    if harness_ctrl is not None:
                        harness_ctrl.on_episode_end(
                            stage_score_pct=stage_pct,
                            stage_success=bool(diagnostics.get("stage_success")),
                            failure_reason=diagnostics.get("failure_reason"),
                            stage_done=stage_done,
                            run_dir=run_dir / f"attempt{attempt}",
                        )
                    if stage_pct > best_stage_pct or (
                        stage_pct == best_stage_pct and diagnostics.get("stage_success")
                    ):
                        best_stage_pct = stage_pct
                        best_stage_done = dict(stage_done)
                        best_goal_success = goal_success
                        best_diagnostics = dict(diagnostics)
                        best_replay = replay
                        best_replay_wrist = replay_wrist
                        best_attempt = attempt
                    ep_logger.info(
                        "attempt=%s stage_score=%.1f stage_success=%s failure_reason=%s",
                        attempt,
                        stage_pct,
                        diagnostics.get("stage_success"),
                        diagnostics.get("failure_reason"),
                    )
                    # v0.14: record this attempt's outcome for duplicate-stall detection.
                    attempt_history.append(
                        {
                            "attempt_idx": attempt,
                            "stage_pct": float(stage_pct),
                            "stalled": next(
                                (sp.name for sp in stage_specs if not stage_done.get(sp.name)),
                                None,
                            ),
                            "completed": [sp.name for sp in stage_specs if stage_done.get(sp.name)],
                            "failure_reason": diagnostics.get("failure_reason"),
                        }
                    )
                    if harness_ctrl is None or not harness_ctrl.should_retry(
                        bool(diagnostics.get("stage_success")),
                        stage_pct,
                    ):
                        break

                stage_pct = best_stage_pct
                stage_done = best_stage_done
                goal_success = best_goal_success
                diagnostics = best_diagnostics
                replay = best_replay
                replay_wrist = best_replay_wrist
                if harness_ctrl is not None:
                    diagnostics = dict(diagnostics)
                    diagnostics["harness_total_attempts"] = harness_ctrl.attempt_idx + 1
                stage_sum += stage_pct
                stage_success_cnt += int(diagnostics["stage_success"])
                goal_cnt += stage_pct / 100.0
                base_name = ec.get_video_basename(
                    task_id,
                    ep,
                    seed,
                    diagnostics["stage_success"],
                )
                stages_str = " | ".join(f"{k}={'Y' if v else 'N'}" for k, v in stage_done.items())
                ep_logger.info(
                    "Episode %s seed=%s stage_score=%.1f stage_success=%s goal=%s failure_reason=%s | %s",
                    ep,
                    seed,
                    stage_pct,
                    int(diagnostics["stage_success"]),
                    f"{stage_pct / 100.0:.3f}",
                    diagnostics["failure_reason"],
                    stages_str,
                )
                with prompt_trace_tsv.open("a", encoding="utf-8") as f:
                    goal_text = f"{stage_pct / 100.0:.4f}"
                    f.write(
                        f"{task_id}\t{ep}\t{seed}\t{args.base_model_dir}\t\t"
                        f"{int(diagnostics['stage_success'])}\t{goal_text}\t{stage_pct:.1f}\t"
                        f"{int(diagnostics['extra_pour_detected'])}\t{diagnostics['failure_reason']}\n"
                    )
                if save_video and replay:
                    try:
                        _write_video(task_video / f"{base_name}.mp4", replay, fps=10)
                    except Exception:
                        ep_logger.exception("Failed to write main video for %s", base_name)
                if save_video and replay_wrist:
                    try:
                        _write_video(task_video / f"{base_name}_wrist.mp4", replay_wrist, fps=10)
                    except Exception:
                        ep_logger.exception("Failed to write wrist video for %s", base_name)
                if hpm_ctrl is not None and status == "completed":
                    attempt_dir = run_dir / f"attempt{best_attempt}"
                    trace_path = find_latest_trace(attempt_dir)
                    hpm_ctrl.consolidate_task(
                        task_id=int(task_id),
                        seed=int(seed),
                        csr=float(stage_pct),
                        stage_done=dict(stage_done),
                        trace_path=trace_path,
                    )
            env.close()
        except Exception as exc:
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        n = max(1, args.num_trials_per_task)
        stage_score = stage_sum / n
        stage_success_rate = stage_success_cnt / n
        goal_success_rate = goal_cnt / n
        dur = round(time.time() - st, 2)
        row = {
            "task_id": task_id,
            "status": status,
            "error": err,
            "stage_score_pct": stage_score,
            "stage_success_rate": stage_success_rate,
            "goal_success_rate": goal_success_rate,
            "video_dir": str(task_video),
            "duration_sec": dur,
        }
        results.append(row)
        summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        with summary_tsv.open("a", encoding="utf-8") as f:
            goal_text = f"{goal_success_rate:.4f}"
            f.write(
                f"{task_id}\t{status}\t{err.replace(chr(9), ' ')}\t{stage_score:.1f}\t"
                f"{stage_success_rate:.4f}\t{goal_text}\t{task_video}\t{dur}\n"
            )
        logging.info(
            "task=%s status=%s stage_score=%.1f stage_success=%.3f goal=%s",
            task_id,
            status,
            stage_score,
            stage_success_rate,
            goal_text,
        )

    planner.close()
    completed = [r for r in results if r["status"] == "completed"]
    aggregate = {
        "macro_stage_score_pct": sum(r["stage_score_pct"] for r in completed) / max(1, len(completed)),
        "macro_stage_success_rate": sum(r["stage_success_rate"] for r in completed) / max(1, len(completed)),
        "macro_goal_success_rate": sum(r["goal_success_rate"] for r in completed) / max(1, len(completed)),
        "num_tasks": len(results),
        "num_goal_scored_tasks": len(completed),
        "harness_enabled": harness_config.enabled,
    }
    if harness_config.enabled:
        aggregate["harness_max_retries"] = harness_config.max_episode_retries
        aggregate["harness_stall_steps"] = harness_config.stall_step_threshold
        aggregate["harness_vlm_context"] = harness_config.inject_vlm_context
        aggregate["harness_vla_hints"] = harness_config.inject_vla_hints
        aggregate["harness_api_planner"] = harness_config.api_planner_enable
    (out_root / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("done aggregate=%s summary=%s", aggregate, summary_tsv)


if __name__ == "__main__":
    main()
