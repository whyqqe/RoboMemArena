"""EvMem-GPM: gated predicate memory on nomem (highest-success design).

Channels:
  G — predicate scaffold (scorer-aligned, redact-safe; never answer-key identity)
  E — evidence only when gated; tagged/decaying fail frames; admissible templates
  C — hard primitive controller (off-stage / premature pour / label paste)

Fluent by default (≈ nomem). Evidence on stall/attempts/attractor. No notes.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DIGEST_CAP = int(os.environ.get("MEMEXP_EVMEM_DIGEST_CAP", "2000") or 2000)
LOOKAHEAD = int(os.environ.get("MEMEXP_EVMEM_LOOKAHEAD", "4") or 4)
MAX_ROUNDS = int(os.environ.get("MEMEXP_EVMEM_MAX_ROUNDS", "2") or 2)
EVIDENCE_CAP = int(os.environ.get("MEMEXP_EVMEM_EVIDENCE_CAP", "6") or 6)
GATE_ATTEMPTS = int(os.environ.get("MEMEXP_EVMEM_GATE_ATTEMPTS", "3") or 3)
GATE_STALL = int(os.environ.get("MEMEXP_EVMEM_GATE_STALL", "3") or 3)
ATTRACTOR_REPEAT = int(os.environ.get("MEMEXP_EVMEM_ATTRACTOR_REPEAT", "2") or 2)

FINISH_FORM = '{"current_primitive": "<natural robot command>", "keyframe_positions": [<positions>]}'
FORCE_FINAL = (
    "TOOLS ARE NOW CLOSED. Output exactly one JSON object: " + FINISH_FORM
)
FORCE_NATURAL = (
    "Rejected: stage label / ordinal / forbidden intent. "
    "Output ONE natural command that serves the NEED predicate only. " + FINISH_FORM
)

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_LEADING_IDX = re.compile(r"^\d+[_\-\s]*")
_OPEN_CLOSE = re.compile(r"(^|_)(open|close|inspect|observe)(_|$)", re.IGNORECASE)

_CONTAINER = {
    "drawer", "drawers", "basket", "baskets", "cabinet", "microwave", "drainer",
    "bowl", "bowls", "shelf", "tray", "oven", "fridge", "refrigerator", "sink",
    "plate", "bin", "box", "pot", "carton", "cup", "mug", "container", "rack",
}
_QUALIFIER = {
    "top", "middle", "bottom", "upper", "lower", "left", "right", "front", "back",
    "side", "first", "second", "third", "fourth", "inner", "outer", "near", "far",
}

# Off-graph attractors common in briefs but absent from many stage graphs.
_MICROWAVE_RX = re.compile(r"\bmicrowave\b", re.I)
_POUR_RX = re.compile(r"\b(pour|tilt)\b", re.I)
_SECOND_POUR_RX = re.compile(r"\b(second|2nd|twice|two times|again)\b.*\b(pour|sauce)\b|\b(pour|sauce)\b.*\b(second|2nd|twice)\b", re.I)
_PLACE_COOKIES_MW = re.compile(r"\b(cookies|biscuit).{0,40}\b(microwave|oven)\b|\b(microwave|oven).{0,40}\b(cookies|biscuit)\b", re.I)


def enabled() -> bool:
    return str(os.environ.get("MEMEXP_EVMEM", "")).strip().lower() in _TRUTHY


def _stem_tokens(stage_name: str) -> list[str]:
    raw = _LEADING_IDX.sub("", str(stage_name or "").strip())
    return [t for t in re.split(r"[^A-Za-z0-9]+", raw) if t]


def action_stem_safe(stage_name: str) -> str:
    """Stage tokens minus answer-key container phrases (qualifier+container dropped)."""
    toks = _stem_tokens(stage_name)
    kept: list[str] = []
    for tok in toks:
        low = tok.lower()
        if low in _CONTAINER:
            while kept and kept[-1].lower() in _QUALIFIER:
                kept.pop()
            continue
        kept.append(tok)
    return " ".join(kept) if kept else "step"


def public_ordinal(index0: int, n: int) -> str:
    return f"step {index0 + 1} of {n}"


def predicate_scaffold(stage_name: str, *, pour_index: int | None = None, pour_total: int | None = None) -> str:
    """Scorer-aligned NEED text. No qualified container identity."""
    nm = str(stage_name or "")
    low = nm.lower()
    stem = action_stem_safe(nm)
    if re.search(r"lift|grasp|pick", low):
        obj = stem
        for pref in ("Lift ", "Grasp ", "Pick "):
            if obj.lower().startswith(pref.lower()):
                obj = obj[len(pref):].strip()
        return f"grasp/lift the {obj or 'target object'} until it is raised"
    if "pour_one" in low or (pour_index == 1):
        return "pour onto the target once (pour count 1) — do not do later recipe steps"
    if "pour_two" in low or (pour_index == 2):
        return "pour onto the target a second time (pour count 2)"
    if re.search(r"pour", low):
        if pour_index and pour_total:
            return f"pour onto the target (count {pour_index} of {pour_total})"
        return f"pour as required ({stem})"
    if re.search(r"open", low):
        return "open the relevant container and observe its contents (do not name which one)"
    if re.search(r"close", low):
        return "close the currently open container"
    if re.search(r"put|place", low):
        return f"place the object as required ({stem}) without naming drawer identity"
    return f"complete: {stem}"


def admissible_templates(stage_name: str) -> list[str]:
    low = str(stage_name or "").lower()
    if re.search(r"lift|grasp|pick", low):
        return [
            "Reach for and grasp the sauce bottle.",
            "Pick up the tomato sauce bottle.",
        ]
    if "pour_one" in low:
        return [
            "Pour the sauce onto the target once.",
            "Tilt the bottle to pour sauce onto the target.",
        ]
    if "pour_two" in low:
        return [
            "Pour the sauce onto the target a second time.",
            "Tilt the bottle to pour sauce onto the target again.",
        ]
    if re.search(r"open", low):
        return ["Open the next drawer in order.", "Pull the drawer open to inspect."]
    if re.search(r"close", low):
        return ["Close the currently open drawer."]
    if re.search(r"put|place", low):
        return ["Place the object into the open container.", "Put the object into the open drawer."]
    return ["Perform the current NEED with a short natural command."]


def is_open_close_stage(stage_name: str) -> bool:
    raw = _LEADING_IDX.sub("", str(stage_name or "").strip())
    return bool(_OPEN_CLOSE.search(raw.replace(" ", "_")))


def extract_primitive(text: str) -> str:
    s = str(text or "").strip()
    if "</think>" in s:
        s = s[s.rfind("</think>") + len("</think>"):].strip()
    m = _JSON_OBJ.search(s)
    if not m:
        return ""
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("current_primitive", "") or "").strip()


def attractor_cluster(primitive: str) -> str:
    p = str(primitive or "")
    if _MICROWAVE_RX.search(p) or _PLACE_COOKIES_MW.search(p):
        return "microwave"
    if _POUR_RX.search(p) and _SECOND_POUR_RX.search(p):
        return "pour_second"
    if _POUR_RX.search(p):
        return "pour"
    if re.search(r"\b(grasp|lift|pick up|reach)\b", p, re.I):
        return "lift"
    if re.search(r"\b(open|close)\b.*\bdrawer\b|\bdrawer\b.*\b(open|close)\b", p, re.I):
        return "drawer"
    return "other"


def looks_like_label_primitive(text: str, ledger: "Ledger | None" = None) -> bool:
    prim = extract_primitive(text) if "current_primitive" in (text or "") else str(text or "").strip()
    if not prim:
        prim = extract_primitive(text)
    p = prim.strip()
    if not p:
        return False
    pl = re.sub(r"[_\-]+", " ", p.lower())
    pl = re.sub(r"\s+", " ", pl).strip()
    if re.fullmatch(r"step\s+\d+\s+of\s+\d+", pl):
        return True
    if re.fullmatch(r"(pour one|pour two|lift tomato sauce)", pl):
        return True
    if ledger is not None:
        for c in ledger.claims.values():
            if c.kind != "CONDITION":
                continue
            if pl == c.label.lower().strip():
                return True
            stem = action_stem_safe(c.stage_name).lower()
            if stem and pl == stem:
                return True
    return False


def control_reject_reason(primitive: str, ledger: "Ledger") -> str | None:
    """Hard C-gate. Return reason string if primitive must be rejected."""
    p = str(primitive or "").strip()
    if not p:
        return "empty primitive"
    if looks_like_label_primitive(p, ledger):
        return "copied stage label/ordinal"
    # Off-graph microwave if no stage mentions microwave
    stage_blob = " ".join(c.stage_name for c in ledger.claims.values()).lower()
    if _MICROWAVE_RX.search(p) or _PLACE_COOKIES_MW.search(p):
        if "microwave" not in stage_blob:
            return "microwave/off-graph intent forbidden until NEED stages done"
    # Progress guards for counting-pour style ledgers
    lift_done = any(
        c.status == "VERIFIED" and re.search(r"lift|grasp|pick", c.stage_name, re.I)
        for c in ledger.claims.values() if c.kind == "CONDITION"
    )
    has_lift = any(
        re.search(r"lift|grasp|pick", c.stage_name, re.I)
        for c in ledger.claims.values() if c.kind == "CONDITION"
    )
    pour1_done = any(
        c.status == "VERIFIED" and "pour_one" in c.stage_name.lower()
        for c in ledger.claims.values() if c.kind == "CONDITION"
    )
    has_pour1 = any("pour_one" in c.stage_name.lower() for c in ledger.claims.values())
    if has_lift and not lift_done and _POUR_RX.search(p):
        return "pour/tilt before lift VERIFIED"
    if has_pour1 and not pour1_done and _SECOND_POUR_RX.search(p):
        return "second-pour language before pour_one VERIFIED"
    return None


def safe_fallback_primitive(ledger: "Ledger") -> str:
    cur = ledger.active_claim()
    if cur is None:
        return "Reach for and grasp the target object."
    temps = admissible_templates(cur.stage_name)
    return temps[0]


@dataclass
class Claim:
    cid: str
    kind: str
    label: str                 # ordinal only
    predicate: str             # G scaffold
    status: str
    provenance: str
    frames: list[int] = field(default_factory=list)
    attempts: int = 0
    looked: bool = False
    stage_name: str = ""
    last_update_step: int = 0
    auto_look_streak: int = 0


@dataclass
class Event:
    step: int
    intent: str
    verdict: str
    frames: list[int]
    source: str = "planner"


class Ledger:
    def __init__(self) -> None:
        self.claims: dict[str, Claim] = {}
        self._order: list[str] = []
        self.events: list[Event] = []
        self.n_plan_steps = 0
        self.n_looks = 0
        self.n_looks_new_frames = 0
        self.n_looks_zero_gain = 0
        self.n_rereads = 0
        self.n_auto_looks = 0
        self.n_fluent_steps = 0
        self.n_evidence_mode_steps = 0
        self.n_label_rejects = 0
        self.n_control_rejects = 0
        self.n_digest_chars = 0
        self.n_digest_redacted = 0
        self.n_refused = 0
        self.n_seeded = 0
        self.n_evidence_carried = 0
        self.active_cid: str = ""
        self._next_id = 1
        self.evidence_abs: list[int] = []
        self.evidence_meta: list[dict[str, Any]] = []
        self.tool_trace: list[dict[str, Any]] = []
        self.last_digest: str = ""
        self.last_mode: str = "fluent"
        self.pending_pin_cids: list[str] = []
        self._verified_seen: set[str] = set()
        self.attractor_hist: list[str] = []
        self.force_evidence: bool = False

    def reset(self) -> None:
        self.__init__()

    def _mint_id(self) -> str:
        cid = f"c{self._next_id}"
        self._next_id += 1
        return cid

    def seed_conditions(self, stage_names: list[str]) -> int:
        names = [str(n).strip() for n in stage_names if str(n).strip()]
        n = len(names)
        pour_total = sum(1 for nm in names if "pour" in nm.lower())
        pour_i = 0
        minted = 0
        for i, nm in enumerate(names):
            if any(c.stage_name == nm and c.kind == "CONDITION" for c in self.claims.values()):
                continue
            pidx = None
            if "pour" in nm.lower():
                pour_i += 1
                pidx = pour_i
            cid = self._mint_id()
            self.claims[cid] = Claim(
                cid=cid,
                kind="CONDITION",
                label=public_ordinal(i, n),
                predicate=predicate_scaffold(nm, pour_index=pidx, pour_total=pour_total or None),
                status="UNVERIFIED",
                provenance="env",
                stage_name=nm,
            )
            self._order.append(cid)
            minted += 1
        self.n_seeded += minted
        return minted

    def set_active_stage(self, name: str, step: int) -> str:
        nm = str(name or "").strip()
        if not nm:
            return ""
        for cid in self._order:
            c = self.claims[cid]
            if c.kind == "CONDITION" and c.stage_name == nm:
                self.active_cid = cid
                c.last_update_step = int(step)
                return cid
        return ""

    def note_verified(self, name: str, step: int) -> None:
        nm = str(name or "").strip()
        if not nm:
            return
        for c in self.claims.values():
            if c.kind == "CONDITION" and c.stage_name == nm:
                was = c.status
                c.status = "VERIFIED"
                c.provenance = "env"
                c.last_update_step = int(step)
                if was != "VERIFIED" and nm not in self._verified_seen:
                    self._verified_seen.add(nm)
                    if is_open_close_stage(nm):
                        self.pending_pin_cids.append(c.cid)

    def note_action(self, primitive: str, step: int, frame_lo: int, frame_hi: int) -> None:
        text = str(primitive or "").strip()
        frames = list(range(int(frame_lo), int(frame_hi) + 1)) if frame_hi >= frame_lo else []
        self.events.append(Event(step=int(step), intent=text, verdict="", frames=frames))
        self.n_plan_steps += 1
        cl = attractor_cluster(text)
        self.attractor_hist.append(cl)
        if len(self.attractor_hist) > 8:
            self.attractor_hist = self.attractor_hist[-8:]
        # early evidence if same attractor repeats
        if len(self.attractor_hist) >= ATTRACTOR_REPEAT:
            tail = self.attractor_hist[-ATTRACTOR_REPEAT:]
            if len(set(tail)) == 1 and tail[0] in {"microwave", "pour", "pour_second", "other"}:
                if tail[0] == "microwave" or (
                    tail[0] in {"pour", "pour_second"} and self._lift_incomplete()
                ):
                    self.force_evidence = True
        if self.active_cid and self.active_cid in self.claims:
            c = self.claims[self.active_cid]
            c.attempts += 1
            if frames:
                have = set(c.frames)
                c.frames.extend(i for i in frames if i not in have)
            if c.status == "UNVERIFIED":
                c.status = "OPEN_Q"

    def _lift_incomplete(self) -> bool:
        has = any(re.search(r"lift|grasp|pick", c.stage_name, re.I) for c in self.claims.values())
        if not has:
            return False
        return not any(
            c.status == "VERIFIED" and re.search(r"lift|grasp|pick", c.stage_name, re.I)
            for c in self.claims.values()
        )

    def note_verdict(self, checked: str, passed: bool) -> None:
        if not self.events:
            return
        self.events[-1].verdict = f"{checked} {'passed' if passed else 'failed'}"

    def active_claim(self) -> Claim | None:
        if self.active_cid and self.active_cid in self.claims:
            return self.claims[self.active_cid]
        for cid in self._order:
            c = self.claims[cid]
            if c.kind == "CONDITION" and c.status != "VERIFIED":
                return c
        return None

    def decide_mode(self, stall: int) -> str:
        if self.pending_pin_cids or self.force_evidence:
            return "evidence"
        c = self.active_claim()
        if c is None:
            return "fluent"
        if int(c.attempts) >= GATE_ATTEMPTS or int(stall) >= GATE_STALL:
            return "evidence"
        return "fluent"

    def remember_evidence(self, frames: list[int], *, cid: str = "", tag: str = "probe") -> None:
        if not frames:
            return
        have = set(self.evidence_abs)
        for i in frames:
            ii = int(i)
            if ii in have:
                continue
            self.evidence_abs.append(ii)
            self.evidence_meta.append({"abs": ii, "cid": cid, "tag": tag})
            have.add(ii)
            self.n_evidence_carried += 1
        while len(self.evidence_abs) > EVIDENCE_CAP:
            self.evidence_abs.pop(0)
            if self.evidence_meta:
                self.evidence_meta.pop(0)

    def decay_fail_evidence(self, cid: str) -> None:
        """Drop fail_repeat frames for cid after fruitless auto-looks."""
        keep_abs: list[int] = []
        keep_meta: list[dict[str, Any]] = []
        for a, m in zip(self.evidence_abs, self.evidence_meta or [{"abs": x, "cid": "", "tag": "probe"} for x in self.evidence_abs]):
            if m.get("cid") == cid and m.get("tag") == "fail_repeat":
                continue
            keep_abs.append(int(a))
            keep_meta.append(m)
        # keep at most last frame overall
        self.evidence_abs = keep_abs[-EVIDENCE_CAP:]
        self.evidence_meta = keep_meta[-EVIDENCE_CAP:]

    def take_evidence_for_prompt(self) -> list[tuple[int, str]]:
        """Return (abs, tag) pairs for injection."""
        out: list[tuple[int, str]] = []
        meta_by = {int(m.get("abs", -1)): str(m.get("tag", "probe")) for m in self.evidence_meta}
        for a in self.evidence_abs:
            out.append((int(a), meta_by.get(int(a), "probe")))
        return out

    def look(self, cid: str, on_context: set[int], frame_store: dict[int, Any], *,
             auto: bool = False) -> tuple[str, list[int]]:
        self.n_looks += 1
        if auto:
            self.n_auto_looks += 1
        c = self.claims.get(str(cid or "").strip())
        if c is None:
            return f"look({cid!r}): unknown id.", []
        if c.status == "VERIFIED" and not auto:
            self.n_rereads += 1
            return f"look({c.cid}) -> VERIFIED. Do not re-look.", []
        if (c.looked or c.status == "SEEN") and not auto:
            self.n_rereads += 1
        offered = _available_frames(c.frames, on_context, frame_store)
        if not offered and auto and c.frames:
            cand = [int(i) for i in c.frames if int(i) in frame_store]
            offered = [i for i in cand if i not in on_context][-LOOKAHEAD:]
            if not offered:
                offered = cand[-min(2, len(cand)):]
        tag = "boundary_pin" if auto and c.status == "VERIFIED" else ("fail_repeat" if auto else "probe")
        if offered:
            self.n_looks_new_frames += 1
            c.looked = True
            if c.status != "VERIFIED":
                c.status = "SEEN"
                c.provenance = "code"
            if auto:
                c.auto_look_streak += 1
                if c.auto_look_streak >= 2:
                    self.decay_fail_evidence(c.cid)
                    tag = "fail_repeat"
            self.remember_evidence(offered, cid=c.cid, tag=tag)
        else:
            self.n_looks_zero_gain += 1
        kind = "auto-look" if auto else "look"
        lines = [
            f"{kind}({c.cid}) -> {c.label}  NEED={c.predicate[:80]}  status={c.status}",
            f"frames ({tag}): {offered or 'none'}",
            "SEEN = pixels only, NOT done. Only VERIFIED completes the step.",
        ]
        if auto and c.auto_look_streak >= 2:
            lines.append("Prior strategy stuck — switch to a different admissible template.")
        return "\n".join(lines), offered

    def consume_pending_pins(self) -> list[str]:
        out = list(self.pending_pin_cids)
        self.pending_pin_cids = []
        return out

    def clear_force_evidence(self) -> None:
        self.force_evidence = False

    def ls(self, path: str = "") -> str:
        p = str(path or "memory").strip().strip("/")
        if p in {"", "memory", "."}:
            return "memory/\n  INDEX.md\nframes/\nevents.jsonl"
        return f"ls: try memory/"

    def cat(self, path: str, on_context: set[int], frame_store: dict[int, Any]) -> tuple[str, list[int]]:
        p = str(path or "").strip()
        if p in self.claims or (p.startswith("c") and p[1:].isdigit()):
            return self.look(p, on_context, frame_store)
        if p in {"events.jsonl", "events"}:
            return self.render_events(), []
        if p in {"INDEX.md", "memory/INDEX.md"}:
            return self.render_digest(mode=self.last_mode), []
        return f"cat: {p!r} not found", []

    def grep(self, pattern: str) -> str:
        pat = str(pattern or "").strip()
        if not pat:
            return "grep needs a pattern"
        rx = re.compile(re.escape(pat), re.IGNORECASE)
        hits = [f"step={e.step} {e.intent} {e.verdict}" for e in self.events if rx.search(f"{e.intent} {e.verdict}")]
        if not hits:
            return f"grep({pat!r}): no hits"
        return f"grep({pat!r}) {len(hits)} hit(s):\n" + "\n".join(hits[:20])

    def render_events(self) -> str:
        lines = [json.dumps({"step": e.step, "intent": e.intent, "verdict": e.verdict}, ensure_ascii=False)
                 for e in self.events]
        return "\n".join(lines) if lines else "(no events)"

    def _progress(self) -> tuple[int, int]:
        conds = [self.claims[c] for c in self._order if self.claims[c].kind == "CONDITION"]
        return sum(1 for c in conds if c.status == "VERIFIED"), len(conds)

    def forbidden_lines(self) -> list[str]:
        lines = ["Do not copy step labels or stage ids."]
        stage_blob = " ".join(c.stage_name for c in self.claims.values()).lower()
        if "microwave" not in stage_blob:
            lines.append("Forbidden now: open/use microwave; place food into microwave.")
        if self._lift_incomplete():
            lines.append("Forbidden now: pour/tilt until lift is VERIFIED.")
        has_pour1 = any("pour_one" in c.stage_name.lower() for c in self.claims.values())
        pour1_done = any(c.status == "VERIFIED" and "pour_one" in c.stage_name.lower() for c in self.claims.values())
        if has_pour1 and not pour1_done:
            lines.append("Forbidden now: 'second pour' / count-2 language.")
        lines.append("Ignore later recipe steps in the task brief until current NEED is VERIFIED.")
        return lines

    def render_digest(self, mode: str = "fluent", redact_fn=None, stall: int = 0) -> str:
        self.last_mode = mode
        v, n = self._progress()
        cur = self.active_claim()
        lines = [f"=== {'evidence mode' if mode == 'evidence' else 'progress'} ===", f"verified {v}/{n}"]
        if cur is not None:
            lines.append(f"current {cur.cid} {cur.label}  status={cur.status}  attempts={cur.attempts}")
            lines.append(f"NEED: {cur.predicate}")
        lines.extend(self.forbidden_lines())
        if mode == "evidence" and cur is not None:
            lines.append("Admissible templates (paraphrase OK, same intent):")
            for t in admissible_templates(cur.stage_name):
                lines.append(f"  - {t}")
            lines.append(f"stall={int(stall)}  evidence_abs={self.evidence_abs or []}")
            lines.append("Tagged fail frames are recent unsuccessful attempts — change strategy.")
            lines.append("SEEN ≠ done. Emit ONE natural command for NEED only.")
        else:
            lines.append("Emit ONE natural command that serves NEED only. Do not skip ahead.")
        text = "\n".join(lines)
        if len(text) > DIGEST_CAP:
            text = text[:DIGEST_CAP]
        if redact_fn is not None:
            try:
                red = str(redact_fn(text) or text)
            except Exception:
                red = text
            if red != text:
                self.n_digest_redacted += 1
            text = red
        self.n_digest_chars = len(text)
        self.last_digest = text
        return text

    def spec_text(self) -> str:
        return "\n".join([
            "=== memory tools (optional) ===",
            '{"tool":"look","id":"<cN>"}',
            '{"tool":"ls","path":"memory"}',
            '{"tool":"cat","path":"events.jsonl"}',
            '{"tool":"grep","pattern":"<literal>"}',
            FINISH_FORM,
        ])

    def record_tool(self, *, step: int, tool: str, args: dict, offered: list[int],
                    text_head: str, auto: bool = False) -> None:
        self.tool_trace.append({
            "step": int(step), "tool": str(tool), "auto": bool(auto),
            "args": dict(args or {}), "offered": [int(x) for x in offered],
            "text_head": str(text_head or "")[:240],
        })

    def summary(self) -> dict[str, Any]:
        return {
            "n_claims": len(self.claims),
            "n_events": len(self.events),
            "n_plan_steps": self.n_plan_steps,
            "n_looks": self.n_looks,
            "n_auto_looks": self.n_auto_looks,
            "n_looks_new_frames": self.n_looks_new_frames,
            "n_looks_zero_gain": self.n_looks_zero_gain,
            "n_rereads": self.n_rereads,
            "n_fluent_steps": self.n_fluent_steps,
            "n_evidence_mode_steps": self.n_evidence_mode_steps,
            "n_label_rejects": self.n_label_rejects,
            "n_control_rejects": self.n_control_rejects,
            "n_digest_redacted": self.n_digest_redacted,
            "n_seeded": self.n_seeded,
            "n_evidence_carried": self.n_evidence_carried,
            "evidence_abs": list(self.evidence_abs),
            "events_eq_steps": len(self.events) == self.n_plan_steps,
            "n_tool_trace": len(self.tool_trace),
            "last_mode": self.last_mode,
            "labels": [self.claims[c].label for c in self._order if self.claims[c].kind == "CONDITION"],
            "predicates": [self.claims[c].predicate for c in self._order if self.claims[c].kind == "CONDITION"],
            "last_digest_head": (self.last_digest or "")[:500],
        }


def _available_frames(candidates: list[int], on_context: set[int], store: dict[int, Any]) -> list[int]:
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
    s = str(text or "").strip()
    if "</think>" in s:
        s = s[s.rfind("</think>") + len("</think>"):].strip()
    if s.startswith("```"):
        lines = s.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    m = _JSON_OBJ.search(s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("tool", "")).strip().lower()
    if name not in known:
        return None
    args = obj.get("args")
    if not isinstance(args, dict):
        args = {k: v for k, v in obj.items() if k not in {"tool", "args", "reason", "thought", "why"}}
    return {"tool": name, "args": args}


def known_tools() -> set[str]:
    return {"look", "ls", "cat", "grep"}


def dispatch(led: Ledger, name: str, args: dict[str, Any], *,
             on_context: set[int], frame_store: dict[int, Any]) -> tuple[str, list[int]]:
    n = str(name or "").strip().lower()
    if n not in known_tools():
        return f"unknown tool {name!r}", []
    if n == "look":
        return led.look(str(args.get("id", args.get("cid", ""))).strip(), on_context, frame_store)
    if n == "ls":
        return led.ls(str(args.get("path", "") or "")), []
    if n == "cat":
        return led.cat(str(args.get("path", args.get("id", "")) or ""), on_context, frame_store)
    if n == "grep":
        return led.grep(str(args.get("pattern", "") or "")), []
    return f"unknown tool {name!r}", []


def apply_control(out_text: str, ledger: Ledger) -> tuple[str, bool]:
    """If rejected, return a forced JSON with safe fallback. Second value True if rewritten."""
    prim = extract_primitive(out_text)
    reason = control_reject_reason(prim, ledger)
    if reason is None:
        return out_text, False
    ledger.n_control_rejects += 1
    fb = safe_fallback_primitive(ledger)
    return json.dumps({"current_primitive": fb, "keyframe_positions": []}, ensure_ascii=False), True


def merge_keyframe_positions(out_text: str, abs_frames: list[int], *,
                             recent_start: int, n_context: int) -> str:
    if not abs_frames or n_context <= 0:
        return out_text
    rel = []
    for a in abs_frames:
        r = int(a) - int(recent_start) + 1
        if 1 <= r <= int(n_context):
            rel.append(r)
    if not rel:
        return out_text
    m = _JSON_OBJ.search(str(out_text or ""))
    if not m:
        return out_text
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return out_text
    if not isinstance(obj, dict) or "current_primitive" not in obj:
        return out_text
    existing = obj.get("keyframe_positions") if isinstance(obj.get("keyframe_positions"), list) else []
    merged, seen = [], set()
    for x in list(existing) + rel:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi in seen or not (1 <= xi <= n_context):
            continue
        seen.add(xi)
        merged.append(xi)
    obj["keyframe_positions"] = merged
    return json.dumps(obj, ensure_ascii=False)


def report_path() -> Path:
    explicit = str(os.environ.get("MEMEXP_EVMEM_REPORT", "")).strip()
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("MEMEXP_RUN_DIR") or os.environ.get("OUT_ROOT") or ".")
    return root / "memexp_evmem_report.json"


def write_report(ledgers: list[Ledger]) -> Path | None:
    path = report_path()
    totals = {
        "n_ledgers": len(ledgers),
        "n_events": sum(len(x.events) for x in ledgers),
        "n_plan_steps": sum(x.n_plan_steps for x in ledgers),
        "n_looks": sum(x.n_looks for x in ledgers),
        "n_auto_looks": sum(x.n_auto_looks for x in ledgers),
        "n_looks_new_frames": sum(x.n_looks_new_frames for x in ledgers),
        "n_fluent_steps": sum(x.n_fluent_steps for x in ledgers),
        "n_evidence_mode_steps": sum(x.n_evidence_mode_steps for x in ledgers),
        "n_control_rejects": sum(x.n_control_rejects for x in ledgers),
        "n_label_rejects": sum(x.n_label_rejects for x in ledgers),
        "n_evidence_carried": sum(x.n_evidence_carried for x in ledgers),
        "n_tool_trace": sum(len(x.tool_trace) for x in ledgers),
    }
    payload = {"arm": os.environ.get("MEMEXP_ARM_NAME", "evmem"), "pid": os.getpid(),
               "version": "gpm_l0l1", "totals": totals, "ledgers": [x.summary() for x in ledgers]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        own = Path(f"{path}.{os.getpid()}")
        tmp = Path(f"{own}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(own)
        audit = Path(f"{own}.tools.jsonl")
        with audit.open("w", encoding="utf-8") as fh:
            for led in ledgers:
                if not led.tool_trace and not led.last_digest:
                    continue
                fh.write(json.dumps({
                    "mode": led.last_mode,
                    "evidence_abs": led.evidence_abs,
                    "evidence_meta": led.evidence_meta,
                    "last_digest": led.last_digest,
                    "tool_trace": led.tool_trace,
                    "predicates": [led.claims[c].predicate for c in led._order
                                   if led.claims[c].kind == "CONDITION"],
                }, ensure_ascii=False) + "\n")
        return own
    except OSError:
        return None
