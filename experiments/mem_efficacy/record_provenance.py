"""mem_efficacy / record WHICH CODE produced a result, next to the result.

WHY THIS EXISTS
---------------
These arms do not run the harness as it sits on disk. They REPLACE parts of it at runtime: the
`pullmem` arm wraps two functions on `harness.api_vlm_planner`, and both memory arms replace the
memory renderer and the context refresher on `harness.memory_reason` / `harness.controller`. So a
score is a property of (this experiment's files) x (the upstream harness revision), and the second
factor is not recorded anywhere today.

That matters concretely:
  * if upstream `harness/memory_reason.py` is edited, the patch target changes and an archived
    score can no longer be reproduced -- but nothing in the results directory would show it;
  * if `memexp_memfix.py` is edited, `MEMEXP_MEMFIX_ENABLE=1` still reads as "corrected", so two
    runs with different corrections would look like the same arm.

`resolved_env.txt` (written by the runner) answers "what was this configured with". This file
answers the different question "what code was this", by hashing every file that can change the
answer. A difference in that hash means the comparison is not a comparison.

WHAT IS NOT HASHED
------------------
The task data, checkpoints and the frozen VLA are not hashed: they are large, read-only, and
identified by the protocol knobs already recorded in `resolved_env.txt`. This file is about code
that a person can edit by accident.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys

# Every file that can change what an arm does, or what the numbers mean:
#   * this experiment's own modules and drivers;
#   * the upstream modules the arms PATCH AT RUNTIME -- these are the load-bearing ones, because
#     an edit here silently changes the meaning of an already-recorded score;
#   * the evaluator entry points the protocol knobs are passed to.
RELATIVE_PATHS = [
    # this experiment
    "experiments/mem_efficacy/run_26x1.sbatch",
    "experiments/mem_efficacy/validate_arm.py",
    "experiments/mem_efficacy/census_channels.py",
    "experiments/mem_efficacy/selftest_pull.py",
    "experiments/mem_efficacy/selftest_memfix.py",
    "experiments/mem_efficacy/memexp_memfix.py",
    "experiments/mem_efficacy/memexp_bind.py",
    "experiments/mem_efficacy/memexp_substrate.py",
    "experiments/mem_efficacy/memexp_tools.py",
    # The SEARCH registry, selected by the `pullmem` arm via `MEMEXP_TOOLS_MODULE`. It decides the
    # arm's tool SET, its spec wording and its finishing form, so an edit here changes which tool
    # design the arm tested and can invalidate an archived score exactly as an edit to the binding
    # can. It replaces the address-only registry's role for the pull arm and must be hashed for the
    # same reason.
    "experiments/mem_efficacy/memexp_tools_search.py",
    "experiments/mem_efficacy/pysite/sitecustomize.py",
    "experiments/mem_efficacy/arms/official_protocol.sh",
    "experiments/mem_efficacy/arms/_memexp_pysite.sh",
    "experiments/mem_efficacy/arms/nomem.sh",
    "experiments/mem_efficacy/arms/pushmem.sh",
    "experiments/mem_efficacy/arms/pullmem.sh",
    # upstream, patched at runtime by the arms above
    "evaluation_benchmark/harness/memory_reason.py",
    "evaluation_benchmark/harness/controller.py",
    "evaluation_benchmark/harness/api_vlm_planner.py",
    # `memory_system/config.py` -- NOT merely a hyperparameter file. It defines `kf_spread` and
    # `nomination_prompt`, i.e. whether the Planner's prompt gains the keyframe policy AND whether
    # the bank gets temporally spread frames, plus `bank_max` / `cluster_distance` / `Stage_anchor`
    # which decide the bank's shape. An archived score therefore cannot be tied to a revision of
    # the memory mechanism without it: two runs with identical provenance fingerprints could have
    # had different prompts and different banks. Found by diffing the tracked set against the arm
    # that reads it -- the file was missing while its four consumers were all present.
    "evaluation_benchmark/memory_system/config.py",
      "evaluation_benchmark/memory_system/keyframe_bank.py",
      "evaluation_benchmark/harness/api_planner.py",
    "evaluation_benchmark/harness/external_memory.py",
    "evaluation_benchmark/harness/config.py",
    "evaluation_benchmark/harness/stage_mapper.py",
    # Channel B's builders. Lives under `openpi_minimal_runtime/`, NOT under `harness/` -- the
    # first version of this list guessed `harness/keyframe_selection.py`, which does not exist, and
    # the missing-file check below is what caught it. That is the check earning its place: a
    # silently missing entry here would mean a file that decides channel B was never hashed.
    "evaluation_benchmark/openpi_minimal_runtime/keyframe_selection.py",
    # the evaluator the protocol knobs are handed to
    "evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py",
    "evaluation_benchmark/async_vlm26_reference/run_fullvlm26_async_vlm_vla_csr_tsr.sh",
    # The task's stage table, and the reason it is hashed: the `pullmem` arm's task-spec seed
    # EXPANDS this list into the memory bank's addresses (`memexp_bind._seed_from_task_spec` ->
    # `Substrate.seed_from_stages`). So an edit here changes which containers the bank can hold --
    # i.e. it decides what that arm's memory is able to represent -- exactly as an edit to
    # `memexp_tools_search.py` decides its tool set. It is also the file the seed and the stage
    # SCORER share, which is what keeps them from disagreeing about what the task contains, and
    # a third copy of the stage list anywhere else would break that without any flag changing.
    "evaluation_benchmark/scripts/task2_26_reference_stage.py",
]


def sha256_of(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def git(*args: str, root: str) -> str | None:
    try:
        p = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=30)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def collect(root: str) -> dict:
    files: dict[str, str | None] = {}
    for rel in RELATIVE_PATHS:
        files[rel] = sha256_of(os.path.join(root, rel))

    # The arms patch upstream files, so a DIRTY upstream tree is the normal state of this
    # repository and "clean/dirty" alone says nothing. What matters is the hash above. The git
    # revision is recorded because it is how a human finds the right tree again.
    dirty = git("status", "--porcelain", root=root)
    return {
        "root": root,
        "git_head": git("rev-parse", "HEAD", root=root),
        "git_describe": git("describe", "--always", "--dirty", root=root),
        "git_dirty_file_count": len(dirty.splitlines()) if dirty is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "file_sha256": files,
        "missing_files": sorted(k for k, v in files.items() if v is None),
    }


def main(argv: list[str]) -> int:
    root = os.environ.get("ROOT") or (argv[1] if len(argv) > 1 else os.getcwd())
    out_dir = os.environ.get("OUT_ROOT") or (argv[2] if len(argv) > 2 else root)
    arm = os.environ.get("MEMEXP_ARM_NAME", "unspecified")

    data = collect(root)
    data["arm"] = arm
    data["job"] = os.environ.get("SLURM_JOB_ID")
    data["job_tag"] = os.environ.get("JOB_TAG")

    path = os.path.join(out_dir, "code_provenance.json")
    try:
        os.makedirs(out_dir, exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[provenance] could not write {path}: {exc}", file=sys.stderr)
        return 1

    # A file the arms patch that is MISSING means the patch target does not exist, so the patch
    # will silently not apply and the arm will silently run as its baseline. That is the exact
    # failure mode this experiment has already hit twice, so it is surfaced here at run time.
    if data["missing_files"]:
        print("[provenance] !! these patch targets / inputs are MISSING, so the arms that need "
              "them cannot behave as declared:", file=sys.stderr)
        for m in data["missing_files"]:
            print(f"[provenance]      {m}", file=sys.stderr)

    print(f"[provenance] wrote {path}")
    print(f"[provenance] arm={arm} git={data['git_describe']} "
          f"({len(data['file_sha256']) - len(data['missing_files'])} files hashed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
