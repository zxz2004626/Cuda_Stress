"""Directory-based task queue with atomic state transitions.

    generated -> compiling -> ready_to_run -> running -> triage -> done | rejected

Every transition is an atomic rename of a case directory, so a process crash leaves the
task in a well-defined stage that `resume` can pick back up.  A case's identity is its
directory name (case_id) = content-derived (dedup happens automatically).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

STAGES = ["generated", "compiling", "ready_to_run", "running", "triage", "done", "rejected"]


class TaskQueue:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for s in STAGES:
            (self.root / s).mkdir(parents=True, exist_ok=True)

    # -- introspection ------------------------------------------------------
    def list(self, stage: str) -> List[str]:
        d = self.root / stage
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    def stage_of(self, case_id: str) -> Optional[str]:
        for s in STAGES:
            if (self.root / s / case_id).is_dir():
                return s
        return None

    def count(self, stage: str) -> int:
        return len(self.list(stage))

    # -- transitions ----------------------------------------------------------
    def _move(self, case_id: str, src: str, dst: str) -> Path:
        src_dir = self.root / src / case_id
        dst_dir = self.root / dst / case_id
        if not src_dir.exists():
            raise FileNotFoundError(f"task {case_id} not in {src}")
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(src_dir), str(dst_dir))     # atomic on same filesystem
        return dst_dir

    def enter(self, case_dir: Path) -> Path:
        """Move an already-materialized case dir into generated/."""
        case_id = case_dir.name
        dst = self.root / "generated" / case_id
        os.replace(str(case_dir), str(dst))
        return dst

    def to_compiling(self, case_id: str) -> Path:
        return self._move(case_id, "generated", "compiling")

    def to_ready(self, case_id: str) -> Path:
        src = self.stage_of(case_id)
        if src in ("compiling", "running"):
            return self._move(case_id, src, "ready_to_run")
        if src == "generated":
            return self._move(case_id, "generated", "ready_to_run")
        return self.root / "ready_to_run" / case_id

    def to_running(self, case_id: str) -> Path:
        return self._move(case_id, "ready_to_run", "running")

    def to_triage(self, case_id: str) -> Path:
        return self._move(case_id, "running", "triage")

    def to_done(self, case_id: str, category: str = "") -> Path:
        return self._move(case_id, "triage", "done")

    def to_rejected(self, case_id: str, reason: str = "") -> Path:
        dst = self._move(case_id, "triage", "rejected")
        taskf = dst / "task.json"
        rec = {}
        if taskf.exists():
            try:
                rec = json.loads(taskf.read_text())
            except Exception:  # noqa: BLE001
                rec = {}
        rec["reject_reason"] = reason
        rec["category"] = "invalid"
        taskf.write_text(json.dumps(rec, indent=2))
        return dst

    def move_to_done_with_reason(self, case_id: str, category: str, reason: str = "") -> Path:
        src = self.stage_of(case_id)
        if src in ("triage", "running", "ready_to_run", "compiling"):
            dst = self._move(case_id, src, "done")
        else:
            dst = self.root / "done" / case_id
        taskf = dst / "task.json"
        rec = {}
        if taskf.exists():
            try:
                rec = json.loads(taskf.read_text())
            except Exception:  # noqa: BLE001
                rec = {}
        rec["category"] = category
        rec["reason"] = reason
        taskf.write_text(json.dumps(rec, indent=2))
        return dst

    # -- recovery -------------------------------------------------------------
    def recover(self) -> Dict[str, int]:
        """Move interrupted tasks back to a runnable stage after a crash/restart."""
        moved = {}
        # compile interrupted -> ready (compile is idempotent; recompile overwrites)
        for case_id in self.list("compiling"):
            self._move(case_id, "compiling", "ready_to_run")
            moved.setdefault("compiling->ready", 0)
            moved["compiling->ready"] += 1
        # running interrupted -> ready
        for case_id in self.list("running"):
            self._move(case_id, "running", "ready_to_run")
            moved.setdefault("running->ready", 0)
            moved["running->ready"] += 1
        # triage is idempotent: leave in place
        return moved

    def case_dir(self, stage: str, case_id: str) -> Path:
        return self.root / stage / case_id

    def write_state(self, state: Dict) -> None:
        tmp = self.root / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        os.replace(str(tmp), str(self.root / "state.json"))

    def read_state(self) -> Dict:
        try:
            return json.loads((self.root / "state.json").read_text())
        except Exception:  # noqa: BLE001
            return {}

    def write_run_id(self, run_id: str) -> None:
        tmp = self.root / "run_id.txt.tmp"
        tmp.write_text(run_id)
        os.replace(str(tmp), str(self.root / "run_id.txt"))

    def read_run_id(self) -> Optional[str]:
        try:
            return (self.root / "run_id.txt").read_text().strip()
        except Exception:  # noqa: BLE001
            return None
