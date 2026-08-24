from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

DEFAULT_STATE = {
    "project_name": "",
    "workflow_mode": "Quick-start without LLM",
    "created_at": "",
    "updated_at": "",
    "status": "initialized",
}


def get_project_root(project_name: str, base_dir: Path | str = Path("runs") / "web_projects") -> Path:
    safe_name = project_name.strip().replace(" ", "_") or "untitled_project"
    return Path(base_dir) / safe_name


def state_path(project_root: Path) -> Path:
    return project_root / "project_state.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_project_state(project_root: Path, project_name: str, mode: str) -> Dict[str, Any]:
    project_root.mkdir(parents=True, exist_ok=True)

    path = state_path(project_root)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = DEFAULT_STATE.copy()
        state["created_at"] = _timestamp()

    state.update(
        {
            "project_name": project_name,
            "workflow_mode": mode,
            "updated_at": _timestamp(),
            "status": state.get("status", "initialized"),
        }
    )

    save_project_state(project_root, state)
    return state


def save_project_state(project_root: Path, state: Dict[str, Any]) -> Path:
    project_root.mkdir(parents=True, exist_ok=True)
    path = state_path(project_root)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return path


def set_project_mode(project_root: Path, mode: str) -> Dict[str, Any]:
    path = state_path(project_root)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = DEFAULT_STATE.copy()
        state["created_at"] = _timestamp()
    state["workflow_mode"] = mode
    state["updated_at"] = _timestamp()
    save_project_state(project_root, state)
    return state
