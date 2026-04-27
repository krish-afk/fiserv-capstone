from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from src.utils.config import config as base_config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runs_root(cfg: Optional[dict] = None) -> Path:
    cfg = cfg or base_config
    root = Path(cfg["paths"]["experiments"]) / "dashboard_runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_dir(run_id: str, cfg: Optional[dict] = None) -> Path:
    return _runs_root(cfg) / run_id


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)


def _read_json(path: Path, default: Optional[dict] = None) -> Optional[dict]:
    if not path.exists():
        return default
    with open(path, "r") as f:
        return json.load(f)


def generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid4().hex[:8]}"


def create_run(payload: Optional[dict] = None, cfg: Optional[dict] = None) -> dict:
    run_id = generate_run_id()
    run_dir = _run_dir(run_id, cfg=cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    request_payload = payload or {}
    request_path = run_dir / "request.json"
    meta_path = run_dir / "meta.json"

    meta = {
        "run_id": run_id,
        "status": "running",
        "created_at": _now_iso(),
        "started_at": _now_iso(),
        "completed_at": None,
        "error": None,
        "has_results": False,
        "results_path": None,
        "request_path": str(request_path),
        "pipeline_run_id": None,
        "pipeline_run_dir": None,
        "selection": None,
    }

    _write_json(request_path, request_payload)
    _write_json(meta_path, meta)
    return meta


def complete_run(run_id: str, result: dict, cfg: Optional[dict] = None) -> dict:
    run_dir = _run_dir(run_id, cfg=cfg)
    meta_path = run_dir / "meta.json"
    results_path = run_dir / "results.json"

    meta = _read_json(meta_path, default={}) or {}
    _write_json(results_path, result)

    meta.update(
        {
            "status": "completed",
            "completed_at": _now_iso(),
            "error": None,
            "has_results": True,
            "results_path": str(results_path),
            "pipeline_run_id": result.get("pipeline_run_id"),
            "pipeline_run_dir": result.get("run_dir"),
            "selection": result.get("selection"),
        }
    )

    _write_json(meta_path, meta)
    return meta


def fail_run(run_id: str, error: str, cfg: Optional[dict] = None) -> dict:
    run_dir = _run_dir(run_id, cfg=cfg)
    meta_path = run_dir / "meta.json"

    meta = _read_json(meta_path, default={}) or {}
    meta.update(
        {
            "status": "failed",
            "completed_at": _now_iso(),
            "error": str(error),
            "has_results": False,
        }
    )
    _write_json(meta_path, meta)
    return meta


def get_run(run_id: str, cfg: Optional[dict] = None) -> Optional[dict]:
    run_dir = _run_dir(run_id, cfg=cfg)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    return _read_json(meta_path)


def get_run_results(run_id: str, cfg: Optional[dict] = None) -> Optional[dict]:
    run_dir = _run_dir(run_id, cfg=cfg)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None
    return _read_json(results_path)


def list_runs(limit: int = 20, cfg: Optional[dict] = None) -> list[dict]:
    root = _runs_root(cfg=cfg)
    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    run_dirs = sorted(run_dirs, key=lambda p: p.name, reverse=True)

    out = []
    for run_dir in run_dirs[:limit]:
        meta = _read_json(run_dir / "meta.json")
        if meta:
            out.append(meta)

    return out