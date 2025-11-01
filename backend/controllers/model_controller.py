import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from flask import Blueprint, jsonify, request

from store import store

model_bp = Blueprint("model", __name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_SAVE_DIR = BACKEND_DIR / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_file_path(model_id: str) -> Path:
    return MODEL_SAVE_DIR / f"model_{model_id}.pkl"


def _error_response(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _build_model_summary(model_entry: dict, include_runs: bool = False) -> dict:
    model_copy = copy.deepcopy(model_entry)
    runs = store.list_runs(model_copy["model_id"])
    runs_copy = [copy.deepcopy(run) for run in runs]

    model_copy["runs_total"] = len(runs_copy)

    succeeded_runs = [
        run for run in runs_copy if run.get("state") == "succeeded"
    ]
    if succeeded_runs:
        succeeded_runs.sort(
            key=lambda r: r.get("completed_at") or r.get("created_at") or "",
            reverse=True,
        )
        latest = succeeded_runs[0]
        model_copy["trained"] = True
        model_copy["saved_model_path"] = latest.get("saved_model_path") or model_copy.get(
            "saved_model_path"
        )
        model_copy["last_trained_at"] = (
            latest.get("completed_at") or model_copy.get("last_trained_at")
        )
    else:
        model_copy["trained"] = bool(model_copy.get("trained"))
        model_copy.setdefault("saved_model_path", None)
        model_copy.setdefault("last_trained_at", None)

    saved_path = model_copy.get("saved_model_path")
    if saved_path:
        path = Path(saved_path)
        model_copy["saved_model_exists"] = path.exists()
        if not path.exists():
            model_copy["saved_model_path"] = str(path)
    else:
        expected_path = _model_file_path(model_copy["model_id"])
        model_copy["saved_model_exists"] = expected_path.exists()
        if expected_path.exists():
            model_copy["trained"] = True
            model_copy["saved_model_path"] = str(expected_path)

    if include_runs:
        model_copy["runs"] = runs_copy

    return model_copy


@model_bp.route("/api/models", methods=["GET"])
def list_models():
    models = [_build_model_summary(model) for model in store.list_models()]
    return jsonify(models), 200


@model_bp.route("/api/models", methods=["POST"])
def create_model():
    if not request.is_json:
        return _error_response("Expected JSON payload.", status=415)

    try:
        payload = request.get_json(force=True)
    except Exception:
        return _error_response("Malformed JSON payload.")

    if not isinstance(payload, dict):
        return _error_response("Payload must be a JSON object.")

    model_id_raw: Optional[str] = payload.get("model_id")
    if model_id_raw is not None:
        if not isinstance(model_id_raw, str) or not model_id_raw.strip():
            return _error_response("`model_id` must be a non-empty string.", status=422)
        model_id = model_id_raw.strip()
    else:
        model_id = f"m_{uuid4().hex}"

    name = payload.get("name")
    if name is not None:
        name = str(name).strip() or None

    description = payload.get("description")
    if description is not None:
        description = str(description).strip() or None

    architecture = payload.get("architecture")
    if architecture is not None and not isinstance(architecture, dict):
        return _error_response("`architecture` must be an object.", status=422)

    hyperparams = payload.get("hyperparams")
    if hyperparams is not None and not isinstance(hyperparams, dict):
        return _error_response("`hyperparams` must be an object.", status=422)

    existing_model = store.get_model(model_id)
    created = existing_model is None

    if created:
        model_file = _model_file_path(model_id)
        trained = model_file.exists()
        last_trained_at = None
        if trained:
            try:
                last_trained_at = datetime.fromtimestamp(
                    model_file.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                last_trained_at = None
        model_data = {
            "model_id": model_id,
            "name": name,
            "description": description,
            "architecture": copy.deepcopy(architecture) if architecture is not None else {},
            "hyperparams": copy.deepcopy(hyperparams) if hyperparams is not None else {},
            "created_at": _utcnow_iso(),
            "trained": trained,
            "saved_model_path": str(model_file) if trained else None,
            "last_trained_at": last_trained_at,
        }
        store.add_model(model_id, model_data)
        model_entry = model_data
    else:
        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if architecture is not None:
            updates["architecture"] = copy.deepcopy(architecture)
        if hyperparams is not None:
            updates["hyperparams"] = copy.deepcopy(hyperparams)

        if "trained" in payload:
            updates["trained"] = bool(payload["trained"])

        if "saved_model_path" in payload:
            saved_path = payload["saved_model_path"]
            if saved_path is not None:
                path = Path(saved_path)
                if not path.exists():
                    return _error_response("`saved_model_path` does not exist on disk.", status=422)
                updates["saved_model_path"] = str(path)
                updates["trained"] = True
                if "last_trained_at" not in payload:
                    try:
                        updates["last_trained_at"] = datetime.fromtimestamp(
                            path.stat().st_mtime, tz=timezone.utc
                        ).isoformat()
                    except OSError:
                        pass
            else:
                updates["saved_model_path"] = None

        if "last_trained_at" in payload:
            updates["last_trained_at"] = payload["last_trained_at"]

        if updates:
            store.update_model(model_id, updates)

        model_entry = store.get_model(model_id)
        if model_entry is None:
            return _error_response("Failed to update model.", status=500)

    summary = _build_model_summary(model_entry)
    status = 201 if created else 200
    return jsonify(summary), status


@model_bp.route("/api/models/<id>", methods=["GET"])
def get_model(id: str):
    model_entry = store.get_model(id)
    if model_entry is None:
        return _error_response("Unknown model_id.", status=404)

    summary = _build_model_summary(model_entry, include_runs=True)
    return jsonify(summary), 200
