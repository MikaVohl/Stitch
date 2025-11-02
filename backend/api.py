import copy
from datetime import datetime, timezone
import json
import logging
import queue
import threading
import traceback
import uuid
import random
from pathlib import Path
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
import torch
from torch import nn
from controllers.model_controller import model_bp
from controllers.chat_controller import chat_bp
from services.model_service import (
    build_model,
    prepare_dataloaders,
    configure_optimizer,
    tensor_from_pixels,
)
from store import store
from utils.validation import validate_architecture, validate_hyperparams

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


"""
Expected request payload:
{
  "architecture": {
    "input_size": 784,
    "input_channels": 1,
    "input_height": 28,
    "input_width": 28,
    "layers": [
      {"type": "conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": "same"},
      {"type": "relu"},
      {"type": "maxpool2d", "kernel_size": 2, "stride": 2, "padding": 0},
      {"type": "conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": "same"},
      {"type": "relu"},
      {"type": "maxpool2d", "kernel_size": 2, "stride": 2, "padding": 0},
      {"type": "flatten"},
      {"type": "linear", "in": 3136, "out": 128},
      {"type": "relu"},
      {"type": "dropout", "p": 0.5},
      {"type": "linear", "in": 128, "out": 10},
      {"type": "softmax"}
    ]
  },
  "hyperparams": {
    "epochs": 5,
    "batch_size": 64,
    "optimizer": {"type": "sgd", "lr": 0.1, "momentum": 0.0},
    "loss": "cross_entropy",
    "seed": 42,
    "train_split": 0.9,
    "shuffle": true,
    "max_samples": 4096
  }
}
"""

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_SAVE_DIR = BACKEND_DIR / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
MNIST_DATA_ROOT = BACKEND_DIR / "data" / "mnist"
MNIST_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _model_file_path(model_id: str) -> Path:
    return MODEL_SAVE_DIR / f"model_{model_id}.pkl"


app = Flask(__name__)
CORS(app)

# Register blueprints

app.register_blueprint(model_bp)
app.register_blueprint(chat_bp)


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _train_with_torch(model, train_loader, val_loader, hyperparams, on_checkpoint=None):
    import time

    device = torch.device("cpu")
    model.to(device)

    if hyperparams["seed"] is not None:
        torch.manual_seed(hyperparams["seed"])
        random.seed(hyperparams["seed"])

    criterion = nn.CrossEntropyLoss()
    optimizer = configure_optimizer(hyperparams["optimizer"], model.parameters())

    epochs = hyperparams["epochs"]
    metrics = []

    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += inputs.size(0)

        avg_train_loss = train_loss / max(1, train_total)
        train_accuracy = train_correct / max(1, train_total)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += inputs.size(0)

        avg_val_loss = val_loss / max(1, val_total)
        val_accuracy = val_correct / max(1, val_total)

        # Calculate timing and progress metrics
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        avg_epoch_time = elapsed_time / epoch
        eta_seconds = avg_epoch_time * (epochs - epoch)
        progress = epoch / epochs

        # Get learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        metric_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_accuracy": round(val_accuracy, 4),
            "learning_rate": round(current_lr, 6),
            "epoch_time": round(epoch_time, 2),
            "samples_per_sec": round(train_total / epoch_time, 1)
            if epoch_time > 0
            else 0,
            "progress": round(progress, 4),
            "eta_seconds": round(eta_seconds, 1),
        }
        metrics.append(metric_entry)
        if on_checkpoint is not None:
            on_checkpoint(metric_entry)

    test_accuracy = metrics[-1]["val_accuracy"] if metrics else 0.0
    return metrics, test_accuracy


def _format_sse(event_name, data):
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


def _persist_model_weights(model_id: str, model: nn.Module) -> Path:
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu")
    output_path = _model_file_path(model_id)
    torch.save(model_cpu.state_dict(), output_path)
    return output_path


def _collect_sample_predictions(model: nn.Module, data_loader, limit: int = 8):
    samples = []
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()

    total_collected = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            batch_size = inputs.size(0)
            for idx in range(batch_size):
                image_tensor = inputs[idx].detach().cpu()
                if image_tensor.dim() == 3 and image_tensor.size(0) == 1:
                    image_tensor = image_tensor.squeeze(0)
                grid = image_tensor.mul(255).clamp(0, 255).to(torch.uint8).tolist()
                confidence = float(
                    probabilities[idx, predictions[idx]].detach().cpu().item()
                )
                samples.append(
                    {
                        "grid": grid,
                        "label": int(targets[idx].detach().cpu().item()),
                        "prediction": int(predictions[idx].detach().cpu().item()),
                        "confidence": confidence,
                    }
                )
                total_collected += 1
                if total_collected >= limit:
                    break
            if total_collected >= limit:
                break

    return samples


@app.route("/api/models/save", methods=["POST"])
def save_trained_model():
    if not request.is_json:
        return _error_response("Expected JSON payload.", status=415)

    try:
        payload = request.get_json(force=True)
    except Exception:
        return _error_response("Malformed JSON payload.")

    if not isinstance(payload, dict):
        return _error_response("Payload must be a JSON object.")

    run_id = payload.get("run_id")
    model_name = payload.get("name")

    if not isinstance(run_id, str) or not run_id:
        return _error_response("`run_id` is required.", status=400)

    run_entry = store.get_run(run_id)
    if run_entry is None:
        return _error_response("Run does not exist.", status=404)

    if run_entry.get("state") != "succeeded":
        return _error_response("Run is not ready for saving.", status=409)

    saved_model_path = run_entry.get("saved_model_path")
    if not saved_model_path:
        return _error_response("Model weights have not been persisted.", status=409)

    output_path = Path(saved_model_path)
    if not output_path.exists():
        return _error_response("Persisted model file is missing.", status=500)

    # Create new model
    model_id = _generate_id("m")
    created_at = _utcnow_iso()
    architecture = run_entry.get("architecture", {})
    hyperparams = run_entry.get("hyperparams", {})

    # Rename model file from run_id to model_id
    new_model_path = _model_file_path(model_id)
    output_path.rename(new_model_path)

    model_entry = {
        "model_id": model_id,
        "name": model_name,
        "description": None,
        "architecture": copy.deepcopy(architecture),
        "hyperparams": copy.deepcopy(hyperparams),
        "created_at": created_at,
        "trained": True,
        "saved_model_path": str(new_model_path),
        "last_trained_at": created_at,
    }
    store.add_model(model_id, model_entry)

    # Link run to model
    store.update_run(
        run_id,
        {
            "model_id": model_id,
            "saved_model_path": str(new_model_path),
        },
    )

    response = {
        "model_id": model_id,
        "run_id": run_id,
        "saved_path": str(new_model_path),
        "trained": True,
        "name": model_name,
        "architecture": copy.deepcopy(architecture),
        "hyperparams": copy.deepcopy(hyperparams),
    }

    return jsonify(response), 201


def _start_training_thread(model_id, run_id, architecture, hyperparams):
    event_queue = queue.Queue()
    store.add_event_queue(run_id, event_queue)

    def emit(event_name, data):
        payload = dict(data)
        payload.setdefault("run_id", run_id)
        event_queue.put({"event": event_name, "data": payload})

    def worker():
        try:
            model = build_model(architecture)
            train_loader, val_loader = prepare_dataloaders(
                hyperparams["batch_size"],
                hyperparams["train_split"],
                hyperparams["shuffle"],
                hyperparams["max_samples"],
                hyperparams["seed"],
            )

            captured_metrics = []

            store.update_run(run_id, {"state": "running"})

            emit("state", {"state": "running"})

            def on_checkpoint(metric):
                metric_copy = dict(metric)
                captured_metrics.append(metric_copy)
                emit("metric", metric_copy)
                store.update_run(
                    run_id,
                    {
                        "metrics": list(captured_metrics),
                        "epoch": metric_copy["epoch"],
                    },
                )

            metrics, test_accuracy = _train_with_torch(
                model,
                train_loader,
                val_loader,
                hyperparams,
                on_checkpoint=on_checkpoint,
            )

            sample_predictions = _collect_sample_predictions(model, val_loader, limit=8)

            # Save model weights to temporary location
            temp_model_id = run_id  # Use run_id for temporary storage
            output_path = _persist_model_weights(temp_model_id, model)
            completed_at = _utcnow_iso()
            store.update_run(
                run_id,
                {
                    "state": "succeeded",
                    "metrics": metrics,
                    "test_accuracy": test_accuracy,
                    "completed_at": completed_at,
                    "saved_model_path": str(output_path),
                    "sample_predictions": sample_predictions,
                },
            )

            emit(
                "state",
                {
                    "state": "succeeded",
                    "test_accuracy": test_accuracy,
                    "sample_predictions": sample_predictions,
                },
            )
        except Exception as exc:
            error_message = str(exc)
            logger.error(f"Training failed for run {run_id}: {error_message}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            store.update_run(
                run_id,
                {
                    "state": "failed",
                    "error": error_message,
                    "completed_at": _utcnow_iso(),
                },
            )
            emit("state", {"state": "failed", "error": error_message})
        finally:
            event_queue.put(None)
            store.remove_event_queue(run_id)

    # Emit initial queued state before the worker starts.
    emit("state", {"state": "queued"})
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


def _error_response(message, status=400):
    return jsonify({"error": message}), status


@app.route("/api/train", methods=["POST"])
def train_model():
    if not request.is_json:
        return _error_response("Expected JSON payload.", status=415)

    try:
        payload = request.get_json(force=True)
    except Exception:
        return _error_response("Malformed JSON payload.")

    if not isinstance(payload, dict):
        return _error_response("Payload must be a JSON object.")

    architecture_raw = payload.get("architecture")
    hyperparams_raw = payload.get("hyperparams")

    try:
        architecture = validate_architecture(architecture_raw)
        hyperparams = validate_hyperparams(hyperparams_raw)
    except ValueError as exc:
        return _error_response(str(exc))

    run_id = _generate_id("r")
    created_at = _utcnow_iso()

    # Work with deep copies to avoid sharing references across threads.
    architecture = json.loads(json.dumps(architecture))
    hyperparams = json.loads(json.dumps(hyperparams))

    store.add_run(
        run_id,
        {
            "run_id": run_id,
            "model_id": None,
            "state": "queued",
            "epochs_total": hyperparams["epochs"],
            "metrics": [],
            "test_accuracy": None,
            "created_at": created_at,
            "events_url": f"/api/runs/{run_id}/events",
            "hyperparams": hyperparams,
            "architecture": architecture,
            "saved_model_path": None,
            "sample_predictions": [],
        },
    )

    _start_training_thread(None, run_id, architecture, hyperparams)

    response = {
        "run_id": run_id,
        "status": "queued",
        "created_at": created_at,
        "epochs_total": hyperparams["epochs"],
        "metrics": [],
        "test_accuracy": None,
        "events_url": f"/api/runs/{run_id}/events",
    }

    return jsonify(response), 202


@app.route("/api/infer", methods=["POST"])
def infer_single_pixel_map():
    if not request.is_json:
        return _error_response("Expected JSON payload.", status=415)

    try:
        payload = request.get_json(force=True)
    except Exception:
        return _error_response("Malformed JSON payload.")

    if not isinstance(payload, dict):
        return _error_response("Payload must be a JSON object.")

    run_id = payload.get("run_id")
    pixels = payload.get("pixels")

    if not isinstance(run_id, str) or not run_id:
        return _error_response("`run_id` is required.", status=400)

    try:
        input_tensor = tensor_from_pixels(pixels)
    except ValueError as exc:
        return _error_response(str(exc), status=422)

    run_entry = store.get_run(run_id)
    if run_entry is None:
        return _error_response("Unknown run_id.", status=404)
    state = run_entry.get("state")
    saved_model_path = run_entry.get("saved_model_path")
    model_id = run_entry.get("model_id")

    model_entry = store.get_model(model_id) if model_id else None

    if state != "succeeded":
        return _error_response("Run is not ready for inference.", status=409)
    if model_entry is None:
        return _error_response("Associated model not found.", status=404)
    if not saved_model_path:
        return _error_response("Persisted model file not available.", status=409)

    model_path = Path(saved_model_path)
    if not model_path.exists():
        return _error_response("Persisted model file is missing.", status=500)

    architecture = model_entry["architecture"]
    model = build_model(architecture)

    try:
        state_dict = torch.load(model_path, map_location="cpu")
    except Exception:
        return _error_response("Failed to load persisted model.", status=500)

    model.load_state_dict(state_dict)
    del state_dict
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_label = int(probabilities.argmax().item())
    response = {
        "run_id": run_id,
        "label": predicted_label,
        "probabilities": [float(p) for p in probabilities.tolist()],
    }

    return jsonify(response), 200


@app.route("/api/runs/<run_id>/events", methods=["GET"])
def stream_run_events(run_id):
    run = store.get_run(run_id)
    event_queue = store.get_event_queue(run_id)

    if run is None:
        return _error_response("Unknown run_id.", status=404)

    def event_generator():
        if event_queue is None:
            for metric in run.get("metrics", []):
                yield _format_sse("metric", {"run_id": run_id, **metric})
            yield _format_sse(
                "state",
                {
                    "run_id": run_id,
                    "state": run.get("state"),
                    "test_accuracy": run.get("test_accuracy"),
                    "error": run.get("error"),
                },
            )
            return

        while True:
            try:
                item = event_queue.get(timeout=1.0)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield _format_sse(item["event"], item["data"])

    return Response(
        stream_with_context(event_generator()), mimetype="text/event-stream"
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)
