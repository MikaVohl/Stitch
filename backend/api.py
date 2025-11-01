from datetime import datetime, timezone
import threading
import uuid
import random

from flask import Flask, jsonify, request

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
except ModuleNotFoundError as exc:
    raise RuntimeError("PyTorch is required to run the training service.") from exc

try:
    from torchvision import datasets, transforms

    _HAS_TORCHVISION = True
except ModuleNotFoundError:
    _HAS_TORCHVISION = False

"""
Expected request payload:
{
  "architecture": {
    "input_size": 784,
    "layers": [
      {"type": "linear", "in": 784, "out": 128},
      {"type": "relu"},
      {"type": "linear", "in": 128, "out": 10}
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

ALLOWED_LAYER_TYPES = {"linear", "relu", "sigmoid", "tanh", "softmax"}
ALLOWED_LOSSES = {"cross_entropy"}
ALLOWED_OPTIMIZERS = {"sgd", "adam"}
MNIST_INPUT_SIZE = 28 * 28
DATA_ROOT = "data"

# In-memory store for models and runs to keep the example self-contained.
_store_lock = threading.Lock()
_models = {}
_runs = {}

app = Flask(__name__)


class ValidationError(ValueError):
    """Raised when incoming payload fails validation rules."""


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _validate_architecture(payload):
    if not isinstance(payload, dict):
        raise ValidationError("`architecture` must be an object.")

    if "input_size" not in payload:
        raise ValidationError("`architecture.input_size` is required.")
    if "layers" not in payload:
        raise ValidationError("`architecture.layers` is required.")

    input_size = payload["input_size"]
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValidationError("`architecture.input_size` must be a positive integer.")
    if input_size != MNIST_INPUT_SIZE:
        raise ValidationError("`architecture.input_size` must be 784 for MNIST images.")

    layers = payload["layers"]
    if not isinstance(layers, list) or not layers:
        raise ValidationError("`architecture.layers` must be a non-empty list.")

    sanitized_layers = []
    prev_dim = input_size
    last_linear_out = None

    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValidationError(f"Layer {idx} must be an object.")

        layer_type = layer.get("type")
        if layer_type not in ALLOWED_LAYER_TYPES:
            raise ValidationError(f"Layer {idx} has unsupported type `{layer_type}`.")

        if layer_type == "linear":
            in_dim = layer.get("in")
            out_dim = layer.get("out")
            if not isinstance(in_dim, int) or not isinstance(out_dim, int):
                raise ValidationError(f"`linear` layer {idx} must define integer `in` and `out`.")
            if in_dim <= 0 or out_dim <= 0:
                raise ValidationError(f"`linear` layer {idx} must have positive `in` and `out`.")
            if in_dim != prev_dim:
                raise ValidationError(
                    f"`linear` layer {idx} expected `in={prev_dim}` but received `in={in_dim}`."
                )
            prev_dim = out_dim
            last_linear_out = out_dim
            sanitized_layers.append({"type": "linear", "in": in_dim, "out": out_dim})
        else:
            if layer_type == "softmax" and idx != len(layers) - 1:
                raise ValidationError("`softmax` layers are only allowed at the final position.")
            sanitized_layers.append({"type": layer_type})

    if last_linear_out is None:
        raise ValidationError("Architecture must include at least one `linear` layer.")

    if last_linear_out != 10:
        raise ValidationError("Final linear layer must produce 10 outputs for MNIST.")

    if layers[-1]["type"] == "softmax" and prev_dim != 10:
        raise ValidationError("`softmax` output must operate on 10 logits.")

    return {"input_size": input_size, "layers": sanitized_layers}


def _validate_hyperparams(payload):
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValidationError("`hyperparams` must be an object.")

    result = {
        "epochs": 5,
        "batch_size": 64,
        "optimizer": {"type": "sgd", "lr": 0.1, "momentum": 0.0},
        "loss": "cross_entropy",
        "seed": None,
        "train_split": 0.9,
        "shuffle": True,
        "max_samples": 4096,
    }

    if "epochs" in payload:
        epochs = payload["epochs"]
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValidationError("`hyperparams.epochs` must be a positive integer.")
        if epochs > 2000:
            raise ValidationError("`hyperparams.epochs` must be <= 2000 to protect server load.")
        result["epochs"] = epochs

    if "batch_size" in payload:
        batch_size = payload["batch_size"]
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValidationError("`hyperparams.batch_size` must be a positive integer.")
        result["batch_size"] = batch_size

    if "train_split" in payload:
        train_split = payload["train_split"]
        if not isinstance(train_split, (float, int)) or not (0 < float(train_split) < 1):
            raise ValidationError("`hyperparams.train_split` must be between 0 and 1.")
        result["train_split"] = float(train_split)

    if "shuffle" in payload:
        shuffle = payload["shuffle"]
        if not isinstance(shuffle, bool):
            raise ValidationError("`hyperparams.shuffle` must be a boolean.")
        result["shuffle"] = shuffle

    if "loss" in payload:
        loss = payload["loss"]
        if not isinstance(loss, str) or loss.lower() not in ALLOWED_LOSSES:
            raise ValidationError(f"`hyperparams.loss` must be one of {sorted(ALLOWED_LOSSES)}.")
        result["loss"] = loss.lower()

    if "seed" in payload:
        seed = payload["seed"]
        if seed is not None and not isinstance(seed, int):
            raise ValidationError("`hyperparams.seed` must be an integer or null.")
        result["seed"] = seed

    if "max_samples" in payload:
        max_samples = payload["max_samples"]
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValidationError("`hyperparams.max_samples` must be a positive integer.")
        result["max_samples"] = max_samples

    optimizer = payload.get("optimizer")
    if optimizer is not None:
        if not isinstance(optimizer, dict):
            raise ValidationError("`hyperparams.optimizer` must be an object.")
        opt_type = optimizer.get("type")
        if not isinstance(opt_type, str):
            raise ValidationError("`hyperparams.optimizer.type` is required.")
        opt_type = opt_type.lower()
        if opt_type not in ALLOWED_OPTIMIZERS:
            raise ValidationError(f"`hyperparams.optimizer.type` must be one of {sorted(ALLOWED_OPTIMIZERS)}.")
        sanitized_optimizer = {"type": opt_type}

        lr = optimizer.get("lr", result["optimizer"].get("lr"))
        if lr is not None:
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValidationError("`hyperparams.optimizer.lr` must be a positive number.")
            sanitized_optimizer["lr"] = float(lr)

        if opt_type == "sgd":
            momentum = optimizer.get("momentum", result["optimizer"].get("momentum", 0.0))
            if not isinstance(momentum, (int, float)) or not (0 <= momentum < 1):
                raise ValidationError("`hyperparams.optimizer.momentum` must be in [0, 1) for SGD.")
            sanitized_optimizer["momentum"] = float(momentum)
        elif opt_type == "adam":
            beta1 = optimizer.get("beta1", 0.9)
            beta2 = optimizer.get("beta2", 0.999)
            eps = optimizer.get("eps", 1e-8)
            for name, value in [("beta1", beta1), ("beta2", beta2), ("eps", eps)]:
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValidationError(f"`hyperparams.optimizer.{name}` must be a positive number.")
            sanitized_optimizer["beta1"] = float(beta1)
            sanitized_optimizer["beta2"] = float(beta2)
            sanitized_optimizer["eps"] = float(eps)

        result["optimizer"] = sanitized_optimizer

    return result


def _build_model(architecture):
    layers = []
    for layer_spec in architecture["layers"]:
        layer_type = layer_spec["type"]
        if layer_type == "linear":
            layers.append(nn.Linear(layer_spec["in"], layer_spec["out"]))
        elif layer_type == "relu":
            layers.append(nn.ReLU())
        elif layer_type == "sigmoid":
            layers.append(nn.Sigmoid())
        elif layer_type == "tanh":
            layers.append(nn.Tanh())
        elif layer_type == "softmax":
            layers.append(nn.Softmax(dim=1))
        else:
            raise ValidationError(f"Unsupported layer type `{layer_type}`.")
    return nn.Sequential(*layers)


def _synthetic_dataset(size, input_size, seed):
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    data = torch.rand(size, input_size, generator=generator)
    labels = torch.randint(0, 10, (size,), generator=generator)
    return TensorDataset(data, labels)


def _prepare_dataloaders(batch_size, train_split, shuffle, max_samples, seed):
    if _HAS_TORCHVISION:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        try:
            dataset = datasets.MNIST(
                root=DATA_ROOT, train=True, download=True, transform=transform
            )
        except Exception:
            dataset = _synthetic_dataset(max_samples, MNIST_INPUT_SIZE, seed)
    else:
        dataset = _synthetic_dataset(max_samples, MNIST_INPUT_SIZE, seed)

    if max_samples and len(dataset) > max_samples:
        indices = list(range(len(dataset)))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        dataset = Subset(dataset, indices[:max_samples])

    train_len = max(1, int(len(dataset) * train_split))
    val_len = len(dataset) - train_len
    if val_len == 0:
        val_len = 1
        train_len = len(dataset) - 1

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset, [train_len, val_len], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _configure_optimizer(optimizer_cfg, parameters):
    opt_type = optimizer_cfg["type"]
    lr = optimizer_cfg.get("lr", 0.001)
    if opt_type == "sgd":
        momentum = optimizer_cfg.get("momentum", 0.0)
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    if opt_type == "adam":
        beta1 = optimizer_cfg.get("beta1", 0.9)
        beta2 = optimizer_cfg.get("beta2", 0.999)
        eps = optimizer_cfg.get("eps", 1e-8)
        return torch.optim.Adam(parameters, lr=lr, betas=(beta1, beta2), eps=eps)
    raise ValidationError(f"Unsupported optimizer `{opt_type}`.")


def _train_with_torch(model, train_loader, val_loader, hyperparams):
    device = torch.device("cpu")
    model.to(device)

    if hyperparams["seed"] is not None:
        torch.manual_seed(hyperparams["seed"])
        random.seed(hyperparams["seed"])

    criterion = nn.CrossEntropyLoss()
    optimizer = _configure_optimizer(hyperparams["optimizer"], model.parameters())

    epochs = hyperparams["epochs"]
    metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
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
                inputs = inputs.view(inputs.size(0), -1).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += inputs.size(0)

        avg_val_loss = val_loss / max(1, val_total)
        val_accuracy = val_correct / max(1, val_total)

        if epoch % 100 == 0 or epoch == epochs:
            metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": round(avg_train_loss, 4),
                    "val_loss": round(avg_val_loss, 4),
                    "train_accuracy": round(train_accuracy, 4),
                    "val_accuracy": round(val_accuracy, 4),
                }
            )

    test_accuracy = metrics[-1]["val_accuracy"] if metrics else 0.0
    return metrics, test_accuracy


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
        architecture = _validate_architecture(architecture_raw)
        hyperparams = _validate_hyperparams(hyperparams_raw)
    except ValidationError as exc:
        return _error_response(str(exc))

    model_id = _generate_id("m")
    run_id = _generate_id("r")
    created_at = _utcnow_iso()

    model = _build_model(architecture)
    train_loader, val_loader = _prepare_dataloaders(
        hyperparams["batch_size"],
        hyperparams["train_split"],
        hyperparams["shuffle"],
        hyperparams["max_samples"],
        hyperparams["seed"],
    )

    metrics, test_accuracy = _train_with_torch(model, train_loader, val_loader, hyperparams)

    with _store_lock:
        _models[model_id] = {
            "model_id": model_id,
            "architecture": architecture,
            "hyperparams": hyperparams,
            "created_at": created_at,
        }
        _runs[run_id] = {
            "run_id": run_id,
            "model_id": model_id,
            "state": "succeeded",
            "epochs_total": hyperparams["epochs"],
            "metrics": metrics,
            "test_accuracy": test_accuracy,
            "completed_at": _utcnow_iso(),
        }

    response = {
        "model_id": model_id,
        "run_id": run_id,
        "status": "succeeded",
        "created_at": created_at,
        "epochs_total": hyperparams["epochs"],
        "metrics": metrics,
        "test_accuracy": test_accuracy,
    }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True)
