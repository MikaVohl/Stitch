import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import math
from torch.utils.data import Subset
from torchvision import datasets, transforms
from pathlib import Path

from utils.validation import DEFAULT_HYPERPARAMS, MNIST_INPUT_SIZE


BACKEND_DIR = Path(__file__).resolve().parent
MNIST_DATA_ROOT = BACKEND_DIR / "data" / "mnist"


BACKEND_DIR = Path(__file__).resolve().parent
MNIST_DATA_ROOT = BACKEND_DIR / "data" / "mnist"


def build_model(architecture):
    layers = []
    for layer_spec in architecture["layers"]:
        layer_type = str(layer_spec["type"]).lower()
        if layer_type == "linear":
            layers.append(nn.Linear(layer_spec["in"], layer_spec["out"]))
        elif layer_type == "conv2d":
            padding = layer_spec.get("padding", 0)
            layers.append(
                nn.Conv2d(
                    layer_spec["in_channels"],
                    layer_spec["out_channels"],
                    kernel_size=layer_spec["kernel_size"],
                    stride=layer_spec.get("stride", 1),
                    padding=padding,
                )
            )
        elif layer_type == "flatten":
            layers.append(nn.Flatten())
        elif layer_type == "relu":
            layers.append(nn.ReLU())
        elif layer_type == "sigmoid":
            layers.append(nn.Sigmoid())
        elif layer_type == "tanh":
            layers.append(nn.Tanh())
        elif layer_type == "softmax":
            layers.append(nn.Softmax(dim=1))
        else:
            raise ValueError(f"Unsupported layer type `{layer_type}`.")
    return nn.Sequential(*layers)


def _load_mnist_dataset(train: bool):
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        dataset = datasets.MNIST(
            root=str(MNIST_DATA_ROOT),
            train=train,
            download=True,
            transform=transform,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load MNIST dataset: {exc}") from exc
    return dataset


def prepare_dataloaders(batch_size, train_split, shuffle, max_samples, seed):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    dataset = _load_mnist_dataset(train=True)
    dataset_size = len(dataset)

    desired_samples = max_samples or dataset_size
    desired_samples = max(desired_samples, batch_size * 2)
    desired_samples = min(desired_samples, dataset_size)
    desired_samples = max(2, desired_samples)

    if desired_samples < dataset_size:
        indices = torch.randperm(dataset_size, generator=generator)[:desired_samples]
        dataset = Subset(dataset, indices.tolist())

    train_len = max(1, int(len(dataset) * train_split))
    if train_len >= len(dataset):
        train_len = len(dataset) - 1
    val_len = max(1, len(dataset) - train_len)

    train_dataset, val_dataset = random_split(
        dataset, [train_len, val_len], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, generator=generator
    )
    return train_loader, val_loader


def configure_optimizer(optimizer_cfg, parameters):
    opt_type = str(
        optimizer_cfg.get("type", DEFAULT_HYPERPARAMS["optimizer"]["type"])
    ).lower()
    lr = float(optimizer_cfg.get("lr", DEFAULT_HYPERPARAMS["optimizer"]["lr"]))
    if opt_type == "sgd":
        momentum = float(
            optimizer_cfg.get(
                "momentum", DEFAULT_HYPERPARAMS["optimizer"].get("momentum", 0.0)
            )
        )
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    if opt_type == "adam":
        beta1 = float(optimizer_cfg.get("beta1", 0.9))
        beta2 = float(optimizer_cfg.get("beta2", 0.999))
        eps = float(optimizer_cfg.get("eps", 1e-8))
        return torch.optim.Adam(parameters, lr=lr, betas=(beta1, beta2), eps=eps)
    raise ValueError(f"Unsupported optimizer `{opt_type}`.")


def tensor_from_pixels(pixels):
    if not isinstance(pixels, (list, tuple)):
        raise ValueError("`pixels` must be a list of numbers.")
    if len(pixels) != MNIST_INPUT_SIZE:
        raise ValueError(f"`pixels` must contain exactly {MNIST_INPUT_SIZE} values.")
    try:
        flattened = [float(value) for value in pixels]
    except (TypeError, ValueError) as exc:
        raise ValueError("`pixels` must be numeric.") from exc
    side = int(round(math.sqrt(MNIST_INPUT_SIZE)))
    if side * side != MNIST_INPUT_SIZE:
        raise ValueError("Input size does not correspond to a square image.")
    tensor = torch.tensor(flattened, dtype=torch.float32).view(1, 1, side, side)
    return tensor
