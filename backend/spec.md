<!-- # API

Base URL: `/api`
Content type: `application/json`

## JSON types

### Architecture

```
{
  "input_size": 784,
  "layers": [
    {"type": "linear", "in": 784, "out": 128},
    {"type": "relu"},
    {"type": "linear", "in": 128, "out": 10}
  ]
}
```

Rules:

* Allowed: `linear`, `relu`, `sigmoid`, `tanh`, `softmax` (softmax only as final layer).
* Validate dimension chaining. Final output must be 10 for MNIST.

### Hyperparameters

```
{
  "epochs": 5,
  "batch_size": 64,
  "optimizer": {"type": "sgd", "lr": 0.1, "momentum": 0.0},
  "loss": "cross_entropy",
  "seed": 42,
  "train_split": 0.9,
  "shuffle": true
}
```

### Parameters

```
{
  "layers": [
    {"type": "linear", "W": [[...],[...]], "b": [...]},
    {"type": "relu"},
    {"type": "linear", "W": [[...],[...]], "b": [...]}
  ],
  "dtype": "float32",
  "layout": "row_major"
}
```

## Identifiers

* `model_id`: unique UUID for each created model definition
* `run_id`: UUID for each training job
* `param_id`: UUID for each trained parameter set

IDs are purely for tracking; they are not reused or deduplicated.

## Endpoints

### 1) Create a model

`POST /models`

* Body: `{ "architecture": {...}, "hyperparams": {...} }`
* Creates a new model definition.
* 200 Response:

```
{
  "model_id": "m_...",
  "architecture": {...},
  "hyperparams": {...},
  "created_at": "ISO8601"
}
```

* 400 on invalid dimensions.

### 2) Start training

`POST /models/{model_id}/train`

* Body: optional `{ "hyperparams": {...} }`
* 202 Response:

```
{
  "run_id": "r_...",
  "model_id": "m_...",
  "status": "queued"
}
```

### 3) Check training status

`GET /runs/{run_id}`

* 200 Response:

```
{
  "run_id": "r_...",
  "model_id": "m_...",
  "state": "queued|running|succeeded|failed",
  "epoch": 3,
  "epochs_total": 5,
  "metrics": {
    "train_loss": 0.23,
    "train_accuracy": 0.93,
    "val_loss": 0.20,
    "val_accuracy": 0.94
  },
  "param_id": "p_..."  // only if succeeded
}
```

### 4) List runs for a model

`GET /models/{model_id}/runs?limit=20&cursor=...`

* 200 Response:

```
{ "items": [ {run...}, ... ], "next_cursor": "..." }
```

### 5) Get parameters

`GET /params/{param_id}`

* 200 Response:

```
{
  "param_id": "p_...",
  "model_id": "m_...",
  "parameters": {...},
  "size_bytes": 123456
}
```

### 6) Inference using trained parameters

`POST /infer`

* Body:

```
{
  "param_id": "p_...",
  "inputs": [[...784 floats...], ...],
  "return": { "logits": false, "probabilities": true, "labels": true }
}
```

* 200 Response:

```
{
  "model_id": "m_...",
  "param_id": "p_...",
  "outputs": {
    "labels": [7,2,1,...],
    "probabilities": [[0.01, ..., 0.97], ...]
  },
  "latency_ms": 3
}
```

* 422 on invalid input shape.

### 7) Ad-hoc inference

`POST /infer_adhoc`

* Body:

```
{
  "architecture": {...},
  "parameters": {...},
  "inputs": [[...], ...],
  "return": {"probabilities": true, "labels": true}
}
```

* Performs direct forward pass without saving any artifacts.

## Errors

* 400 invalid JSON or shape mismatch
* 404 unknown `model_id`, `run_id`, or `param_id`
* 409 inconsistent hyperparameters for an existing run
* 422 input validation failed
* 500 internal error

## Limits

* Max batch size for `/infer`: 512
* Max layers: 5 linear layers
* Dtypes: float32 only

## Security

* All endpoints require `Authorization: Bearer <token>`
* Enforce upload size limits

## Minimal flow

1. `POST /models` → get `model_id`
2. `POST /models/{model_id}/train` → get `run_id`
3. Poll `GET /runs/{run_id}` until `succeeded`
4. `POST /infer` with `param_id` and inputs -->