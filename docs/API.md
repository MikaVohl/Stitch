# API Reference

All endpoints are served from the Flask backend. When running the development stack via Vite, requests to `/api/**` are proxied to `http://127.0.0.1:8080`.

## Models

### `GET /api/models`
Returns a list of model summaries currently tracked in the in-memory store.

Each summary contains:
- `model_id`, `name`, `description`
- stored `architecture` and `hyperparams`
- training flags such as `trained`, `saved_model_path`, `saved_model_exists`, `last_trained_at`
- `runs_total` plus the most recent succeeded run metadata

### `POST /api/models`
Create a new model or update an existing definition.

**Body**
```jsonc
{
  "model_id": "optional explicit id",
  "name": "optional",
  "description": "optional",
  "architecture": { /* optional JSON describing layers */ },
  "hyperparams": { /* optional JSON describing hyperparameters */ },
  "trained": false,             // optional flag when updating
  "saved_model_path": "/path/to/model.pt" // optional when updating
}
```
- If `model_id` is omitted a new ID `m_<uuid>` is generated.
- When updating, only provided fields are touched.
- Returns the updated summary. Status `201` for creation, `200` for update.

### `GET /api/models/<id>`
Fetch the summary for a single model, including its recorded training runs.

Returns `404` if the model is unknown.

### `POST /api/models/<model_id>/save`
Persist a model definition or finalize a trained run.

This endpoint supports two workflows:

1. **Register / update an untrained model definition**
   ```json
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
   ```
   - No `run_id` is supplied.
   - Stores the architecture/hyperparameters under `model_id`, marks `trained: false`, and clears any previous `saved_model_path`.
   - Returns the stored definition (status `200`).

2. **Promote a completed training run**
   ```json
   { "run_id": "r_<uuid>" }
   ```
   - Requires that the run exists, is associated with the model, finished with `state: "succeeded"`, and has a persisted weight file.
   - Marks the model as trained, records `saved_model_path`, and echoes back architecture/hyperparameters alongside `run_id`.

## Training

### `POST /api/train`
Launch an asynchronous training job for a model definition.

**Body**
```json
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
```
- Returns `202 Accepted` with a payload containing the generated `model_id`, `run_id`, and `events_url` (`/api/runs/<run_id>/events`).
- Training runs asynchronously in a background thread. Intermediate state and metrics are stored in the in-memory store and streamed via Server-Sent Events.
- SSE `state` events can progress through `queued → running → succeeded/failed/cancelled`.
- Only one training run may be active at a time; additional requests while a run is in progress return `409 Conflict`.

### `POST /api/train/<run_id>/cancel`
Request cancellation of an active training run.

- Only runs in `queued` or `running` state can be cancelled.
- Returns `202 Accepted` when the cancellation signal has been issued.
- A `state` event with `{"state": "cancelled"}` is emitted once the worker stops.

### `GET /api/runs/<run_id>/events`
Server-Sent Events (SSE) stream for a training run.

- Emits historical metrics/states immediately if the run has already completed.
- While training is active, metrics (`event: metric`) and state transitions (`event: state`) are pushed as soon as they are produced.
- Idle connections receive periodic `: keep-alive` comments to keep proxies from buffering.
- Response headers set `Cache-Control: no-cache`, `X-Accel-Buffering: no`, and `Connection: keep-alive`.

### `POST /api/infer`
Run inference using a trained model produced by a completed run.

**Body**
```json
{
  "run_id": "r_<uuid from training response>",
  "pixels": [0.0, 0.1, ... , 0.9] // length must be 784 (28x28 flattened)
}
```
- Requires that the referenced run has state `succeeded` and its `saved_model_path` exists on disk.
- Response (`200 OK`):
  ```json
  {
    "run_id": "r_<uuid>",
    "label": 3,
    "probabilities": [0.01, 0.02, ..., 0.75]
  }
  ```
- Validation failures return `400`/`422`; missing models or weights return `404`/`409`/`500` accordingly.

## Error Format

Errors are returned as JSON:
```json
{ "error": "Human-readable message." }
```
The HTTP status code conveys the failure category (e.g., `400` validation, `404` unknown resources, `409` invalid state, `500` internal issues).
