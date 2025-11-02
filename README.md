# STITCH – Interactive MNIST Playground

This project combines a React/TypeScript front end with a FastAPI/PyTorch back end to let users build, train, inspect, and test MNIST digit classifiers. It powers an interactive UI where you can draw digits, launch hyper‑parameter sweeps, inspect layer architectures, and visualize how pixels flow through the network.

## At a Glance

- **Frontend**: React 19 + Vite + Tailwind + React Flow for rich network visualizations.
- **Backend**: FastAPI + PyTorch for model management, training, and inference.
- **Storage**: Lightweight file system persistence (pickled PyTorch models plus JSON metadata).
- **Tooling**: pnpm for front-end dependencies, uv/pip for Python, and React Query for data fetching.

## Repository Layout

```
.
├── backend/                # FastAPI application code
│   ├── api.py              # App entry point & route registration
│   ├── controllers/        # Request handlers
│   ├── services/           # Training/inference logic (PyTorch)
│   ├── saved_models/       # Serialized model checkpoints (*.pkl)
│   └── requirements.txt    # Python dependency list
├── frontend/               # React client
│   ├── public/             # Static assets (SVGs, favicons)
│   ├── src/                # Application source
│   │   ├── components/     # Reusable UI and visualization pieces
│   │   ├── hooks/          # Data fetching & model hooks
│   │   └── routes/         # Page-level components
│   └── package.json        # Front-end scripts & dependencies
├── .env                    # Sample environment variables (optional)
└── README.md               # You are here
```

## Prerequisites

- **Node.js** ≥ 20 (pnpm recommended)
- **Python** ≥ 3.10
- (Optional) **uv** for faster Python environment creation

## Quick Start

Clone the repo and install both stacks:

```bash
git clone https://github.com/your-org/mais-2025.git
cd mais-2025
```

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer [uv](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then start the API:

```bash
uvicorn api:app --reload
```

The server defaults to `http://127.0.0.1:8000`.

### Frontend Setup

```bash
cd frontend
pnpm install
pnpm dev
```

The Vite dev server runs on `http://127.0.0.1:5173`.

> Make sure the backend is running before using pages that fetch data (Models, Playground, Test).

## Environment Variables

Create a `.env` in the project root if you need overrides. Common keys:

```
BACKEND_PORT=8000
BACKEND_HOST=127.0.0.1
FRONTEND_PORT=5173
```

The frontend proxies `/api/*` to the FastAPI server during development (see `frontend/vite.config.ts`).

## Key Features

- **Model Catalog**: Browse trained models, inspect architecture layers, and jump straight to testing with deep links (`/test/:modelId`).
- **Interactive Drawing Grid**: Draw digits with a pixel brush; the UI flattens and normalizes them to MNIST format before inference.
- **Visualization**: React Flow graph showing:
  - 28×28 input pixel nodes
  - Sampled convolutions & dense layers
  - Operations (max pool, dropout, flatten) as labeled boxes
  - Highlighted output neuron corresponding to the latest prediction
- **Inference Runner**: Converts the canvas drawing to a tensor, calls `/api/infer`, and surfaces predictions inline, lighting up the relevant output node.
- **Training Hooks**: Model metadata includes hyperparameters, runs, and saved checkpoints; the backend service abstracts loading, training, and inference.

## Useful Scripts

| Location   | Command                  | Purpose                                    |
| ---------- | ------------------------ | ------------------------------------------ |
| `frontend` | `pnpm dev`               | Run Vite development server                |
|            | `pnpm build`             | Production build                           |
|            | `pnpm lint` *(if added)* | Run linting (configure ESLint if desired)  |
| `backend`  | `uvicorn api:app --reload` | Start FastAPI in auto-reload mode        |
|            | `pytest` *(if added)*    | Run backend tests                          |

## API Overview

The FastAPI app exposes run-time model management endpoints (simplified view):

- `GET /api/models` – List all stored models with metadata.
- `GET /api/models/{model_id}` – Detailed view including runs.
- `POST /api/models` – Create a new model definition.
- `POST /api/train` – Launch training for a model/run configuration.
- `POST /api/infer` – Run inference with `run_id` + flattened pixel array.

Check `backend/api.py` and `backend/controllers/` for full route implementations.

## Testing & Quality

There are no automated tests bundled yet. To add coverage:

- Frontend: configure Jest/Testing Library or Vitest.
- Backend: add unit/integration tests under `backend/tests/` and run with `pytest`.

Linting (ESLint/Prettier, Ruff) can be wired in via `pnpm lint` or `pre-commit` hooks.

## Troubleshooting

- **Port already in use (5173/8000)**: stop conflicting processes or change ports in `.env` / CLI.
- **Inference returns uniform values**: ensure input normalization matches training (`flattenDrawing` -> `[0, 1]`, backend expects normalized tensors).
- **Missing successful runs**: the inference UI filters training runs by state; confirm at least one run has `state === "succeeded"` and a saved checkpoint.

## Contributing

1. Create a new branch off `main`.
2. Tackle your feature or fix; keep commits scoped.
3. Run relevant tests/build steps.
4. Submit a PR with context and screenshots when appropriate.

## License

Specify the project license here (MIT, Apache 2.0, etc.). If none has been chosen yet, consider adding one before distributing the code.

---

Happy experimenting! Feel free to open issues or discussions for questions, bug reports, or feature proposals.
