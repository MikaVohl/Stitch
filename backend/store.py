import threading
from typing import Any, Optional


class Store:
    """Thread-safe in-memory store for models, runs, and event queues."""

    def __init__(self):
        self._lock = threading.Lock()
        self._models = {}
        self._runs = {}
        self._run_event_queues = {}

    # Model operations
    def add_model(self, model_id: str, model_data: dict) -> None:
        """Add a model to the store."""
        with self._lock:
            self._models[model_id] = model_data

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get a model by ID."""
        with self._lock:
            return self._models.get(model_id)

    def update_model(self, model_id: str, updates: dict) -> None:
        """Update a model with new data."""
        with self._lock:
            model = self._models.get(model_id)
            if model is not None:
                model.update(updates)

    def list_models(self) -> list:
        """List all models."""
        with self._lock:
            return list(self._models.values())

    # Run operations
    def add_run(self, run_id: str, run_data: dict) -> None:
        """Add a run to the store."""
        with self._lock:
            self._runs[run_id] = run_data

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a run by ID."""
        with self._lock:
            return self._runs.get(run_id)

    def update_run(self, run_id: str, updates: dict) -> None:
        """Update a run with new data."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                run.update(updates)

    def list_runs(self, model_id: Optional[str] = None) -> list:
        """List all runs, optionally filtered by model_id."""
        with self._lock:
            runs = list(self._runs.values())
            if model_id:
                runs = [r for r in runs if r.get("model_id") == model_id]
            return runs

    # Event queue operations
    def add_event_queue(self, run_id: str, queue: Any) -> None:
        """Add an event queue for a run."""
        with self._lock:
            self._run_event_queues[run_id] = queue

    def get_event_queue(self, run_id: str) -> Optional[Any]:
        """Get an event queue by run ID."""
        with self._lock:
            return self._run_event_queues.get(run_id)

    def remove_event_queue(self, run_id: str) -> None:
        """Remove an event queue for a run."""
        with self._lock:
            self._run_event_queues.pop(run_id, None)


# Global singleton instance
store = Store()
