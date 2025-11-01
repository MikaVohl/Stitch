from flask import Blueprint, jsonify
from store import store

model_bp = Blueprint("model", __name__)


@model_bp.route("/api/models", methods=["GET"])
def list_models():
    models = store.list_models()
    return jsonify(models), 200


# Create and name a model
@model_bp.route("/api/models", methods=["POST"])
def create_model():
    pass


# Get Model and Runs
@model_bp.route("/api/models/<id>", methods=["GET"])
def get_model(id: str):
    pass
