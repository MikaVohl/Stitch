from flask import Blueprint, request, Response, stream_with_context
from utils.response import error_response
from openai import OpenAI
import os
import json
import traceback
import logging

chat_bp = Blueprint("chat", __name__)


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@chat_bp.route("/api/chat", methods=["POST"])
def chat_with_assistant():
    if not request.is_json:
        return error_response("Expected JSON payload.", status=415)

    try:
        payload = request.get_json(force=True)
    except Exception:
        return error_response("Malformed JSON payload.")

    if not isinstance(payload, dict):
        return error_response("Payload must be a JSON object.")

    message = payload.get("message")
    current_schema = payload.get("currentSchema")

    if not isinstance(message, str) or not message:
        return error_response("`message` is required.", status=400)

    # Get OpenAI API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"API Key: {api_key}")
    if not api_key:
        return error_response("OPENAI_API_KEY not configured.", status=500)

    client = OpenAI(api_key=api_key)

    # Build system prompt
    system_prompt = """You are Stitchy, a helpful AI assistant for a neural network architecture builder for MNIST.
Answer user questions conversationally and helpfully.

START General Info FOR USERS about Stitch:
Stitch is a no-code tool for building, running, and testing your own neural network. Think of it like scratch but for neural networks. Choose between different layer types, hyperparameters, and training configurations, all with a simple drag and drop interface. Stitch lets you observe how your model's performance changed during training, and test your models in real time with your own input, helping you learn the vital concepts in machine learning without the burden of a massive codebase. Your model will be trained and evalutated on the MNIST dataset, and after training you will be able to draw your own digits to test your model.
How to use: Drag and drop layers from the left panel onto the canvas to build your network. Connect layers by dragging from output to input ports. I can help you optimize your architecture, explain concepts, or suggest improvements!
END General Info FOR USERS about Stitch.

The builder supports CNN-focused workflows: users can place Input, Convolution, Pooling, Flatten (to vectorize), Dense, Dropout, and Output nodes to build MNIST classifiers.

IMPORTANT: Only propose architecture changes when the user EXPLICITLY asks you to modify, improve, add, remove, or change their neural network architecture.

Examples of when to propose changes:
- "add a dropout layer"
- "improve this model"
- "make it deeper"
- "change the architecture for image classification"
- "remove the second dense layer"

Examples of when NOT to propose changes (just answer conversationally):
- "what does this model do?"
- "how many parameters are there?"
- "explain this architecture"
- "what is a convolutional layer?"
- "tell me about neural networks"

When you DO propose changes:
1. Provide a clear, conversational explanation of WHAT you're changing and WHY
2. Focus on the neural network concepts, not implementation details
3. After your explanation, write exactly this sentence on its own line: "Certainly! Here is your current architecture schema, re-implemented as requested:"
4. Immediately after that sentence, include the complete JSON schema wrapped in a ```json code block

Schema format (use this but don't explain it to the user):
{
  "layers": {
    "layer-id": {
      "id": "layer-id",
      "kind": "Input" | "Convolution" | "Pooling" | "Flatten" | "Dense" | "Dropout" | "Output",
      "params": { ... },
      "position": { "x": number, "y": number }
    }
  },
  "edges": [
    {
      "id": "edge-id",
      "source": "source-layer-id",
      "target": "target-layer-id"
    }
  ]
}

Layer params:
- Input: { "size": number, "channels": 1, "height": 28, "width": 28 }
- Convolution: { "filters": number, "kernel": number, "stride": number, "padding": "same" | "valid", "activation": "relu" | "sigmoid" | "tanh" }
- Pooling: { "type": "max", "pool_size": number, "stride": number, "padding": number }
- Flatten: { }
- Dense: { "units": number, "activation": "relu" | "sigmoid" | "tanh" | "none" }
- Dropout: { "rate": number }
- Output: { "classes": number, "activation": "softmax" }

Only use these layer types and params—do not invent new kinds of nodes or fields.

Position layers horizontally with spacing (x: 0, 300, 600, etc., y: 200).
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Always include current schema for context
    if current_schema:
        schema_context = f"Current architecture:\n```json\n{json.dumps(current_schema, indent=2)}\n```\n\nUser question: {message}"
        messages.append({"role": "user", "content": schema_context})
    else:
        messages.append({"role": "user", "content": message})

    def event_generator():
        try:
            # TODO: Type errors
            stream = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                stream=True,
                temperature=0.7,
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield f"event: token\ndata: {json.dumps({'content': token})}\n\n"

            # Try to extract JSON schema from response
            import re

            json_match = re.search(
                r"```json\s*(\{.*?\})\s*```", full_response, re.DOTALL
            )
            if json_match:
                try:
                    proposed_schema = json.loads(json_match.group(1))
                    # Send schema event
                    yield f"event: schema\ndata: {json.dumps({'proposedSchema': proposed_schema})}\n\n"
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse proposed schema: {e}")

            yield f"event: done\ndata: {json.dumps({})}\n\n"

        except Exception as exc:
            logger.error(f"Chat error: {exc}")
            logger.error(traceback.format_exc())
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(event_generator()), mimetype="text/event-stream"
    )
