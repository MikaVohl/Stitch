# Communication Architecture

## Overview

This repository uses **Server-Sent Events (SSE)** for real-time, unidirectional communication from the server to the client. **WebSockets are NOT used** in this application.

## Server-Sent Events (SSE) Implementation

### What are Server-Sent Events?

Server-Sent Events is a standard for pushing real-time updates from the server to the client over HTTP. Unlike WebSockets, SSE is:
- **Unidirectional**: Server → Client only
- **HTTP-based**: Uses standard HTTP connections
- **Simpler**: Built on top of standard HTTP, no special protocol needed
- **Auto-reconnection**: Browsers automatically reconnect on connection loss

### Use Cases in Stitch

Stitch uses SSE for two primary purposes:

#### 1. Training Progress Updates

**Endpoint**: `/api/runs/<run_id>/events`

The backend streams real-time training metrics and state updates as the neural network trains.

**Backend Implementation** (`backend/api.py`):
```python
@app.route("/api/runs/<run_id>/events", methods=["GET"])
def stream_run_events(run_id):
    def event_generator():
        # ... event generation logic
        yield _format_sse("metric", {"run_id": run_id, **metric})
        yield _format_sse("state", {"state": state, ...})
    
    return Response(
        stream_with_context(event_generator()), 
        mimetype="text/event-stream"
    )
```

**Frontend Implementation** (`frontend/src/api/training.ts`):
```typescript
export function subscribeToTrainingEvents(
  eventsUrl: string,
  callbacks: TrainingEventCallbacks
): () => void {
  const eventSource = new EventSource(eventsUrl)
  
  eventSource.addEventListener('metric', (e) => {
    const data = JSON.parse(e.data) as MetricData
    callbacks.onMetric(data)
  })
  
  eventSource.addEventListener('state', (e) => {
    const data = JSON.parse(e.data) as TrainingState
    callbacks.onState(data)
  })
  
  return () => eventSource.close()
}
```

**Events Streamed**:
- `metric`: Training metrics after each epoch (loss, accuracy, learning rate, etc.)
- `state`: Training state changes (queued → running → succeeded/failed/cancelled)

#### 2. AI Chat Streaming

**Endpoint**: `/api/chat`

The backend streams AI assistant responses token-by-token for a better user experience.

**Backend Implementation** (`backend/controllers/chat_controller.py`):
```python
@chat_bp.route("/api/chat", methods=["POST"])
def chat_with_assistant():
    def event_generator():
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                yield f"event: token\ndata: {json.dumps({'content': token})}\n\n"
        
        yield f"event: done\ndata: {json.dumps({})}\n\n"
    
    return Response(
        stream_with_context(event_generator()), 
        mimetype="text/event-stream"
    )
```

**Frontend Implementation** (`frontend/src/hooks/useChat.ts`):
```typescript
const response = await fetch('http://127.0.0.1:8080/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message, currentSchema })
})

const reader = response.body?.getReader()
const decoder = new TextDecoder()

while (true) {
  const { done, value } = await reader.read()
  if (done) break
  
  // Parse SSE format and handle events
  // ...
}
```

**Events Streamed**:
- `token`: Individual tokens of the AI response as they're generated
- `schema`: Proposed architecture schema (if AI suggests changes)
- `done`: Completion signal
- `error`: Error information

### SSE Message Format

All SSE messages follow the standard format:
```
event: <event_name>
data: <JSON_payload>

```

The backend helper function formats messages:
```python
def _format_sse(event_name, data):
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"
```

## Why SSE Instead of WebSockets?

### Advantages for This Application

1. **Simpler Implementation**: SSE uses standard HTTP, requiring less infrastructure
2. **Unidirectional Communication**: All real-time updates flow server → client, which matches Stitch's needs perfectly
3. **Built-in Reconnection**: Browsers handle reconnection automatically
4. **Standard HTTP**: Works through firewalls and proxies more reliably
5. **Lower Overhead**: No need for WebSocket handshake and protocol upgrade

### When WebSockets Would Be Better

WebSockets would be beneficial if Stitch needed:
- Bidirectional real-time communication
- Client → Server real-time updates
- Low-latency, high-frequency bidirectional messaging
- Binary data streaming in both directions

## Technical Stack

### Backend
- **Framework**: Flask with `flask_cors`
- **SSE Support**: `flask.stream_with_context()` for streaming responses
- **Content-Type**: `text/event-stream` for SSE endpoints

### Frontend
- **SSE Client**: Native browser `EventSource` API for training events
- **Fetch API**: `fetch()` with `ReadableStream` for chat streaming (more flexible parsing)

## Connection Management

### Training Events
- Connection opened when training starts
- Automatically closed when training completes (succeeded/failed/cancelled)
- Cleanup function returned for manual closure if component unmounts

### Chat Streaming
- Connection per chat message
- Closed when response is complete
- AbortController used for request cancellation

## Code References

### Backend
- Main API: `backend/api.py`
  - `/api/runs/<run_id>/events` - Training event stream
- Chat Controller: `backend/controllers/chat_controller.py`
  - `/api/chat` - AI chat stream

### Frontend
- Training: `frontend/src/api/training.ts`
  - `subscribeToTrainingEvents()` - EventSource implementation
- Chat: `frontend/src/hooks/useChat.ts`
  - `sendMessage()` - Fetch + ReadableStream implementation
- Usage: `frontend/src/hooks/useTraining.ts`
  - Integrates SSE with React Query

## Summary

**Answer to the Question: Does this repo use server-side events or WebSocket?**

✅ **Server-Side Events (SSE)**: YES - Used extensively for:
  - Real-time training progress updates
  - Streaming AI chat responses

❌ **WebSockets**: NO - Not used anywhere in the codebase

The choice of SSE is well-suited for Stitch's architecture, which only requires unidirectional server-to-client updates for training metrics and chat responses.
