import type { TrainingRequest, TrainingResponse, MetricData, TrainingState } from './types'

export class TrainingError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TrainingError'
  }
}

export async function startTraining(request: TrainingRequest): Promise<TrainingResponse> {
  const response = await fetch('/api/train', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Training request failed' }))
    throw new TrainingError(error.error || 'Training request failed')
  }

  return response.json()
}

export interface TrainingEventCallbacks {
  onMetric: (data: MetricData) => void
  onState: (data: TrainingState) => void
  onError: (error: Error) => void
}

export function subscribeToTrainingEvents(
  eventsUrl: string,
  callbacks: TrainingEventCallbacks
): () => void {
  const eventSource = new EventSource(eventsUrl)

  eventSource.addEventListener('metric', (e) => {
    try {
      const data = JSON.parse(e.data) as MetricData
      callbacks.onMetric(data)
    } catch (error) {
      callbacks.onError(new Error('Failed to parse metric data'))
    }
  })

  eventSource.addEventListener('state', (e) => {
    try {
      const data = JSON.parse(e.data) as TrainingState
      callbacks.onState(data)

      if (data.state === 'succeeded' || data.state === 'failed') {
        eventSource.close()
      }
    } catch (error) {
      callbacks.onError(new Error('Failed to parse state data'))
    }
  })

  eventSource.onerror = (error) => {
    callbacks.onError(new Error('Event stream connection error'))
    eventSource.close()
  }

  // Return cleanup function
  return () => {
    eventSource.close()
  }
}
