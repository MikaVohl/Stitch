import { useMutation } from '@tanstack/react-query'
import { useEffect, useRef, useState, useCallback } from 'react'
import { startTraining, subscribeToTrainingEvents } from '@/api/training'
import type { TrainingRequest, MetricData, TrainingState } from '@/api/types'
import { toast } from 'sonner'

export function useStartTraining() {
  return useMutation({
    mutationFn: startTraining,
    onError: (error: Error) => {
      toast.error('Failed to start training', {
        description: error.message,
      })
    },
  })
}

interface UseTrainingMetricsReturn {
  metrics: MetricData[]
  currentState: TrainingState['state'] | null
  isTraining: boolean
  runId: string | undefined
  testAccuracy: number | undefined
  startTraining: (request: TrainingRequest) => void
  resetMetrics: () => void
}

export function useTrainingMetrics(): UseTrainingMetricsReturn {
  const [metrics, setMetrics] = useState<MetricData[]>([])
  const [currentState, setCurrentState] = useState<TrainingState['state'] | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [runId, setRunId] = useState<string | undefined>(undefined)
  const [testAccuracy, setTestAccuracy] = useState<number | undefined>(undefined)
  const eventSourceCleanupRef = useRef<(() => void) | null>(null)

  const startTrainingMutation = useStartTraining()

  const resetMetrics = useCallback(() => {
    setMetrics([])
    setCurrentState(null)
    setTestAccuracy(undefined)
  }, [])

  const handleStartTraining = useCallback(
    (request: TrainingRequest) => {
      if (isTraining) {
        toast.info('Training already in progress')
        return
      }

      setIsTraining(true)
      resetMetrics()

      startTrainingMutation.mutate(request, {
        onSuccess: (data) => {
          setRunId(data.run_id)
          console.log('✅ Training job created:', data)
          toast.success('Training started!', {
            description: `Run ID: ${data.run_id}`,
          })

          // Subscribe to SSE events
          console.log('🔌 Connecting to event stream:', data.events_url)
          const cleanup = subscribeToTrainingEvents(data.events_url, {
            onMetric: (metricData) => {
              console.log('📈 Metric:', metricData)
              setMetrics((prev) => [...prev, metricData])
            },
            onState: (stateData) => {
              console.log('🔄 State:', stateData)
              setCurrentState(stateData.state)

              if (stateData.state === 'succeeded') {
                setTestAccuracy(stateData.test_accuracy)
                toast.success('Training completed!', {
                  description: stateData.test_accuracy
                    ? `Test accuracy: ${(stateData.test_accuracy * 100).toFixed(2)}%`
                    : undefined,
                })
                setIsTraining(false)
              } else if (stateData.state === 'failed') {
                toast.error('Training failed', {
                  description: stateData.error || 'Unknown error',
                })
                setIsTraining(false)
              }
            },
            onError: (error) => {
              console.error('❌ EventSource error:', error)
              toast.error('Connection error', {
                description: error.message,
              })
              setIsTraining(false)
            },
          })

          eventSourceCleanupRef.current = cleanup
        },
        onError: () => {
          setIsTraining(false)
        },
      })
    },
    [isTraining, resetMetrics, startTrainingMutation]
  )

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceCleanupRef.current) {
        eventSourceCleanupRef.current()
      }
    }
  }, [])

  return {
    metrics,
    currentState,
    isTraining,
    runId,
    testAccuracy,
    startTraining: handleStartTraining,
    resetMetrics,
  }
}
