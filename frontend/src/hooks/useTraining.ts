import { useMutation } from '@tanstack/react-query'
import { useEffect, useRef, useState, useCallback } from 'react'
import { startTraining, subscribeToTrainingEvents, cancelTraining } from '@/api/training'
import type { TrainingRequest, MetricData, TrainingState, MnistSample } from '@/api/types'
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
  samplePredictions: MnistSample[]
  startTraining: (request: TrainingRequest) => void
  resetMetrics: () => void
  cancelTraining: () => Promise<void>
  isCancelling: boolean
}

export function useTrainingMetrics(): UseTrainingMetricsReturn {
  const [metrics, setMetrics] = useState<MetricData[]>([])
  const [currentState, setCurrentState] = useState<TrainingState['state'] | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [runId, setRunId] = useState<string | undefined>(undefined)
  const [testAccuracy, setTestAccuracy] = useState<number | undefined>(undefined)
  const [samplePredictions, setSamplePredictions] = useState<MnistSample[]>([])
  const [isCancelling, setIsCancelling] = useState(false)
  const eventSourceCleanupRef = useRef<(() => void) | null>(null)

  const startTrainingMutation = useStartTraining()

  const resetMetrics = useCallback(() => {
    setMetrics([])
    setCurrentState(null)
    setTestAccuracy(undefined)
    setSamplePredictions([])
  }, [])

  const handleStartTraining = useCallback(
    (request: TrainingRequest) => {
      if (isTraining) {
        toast.info('Training already in progress')
        return
      }

      setIsTraining(true)
      resetMetrics()
      setSamplePredictions([])

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
                setSamplePredictions(stateData.sample_predictions ?? [])
                toast.success('Training completed!', {
                  description: stateData.test_accuracy
                    ? `Test accuracy: ${(stateData.test_accuracy * 100).toFixed(2)}%`
                    : undefined,
                })
                setIsTraining(false)
                if (eventSourceCleanupRef.current) {
                  eventSourceCleanupRef.current()
                  eventSourceCleanupRef.current = null
                }
              } else if (stateData.state === 'failed') {
                toast.error('Training failed', {
                  description: stateData.error || 'Unknown error',
                })
                setSamplePredictions([])
                setIsTraining(false)
                if (eventSourceCleanupRef.current) {
                  eventSourceCleanupRef.current()
                  eventSourceCleanupRef.current = null
                }
              } else if (stateData.state === 'cancelled') {
                toast.info('Training cancelled', {
                  description: 'Your training run has been stopped.',
                })
                setSamplePredictions([])
                setIsTraining(false)
                if (eventSourceCleanupRef.current) {
                  eventSourceCleanupRef.current()
                  eventSourceCleanupRef.current = null
                }
              }
            },
            onError: (error) => {
              console.error('❌ EventSource error:', error)
              toast.error('Connection error', {
                description: error.message,
              })
              setIsTraining(false)
              if (eventSourceCleanupRef.current) {
                eventSourceCleanupRef.current()
                eventSourceCleanupRef.current = null
              }
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

  const cancelActiveTraining = useCallback(async () => {
    if (!runId) {
      toast.info('No active training run to cancel')
      return
    }

    if (!isTraining && currentState !== 'running' && currentState !== 'queued') {
      toast.info('No active training run to cancel')
      return
    }

    if (isCancelling) {
      return
    }

    setIsCancelling(true)
    try {
      await cancelTraining(runId)
      toast.info('Cancelling training...', {
        description: 'Please wait while the current epoch finishes.',
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to cancel training'
      toast.error('Failed to cancel training', {
        description: message,
      })
    } finally {
      setIsCancelling(false)
    }
  }, [runId, isTraining, currentState, isCancelling])

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
    samplePredictions,
    startTraining: handleStartTraining,
    resetMetrics,
    cancelTraining: cancelActiveTraining,
    isCancelling,
  }
}
