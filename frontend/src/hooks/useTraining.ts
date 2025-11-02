import { useMutation } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { startTraining, subscribeToTrainingEvents, cancelTraining } from '@/api/training'
import type { TrainingRequest, MetricData, TrainingState, MnistSample } from '@/api/types'
import { toast } from 'sonner'
import { useTrainingStateStore } from '@/store/trainingStateStore'

const EMPTY_METRICS: MetricData[] = []
const EMPTY_PREDICTIONS: MnistSample[] = []

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
  lastRunArchitecture: TrainingRequest['architecture'] | undefined
  lastRunHyperparams: TrainingRequest['hyperparams'] | undefined
  startTraining: (request: TrainingRequest) => void
  resetMetrics: () => void
  cancelTraining: () => Promise<void>
  isCancelling: boolean
}

export function useTrainingMetrics(): UseTrainingMetricsReturn {
  const currentState = useTrainingStateStore((state) => state.currentState)
  const setTrainingState = useTrainingStateStore((state) => state.setCurrentState)
  const metrics = useTrainingStateStore((state) => state.lastRun?.metrics ?? EMPTY_METRICS)
  const testAccuracy = useTrainingStateStore((state) => state.lastRun?.testAccuracy)
  const samplePredictions = useTrainingStateStore((state) => state.lastRun?.samplePredictions ?? EMPTY_PREDICTIONS)
  const initializeRun = useTrainingStateStore((state) => state.initializeRun)
  const appendMetric = useTrainingStateStore((state) => state.appendMetric)
  const setRunIdInStore = useTrainingStateStore((state) => state.setRunId)
  const setRunResult = useTrainingStateStore((state) => state.setRunResult)
  const clearRun = useTrainingStateStore((state) => state.clearRun)
  const lastRunArchitecture = useTrainingStateStore((state) => state.lastRun?.architecture)
  const lastRunHyperparams = useTrainingStateStore((state) => state.lastRun?.hyperparams)
  const lastRunRunId = useTrainingStateStore((state) => state.lastRun?.runId)

  const [isTraining, setIsTraining] = useState(false)
  const [runId, setRunId] = useState<string | undefined>(lastRunRunId)
  const [isCancelling, setIsCancelling] = useState(false)
  const eventSourceCleanupRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    if (!runId && lastRunRunId) {
      setRunId(lastRunRunId)
    }
  }, [runId, lastRunRunId])

  const startTrainingMutation = useStartTraining()

  const resetMetrics = useCallback(() => {
    setTrainingState(null)
    clearRun()
    setRunId(undefined)
  }, [clearRun, setTrainingState])

  const handleStartTraining = useCallback(
    (request: TrainingRequest) => {
      if (isTraining) {
        toast.info('Training already in progress')
        return
      }

      setIsTraining(true)
      resetMetrics()
      initializeRun({
        architecture: request.architecture,
        hyperparams: request.hyperparams,
      })

      startTrainingMutation.mutate(request, {
        onSuccess: (data) => {
          setRunId(data.run_id)
          setRunIdInStore(data.run_id)
          console.log('✅ Training job created:', data)
          toast.success('Training started!', {
            description: `Run ID: ${data.run_id}`,
          })

          console.log('🔌 Connecting to event stream:', data.events_url)
          const cleanup = subscribeToTrainingEvents(data.events_url, {
            onMetric: (metricData) => {
              console.log('📈 Metric:', metricData)
              appendMetric(metricData)
            },
            onState: (stateData) => {
              console.log('🔄 State:', stateData)
              setTrainingState(stateData.state)

              if (stateData.state === 'succeeded') {
                setRunResult({
                  testAccuracy: stateData.test_accuracy,
                  samplePredictions: stateData.sample_predictions ?? [],
                })
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
                setRunResult({ samplePredictions: [] })
                setIsTraining(false)
                if (eventSourceCleanupRef.current) {
                  eventSourceCleanupRef.current()
                  eventSourceCleanupRef.current = null
                }
              } else if (stateData.state === 'cancelled') {
                toast.info('Training cancelled', {
                  description: 'Your training run has been stopped.',
                })
                setRunResult({ samplePredictions: [] })
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
    [appendMetric, initializeRun, isTraining, resetMetrics, setRunIdInStore, setRunResult, setTrainingState, startTrainingMutation]
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

  useEffect(() => {
    if (!isTraining && !runId && metrics.length === 0) {
      setTrainingState(null)
    }
  }, [isTraining, runId, metrics.length, setTrainingState])

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
    lastRunArchitecture,
    lastRunHyperparams,
    startTraining: handleStartTraining,
    resetMetrics,
    cancelTraining: cancelActiveTraining,
    isCancelling,
  }
}
