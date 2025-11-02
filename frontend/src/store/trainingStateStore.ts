import { create } from 'zustand'
import type { TrainingState, TrainingRequest, MetricData, MnistSample } from '@/api/types'

interface LastRunState {
  runId?: string
  architecture?: TrainingRequest['architecture']
  hyperparams?: TrainingRequest['hyperparams']
  metrics: MetricData[]
  testAccuracy?: number
  samplePredictions: MnistSample[]
}

interface TrainingStateStore {
  currentState: TrainingState['state'] | null
  lastRun: LastRunState | null
  setCurrentState: (state: TrainingState['state'] | null) => void
  initializeRun: (payload: {
    architecture: TrainingRequest['architecture']
    hyperparams: TrainingRequest['hyperparams']
  }) => void
  appendMetric: (metric: MetricData) => void
  setRunId: (runId: string) => void
  setRunResult: (payload: { testAccuracy?: number; samplePredictions?: MnistSample[] }) => void
  clearRun: () => void
}

export const useTrainingStateStore = create<TrainingStateStore>((set) => ({
  currentState: null,
  lastRun: null,
  setCurrentState: (currentState) => set({ currentState }),
  initializeRun: ({ architecture, hyperparams }) =>
    set({
      lastRun: {
        architecture,
        hyperparams,
        metrics: [],
        samplePredictions: [],
      },
    }),
  appendMetric: (metric) =>
    set((state) =>
      state.lastRun
        ? {
            lastRun: {
              ...state.lastRun,
              metrics: [...state.lastRun.metrics, metric],
            },
          }
        : state
    ),
  setRunId: (runId) =>
    set((state) =>
      state.lastRun
        ? {
            lastRun: {
              ...state.lastRun,
              runId,
            },
          }
        : state
    ),
  setRunResult: ({ testAccuracy, samplePredictions }) =>
    set((state) =>
      state.lastRun
        ? {
            lastRun: {
              ...state.lastRun,
              testAccuracy: testAccuracy ?? state.lastRun.testAccuracy,
              samplePredictions: samplePredictions ?? state.lastRun.samplePredictions,
            },
          }
        : state
    ),
  clearRun: () => set({ lastRun: null }),
}))
