import { useQuery } from "@tanstack/react-query";
import { useMutation } from "@tanstack/react-query";
import axios from "axios";
import type { MetricData } from "@/api/types";

export type StoredLayer = {
  type: string
  in?: number
  out?: number
  [key: string]: unknown
}

export type TrainingRun = {
  run_id: string
  model_id: string
  state: 'queued' | 'running' | 'succeeded' | 'failed'
  epochs_total: number
  epoch?: number
  metrics: MetricData[]
  test_accuracy?: number
  created_at: string
  completed_at?: string
  error?: string
  saved_model_path?: string
  hyperparams?: Record<string, unknown>
}

export type StoredModel = {
  name: string
  model_id: string
  created_at?: string
  architecture?: {
    input_size?: number
    layers?: StoredLayer[]
  }
  hyperparams?: Record<string, unknown>
  runs?: TrainingRun[]
  runs_total?: number
  trained?: boolean
  last_trained_at?: string
}

export function SaveModel() {
  return useMutation<unknown, Error, { name: string }>({
    mutationFn: (name) => {
      return axios.post('/api/models', name)
    },
  })
}

export function useModel(id?: string) {
  return useQuery({
    queryKey: ['models', id],
    queryFn: async (): Promise<StoredModel> => {

      const response = await fetch(`/api/models/${id}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      return response.json()
    },
    enabled: !!id
  })
}


export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: async (): Promise<StoredModel[]> => {

      const response = await fetch('/api/models', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      return response.json()
    }
  })
}

