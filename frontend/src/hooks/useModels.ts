import { useQuery } from "@tanstack/react-query";
import { useMutation } from "@tanstack/react-query";
import axios from "axios";

export type StoredLayer = {
  type: string
  in?: number
  out?: number
  [key: string]: unknown
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
}

export function SaveModel() {
  return useMutation<unknown, Error, { name: string }>({
    mutationFn: (name) => {
      return axios.post('/api/models', name)
    },
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

