import { useQuery } from "@tanstack/react-query";

export type StoredModel = {
  model_id: string
  created_at?: string
  architecture?: {
    input_size?: number
    layers?: StoredLayer[]
  }
  hyperparams?: Record<string, unknown>
}


export type StoredLayer = {
  type: string
  in?: number
  out?: number
  [key: string]: unknown
}


export default function useModels() {
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
