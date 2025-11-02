import { DrawingGrid } from '@/components/DrawingGrid'
import { useModel, useModels } from '@/hooks/useModels'
import { useEffect, useMemo, useState } from 'react'
import { NetworkVisualization } from '@/components/NetworkVisualization'

export default function Test() {
  const { data: models, isLoading, isError, error } = useModels()
  const [selectedModelId, setSelectedModelId] = useState<string>('')
  const [currentDrawing, setCurrentDrawing] = useState<number[][]>()
  const [isRunning, setIsRunning] = useState(false)
  const [prediction, setPrediction] = useState<number | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string>('')

  const selectedModel = models?.find(m => m.model_id === selectedModelId)
  const { data: selectedModelDetail } = useModel(selectedModelId)
  const succeededRuns = useMemo(() => {
    if (!selectedModelDetail?.runs) return []
    return [...selectedModelDetail.runs]
      .filter((run) => run.state === 'succeeded')
      .sort((a, b) => {
        const aTime = new Date(a.completed_at ?? a.created_at ?? 0).getTime()
        const bTime = new Date(b.completed_at ?? b.created_at ?? 0).getTime()
        return bTime - aTime
      })
  }, [selectedModelDetail])

  const latestRunId = succeededRuns.length > 0 ? succeededRuns[0].run_id : ''

  useEffect(() => {
    setSelectedRunId('')
  }, [selectedModelId])

  const flattenDrawing = (grid?: number[][]): number[] | null => {
    if (!grid || grid.length === 0 || grid[0]?.length === 0) return null
    const flattened: number[] = []
    for (const row of grid) {
      for (const value of row) {
        flattened.push(Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0)))
      }
    }
    return flattened
  }

  const handleInference = async () => {
    const flattened = flattenDrawing(currentDrawing)
    const runId = selectedRunId || latestRunId
    if (!flattened || !runId) return

    setIsRunning(true)
    try {
      const response = await fetch('/api/infer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          run_id: runId,
          pixels: flattened,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to run inference')
      }

      const data = await response.json()
      const detected = typeof data?.label === 'number'
        ? data.label
        : typeof data?.prediction === 'number'
        ? data.prediction
        : null
      setPrediction(detected)
    } catch (err) {
      console.error('Inference failed:', err)
    } finally {
      setIsRunning(false)
    }
  }

  if (isLoading) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-8 text-gray-600 shadow-sm">
        Loading your models...
      </div>
    )
  }

  if (isError) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-8 text-red-700 shadow-sm">
        {error.message}
      </div>
    )
  }

  return (
    <main className="min-h-[calc(100vh-4rem)] bg-gray-50 py-10">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold text-gray-900">Test Your Models</h1>
          <p className="text-sm text-gray-600">
            Draw digits to test your trained models. The drawing will be converted to a 28x28 pixel image,
            similar to the MNIST dataset format.
          </p>
        </header>

        <div className="grid gap-6 md:grid-cols-[1fr,auto]">
          <section className="rounded-xl border border-gray-200 bg-white p-8">
            <h2 className="text-xl font-semibold text-gray-900">Drawing Canvas</h2>
            <p className="mt-2 text-sm text-gray-600">
              Draw a digit (0-9) in the grid below and select a model to test.
            </p>
            <div className="mt-6 grid grid-cols-2 items-start gap-6">
              <div className="space-y-6">
                <DrawingGrid
                  onDrawingComplete={(pixels) => {
                    setCurrentDrawing(pixels)
                    setPrediction(null)
                  }}
                />
              </div>
              
              <div className="flex w-full max-w-md flex-col gap-4">
                <select
                  value={selectedModelId}
                  onChange={(e) => setSelectedModelId(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                  <option value="">Select a model</option>
                  {models?.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.name}
                    </option>
                  ))}
                </select>

                <select
                  value={selectedRunId || latestRunId}
                  onChange={(e) => setSelectedRunId(e.target.value)}
                  disabled={succeededRuns.length === 0}
                  className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100"
                >
                  {succeededRuns.length === 0 ? (
                    <option value="">No successful runs available</option>
                  ) : (
                    succeededRuns.map((run) => (
                      <option key={run.run_id} value={run.run_id}>
                        {run.run_id} — {run.completed_at ? new Date(run.completed_at).toLocaleString() : 'Completed'}
                      </option>
                    ))
                  )}
                </select>

                <button
                  onClick={handleInference}
                  disabled={!selectedModelId || (!selectedRunId && !latestRunId) || !currentDrawing || isRunning}
                  className="flex items-center justify-center gap-2 cursor-pointer rounded-lg bg-blue-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-600 disabled:bg-gray-300"
                >
                  {isRunning ? (
                    <>
                      <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Running...
                    </>
                  ) : (
                    'Run Inference'
                  )}
                </button>
                {prediction !== null && (
                  <div className="rounded-md border border-blue-100 bg-blue-50 px-3 py-2 text-center text-sm text-blue-800">
                    Predicted digit: <span className="font-semibold">{prediction}</span>
                  </div>
                )}
              </div>
            </div>
          </section>

        </div>
                        
                {selectedModel && (
                  <div className="col-span-2 rounded-lg border border-gray-200 bg-gray-50 p-4">
                    <h3 className="mb-3 text-sm font-medium text-gray-900">Network Architecture</h3>
                    <NetworkVisualization 
                      layers={selectedModel.architecture?.layers ?? []} 
                      currentDrawing={currentDrawing}
                    />
                  </div>
                )}
      </div>
    </main>
  )
}
