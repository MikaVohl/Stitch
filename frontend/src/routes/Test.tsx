import { DrawingGrid } from '@/components/DrawingGrid'
import { NetworkVisualization } from '@/components/NetworkVisualization'
import { useModel, useModels } from '@/hooks/useModels'
import { useEffect, useMemo, useState } from 'react'

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

  /**
   * Converts a 2D drawing grid (values 0–255) into a flattened
   * 1D array of floats [0, 1], suitable for tensor_from_pixels().
   */
  const flattenDrawing = (grid?: number[][]): number[] | null => {
    if (!grid || grid.length === 0 || grid[0]?.length === 0) return null

    const flattened: number[] = []

    for (const row of grid) {
      for (const value of row) {
        // Ensure numeric and clamp to valid range
        const numeric = Number.isFinite(value) ? value : 0
        const normalized = Math.min(1, Math.max(0, numeric / 255))
        flattened.push(normalized)
      }
    }

    // Ensure correct size (28x28 = 784)
    if (flattened.length !== 784) {
      console.warn(`Expected 784 pixels, got ${flattened.length}`)
      return null
    }

    return flattened
  }

  const flattenedPixels = useMemo(() => flattenDrawing(currentDrawing), [currentDrawing])

  const handleInference = async () => {
    const flattened = flattenedPixels
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
    <main className="min-h-[calc(100vh-4rem)] bg-slate-50 py-12">
      <div className="mx-auto flex max-w-6xl flex-col gap-10 px-4 sm:px-6 lg:px-8">
        <header className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Playground</p>
          <h1 className="text-3xl font-semibold text-slate-900">
            Test your trained digits model
          </h1>
          <p className="max-w-2xl text-sm text-slate-600">
            Sketch a digit, pick a saved run, and preview how the network interprets the pixels in real time.
            The flattened pixel array mirrors the MNIST preprocessing used during training.
          </p>
        </header>

        <div className="grid gap-8 lg:grid-cols-[380px,1fr]">
          <section className="space-y-6">
            <div className="overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
              <div className="border-b border-slate-200 bg-slate-50 px-6 py-4">
                <h2 className="text-sm font-semibold text-slate-900">Draw &amp; Configure</h2>
                <p className="text-xs text-slate-500">
                  Select a trained model, choose a successful run, then draw in the grid.
                </p>
              </div>
              <div className="space-y-6 px-6 pb-6 pt-5">
                <div className="flex flex-col gap-4">
                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Model
                    <select
                      value={selectedModelId}
                      onChange={(e) => setSelectedModelId(e.target.value)}
                      className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-inner focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                    >
                      <option value="">Select a model</option>
                      {models?.map((model) => (
                        <option key={model.model_id} value={model.model_id}>
                          {model.name}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Successful run
                    <select
                      value={selectedRunId || latestRunId}
                      onChange={(e) => setSelectedRunId(e.target.value)}
                      disabled={succeededRuns.length === 0}
                      className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-inner focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200 disabled:cursor-not-allowed disabled:bg-slate-100"
                    >
                      {succeededRuns.length === 0 ? (
                        <option value="">No successful runs available</option>
                      ) : (
                        succeededRuns.map((run) => (
                          <option key={run.run_id} value={run.run_id}>
                            {run.run_id} —{' '}
                            {run.completed_at ? new Date(run.completed_at).toLocaleString() : 'Completed'}
                          </option>
                        ))
                      )}
                    </select>
                  </label>
                </div>

                <div className="rounded-2xl bg-slate-900 p-4 text-slate-50 shadow-inner">
                  <p className="text-xs uppercase tracking-wider text-slate-400">Drawing Grid</p>
                  <p className="mt-1 text-sm text-slate-100">
                    Use your cursor to sketch a digit. Click <span className="font-semibold">Clear</span> to reset.
                  </p>
                </div>

                <div className="flex flex-col items-center justify-center">
                  <DrawingGrid
                    onDrawingComplete={(pixels) => {
                      setCurrentDrawing(pixels)
                      setPrediction(null)
                    }}
                  />
                </div>

                <button
                  onClick={handleInference}
                  disabled={!selectedModelId || (!selectedRunId && !latestRunId) || !flattenedPixels || isRunning}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-all hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  {isRunning ? (
                    <>
                      <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Running inference…
                    </>
                  ) : (
                    <>
                      <span>Run inference</span>
                      <span aria-hidden className="text-xs text-blue-200">⌘⏎</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className="overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
              <div className="border-b border-slate-200 bg-slate-50 px-6 py-4">
                <h3 className="text-sm font-semibold text-slate-900">Input preview</h3>
                <p className="text-xs text-slate-500">
                  Flattened pixel values ready for <code className="font-mono text-orange-600">tensor_from_pixels()</code>.
                </p>
              </div>
              <div className="space-y-4 px-6 pb-6 pt-5 text-xs text-slate-700">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-900">Length</span>
                  <span className="rounded-full bg-slate-100 px-3 py-1 font-mono text-xs text-slate-700">
                    {flattenedPixels?.length ?? 0}
                  </span>
                </div>
                <div className="max-h-32 overflow-auto rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-[11px] leading-relaxed text-slate-700">
                  {flattenedPixels
                    ? `[${flattenedPixels
                        .slice(0, 120)
                        .map((value) => value.toFixed(3))
                        .join(', ')}${flattenedPixels.length > 120 ? ', …' : ''}]`
                    : 'Draw on the grid to preview the flattened array.'}
                </div>
                {prediction !== null && (
                  <div className="rounded-xl border border-blue-100 bg-blue-50 px-4 py-3 text-center text-sm text-blue-800">
                    Predicted digit:&nbsp;
                    <span className="font-semibold text-blue-900">{prediction}</span>
                  </div>
                )}
              </div>
            </div>
          </section>

          <section className="flex h-full flex-col">
            <div className="flex-1 overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
              <div className="border-b border-slate-200 bg-slate-50 px-6 py-4">
                <h2 className="text-sm font-semibold text-slate-900">Network visualization</h2>
                <p className="text-xs text-slate-500">
                  Input pixels, sampled hidden layers, and all output neurons rendered with React Flow.
                </p>
              </div>
              <div className="p-4">
                <NetworkVisualization
                  layers={selectedModel?.architecture?.layers ?? []}
                  currentDrawing={currentDrawing}
                />
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  )
}
