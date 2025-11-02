import type { FC } from 'react'
import { useState } from 'react'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from './ui/sheet'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { MetricData, MnistSample } from '@/api/types'
import { Button } from './ui/button'
import { Input } from './ui/input'

interface TrainingMetricsSlideOverProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  isTraining: boolean
  metrics: MetricData[]
  currentState: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | null
  runId?: string
  samplePredictions?: MnistSample[]
  onCancel?: () => void
  canCancel?: boolean
  isCancelling?: boolean
}

function formatTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  return `${mins}m ${secs}s`
}

export const TrainingMetricsSlideOver: FC<TrainingMetricsSlideOverProps> = ({
  open,
  onOpenChange,
  isTraining,
  metrics,
  currentState,
  runId,
  samplePredictions = [],
  onCancel,
  canCancel = false,
  isCancelling = false,
}) => {
  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null
  const [modelName, setModelName] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [savedModelId, setSavedModelId] = useState<string | null>(null)
  const hasSamplePredictions = samplePredictions.length > 0

  const renderSampleGrid = (grid: number[][]) => (
    <div
      className="grid"
      style={{
        gridTemplateColumns: 'repeat(28, 1fr)',
        gridAutoRows: '1fr',
        width: '112px',
        height: '112px',
        gap: '0',
      }}
    >
      {grid.map((row, rowIndex) =>
        row.map((value, columnIndex) => {
          const shade = Math.max(0, Math.min(255, value))
          return (
            <div
              key={`${rowIndex}-${columnIndex}`}
              style={{
                backgroundColor: `rgb(${shade}, ${shade}, ${shade})`,
                width: '100%',
                height: '100%',
              }}
            />
          )
        })
      )}
    </div>
  )

  const handleSaveModel = async () => {
    if (!runId || !modelName.trim()) return

    setIsSaving(true)
    try {
      const response = await fetch('/api/models/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          run_id: runId,
          name: modelName.trim(),
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setSavedModelId(data.model_id)
        setModelName('')
      } else {
        const error = await response.json()
        alert(`Failed to save model: ${error.error}`)
      }
    } catch (error) {
      alert('Failed to save model')
    } finally {
      setIsSaving(false)
    }
  }

  const showCancelButton = Boolean(onCancel) && canCancel

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-3xl overflow-y-auto px-8 pb-8">
        <SheetHeader>
          <SheetTitle>Training Metrics</SheetTitle>
          <SheetDescription>
            {runId && <span className="text-xs font-mono">Run: {runId}</span>}
          </SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* Save Model Section - Show when training succeeded */}
          {currentState === 'succeeded' && !savedModelId && (
            <div className="space-y-3 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="text-sm font-semibold text-green-900">Save Model</h3>
              <p className="text-xs text-green-700">
                Training completed successfully! Save this model to use it later.
              </p>
              <div className="flex gap-2">
                <Input
                  placeholder="Model name"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  className="flex-1"
                />
                <Button
                  onClick={handleSaveModel}
                  disabled={!modelName.trim() || isSaving}
                  className="bg-green-600 hover:bg-green-700"
                >
                  {isSaving ? 'Saving...' : 'Save'}
                </Button>
              </div>
            </div>
          )}

          {savedModelId && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="text-sm font-semibold text-green-900">Model Saved!</h3>
              <p className="text-xs text-green-700 mt-1">
                Model ID: <span className="font-mono">{savedModelId}</span>
              </p>
              <a
                href={`/models/${savedModelId}`}
                className="text-xs text-green-600 hover:underline mt-2 inline-block"
              >
                View model →
              </a>
            </div>
          )}
          {/* Status */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-700">Status</h3>
            <div className="flex items-center gap-2">
              {isTraining && (
                <svg
                  className="w-4 h-4 animate-spin text-blue-500"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              )}
              <span
                className={`text-sm font-medium ${currentState === 'running'
                  ? 'text-blue-600'
                  : currentState === 'succeeded'
                    ? 'text-green-600'
                    : currentState === 'failed'
                      ? 'text-red-600'
                      : currentState === 'cancelled'
                        ? 'text-gray-600'
                        : 'text-gray-600'
                  }`}
              >
                {currentState === 'running'
                  ? 'Training...'
                  : currentState === 'succeeded'
                    ? 'Completed'
                    : currentState === 'failed'
                      ? 'Failed'
                      : currentState === 'cancelled'
                        ? 'Cancelled'
                        : currentState === 'queued'
                          ? 'Queued'
                          : 'Idle'}
              </span>
            </div>
            {showCancelButton && (
              <Button
                variant="outline"
                size="sm"
                onClick={onCancel}
                disabled={isCancelling}
              >
                {isCancelling ? 'Cancelling...' : 'Cancel training'}
              </Button>
            )}
          </div>

          {hasSamplePredictions && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-gray-700">Sample Predictions</h3>
              <p className="text-xs text-gray-500">
                Eight validation samples, their true labels, and the model&apos;s predictions.
              </p>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {samplePredictions.slice(0, 8).map((sample, index) => (
                  <div
                    key={`${sample.label}-${sample.prediction}-${index}`}
                    className="flex flex-col items-center gap-2 rounded-lg border border-gray-200 bg-white p-3 shadow-sm"
                  >
                    {renderSampleGrid(sample.grid)}
                    <div className="text-xs text-gray-700 text-center space-y-1">
                      <div>
                        <span className="font-semibold text-gray-900">True:</span> {sample.label}
                      </div>
                      <div>
                        <span className="font-semibold text-gray-900">Pred:</span> {sample.prediction}
                      </div>
                      {typeof sample.confidence === 'number' && (
                        <div>
                          <span className="font-semibold text-gray-900">Confidence:</span>{' '}
                          {(sample.confidence * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Progress */}
          {latestMetric && (
            <>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold text-gray-700">Progress</span>
                  <span className="text-gray-600">
                    Epoch {latestMetric.epoch}
                    {latestMetric.progress !== undefined &&
                      ` (${(latestMetric.progress * 100).toFixed(0)}%)`}
                  </span>
                </div>
                {latestMetric.progress !== undefined && (
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${latestMetric.progress * 100}%` }}
                    />
                  </div>
                )}
              </div>

              {/* ETA */}
              {latestMetric.eta_seconds !== undefined && latestMetric.eta_seconds > 0 && (
                <div className="space-y-1">
                  <span className="text-sm font-semibold text-gray-700">
                    Estimated Time Remaining
                  </span>
                  <p className="text-sm text-gray-600">
                    {formatTime(latestMetric.eta_seconds)}
                  </p>
                </div>
              )}

              {/* Charts */}
              {metrics.length > 1 && (
                <>
                  <div className="space-y-2">
                    <h3 className="text-sm font-semibold text-gray-700">Loss Over Time</h3>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={metrics}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="epoch"
                            label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                          <Tooltip />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="train_loss"
                            stroke="#dc2626"
                            name="Train Loss"
                            strokeWidth={2}
                            dot={false}
                          />
                          <Line
                            type="monotone"
                            dataKey="val_loss"
                            stroke="#2563eb"
                            name="Val Loss"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h3 className="text-sm font-semibold text-gray-700">Accuracy Over Time</h3>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={metrics}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="epoch"
                            label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis
                            label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }}
                            domain={[0, 1]}
                            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                          />
                          <Tooltip
                            formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                          />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="train_accuracy"
                            stroke="#16a34a"
                            name="Train Acc"
                            strokeWidth={2}
                            dot={false}
                          />
                          <Line
                            type="monotone"
                            dataKey="val_accuracy"
                            stroke="#9333ea"
                            name="Val Acc"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </>
              )}

              {/* Latest Metrics */}
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-gray-700">Latest Metrics</h3>
                <div className="grid grid-cols-4 gap-3">
                  <div className="bg-red-50 p-3 rounded-lg">
                    <p className="text-xs text-gray-600">Train Loss</p>
                    <p className="text-lg font-semibold text-red-700">
                      {latestMetric.train_loss.toFixed(4)}
                    </p>
                  </div>
                  <div className="bg-blue-50 p-3 rounded-lg">
                    <p className="text-xs text-gray-600">Val Loss</p>
                    <p className="text-lg font-semibold text-blue-700">
                      {latestMetric.val_loss.toFixed(4)}
                    </p>
                  </div>
                  <div className="bg-green-50 p-3 rounded-lg">
                    <p className="text-xs text-gray-600">Train Acc</p>
                    <p className="text-lg font-semibold text-green-700">
                      {(latestMetric.train_accuracy * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="bg-purple-50 p-3 rounded-lg">
                    <p className="text-xs text-gray-600">Val Acc</p>
                    <p className="text-lg font-semibold text-purple-700">
                      {(latestMetric.val_accuracy * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
              </div>

              {/* Additional Info */}
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-gray-700">Training Info</h3>
                <div className="space-y-1 text-sm">
                  {latestMetric.learning_rate !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Learning Rate:</span>
                      <span className="font-mono text-gray-900">
                        {latestMetric.learning_rate.toFixed(6)}
                      </span>
                    </div>
                  )}
                  {latestMetric.epoch_time !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Epoch Time:</span>
                      <span className="font-mono text-gray-900">
                        {latestMetric.epoch_time.toFixed(2)}s
                      </span>
                    </div>
                  )}
                  {latestMetric.samples_per_sec !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Samples/sec:</span>
                      <span className="font-mono text-gray-900">
                        {latestMetric.samples_per_sec.toFixed(1)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}

          {/* Metrics History */}
          {metrics.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-gray-700">History</h3>
              <div className="max-h-64 overflow-y-auto border rounded-lg">
                <table className="w-full text-xs">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="px-2 py-1 text-left">Epoch</th>
                      <th className="px-2 py-1 text-right">Train Loss</th>
                      <th className="px-2 py-1 text-right">Val Loss</th>
                      <th className="px-2 py-1 text-right">Val Acc</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics
                      .slice()
                      .reverse()
                      .map((metric, idx) => (
                        <tr
                          key={idx}
                          className={idx === 0 ? 'bg-blue-50 font-semibold' : ''}
                        >
                          <td className="px-2 py-1">{metric.epoch}</td>
                          <td className="px-2 py-1 text-right font-mono">
                            {metric.train_loss.toFixed(4)}
                          </td>
                          <td className="px-2 py-1 text-right font-mono">
                            {metric.val_loss.toFixed(4)}
                          </td>
                          <td className="px-2 py-1 text-right font-mono">
                            {(metric.val_accuracy * 100).toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Loading State */}
          {isTraining && metrics.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 space-y-4">
              <svg
                className="w-12 h-12 animate-spin text-blue-500"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <p className="text-sm text-gray-600">Initializing training...</p>
            </div>
          )}

          {/* Empty State */}
          {!isTraining && metrics.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <p className="text-sm text-gray-600">
                No training metrics yet. Start a training run to see live updates here.
              </p>
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}
