import { useModel, type TrainingRun } from "@/hooks/useModels";
import { useParams } from "react-router-dom"
import { summarizeArchitecture, summarizeHyperparams } from "./Models";

export default function ModelPage() {
  const { id } = useParams();
  const { data: model, isLoading } = useModel(id);

  if (isLoading || !model) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-8 text-gray-600 shadow-sm">
        Loading your model...
      </div>
    )
  }

  const succeededRuns = model.runs?.filter(run => run.state === 'succeeded') ?? [];

  return (
    <div className="max-w-6xl mx-auto space-y-8 p-12">
      {/* Model Info */}
      <div>
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{model.name}</h1>
            <p className="mt-1 text-sm text-gray-500">
              {model.created_at
                ? new Date(model.created_at).toLocaleString()
                : 'Creation time unknown'}
            </p>
          </div>
          <span className="rounded-md bg-gray-100 px-3 py-1 text-sm font-medium text-gray-700">
            {model.architecture?.layers?.length ?? 0} layers
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Architecture</h3>
            <div className="text-gray-600 font-mono text-xs bg-gray-50 p-3 rounded">
              <div className="mb-1">Input: {model.architecture?.input_size ?? '—'}</div>
              {summarizeArchitecture(model.architecture?.layers)}
            </div>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Hyperparameters</h3>
            <div className="text-gray-600 text-xs bg-gray-50 p-3 rounded">
              {summarizeHyperparams(model.hyperparams)}
            </div>
          </div>
        </div>
      </div>

      {model.runs && model.runs.length > 0 && (
        <div>
          <h2 className="text-xl font-bold text-gray-900 mb-4">
            Training Runs ({model.runs_total ?? model.runs.length})
          </h2>
          <div className="space-y-3">
            {model.runs.map(run => (
              <RunDetails key={run.run_id} run={run} />
            ))}
          </div>
        </div>
      )}

      {/* Training Metrics */}
      {succeededRuns.length > 0 && (
        <div>
          <h2 className="text-xl font-bold text-gray-900 mb-4">Training Metrics</h2>
          <MetricsVisualization runs={succeededRuns} />
        </div>
      )}
    </div>
  )
}

function RunDetails({ run }: { run: TrainingRun }) {
  const stateColors: Record<string, string> = {
    queued: 'bg-yellow-50 text-yellow-700 border-yellow-200',
    running: 'bg-blue-50 text-blue-700 border-blue-200',
    succeeded: 'bg-green-50 text-green-700 border-green-200',
    failed: 'bg-red-50 text-red-700 border-red-200',
    cancelled: 'bg-gray-50 text-gray-600 border-gray-200',
  };

  const badgeClass = stateColors[run.state] ?? 'bg-gray-50 text-gray-700 border-gray-200';

  return (
    <div className="border-l-4 pl-4 py-2">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium border ${badgeClass}`}>
              {run.state}
            </span>
            <span className="text-sm font-mono text-gray-600">{run.run_id}</span>
            <span className="text-xs text-gray-400">
              {new Date(run.created_at).toLocaleString()}
            </span>
          </div>

          {run.state === 'succeeded' && run.metrics.length > 0 && (
            <div className="mt-2 flex gap-6 text-sm">
              <div>
                <span className="text-gray-500">Epochs:</span>
                <span className="ml-1.5 font-medium text-gray-900">{run.epochs_total}</span>
              </div>
              {run.test_accuracy !== undefined && (
                <div>
                  <span className="text-gray-500">Accuracy:</span>
                  <span className="ml-1.5 font-medium text-gray-900">
                    {(run.test_accuracy * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              <div>
                <span className="text-gray-500">Final Loss:</span>
                <span className="ml-1.5 font-medium text-gray-900">
                  {run.metrics[run.metrics.length - 1].train_loss.toFixed(4)}
                </span>
              </div>
            </div>
          )}

          {run.error && (
            <div className="mt-2 text-sm text-red-600">
              {run.error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricsVisualization({ runs }: { runs: TrainingRun[] }) {
  const latestRun = runs[0];

  if (!latestRun || !latestRun.metrics || latestRun.metrics.length === 0) {
    return <p className="text-sm text-gray-500">No metrics available</p>;
  }

  const metrics = latestRun.metrics;
  const maxEpoch = metrics.length;
  const maxLoss = Math.max(...metrics.map(m => Math.max(m.train_loss, m.val_loss)));

  return (
    <div className="space-y-6">
      <div>
        <h4 className="text-sm font-medium text-gray-900 mb-2">Loss Over Time</h4>
        <div className="relative h-48 border border-gray-200 rounded-lg p-4 bg-gray-50">
          <svg className="w-full h-full" viewBox="0 0 400 160" preserveAspectRatio="none">
            <polyline
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2"
              points={metrics.map((m, i) =>
                `${(i / maxEpoch) * 400},${160 - (m.train_loss / maxLoss) * 160}`
              ).join(' ')}
            />
            <polyline
              fill="none"
              stroke="#ef4444"
              strokeWidth="2"
              points={metrics.map((m, i) =>
                `${(i / maxEpoch) * 400},${160 - (m.val_loss / maxLoss) * 160}`
              ).join(' ')}
            />
          </svg>
          <div className="absolute top-2 right-2 flex gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-blue-500 rounded"></div>
              <span>Train Loss</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span>Val Loss</span>
            </div>
          </div>
        </div>
      </div>

      {/* Accuracy Chart */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 mb-2">Accuracy Over Time</h4>
        <div className="relative h-48 border border-gray-200 rounded-lg p-4 bg-gray-50">
          <svg className="w-full h-full" viewBox="0 0 400 160" preserveAspectRatio="none">
            {/* Train Accuracy Line */}
            <polyline
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2"
              points={metrics.map((m, i) =>
                `${(i / maxEpoch) * 400},${160 - m.train_accuracy * 160}`
              ).join(' ')}
            />
            {/* Val Accuracy Line */}
            <polyline
              fill="none"
              stroke="#ef4444"
              strokeWidth="2"
              points={metrics.map((m, i) =>
                `${(i / maxEpoch) * 400},${160 - m.val_accuracy * 160}`
              ).join(' ')}
            />
          </svg>
          <div className="absolute top-2 right-2 flex gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-blue-500 rounded"></div>
              <span>Train Acc</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span>Val Acc</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Table */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 mb-2">Epoch Details</h4>
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-gray-700">Epoch</th>
                <th className="px-3 py-2 text-right font-medium text-gray-700">Train Loss</th>
                <th className="px-3 py-2 text-right font-medium text-gray-700">Val Loss</th>
                <th className="px-3 py-2 text-right font-medium text-gray-700">Train Acc</th>
                <th className="px-3 py-2 text-right font-medium text-gray-700">Val Acc</th>
                {metrics[0].learning_rate !== undefined && (
                  <th className="px-3 py-2 text-right font-medium text-gray-700">LR</th>
                )}
                {metrics[0].epoch_time !== undefined && (
                  <th className="px-3 py-2 text-right font-medium text-gray-700">Time (s)</th>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {metrics.map((metric) => (
                <tr key={metric.epoch} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-gray-900">{metric.epoch}</td>
                  <td className="px-3 py-2 text-right text-gray-600">{metric.train_loss.toFixed(4)}</td>
                  <td className="px-3 py-2 text-right text-gray-600">{metric.val_loss.toFixed(4)}</td>
                  <td className="px-3 py-2 text-right text-gray-600">{(metric.train_accuracy * 100).toFixed(2)}%</td>
                  <td className="px-3 py-2 text-right text-gray-600">{(metric.val_accuracy * 100).toFixed(2)}%</td>
                  {metric.learning_rate !== undefined && (
                    <td className="px-3 py-2 text-right text-gray-600">{metric.learning_rate.toFixed(6)}</td>
                  )}
                  {metric.epoch_time !== undefined && (
                    <td className="px-3 py-2 text-right text-gray-600">{metric.epoch_time.toFixed(2)}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
