import { useModel, type TrainingRun } from "@/hooks/useModels";
import { Link, useParams } from "react-router-dom"
import { summarizeArchitecture } from "./Models";
import { MetricsCharts } from "@/components/MetricsCharts";
import { HyperparamsTable } from "@/components/HyperparamsTable";

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
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{model.name}</h1>
            <p className="mt-1 text-sm text-gray-500">
              {model.created_at
                ? new Date(model.created_at).toLocaleString()
                : 'Creation time unknown'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-gray-100 px-3 py-1 text-sm font-medium text-gray-700">
              {model.architecture?.layers?.length ?? 0} layers
            </span>
            <Link
              to={`/test/${model.model_id}`}
              className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-blue-500 transition-transform hover:translate-x-0.5"
            >
              Test this model
              <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M3 10a.75.75 0 01.75-.75h9.69l-3.22-3.22a.75.75 0 111.06-1.06l4.5 4.5a.75.75 0 010 1.06l-4.5 4.5a.75.75 0 11-1.06-1.06l3.22-3.22H3.75A.75.75 0 013 10z" clipRule="evenodd" />
              </svg>
            </Link>
          </div>
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
            <HyperparamsTable hyperparams={model.hyperparams} />
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

  return (
    <div className="space-y-6">
      {/* Charts */}
      <MetricsCharts metrics={metrics} />

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
