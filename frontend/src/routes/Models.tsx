import { useState } from 'react';
import { HyperparamsTable } from '@/components/HyperparamsTable';
import { useModels, type StoredLayer } from '@/hooks/useModels';
import { Link } from 'react-router-dom';

export function summarizeArchitecture(layers?: StoredLayer[]): string {
  if (!layers || layers.length === 0) return 'No layers recorded.'

  return layers
    .map((layer) => {
      const type = layer.type?.toUpperCase?.() ?? 'LAYER'
      if (layer.type === 'linear') {
        return `${type} (${layer.in ?? '?'} → ${layer.out ?? '?'})`
      }
      return type
    })
    .join('  •  ')
}

export function summarizeHyperparams(hyperparams?: Record<string, unknown>): string {
  if (!hyperparams || Object.keys(hyperparams).length === 0) {
    return 'No hyperparameters recorded.'
  }

  const parts: string[] = []
  const entries = Object.entries(hyperparams)

  for (const [key, value] of entries) {
    if (typeof value === 'object' && value !== null) {
      parts.push(`${key}: ${JSON.stringify(value)}`)
    } else {
      parts.push(`${key}: ${String(value)}`)
    }
  }

  return parts.join('  •  ')
}

export default function Models() {
  const { data: models, isLoading, isError, error } = useModels();
  const [sortBy, setSortBy] = useState<'newest' | 'oldest' | 'accuracy'>('newest');

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

  if (models?.length === 0) {
    return (

      <main className="min-h-[calc(100vh-4rem)] bg-gray-50 py-10">
        <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 sm:px-6 lg:px-8">
          <header className="flex flex-col gap-2">
            <h1 className="text-3xl font-semibold text-gray-900">Models</h1>
            <p className="text-sm text-gray-600">
              Every training run creates a snapshot of the architecture and hyperparameters you used. Review them here
              before jumping back into the arena.
            </p>
          </header>
          <div>
            <div className="rounded-xl border border-gray-200 bg-white p-8 text-gray-600 shadow-sm">
              <p>No models saved yet. Train a network in the Playground to populate this list.</p>
            </div>
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-[calc(100vh-4rem)] bg-gray-50 py-10">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-semibold text-gray-900">Models</h1>
            <div className="flex items-center gap-2">
              <label htmlFor="sort" className="text-sm font-medium text-gray-700">Sort by:</label>
              <select
                id="sort"
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as 'newest' | 'oldest' | 'accuracy')}
                className="rounded-md border cursor-pointer border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="newest">Newest first</option>
                <option value="oldest">Oldest first</option>
                <option value="accuracy">Highest accuracy</option>
              </select>
            </div>
          </div>
          <p className="text-sm text-gray-600">
            Every training run creates a snapshot of the architecture and hyperparameters you used. Review them here
            before jumping back into the arena.
          </p>
        </header>
        <div>

          <div className="grid gap-6 lg:grid-cols-2">
            {(models ?? [])
              .sort((a, b) => {
                if (sortBy === 'accuracy') {
                  const aAcc = a.highest_accuracy ?? -1;
                  const bAcc = b.highest_accuracy ?? -1;
                  return bAcc - aAcc;
                }
                const aDate = a.created_at ? new Date(a.created_at).getTime() : 0;
                const bDate = b.created_at ? new Date(b.created_at).getTime() : 0;
                return sortBy === 'newest' ? bDate - aDate : aDate - bDate;
              })
              .map((model) => (
              <Link to={`./${model.model_id}`} key={model.model_id} className="block">
                <article
                  className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm transition-transform hover:-translate-y-1 hover:shadow-lg"
                >
                  <header className="flex items-start justify-between">
                    <div>
                      <h3 className="text-2xl font-semibold text-gray-900">{model.name}</h3>
                      <p className="mt-1 text-sm text-gray-500">
                        {model.created_at
                          ? new Date(model.created_at).toLocaleString()
                          : 'Creation time unknown'}
                      </p>
                    </div>
                    <span className="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-600">
                      {model.architecture?.layers?.length ?? 0} layers
                    </span>
                  </header>

                  <dl className="mt-4 space-y-3 text-sm text-gray-600">
                    <div className="flex gap-2">
                      <dt className="w-32 font-medium text-gray-900">Input size</dt>
                      <dd>{model.architecture?.input_size ?? '—'}</dd>
                    </div>
                    <div className="flex gap-2">
                      <dt className="w-32 font-medium text-gray-900">Highest accuracy</dt>
                      <dd>
                        {model.highest_accuracy != null
                          ? `${(model.highest_accuracy * 100).toFixed(1)}%`
                          : '—'}
                      </dd>
                    </div>
                    <div>
                      <dt className="font-medium text-gray-900">Architecture</dt>
                      <dd className="mt-1 rounded-lg bg-gray-50 p-3 text-xs font-mono text-gray-700">
                        {summarizeArchitecture(model.architecture?.layers)}
                      </dd>
                    </div>
                  </dl>
                </article>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </main>
  )
}
