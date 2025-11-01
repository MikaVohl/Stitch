import { useState } from 'react'
import clsx from 'clsx'

interface Hyperparams {
  epochs: number
  batch_size: number
  optimizer: {
    type: string
    lr: number
    momentum: number
  }
  loss: string
  seed: number
  train_split: number
  shuffle: boolean
}

const DEFAULT_HYPERPARAMS: Hyperparams = {
  epochs: 5,
  batch_size: 64,
  optimizer: { type: 'sgd', lr: 0.1, momentum: 0.0 },
  loss: 'cross_entropy',
  seed: 42,
  train_split: 0.9,
  shuffle: true,
}

export function HyperparamsPanel({
  onParamsChange,
  className,
}: {
  onParamsChange?: (params: Hyperparams) => void
  className?: string
}) {
  const [params, setParams] = useState<Hyperparams>(DEFAULT_HYPERPARAMS)

  const updateParam = <K extends keyof Hyperparams>(key: K, value: Hyperparams[K]) => {
    const updated = { ...params, [key]: value }
    setParams(updated)
    onParamsChange?.(updated)
  }

  const updateOptimizer = <K extends keyof Hyperparams['optimizer']>(
    key: K,
    value: Hyperparams['optimizer'][K]
  ) => {
    const updated = {
      ...params,
      optimizer: { ...params.optimizer, [key]: value },
    }
    setParams(updated)
    onParamsChange?.(updated)
  }

  return (
    <div className={clsx('bg-white rounded-lg shadow-lg border border-gray-200', className)}>
      {/* Header */}
      <button
        className="w-full px-4 py-2.5 flex items-center justify-between rounded-t-lg transition-colors"
      >
        <span className="font-semibold text-gray-700 text-sm">Hyperparameters</span>
      </button>

      {/* Expanded Panel */}
      
        <div className="p-4 space-y-3 border-t border-gray-200 min-w-[280px]">
          {/* Epochs */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Epochs</label>
            <input
              type="number"
              value={params.epochs}
              onChange={(e) => updateParam('epochs', parseInt(e.target.value) || 1)}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1"
            />
          </div>

          {/* Batch Size */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Batch Size</label>
            <input
              type="number"
              value={params.batch_size}
              onChange={(e) => updateParam('batch_size', parseInt(e.target.value) || 1)}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1"
            />
          </div>

          {/* Optimizer Type */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Optimizer</label>
            <select
              value={params.optimizer.type}
              onChange={(e) => updateOptimizer('type', e.target.value)}
              className="w-24 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="sgd">SGD</option>
              <option value="adam">Adam</option>
              <option value="rmsprop">RMSprop</option>
            </select>
          </div>

          {/* Learning Rate */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Learning Rate</label>
            <input
              type="number"
              value={params.optimizer.lr}
              onChange={(e) => updateOptimizer('lr', parseFloat(e.target.value) || 0.001)}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              step="0.001"
              min="0"
            />
          </div>

          {/* Momentum */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Momentum</label>
            <input
              type="number"
              value={params.optimizer.momentum}
              onChange={(e) => updateOptimizer('momentum', parseFloat(e.target.value) || 0)}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              step="0.1"
              min="0"
              max="1"
            />
          </div>

          {/* Loss */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Loss</label>
            <select
              value={params.loss}
              onChange={(e) => updateParam('loss', e.target.value)}
              className="w-32 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="cross_entropy">Cross Entropy</option>
              <option value="mse">MSE</option>
            </select>
          </div>

          {/* Seed */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Seed</label>
            <input
              type="number"
              value={params.seed}
              onChange={(e) => updateParam('seed', parseInt(e.target.value) || 0)}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Train Split */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Train Split</label>
            <input
              type="number"
              value={params.train_split}
              onChange={(e) => updateParam('train_split', parseFloat(e.target.value) || 0.5)}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              step="0.1"
              min="0"
              max="1"
            />
          </div>

          {/* Shuffle */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Shuffle</label>
            <input
              type="checkbox"
              checked={params.shuffle}
              onChange={(e) => updateParam('shuffle', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
    </div>
  )
}

export { type Hyperparams, DEFAULT_HYPERPARAMS }
