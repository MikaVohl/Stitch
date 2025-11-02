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
  epochs: 15,
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
  const [isOpen, setIsOpen] = useState(true)

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
    <div
      className={clsx(
        'bg-white rounded-lg shadow-lg border border-gray-200 w-[200px] md:w-[280px]',
        className
      )}
    >
      {/* Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-2.5 flex items-center cursor-pointer justify-between rounded-t-lg transition-colors hover:bg-gray-50 border-b border-gray-200"
      >
        <span className="font-semibold text-gray-700 text-sm">Hyperparameters</span>
        <svg
          className={clsx(
            'w-4 h-4 text-gray-500 transform transition-transform duration-200',
            isOpen ? 'rotate-180' : 'rotate-0'
          )}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown (Collapsible Content) */}
      {isOpen && (
        <div className="p-4 space-y-3">
          {/* Epochs */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Epochs</label>
            <input
              type="number"
              value={params.epochs ?? ''}
              onChange={(e) => {
                const val = e.target.value
                // allow empty string
                if (val === '') {
                  updateParam('epochs', '' as any) // temporarily store empty
                } else {
                  updateParam('epochs', parseInt(val))
                }
              }}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1"
            />
          </div>

          {/* Batch Size */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Batch Size</label>
            <input
              type="number"
              value={params.batch_size ?? ''}
              onChange={(e) => {
                const val = e.target.value
                if (val === '') {
                  updateParam('batch_size', '' as any)
                } else {
                  updateParam('batch_size', parseInt(val))
                }
              }}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1"
            />
          </div>

          {/* Optimizer */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Optimizer</label>
            <select
              value={params.optimizer.type}
              onChange={(e) => updateOptimizer('type', e.target.value)}
              className="w-20 px-1 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
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
              value={params.optimizer.lr ?? ''}
              onChange={(e) => {
                const val = e.target.value
                if (val === '') {
                  updateOptimizer('lr', '' as any)
                } else {
                  updateOptimizer('lr', parseFloat(val))
                }
              }}
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
              value={params.optimizer.momentum ?? ''}
              onChange={(e) => {
                const val = e.target.value
                if (val === '') {
                  updateOptimizer('momentum', '' as any)
                } else {
                  updateOptimizer('momentum', parseFloat(val))
                }
              }}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              step="0.1"
              min="0"
              max="1"
            />
          </div>

          {/* Seed */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Seed</label>
            <input
              type="number"
              value={params.seed ?? ''}
              onChange={(e) => {
                const val = e.target.value
                if (val === '') {
                  updateParam('seed', '' as any)
                } else {
                  updateParam('seed', parseInt(val))
                }
              }}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Train Split */}
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-600">Train Split</label>
            <input
              type="number"
              value={params.train_split ?? ''}
              onChange={(e) => {
                const val = e.target.value
                if (val === '') {
                  updateParam('train_split', '' as any)
                } else {
                  updateParam('train_split', parseFloat(val))
                }
              }}
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
      )}
    </div>
  )
}

export { type Hyperparams, DEFAULT_HYPERPARAMS }
