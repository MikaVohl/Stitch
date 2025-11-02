import { useState } from 'react'
import clsx from 'clsx'
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Info } from 'lucide-react'

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
  const [isOpen, setIsOpen] = useState(false)

  const updateParam = <K extends keyof Hyperparams>(key: K, value: Hyperparams[K]) => {
    const updated = { ...params, [key]: value }
    setParams(updated)
    onParamsChange?.(updated)
  }

  const updateOptimizer = <K extends keyof Hyperparams['optimizer']>(
    key: K,
    value: Hyperparams['optimizer'][K]
  ) => {
    const updated = { ...params, optimizer: { ...params.optimizer, [key]: value } }
    setParams(updated)
    onParamsChange?.(updated)
  }

  const renderLabelWithInfo = (label: string, tooltipContent: any) => (
  <div className="flex items-center gap-1">
    <span className="text-xs font-medium text-gray-600">{label}</span>
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          className="p-0 text-blue-600 hover:text-blue-800 focus:outline-none"
          aria-label={`Info about ${label}`}
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top">{tooltipContent}</TooltipContent>
    </Tooltip>
  </div>
)

  return (
    <div
      className={clsx(
        'bg-white rounded-lg shadow-lg border border-gray-200 w-[200px] md:w-[280px] pointer-events-auto',
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

      {/* Collapsible Content */}
      {isOpen && (
        <div className="p-4 space-y-3">

          {/* Epochs */}
          <div className="flex items-center justify-between">
            {renderLabelWithInfo('Epochs', <p>Number of complete passes through the training dataset.</p>)}
            <input
              type="number"
              value={params.epochs ?? ''}
              onChange={e => updateParam('epochs', e.target.value === '' ? ('' as any) : parseInt(e.target.value))}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1"
            />
          </div>

          {/* Batch Size */}
          <div className="flex items-center justify-between">
            {renderLabelWithInfo('Batch Size', <p>Number of samples processed before the model updates weights.</p>)}
            <input
              type="number"
              value={params.batch_size ?? ''}
              onChange={e => updateParam('batch_size', e.target.value === '' ? ('' as any) : parseInt(e.target.value))}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1"
            />
          </div>

          {/* Optimizer */}
          <div className="flex items-center justify-between">
            {renderLabelWithInfo('Optimizer', <p>Algorithm used to update model weights during training.</p>)}
            <select
              value={params.optimizer.type}
              onChange={e => updateOptimizer('type', e.target.value)}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="sgd">SGD</option>
              <option value="adam">Adam</option>
              <option value="rmsprop">RMSprop</option>
            </select>
          </div>

          {/* Learning Rate */}
          <div className="flex items-center justify-between">
            {renderLabelWithInfo('Learning Rate', <p>Controls how much to adjust model weights per update.</p>)}
            <input
              type="number"
              value={params.optimizer.lr ?? ''}
              onChange={e => updateOptimizer('lr', e.target.value === '' ? ('' as any) : parseFloat(e.target.value))}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              step="0.001"
              min="0"
            />
          </div>

          {/* Momentum */}
          <div className="flex items-center justify-between">
            {renderLabelWithInfo('Momentum', <p>Helps accelerate gradients and smooth convergence.</p>)}
            <input
              type="number"
              value={params.optimizer.momentum ?? ''}
              onChange={e => updateOptimizer('momentum', e.target.value === '' ? ('' as any) : parseFloat(e.target.value))}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              step="0.1"
              min="0"
              max="1"
            />
          </div>

          {/* Seed */}
          <div className="flex items-center justify-between">
            {renderLabelWithInfo('Seed', <p>Sets the random seed for reproducible training runs.</p>)}
            <input
              type="number"
              value={params.seed ?? ''}
              onChange={e => updateParam('seed', e.target.value === '' ? ('' as any) : parseInt(e.target.value))}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Train Split */}
          <div className="flex items-center justify-between">
            {renderLabelWithInfo('Train Split', <p>Proportion of data used for training vs validation.</p>)}
            <input
              type="number"
              value={params.train_split ?? ''}
              onChange={e => updateParam('train_split', e.target.value === '' ? ('' as any) : parseFloat(e.target.value))}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              step="0.1"
              min="0"
              max="1"
            />
          </div>

          {/* Shuffle */}
          <div className="flex items-center justify-between">
            {renderLabelWithInfo('Shuffle', <p>Whether to shuffle data at the start of each epoch.</p>)}
            <input
              type="checkbox"
              checked={params.shuffle}
              onChange={e => updateParam('shuffle', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
            />
          </div>

        </div>
      )}
    </div>
  )
}

export { type Hyperparams, DEFAULT_HYPERPARAMS }
