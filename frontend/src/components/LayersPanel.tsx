import { useState } from 'react'
import clsx from 'clsx'
import { Info } from 'lucide-react'
import type { DragEvent } from 'react'
import type { ActivationType } from '@/types/graph'
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from '@/components/ui/tooltip'

// --- Types ---
type DenseTemplate = {
  id: string
  label: string
  description: string
  kind: 'Dense'
  params: { units: number; activation: ActivationType }
  docsUrl: string
}
type ConvTemplate = {
  id: string
  label: string
  description: string
  kind: 'Convolution'
  params: {
    filters: number
    kernel: number
    stride: number
    padding: 'valid' | 'same'
    activation: Exclude<ActivationType, 'softmax'>
  }
  docsUrl: string
}
type FlattenTemplate = {
  id: string
  label: string
  description: string
  kind: 'Flatten'
  params: Record<string, never>
  docsUrl: string
}
type DropoutTemplate = {
  id: string
  label: string
  description: string
  kind: 'Dropout'
  params: { rate: number }
  docsUrl: string
}
type PoolingTemplate = {
  id: string
  label: string
  description: string
  kind: 'Pooling'
  params: {
    type: 'max'
    pool_size: number
    stride: number
    padding: number
  }
  docsUrl: string
}

type LayerTemplate =
  | DenseTemplate
  | ConvTemplate
  | FlattenTemplate
  | DropoutTemplate
  | PoolingTemplate

// --- Styles ---
const TEMPLATE_STYLES: Record<
  LayerTemplate['kind'],
  {
    border: string
    background: string
    hover: string
    label: string
    description: string
    icon: string
    iconHover: string
  }
> = {
  Dense: {
    border: 'border-blue-300',
    background: 'bg-blue-50',
    hover: 'hover:bg-blue-100',
    label: 'text-blue-700',
    description: 'text-blue-600',
    icon: 'text-blue-600',
    iconHover: 'hover:text-blue-800',
  },
  Convolution: {
    border: 'border-indigo-300',
    background: 'bg-indigo-50',
    hover: 'hover:bg-indigo-100',
    label: 'text-indigo-700',
    description: 'text-indigo-600',
    icon: 'text-indigo-600',
    iconHover: 'hover:text-indigo-800',
  },
  Flatten: {
    border: 'border-yellow-300',
    background: 'bg-yellow-50',
    hover: 'hover:bg-yellow-100',
    label: 'text-yellow-700',
    description: 'text-yellow-600',
    icon: 'text-yellow-600',
    iconHover: 'hover:text-yellow-800',
  },
  Dropout: {
    border: 'border-orange-300',
    background: 'bg-orange-50',
    hover: 'hover:bg-orange-100',
    label: 'text-orange-700',
    description: 'text-orange-600',
    icon: 'text-orange-600',
    iconHover: 'hover:text-orange-800',
  },
  Pooling: {
    border: 'border-emerald-300',
    background: 'bg-emerald-50',
    hover: 'hover:bg-emerald-100',
    label: 'text-emerald-700',
    description: 'text-emerald-600',
    icon: 'text-emerald-600',
    iconHover: 'hover:text-emerald-800',
  },
}

// --- Templates ---
const LAYER_TEMPLATES: LayerTemplate[] = [
  {
    id: 'dense-layer',
    label: 'Dense Layer',
    description: 'Fully connected layer connecting all inputs to all outputs.',
    kind: 'Dense',
    params: { units: 64, activation: 'relu' },
    docsUrl: 'https://keras.io/api/layers/core_layers/dense/',
  },
  {
    id: 'conv-layer',
    label: 'Conv Layer',
    description: 'Applies 2D convolution operations to extract spatial features.',
    kind: 'Convolution',
    params: { filters: 32, kernel: 3, stride: 1, padding: 'same', activation: 'relu' },
    docsUrl: 'https://keras.io/api/layers/convolution_layers/convolution2d/',
  },
  {
    id: 'flatten-layer',
    label: 'Flatten',
    description: 'Converts multi-dimensional features into a single vector.',
    kind: 'Flatten',
    params: {},
    docsUrl: 'https://keras.io/api/layers/reshaping_layers/flatten/',
  },
  {
    id: 'dropout-layer',
    label: 'Dropout',
    description: 'Randomly drops a fraction of neurons during training to prevent overfitting.',
    kind: 'Dropout',
    params: { rate: 0.2 },
    docsUrl: 'https://keras.io/api/layers/regularization_layers/dropout/',
  },
  {
    id: 'maxpool-layer',
    label: 'Max Pooling',
    description: 'Downsamples feature maps by taking the maximum value in each window.',
    kind: 'Pooling',
    params: { type: 'max', pool_size: 2, stride: 2, padding: 0 },
    docsUrl: 'https://keras.io/api/layers/pooling_layers/max_pooling2d/',
  },
]

// --- Drag Handler ---
function createDragStartHandler(template: LayerTemplate) {
  return (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.effectAllowed = 'copy'
    event.dataTransfer.setData(
      'application/layer-template',
      JSON.stringify({
        kind: template.kind,
        params: template.params,
      })
    )
  }
}

// --- Component ---
export function LayersPanel({ className }: { className?: string }) {
  const [isOpen, setIsOpen] = useState(false)

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
        className="w-full px-4 py-2.5 flex cursor-pointer items-center justify-between rounded-t-lg transition-colors hover:bg-gray-50 border-b border-gray-200"
      >
        <span className="font-semibold text-gray-700 text-sm">Layer Palette</span>
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

      {/* Content */}
      {isOpen && (
        <div className="p-2 flex flex-col gap-3 min-w-[280px]">
          <p className="text-xs text-gray-500">
            Drag a preset onto the canvas, then adjust its parameters.
          </p>

          {LAYER_TEMPLATES.map((template) => {
            const style = TEMPLATE_STYLES[template.kind]

            return (
              <div
                key={template.id}
                draggable
                onDragStart={createDragStartHandler(template)}
                className={clsx(
                  'flex flex-col border border-dashed rounded-lg px-3 py-2 cursor-grab active:cursor-grabbing transition-colors',
                  style.border,
                  style.background,
                  style.hover
                )}
              >
                <div className="flex items-center gap-1">
                  <div className={`text-sm font-semibold ${style.label}`}>{template.label}</div>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        className={clsx(
                          'rounded-full p-1 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-1',
                          style.icon,
                          style.iconHover
                        )}
                        aria-label={`Info about ${template.label}`}
                      >
                        <Info className="h-3.5 w-3.5" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-[260px] text-xs">
                      <p className="mb-2">{template.description}</p>
                      <a
                        href={template.docsUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={clsx('underline font-medium', style.icon)}
                      >
                        Learn more → Keras Docs
                      </a>
                    </TooltipContent>
                  </Tooltip>
                </div>

                <div className={`text-xs ${style.description}`}>{template.description}</div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}