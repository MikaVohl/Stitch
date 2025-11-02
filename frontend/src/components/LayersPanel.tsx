import { useState } from 'react'
import clsx from 'clsx'
import type { DragEvent } from 'react'
import type { ActivationType } from '@/types/graph'

type DenseTemplate = {
  id: string
  label: string
  description: string
  kind: 'Dense'
  params: {
    units: number
    activation: ActivationType
  }
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
}

type FlattenTemplate = {
  id: string
  label: string
  description: string
  kind: 'Flatten'
  params: Record<string, never>
}

type DropoutTemplate = {
  id: string
  label: string
  description: string
  kind: 'Dropout'
  params: {
    rate: number
  }
}

type LayerTemplate = DenseTemplate | ConvTemplate | FlattenTemplate | DropoutTemplate

const TEMPLATE_STYLES: Record<
  LayerTemplate['kind'],
  {
    border: string
    background: string
    hover: string
    label: string
    description: string
  }
> = {
  Dense: {
    border: 'border-blue-300',
    background: 'bg-blue-50',
    hover: 'hover:bg-blue-100',
    label: 'text-blue-700',
    description: 'text-blue-600',
  },
  Convolution: {
    border: 'border-indigo-300',
    background: 'bg-indigo-50',
    hover: 'hover:bg-indigo-100',
    label: 'text-indigo-700',
    description: 'text-indigo-600',
  },
  Flatten: {
    border: 'border-yellow-300',
    background: 'bg-yellow-50',
    hover: 'hover:bg-yellow-100',
    label: 'text-yellow-700',
    description: 'text-yellow-600',
  },
  Dropout: {
    border: 'border-orange-300',
    background: 'bg-orange-50',
    hover: 'hover:bg-orange-100',
    label: 'text-orange-700',
    description: 'text-orange-600',
  },
}

const DENSE_LAYER_TEMPLATE: DenseTemplate = {
  id: 'dense-layer',
  label: 'Dense Layer',
  description: 'Fully connected linear layer',
  kind: 'Dense',
  params: {
    units: 64,
    activation: 'relu',
  },
}

const CONV_LAYER_TEMPLATE: ConvTemplate = {
  id: 'conv-layer',
  label: 'Conv Layer',
  description: '2D convolutional layer',
  kind: 'Convolution',
  params: {
    filters: 32,
    kernel: 3,
    stride: 1,
    padding: 'same',
    activation: 'relu',
  },
}

const FLATTEN_LAYER_TEMPLATE: FlattenTemplate = {
  id: 'flatten-layer',
  label: 'Flatten',
  description: 'Convert image features into a vector',
  kind: 'Flatten',
  params: {},
}

const DROPOUT_LAYER_TEMPLATE: DropoutTemplate = {
  id: 'dropout-layer',
  label: 'Dropout',
  description: 'Randomly drop a fraction of activations',
  kind: 'Dropout',
  params: {
    rate: 0.2,
  },
}

const LAYER_TEMPLATES: LayerTemplate[] = [
  DENSE_LAYER_TEMPLATE,
  CONV_LAYER_TEMPLATE,
  FLATTEN_LAYER_TEMPLATE,
  DROPOUT_LAYER_TEMPLATE,
]

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

export function LayersPanel({ className }: { className?: string }) {
  const [isOpen, setIsOpen] = useState(true)

  return (
    <div className={clsx('bg-white rounded-lg shadow-lg border border-gray-200 w-[200px] md:w-[280px]', className)}>
      {/* Header with rotating arrow */}
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

      {/* Collapsible content */}
      {isOpen && (
        <div className="p-4 flex flex-col gap-3 min-w-[280px]">
          <p className="text-xs text-gray-500">
            Drag a preset onto the canvas, then tune the settings inline.
          </p>

          {LAYER_TEMPLATES.map((template) => {
            const style = TEMPLATE_STYLES[template.kind]
            return (
              <div
                key={template.id}
                draggable
                onDragStart={createDragStartHandler(template)}
                className={clsx(
                  'border border-dashed rounded-lg px-3 py-2 cursor-grab active:cursor-grabbing transition-colors',
                  style.border,
                  style.background,
                  style.hover
                )}
              >
                <div className={`text-sm font-semibold ${style.label}`}>{template.label}</div>
                <div className={`text-xs ${style.description}`}>{template.description}</div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
