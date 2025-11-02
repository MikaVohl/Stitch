import type { AnyLayer, GraphEdge } from '@/types/graph'

export type PresetType = 'blank' | 'simple' | 'complex'

interface PresetChipsProps {
  onPresetSelect: (preset: PresetType) => void
}

export function PresetChips({ onPresetSelect }: PresetChipsProps) {
  const baseClasses =
    'px-3 py-1.5 text-xs font-medium rounded-full transition-colors cursor-pointer border backdrop-blur-sm'

  return (
    <div className="px-3 py-2 flex gap-2 items-center pointer-events-auto">
      <span className="text-xs font-semibold text-gray-600">Preset:</span>
      <button
        onClick={() => onPresetSelect('blank')}
        className={
          `${baseClasses} bg-blue-50/80 text-blue-700 hover:bg-blue-100 border-blue-200/50`
        }
      >
        Blank
      </button>
      <button
        onClick={() => onPresetSelect('simple')}
        className={
          `${baseClasses} bg-blue-50/80 text-blue-700 hover:bg-blue-100 border-blue-200/50`
        }
      >
        Simple
      </button>
      <button
        onClick={() => onPresetSelect('complex')}
        className={
          `${baseClasses} bg-blue-50/80 text-blue-700 hover:bg-blue-100 border-blue-200/50`
        }
      >
        Complex
      </button>
    </div>
  )
}

// Preset definitions
export function getPresetGraph(preset: PresetType): {
  layers: Record<string, AnyLayer>
  edges: GraphEdge[]
} {
  const makeEdge = (id: string, source: string, target: string): GraphEdge => ({
    id,
    source,
    target,
    sourceHandle: 'output',
    targetHandle: 'input',
  })

  switch (preset) {
    case 'blank':
      return {
        layers: {
          'input-1': {
            id: 'input-1',
            kind: 'Input',
            params: { size: 784, channels: 1, height: 28, width: 28 },
            position: { x: 50, y: 200 },
          },
          'output-1': {
            id: 'output-1',
            kind: 'Output',
            params: { classes: 10, activation: 'softmax' },
            position: { x: 350, y: 200 },
          },
        },
        edges: [
          makeEdge('input-1-output-1', 'input-1', 'output-1'),
        ],
      }

    case 'simple':
      return {
        layers: {
          'input-1': {
            id: 'input-1',
            kind: 'Input',
            params: { size: 784, channels: 1, height: 28, width: 28 },
            position: { x: 50, y: 200 },
          },
          'flatten-1': {
            id: 'flatten-1',
            kind: 'Flatten',
            params: {},
            position: { x: 250, y: 200 },
          },
          'dense-1': {
            id: 'dense-1',
            kind: 'Dense',
            params: { units: 128, activation: 'relu' },
            position: { x: 450, y: 200 },
          },
          'output-1': {
            id: 'output-1',
            kind: 'Output',
            params: { classes: 10, activation: 'softmax' },
            position: { x: 650, y: 200 },
          },
        },
        edges: [
          makeEdge('input-1-flatten-1', 'input-1', 'flatten-1'),
          makeEdge('flatten-1-dense-1', 'flatten-1', 'dense-1'),
          makeEdge('dense-1-output-1', 'dense-1', 'output-1'),
        ],
      }

    case 'complex':
      return {
        layers: {
          'input-1': {
            id: 'input-1',
            kind: 'Input',
            params: { size: 784, channels: 1, height: 28, width: 28 },
            position: { x: 50, y: 200 },
          },
          'conv-1': {
            id: 'conv-1',
            kind: 'Convolution',
            params: { filters: 32, kernel: 3, stride: 1, padding: 'same', activation: 'relu' },
            position: { x: 250, y: 200 },
          },
          'pool-1': {
            id: 'pool-1',
            kind: 'Pooling',
            params: { type: 'max', pool_size: 2, stride: 2, padding: 0 },
            position: { x: 450, y: 200 },
          },
          'conv-2': {
            id: 'conv-2',
            kind: 'Convolution',
            params: { filters: 64, kernel: 3, stride: 1, padding: 'same', activation: 'relu' },
            position: { x: 650, y: 200 },
          },
          'pool-2': {
            id: 'pool-2',
            kind: 'Pooling',
            params: { type: 'max', pool_size: 2, stride: 2, padding: 0 },
            position: { x: 850, y: 200 },
          },
          'flatten-1': {
            id: 'flatten-1',
            kind: 'Flatten',
            params: {},
            position: { x: 1050, y: 200 },
          },
          'dense-1': {
            id: 'dense-1',
            kind: 'Dense',
            params: { units: 128, activation: 'relu' },
            position: { x: 1250, y: 200 },
          },
          'dropout-1': {
            id: 'dropout-1',
            kind: 'Dropout',
            params: { rate: 0.5 },
            position: { x: 1450, y: 200 },
          },
          'output-1': {
            id: 'output-1',
            kind: 'Output',
            params: { classes: 10, activation: 'softmax' },
            position: { x: 1650, y: 200 },
          },
        },
        edges: [
          makeEdge('input-1-conv-1', 'input-1', 'conv-1'),
          makeEdge('conv-1-pool-1', 'conv-1', 'pool-1'),
          makeEdge('pool-1-conv-2', 'pool-1', 'conv-2'),
          makeEdge('conv-2-pool-2', 'conv-2', 'pool-2'),
          makeEdge('pool-2-flatten-1', 'pool-2', 'flatten-1'),
          makeEdge('flatten-1-dense-1', 'flatten-1', 'dense-1'),
          makeEdge('dense-1-dropout-1', 'dense-1', 'dropout-1'),
          makeEdge('dropout-1-output-1', 'dropout-1', 'output-1'),
        ],
      }
  }
}
