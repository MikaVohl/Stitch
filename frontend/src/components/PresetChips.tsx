import type { AnyLayer, GraphEdge } from '@/types/graph'

export type PresetType = 'blank' | 'simple' | 'complex'

interface PresetChipsProps {
  onPresetSelect: (preset: PresetType) => void
  currentPreset?: PresetType
}

export function PresetChips({ onPresetSelect }: PresetChipsProps) {
  const chipClasses = 'px-3 py-1.5 text-xs font-medium rounded-full transition-colors cursor-pointer bg-blue-50/80 text-blue-700 hover:bg-blue-100 backdrop-blur-sm border border-blue-200/50'

  return (
    <div className="px-3 py-2 flex gap-2 items-center pointer-events-auto">
      <span className="text-xs font-semibold text-gray-600">Preset:</span>
      <button
        onClick={() => onPresetSelect('blank')}
        className={chipClasses}
      >
        Blank
      </button>
      <button
        onClick={() => onPresetSelect('simple')}
        className={chipClasses}
      >
        Simple
      </button>
      <button
        onClick={() => onPresetSelect('complex')}
        className={chipClasses}
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
          { id: 'input-1-output-1', source: 'input-1', target: 'output-1' },
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
          { id: 'input-1-flatten-1', source: 'input-1', target: 'flatten-1' },
          { id: 'flatten-1-dense-1', source: 'flatten-1', target: 'dense-1' },
          { id: 'dense-1-output-1', source: 'dense-1', target: 'output-1' },
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
          { id: 'input-1-conv-1', source: 'input-1', target: 'conv-1' },
          { id: 'conv-1-pool-1', source: 'conv-1', target: 'pool-1' },
          { id: 'pool-1-conv-2', source: 'pool-1', target: 'conv-2' },
          { id: 'conv-2-pool-2', source: 'conv-2', target: 'pool-2' },
          { id: 'pool-2-flatten-1', source: 'pool-2', target: 'flatten-1' },
          { id: 'flatten-1-dense-1', source: 'flatten-1', target: 'dense-1' },
          { id: 'dense-1-dropout-1', source: 'dense-1', target: 'dropout-1' },
          { id: 'dropout-1-output-1', source: 'dropout-1', target: 'output-1' },
        ],
      }
  }
}
