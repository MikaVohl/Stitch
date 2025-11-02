import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useGraphStore } from '../../store/graphStore'
import type { ConvLayer } from '../../types/graph'
import { formatShape, calculateParams } from '../../types/graph'

const activationColors: Record<string, string> = {
  relu: 'bg-indigo-100 text-indigo-700 border-indigo-300',
  sigmoid: 'bg-purple-100 text-purple-700 border-purple-300',
  tanh: 'bg-green-100 text-green-700 border-green-300',
  none: 'bg-gray-100 text-gray-700 border-gray-300',
}

const paddingOptions = ['valid', 'same'] as const

export function ConvLayerNode({ id }: NodeProps) {
  const layer = useGraphStore((state) => state.layers[id]) as ConvLayer | undefined
  const inputShape = useGraphStore((state) => state.getInputShape(id))
  const updateLayerParams = useGraphStore((state) => state.updateLayerParams)
  const removeLayer = useGraphStore((state) => state.removeLayer)

  if (!layer) return null

  const paramCount = calculateParams(layer, inputShape)

  return (
    <div className="relative bg-indigo-50 border-2 border-indigo-500 rounded-lg shadow-lg min-w-[180px]">
      <button
        type="button"
        onClick={() => removeLayer(id)}
        className="absolute -right-2 -top-2 w-6 h-6 flex items-center justify-center rounded-full bg-indigo-500 text-white text-xs font-bold shadow-md hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-300"
        aria-label="Remove convolution layer"
      >
        ×
      </button>
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="bg-indigo-500! size-5! border-2! border-white!"
      />

      <div className="bg-indigo-500 text-white px-3 py-1.5 rounded-t-md text-sm font-semibold">
        Convolution Layer
      </div>

      <div className="p-3 space-y-3">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Filters:</label>
          <input
            type="number"
            value={layer.params.filters}
            onChange={(e) =>
              updateLayerParams(id, { filters: Math.max(1, parseInt(e.target.value) || 1) })
            }
            min={1}
            className="w-16 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Kernel:</label>
          <input
            type="number"
            value={layer.params.kernel}
            onChange={(e) =>
              updateLayerParams(id, { kernel: Math.max(1, parseInt(e.target.value) || 1) })
            }
            min={1}
            className="w-16 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <span className="text-xs text-gray-500">× {layer.params.kernel}</span>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Stride:</label>
          <input
            type="number"
            value={layer.params.stride}
            onChange={(e) =>
              updateLayerParams(id, { stride: Math.max(1, parseInt(e.target.value) || 1) })
            }
            min={1}
            className="w-16 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Padding:</label>
          <select
            value={layer.params.padding}
            onChange={(e) => updateLayerParams(id, { padding: e.target.value as 'valid' | 'same' })}
            className="px-2 py-1 text-xs rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {paddingOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Activation:</label>
          <select
            value={layer.params.activation}
            onChange={(e) => updateLayerParams(id, { activation: e.target.value })}
            className={`px-2 py-1 text-xs rounded border ${activationColors[layer.params.activation] ?? activationColors.none
              } font-medium`}
          >
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
            <option value="none">None</option>
          </select>
        </div>

        <div className="text-xs text-gray-600">
          <span className="font-medium">Input:</span> {formatShape(inputShape)}
        </div>
        <div className="text-xs text-gray-600">
          <span className="font-medium">Shape:</span> {formatShape(layer.shapeOut)}
        </div>
        {paramCount > 0 && (
          <div className="text-xs text-gray-600">
            <span className="font-medium">Params:</span> {paramCount.toLocaleString()}
          </div>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="bg-indigo-500! size-5! border-2! border-white!"
      />
    </div>
  )
}
