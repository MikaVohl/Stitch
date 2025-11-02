import type { ChangeEvent } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useGraphStore } from '@/store/graphStore'
import type { DropoutLayer } from '@/types/graph'
import { formatShape } from '@/types/graph'

export function DropoutLayerNode({ id }: NodeProps) {
  const layer = useGraphStore(state => state.layers[id]) as DropoutLayer | undefined
  const inputShape = useGraphStore(state => state.getInputShape(id))
  const updateLayerParams = useGraphStore(state => state.updateLayerParams)
  const removeLayer = useGraphStore(state => state.removeLayer)

  if (!layer) return null

  const handleRateChange = (event: ChangeEvent<HTMLInputElement>) => {
    const value = Math.min(1, Math.max(0, parseFloat(event.target.value) || 0))
    updateLayerParams(id, { rate: Number.isNaN(value) ? 0 : value })
  }

  return (
    <div className="relative bg-orange-50 border-2 border-orange-500 rounded-lg shadow-lg min-w-[180px]">
      <button
        type="button"
        onClick={() => removeLayer(id)}
        className="absolute -right-2 -top-2 w-6 h-6 flex items-center justify-center rounded-full bg-orange-500 text-white text-xs font-bold shadow-md hover:bg-orange-600 focus:outline-none focus:ring-2 focus:ring-orange-300"
        aria-label="Remove dropout layer"
      >
        ×
      </button>

      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="bg-indigo-500! size-5! border-2! border-white!"
      />

      <div className="bg-orange-500 text-white px-3 py-1.5 rounded-t-md text-sm font-semibold">
        Dropout
      </div>

      <div className="p-3 space-y-3">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Rate:</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={layer.params.rate.toFixed(2)}
            onChange={handleRateChange}
            className="w-20 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
          />
        </div>

        <div className="text-xs text-gray-600">
          <span className="font-medium">Shape:</span> {formatShape(layer.shapeOut ?? inputShape)}
        </div>
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
