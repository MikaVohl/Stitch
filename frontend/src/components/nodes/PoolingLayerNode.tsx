import type { ChangeEvent } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useGraphStore } from '@/store/graphStore'
import type { PoolingLayer } from '@/types/graph'
import { formatShape } from '@/types/graph'

export function PoolingLayerNode({ id }: NodeProps) {
  const layer = useGraphStore(state => state.layers[id]) as PoolingLayer | undefined
  const inputShape = useGraphStore(state => state.getInputShape(id))
  const updateLayerParams = useGraphStore(state => state.updateLayerParams)
  const removeLayer = useGraphStore(state => state.removeLayer)

  if (!layer) return null

  const handleNumberChange = (field: 'pool_size' | 'stride' | 'padding') => (event: ChangeEvent<HTMLInputElement>) => {
    const value = Math.max(0, parseInt(event.target.value) || 0)
    updateLayerParams(id, { [field]: field === 'pool_size' ? Math.max(1, value) : value })
  }

  return (
    <div className="relative bg-emerald-50 border-2 border-emerald-500 rounded-lg shadow-lg min-w-[190px]">
      <button
        type="button"
        onClick={() => removeLayer(id)}
        className="absolute -right-2 -top-2 w-6 h-6 flex items-center justify-center rounded-full bg-emerald-500 text-white text-xs font-bold shadow-md hover:bg-emerald-600 focus:outline-none focus:ring-2 focus:ring-emerald-300"
        aria-label="Remove pooling layer"
      >
        ×
      </button>

      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="bg-indigo-500! !w-5 !h-5 !border-2 !border-white"
      />

      <div className="bg-emerald-500 text-white px-3 py-1.5 rounded-t-md text-sm font-semibold">
        Max Pooling
      </div>

      <div className="p-3 space-y-2">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Pool:</label>
          <input
            type="number"
            min={1}
            value={layer.params.pool_size}
            onChange={handleNumberChange('pool_size')}
            className="w-14 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Stride:</label>
          <input
            type="number"
            min={1}
            value={layer.params.stride}
            onChange={handleNumberChange('stride')}
            className="w-14 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Padding:</label>
          <input
            type="number"
            min={0}
            value={layer.params.padding}
            onChange={handleNumberChange('padding')}
            className="w-14 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
        </div>

        <div className="text-xs text-gray-600">
          <span className="font-medium">Out shape:</span> {formatShape(layer.shapeOut ?? inputShape)}
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="bg-indigo-500! !w-5 !h-5 !border-2 !border-white"
      />
    </div>
  )
}
