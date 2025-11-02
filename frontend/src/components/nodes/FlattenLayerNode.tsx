import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useGraphStore } from '@/store/graphStore'
import type { FlattenLayer } from '@/types/graph'
import { formatShape } from '@/types/graph'

export function FlattenLayerNode({ id }: NodeProps) {
  const layer = useGraphStore(state => state.layers[id]) as FlattenLayer | undefined
  const inputShape = useGraphStore(state => state.getInputShape(id))
  const removeLayer = useGraphStore(state => state.removeLayer)

  if (!layer) return null

  return (
    <div className="relative bg-yellow-50 border-2 border-yellow-500 rounded-lg shadow-lg min-w-40">
      <button
        type="button"
        onClick={() => removeLayer(id)}
        className="absolute -right-2 -top-2 w-6 h-6 flex items-center justify-center rounded-full bg-yellow-500 text-white text-xs font-bold shadow-md hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-yellow-300"
        aria-label="Remove flatten layer"
      >
        ×
      </button>

      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="bg-indigo-500! size-5! border-2! border-white!"
      />

      <div className="bg-yellow-500 text-white px-3 py-1.5 rounded-t-md text-sm font-semibold">
        Flatten
      </div>

      <div className="p-3 space-y-2">
        <div className="text-xs text-gray-600">
          <span className="font-medium">Input:</span> {formatShape(inputShape)}
        </div>
        <div className="text-xs text-gray-600">
          <span className="font-medium">Output:</span> {formatShape(layer.shapeOut)}
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
