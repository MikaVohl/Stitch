import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useGraphStore } from '../../store/graphStore'
import type { InputLayer } from '../../types/graph'
import { formatShape } from '../../types/graph'

export function InputLayerNode({ id }: NodeProps) {
  const layer = useGraphStore(state => state.layers[id]) as InputLayer | undefined

  if (!layer) return null

  return (
    <div className="bg-red-50 border-2 border-red-500 rounded-lg shadow-lg min-w-40">
      <div className="bg-red-500 text-white px-3 py-1.5 rounded-t-md text-sm font-semibold">
        Input Layer
      </div>

      <div className="p-3 space-y-2">
        <div className="text-xs text-gray-600">
          <span className="font-medium">Size:</span> {layer.params.size}
        </div>

        <div className="text-xs text-gray-600">
          <span className="font-medium">Shape:</span> {formatShape(layer.shapeOut)}
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
