import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useGraphStore } from '../../store/graphStore';
import type { OutputLayer } from '../../types/graph';
import { formatShape } from '../../types/graph';

export function OutputLayerNode({ id }: NodeProps) {
  const layer = useGraphStore(state => state.layers[id]) as OutputLayer | undefined;

  if (!layer) return null;

  return (
    <div className="bg-green-50 border-2 border-green-500 rounded-lg shadow-lg min-w-40">
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="bg-indigo-500! size-5! border-2! border-white!"
      />

      <div className="bg-green-500 text-white px-3 py-1.5 rounded-t-md text-sm font-semibold">
        Output Layer
      </div>

      <div className="p-3 space-y-2">
        <div className="text-xs text-gray-600">
          <span className="font-medium">Classes:</span> {layer.params.classes}
        </div>

        <div className="text-xs text-gray-600">
          <span className="font-medium">Activation:</span>{' '}
          <span className="px-2 py-0.5 bg-pink-100 text-pink-700 rounded border border-pink-300 font-medium">
            {layer.params.activation}
          </span>
        </div>

        <div className="text-xs text-gray-600">
          <span className="font-medium">Shape:</span> {formatShape(layer.shapeOut)}
        </div>
      </div>
    </div>
  );
}
