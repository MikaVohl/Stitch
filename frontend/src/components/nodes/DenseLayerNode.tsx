import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useGraphStore } from '../../store/graphStore';
import type { DenseLayer } from '../../types/graph';
import { formatShape, calculateParams } from '../../types/graph';

export function DenseLayerNode({ id, data }: NodeProps) {
  const layer = useGraphStore(state => state.layers[id]) as DenseLayer | undefined;
  const inputShape = useGraphStore(state => state.getInputShape(id));
  const updateLayerParams = useGraphStore(state => state.updateLayerParams);

  if (!layer) return null;

  const paramCount = calculateParams(layer, inputShape);

  const activationColors: Record<string, string> = {
    relu: 'bg-blue-100 text-blue-700 border-blue-300',
    sigmoid: 'bg-purple-100 text-purple-700 border-purple-300',
    tanh: 'bg-green-100 text-green-700 border-green-300',
    softmax: 'bg-pink-100 text-pink-700 border-pink-300',
    none: 'bg-gray-100 text-gray-700 border-gray-300',
  };

  return (
    <div className="bg-blue-50 border-2 border-blue-500 rounded-lg shadow-lg min-w-[180px]">
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-blue-500 !w-3 !h-3 !border-2 !border-white"
      />

      {/* Header */}
      <div className="bg-blue-500 text-white px-3 py-1.5 rounded-t-md text-sm font-semibold">
        Dense Layer
      </div>

      {/* Body */}
      <div className="p-3 space-y-2">
        {/* Units input */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Units:</label>
          <input
            type="number"
            value={layer.params.units}
            onChange={(e) => updateLayerParams(id, { units: parseInt(e.target.value) || 1 })}
            className="w-16 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            min="1"
            max="512"
          />
        </div>

        {/* Activation chip */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Activation:</label>
          <select
            value={layer.params.activation}
            onChange={(e) => updateLayerParams(id, { activation: e.target.value })}
            className={`px-2 py-1 text-xs rounded border ${activationColors[layer.params.activation]} font-medium`}
          >
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
            <option value="none">None</option>
          </select>
        </div>

        {/* Output shape */}
        <div className="text-xs text-gray-600">
          <span className="font-medium">Shape:</span> {formatShape(layer.shapeOut)}
        </div>

        {/* Params count */}
        {paramCount > 0 && (
          <div className="text-xs text-gray-600">
            <span className="font-medium">Params:</span> {paramCount.toLocaleString()}
          </div>
        )}
      </div>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-blue-500 !w-3 !h-3 !border-2 !border-white"
      />
    </div>
  );
}
