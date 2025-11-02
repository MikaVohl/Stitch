import { useMemo } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  type Node,
  type Edge,
  type NodeTypes,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { AnyLayer, GraphEdge, LayerKind } from '@/types/graph'
import { formatShape } from '@/types/graph'

type ChangeType = 'added' | 'removed' | 'modified' | 'unchanged'

interface SchemaProposalNodeData {
  layer: AnyLayer
  changeType: ChangeType
}

type SchemaProposalNode = Node<SchemaProposalNodeData, 'proposal'>

const layerTitles: Record<LayerKind, string> = {
  Input: 'Input Layer',
  Dense: 'Dense Layer',
  Convolution: 'Convolution Layer',
  Pooling: 'Pooling Layer',
  Flatten: 'Flatten Layer',
  Dropout: 'Dropout Layer',
  Output: 'Output Layer',
}

const headerColors: Record<LayerKind, string> = {
  Input: 'bg-red-500',
  Dense: 'bg-blue-500',
  Convolution: 'bg-indigo-500',
  Pooling: 'bg-teal-500',
  Flatten: 'bg-amber-500',
  Dropout: 'bg-rose-500',
  Output: 'bg-emerald-500',
}

const changeBadgeStyles: Record<ChangeType, string> = {
  added: 'bg-green-100 text-green-700 border-green-200',
  removed: 'bg-red-100 text-red-700 border-red-200',
  modified: 'bg-yellow-100 text-yellow-700 border-yellow-200',
  unchanged: 'bg-gray-100 text-gray-600 border-gray-200',
}

function describeLayer(layer: AnyLayer): Array<{ label: string; value: string }> {
  switch (layer.kind) {
    case 'Input':
      return [
        { label: 'Size', value: `${layer.params.size}` },
        layer.params.channels && layer.params.height && layer.params.width
          ? {
            label: 'Shape',
            value: `${layer.params.channels}×${layer.params.height}×${layer.params.width}`,
          }
          : { label: 'Shape', value: 'Vector' },
      ]
    case 'Dense':
      return [
        { label: 'Units', value: `${layer.params.units}` },
        { label: 'Activation', value: layer.params.activation },
      ]
    case 'Convolution':
      return [
        { label: 'Filters', value: `${layer.params.filters}` },
        { label: 'Kernel', value: `${layer.params.kernel}×${layer.params.kernel}` },
        { label: 'Stride', value: `${layer.params.stride}` },
        { label: 'Padding', value: layer.params.padding },
        { label: 'Activation', value: layer.params.activation },
      ]
    case 'Pooling':
      return [
        { label: 'Type', value: layer.params.type },
        { label: 'Pool', value: `${layer.params.pool_size}` },
        { label: 'Stride', value: `${layer.params.stride}` },
      ]
    case 'Flatten':
      return [{ label: 'Operation', value: 'Flatten tensor' }]
    case 'Dropout':
      return [{ label: 'Rate', value: `${layer.params.rate}` }]
    case 'Output':
      return [
        { label: 'Classes', value: `${layer.params.classes}` },
        { label: 'Activation', value: layer.params.activation },
      ]
    default:
      return []
  }
}

function SchemaProposalNodeView({ data }: NodeProps<SchemaProposalNode>) {
  const layer = data?.layer
  if (!layer) return null

  const changeType = data.changeType ?? 'unchanged'

  return (
    <>
      <Handle type="target" position={Position.Left} />
      <div className="relative bg-white border border-gray-200 rounded-xl shadow-md overflow-hidden min-w-[180px]">
        <div className={`px-3 py-1.5 text-xs font-semibold text-white ${headerColors[layer.kind]}`}>
          {layerTitles[layer.kind]}
        </div>

        <div className="absolute top-2 right-2">
          <span className={`px-2 py-0.5 text-[10px] font-semibold rounded-full border ${changeBadgeStyles[changeType]}`}>
            {changeType.charAt(0).toUpperCase() + changeType.slice(1)}
          </span>
        </div>

        <div className="p-3 space-y-1.5 text-xs text-gray-700">
          {describeLayer(layer).map((item) => (
            <div key={item.label}>
              <span className="font-medium">{item.label}:</span> {item.value}
            </div>
          ))}
          <div className="pt-1 text-gray-600">
            <span className="font-medium">Output:</span> {formatShape(layer.shapeOut)}
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </>
  )
}

const nodeTypes: NodeTypes = {
  proposal: SchemaProposalNodeView,
}

const layerKindToNodeType: Record<LayerKind, keyof typeof nodeTypes> = {
  Input: 'proposal',
  Dense: 'proposal',
  Convolution: 'proposal',
  Pooling: 'proposal',
  Flatten: 'proposal',
  Dropout: 'proposal',
  Output: 'proposal',
}

interface SchemaProposalPreviewProps {
  currentLayers: Record<string, AnyLayer>
  currentEdges: GraphEdge[]
  proposedLayers: Record<string, AnyLayer>
  proposedEdges: GraphEdge[]
  onApply: () => void
  onReject: () => void
}

export function SchemaProposalPreview({
  currentLayers,
  currentEdges,
  proposedLayers,
  proposedEdges,
  onApply,
  onReject,
}: SchemaProposalPreviewProps) {
  const { currentNodes, proposedNodes } = useMemo(() => {
    // Detect layer changes
    const layerChanges = new Map<string, ChangeType>()

    // Check for removed and modified layers
    Object.keys(currentLayers).forEach((id) => {
      if (!proposedLayers[id]) {
        layerChanges.set(id, 'removed')
      } else {
        // Check if layer was modified
        const current = currentLayers[id]
        const proposed = proposedLayers[id]
        const isModified =
          current.kind !== proposed.kind ||
          JSON.stringify(current.params) !== JSON.stringify(proposed.params)

        layerChanges.set(id, isModified ? 'modified' : 'unchanged')
      }
    })

    // Check for added layers
    Object.keys(proposedLayers).forEach((id) => {
      if (!currentLayers[id]) {
        layerChanges.set(id, 'added')
      }
    })

    const getNodeColor = (changeType: ChangeType): string => {
      switch (changeType) {
        case 'added':
          return '#10b981' // green
        case 'removed':
          return '#ef4444' // red
        case 'modified':
          return '#f59e0b' // yellow
        default:
          return '#6b7280' // gray
      }
    }

    const createNodes = (layers: Record<string, AnyLayer>): Node[] => {
      return Object.values(layers).map((layer, index) => {
        const changeType = layerChanges.get(layer.id) || 'unchanged'
        const borderColor = getNodeColor(changeType)
        const nodeType = layerKindToNodeType[layer.kind]

        return {
          id: layer.id,
          type: nodeType,
          position: layer.position ?? { x: index * 300 + 50, y: 250 },
          data: {
            layer,
            changeType,
          },
          draggable: false,
          style: {
            background: 'transparent',
            border: `3px solid ${borderColor}`,
            borderRadius: '8px',
            padding: 0,
            boxShadow: `0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 0 0 3px ${borderColor}33`,
          },
        }
      })
    }

    const currentNodes = createNodes(currentLayers)
    const proposedNodes = createNodes(proposedLayers)

    return { currentNodes, proposedNodes }
  }, [currentLayers, proposedLayers])

  const { currentReactFlowEdges, proposedReactFlowEdges } = useMemo(() => {
    const sanitizeHandle = (handle?: string | null): string | undefined => {
      if (handle == null) return undefined
      const trimmed = String(handle).trim()
      if (!trimmed || trimmed === 'null' || trimmed === 'undefined') return undefined
      return trimmed
    }

    const createEdges = (edges: GraphEdge[]): Edge[] => {
      return edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: sanitizeHandle(edge.sourceHandle),
        targetHandle: sanitizeHandle(edge.targetHandle),
        label: edge.label,
        animated: true,
      }))
    }

    return {
      currentReactFlowEdges: createEdges(currentEdges),
      proposedReactFlowEdges: createEdges(proposedEdges),
    }
  }, [currentEdges, proposedEdges])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-6 sm:p-10">
      <div className="bg-white rounded-2xl shadow-2xl w-full h-full max-w-6xl max-h-[90vh] flex flex-col overflow-hidden border border-slate-200">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900">Architecture Proposal</h2>
          <p className="text-sm text-gray-600 mt-1">
            Review the proposed changes to your neural network architecture
          </p>
          <div className="flex gap-4 mt-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span>Added</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-yellow-500 rounded"></div>
              <span>Modified</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span>Removed</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-gray-500 rounded"></div>
              <span>Unchanged</span>
            </div>
          </div>
        </div>

        {/* Side by side comparison */}
        <div className="flex-1 flex overflow-hidden">
          {/* Current */}
          <div className="flex-1 flex flex-col border-r border-gray-200">
            <div className="p-4 bg-gray-50 border-b border-gray-200">
              <h3 className="font-semibold text-gray-900">Current Architecture</h3>
            </div>
            <div className="flex-1 min-h-[24rem]">
              <ReactFlow
                nodes={currentNodes}
                edges={currentReactFlowEdges}
                nodeTypes={nodeTypes}
                fitView
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable={false}
              >
                <Background />
                <Controls showInteractive={false} />
              </ReactFlow>
            </div>
          </div>

          {/* Proposed */}
          <div className="flex-1 flex flex-col">
            <div className="p-4 bg-blue-50 border-b border-blue-200">
              <h3 className="font-semibold text-blue-900">Proposed Architecture</h3>
            </div>
            <div className="flex-1 min-h-[24rem]">
              <ReactFlow
                nodes={proposedNodes}
                edges={proposedReactFlowEdges}
                nodeTypes={nodeTypes}
                fitView
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable={false}
              >
                <Background />
                <Controls showInteractive={false} />
              </ReactFlow>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="p-6 border-t border-gray-200 flex justify-end gap-4">
          <button
            onClick={onReject}
            className="px-6 py-2.5 bg-gray-200 hover:bg-gray-300 text-gray-900 font-semibold rounded-lg transition-colors cursor-pointer"
          >
            Reject
          </button>
          <button
            onClick={onApply}
            className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors cursor-pointer"
          >
            Apply Changes
          </button>
        </div>
      </div>
    </div>
  )
}
