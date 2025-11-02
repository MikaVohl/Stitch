import { useMemo } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Panel,
  type Node,
  type Edge,
  type NodeTypes,
  type NodeProps,
  Position,
  Handle,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { StoredLayer } from '@/hooks/useModels'

const MNIST_SIDE = 28
const HORIZONTAL_SPACING = 320
const VERTICAL_SPACING = 72
const SAMPLE_FACTOR = 8
const CONV_ROW_SPACING = 48
const CONV_COL_X_SHIFT = 28
const CONV_COL_Y_OFFSET = 12
const INPUT_SPACING = 36
const INPUT_X_SHIFT = -HORIZONTAL_SPACING * 2

type InputNeuronData = {
  activation: number
}

type NeuronNodeData = {
  layerIndex: number
  totalNeurons: number
  sampledIndex: number
  isOutput: boolean
  highlighted?: boolean
}

type ConvNeuronNodeData = {
  filters: number
  sampledFilters: number
  index: number
  columns: number
}

type OperationNodeData = {
  label: string
}

interface NetworkVisualizationProps {
  layers: StoredLayer[]
  currentDrawing?: number[][]
  activeOutput?: number | null
}

const DENSE_LAYER_TYPES = new Set(['linear', 'dense', 'output'])
const CONV_LAYER_TYPES = new Set(['conv2d'])
const OPERATION_LAYER_TYPES = new Set(['maxpool2d', 'flatten', 'dropout'])

function sanitizeLayers(layers: StoredLayer[]): StoredLayer[] {
  return layers.filter((layer) => {
    const type = layer.type.toLowerCase()
    return (
      type === 'input' ||
      DENSE_LAYER_TYPES.has(type) ||
      CONV_LAYER_TYPES.has(type) ||
      OPERATION_LAYER_TYPES.has(type)
    )
  })
}

function detectNeuronCount(layer: StoredLayer): number {
  if (typeof layer.out === 'number') return Math.max(1, layer.out)
  const sized = layer as Record<string, unknown>
  if (typeof sized.units === 'number') return Math.max(1, sized.units as number)
  if (typeof sized.size === 'number') return Math.max(1, sized.size as number)
  if (typeof sized.filters === 'number') return Math.max(1, sized.filters as number)
  if (typeof sized.out_channels === 'number') return Math.max(1, sized.out_channels as number)
  if (typeof layer.in === 'number') return Math.max(1, layer.in)
  return 1
}

function detectFilterCount(layer: StoredLayer): number {
  const sized = layer as Record<string, unknown>
  if (typeof sized.filters === 'number') return Math.max(1, sized.filters as number)
  if (typeof sized.out_channels === 'number') return Math.max(1, sized.out_channels as number)
  if (typeof sized.channels === 'number') return Math.max(1, sized.channels as number)
  if (typeof sized.depth === 'number') return Math.max(1, sized.depth as number)
  if (typeof layer.out === 'number') return Math.max(1, layer.out)
  return 1
}

function clampedMatrix(drawing?: number[][]): number[][] | null {
  if (!drawing || drawing.length !== MNIST_SIDE || drawing[0]?.length !== MNIST_SIDE) return null
  return drawing
}

function createInputNodes(drawing: number[][] | null): {
  nodes: Node<InputNeuronData>[]
  nodeIds: string[]
} {
  const nodes: Node<InputNeuronData>[] = []
  const nodeIds: string[] = []

  const gridWidth = (MNIST_SIDE - 1) * INPUT_SPACING
  const halfWidth = gridWidth / 2

  for (let row = 0; row < MNIST_SIDE; row++) {
    for (let col = 0; col < MNIST_SIDE; col++) {
      const id = `input-${row}-${col}`
      const activation = drawing?.[row]?.[col] ?? 0
      nodeIds.push(id)
      nodes.push({
        id,
        type: 'inputNeuron',
        position: {
          x: col * INPUT_SPACING - halfWidth + INPUT_X_SHIFT,
          y: row * INPUT_SPACING - halfWidth,
        },
        data: {
          activation: Math.min(1, Math.max(0, activation / 255)),
        },
        draggable: false,
        selectable: false,
        sourcePosition: Position.Right,
      })
    }
  }

  return { nodes, nodeIds }
}

function buildNodesAndEdges(
  layers: StoredLayer[],
  drawing: number[][] | null,
  activeOutput?: number | null
): { nodes: Node<InputNeuronData | NeuronNodeData | ConvNeuronNodeData | OperationNodeData>[]; edges: Edge[] } {
  const filtered = sanitizeLayers(layers)
  const nodes: Node<InputNeuronData | NeuronNodeData | ConvNeuronNodeData | OperationNodeData>[] = []
  const edges: Edge[] = []
  const edgeIds = new Set<string>()
  const convColumns = new Map<string, { column: number; totalColumns: number }>()

  const { nodes: inputNodes, nodeIds: inputIds } = createInputNodes(drawing)
  nodes.push(...inputNodes)

  const EDGE_STEP_FIRST_LAYER = 1
  const rightmostColumnIds = inputIds.filter((id) => id.endsWith(`-${MNIST_SIDE - 1}`))
  const initialEdgeSources = rightmostColumnIds.length > 0 ? rightmostColumnIds : inputIds

  let previousNodeIds: string[] = initialEdgeSources
  let renderIndex = 1

  filtered.forEach((layer, idx) => {
    const layerType = layer.type.toLowerCase()
    if (layerType === 'input') {
      return
    }

    const isLast = idx === filtered.length - 1
    const currentIds: string[] = []

    if (CONV_LAYER_TYPES.has(layerType)) {
      const filters = detectFilterCount(layer)
      const baseSample = Math.max(1, Math.ceil(filters / SAMPLE_FACTOR))
      const sampleCount = Math.max(1, baseSample * baseSample)
      const columns = Math.max(1, Math.ceil(Math.sqrt(sampleCount)))
      const rows = Math.max(1, Math.ceil(sampleCount / columns))
      const yOffset = ((rows - 1) * CONV_ROW_SPACING) / 2
      const baseX = renderIndex * HORIZONTAL_SPACING

      for (let i = 0; i < sampleCount; i++) {
        const row = Math.floor(i / columns)
        const col = i % columns
        const nodeId = `layer-${renderIndex}-conv-${i}`
        currentIds.push(nodeId)

        nodes.push({
          id: nodeId,
          type: 'convNeuron',
          position: {
            x: baseX + col * CONV_COL_X_SHIFT,
            y: row * CONV_ROW_SPACING - yOffset + col * CONV_COL_Y_OFFSET,
          },
          data: {
            filters,
            sampledFilters: sampleCount,
            index: i,
            columns,
          },
          draggable: false,
          selectable: false,
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        })
        convColumns.set(nodeId, { column: col, totalColumns: columns })
      }
    } else if (DENSE_LAYER_TYPES.has(layerType)) {
      const neuronCount = detectNeuronCount(layer)
      const displayCount = isLast
        ? neuronCount
        : Math.max(1, Math.ceil(neuronCount / SAMPLE_FACTOR))
      const yOffset = ((displayCount - 1) * VERTICAL_SPACING) / 2

      for (let i = 0; i < displayCount; i++) {
        const nodeId = `layer-${renderIndex}-${i}`
        currentIds.push(nodeId)

        nodes.push({
          id: nodeId,
          type: 'neuron',
          position: {
            x: renderIndex * HORIZONTAL_SPACING,
            y: i * VERTICAL_SPACING - yOffset,
          },
          data: {
            layerIndex: renderIndex,
            totalNeurons: neuronCount,
            sampledIndex: isLast ? i : Math.min(i * SAMPLE_FACTOR, neuronCount - 1),
            isOutput: isLast,
            highlighted: isLast && typeof activeOutput === 'number' && activeOutput === i,
          },
          draggable: false,
          selectable: false,
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        })
      }
    } else if (OPERATION_LAYER_TYPES.has(layerType)) {
      const nodeId = `layer-${renderIndex}-operation`
      currentIds.push(nodeId)

      const label = layerType === 'maxpool2d'
        ? 'Max Pool'
        : layerType.charAt(0).toUpperCase() + layerType.slice(1)

      nodes.push({
        id: nodeId,
        type: 'operation',
        position: {
          x: renderIndex * HORIZONTAL_SPACING,
          y: 0,
        },
        data: {
          label,
        },
        draggable: false,
        selectable: false,
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      })
    } else {
      renderIndex += 1
      return
    }

    const selectConvBoundary = (ids: string[]): string[] => {
      const convCandidates = ids.filter((id) => convColumns.has(id))
      if (convCandidates.length === 0) return []
      const boundary = convCandidates.filter((id) => {
        const info = convColumns.get(id)!
        return info.column === 0 || info.column === info.totalColumns - 1
      })
      return Array.from(new Set(boundary))
    }

    const selectWithStep = (ids: string[], step: number): string[] => {
      if (ids.length <= 1) return ids
      const chosen: string[] = []
      for (let i = 0; i < ids.length; i += step) {
        chosen.push(ids[i])
      }
      const last = ids[ids.length - 1]
      if (last) {
        chosen.push(last)
      }
      return Array.from(new Set(chosen))
    }

    const isFirstHiddenLayer = renderIndex === 1
    const previousIsConv = previousNodeIds.some((id) => convColumns.has(id))
    const currentIsConv = currentIds.some((id) => convColumns.has(id))

    const previousStep = isFirstHiddenLayer
      ? EDGE_STEP_FIRST_LAYER
      : Math.max(1, Math.ceil(previousNodeIds.length / 64))
    const currentStep = isFirstHiddenLayer
      ? 1
      : Math.max(1, Math.ceil(currentIds.length / 32))

    const sourceSelection = previousIsConv
      ? selectConvBoundary(previousNodeIds)
      : selectWithStep(previousNodeIds, previousStep)
    const targetSelection = currentIsConv
      ? selectConvBoundary(currentIds)
      : selectWithStep(currentIds, currentStep)

    sourceSelection.forEach((sourceId) => {
      targetSelection.forEach((targetId) => {
        if (!sourceId || !targetId) return
        const edgeId = `edge-${sourceId}-${targetId}`
        if (edgeIds.has(edgeId)) return
        edgeIds.add(edgeId)
        edges.push({
          id: edgeId,
          source: sourceId,
          target: targetId,
          type: 'straight',
          animated: false,
          style: { stroke: '#1f293780', strokeWidth: 1 },
        })
      })
    })

    previousNodeIds = currentIds
    renderIndex += 1
  })

  return { nodes, edges }
}

function NeuronNode(props: NodeProps) {
  const data = props.data as NeuronNodeData
  const highlighted = !!data.highlighted

  const baseClasses =
    'relative flex h-12 w-12 items-center justify-center rounded-full border-2 shadow-sm transition-colors'
  const visualClasses = highlighted
    ? 'border-blue-500 bg-blue-100 text-blue-900 ring-2 ring-blue-200'
    : 'border-slate-300 bg-white text-slate-700'

  return (
    <div className={`${baseClasses} ${visualClasses}`}>
      {!data.isOutput && (
        <Handle
          type="source"
          position={Position.Right}
          className={`h-2.5! w-2.5! border-0! ${highlighted ? 'bg-blue-400!' : 'bg-orange-300!'}`}
        />
      )}
      <Handle
        type="target"
        position={Position.Left}
        className={`h-2.5! w-2.5! border-0! ${highlighted ? 'bg-blue-400!' : 'bg-orange-300!'}`}
      />
      {data.isOutput && (
        <span className="text-sm font-semibold">
          {Math.min(9, Math.max(0, data.sampledIndex)).toString()}
        </span>
      )}
    </div>
  )
}

function OperationNode(props: NodeProps) {
  const data = props.data as OperationNodeData

  return (
    <div className="flex h-16 w-32 flex-col items-center justify-center rounded-xl border-2 border-slate-300 bg-slate-100 text-sm font-semibold text-slate-700 shadow-sm">
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2.5 !w-2.5 !border-0 !bg-slate-400"
      />
      <span className="uppercase text-lg font-bold tracking-wide">{data.label}</span>
      <Handle
        type="source"
        position={Position.Right}
        className="!h-2.5 !w-2.5 !border-0 !bg-slate-400"
      />
    </div>
  )
}

function InputNeuronNode(props: NodeProps) {
  const intensity = Math.round((props.data as InputNeuronData).activation * 255)
  const color = `rgb(${intensity}, ${intensity}, ${intensity})`

  return (
    <div
      className="flex h-[32px] w-[32px] items-center justify-center rounded-full border border-slate-300 shadow-sm"
      style={{ backgroundColor: color }}
    >
      <Handle
        type="source"
        position={Position.Right}
        className="!h-2.5 !w-2.5 !border-0 !bg-slate-400"
      />
    </div>
  )
}

function ConvNeuronNode(props: NodeProps) {
  return (
    <div className="relative flex h-12 w-12 items-center justify-center rounded-full border-2 border-indigo-400 bg-white shadow-sm">
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2.5 !w-2.5 !border-0 !bg-indigo-400"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!h-2.5 !w-2.5 !border-0 !bg-indigo-400"
      />
    </div>
  )
}

const nodeTypes: NodeTypes = {
  neuron: NeuronNode,
  inputNeuron: InputNeuronNode,
  convNeuron: ConvNeuronNode,
  operation: OperationNode,
}

export function NetworkVisualization({ layers, currentDrawing, activeOutput }: NetworkVisualizationProps) {
  const drawingMatrix = useMemo(() => clampedMatrix(currentDrawing), [currentDrawing])

  const { nodes, edges } = useMemo(
    () => buildNodesAndEdges(layers, drawingMatrix, activeOutput),
    [layers, drawingMatrix, activeOutput]
  )

  return (
    <div className="relative h-[520px] w-full rounded-xl border border-slate-200 bg-white">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.2}
        maxZoom={1.4}
        nodesDraggable={false}
        nodesConnectable={false}
        edgesFocusable={false}
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={24} color="#e2e8f0" />
        <Controls showInteractive={false} className="rounded-lg border border-slate-200 bg-white shadow-sm" />
        <Panel
          position="top-center"
          className="pointer-events-none w-full px-6 pt-4"
          style={{ width: '100%', left: '50%', transform: 'translate(-50%, 0)' }}
        >
          <div className="pointer-events-auto flex w-full flex-col rounded-xl border border-slate-200 bg-white/95 px-6 py-3 text-xs text-slate-600 shadow-sm backdrop-blur">
            <p className="text-center text-sm font-semibold text-slate-900">Legend</p>
            <p className="text-center text-xs font-light italic text-slate-400">Each node represents 8 neurons</p>

            <ul className="mt-2 grid grid-cols-2 gap-x-4 justify-items-center gap-y-1 sm:grid-cols-4 lg:grid-cols-5">
            <li className="flex items-center gap-2">
              <span className="inline-flex h-3 w-3 rounded-full bg-slate-900" />
              <span>Input pixels</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-flex h-3 w-3 rounded-full border border-orange-300" />
              <span>Dense neurons</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-flex h-3 w-3 rounded-full border border-indigo-400 bg-white" />
              <span>Convolutional Neurons</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-flex h-3 w-5 rounded bg-slate-200 border border-slate-400" />
              <span>Operations </span>
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-flex h-px w-6 bg-slate-500/60" />
              <span>Connections</span>
            </li>
            </ul>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  )
}
