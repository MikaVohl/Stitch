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
const HORIZONTAL_SPACING = 240
const VERTICAL_SPACING = 72
const SAMPLE_FACTOR = 8
const INPUT_SPACING = 32
const INPUT_X_SHIFT = -HORIZONTAL_SPACING * 0.75

type InputNeuronData = {
  activation: number
}

type NeuronNodeData = {
  layerIndex: number
  totalNeurons: number
  sampledIndex: number
  isOutput: boolean
}

interface NetworkVisualizationProps {
  layers: StoredLayer[]
  currentDrawing?: number[][]
}

function sanitizeLayers(layers: StoredLayer[]): StoredLayer[] {
  return layers.filter((layer) => {
    const type = layer.type.toLowerCase()
    return type === 'input' || type === 'linear' || type === 'dense' || type === 'output'
  })
}

function detectNeuronCount(layer: StoredLayer): number {
  if (typeof layer.out === 'number') return Math.max(1, layer.out)
  const sized = layer as Record<string, unknown>
  if (typeof sized.units === 'number') return Math.max(1, sized.units as number)
  if (typeof sized.size === 'number') return Math.max(1, sized.size as number)
  if (typeof layer.in === 'number') return Math.max(1, layer.in)
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
  drawing: number[][] | null
): { nodes: Node<InputNeuronData | NeuronNodeData>[]; edges: Edge[] } {
  const filtered = sanitizeLayers(layers)
  const nodes: Node<InputNeuronData | NeuronNodeData>[] = []
  const edges: Edge[] = []
  const edgeIds = new Set<string>()

  const { nodes: inputNodes, nodeIds: inputIds } = createInputNodes(drawing)
  nodes.push(...inputNodes)

  let previousNodeIds: string[] = inputIds
  let renderIndex = 1

  filtered.forEach((layer, idx) => {
    const layerType = layer.type.toLowerCase()
    if (layerType === 'input') {
      return
    }

    const isLast = idx === filtered.length - 1
    const neuronCount = detectNeuronCount(layer)
    const displayCount = isLast ? neuronCount : Math.max(1, Math.ceil(neuronCount / SAMPLE_FACTOR))
    const yOffset = ((displayCount - 1) * VERTICAL_SPACING) / 2

    const currentIds: string[] = []

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
        },
        draggable: false,
        selectable: false,
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      })
    }

    const isFirstHiddenLayer = renderIndex === 1
    const previousStep = isFirstHiddenLayer
      ? 1
      : Math.max(1, Math.ceil(previousNodeIds.length / 64))
    const currentStep = isFirstHiddenLayer
      ? 1
      : Math.max(1, Math.ceil(currentIds.length / 32))

    for (let s = 0; s < previousNodeIds.length; s += previousStep) {
      const sourceId = previousNodeIds[s]
      if (!sourceId) continue
      for (let t = 0; t < currentIds.length; t += currentStep) {
        const targetId = currentIds[t]
        if (!targetId) continue

        const edgeId = `edge-${sourceId}-${targetId}`
        if (edgeIds.has(edgeId)) continue

        edgeIds.add(edgeId)
        edges.push({
          id: edgeId,
          source: sourceId,
          target: targetId,
          type: 'straight',
          animated: false,
          style: { stroke: '#1f293780', strokeWidth: 1 },
        })
      }
    }

    const lastSource = previousNodeIds[previousNodeIds.length - 1]
    const lastTarget = currentIds[currentIds.length - 1]
    if (lastSource && lastTarget) {
      const edgeId = `edge-${lastSource}-${lastTarget}`
      if (!edgeIds.has(edgeId)) {
        edgeIds.add(edgeId)
        edges.push({
          id: edgeId,
          source: lastSource,
          target: lastTarget,
          type: 'straight',
          animated: false,
          style: { stroke: '#1f293780', strokeWidth: 1 },
        })
      }
    }

    previousNodeIds = currentIds
    renderIndex += 1
  })

  return { nodes, edges }
}

function NeuronNode({ data }: NodeProps<any>) {
  return (
    <div className="relative flex h-12 w-12 items-center justify-center rounded-full border-2 border-slate-300 bg-white shadow-sm">
      {!data.isOutput && (
        <Handle
          type="source"
          position={Position.Right}
          className="!h-2.5 !w-2.5 !border-0 !bg-slate-400"
        />
      )}
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2.5 !w-2.5 !border-0 !bg-slate-400"
      />
    </div>
  )
}

function InputNeuronNode({ data }: NodeProps<any>) {
  const intensity = Math.round(data.activation * 255)
  const color = `rgb(${intensity}, ${intensity}, ${intensity})`

  return (
    <div
      className="flex h-[18px] w-[18px] items-center justify-center rounded-full border border-slate-300 shadow-sm"
      style={{ backgroundColor: color }}
    />
  )
}

const nodeTypes: NodeTypes = {
  neuron: NeuronNode as any,
  inputNeuron: InputNeuronNode as any,
}

export function NetworkVisualization({ layers, currentDrawing }: NetworkVisualizationProps) {
  const drawingMatrix = useMemo(() => clampedMatrix(currentDrawing), [currentDrawing])

  const { nodes, edges } = useMemo(
    () => buildNodesAndEdges(layers, drawingMatrix),
    [layers, drawingMatrix]
  )

  return (
    <div className="h-[520px] w-full rounded-xl border border-slate-200 bg-white">
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
        <MiniMap
          className="!bg-white !border !border-slate-200 !rounded-lg !shadow-sm"
          nodeBorderRadius={6}
          pannable
          zoomable
          nodeColor={(node) =>
            node.type === 'inputNeuron' ? '#cbd5f530' : '#94a3b8'
          }
        />
        <Controls showInteractive={false} className="rounded-lg border border-slate-200 bg-white shadow-sm" />
        <Panel position="top-right" className="rounded-lg border border-slate-200 bg-white/95 p-3 text-xs text-slate-600 shadow-sm backdrop-blur">
          <p className="font-semibold text-slate-900">Legend</p>
          <ul className="mt-2 space-y-1">
            <li className="flex items-center gap-2">
              <span className="inline-flex h-3 w-3 rounded-full bg-slate-300" />
              <span>Input pixels</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-flex h-3 w-3 rounded-full border border-slate-400" />
              <span>Sampled hidden/output neurons</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-flex h-px w-6 bg-slate-500/60" />
              <span>Connections (thinned for clarity)</span>
            </li>
          </ul>
        </Panel>
      </ReactFlow>
    </div>
  )
}
