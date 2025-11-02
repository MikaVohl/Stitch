import { useMemo, useEffect, useRef } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeProps,
  type NodeTypes,
  Position,
  Handle,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { StoredLayer } from '@/hooks/useModels'

const MNIST_SIDE = 28
const HORIZONTAL_SPACING = 220
const VERTICAL_SPACING = 70
const SAMPLE_FACTOR = 8
const OUTPUT_NEURONS = 10

type GridNodeData = {
  drawing: number[][] | null
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
    const layerType = layer.type.toLowerCase()
    return (
      layerType === 'input' ||
      layerType === 'linear' ||
      layerType === 'dense' ||
      layerType === 'output'
    )
  })
}

function detectNeuronCount(layer: StoredLayer): number {
  if (typeof layer.out === 'number') return Math.max(1, layer.out)

  const record = layer as Record<string, unknown>
  const maybeUnits = record.units
  if (typeof maybeUnits === 'number') return Math.max(1, maybeUnits)

  const maybeSize = record.size
  if (typeof maybeSize === 'number') return Math.max(1, maybeSize)

  if (typeof layer.in === 'number') return Math.max(1, layer.in)
  return 1
}

function clampedMatrix(drawing?: number[][]): number[][] | null {
  if (!drawing || drawing.length === 0 || drawing[0]?.length === 0) return null
  if (drawing.length !== MNIST_SIDE || drawing[0].length !== MNIST_SIDE) return null
  return drawing
}

function buildNodesAndEdges(
  layers: StoredLayer[],
  drawing: number[][] | null
): { nodes: Node<GridNodeData | NeuronNodeData>[]; edges: Edge[] } {
  const filtered = sanitizeLayers(layers)
  const nodes: Node<GridNodeData | NeuronNodeData>[] = []
  const edges: Edge[] = []

  // Input grid node always present so the input layer can be skipped later.
  nodes.push({
    id: 'grid-0',
    type: 'drawingGrid',
    position: { x: 0, y: 0 },
    data: { drawing },
    draggable: false,
    selectable: false,
    sourcePosition: Position.Right,
  })

  let previousNodeIds: string[] = ['grid-0']
  let renderIndex = 1

  filtered.forEach((layer, layerIndex) => {
    const layerType = layer.type.toLowerCase()

    // Skip explicit input layers; the drawing grid already represents them visually.
    if (layerType === 'input') {
      previousNodeIds = ['grid-0']
      return
    }

    const isOutputLayer = layerIndex === filtered.length - 1
    const neuronCount = detectNeuronCount(layer)
    const displayCount = isOutputLayer
      ? OUTPUT_NEURONS
      : Math.max(1, Math.ceil(neuronCount / SAMPLE_FACTOR))
    const yOffset = ((displayCount - 1) * VERTICAL_SPACING) / 2

    const currentNodeIds: string[] = []

    for (let i = 0; i < displayCount; i++) {
      const nodeId = `layer-${renderIndex}-neuron-${i}`
      currentNodeIds.push(nodeId)

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
          sampledIndex: isOutputLayer
            ? i
            : Math.min(i * SAMPLE_FACTOR, neuronCount - 1),
          isOutput: isOutputLayer,
        },
        draggable: false,
        selectable: false,
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      })
    }

    for (const sourceId of previousNodeIds) {
      for (const targetId of currentNodeIds) {
        edges.push({
          id: `edge-${sourceId}-${targetId}`,
          source: sourceId,
          target: targetId,
          type: 'straight',
          animated: false,
          style: { stroke: '#93c5fd', strokeWidth: 1 },
        })
      }
    }

    previousNodeIds = currentNodeIds
    renderIndex += 1
  })

  return { nodes, edges }
}

function DrawingGridNode({ data }: NodeProps<GridNodeData>) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const size = canvas.width
    ctx.clearRect(0, 0, size, size)

    ctx.fillStyle = '#f8fafc'
    ctx.fillRect(0, 0, size, size)

    if (!data.drawing) {
      ctx.strokeStyle = '#93c5fd'
      ctx.lineWidth = 2
      ctx.strokeRect(0, 0, size, size)
      return
    }

    const cellSize = size / MNIST_SIDE
    const gap = Math.max(0.6, cellSize * 0.1)
    const effectiveCell = cellSize - gap

    for (let row = 0; row < MNIST_SIDE; row++) {
      for (let col = 0; col < MNIST_SIDE; col++) {
        const raw = data.drawing[row]?.[col] ?? 0
        const clamped = Math.min(1, Math.max(0, raw))
        const intensity = Math.round(clamped * 255)
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`
        ctx.fillRect(
          col * cellSize + gap / 2,
          row * cellSize + gap / 2,
          effectiveCell,
          effectiveCell
        )
      }
    }

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= MNIST_SIDE; i++) {
      const offset = i * cellSize
      ctx.beginPath()
      ctx.moveTo(offset, 0)
      ctx.lineTo(offset, size)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, offset)
      ctx.lineTo(size, offset)
      ctx.stroke()
    }

    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.strokeRect(0, 0, size, size)
  }, [data.drawing])

  return (
    <div className="rounded-xl border border-blue-200 bg-white p-3 shadow-sm">
      <canvas
        ref={canvasRef}
        width={168}
        height={168}
        className="h-44 w-44 rounded-md border border-blue-100"
      />
      <Handle type="source" position={Position.Right} className="!bg-blue-400" />
    </div>
  )
}

function NeuronNode({ data }: NodeProps<NeuronNodeData>) {
  return (
    <div className="relative flex h-12 w-12 items-center justify-center rounded-full border-2 border-blue-200 bg-blue-50 shadow-sm">
      {!data.isOutput && (
        <Handle
          type="source"
          position={Position.Right}
          className="!h-2.5 !w-2.5 !border-0 !bg-blue-300"
        />
      )}
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2.5 !w-2.5 !border-0 !bg-blue-300"
      />
    </div>
  )
}

const nodeTypes: NodeTypes = {
  drawingGrid: DrawingGridNode,
  neuron: NeuronNode,
}

export function NetworkVisualization({
  layers,
  currentDrawing,
}: NetworkVisualizationProps) {
  const drawingMatrix = useMemo(() => clampedMatrix(currentDrawing), [currentDrawing])

  const { nodes, edges } = useMemo(
    () => buildNodesAndEdges(layers, drawingMatrix),
    [layers, drawingMatrix]
  )

  return (
    <div className="relative h-[460px] w-full rounded-lg border border-gray-200 bg-white">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.25 }}
        minZoom={0.2}
        maxZoom={1.4}
        defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        nodesDraggable={false}
        nodesConnectable={false}
        edgesFocusable={false}
        edgesUpdatable={false}
        panOnScroll
        zoomOnScroll
        zoomOnPinch
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={20} color="#e5e7eb" />
        <Controls showInteractive={false} />
      </ReactFlow>

      {nodes.length === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-slate-500">
          No layers available to visualize.
        </div>
      )}
    </div>
  )
}
