import { create } from 'zustand'
import type { AnyLayer, GraphEdge, TensorShape } from '../types/graph'

interface GraphState {
  layers: Record<string, AnyLayer>
  edges: GraphEdge[]

  // Actions
  addLayer: (layer: AnyLayer) => void
  removeLayer: (id: string) => void
  updateLayerParams: (id: string, params: Record<string, any>) => void
  addEdge: (edge: GraphEdge) => void
  removeEdge: (id: string) => void
  recomputeShapes: () => void
  getInputShape: (layerId: string) => TensorShape | undefined
}

// Compute output shape for a layer given its input shape
function computeOutputShape(layer: AnyLayer, inputShape?: TensorShape): TensorShape {
  switch (layer.kind) {
    case 'Input':
      return { type: 'vector', size: layer.params.size }

    case 'Dense':
      if (inputShape?.type === 'vector') {
        return { type: 'vector', size: layer.params.units }
      }
      return { type: 'unknown' }

    case 'Output':
      if (inputShape?.type === 'vector') {
        return { type: 'vector', size: layer.params.classes }
      }
      return { type: 'unknown' }

    default:
      return { type: 'unknown' }
  }
}

export const useGraphStore = create<GraphState>((set, get) => ({
  layers: {},
  edges: [],

  addLayer: (layer) => {
    set((state) => ({
      layers: { ...state.layers, [layer.id]: layer }
    }))
    get().recomputeShapes()
  },

  removeLayer: (id) => set((state) => {
    const { [id]: removed, ...rest } = state.layers
    return {
      layers: rest,
      edges: state.edges.filter(e => e.source !== id && e.target !== id)
    }
  }),

  updateLayerParams: (id, params) => {
    set((state) => {
      const layer = state.layers[id]
      if (!layer) return state

      return {
        layers: {
          ...state.layers,
          [id]: { ...layer, params: { ...layer.params, ...params } }
        }
      }
    })
    get().recomputeShapes()
  },

  addEdge: (edge) => {
    set((state) => ({
      edges: [...state.edges, edge]
    }))
    get().recomputeShapes()
  },

  removeEdge: (id) => {
    set((state) => ({
      edges: state.edges.filter(e => e.id !== id)
    }))
    get().recomputeShapes()
  },

  getInputShape: (layerId: string): TensorShape | undefined => {
    const state = get()
    const incomingEdge = state.edges.find(e => e.target === layerId)

    if (!incomingEdge) return undefined

    const sourceLayer = state.layers[incomingEdge.source]
    return sourceLayer?.shapeOut
  },

  recomputeShapes: () => set((state) => {
    const updatedLayers = { ...state.layers }
    const visited = new Set<string>()

    // Topological sort helper
    function visit(layerId: string) {
      if (visited.has(layerId)) return
      visited.add(layerId)

      const layer = updatedLayers[layerId]
      if (!layer) return

      // Find incoming edge
      const incomingEdge = state.edges.find(e => e.target === layerId)

      // Visit source first if exists
      if (incomingEdge) {
        visit(incomingEdge.source)
      }

      // Get input shape
      const inputShape = incomingEdge
        ? updatedLayers[incomingEdge.source]?.shapeOut
        : undefined

      // Compute output shape
      const outputShape = computeOutputShape(layer, inputShape)
      updatedLayers[layerId] = { ...layer, shapeOut: outputShape }
    }

    // Visit all layers
    Object.keys(updatedLayers).forEach(visit)

    // Update edge labels with shapes
    const updatedEdges = state.edges.map(edge => {
      const sourceLayer = updatedLayers[edge.source]
      const label = sourceLayer?.shapeOut
        ? `${sourceLayer.shapeOut.type === 'vector' ? sourceLayer.shapeOut.size : '?'}`
        : undefined
      return { ...edge, label }
    })

    return {
      layers: updatedLayers,
      edges: updatedEdges
    }
  })
}))

// Convert graph to backend architecture format
export function graphToArchitecture(layers: Record<string, AnyLayer>, edges: GraphEdge[]) {
  // Build adjacency map for graph traversal
  const adjacency = new Map<string, string>()
  edges.forEach(edge => {
    adjacency.set(edge.target, edge.source)
  })

  // Find input layer (no incoming edges)
  const inputLayer = Object.values(layers).find(
    layer => layer.kind === 'Input' && !edges.some(e => e.target === layer.id)
  )

  if (!inputLayer || inputLayer.kind !== 'Input') {
    throw new Error('No input layer found')
  }

  const input_size = inputLayer.params.size
  const backendLayers: any[] = []

  // Traverse graph in topological order starting from input
  let currentId: string | undefined = inputLayer.id
  let prevSize = input_size

  while (currentId) {
    const layer = layers[currentId]
    if (!layer) break

    // Convert layer to backend format
    if (layer.kind === 'Dense') {
      const units = layer.params.units
      backendLayers.push({
        type: 'linear',
        in: prevSize,
        out: units,
      })

      // Add activation if not 'none'
      const activation = layer.params.activation
      if (activation && activation !== 'none') {
        backendLayers.push({ type: activation })
      }

      prevSize = units
    } else if (layer.kind === 'Output') {
      const classes = layer.params.classes
      backendLayers.push({
        type: 'linear',
        in: prevSize,
        out: classes,
      })

      // Add softmax activation
      if (layer.params.activation === 'softmax') {
        backendLayers.push({ type: 'softmax' })
      }

      prevSize = classes
    }

    // Move to next connected layer
    const nextEdge = edges.find(e => e.source === currentId)
    currentId = nextEdge?.target
  }

  return {
    input_size,
    layers: backendLayers,
  }
}
