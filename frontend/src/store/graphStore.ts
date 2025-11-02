import { create } from 'zustand'
import type { AnyLayer, GraphEdge, TensorShape } from '../types/graph'
import { formatShape } from '../types/graph'
import type { TrainingRequest } from '@/api/types'

interface GraphState {
  layers: Record<string, AnyLayer>
  edges: GraphEdge[]

  // Actions
  addLayer: (layer: AnyLayer) => void
  removeLayer: (id: string) => void
  updateLayerParams: (id: string, params: Record<string, any>) => void
  updateLayerPosition: (id: string, position: { x: number; y: number }) => void
  addEdge: (edge: GraphEdge) => void
  removeEdge: (id: string) => void
  recomputeShapes: () => void
  getInputShape: (layerId: string) => TensorShape | undefined
  applyProposedSchema: (schema: { layers: Record<string, AnyLayer>, edges: GraphEdge[] }) => void
  clearGraph: () => void
  loadGraph: (layers: Record<string, AnyLayer>, edges: GraphEdge[]) => void
}

// Compute output shape for a layer given its input shape
function computeOutputShape(layer: AnyLayer, inputShape?: TensorShape): TensorShape {
  switch (layer.kind) {
    case 'Input': {
      const { channels, height, width, size } = layer.params
      if (channels && height && width) {
        return { type: 'image', channels, height, width }
      }
      return { type: 'vector', size }
    }

    case 'Dense':
      if (inputShape?.type === 'vector') {
        return { type: 'vector', size: layer.params.units }
      }
      return { type: 'unknown' }

    case 'Convolution': {
      let imageShape = inputShape
      if (imageShape?.type === 'vector') {
        imageShape = inferImageShapeFromSize(imageShape.size)
      }
      if (!imageShape || imageShape.type !== 'image') {
        return { type: 'unknown' }
      }

      const { kernel, stride, padding, filters } = layer.params
      const nextHeight = computeConvOutputDim(imageShape.height, kernel, stride, padding)
      const nextWidth = computeConvOutputDim(imageShape.width, kernel, stride, padding)
      return { type: 'image', channels: filters, height: nextHeight, width: nextWidth }
    }

    case 'Pooling': {
      let imageShape = inputShape
      if (imageShape?.type === 'vector') {
        imageShape = inferImageShapeFromSize(imageShape.size)
      }
      if (!imageShape || imageShape.type !== 'image') {
        return { type: 'unknown' }
      }

      const { pool_size, stride, padding } = layer.params
      const nextHeight = computePoolOutputDim(imageShape.height, pool_size, stride, padding)
      const nextWidth = computePoolOutputDim(imageShape.width, pool_size, stride, padding)
      return {
        type: 'image',
        channels: imageShape.channels,
        height: nextHeight,
        width: nextWidth,
      }
    }

    case 'Flatten':
      if (!inputShape) return { type: 'unknown' }
      if (inputShape.type === 'vector') return inputShape
      if (inputShape.type === 'image') {
        const flattened = inputShape.channels * inputShape.height * inputShape.width
        return { type: 'vector', size: flattened }
      }
      return { type: 'unknown' }

    case 'Dropout':
      return inputShape ?? { type: 'unknown' }

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
      layers: { ...state.layers, [layer.id]: layer } as Record<string, AnyLayer>
    }))
    get().recomputeShapes()
  },

  removeLayer: (id) => set((state) => {
    const { [id]: removed, ...rest } = state.layers
    return {
      layers: rest as Record<string, AnyLayer>,
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
          [id]: { ...layer, params: { ...layer.params, ...params } } as AnyLayer
        } as Record<string, AnyLayer>
      }
    })
    get().recomputeShapes()
  },

  updateLayerPosition: (id, position) => {
    set((state) => {
      const layer = state.layers[id]
      if (!layer) return state
      return {
        layers: {
          ...state.layers,
          [id]: { ...layer, position } as AnyLayer,
        } as Record<string, AnyLayer>,
      }
    })
  },

  addEdge: (edge) => {
    const normalizeHandle = (handle?: string | null) => handle ?? null

    set((state) => {
      const nextEdges = state.edges.filter((existing) => {
        const sameSourcePort =
          existing.source === edge.source &&
          normalizeHandle(existing.sourceHandle) === normalizeHandle(edge.sourceHandle)

        const sameTargetPort =
          existing.target === edge.target &&
          normalizeHandle(existing.targetHandle) === normalizeHandle(edge.targetHandle)

        return !sameSourcePort && !sameTargetPort
      })

      const updatedEdges = [...nextEdges, edge] as GraphEdge[]
      return {
        edges: updatedEdges
      }
    })
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
      const label = sourceLayer?.shapeOut ? formatShape(sourceLayer.shapeOut) : undefined
      return { ...edge, label }
    })

    return {
      layers: updatedLayers as Record<string, AnyLayer>,
      edges: updatedEdges
    }
  }),

  applyProposedSchema: (schema) => {
    get().clearGraph()
    Object.values(schema.layers).forEach((layer) => {
      get().addLayer({
        ...layer,
        params: { ...layer.params },
        position: layer.position ? { ...layer.position } : undefined,
      })
    })
    schema.edges.forEach((edge) => {
      get().addEdge({
        ...edge,
        sourceHandle: edge.sourceHandle ?? 'output',
        targetHandle: edge.targetHandle ?? 'input',
      })
    })
  },

  clearGraph: () => {
    set({
      layers: {},
      edges: []
    })
  },

  loadGraph: (layers, edges) => {
    const normalizedLayers = Object.values(layers).map((layer) => ({
      ...layer,
      params: { ...layer.params },
      position: layer.position ? { ...layer.position } : undefined,
    }))

    const normalizedEdges = edges.map((edge) => ({
      ...edge,
      sourceHandle: edge.sourceHandle ?? 'output',
      targetHandle: edge.targetHandle ?? 'input',
    })) as GraphEdge[]

    set({
      layers: {},
      edges: [],
    })

    normalizedLayers.forEach((layer) => {
      get().addLayer(layer)
    })

    normalizedEdges.forEach((edge) => {
      get().addEdge(edge)
    })
  }
}))

// Convert graph to backend architecture format
type VectorShapeState = Extract<TensorShape, { type: 'vector' }>
type ImageShapeState = Extract<TensorShape, { type: 'image' }>
type ShapeState = VectorShapeState | ImageShapeState

function inferImageShapeFromSize(size: number): ImageShapeState | undefined {
  const side = Math.round(Math.sqrt(size))
  if (side * side !== size) return undefined
  return { type: 'image', channels: 1, height: side, width: side }
}

function ensureFlattenLayer(
  state: ShapeState,
  layers: TrainingRequest['architecture']['layers']
): VectorShapeState {
  if (state.type === 'vector') return state
  const flattenedSize = state.channels * state.height * state.width
  layers.push({ type: 'flatten' })
  return { type: 'vector', size: flattenedSize }
}

function computeConvOutputDim(
  input: number,
  kernel: number,
  stride: number,
  padding: 'same' | 'valid'
): number {
  if (padding === 'same') {
    return Math.ceil(input / Math.max(1, stride))
  }
  const effectiveStride = Math.max(1, stride)
  return Math.max(1, Math.floor((input - kernel) / effectiveStride + 1))
}

function computePoolOutputDim(input: number, kernel: number, stride: number, padding: number): number {
  const effectiveStride = Math.max(1, stride)
  return Math.max(1, Math.floor((input + 2 * padding - kernel) / effectiveStride + 1))
}

export function graphToArchitecture(
  layers: Record<string, AnyLayer>,
  edges: GraphEdge[]
): TrainingRequest['architecture'] {
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

  const {
    size: inputLayerSize,
    channels: inputChannelsParam,
    height: inputHeightParam,
    width: inputWidthParam,
  } = inputLayer.params

  const explicitImageShape: ImageShapeState | undefined =
    inputChannelsParam && inputHeightParam && inputWidthParam
      ? {
          type: 'image',
          channels: inputChannelsParam,
          height: inputHeightParam,
          width: inputWidthParam,
        }
      : undefined

  const productSize =
    explicitImageShape?.channels && explicitImageShape.height && explicitImageShape.width
      ? explicitImageShape.channels * explicitImageShape.height * explicitImageShape.width
      : undefined

  const inputSize = productSize ?? inputLayerSize
  const initialImageShape = explicitImageShape ?? inferImageShapeFromSize(inputSize)
  let currentShape: ShapeState = initialImageShape ?? { type: 'vector', size: inputSize }
  const backendLayers: TrainingRequest['architecture']['layers'] = []

  // Traverse graph in topological order starting from input
  let currentId: string | undefined = inputLayer.id

  while (currentId) {
    const layer = layers[currentId]
    if (!layer) break

    if (layer.kind === 'Input') {
      // Skip adding the input layer itself; move to next
    } else if (layer.kind === 'Dense') {
      currentShape = ensureFlattenLayer(currentShape, backendLayers)
      const units = layer.params.units
      const inputDim = currentShape.size
      backendLayers.push({
        type: 'linear',
        in: inputDim,
        out: units,
      })

      const activation = layer.params.activation
      if (activation && activation !== 'none') {
        backendLayers.push({ type: activation })
      }

      currentShape = { type: 'vector', size: units }
    } else if (layer.kind === 'Convolution') {
      if (currentShape.type === 'vector') {
        const inferred = inferImageShapeFromSize(currentShape.size)
        if (!inferred) {
          throw new Error('Cannot infer image shape for convolution layer input')
        }
        currentShape = inferred
      }

      if (currentShape.type !== 'image') {
        throw new Error('Expected image input for convolution layer')
      }

      const { filters, kernel, stride, padding, activation } = layer.params
      const strideValue = Math.max(1, stride)
      const paddingMode: 'same' | 'valid' = padding === 'same' ? 'same' : 'valid'

      backendLayers.push({
        type: 'conv2d',
        in_channels: currentShape.channels,
        out_channels: filters,
        kernel_size: kernel,
        stride: strideValue,
        padding: paddingMode,
      })

      if (activation && activation !== 'none') {
        backendLayers.push({ type: activation })
      }

      const nextHeight = computeConvOutputDim(currentShape.height, kernel, strideValue, paddingMode)
      const nextWidth = computeConvOutputDim(currentShape.width, kernel, strideValue, paddingMode)

      currentShape = {
        type: 'image',
        channels: filters,
        height: nextHeight,
        width: nextWidth,
      }
    } else if (layer.kind === 'Pooling') {
      if (currentShape.type === 'vector') {
        const inferred = inferImageShapeFromSize(currentShape.size)
        if (!inferred) {
          throw new Error('Cannot infer image shape for pooling layer input')
        }
        currentShape = inferred
      }

      if (currentShape.type !== 'image') {
        throw new Error('Expected image input for pooling layer')
      }

      if (layer.params.type !== 'max') {
        throw new Error(`Unsupported pooling type: ${layer.params.type}`)
      }

      const poolSize = Math.max(1, layer.params.pool_size)
      const strideValue = Math.max(1, layer.params.stride ?? poolSize)
      const paddingValue = Math.max(0, layer.params.padding ?? 0)

      backendLayers.push({
        type: 'maxpool2d',
        kernel_size: poolSize,
        stride: strideValue,
        padding: paddingValue,
      })

      const nextHeight = computePoolOutputDim(currentShape.height, poolSize, strideValue, paddingValue)
      const nextWidth = computePoolOutputDim(currentShape.width, poolSize, strideValue, paddingValue)

      currentShape = {
        type: 'image',
        channels: currentShape.channels,
        height: nextHeight,
        width: nextWidth,
      }
    } else if (layer.kind === 'Flatten') {
      backendLayers.push({ type: 'flatten' })
      if (currentShape.type === 'image') {
        const flattenedSize = currentShape.channels * currentShape.height * currentShape.width
        currentShape = { type: 'vector', size: flattenedSize }
      }
    } else if (layer.kind === 'Dropout') {
      const rate = Math.min(1, Math.max(0, layer.params.rate))
      backendLayers.push({ type: 'dropout', p: rate })
    } else if (layer.kind === 'Output') {
      currentShape = ensureFlattenLayer(currentShape, backendLayers)

      const classes = layer.params.classes
      const inputDim = currentShape.size
      backendLayers.push({
        type: 'linear',
        in: inputDim,
        out: classes,
      })

      backendLayers.push({ type: 'softmax' })

      currentShape = { type: 'vector', size: classes }
    }

    // Move to next connected layer
    const nextEdge = edges.find(e => e.source === currentId)
    currentId = nextEdge?.target
  }

  const architecture: TrainingRequest['architecture'] = {
    input_size: inputSize,
    layers: backendLayers,
  }

  if (initialImageShape) {
    architecture.input_channels = initialImageShape.channels
    architecture.input_height = initialImageShape.height
    architecture.input_width = initialImageShape.width
  }

  return architecture
}
