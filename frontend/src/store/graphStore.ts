import { create } from 'zustand';
import type { AnyLayer, GraphEdge, TensorShape } from '../types/graph';

interface GraphState {
  layers: Record<string, AnyLayer>;
  edges: GraphEdge[];

  // Actions
  addLayer: (layer: AnyLayer) => void;
  removeLayer: (id: string) => void;
  updateLayerParams: (id: string, params: Record<string, any>) => void;
  addEdge: (edge: GraphEdge) => void;
  removeEdge: (id: string) => void;
  recomputeShapes: () => void;
  getInputShape: (layerId: string) => TensorShape | undefined;
}

// Compute output shape for a layer given its input shape
function computeOutputShape(layer: AnyLayer, inputShape?: TensorShape): TensorShape {
  switch (layer.kind) {
    case 'Input':
      return { type: 'vector', size: layer.params.size };

    case 'Dense':
      if (inputShape?.type === 'vector') {
        return { type: 'vector', size: layer.params.units };
      }
      return { type: 'unknown' };

    case 'Output':
      if (inputShape?.type === 'vector') {
        return { type: 'vector', size: layer.params.classes };
      }
      return { type: 'unknown' };

    default:
      return { type: 'unknown' };
  }
}

export const useGraphStore = create<GraphState>((set, get) => ({
  layers: {},
  edges: [],

  addLayer: (layer) => {
    set((state) => ({
      layers: { ...state.layers, [layer.id]: layer }
    }));
    get().recomputeShapes();
  },

  removeLayer: (id) => set((state) => {
    const { [id]: removed, ...rest } = state.layers;
    return {
      layers: rest,
      edges: state.edges.filter(e => e.source !== id && e.target !== id)
    };
  }),

  updateLayerParams: (id, params) => {
    set((state) => {
      const layer = state.layers[id];
      if (!layer) return state;

      return {
        layers: {
          ...state.layers,
          [id]: { ...layer, params: { ...layer.params, ...params } }
        }
      };
    });
    get().recomputeShapes();
  },

  addEdge: (edge) => {
    set((state) => ({
      edges: [...state.edges, edge]
    }));
    get().recomputeShapes();
  },

  removeEdge: (id) => {
    set((state) => ({
      edges: state.edges.filter(e => e.id !== id)
    }));
    get().recomputeShapes();
  },

  getInputShape: (layerId: string): TensorShape | undefined => {
    const state = get();
    const incomingEdge = state.edges.find(e => e.target === layerId);

    if (!incomingEdge) return undefined;

    const sourceLayer = state.layers[incomingEdge.source];
    return sourceLayer?.shapeOut;
  },

  recomputeShapes: () => set((state) => {
    const updatedLayers = { ...state.layers };
    const visited = new Set<string>();

    // Topological sort helper
    function visit(layerId: string) {
      if (visited.has(layerId)) return;
      visited.add(layerId);

      const layer = updatedLayers[layerId];
      if (!layer) return;

      // Find incoming edge
      const incomingEdge = state.edges.find(e => e.target === layerId);

      // Visit source first if exists
      if (incomingEdge) {
        visit(incomingEdge.source);
      }

      // Get input shape
      const inputShape = incomingEdge
        ? updatedLayers[incomingEdge.source]?.shapeOut
        : undefined;

      // Compute output shape
      const outputShape = computeOutputShape(layer, inputShape);
      updatedLayers[layerId] = { ...layer, shapeOut: outputShape };
    }

    // Visit all layers
    Object.keys(updatedLayers).forEach(visit);

    // Update edge labels with shapes
    const updatedEdges = state.edges.map(edge => {
      const sourceLayer = updatedLayers[edge.source];
      const label = sourceLayer?.shapeOut
        ? `${sourceLayer.shapeOut.type === 'vector' ? sourceLayer.shapeOut.size : '?'}`
        : undefined;
      return { ...edge, label };
    });

    return {
      layers: updatedLayers,
      edges: updatedEdges
    };
  })
}));
