// Shape types for tensor validation
export type TensorShape =
  | { type: 'vector'; size: number }
  | { type: 'unknown' };

export type LayerKind = 'Input' | 'Dense' | 'Output';

export type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'none';

// Base layer interface
export interface Layer {
  id: string;
  kind: LayerKind;
  params: Record<string, any>;
  shapeOut?: TensorShape;
  position?: {
    x: number;
    y: number;
  };
}

// Specific layer types
export interface InputLayer extends Layer {
  kind: 'Input';
  params: {
    size: number;
  };
}

export interface DenseLayer extends Layer {
  kind: 'Dense';
  params: {
    units: number;
    activation: ActivationType;
    use_bias?: boolean;
  };
}

export interface OutputLayer extends Layer {
  kind: 'Output';
  params: {
    classes: number;
    activation: 'softmax';
  };
}

export type AnyLayer = InputLayer | DenseLayer | OutputLayer;

// Edge with shape label
export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

// Helper to format shape as string
export function formatShape(shape?: TensorShape): string {
  if (!shape) return 'unknown';
  if (shape.type === 'vector') return `(${shape.size})`;
  return 'unknown';
}

// Calculate parameter count for a layer
export function calculateParams(layer: AnyLayer, inputShape?: TensorShape): number {
  if (layer.kind === 'Dense' && inputShape?.type === 'vector') {
    const weights = inputShape.size * layer.params.units;
    const useBias = layer.params.use_bias ?? true;
    const bias = useBias ? layer.params.units : 0;
    return weights + bias;
  }
  return 0;
}
