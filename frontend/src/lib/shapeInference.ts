import type { AnyLayer, TensorShape } from '../types/graph';
import { toast } from 'sonner';

/**
 * Validates if a connection between two layers is valid based on shape compatibility
 */
export function validateConnection(
  sourceLayer: AnyLayer,
  targetLayer: AnyLayer
): { valid: boolean; error?: string } {
  const outputShape = sourceLayer.shapeOut;

  if (!outputShape || outputShape.type === 'unknown') {
    return {
      valid: false,
      error: 'Source layer has unknown output shape'
    };
  }

  // Input layers cannot receive connections
  if (targetLayer.kind === 'Input') {
    return {
      valid: false,
      error: 'Cannot connect to Input layer'
    };
  }

  // Dense and Output layers expect vector input
  if (targetLayer.kind === 'Dense' || targetLayer.kind === 'Output') {
    if (outputShape.type !== 'vector') {
      return {
        valid: false,
        error: `${targetLayer.kind} layer expects vector input, got ${outputShape.type}`
      };
    }
  }

  return { valid: true };
}

/**
 * Shows a toast notification for connection validation errors
 */
export function notifyConnectionError(error: string) {
  toast.error('Invalid Connection', {
    description: error,
    duration: 3000,
  });
}

/**
 * Checks if a layer already has an incoming connection
 */
export function hasIncomingConnection(layerId: string, edges: Array<{ source: string; target: string }>): boolean {
  return edges.some(edge => edge.target === layerId);
}
