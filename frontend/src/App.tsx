import { useCallback, useEffect, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type Connection,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { toast, Toaster } from "sonner"
import { useGraphStore } from './store/graphStore';
import { InputLayerNode } from './components/nodes/InputLayerNode';
import { DenseLayerNode } from './components/nodes/DenseLayerNode';
import { OutputLayerNode } from './components/nodes/OutputLayerNode';
import { HyperparamsPanel } from './components/HyperparamsPanel';
import { validateConnection, notifyConnectionError, hasIncomingConnection } from './lib/shapeInference';

const nodeTypes: NodeTypes = {
  input: InputLayerNode,
  dense: DenseLayerNode,
  output: OutputLayerNode,
};

export default function App() {
  const { layers, edges, addLayer, addEdge, removeEdge } = useGraphStore();

  // Convert store state to ReactFlow format with auto-layout
  const reactFlowNodes = useMemo((): Node[] => {
    const layerArray = Object.values(layers);
    const HORIZONTAL_SPACING = 300;
    const VERTICAL_CENTER = 250;

    return layerArray.map((layer, index) => ({
      id: layer.id,
      type: layer.kind.toLowerCase(),
      position: { x: index * HORIZONTAL_SPACING + 50, y: VERTICAL_CENTER },
      data: {},
    }));
  }, [layers]);

  const reactFlowEdges = useMemo((): Edge[] => {
    return edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      animated: true,
    }));
  }, [edges]);


  // Handle new connections with validation
  const onConnect = useCallback(
    (connection: Connection) => {
      const { source, target } = connection;
      if (!source || !target) return;

      const sourceLayer = layers[source];
      const targetLayer = layers[target];

      if (!sourceLayer || !targetLayer) return;

      // Check if target already has an incoming connection
      if (hasIncomingConnection(target, edges)) {
        notifyConnectionError('Layer already has an incoming connection');
        return;
      }

      // Validate shape compatibility
      const validation = validateConnection(sourceLayer, targetLayer);

      if (!validation.valid) {
        notifyConnectionError(validation.error || 'Invalid connection');
        return;
      }

      // Add the edge
      const edgeId = `${source}-${target}`;
      addEdge({ id: edgeId, source, target });
    },
    [layers, edges, addEdge]
  );

  const onEdgesChange = useCallback(
    (changes: any[]) => {
      changes.forEach(change => {
        if (change.type === 'remove') {
          removeEdge(change.id);
        }
      });
    },
    [removeEdge]
  );


  // Add sample layers on mount with positions
  useEffect(() => {
    addLayer({
      id: 'input-1',
      kind: 'Input',
      params: { size: 784 },
    });

    addLayer({
      id: 'dense-1',
      kind: 'Dense',
      params: { units: 128, activation: 'relu' }
    });

    addLayer({
      id: 'dense-2',
      kind: 'Dense',
      params: { units: 64, activation: 'relu' }
    });

    addLayer({
      id: 'output-1',
      kind: 'Output',
      params: { classes: 10, activation: 'softmax' },
    });

    // Connect the layers
    addEdge({ id: 'input-1-dense-1', source: 'input-1', target: 'dense-1' });
    addEdge({ id: 'dense-1-dense-2', source: 'dense-1', target: 'dense-2' });
    addEdge({ id: 'dense-2-output-1', source: 'dense-2', target: 'output-1' });
  }, [addLayer, addEdge]);

  const handleRun = useCallback(() => {
    toast.success('Training started!', {
      description: 'Mock inference - backend not connected yet',
    });
  }, []);

  return (
    <>
      <Toaster position="top-right" />
      <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
        {/* Hyperparameters Panel */}
        <HyperparamsPanel />

        {/* Floating Train Button */}
        <button
          onClick={handleRun}
          className="absolute top-4 right-4 z-10 bg-green-500 hover:bg-green-600 text-white font-semibold px-6 py-2.5 rounded-lg shadow-lg transition-colors flex items-center gap-2 cursor-pointer"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
          </svg>
          Train
        </button>

        <ReactFlow
          nodes={reactFlowNodes}
          edges={reactFlowEdges}
          onConnect={onConnect}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          snapToGrid
          snapGrid={[15, 15]}
          defaultEdgeOptions={{ animated: true }}
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>
    </>
  );
}
