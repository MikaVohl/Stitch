import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type Connection,
  type NodeTypes,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useGraphStore, graphToArchitecture } from '@/store/graphStore'
import { InputLayerNode } from '@/components/nodes/InputLayerNode'
import { DenseLayerNode } from '@/components/nodes/DenseLayerNode'
import { OutputLayerNode } from '@/components/nodes/OutputLayerNode'
import { HyperparamsPanel, type Hyperparams, DEFAULT_HYPERPARAMS } from '@/components/HyperparamsPanel'
import { validateConnection, notifyConnectionError, hasIncomingConnection } from '@/lib/shapeInference'
import { TrainingMetricsSlideOver } from '@/components/TrainingMetricsSlideOver'
import { useTrainingMetrics } from '@/hooks/useTraining'


const nodeTypes: NodeTypes = {
  input: InputLayerNode,
  dense: DenseLayerNode,
  output: OutputLayerNode,
};

export default function Playground() {
  const { layers, edges, addLayer, addEdge, removeEdge } = useGraphStore()
  const [hyperparams, setHyperparams] = useState<Hyperparams>(DEFAULT_HYPERPARAMS)
  const [metricsSlideOverOpen, setMetricsSlideOverOpen] = useState(false)
  const {
    metrics,
    currentState,
    isTraining,
    runId,
    startTraining,
  } = useTrainingMetrics()

  function save() {
    
  }

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
    // Only initialize if no layers exist
    if (Object.keys(layers).length === 0) {
      addLayer({
        id: 'input-1',
        kind: 'Input',
        params: { size: 784 },
      })

      addLayer({
        id: 'dense-1',
        kind: 'Dense',
        params: { units: 128, activation: 'relu' }
      })

      addLayer({
        id: 'dense-2',
        kind: 'Dense',
        params: { units: 64, activation: 'relu' }
      })

      addLayer({
        id: 'output-1',
        kind: 'Output',
        params: { classes: 10, activation: 'softmax' },
      })

      // Connect the layers
      addEdge({ id: 'input-1-dense-1', source: 'input-1', target: 'dense-1' })
      addEdge({ id: 'dense-1-dense-2', source: 'dense-1', target: 'dense-2' })
      addEdge({ id: 'dense-2-output-1', source: 'dense-2', target: 'output-1' })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRun = useCallback(() => {
    try {
      // Convert graph to backend architecture format
      const architecture = graphToArchitecture(layers, edges)

      console.log('🚀 Starting training with architecture:', architecture)
      console.log('📊 Hyperparameters:', hyperparams)

      setMetricsSlideOverOpen(true)

      startTraining({
        architecture,
        hyperparams,
      })
    } catch (error) {
      console.error('❌ Training error:', error)
    }
  }, [layers, edges, hyperparams, startTraining])
  return (
    <>
      <div style={{ width: '100vw', height: 'calc(100vh - 4rem)', position: 'relative' }}>
        {/* Hyperparameters Panel */}
        <HyperparamsPanel onParamsChange={setHyperparams} />

        <button
          onClick={() => save()}
          className="absolute top-4 right-4 z-10 bg-blue-500 hover:bg-blue-600 text-white font-semibold px-6 py-2.5 rounded-lg shadow-lg transition-colors flex items-center gap-2 cursor-pointer"
        >
          Save
        </button>

        {/* Floating Train Button */}
        <button
          onClick={handleRun}
          disabled={isTraining}
          className={`absolute top-4 right-26 z-10 ${isTraining
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-green-500 hover:bg-green-600 cursor-pointer'
            } text-white font-semibold px-6 py-2.5 rounded-lg shadow-lg transition-colors flex items-center gap-2`}
        >
          {isTraining ? (
            <>
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Training...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
              </svg>
              Train
            </>
          )}
        </button>

        {(metrics.length > 0 || currentState !== null) && (
          <button
            onClick={() => setMetricsSlideOverOpen(true)}
            className="absolute top-4 right-54 z-10 bg-blue-500 hover:bg-blue-600 text-white font-semibold px-6 py-2.5 rounded-lg shadow-lg transition-colors flex items-center gap-2 cursor-pointer"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
            </svg>
            Metrics
          </button>
        )}

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
          <Controls position="bottom-left" showInteractive={false} className="shadow-lg" />
        </ReactFlow>
      </div>

      <TrainingMetricsSlideOver
        open={metricsSlideOverOpen}
        onOpenChange={setMetricsSlideOverOpen}
        isTraining={isTraining}
        metrics={metrics}
        currentState={currentState}
        runId={runId}
      />
    </>
  )
}
