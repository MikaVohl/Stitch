import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type Connection,
  type NodeTypes,
  type NodeChange,
  type ReactFlowInstance,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useGraphStore, graphToArchitecture } from '@/store/graphStore'
import { InputLayerNode } from '@/components/nodes/InputLayerNode'
import { DenseLayerNode } from '@/components/nodes/DenseLayerNode'
import { ConvLayerNode } from '@/components/nodes/ConvLayerNode'
import { OutputLayerNode } from '@/components/nodes/OutputLayerNode'
import { PoolingLayerNode } from '@/components/nodes/PoolingLayerNode'
import { FlattenLayerNode } from '@/components/nodes/FlattenLayerNode'
import { DropoutLayerNode } from '@/components/nodes/DropoutLayerNode'
import { HyperparamsPanel, type Hyperparams, DEFAULT_HYPERPARAMS } from '@/components/HyperparamsPanel'
import { validateConnection, notifyConnectionError } from '@/lib/shapeInference'
import { TrainingMetricsSlideOver } from '@/components/TrainingMetricsSlideOver'
import { useTrainingMetrics } from '@/hooks/useTraining'
import { LayersPanel } from '@/components/LayersPanel'
import type { ActivationType, LayerKind, AnyLayer } from '@/types/graph'
import { ChatbotPanel } from '@/components/ChatbotPanel'
import { SchemaProposalPreview } from '@/components/SchemaProposalPreview'
import { useChat } from '@/hooks/useChat'
import { PresetChips, getPresetGraph, type PresetType } from '@/components/PresetChips'

const nodeTypes: NodeTypes = {
  input: InputLayerNode,
  dense: DenseLayerNode,
  conv: ConvLayerNode,
  pool: PoolingLayerNode,
  flatten: FlattenLayerNode,
  dropout: DropoutLayerNode,
  output: OutputLayerNode,
};

const layerKindToNodeType: Record<LayerKind, keyof typeof nodeTypes> = {
  Input: 'input',
  Dense: 'dense',
  Convolution: 'conv',
  Pooling: 'pool',
  Flatten: 'flatten',
  Dropout: 'dropout',
  Output: 'output',
}

const layerIdPrefixes: Record<LayerKind, string> = {
  Input: 'input',
  Dense: 'dense',
  Convolution: 'conv',
  Pooling: 'pool',
  Flatten: 'flatten',
  Dropout: 'dropout',
  Output: 'output',
}

const duplicableLayerKinds = new Set<LayerKind>(['Dense', 'Convolution', 'Pooling', 'Dropout', 'Flatten'])

function generateLayerId(kind: LayerKind) {
  const prefix = layerIdPrefixes[kind]
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1000)}`
}

export default function Playground() {
  const { layers, edges, addLayer, addEdge, removeEdge, updateLayerPosition, removeLayer, applyProposedSchema, loadGraph } = useGraphStore()
  const [hyperparams, setHyperparams] = useState<Hyperparams>(DEFAULT_HYPERPARAMS)
  const [metricsSlideOverOpen, setMetricsSlideOverOpen] = useState(false)
  const [currentPreset, setCurrentPreset] = useState<PresetType>('blank')
  const [showProposalPreview, setShowProposalPreview] = useState(false)
  const {
    metrics,
    currentState,
    isTraining,
    runId,
    samplePredictions,
    startTraining,
  } = useTrainingMetrics()
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([])
  const [selectedEdgeIds, setSelectedEdgeIds] = useState<string[]>([])
  const [copiedLayer, setCopiedLayer] = useState<{
    kind: LayerKind
    params: Record<string, any>
    basePosition: { x: number; y: number } | null
    offset: number
  } | null>(null)
  const { messages, isStreaming, isGeneratingSchema, proposedSchema, sendMessage, clearProposedSchema } = useChat()

  // Convert store state to ReactFlow format with auto-layout
  const reactFlowNodes = useMemo((): Node[] => {
    const layerArray = Object.values(layers);
    return layerArray.map((layer, index) => ({
      id: layer.id,
      type: layerKindToNodeType[layer.kind],
      position: layer.position ?? { x: index * 300 + 50, y: 250 },
      data: {},
      draggable: true,
      style: {
        background: 'transparent',
        border: 'none',
        padding: 0,
        boxShadow: 'none',
      },
    }));
  }, [layers]);

  const reactFlowEdges = useMemo((): Edge[] => {
    return edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.sourceHandle ?? undefined,
      targetHandle: edge.targetHandle ?? undefined,
      label: edge.label,
      animated: true,
    }));
  }, [edges]);

  const onConnect = useCallback(
    (connection: Connection) => {
      const { source, target } = connection;
      if (!source || !target) return;

      const sourceLayer = layers[source];
      const targetLayer = layers[target];

      if (!sourceLayer || !targetLayer) return;

      const validation = validateConnection(sourceLayer, targetLayer);

      if (!validation.valid) {
        notifyConnectionError(validation.error || 'Invalid connection');
        return;
      }

      const handleKey = (handle?: string | null) => handle ?? 'default';
      const edgeId = `${source}:${handleKey(connection.sourceHandle)}->${target}:${handleKey(connection.targetHandle)}`;
      addEdge({
        id: edgeId,
        source,
        target,
        sourceHandle: connection.sourceHandle ?? null,
        targetHandle: connection.targetHandle ?? null,
      });
    },
    [layers, addEdge]
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

  const reactFlowWrapper = useRef<HTMLDivElement | null>(null)
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null)

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      changes.forEach((change) => {
        if (change.type === 'position' && change.position) {
          updateLayerPosition(change.id, change.position)
        }
        if (change.type === 'remove') {
          removeLayer(change.id)
        }
      })
    },
    [updateLayerPosition, removeLayer]
  )

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }, [])

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      if (!reactFlowWrapper.current || !reactFlowInstance) return
      const raw = event.dataTransfer.getData('application/layer-template')
      if (!raw) return

      type LayerTemplatePayload =
        | { kind: 'Dense'; params: { units: number; activation: ActivationType } }
        | {
            kind: 'Convolution'
            params: {
              filters: number
              kernel: number
              stride: number
              padding: 'valid' | 'same'
              activation: Exclude<ActivationType, 'softmax'>
            }
          }
        | { kind: 'Flatten'; params: Record<string, never> }
        | { kind: 'Dropout'; params: { rate: number } }

      try {
        const payload = JSON.parse(raw) as LayerTemplatePayload

        const position = reactFlowInstance.screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        })

        if (payload.kind === 'Dense') {
          addLayer({
            id: generateLayerId('Dense'),
            kind: 'Dense',
            params: {
              units: payload.params.units,
              activation: payload.params.activation,
            },
            position,
          })
        } else if (payload.kind === 'Convolution') {
          addLayer({
            id: generateLayerId('Convolution'),
            kind: 'Convolution',
            params: {
              filters: payload.params.filters,
              kernel: payload.params.kernel,
              stride: payload.params.stride,
              padding: payload.params.padding,
              activation: payload.params.activation,
            },
            position,
          })
        } else if (payload.kind === 'Flatten') {
          addLayer({
            id: generateLayerId('Flatten'),
            kind: 'Flatten',
            params: {},
            position,
          })
        } else if (payload.kind === 'Dropout') {
          addLayer({
            id: generateLayerId('Dropout'),
            kind: 'Dropout',
            params: {
              rate: payload.params.rate,
            },
            position,
          })
        }
      } catch (error) {
      }
    },
    [addLayer, reactFlowInstance]
  )
  const onSelectionChange = useCallback((params: { nodes: Node[]; edges: Edge[] }) => {
    setSelectedNodeIds(params.nodes.map((node) => node.id))
    setSelectedEdgeIds(params.edges.map((edge) => edge.id))
  }, [])

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      if (target) {
        const tagName = target.tagName
        if (tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT' || target.isContentEditable) {
          return
        }
      }

      const isCopy = (event.key === 'c' || event.key === 'C') && (event.ctrlKey || event.metaKey)
      const isPaste = (event.key === 'v' || event.key === 'V') && (event.ctrlKey || event.metaKey)
      const isDeleteKey = event.key === 'Delete' || event.key === 'Backspace'

      if (isCopy) {
        if (selectedNodeIds.length === 1) {
          const layer = layers[selectedNodeIds[0]] as AnyLayer | undefined
          if (layer && duplicableLayerKinds.has(layer.kind)) {
            setCopiedLayer({
              kind: layer.kind,
              params: JSON.parse(JSON.stringify(layer.params)),
              basePosition: layer.position ? { ...layer.position } : null,
              offset: 1,
            })
            event.preventDefault()
          }
        }
        return
      }

      if (isPaste) {
        if (copiedLayer && duplicableLayerKinds.has(copiedLayer.kind)) {
          const base = copiedLayer.basePosition ?? { x: 100, y: 100 }
          const offsetDistance = 30 * copiedLayer.offset
          const position = {
            x: base.x + offsetDistance,
            y: base.y + offsetDistance,
          }
          const newId = generateLayerId(copiedLayer.kind)

          addLayer({
            id: newId,
            kind: copiedLayer.kind,
            params: JSON.parse(JSON.stringify(copiedLayer.params)),
            position,
          })

          setCopiedLayer((prev) =>
            prev
              ? {
                ...prev,
                offset: prev.offset + 1,
              }
              : prev
          )
          event.preventDefault()
        }
        return
      }

      if (isDeleteKey) {
        if (selectedNodeIds.length > 0 || selectedEdgeIds.length > 0) {
          event.preventDefault()
          selectedNodeIds.forEach((id) => removeLayer(id))
          selectedEdgeIds.forEach((id) => removeEdge(id))
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [selectedNodeIds, selectedEdgeIds, layers, copiedLayer, addLayer, removeLayer, removeEdge])

  const handlePresetSelect = useCallback((preset: PresetType) => {
    const presetGraph = getPresetGraph(preset)
    loadGraph(presetGraph.layers, presetGraph.edges)
    setCurrentPreset(preset)
  }, [loadGraph])

  useEffect(() => {
    if (Object.keys(layers).length === 0) {
      const blankPreset = getPresetGraph('blank')
      loadGraph(blankPreset.layers, blankPreset.edges)
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

  const handleApplyProposal = useCallback(() => {
    if (proposedSchema) {
      console.log('Applying proposed schema:', proposedSchema)
      applyProposedSchema(proposedSchema)
      setShowProposalPreview(false)
      clearProposedSchema()
    }
  }, [proposedSchema, applyProposedSchema, clearProposedSchema])

  const handleRejectProposal = useCallback(() => {
    setShowProposalPreview(false)
    clearProposedSchema()
  }, [clearProposedSchema])

  return (
    <>
      <div style={{ width: '100vw', height: 'calc(100vh - 4rem)', position: 'relative' }}>
        <div className="absolute top-4 left-4 z-10 flex flex-col gap-4 pointer-events-none">
          <div className="flex flex-row gap-4 items-start">
            <HyperparamsPanel onParamsChange={setHyperparams} />
            <PresetChips onPresetSelect={handlePresetSelect} />
          </div>
          <LayersPanel />
        </div>

        <div className="absolute top-4 right-4 z-10 flex flex-row gap-4">
          {(metrics.length > 0 || currentState !== null) && (
            <button
              onClick={() => setMetricsSlideOverOpen(true)}
              className="z-10 bg-blue-500 hover:bg-blue-600 text-white font-semibold px-6 py-2.5 rounded-lg shadow-lg transition-colors flex items-center gap-2 cursor-pointer"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
              </svg>
              Metrics
            </button>
          )}
          <button
            onClick={handleRun}
            disabled={isTraining}
            className={`${isTraining
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

        </div>

        <div className="h-full w-full" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={reactFlowNodes}
            edges={reactFlowEdges}
            onConnect={onConnect}
            onEdgesChange={onEdgesChange}
            onNodesChange={onNodesChange}
            onSelectionChange={onSelectionChange}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodesDraggable={!isTraining}
            nodesConnectable={!isTraining}
            nodeTypes={nodeTypes}
            fitView
            snapToGrid
            snapGrid={[15, 15]}
            defaultEdgeOptions={{ animated: true }}
            onInit={setReactFlowInstance}
          >
            <Background />
            <Controls position="bottom-left" showInteractive={false} className="shadow-lg" />
          </ReactFlow>
        </div>
      </div>

      <TrainingMetricsSlideOver
        open={metricsSlideOverOpen}
        onOpenChange={setMetricsSlideOverOpen}
        isTraining={isTraining}
        metrics={metrics}
        currentState={currentState}
        runId={runId}
        samplePredictions={samplePredictions}
      />

      <ChatbotPanel
        onViewProposal={() => setShowProposalPreview(true)}
        messages={messages}
        isStreaming={isStreaming}
        isGeneratingSchema={isGeneratingSchema}
        proposedSchema={proposedSchema}
        sendMessage={sendMessage}
      />

      {showProposalPreview && proposedSchema && (
        <SchemaProposalPreview
          currentLayers={layers}
          currentEdges={edges}
          proposedLayers={proposedSchema.layers}
          proposedEdges={proposedSchema.edges}
          onApply={handleApplyProposal}
          onReject={handleRejectProposal}
        />
      )}
    </>
  )
}
