import type { DragEvent } from 'react'

const DENSE_LAYER_TEMPLATE = {
  id: 'dense-layer',
  label: 'Dense Layer',
  description: 'Default 64 units · ReLU',
  units: 64,
  activation: 'relu' as const,
}

function handleDragStart() {
  const template = DENSE_LAYER_TEMPLATE
  return (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.effectAllowed = 'copy'
    event.dataTransfer.setData(
      'application/layer-template',
      JSON.stringify({
        kind: 'Dense',
        params: {
          units: template.units,
          activation: template.activation,
        },
      })
    )
  }
}

export function LayersPanel({ className }: { className?: string }) {
  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className ?? ''}`}>
      <div className="w-full px-4 py-2.5 flex items-center justify-between rounded-t-lg border-b border-gray-200">
        <span className="font-semibold text-gray-700 text-sm">Layer Palette</span>
      </div>
      <div className="p-4 flex flex-col gap-3 min-w-[280px]">
        <p className="text-xs text-gray-500">
          Drag the preset below onto the canvas, then adjust units or activation inline.
        </p>
        <div
          key={DENSE_LAYER_TEMPLATE.id}
          draggable
          onDragStart={handleDragStart()}
          className="border border-dashed border-blue-300 rounded-lg px-3 py-2 bg-blue-50 hover:bg-blue-100 cursor-grab active:cursor-grabbing transition-colors"
        >
          <div className="text-sm font-semibold text-blue-700">{DENSE_LAYER_TEMPLATE.label}</div>
          <div className="text-xs text-blue-600">{DENSE_LAYER_TEMPLATE.description}</div>
        </div>
      </div>
    </div>
  )
}
