import { useCallback, useEffect, useRef, useState } from 'react'

interface DrawingGridProps {
  width?: number
  height?: number
  onDrawingComplete?: (pixels: number[][]) => void
}

export function DrawingGrid({ width = 28, height = 28, onDrawingComplete }: DrawingGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [context, setContext] = useState<CanvasRenderingContext2D | null>(null)
  const scaleFactor = 10 // Each pixel will be 10x10 on screen

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set actual dimensions
    canvas.width = width
    canvas.height = height

    // Set display size
    canvas.style.width = `${width * scaleFactor}px`
    canvas.style.height = `${height * scaleFactor}px`

    // Set initial state
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, width, height)
    ctx.fillStyle = 'white'

    setContext(ctx)
  }, [width, height])

    const draw = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !context || !canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const scaleX = width / rect.width
    const scaleY = height / rect.height

    const x = (event.clientX - rect.left) * scaleX
    const y = (event.clientY - rect.top) * scaleY

    context.fillRect(x - 1, y - 1, 2, 2)
  }, [isDrawing, context, width, height])

  const startDrawing = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true)
    draw(event)
  }, [draw])

  const stopDrawing = useCallback(() => {
    setIsDrawing(false)
    
    if (onDrawingComplete && context) {
      const imageData = context.getImageData(0, 0, width, height)
      const pixels: number[][] = []
      
      for (let y = 0; y < height; y++) {
        pixels[y] = []
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4
        pixels[y][x] = imageData.data[i]
      }
      }
      
      onDrawingComplete(pixels)
    }
  }, [context, width, height, onDrawingComplete])



  const clearCanvas = useCallback(() => {
    if (!context) return
    context.fillStyle = 'black'
    context.fillRect(0, 0, width, height)
    context.fillStyle = 'white'
  }, [context, width, height])

  return (
    <div className="flex flex-col items-center gap-4">
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseUp={stopDrawing}
        onMouseOut={stopDrawing}
        onMouseMove={draw}
        className="cursor-crosshair rounded border border-gray-300 bg-white"
        style={{
          imageRendering: 'pixelated'
        }}
      />
      <button
        onClick={clearCanvas}
        className="rounded-md bg-gray-100 px-3 cursor-pointer py-2 text-sm font-medium text-gray-700 hover:bg-gray-200"
      >
        Clear
      </button>
    </div>
  )
}
