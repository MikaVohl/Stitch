import type { FC } from 'react'
import { memo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { MetricData } from '@/api/types'

interface MetricsChartsProps {
  metrics: MetricData[]
}

export const MetricsCharts: FC<MetricsChartsProps> = memo(({ metrics }) => {
  if (metrics.length <= 1) return null

  return (
    <>
      {/* ---------- LOSS CHART ---------- */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 text-center">
          Loss Over Time
        </h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="epoch"
                label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                label={{
                  value: 'Loss',
                  angle: -90,
                  position: 'insideLeft',
                  dy: 20,
                  dx: -4.3 // slightly lower for spacing
                }}
              />
              <Tooltip />
              <Legend
                verticalAlign="bottom"
                align="center"
                wrapperStyle={{
                  bottom: -7,
                  left: 42,
                }}
              />
              <Line
                type="monotone"
                dataKey="train_loss"
                stroke="#dc2626"
                name="Train Loss"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke="#2563eb"
                name="Val Loss"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ---------- ACCURACY CHART ---------- */}
      <div className="space-y-2 mt-6">
        <h3 className="text-sm font-semibold text-gray-700 text-center">
          Accuracy Over Time
        </h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="epoch"
                label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                label={{
                  value: 'Accuracy',
                  angle: -90,
                  position: 'insideLeft',
                  dy: 34,
                  dx: -4.3 // space tweak for consistency
                }}
                domain={[0, 1]}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
              <Legend
                verticalAlign="bottom"
                align="center"
                wrapperStyle={{
                  bottom: -7,
                  left: 42,
                }}
              />
              <Line
                type="monotone"
                dataKey="train_accuracy"
                stroke="#16a34a"
                name="Train Acc"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_accuracy"
                stroke="#9333ea"
                name="Val Acc"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  )
})

MetricsCharts.displayName = 'MetricsCharts'
