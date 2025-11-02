import '@xyflow/react/dist/style.css'
import { Toaster } from 'sonner'
import { BrowserRouter, NavLink, Route, Routes, Navigate } from 'react-router-dom'
import Playground from './routes/Playground'
import Models from './routes/Models'
import ModelPage from './routes/Model'
import Test from './routes/Test'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient()

export default function App() {
  return (
    <>
      <Toaster position="top-right" />
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <header className="h-16 border-b border-gray-200 bg-white shadow-sm">
            <div className="h-full grid grid-cols-3 max-w-7xl mx-auto px-4 flex gap-8">
              <h1 className="text-xl flex items-center gap-2 font-bold text-gray-900"><img src="/kiwi.svg" alt="" className="h-6 w-6" />Stitch</h1>
                <nav className="flex gap-4 text-sm justify-center font-medium">
                <NavLink
                  to="/playground"
                  className="group flex items-center gap-2 hover:text-gray-600 transition-colors"
                >
                  <img src="/build.svg" alt="" className="h-4 w-4 transition-all group-hover:-rotate-45 group-hover:opacity-60" />
                  Build
                </NavLink>
                <NavLink
                  to="/models"
                  className="group flex items-center gap-2 hover:text-gray-600 transition-colors"
                >
                  <img src="/models.svg" alt="" className="h-4 w-4 transition-all group-hover:-translate-y-1 group-hover:opacity-60" />
                  Models
                </NavLink>
                <NavLink
                  to="/test"
                  className="group flex items-center gap-2 hover:text-gray-600 transition-colors"
                >
                  <img src="/test.svg" alt="" className="h-4 w-4 transition-all group-hover:translate-x-1 group-hover:-translate-y-1 group-hover:opacity-60" />
                  Test
                </NavLink>
                </nav>
            </div>
          </header>


          <Routes>
            <Route path="/" element={<Navigate to="/playground" replace />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/models" element={<Models />} />
            <Route path="/models/:id" element={<ModelPage />} />
            <Route path="/test" element={<Test />} />
            <Route path="/test/:modelId" element={<Test />} />
          </Routes>
        </BrowserRouter>

      </QueryClientProvider>
    </>
  )
}
