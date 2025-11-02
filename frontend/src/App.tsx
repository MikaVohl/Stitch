import '@xyflow/react/dist/style.css'
import { Toaster } from 'sonner'
import { BrowserRouter, NavLink, Route, Routes, Navigate } from 'react-router-dom'
import Playground from './routes/Playground'
import Models from './routes/Models'
import Arena from './routes/Arena'
import ModelPage from './routes/Model'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient()

export default function App() {
  return (
    <>
      <Toaster position="top-right" />
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <header className="h-16 border-b border-gray-200 bg-white shadow-sm">
            <div className="h-full max-w-7xl mx-auto px-4 flex items-center gap-8">
              <h1 className="text-xl flex items-center gap-2 font-bold text-gray-900"><img src="/Group 9.svg" alt="" className="h-4 w-14" />NETBURGER</h1>
              <nav className="flex gap-4 text-sm font-medium">
                <NavLink
                  to="/playground"
                  className="flex items-center gap-2"
                >
                  <img src="/build.svg" alt="" className="h-4 w-4" />
                  Build
                </NavLink>
                <NavLink
                  to="/models"
                  className="flex items-center gap-2"
                >
                  <img src="/models.svg" alt="" className="h-4 w-4" />
                  Models
                </NavLink>
                <NavLink
                  to="/arena"
                  className="flex items-center gap-2"
                >
                  <img src="/arena.svg" alt="" className="h-4 w-4" />
                  Arena
                </NavLink>
              </nav>
            </div>
          </header>


          <Routes>
            <Route path="/" element={<Navigate to="/playground" replace />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/models" element={<Models />} />
            <Route path="/models/:id" element={<ModelPage />} />
            <Route path="/arena" element={<Arena />} />
          </Routes>
        </BrowserRouter>

      </QueryClientProvider>
    </>
  )
}
