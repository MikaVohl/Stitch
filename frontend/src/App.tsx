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
              <h1 className="text-xl flex items-center gap-2 font-bold text-gray-900"><img src="/Group 9.svg" alt="" className="h-4 w-14" />NETBURGER™</h1>
              <nav className="flex gap-4 text-sm justify-center font-medium">
                <NavLink
                  to="/playground"
                  className="flex items-center gap-2"
                >
                  <img src="/build.svg" alt="" className="h-4 w-4 transition-transform hover:-rotate-45" />
                  Build
                </NavLink>
                <NavLink
                  to="/models"
                  className="flex items-center gap-2"
                >
                  <img src="/models.svg" alt="" className="h-4 w-4 transition-transform hover:-translate-y-1" />
                  Models
                </NavLink>
                <NavLink
                  to="/test"
                  className="flex items-center gap-2"
                >
                  <img src="/test.svg" alt="" className="h-4 w-4 transition-transform hover:translate-x-1 hover:-translate-y-1" />
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
          </Routes>
        </BrowserRouter>

      </QueryClientProvider>
    </>
  )
}
