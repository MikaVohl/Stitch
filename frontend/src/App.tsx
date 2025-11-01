import '@xyflow/react/dist/style.css'
import { Toaster } from 'sonner'
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import Playground from './routes/Playground'


export default function App() {
  return (
    <>
      <Toaster position="top-right" />
      <BrowserRouter>
        <header className="h-16 border-b border-gray-200 bg-white shadow-sm">
          <div className="h-full max-w-7xl mx-auto px-4 flex items-center gap-8">
            <h1 className="text-xl font-bold text-gray-900">MAIS 2025</h1>
            <nav className="flex gap-6">
              <Link
                to="/playground"
                className="text-gray-600 hover:text-gray-900 font-medium transition-colors"
              >
                Playground
              </Link>
              <Link
                to="/models"
                className="text-gray-600 hover:text-gray-900 font-medium transition-colors"
              >
                Models
              </Link>
            </nav>
          </div>
        </header>
        <Routes>
          <Route path="/playground" element={<Playground />} />
        </Routes>
      </BrowserRouter>
    </>
  )
}
