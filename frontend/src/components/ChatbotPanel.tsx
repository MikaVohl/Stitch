import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { useGraphStore } from '@/store/graphStore'
import type { ChatMessage, ProposedSchema } from '@/hooks/useChat'
import type { AnyLayer, GraphEdge } from '@/types/graph'

interface ChatbotPanelProps {
  onViewProposal: () => void
  messages: ChatMessage[]
  isStreaming: boolean
  isGeneratingSchema: boolean
  proposedSchema: ProposedSchema | null
  sendMessage: (
    message: string,
    requestSchemaChange: boolean,
    currentSchema?: { layers: Record<string, AnyLayer>; edges: GraphEdge[] }
  ) => void
}

const FUNNY_LOADING_MESSAGES = [
  "Summoning neural spirits...",
  "Teaching tensors to dance...",
  "Waking up the gradient descent elves...",
  "Consulting the architecture oracle...",
  "Brewing some fresh gradients...",
  "Aligning the hidden layers...",
  "Asking the AI overlords nicely...",
  "Defragmenting the neural pathways...",
  "Spinning up the backpropagation hamster wheel...",
  "Calibrating the activation functions..."
]

const SUGGESTED_PROMPTS = [
  "Improve my architecture",
  "Add a convolutional layer",
  "Explain batch normalization",
  "When should I use dropout?",
  "What activation function to use?",
  "Add regularization"
]

export function ChatbotPanel({ onViewProposal, messages, isStreaming, isGeneratingSchema, proposedSchema, sendMessage }: ChatbotPanelProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [loadingMessageIndex, setLoadingMessageIndex] = useState(0)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { layers, edges } = useGraphStore()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (!isGeneratingSchema) {
      setLoadingMessageIndex(0)
      return
    }

    const interval = setInterval(() => {
      setLoadingMessageIndex((prev) => (prev + 1) % FUNNY_LOADING_MESSAGES.length)
    }, 2500)

    return () => clearInterval(interval)
  }, [isGeneratingSchema])

  const handleSend = (messageToSend?: string) => {
    const message = messageToSend || inputValue
    if (!message.trim() || isStreaming) return

    sendMessage(message, true, { layers, edges })
    setInputValue('')
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white rounded-full p-4 shadow-lg transition-all z-50 cursor-pointer"
      >
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
          />
        </svg>
      </button>
    )
  }

  return (
    <div className="fixed bottom-4 right-4 w-[480px] h-[700px] bg-white rounded-lg shadow-2xl flex flex-col z-50 border border-gray-200">
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <h3 className="font-semibold text-gray-900">AI Assistant</h3>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="text-gray-400 hover:text-gray-600 transition-colors cursor-pointer"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 text-sm mt-8">
            <p>Hi! I'm your AI assistant.</p>
            <p className="mt-2">Ask me anything about neural networks or request changes to your architecture!</p>
            <div className="mt-6 flex flex-wrap gap-2 justify-center px-4">
              {SUGGESTED_PROMPTS.map((prompt, index) => (
                <button
                  key={index}
                  onClick={() => handleSend(prompt)}
                  disabled={isStreaming}
                  className="px-3 py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 text-xs font-medium rounded-full border border-blue-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : message.role === 'system' ? 'justify-center' : 'justify-start'}`}
          >
            <div
              className={`${message.role === 'system' ? 'max-w-[90%]' : 'max-w-[80%]'} rounded-lg px-4 py-2 ${message.role === 'user'
                ? 'bg-blue-600 text-white'
                : message.role === 'system'
                  ? 'bg-green-50 text-green-800 border border-green-200'
                  : 'bg-gray-100 text-gray-900'
                }`}
            >
              {message.role === 'user' ? (
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              ) : message.role === 'system' ? (
                <p className="text-xs text-center font-medium">{message.content}</p>
              ) : (
                <div className="text-sm prose prose-sm max-w-none">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        ))}

        {isGeneratingSchema && (
          <div className="flex justify-start">
            <div className="bg-gray-100 text-gray-900 rounded-lg px-4 py-3">
              <div className="flex flex-col items-center gap-3">
                <svg
                  className="w-8 h-8 animate-spin text-blue-600"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="3"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                <p className="text-sm text-gray-600 font-medium animate-pulse">
                  {FUNNY_LOADING_MESSAGES[loadingMessageIndex]}
                </p>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {proposedSchema && (
        <div className="px-4 pb-2">
          <button
            onClick={onViewProposal}
            className="w-full bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded-lg text-sm font-medium transition-colors cursor-pointer"
          >
            View Architecture Proposal
          </button>
        </div>
      )}

      <div className="p-4 border-t border-gray-200">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything..."
            disabled={isStreaming}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm disabled:bg-gray-100"
          />
          <button
            onClick={() => handleSend()}
            disabled={!inputValue.trim() || isStreaming}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white px-4 py-2 rounded-lg transition-colors"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
