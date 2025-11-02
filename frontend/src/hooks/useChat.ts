import { useState, useCallback, useRef } from 'react'
import type { AnyLayer, GraphEdge } from '@/types/graph'

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface ProposedSchema {
  layers: Record<string, AnyLayer>
  edges: GraphEdge[]
}

const SCHEMA_MARKER =
  'Certainly! Here is your current architecture schema, re-implemented as requested:'

const removeJsonCodeBlocks = (text: string): string => {
  return text.replace(/```json[\s\S]*?```/gi, '').trim()
}

export interface UseChatReturn {
  messages: ChatMessage[]
  isStreaming: boolean
  isGeneratingSchema: boolean
  proposedSchema: ProposedSchema | null
  sendMessage: (message: string, requestSchemaChange: boolean, currentSchema?: { layers: Record<string, AnyLayer>, edges: GraphEdge[] }) => void
  clearProposedSchema: () => void
  addMessage: (message: ChatMessage) => void
  clearMessages: () => void
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [isGeneratingSchema, setIsGeneratingSchema] = useState(false)
  const [proposedSchema, setProposedSchema] = useState<ProposedSchema | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(
    async (
      message: string,
      requestSchemaChange: boolean,
      currentSchema?: { layers: Record<string, AnyLayer>, edges: GraphEdge[] }
    ) => {
      setMessages((prev) => [...prev, { role: 'user', content: message }])
      setProposedSchema(null)
      setIsStreaming(true)

      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }

      const abortController = new AbortController()
      abortControllerRef.current = abortController

      try {
        const response = await fetch('http://127.0.0.1:8080/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message,
            currentSchema: requestSchemaChange ? currentSchema : null,
            requestSchemaChange,
          }),
          signal: abortController.signal,
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()

        if (!reader) {
          throw new Error('No response body')
        }

        let assistantMessage = ''
        let stopDisplaying = false
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()

          if (done) {
            break
          }

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('event:')) {
              continue
            }

            if (line.startsWith('data:')) {
              const data = line.substring(5).trim()

              try {
                const parsed = JSON.parse(data)

                if (parsed.content) {
                  assistantMessage += parsed.content

                  const markerIndex = assistantMessage.indexOf(SCHEMA_MARKER)

                  if (!stopDisplaying && markerIndex !== -1) {
                    stopDisplaying = true
                    setIsGeneratingSchema(true)
                  }

                  const contentBeforeSchema =
                    stopDisplaying && markerIndex !== -1
                      ? assistantMessage.substring(0, markerIndex + SCHEMA_MARKER.length).trim()
                      : assistantMessage

                  const sanitizedMessage = removeJsonCodeBlocks(contentBeforeSchema)

                  setMessages((prev) => {
                    const newMessages = [...prev]
                    const lastMessage = newMessages[newMessages.length - 1]

                    if (lastMessage?.role === 'assistant') {
                      newMessages[newMessages.length - 1] = {
                        role: 'assistant',
                        content: sanitizedMessage,
                      }
                    } else {
                      newMessages.push({
                        role: 'assistant',
                        content: sanitizedMessage,
                      })
                    }

                    return newMessages
                  })
                } else if (parsed.proposedSchema) {
                  setProposedSchema(parsed.proposedSchema)
                  setIsGeneratingSchema(false)
                } else if (parsed.error) {
                  setMessages((prev) => [
                    ...prev,
                    {
                      role: 'assistant',
                      content: `Error: ${parsed.error}`,
                    },
                  ])
                }
              } catch (e) {
              }
            }
          }
        }
      } catch (error: unknown) {
        if (error instanceof Error && error.name !== 'AbortError') {
          setMessages((prev) => [
            ...prev,
            {
              role: 'assistant',
              content: 'Sorry, I encountered an error. Please try again.',
            },
          ])
        }
      } finally {
        setIsStreaming(false)
        setIsGeneratingSchema(false)
        abortControllerRef.current = null
      }
    },
    []
  )

  const clearProposedSchema = useCallback(() => {
    setProposedSchema(null)
  }, [])

  const addMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message])
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return {
    messages,
    isStreaming,
    isGeneratingSchema,
    proposedSchema,
    sendMessage,
    clearProposedSchema,
    addMessage,
    clearMessages,
  }
}
