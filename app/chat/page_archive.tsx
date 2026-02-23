'use client'
import { useState } from "react"
import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport } from 'ai'

function Chat() {

  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({
      api: "/api/chat_04_stream",
    })
  })

  const [input, setInput] = useState('')

  console.log("Input", input)
  return (
    <div className="max-w-3xl mx-auto w-full mt-20">
      <form onSubmit={e => {
        e.preventDefault()
        sendMessage({ text: input })
        setInput("")
      }}>
        <input type="text" value={input} onChange={e => setInput(e.target.value)} />
        <button type="submit">Send</button>
      </form>
      {(status === 'submitted' || status === 'streaming') && <div>AI กำลังคิด...</div>}
      
      {messages.map(m => (
        <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
          <div>
              {m.parts.map((part, index) =>
                part.type === 'text' ? (
                  <div key={index} className="whitespace-pre-wrap">{part.text}</div>
                ) : null
              )}
          </div>
        </div>
      ))}
    </div>
  )
}

export default Chat
