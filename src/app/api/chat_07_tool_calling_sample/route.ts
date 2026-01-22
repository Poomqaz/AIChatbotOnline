// /app/api/chat/route.ts

import { NextRequest } from "next/server"
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"
import { tool } from "@langchain/core/tools"
import {
  ToolMessage,
  HumanMessage,
  AIMessage,
  SystemMessage,
  BaseMessage,
  AIMessageChunk,
} from "@langchain/core/messages"
import { PostgresChatMessageHistory } from "@langchain/community/stores/message/postgres"
import { toUIMessageStream } from "@ai-sdk/langchain"
import { createUIMessageStreamResponse } from "ai"
import { z } from "zod"
import { getDatabase } from '@/src/lib/database'

export const dynamic = 'force-dynamic'
export const maxDuration = 30

// ===============================================
// 1. Type Definitions
// ===============================================

interface ChatMessage {
  role: string
  content: string
}

interface DBMessageRow {
  message: {
    content?: string;
    text?: string;
    type?: string;
    [key: string]: unknown;
  };
  message_type: string;
  created_at: string;
}

interface DBSessionRow {
  id: string;
  title: string;
  created_at: string;
}

type GenericToolInvoker = { 
  invoke: (input: Record<string, unknown>) => Promise<unknown> 
};

// ===============================================
// 2. Helper Functions
// ===============================================

// ฟังก์ชันกรองและซ่อมแซม Message ที่โหลดมาจาก DB
function sanitizeMessages(messages: BaseMessage[]): BaseMessage[] {
  return messages.map((msg) => {
    // ถ้า content เป็น null/undefined ให้แก้เป็น string ว่าง
    if (msg.content === undefined || msg.content === null) {
      msg.content = "";
    }
    // เช็ค additional_kwargs ถ้าไม่มีให้ใส่ object ว่าง (กัน Error)
    if (!msg.additional_kwargs) {
      msg.additional_kwargs = {};
    }
    return msg;
  });
}

async function consumeStream(stream: ReadableStream<AIMessageChunk>): Promise<string> {
  const reader = stream.getReader();
  let accumulatedContent = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      if (value && typeof value.content === 'string') {
        accumulatedContent += value.content;
      } else if (value && Array.isArray(value.content)) {
         const textPart = value.content.find((c) => typeof c === 'object' && c !== null && 'type' in c && (c as Record<string, unknown>).type === 'text') as { text: string } | undefined;
         if (textPart) accumulatedContent += textPart.text;
      }
    }
  } finally {
    reader.releaseLock();
  }
  
  return accumulatedContent;
}

// ===============================================
// 3. Tool Creation
// ===============================================
const getWeatherTool = tool(
  async ({ city }: { city: string }) => {
    // จำลอง Delay นิดหน่อยเพื่อให้เห็น Loading state
    await new Promise(resolve => setTimeout(resolve, 500)); 

    if (city.toLowerCase().includes("bangkok") || city.includes("กรุงเทพ")) {
      return `The weather in Bangkok is 34°C, Sunny, and very hot!`;
    } else if (city.toLowerCase().includes("chiang mai")) {
      return `The weather in Chiang Mai is 25°C and partly cloudy.`;
    }
    return `Sorry, I don't have weather information for ${city}.`;
  },
  {
    name: "get_weather",
    description: "Get the current weather for a specific city in Thailand.",
    schema: z.object({
      city: z.string().describe("The name of the city in English."),
    }),
  }
)

// ===============================================
// 4. GET Handler
// ===============================================
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const sessionId = searchParams.get('sessionId')
    const userId = searchParams.get('userId')

    if (!sessionId && !userId) {
      return new Response(JSON.stringify({ error: "sessionId or userId is required" }), { status: 400 })
    }

    const client = await getDatabase().connect()
    
    try {
      if (sessionId) {
        const result = await client.query(`
          SELECT message, message->>'type' as message_type, created_at
          FROM chat_messages 
          WHERE session_id = $1 
          ORDER BY created_at ASC
        `, [sessionId])

        const messages = result.rows.map((row: DBMessageRow, index: number) => {
          const messageData = row.message
          
          let role = 'user'
          if (row.message_type === 'ai') role = 'assistant'
          else if (row.message_type === 'human') role = 'user'
          else if (row.message_type === 'tool') role = 'tool'
          else if (row.message_type === 'system') role = 'system'
          
          return {
            id: `history-${index}`,
            role: role,
            content: messageData.content || messageData.text || '',
            createdAt: row.created_at
          }
        })

        return new Response(JSON.stringify({ messages }), { status: 200 })
      } 
      else if (userId) {
        const result = await client.query(`
          SELECT id, title, created_at
          FROM chat_sessions
          WHERE user_id = $1
          ORDER BY created_at DESC
        `, [userId])
        
        const sessions: DBSessionRow[] = result.rows;
        return new Response(JSON.stringify({ sessions }), { status: 200 })
      }
      return new Response(JSON.stringify({ error: "Invalid request" }), { status: 400 })
    } finally {
      client.release()
    }
  } catch (error) {
    console.error("GET Error:", error)
    return new Response(JSON.stringify({ error: "Failed to fetch data" }), { status: 500 })
  }
}

// ===============================================
// 5. DELETE Handler
// ===============================================
export async function DELETE(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const sessionId = searchParams.get('sessionId')
    if (!sessionId) {
      return new Response(JSON.stringify({ error: "sessionId is required" }), { status: 400 })
    }
    const client = await getDatabase().connect()
    try {
      await client.query('DELETE FROM chat_messages WHERE session_id = $1', [sessionId])
      await client.query('DELETE FROM chat_sessions WHERE id = $1', [sessionId])
      return new Response(JSON.stringify({ message: "Session deleted successfully" }), { status: 200 })
    } finally {
      client.release()
    }
  } catch (error) {
    console.error("DELETE Error:", error)
    return new Response(JSON.stringify({ error: "Failed to delete session" }), { status: 500 })
  }
}

// ===============================================
// 6. POST Handler
// ===============================================
export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { messages, sessionId, userId } = body as { 
      messages: ChatMessage[];
      sessionId?: string;
      userId?: string;
    }
    
    // Validation
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return new Response(JSON.stringify({ error: "Messages array is required" }), { status: 400 })
    }

    const lastUserMessageData = messages[messages.length - 1]
    
    // Fallback Content: ป้องกัน Error ถ้า content ว่าง
    let userContent = lastUserMessageData?.content 
      ? String(lastUserMessageData.content).trim() 
      : "";

    if (!userContent) {
       userContent = "(Empty message)"; 
    }

    let currentSessionId = sessionId

    // Create Session
    if (!currentSessionId) {
      if (!userId) throw new Error("User ID is required for new session")
      
      const client = await getDatabase().connect()
      try {
        const title = userContent.slice(0, 50) + (userContent.length > 50 ? '...' : '')
        const result = await client.query(`
          INSERT INTO chat_sessions (title, user_id)
          VALUES ($1, $2)
          RETURNING id
        `, [title, userId])
        currentSessionId = result.rows[0].id
      } finally {
        client.release()
      }
    }

    // Setup History
    const messageHistory = new PostgresChatMessageHistory({
      sessionId: currentSessionId!,
      tableName: "chat_messages",
      pool: getDatabase(),
    })

    // 🔴 Step สำคัญ: โหลด History และ "ซ่อมแซม" (Sanitize) ก่อนใช้งาน
    // เพื่อป้องกัน Error 'additional_kwargs of undefined' จากข้อมูลเก่าที่เสีย
    let dbMessages = await messageHistory.getMessages()
    dbMessages = sanitizeMessages(dbMessages);

    const systemMessage = new SystemMessage(
      "You are a helpful AI assistant. You can check weather information."
    )

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash", 
      temperature: 0.7,
      maxOutputTokens: 8192,
      apiKey: process.env.GOOGLE_API_KEY,
    })
    
    const tools = [getWeatherTool]
    const modelWithTools = model.bindTools(tools)

    const currentUserMessage = new HumanMessage(userContent)

    const coreMessages: BaseMessage[] = [
      systemMessage,
      ...dbMessages,
      currentUserMessage
    ]

    // Invoke AI
    const aiResponse = await modelWithTools.invoke(coreMessages)

    await messageHistory.addMessage(currentUserMessage)

    // CASE A: No Tool Call
    if (!aiResponse.tool_calls || aiResponse.tool_calls.length === 0) {
      const stream = await modelWithTools.stream(coreMessages)
      
      const [streamForUser, streamForSave] = stream.tee();
      (async () => {
        const fullText = await consumeStream(streamForSave);
        await messageHistory.addMessage(new AIMessage(fullText))
      })();

      return createUIMessageStreamResponse({
        stream: toUIMessageStream(streamForUser),
        headers: { 'x-session-id': currentSessionId! },
      })
    }

    // CASE B: Tool Call Execution
    await messageHistory.addMessage(aiResponse)

    const toolObservations: ToolMessage[] = []

    for (const toolCall of aiResponse.tool_calls) {
      const selectedTool = tools.find((t) => t.name === toolCall.name)
      if (selectedTool) {
        
        const toolArgs = toolCall.args as Record<string, unknown>;
        const runner = selectedTool as unknown as GenericToolInvoker;
        const observationResult = await runner.invoke(toolArgs);
        
        // Ensure result is string
        const observation = typeof observationResult === 'string' 
          ? observationResult 
          : JSON.stringify(observationResult || {}); // Handle null/undefined result
        
        const toolMsg = new ToolMessage({
            content: observation,
            tool_call_id: toolCall.id!, // Gemini 2.5 usually provides ID
        });

        toolObservations.push(toolMsg)
        await messageHistory.addMessage(toolMsg)
      }
    }

    // Final Response Stream
    const finalStream = await modelWithTools.stream([
      ...coreMessages,
      aiResponse,
      ...toolObservations
    ])

    const [finalStreamForUser, finalStreamForSave] = finalStream.tee();
    (async () => {
        const fullText = await consumeStream(finalStreamForSave);
        await messageHistory.addMessage(new AIMessage(fullText))
    })();

    return createUIMessageStreamResponse({
      stream: toUIMessageStream(finalStreamForUser),
      headers: { 'x-session-id': currentSessionId! },
    })

  } catch (error) {
    console.error("API Error:", error)
    return new Response(
      JSON.stringify({
        error: "An error occurred",
        details: error instanceof Error ? error.message : 'Unknown error'
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    )
  }
}