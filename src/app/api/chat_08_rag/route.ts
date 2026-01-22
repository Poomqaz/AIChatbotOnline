/**
 * ===============================================
 * API Route สำหรับ Chat (Gemini RAG Version)
 * ===============================================
 */

import { NextRequest } from 'next/server'
import { getDatabase } from '@/src/lib/database'

// LangChain & AI SDK Imports
// 🔄 CHANGED: เปลี่ยนจาก OpenAI เป็น Google GenAI
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts'
import { toUIMessageStream } from '@ai-sdk/langchain'
import { createUIMessageStreamResponse, UIMessage } from 'ai'
import { PostgresChatMessageHistory } from '@langchain/community/stores/message/postgres'
import { BaseMessage, AIMessage, HumanMessage, SystemMessage, MessageContent } from '@langchain/core/messages'
import { trimMessages } from '@langchain/core/messages'
import { StringOutputParser } from '@langchain/core/output_parsers'
// หมายเหตุ: tiktoken ใช้สำหรับนับ token ของ OpenAI แต่เรายังใช้เพื่อประมาณการความยาวข้อความได้
import { encodingForModel } from '@langchain/core/utils/tiktoken' 
import { createClient } from '@supabase/supabase-js'
// เพิ่มบรรทัดนี้ในส่วน Import
import { HarmCategory, HarmBlockThreshold } from "@google/generative-ai";

// Imports for Vector Search (Document RAG)
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase"
import { CacheBackedEmbeddings } from "@langchain/classic/embeddings/cache_backed";
import { InMemoryStore } from "@langchain/core/stores"
import { TaskType } from "@google/generative-ai";

export const dynamic = 'force-dynamic'
export const maxDuration = 30

const pool = getDatabase()

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_OR_ANON_KEY!
)

// ===============================================
// 🔄 MODIFIED: สร้าง Vector Store ด้วย Google Embeddings
// ===============================================
async function createVectorStore() {
  // ใช้ Google Embeddings (text-embedding-004)
  const baseEmbeddings = new GoogleGenerativeAIEmbeddings({
    model: process.env.GOOGLE_EMBEDDING_MODEL_NAME || "text-embedding-004",
    taskType: TaskType.RETRIEVAL_DOCUMENT, // ระบุ Task Type เพื่อความแม่นยำ
    // Google embeddings ปกติจะเป็น 768 dimensions 
    // หาก Table ใน Supabase ตั้งไว้ 1536 อาจต้องแก้ Database หรือ Re-create table
  });

  const cacheStore = new InMemoryStore();
  const embeddings = CacheBackedEmbeddings.fromBytesStore(
    baseEmbeddings,
    cacheStore,
    {
      namespace: "rag_embeddings_gemini" // เปลี่ยน namespace เล็กน้อยเพื่อไม่ให้ตีกับ cache เดิม
    }
  );

  return new SupabaseVectorStore(embeddings, {
    client: supabase,
    tableName: 'documents',
    queryName: 'match_documents'
  });
}

// ===============================================
// ฟังก์ชันสำหรับ RAG (Vector Search)
// ===============================================
async function searchDocuments(query: string, limit: number = 5) {
  try {
    console.log(`🔧 Searching documents (Gemini) with query="${query}", limit=${limit}`);
    
    const vectorStore = await createVectorStore();
    
    const results = await vectorStore.similaritySearchWithScore(query, limit);
    
    if (!results || results.length === 0) {
      return `ไม่พบเอกสารที่เกี่ยวข้องกับ "${query}" ในระบบ`;
    }
    
    console.log(`✅ พบเอกสารที่เกี่ยวข้อง: ${results.length} รายการ`);
    
    const documents = results.map(([doc, score]) => {
      const filename = doc.metadata?.filename || 'ไม่ทราบชื่อไฟล์';
      const type = doc.metadata?.type || 'ไม่ทราบประเภท';
      return `ไฟล์: ${filename} (${type.toUpperCase()})
เนื้อหา: ${doc.pageContent}
ความเกี่ยวข้อง: ${(score * 100).toFixed(1)}%`; // Note: Score ของ Google อาจต่าง scale กับ OpenAI
    }).join('\n\n---\n\n');
    
    return documents;
  } catch (e: unknown) {
    const errorMessage = e instanceof Error ? e.message : String(e);
    console.log('❌ Search error:', errorMessage);
    
    if (errorMessage.includes('connection') || 
        errorMessage.includes('network') || 
        errorMessage.includes('timeout')) {
      throw new Error('ไม่สามารถเข้าถึงระบบค้นหาเอกสารได้ในขณะนี้');
    }
    
    throw new Error(`เกิดข้อผิดพลาดในการค้นหาเอกสาร: ${errorMessage}`);
  }
}

// ===============================================
// ฟังก์ชันสำหรับนับ Token (Tiktoken Proxy)
// ===============================================
// หมายเหตุ: แม้เราใช้ Gemini แต่การใช้ Tiktoken (ของ GPT-4) 
// ยังคงเป็นวิธีที่ดีและเร็วในการประมาณความยาวข้อความเพื่อทำ Trimming
// เพราะเราแค่ต้องการตัดข้อความไม่ให้ยาวเกินไป ไม่ต้องเป๊ะ 100% ตามโมเดล Gemini

type Encoding = {
  encode: (text: string) => number[]
  free?: () => void
}

let encPromise: Promise<Encoding> | undefined

async function getEncoder(): Promise<Encoding> {
  if (!encPromise) {
    // ใช้ gpt-4o เป็น standard reference สำหรับความยาว
    encPromise = encodingForModel("gpt-4o").catch(() =>
      encodingForModel("gpt-4")
    )
  }
  return encPromise
}

async function strTokenCounter(content: MessageContent): Promise<number> {
  const enc = await getEncoder()
  if (typeof content === 'string') return enc.encode(content).length
  if (Array.isArray(content)) {
    return enc.encode(
      content.map(p => (p.type === 'text' ? p.text : JSON.stringify(p))).join(' ')
    ).length
  }
  return enc.encode(String(content ?? '')).length
}

async function tiktokenCounter(messages: BaseMessage[]): Promise<number> {
  let total = 0
  for (const m of messages) {
    const role =
      m instanceof HumanMessage ? 'user'
        : m instanceof AIMessage ? 'assistant'
        : m instanceof SystemMessage ? 'system'
        : 'unknown'
    total += await strTokenCounter(role)
    total += await strTokenCounter(m.content)
  }
  return total
}

// ===============================================
// POST API: ส่งข้อความและรับการตอบกลับแบบ Stream
// ===============================================
export async function POST(req: NextRequest) {
  try {
    const { messages, sessionId, userId }: {
      messages: UIMessage[]
      sessionId?: string
      userId?: string
    } = await req.json()

    // --- Step 2: Session Management (เหมือนเดิม) ---
    let currentSessionId = sessionId
    if (!currentSessionId) {
      const client = await pool.connect()
      try {
        const firstMessage = messages.find(m => m.role === 'user')
        let title = 'New Chat'
        if (firstMessage && Array.isArray(firstMessage.parts) && firstMessage.parts.length > 0) {
          const textPart = firstMessage.parts.find(p => p.type === 'text')
          if (textPart && typeof textPart.text === 'string') {
            title = textPart.text.slice(0, 50) + (textPart.text.length > 50 ? '...' : '')
          }
        }
        
        if (!userId) throw new Error('User ID is required')
        const result = await client.query(
          'INSERT INTO chat_sessions (title, user_id) VALUES ($1, $2) RETURNING id',
          [title, userId]
        )
        currentSessionId = result.rows[0].id
      } finally {
        client.release()
      }
    }

    // --- Step 3: Load Summary (เหมือนเดิม) ---
    const clientForSummary = await pool.connect()
    let persistedSummary = ''
    try {
      const r = await clientForSummary.query(
        'SELECT summary FROM chat_sessions WHERE id = $1 LIMIT 1',
        [currentSessionId]
      )
      persistedSummary = r.rows?.[0]?.summary ?? ''
    } finally {
      clientForSummary.release()
    }

    // ===============================================
    // 🔄 MODIFIED Step 4: ตั้งค่า AI Model (Gemini)
    // ===============================================
    const model = new ChatGoogleGenerativeAI({
      model: process.env.GOOGLE_MODEL_NAME || "gemini-2.5-flash",
      temperature: 0.1,
      maxOutputTokens: 8192, // Gemini รองรับ Output ยาวกว่า
      streaming: true,
        apiKey: process.env.GOOGLE_API_KEY,
        safetySettings: [
            {
            category: HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold: HarmBlockThreshold.BLOCK_NONE,
            },
            {
            category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold: HarmBlockThreshold.BLOCK_NONE,
            },
            {
            category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold: HarmBlockThreshold.BLOCK_NONE,
            },
            {
            category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold: HarmBlockThreshold.BLOCK_NONE,
            },
        ],
    })

    // --- Step 5: History (เหมือนเดิม) ---
    const messageHistory = new PostgresChatMessageHistory({
      sessionId: currentSessionId!,
      tableName: 'chat_messages',
      pool: pool
    })

    const fullHistory = await messageHistory.getMessages()
    
    // --- Step 6: Last User Message (เหมือนเดิม) ---
    const lastUserMessage = messages.filter(m => m.role === 'user').pop()
    let input = ''
    if (lastUserMessage && Array.isArray(lastUserMessage.parts) && lastUserMessage.parts.length > 0) {
      const textPart = lastUserMessage.parts.find(p => p.type === 'text')
      if (textPart) input = textPart.text
    }
    if (!input) return new Response('No valid user input found.', { status: 400 })

    // --- Step 7: Trimming & Summary (เหมือนเดิม) ---
    let recentWindowWithoutCurrentInput: BaseMessage[] = []
    let overflowSummary = ''
    
    if (sessionId && fullHistory.length > 0) {
      // ใช้ tiktokenCounter (ที่เป็น proxy) เพื่อตัดข้อความ
      const trimmedWindow = await trimMessages(fullHistory, {
        maxTokens: 3000, // Gemini Flash รับ Context ได้เยอะ เพิ่ม limit ได้
        strategy: 'last',
        tokenCounter: tiktokenCounter
      })

      recentWindowWithoutCurrentInput = trimmedWindow.filter(msg => {
        if (msg instanceof HumanMessage && msg.content === input) return false
        return true
      })

      const windowSet = new Set(trimmedWindow)
      const overflow = fullHistory.filter(m => !windowSet.has(m))
      if (overflow.length > 0) {
        const summarizerPrompt = ChatPromptTemplate.fromMessages([
          ['system', 'สรุปบทสนทนาให้สั้นที่สุด เป็นภาษาไทย เก็บเฉพาะสาระสำคัญ'],
          ['human', 'สรุปข้อความต่อไปนี้:\n\n{history}']
        ])
        const summarizer = summarizerPrompt.pipe(model).pipe(new StringOutputParser())
        const historyText = overflow
          .map(m => {
            if (m instanceof HumanMessage) return `ผู้ใช้: ${m.content}`
            if (m instanceof AIMessage) return `ผู้ช่วย: ${m.content}`
            return `ระบบ: ${String(m.content)}`
          })
          .join('\n')
        try {
          overflowSummary = await summarizer.invoke({ history: historyText })
        } catch (e) {
          console.warn('overflow summary failed', e)
        }
      }
    }

    const summaryForThisTurn = [persistedSummary, overflowSummary].filter(Boolean).join('\n')

    // ===============================================
    // Step 8: RAG Chain (Prompt เดิม ใช้ได้เลย)
    // ===============================================
    const ragPrompt = ChatPromptTemplate.fromMessages([
      ['system', `คุณคือผู้ช่วย AI อัจฉริยะที่ตอบเป็นภาษาไทย 
      
      คุณมีข้อมูลจากเอกสารที่อัปโหลดไว้ในระบบ เพื่อใช้ตอบคำถาม
      
      **หลักการตอบคำถาม:**
      - ใช้ข้อมูลจากเอกสารที่ให้มาในการตอบคำถาม
      - หากไม่มีข้อมูลที่เกี่ยวข้อง ให้บอกว่าไม่พบข้อมูลที่เกี่ยวข้อง
      - ห้ามเดาหรือสร้างข้อมูลขึ้นมาเอง ให้ใช้ข้อมูลจากเอกสารเท่านั้น
      - ตอบด้วยข้อมูลที่ถูกต้องและครบถ้วน
      
      บริบทการสนทนาก่อนหน้านี้โดยสรุปคือ: {summary}
      
      ข้อมูลจากเอกสารที่เกี่ยวข้อง:
      {context}`],
      new MessagesPlaceholder('chat_history'),
      ['human', '{input}']
    ])

    const ragChain = ragPrompt.pipe(model).pipe(new StringOutputParser())

    // --- Step 9: Search & Stream ---
    let documentContext = '';
    try {
      documentContext = await searchDocuments(input, 3);
    } catch (error) {
      console.warn('⚠️ Search Error:', error instanceof Error ? error.message : String(error));
      documentContext = 'ไม่สามารถเข้าถึงระบบค้นหาเอกสารได้ในขณะนี้';
    }

    const chatHistoryForChain = [...recentWindowWithoutCurrentInput];
    // if (summaryForThisTurn) {
    //     chatHistoryForChain.unshift(new SystemMessage(summaryForThisTurn));
    // }

    const stream = await ragChain.stream({
        input: input,
        chat_history: chatHistoryForChain,
        summary: summaryForThisTurn,
        context: documentContext
    });

    // --- Step 10: Save User Message ---
    let canSaveToDatabase = true
    try {
      await messageHistory.addUserMessage(input)
    } catch (e) {
      console.warn('⚠️ Save user msg failed:', e)
      canSaveToDatabase = false
    }
    
    // --- Step 11: Handle Stream ---
    let assistantText = ''
    let hasSearchError = false
    
    const readable = new ReadableStream({
      async start(controller) {
        try {
        console.log('🚀 Start Streaming response...');
          for await (const chunk of stream) {
            if (typeof chunk === 'string') {
              assistantText += chunk;
              if (chunk.includes('ไม่สามารถเข้าถึงระบบค้นหาเอกสารได้') || 
                  assistantText.includes('ไม่สามารถเข้าถึงระบบค้นหาเอกสารได้')) {
                hasSearchError = true;
                const friendlyMessage = 'ขออภัยครับ ขณะนี้ไม่สามารถเข้าถึงระบบค้นหาเอกสารได้ กรุณาลองใหม่อีกครั้งในภายหลัง';
                controller.enqueue(friendlyMessage);
                assistantText = friendlyMessage;
              } else {
                controller.enqueue(chunk);
              }
            }
          }

          if (!assistantText) {
             console.warn('⚠️ จบ Stream แต่ไม่มีข้อความ (Empty Response)');
          } else {
             console.log('✅ Stream Finished. Total length:', assistantText.length);
          }
          
          // --- Step 12 & 13: Save AI Message & Update Summary ---
          if (assistantText && !hasSearchError && canSaveToDatabase) {
            try {
              await messageHistory.addMessage(new AIMessage(assistantText))
              
              const summarizerPrompt2 = ChatPromptTemplate.fromMessages([
                ['system', 'รวมสาระสำคัญให้สั้นที่สุด ภาษาไทย กระชับ'],
                ['human', 'นี่คือสรุปเดิม:\n{old}\n\nนี่คือข้อความใหม่:\n{delta}\n\nช่วยอัปเดตให้สั้นและครบถ้วน']
              ])
              const summarizer2 = summarizerPrompt2.pipe(model).pipe(new StringOutputParser())
              const updatedSummary = await summarizer2.invoke({
                old: persistedSummary || 'ไม่มีประวัติก่อนหน้า',
                delta: [overflowSummary, `ผู้ใช้: ${input}`, `ผู้ช่วย: ${assistantText}`].filter(Boolean).join('\n')
              })
              
              const clientUpdate = await pool.connect()
              try {
                await clientUpdate.query(
                  'UPDATE chat_sessions SET summary = $1 WHERE id = $2',
                  [updatedSummary, currentSessionId]
                )
              } finally {
                clientUpdate.release()
              }
            } catch (e) {
              console.warn('Update summary failed', e)
            }
          }
          
          controller.close()
        } catch (error) {
          controller.error(error)
        }
      }
    })

    // --- Step 14: Response ---
    return createUIMessageStreamResponse({
      stream: toUIMessageStream(readable),
      headers: currentSessionId ? { 'x-session-id': currentSessionId } : undefined
    })
  } catch (error) {
    console.error('API Error:', error)
    return new Response(
      JSON.stringify({
        error: 'An error occurred',
        details: error instanceof Error ? error.message : 'Unknown error'
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}

// ===============================================
// GET API: (คงเดิม ไม่ต้องแก้)
// ===============================================
export async function GET(req: NextRequest) {
  try {
    // ===============================================
    // Step 1: ตรวจสอบ Session ID จาก URL Parameters
    // ===============================================
    const { searchParams } = new URL(req.url)
    const sessionId = searchParams.get('sessionId')
    if (!sessionId) {
      return new Response(
        JSON.stringify({ error: 'Session ID is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // ===============================================
    // Step 2: Query ข้อมูลประวัติการสนทนาจากฐานข้อมูล
    // ===============================================
    const client = await pool.connect()
    try {
      const result = await client.query(
        `SELECT message, message->>'type' as message_type, created_at
         FROM chat_messages 
         WHERE session_id = $1 
         ORDER BY created_at ASC`,
        [sessionId]
      )
      
      // ===============================================
      // Step 3: แปลงข้อมูลให้อยู่ในรูปแบบที่ UI ต้องการ
      // ===============================================
      const messages = result.rows.map((row, i) => {
        const data = row.message
        let role = 'user'
        if (row.message_type === 'ai') role = 'assistant'
        else if (row.message_type === 'human') role = 'user'
        return {
          id: `history-${i}`,
          role,
          content: data.content || data.text || data.message || '',
          createdAt: row.created_at
        }
      })
      
      // ===============================================
      // Step 4: ส่งข้อมูลกลับ
      // ===============================================
      return new Response(JSON.stringify({ messages }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      })
    } finally {
      client.release()
    }
  } catch (error) {
    console.error('Error fetching messages:', error)
    return new Response(
      JSON.stringify({
        error: 'Failed to fetch messages',
        details: error instanceof Error ? error.message : 'Unknown error'
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}