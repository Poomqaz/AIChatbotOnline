/**
 * ===============================================
 * API Route สำหรับ Chat (Fixed for Gemini Strict Rules & Linting)
 * ===============================================
 */

import { NextRequest } from 'next/server'
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts'
import { toUIMessageStream } from '@ai-sdk/langchain'
import { createUIMessageStreamResponse, UIMessage } from 'ai'
import { PostgresChatMessageHistory } from '@langchain/community/stores/message/postgres'
import { BaseMessage, AIMessage, HumanMessage, SystemMessage, MessageContent } from '@langchain/core/messages'
import { trimMessages } from '@langchain/core/messages'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { encodingForModel } from '@langchain/core/utils/tiktoken'
import { getDatabase } from '@/lib/database'

export const dynamic = 'force-dynamic'
export const maxDuration = 30

const pool = getDatabase()

// ... (ส่วนของ Helper Function นับ Token) ...
type Encoding = {
  encode: (text: string) => number[]
  free?: () => void
}

let encPromise: Promise<Encoding> | undefined

async function getEncoder(): Promise<Encoding> {
  if (!encPromise) {
    encPromise = encodingForModel("gpt-4o-mini").catch(() =>
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
      m instanceof HumanMessage
        ? 'user'
        : m instanceof AIMessage
        ? 'assistant'
        : m instanceof SystemMessage
        ? 'system'
        : 'unknown'
    total += await strTokenCounter(role)
    total += await strTokenCounter(m.content)
  }
  return total
}

// ===============================================
// POST API
// ===============================================
export async function POST(req: NextRequest) {
  try {
    const { messages, sessionId, userId }: {
      messages: UIMessage[]
      sessionId?: string
      userId?: string
    } = await req.json()

    // Map ข้อความ UI -> LangChain
    const mapUIMessagesToLangChainMessages = (messages: UIMessage[]): BaseMessage[] => {
      return messages.map(msg => {
        const content = msg.parts?.find(p => p.type === 'text')?.text ?? '';
        if (msg.role === 'user') return new HumanMessage(content);
        if (msg.role === 'assistant') return new AIMessage(content);
        return new HumanMessage(content);
      });
    };

    const isNewSession = !sessionId;
    let currentSessionId = sessionId;

    // Step 1: Session Management
    if (isNewSession) {
      if (!userId) throw new Error('User ID is required for new sessions');
      currentSessionId = await createNewSession(userId, messages);
    }
    
    if (!currentSessionId) throw new Error("Failed to create session ID");

    // Step 2: ดึงประวัติจาก DB
    let persistedSummary = '';
    let fullHistory: BaseMessage[] = [];

    const [summaryResult, historyResult] = await Promise.all([
      pool.query('SELECT summary FROM chat_sessions WHERE id = $1 LIMIT 1', [currentSessionId]),
      new PostgresChatMessageHistory({ sessionId: currentSessionId, tableName: 'chat_messages', pool }).getMessages()
    ]);
    persistedSummary = summaryResult.rows?.[0]?.summary ?? '';
    const dbHistory = historyResult;

    if (isNewSession) {
      fullHistory = mapUIMessagesToLangChainMessages(messages);
    } else {
      const newMessages = mapUIMessagesToLangChainMessages(messages);
      const latestUserMessage = newMessages.filter(m => m instanceof HumanMessage).pop();
      if (latestUserMessage) {
        fullHistory = [...dbHistory, latestUserMessage];
      } else {
        fullHistory = dbHistory;
      }
    }
    
    // Step 3: ตั้งค่า AI Model (GEMINI)
    const model = new ChatGoogleGenerativeAI({
      model: process.env.GOOGLE_MODEL || 'gemini-2.5-flash',
      temperature: 0.7,
      maxOutputTokens: 8192,
      apiKey: process.env.GOOGLE_API_KEY
    });
    
    const lastUserMessage = fullHistory.filter(m => m instanceof HumanMessage).pop();
    const input = lastUserMessage?.content.toString() ?? '';
    if (!input) return new Response('No valid user input found.', { status: 400 });

    const historyWithoutLastInput = lastUserMessage 
      ? fullHistory.slice(0, fullHistory.lastIndexOf(lastUserMessage))
      : fullHistory;

    // Step 4: Trim ประวัติแชท
    let recentWindow = historyWithoutLastInput.length > 0
      ? await trimMessages(historyWithoutLastInput, { 
          maxTokens: 10000, 
          strategy: 'last', 
          tokenCounter: tiktokenCounter 
        })
      : [];

    // [FIX 1]: Gemini บังคับว่า History ต้องเริ่มด้วย Human Message เท่านั้น
    if (recentWindow.length > 0 && recentWindow[0] instanceof AIMessage) {
        recentWindow = recentWindow.slice(1);
    }
    
    // [FIX 2]: รวม System Message เป็นก้อนเดียว
    const systemInstruction = `คุณคือผู้ช่วยที่ตอบชัดเจน และตอบเป็นภาษาไทยเมื่อผู้ใช้ถามเป็นไทย
    
บริบทก่อนหน้า (สรุป): ${persistedSummary || 'ไม่มีบริบทก่อนหน้า'}`;

    const prompt = ChatPromptTemplate.fromMessages([
      ['system', systemInstruction],
      new MessagesPlaceholder('recent_window'),
      ['human', '{input}']
    ]);

    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    // Step 6: Stream และบันทึก
    let assistantText = '';
    const messageHistory = new PostgresChatMessageHistory({
      sessionId: currentSessionId,
      tableName: 'chat_messages',
      pool: pool,
    });
    
    const stream = await chain.stream({
      input,
      recent_window: recentWindow
    });

    const responseStream = new ReadableStream<string>({
      async start(controller) {
        try {
          for await (const chunk of stream) {
            if (chunk && chunk.length > 0) {
                assistantText += chunk;
                controller.enqueue(chunk);
            }
          }
          
          if (assistantText) {
            try {
              // 1. บันทึกข้อความลง DB
              await messageHistory.addMessages([
                  new HumanMessage(input),
                  new AIMessage(assistantText)
              ]);

              // 2. สร้าง Summary (Background task)
              const allHistoryForSummary = [
                ...dbHistory.map(m => formatMessageForSummary(m)),
                `ผู้ใช้: ${input}`,
                `ผู้ช่วย: ${assistantText}`
              ].join('\n');
              
              await updateSessionSummary(currentSessionId!, persistedSummary, allHistoryForSummary);

            } catch (bgError) {
              console.error("❌ Database save error:", bgError);
            }
          }

          controller.close();
          
        } catch (error) {
          console.error("❌ Stream error:", error);
          controller.error(error);
        }
      }
    });

    return createUIMessageStreamResponse({
      stream: toUIMessageStream(responseStream),
      headers: { 'x-session-id': currentSessionId },
    });

  } catch (error) {
    console.error('API Error:', error);
    return new Response(JSON.stringify({ 
        error: 'An error occurred processing your request',
        details: error instanceof Error ? error.message : String(error)
    }), { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
    });
  }
}

// ===============================================
// Helper Functions
// ===============================================
async function createNewSession(userId: string, messages: UIMessage[]): Promise<string> {
  const client = await pool.connect();
  try {
    const firstMessage = messages.find(m => m.role === 'user');
    let title = 'New Chat';
    if (firstMessage?.parts?.[0]?.type === 'text') {
      title = firstMessage.parts[0].text.slice(0, 50);
    }
    const sessionResult = await client.query(
      'INSERT INTO chat_sessions (title, user_id) VALUES ($1, $2) RETURNING id',
      [title, userId]
    );
    return sessionResult.rows[0].id;
  } catch (error) {
    console.error("❌ Error in createNewSession:", error);
    throw error;
  } finally {
    client.release();
  }
}

async function updateSessionSummary(sessionId: string, oldSummary: string, allHistory: string) {
  try {
    const model = new ChatGoogleGenerativeAI({ 
        model: process.env.GOOGLE_MODEL || 'gemini-2.5-flash',
        apiKey: process.env.GOOGLE_API_KEY
    });
    
    const summarizerPrompt = ChatPromptTemplate.fromMessages([
      ['system', 'สร้างสรุปสั้นๆ ของการสนทนาทั้งหมด ให้ครอบคลุมหัวข้อหลักและประเด็นสำคัญ ใช้ภาษาไทย กระชับ ไม่เกิน 200 คำ'],
      ['human', 'ประวัติการสนทนาทั้งหมด:\n{history}\n\nช่วยสรุปสาระสำคัญของการสนทนานี้']
    ]);
    
    const summarizer = summarizerPrompt.pipe(model).pipe(new StringOutputParser());
    const updatedSummary = await summarizer.invoke({
      history: allHistory,
    });
    
    await pool.query(
      'UPDATE chat_sessions SET summary = $1 WHERE id = $2',
      [updatedSummary, sessionId]
    );
  } catch (e) {
    console.error(`❌ Failed to update summary for session ${sessionId}:`, e);
  }
}

function formatMessageForSummary(m: BaseMessage): string {
    if (m instanceof HumanMessage) return `ผู้ใช้: ${m.content}`;
    if (m instanceof AIMessage) return `ผู้ช่วย: ${m.content}`;
    return `ระบบ: ${String(m.content)}`;
}

// ===============================================
// GET API (Fixed: Added error usage)
// ===============================================
export async function GET(req: NextRequest) {
    try {
      const { searchParams } = new URL(req.url)
      const sessionId = searchParams.get('sessionId')
      if (!sessionId) {
        return new Response(JSON.stringify({ error: 'Session ID is required' }), { status: 400 })
      }
  
      const client = await pool.connect()
      try {
        const result = await client.query(
          `SELECT message, message->>'type' as message_type, created_at
           FROM chat_messages 
           WHERE session_id = $1 
           ORDER BY created_at ASC`,
          [sessionId]
        )
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
        return new Response(JSON.stringify({ messages }), { status: 200 })
      } finally {
        client.release()
      }
    } catch (error) {
      // FIX: นำตัวแปร error มาใช้งาน (Log) เพื่อแก้ Warning
      console.error('GET History Error:', error);
      return new Response(JSON.stringify({ error: 'Failed to fetch history' }), { status: 500 })
    }
}