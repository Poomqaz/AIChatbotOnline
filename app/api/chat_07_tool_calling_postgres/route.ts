import { NextRequest } from 'next/server'
import { getDatabase } from '@/lib/database'

// LangChain & Google GenAI
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { HumanMessage, AIMessage } from '@langchain/core/messages'
import { DynamicStructuredTool } from '@langchain/core/tools'
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// Vercel AI SDK (ตามที่คุณใช้งาน)
import { createUIMessageStreamResponse } from 'ai'
import { toUIMessageStream } from '@ai-sdk/langchain'

// Utils
import { z } from 'zod'
import { createClient } from '@supabase/supabase-js'
import { PostgresChatMessageHistory } from '@langchain/community/stores/message/postgres'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

// ===============================================
// Interfaces & Setup (เหมือนเดิม)
// ===============================================
interface MessagePart {
    type: 'text' | 'image-url';
    text?: string;
    image_url?: { url: string };
}

interface ClientMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system' | 'data';
  content: string; 
  parts?: MessagePart[]; 
}
const pool = getDatabase()
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_OR_ANON_KEY!
)

// ===============================================
// Tools (เหมือนเดิม)
// ===============================================
const getProductInfoTool = new DynamicStructuredTool({
    name: "get_product_info",
    description: "ค้นหาข้อมูลสินค้าจากฐานข้อมูล รวมถึงราคาและจำนวนคงคลัง (stock) โดยรับชื่อสินค้าเป็น input",
    schema: z.object({
      productName: z.string().describe("ชื่อของสินค้าที่ต้องการค้นหา เช่น 'Running Shoes', 'Earbuds', 'Keyboard' เป็นต้น"),
    }),
    func: async ({ productName }) => {
      console.log(`🔧 TOOL CALLED: get_product_info with productName="${productName}"`);
      try {
        // ตรวจสอบการเชื่อมต่อฐานข้อมูล
        const { data, error } = await supabase
          .from("products")
          .select("name, price, stock, description")
          .ilike("name", `%${productName}%`)
          .limit(10); // จำกัดผลลัพธ์ไม่เกิน 10 รายการ
          // .single(); // .single() จะคืนค่า object เดียว หรือ error ถ้าเจอหลายรายการ/ไม่เจอ
        
        if (error) {
          console.log('❌ Supabase error:', error.message);
          // ตรวจสอบว่าเป็น connection error หรือไม่
          if (error.message.includes('connection') || error.message.includes('network') || error.message.includes('timeout')) {
            throw new Error('DATABASE_CONNECTION_ERROR');
          }
          throw new Error(error.message);
        }
        
        if (!data || data.length === 0) {
          console.log(`❌ ไม่พบสินค้าที่ชื่อ '${productName}'`);
          return `ไม่พบสินค้าที่ชื่อ '${productName}' ในฐานข้อมูล`;
        }
        
        console.log('✅ พบข้อมูลสินค้า:', data);
        
        // หากพบหลายสินค้า ให้แสดงรายการทั้งหมด
        if (data.length === 1) {
          const product = data[0];
          return `ข้อมูลสินค้า "${product.name}":
- ราคา: ${product.price} บาท
- จำนวนในสต็อก: ${product.stock} ชิ้น
- รายละเอียด: ${product.description}`;
        } else {
          // แสดงรายการสินค้าทั้งหมดที่พบในรูปแบบตาราง Markdown
          const tableHeader = `| ชื่อสินค้า | ราคา (บาท) | สต็อก (ชิ้น) | รายละเอียด |
|----------|------------|-------------|------------|`;
          
          const tableRows = data.map(product => 
            `| ${product.name} | ${product.price.toLocaleString()} | ${product.stock} | ${product.description} |`
          ).join('\n');
          
          return `พบสินค้าที่ตรงกับคำค้นหา "${productName}" ทั้งหมด ${data.length} รายการ:

${tableHeader}
${tableRows}`;
        }
      } catch (e: unknown) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        console.log('❌ Tool error:', errorMessage);
        
        // ตรวจสอบว่าเป็น database connection error หรือไม่
        if (errorMessage === 'DATABASE_CONNECTION_ERROR' || 
            errorMessage.includes('connection') || 
            errorMessage.includes('network') || 
            errorMessage.includes('timeout')) {
          throw new Error('DATABASE_CONNECTION_ERROR');
        }
        
        return `เกิดข้อผิดพลาดในการดึงข้อมูลสินค้า: ${errorMessage}`;
      }
    },
})

// สร้าง Tool สำหรับดูข้อมูลการขาย
const getSalesDataTool = new DynamicStructuredTool({
    name: "get_sales_data",
    description: "ใช้ tool นี้เพื่อดูประวัติการขายของสินค้า. รับ input เป็นชื่อสินค้า.",
    schema: z.object({
      productName: z.string().describe("ชื่อของสินค้าที่ต้องการดูข้อมูลการขาย"),
    }),
    func: async ({ productName }) => {
      console.log(`TOOL CALLED: get_sales_data with productName=${productName}`);
      try {
        // ขั้นตอนที่ 1: ค้นหา product_id จากชื่อสินค้า
        const { data: product, error: productError } = await supabase
          .from("products").select("id").ilike("name", `%${productName}%`).single();
        if (productError) {
          // ตรวจสอบว่าเป็น connection error หรือไม่
          if (productError.message.includes('connection') || productError.message.includes('network') || productError.message.includes('timeout')) {
            throw new Error('DATABASE_CONNECTION_ERROR');
          }
          throw new Error(productError.message);
        }
        if (!product) return `ไม่พบสินค้าที่ชื่อ '${productName}'`;
        
        // ขั้นตอนที่ 2: ดึงข้อมูลการขายจาก sales table โดยใช้ product_id
        const { data: sales, error: salesError } = await supabase
          .from("sales").select("sale_date, quantity_sold, total_price").eq("product_id", product.id);
        if (salesError) {
          // ตรวจสอบว่าเป็น connection error หรือไม่
          if (salesError.message.includes('connection') || salesError.message.includes('network') || salesError.message.includes('timeout')) {
            throw new Error('DATABASE_CONNECTION_ERROR');
          }
          throw new Error(salesError.message);
        }
        if (!sales || sales.length === 0) return `ยังไม่มีข้อมูลการขายสำหรับสินค้า '${productName}'`;
        
        // หากมีรายการเดียว แสดงแบบง่าย
        if (sales.length === 1) {
          const sale = sales[0];
          return `ประวัติการขายของสินค้า "${productName}":
                  - วันที่ขาย: ${new Date(sale.sale_date).toLocaleDateString('th-TH')}
                  - จำนวนที่ขาย: ${sale.quantity_sold} ชิ้น
                  - ยอดขาย: ${sale.total_price.toLocaleString()} บาท`;
        } else {
          // หากมีหลายรายการ แสดงเป็นตาราง Markdown
          const tableHeader = `| วันที่ขาย | จำนวนที่ขาย (ชิ้น) | ยอดขาย (บาท) |
|-----------|-------------------|---------------|`;
          
          const tableRows = sales.map(sale => 
            `| ${new Date(sale.sale_date).toLocaleDateString('th-TH')} | ${sale.quantity_sold} | ${sale.total_price.toLocaleString()} |`
          ).join('\n');
          
          const totalQuantity = sales.reduce((sum, sale) => sum + sale.quantity_sold, 0);
          const totalSales = sales.reduce((sum, sale) => sum + parseFloat(sale.total_price), 0);
          
          return `ประวัติการขายของสินค้า "${productName}" ทั้งหมด ${sales.length} รายการ:

${tableHeader}
${tableRows}

**สรุป:**
- ขายรวม: ${totalQuantity} ชิ้น
- ยอดขายรวม: ${totalSales.toLocaleString()} บาท`;
        }
      } catch (e: unknown) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        
        // ตรวจสอบว่าเป็น database connection error หรือไม่
        if (errorMessage === 'DATABASE_CONNECTION_ERROR' || 
            errorMessage.includes('connection') || 
            errorMessage.includes('network') || 
            errorMessage.includes('timeout')) {
          throw new Error('DATABASE_CONNECTION_ERROR');
        }
        
        return `เกิดข้อผิดพลาดในการดึงข้อมูลการขาย: ${errorMessage}`;
      }
    },
})


const tools = [getProductInfoTool, getSalesDataTool];

// ===============================================
// POST API
// ===============================================
export async function POST(req: NextRequest) {
  try {
    const { messages, sessionId } = await req.json()

    // 1. Session & History
    const targetSessionId = sessionId || `session-${Date.now()}`; 
    const messageHistory = new PostgresChatMessageHistory({
      sessionId: targetSessionId,
      tableName: "chat_messages",
      pool: getDatabase(),
    })

    // 2. Prepare Input
    const lastUserMessage = messages.findLast((m: ClientMessage) => m.role === 'user');
    let inputContent = lastUserMessage?.content || "";
    // Handle parts if exists
    if (lastUserMessage?.parts?.length) {
        const textPart = lastUserMessage.parts.find((p: MessagePart) => p.type === 'text');
        if (textPart) inputContent = textPart.text;
    }

    if (!inputContent) return new Response("No input", { status: 400 });

    console.log(`🚀 User: ${inputContent}`);
    await messageHistory.addMessage(new HumanMessage(inputContent));

    // 3. Setup Agent
    const model = new ChatGoogleGenerativeAI({
      model: process.env.OPENAI_MODEL || 'gemini-2.5-flash',
      temperature: 0,
      streaming: true,
    })

    const agent = createReactAgent({
      llm: model,
      tools: tools,
      messageModifier: "You are a helpful assistant. Always Answer in Thai.",
    });

    const historyMessages = await messageHistory.getMessages();

    // 4. Create Custom Stream
    // เราสร้าง ReadableStream ขึ้นมาเอง เพื่อ "กรอง" เอาเฉพาะ Text จาก Agent
    // และส่งให้ toUIMessageStream จัดการต่อ
    const customTextStream = new ReadableStream({
      async start(controller) {
        let fullAiResponse = "";

        try {
          // เรียก Agent แบบ Stream Events
          const eventStream = await agent.streamEvents(
            { messages: historyMessages },
            { version: "v2" }
          );

          for await (const { event, data } of eventStream) {
            // จับเฉพาะตอนที่ Model พ่น Text ออกมา (ไม่ใช่ตอนเรียก Tool)
            if (event === "on_chat_model_stream") {
                const chunk = data.chunk;
                
                // เช็คว่าเป็น Text Content
                if (chunk.content && typeof chunk.content === 'string' && chunk.content.length > 0) {
                    
                    // 1. ส่งเข้า Stream (ให้ Frontend)
                    controller.enqueue(chunk.content);
                    
                    // 2. เก็บใส่ตัวแปร (ให้ Database)
                    fullAiResponse += chunk.content;
                }
            }
          }
          
          // ปิด Stream เมื่อเสร็จ
          controller.close();

          // ✅ Save AI Response to DB (ทำหลังจาก Stream จบใน Background)
          if (fullAiResponse) {
             console.log("💾 Saving AI Response...");
             // ไม่ใช้ await เพื่อไม่ให้ block การปิด stream หรือใส่ try-catch แยก
             messageHistory.addMessage(new AIMessage(fullAiResponse)).catch(err => {
                 console.error("Failed to save history:", err);
             });
          }

        } catch (e) {
          console.error("❌ Stream Error:", e);
          controller.error(e);
        }
      },
    });

    // 5. Return Response using AI SDK
    // ใช้ createUIMessageStreamResponse ตาม SDK ของคุณ
    // โดยส่ง customTextStream ที่เราสร้างขึ้นเองเข้าไป
    return createUIMessageStreamResponse({
      stream: toUIMessageStream(customTextStream),
      headers: {
        'x-session-id': targetSessionId
      }
    });

  } catch (error: unknown) {
    console.error("API Error:", error)
    return new Response(JSON.stringify({ error: "Server Error" }), { status: 500 })
  }
}

// GET API (Code เดิม)
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