/**
 * ===============================================
 * API Route สำหรับ Chat (RAG + Agent with Tools Calling)
 * ===============================================
 *
 * ฟีเจอร์หลัก:
 * - 📚 RAG (Retrieval-Augmented Generation) with pgvector
 * - 🤖 Agent with Tool Calling (Supabase + Vector Search) - ใช้ LangGraph
 * - 🗂️ เก็บประวัติการสนทนาใน PostgreSQL
 * - 🧠 ทำ Summary เพื่อประหยัด Token
 * - ✂️ Trim Messages เพื่อไม่ให้เกิน Token Limit
 * - 🌊 Streaming Response สำหรับ Real-time Chat
 * - 🔧 จัดการ Session ID อัตโนมัติ
 * 
 * Tools ที่มีให้ใช้งาน:
 * 1. search_documents - ค้นหาข้อมูลจากเอกสาร (PDF, CSV, TXT) ด้วย Vector Similarity
 * 2. get_product_info - ค้นหาข้อมูลสินค้าจากฐานข้อมูล
 * 3. get_sales_data - ดูประวัติการขาย
*/

import { NextRequest } from 'next/server'
import { getDatabase } from '@/src/lib/database'

// LangChain & AI SDK Imports
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { UIMessage } from 'ai'
import { PostgresChatMessageHistory } from '@langchain/community/stores/message/postgres'
import { z } from 'zod'
import { createClient } from '@supabase/supabase-js'
import { DynamicStructuredTool } from '@langchain/core/tools'

// ✨ NEW: Imports for Vector Search (Document RAG)
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase"
import { OpenAIEmbeddings } from "@langchain/openai"
import { CacheBackedEmbeddings } from "@langchain/classic/embeddings/cache_backed";
import { InMemoryStore } from "@langchain/core/stores"
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { HumanMessage } from "@langchain/core/messages";

export const dynamic = 'force-dynamic'
export const maxDuration = 30

// ===============================================
// ใช้ centralized database utility แทน pool ที่สร้างเอง
// ===============================================
const pool = getDatabase()

// สร้าง Supabase client
const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_OR_ANON_KEY!
)

// ===============================================
// ✨ NEW: สร้าง Vector Store สำหรับ Document Search
// ===============================================
async function createVectorStore() {
    const baseEmbeddings = new OpenAIEmbeddings({
        model: process.env.OPENAI_EMBEDDING_MODEL_NAME || "text-embedding-3-small",
        dimensions: 1536
    });

    // สร้าง Cache-backed embeddings เพื่อลดต้นทุนและเพิ่มความเร็ว
    const cacheStore = new InMemoryStore();
    const embeddings = CacheBackedEmbeddings.fromBytesStore(
        baseEmbeddings,
        cacheStore,
        {
            namespace: "rag_embeddings" // namespace สำหรับ RAG
        }
    );

    return new SupabaseVectorStore(embeddings, {
        client: supabase,
        tableName: 'documents',
        queryName: 'match_documents'
    });
}

// ===============================================
// ✨ NEW: สร้าง Tools สำหรับคุยกับ Supabase และ Vector Search
// ===============================================

// สร้าง Tool สำหรับค้นหาเอกสารจาก Vector Store
const searchDocumentsTool = new DynamicStructuredTool({
    name: "search_documents",
    description: "ค้นหาข้อมูลจากเอกสารที่เก็บไว้ในระบบ เช่น ข้อมูลร้าน, สินค้า, การขาย, หรือข้อมูลอื่นๆ ที่อัปโหลดไว้ในรูปแบบ PDF, CSV, TXT",
    schema: z.object({
        query: z.string().describe("คำค้นหาสำหรับค้นหาข้อมูลในเอกสาร เช่น 'ข้อมูลร้าน', 'สินค้า', 'ราคา', 'การขาย' เป็นต้น"),
        limit: z.number().optional().default(5).describe("จำนวนผลลัพธ์ที่ต้องการ (ค่าเริ่มต้น 5)")
    }),
    func: async ({ query, limit = 5 }) => {
        console.log(`🔧 TOOL CALLED: search_documents with query="${query}", limit=${limit}`);
        try {
            // สร้าง vector store
            const vectorStore = await createVectorStore();

            // ค้นหาเอกสารที่เกี่ยวข้อง
            const results = await vectorStore.similaritySearchWithScore(query, limit);

            if (!results || results.length === 0) {
                return `ไม่พบเอกสารที่เกี่ยวข้องกับ "${query}" ในระบบ`;
            }

            console.log(`✅ พบเอกสารที่เกี่ยวข้อง: ${results.length} รายการ`);

            // จัดรูปแบบผลลัพธ์
            if (results.length === 1) {
                const [doc, score] = results[0];
                const filename = doc.metadata?.filename || 'ไม่ทราบชื่อไฟล์';
                const type = doc.metadata?.type || 'ไม่ทราบประเภท';

                return `พบข้อมูลที่เกี่ยวข้องกับ "${query}":

**ไฟล์:** ${filename} (${type.toUpperCase()})
**เนื้อหา:** ${doc.pageContent}
**ความเกี่ยวข้อง:** ${(score * 100).toFixed(1)}%`;
            } else {
                // หลายผลลัพธ์ - แสดงเป็นรายการ
                const resultList = results.map(([doc, score], index) => {
                    const filename = doc.metadata?.filename || 'ไม่ทราบชื่อไฟล์';
                    const type = doc.metadata?.type || 'ไม่ทราบประเภท';
                    const preview = doc.pageContent.length > 200 ?
                        doc.pageContent.substring(0, 200) + '...' :
                        doc.pageContent;

                    return `**${index + 1}. ${filename}** (${type.toUpperCase()}) - ความเกี่ยวข้อง: ${(score * 100).toFixed(1)}%
${preview}`;
                }).join('\n\n');

                return `พบข้อมูลที่เกี่ยวข้องกับ "${query}" จำนวน ${results.length} รายการ:

${resultList}`;
            }
        } catch (e: unknown) {
            const errorMessage = e instanceof Error ? e.message : String(e);
            console.log('❌ Tool error:', errorMessage);

            if (errorMessage.includes('connection') ||
                errorMessage.includes('network') ||
                errorMessage.includes('timeout')) {
                return `ขออภัยครับ ไม่สามารถเข้าถึงระบบค้นหาเอกสารได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง`;
            }

            return `เกิดข้อผิดพลาดในการค้นหาเอกสาร: ${errorMessage}`;
        }
    },
})

// สร้าง Tool สำหรับค้นหาข้อมูลสินค้า (เดิม)
const getProductInfoTool = new DynamicStructuredTool({
    name: "get_product_info",
    description: "ค้นหาข้อมูลสินค้าจากฐานข้อมูล รวมถึงราคาและจำนวนคงคลัง (stock) โดยรับชื่อสินค้าเป็น input",
    schema: z.object({
        productName: z.string().describe("ชื่อของสินค้าที่ต้องการค้นหา เช่น 'MacBook Pro M3', 'iPhone', 'iPad' เป็นต้น"),
    }),
    func: async ({ productName }) => {
        console.log(`🔧 TOOL CALLED: get_product_info with productName="${productName}"`);
        try {
            // ตรวจสอบการเชื่อมต่อฐานข้อมูล
            const { data, error } = await supabase
                .from("products")
                .select("name, price, stock, description")
                .ilike("name", `%${productName}%`)
                .limit(5); // จำกัดผลลัพธ์ไม่เกิน 5 รายการ

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

const tools = [searchDocumentsTool, getProductInfoTool, getSalesDataTool];



// ===============================================
// POST API: ส่งข้อความและรับการตอบกลับแบบ Stream
// ===============================================
/**
 * ฟังก์ชันหลักสำหรับจัดการ Chat
 * 
 * Flow การทำงาน:
 * 1. สร้าง/ใช้ Session ID
 * 2. โหลด Summary เดิมจากฐานข้อมูล
 * 3. ตั้งค่า AI Model
 * 4. โหลดและ Trim ประวัติการสนทนา
 * 5. สร้าง Agent Graph
 * 6. สร้าง Stream Response
 * 7. บันทึกข้อความลงฐานข้อมูล
 * 8. อัปเดต Summary
 * 9. ส่ง Response กลับ
 */
export async function POST(req: NextRequest) {
    try {
        const { messages, sessionId, userId }: {
            messages: UIMessage[]
            sessionId?: string
            userId?: string
        } = await req.json()

        // 1. Session Handling (เหมือนเดิม)
        let currentSessionId = sessionId
        if (!currentSessionId) {
            const client = await getDatabase().connect()
            try {
                const firstMessage = messages.find(m => m.role === 'user')
                let title = 'New Chat'
                if (firstMessage && Array.isArray(firstMessage.parts)) {
                     const textPart = firstMessage.parts.find((p) => p.type === 'text') as { type: 'text'; text: string } | undefined;
                     if (textPart && textPart.text) title = textPart.text.slice(0, 50)
                }
                
                if (!userId) throw new Error('User ID required')
                const result = await client.query(
                    'INSERT INTO chat_sessions (title, user_id) VALUES ($1, $2) RETURNING id',
                    [title, userId]
                )
                currentSessionId = result.rows[0].id
            } finally {
                client.release()
            }
        }

        // 2. Load History
        const messageHistory = new PostgresChatMessageHistory({
            sessionId: currentSessionId!,
            tableName: "chat_messages",
            pool: getDatabase(),
        })

        const dbMessages = await messageHistory.getMessages();

        const systemPrompt = `คุณคือผู้ช่วย AI ที่ตอบเป็นภาษาไทย

คุณมี tools ที่สามารถใช้ได้ ได้แก่:
      1. **search_documents** - สำหรับค้นหาข้อมูลจากเอกสารที่อัปโหลดไว้ในระบบ (PDF, CSV, TXT)
      2. **get_product_info** - สำหรับค้นหาข้อมูลสินค้า ราคา และจำนวนในสต็อกจากฐานข้อมูล
      3. **get_sales_data** - สำหรับดูประวัติการขาย
      
      **กฎการใช้ tools:**
      
      **สำหรับคำถามเกี่ยวกับข้อมูลทั่วไป เช่น:**
      - ข้อมูลร้าน (ที่อยู่, เบอร์โทร, เวลาเปิด-ปิด)
      - ข้อมูลบริษัท
      - นโยบาย การบริการ
      - ข้อมูลที่อัปโหลดไว้ในรูปแบบเอกสาร
      **→ ใช้ search_documents**
      
      **สำหรับคำถามเกี่ยวกับสินค้าเฉพาะ เช่น:**
      - "Gaming Mouse ราคาเท่าไหร่?"
      - "iPhone มีในสต็อกไหม?"
      - สินค้าที่ระบุชื่อชัดเจน
      **→ ใช้ get_product_info**
      
      **สำหรับคำถามเกี่ยวกับการขาย เช่น:**
      - "Gaming Mouse ขายไปแล้วกี่ชิ้น?"
      - ประวัติการขาย
      **→ ใช้ get_sales_data**
      
      **หลักการตอบคำถาม:**
      - หากไม่แน่ใจว่าควรใช้ tool ไหน ให้ลองใช้ search_documents ก่อน
      - ถ้าผู้ใช้ถามแบบทั่วๆ เช่น "บอกข้อมูลร้าน" ให้ใช้ search_documents
      - ถ้าผู้ใช้ถามสินค้าเฉพาะ ให้ใช้ get_product_info
      - ห้ามเดาหรือสร้างข้อมูลขึ้นมาเอง ให้ใช้ข้อมูลจาก tools เท่านั้น
      
      สำหรับการค้นหาสินค้า:
      - หากผู้ใช้ใช้คำที่อาจมีความหมายคล้าย ให้ลองค้นหาด้วยคำที่เกี่ยวข้อง
      - เช่น "เมาส์" ลองค้นหาด้วย "mouse", "gaming mouse", "เมาส์เกม"
      - เช่น "แมคบุ๊ค" ลองค้นหาด้วย "MacBook", "Mac"
      - เช่น "กาแฟ" ลองค้นหาด้วย "coffee", "espresso"
      
      หากเกิด DATABASE_CONNECTION_ERROR ให้ตอบว่า "ขออภัยครับ ขณะนี้ไม่สามารถเข้าถึงฐานข้อมูลได้ กรุณาลองใหม่อีกครั้งในภายหลัง"

ตอบด้วยข้อมูลจาก tools เท่านั้น ห้ามสร้างข้อมูลเอง`;

        // 3. Setup Model & Agent (ส่วนที่เปลี่ยนใหม่)
        const model = new ChatGoogleGenerativeAI({
            model: 'gemini-1.5-flash', // แนะนำตัวนี้สำหรับ Free Tier
            temperature: 0.7,
            maxOutputTokens: 8192,
            streaming: true,
        })

        // ✅ ใช้ createReactAgent แทน createToolCallingAgent/AgentExecutor
        const agentApp = createReactAgent({
            llm: model,
            tools: tools, // ใช้ตัวแปร tools ที่คุณประกาศไว้ด้านบน
            stateModifier: systemPrompt,
        });

        // 4. Prepare Input
        const lastUserMessage = messages.filter(m => m.role === 'user').pop()
        let inputContent = ""
        if (lastUserMessage && Array.isArray(lastUserMessage.parts)) {
            const textPart = lastUserMessage.parts.find((p) => p.type === 'text') as { type: 'text'; text: string } | undefined;
            if (textPart && textPart.text) inputContent = textPart.text
        }
        if (!inputContent) return new Response("No valid input", { status: 400 })

        // รวมประวัติเก่า + ข้อความใหม่
        const finalMessages = [...dbMessages, new HumanMessage(inputContent)];

        // 5. Run Stream & Custom Response
        // ใช้ streamEvents เพื่อดึง token ทีละคำ (Streaming แท้ๆ)
        const eventStream = await agentApp.streamEvents(
            { messages: finalMessages },
            { version: "v2" }
        );

        const textStream = new ReadableStream({
            async start(controller) {
                let finalResponse = "";

                try {
                    for await (const { event, data } of eventStream) {
                        // จับ event ตอนที่ Model พ่น Token ออกมา
                        if (event === "on_chat_model_stream" && data.chunk && 'content' in data.chunk) {
                            const content = data.chunk.content;
                            // กรองไม่ให้ส่ง Tool Call definition ออกไป (ส่งเฉพาะข้อความคุย)
                            if (typeof content === 'string' && content.length > 0) {
                                controller.enqueue(content);
                                finalResponse += content;
                            }
                        }
                        // (Optional) จับ event ตอนเริ่มเรียก Tool เพื่อแสดงสถานะ
                        if (event === "on_tool_start" && 'name' in data) {
                            controller.enqueue(`\n🔄 กำลังค้นหาข้อมูล (${data.name})...\n`);
                        }
                    }

                    // บันทึกข้อความลง DB (เพราะ LangGraph ไม่ได้ผูกกับ PostgresHistory อัตโนมัติเหมือน AgentExecutor)
                    if (finalResponse) {
                        await messageHistory.addUserMessage(inputContent);
                        await messageHistory.addAIMessage(finalResponse);
                    }

                    controller.close();
                } catch (e) {
                    controller.error(e);
                }
            }
        });

        return new Response(textStream, {
            headers: {
                'Content-Type': 'text/plain; charset=utf-8',
                'x-session-id': currentSessionId!
            }
        })

    } catch (error) {
        console.error("API Error:", error)
        return new Response(JSON.stringify({ error: 'Internal Server Error' }), { status: 500 })
    }
}
// ===============================================
// GET API: ดึงประวัติการสนทนาจาก Session ID
// ===============================================
export async function GET(req: NextRequest) {
    try {
        const { searchParams } = new URL(req.url)
        const sessionId = searchParams.get('sessionId')
        if (!sessionId) {
            return new Response(
                JSON.stringify({ error: 'Session ID is required' }),
                { status: 400, headers: { 'Content-Type': 'application/json' } }
            )
        } const client = await pool.connect()
        try {
            const result = await client.query(
                `SELECT message, message->>'type' as message_type, created_at
     FROM chat_messages 
     WHERE session_id = $1 
     ORDER BY created_at ASC`,
                [sessionId]
            )

            const messages: { id: string; role: string; content: string; createdAt: Date }[] = result.rows.map((row, i) => {
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