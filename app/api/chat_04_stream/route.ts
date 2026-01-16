import { NextRequest } from "next/server";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { toUIMessageStream } from "@ai-sdk/langchain";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { createUIMessageStreamResponse, UIMessage, convertToModelMessages } from "ai";

// กำหนดให้ API นี้ทำงานแบบ Edge Runtime
export const runtime = "edge";
export const maxDuration = 30;

export async function POST(req: NextRequest) {
  try {
    const { messages }: { messages: UIMessage[] } = await req.json();

    // 1. ใส่ await เพื่อดึงข้อมูลออกจาก Promise อย่างถูกต้อง
    const coreMessages = await convertToModelMessages(messages);

    // 2. แปลง (Map) ข้อมูล และดึงเฉพาะ Text ออกมาเพื่อป้องกัน AI ตอบกลับเป็น JSON
    const langchainMessages = coreMessages.map((msg) => {
      let content = "";

      if (typeof msg.content === 'string') {
        content = msg.content;
      } else if (Array.isArray(msg.content)) {
        // 3. ใช้ Inline Type Predicate เพื่อแก้ปัญหา 'any' และดึง text จาก array
        content = msg.content
          .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
          .map((part) => part.text)
          .join("\n");
      }

      // สร้าง Message Object ของ LangChain ตาม Role
      if (msg.role === "user") {
        return new HumanMessage(content);
      } else if (msg.role === "assistant") {
        return new AIMessage(content);
      } else {
        return new SystemMessage(content);
      }
    });

    // 4. สร้าง Prompt Template
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are a helpful and friendly AI assistant."],
      ...langchainMessages,
    ]);

    // 5. ใช้ Model ของ Google (Gemini) ให้ตรงกับ Library ที่ import มา
    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash", 
      temperature: 0.7,
      maxOutputTokens: 2048,
      streaming: true,
      // apiKey: process.env.GOOGLE_API_KEY // ตรวจสอบว่ามีค่านี้ในไฟล์ .env แล้ว
    });

    const chain = prompt.pipe(model);

    const stream = await chain.stream({});

    const response = createUIMessageStreamResponse({
      stream: toUIMessageStream(stream),
    });

    return response;

  } catch (error) {
    console.error("API Error:", error);
    return new Response(
      JSON.stringify({
        error: "An error occurred while processing your request",
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}