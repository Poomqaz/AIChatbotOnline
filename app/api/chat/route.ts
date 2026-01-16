import { NextRequest } from "next/server"
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"
import { toUIMessageStream } from "@ai-sdk/langchain"
import {
  createUIMessageStreamResponse,
  UIMessage,
} from "ai"

export const runtime = "edge"
export const maxDuration = 30

function getTextFromUIMessage(message: UIMessage): string {
  return message.parts
    .filter((p) => p.type === "text")
    .map((p) => p.text)
    .join("")
}

export async function POST(req: NextRequest) {
  try {
    const { messages }: { messages: UIMessage[] } = await req.json()

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash", // หรือ gemini-1.5-pro
      temperature: 0.7,
      maxOutputTokens: 300,
      streaming: true,
    })

    const formattedMessages = [
      { role: "system", content: "You are a helpful and friendly AI assistant." },
      ...messages.map((m) => ({
        role: m.role,
        content: getTextFromUIMessage(m),
      })),
    ]

    const stream = await model.stream(formattedMessages)

    return createUIMessageStreamResponse({
      stream: toUIMessageStream(stream),
    })
  } catch (error) {
    console.error("API Error:", error)
    return new Response(
      JSON.stringify({ error: "Gemini API error" }),
      { status: 500 }
    )
  }
}
