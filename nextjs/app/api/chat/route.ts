import { NextResponse } from "next/server";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";
import { buildCitations, buildContext, selectRelevantChunks } from "@/lib/retrieval";
import type { KnowledgeChunk } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 60;

const ChunkSchema = z.object({
  id: z.string(),
  sourceId: z.string(),
  sourceName: z.string(),
  sourceType: z.enum(["file", "url", "media", "youtube"]),
  content: z.string(),
  metadata: z.record(z.string(), z.union([z.string(), z.number(), z.boolean(), z.null()])).optional(),
});

const RequestSchema = z.object({
  question: z.string().min(1),
  chunks: z.array(ChunkSchema).max(400),
});

export async function POST(request: Request) {
  try {
    const apiKey = process.env.GOOGLE_API_KEY || process.env.GENAI_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: "GOOGLE_API_KEY is not configured for this deployment." },
        { status: 500 },
      );
    }

    const body = RequestSchema.parse(await request.json());
    const relevantChunks = selectRelevantChunks(
      body.question,
      body.chunks as KnowledgeChunk[],
      8,
    );

    if (!relevantChunks.length) {
      return NextResponse.json({
        answer:
          "I could not find matching context in the current knowledge base. Add more documents or ask about a topic that appears in the uploaded material.",
        citations: [],
      });
    }

    const context = buildContext(relevantChunks);
    const model = new ChatGoogleGenerativeAI({
      apiKey,
      model: process.env.GEMINI_MODEL || "gemini-2.0-flash",
      temperature: 0.2,
    });

    const response = await model.invoke([
      [
        "system",
        "You are a concise knowledge assistant. Answer only from the provided context. If the context is insufficient, say what is missing. Cite sources inline using bracket numbers like [1].",
      ],
      [
        "human",
        `Question:\n${body.question}\n\nContext:\n${context}`,
      ],
    ]);

    return NextResponse.json({
      answer: contentToText(response.content),
      citations: buildCitations(relevantChunks),
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to generate answer." },
      { status: 400 },
    );
  }
}

function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "text" in part) {
          return String(part.text);
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}
