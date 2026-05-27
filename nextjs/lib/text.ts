import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import type { KnowledgeChunk, KnowledgeSource, SourceType } from "@/lib/types";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 5000,
  chunkOverlap: 1000,
});

export function normalizeWhitespace(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

export async function createChunks(input: {
  text: string;
  sourceName: string;
  sourceType: SourceType;
  url?: string;
  metadata?: Record<string, string | number | boolean | null>;
}): Promise<{ source: KnowledgeSource; chunks: KnowledgeChunk[] }> {
  const cleaned = normalizeWhitespace(input.text);
  if (!cleaned) {
    throw new Error("No readable text was found.");
  }

  const sourceId = crypto.randomUUID();
  const pieces = await splitter.splitText(cleaned);
  const source: KnowledgeSource = {
    id: sourceId,
    name: input.sourceName,
    type: input.sourceType,
    url: input.url,
    createdAt: new Date().toISOString(),
    chunkCount: pieces.length,
    characterCount: cleaned.length,
  };

  const chunks = pieces.map((content, index) => ({
    id: `${sourceId}:${index}`,
    sourceId,
    sourceName: input.sourceName,
    sourceType: input.sourceType,
    content,
    metadata: {
      chunkIndex: index + 1,
      ...(input.metadata ?? {}),
    },
  }));

  return { source, chunks };
}
