import type { ChatCitation, KnowledgeChunk } from "@/lib/types";

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "has",
  "have",
  "i",
  "in",
  "is",
  "it",
  "of",
  "on",
  "or",
  "that",
  "the",
  "this",
  "to",
  "was",
  "were",
  "what",
  "when",
  "where",
  "who",
  "why",
  "with",
]);

function tokenize(text: string) {
  return text
    .toLowerCase()
    .match(/[a-z0-9][a-z0-9_-]{1,}/g)
    ?.filter((term) => !STOP_WORDS.has(term)) ?? [];
}

export function selectRelevantChunks(
  question: string,
  chunks: KnowledgeChunk[],
  limit = 8,
) {
  const queryTerms = tokenize(question);
  const uniqueTerms = [...new Set(queryTerms)];

  if (!uniqueTerms.length) {
    return chunks.slice(0, limit);
  }

  return chunks
    .map((chunk) => {
      const text = chunk.content.toLowerCase();
      let score = 0;
      for (const term of uniqueTerms) {
        const matches = text.match(new RegExp(`\\b${escapeRegExp(term)}\\b`, "g"));
        if (matches) score += matches.length;
      }
      if (text.includes(question.toLowerCase())) score += 8;
      score += Math.min(chunk.content.length / 3000, 1);
      return { chunk, score };
    })
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((entry) => entry.chunk);
}

export function buildCitations(chunks: KnowledgeChunk[]): ChatCitation[] {
  return chunks.map((chunk, index) => ({
    index: index + 1,
    chunkId: chunk.id,
    sourceId: chunk.sourceId,
    sourceName: chunk.sourceName,
    snippet:
      chunk.content.length > 240
        ? `${chunk.content.slice(0, 240).trim()}...`
        : chunk.content,
  }));
}

export function buildContext(chunks: KnowledgeChunk[]) {
  return chunks
    .map((chunk, index) => {
      const url = chunk.metadata?.url ? `\nURL: ${chunk.metadata.url}` : "";
      return `[${index + 1}] Source: ${chunk.sourceName}${url}\n${chunk.content}`;
    })
    .join("\n\n");
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
