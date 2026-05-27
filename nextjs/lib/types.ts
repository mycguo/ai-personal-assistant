export type SourceType = "file" | "url" | "media" | "youtube";

export type KnowledgeSource = {
  id: string;
  name: string;
  type: SourceType;
  url?: string;
  createdAt: string;
  chunkCount: number;
  characterCount: number;
};

export type KnowledgeChunk = {
  id: string;
  sourceId: string;
  sourceName: string;
  sourceType: SourceType;
  content: string;
  metadata?: Record<string, string | number | boolean | null>;
};

export type ChatCitation = {
  index: number;
  chunkId: string;
  sourceId: string;
  sourceName: string;
  snippet: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: ChatCitation[];
  createdAt: string;
};

export type IngestResponse = {
  source: KnowledgeSource;
  chunks: KnowledgeChunk[];
};
