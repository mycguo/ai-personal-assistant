"use client";

import {
  AlertTriangle,
  AudioLines,
  Copy,
  Database,
  Download,
  Eye,
  FileText,
  Globe2,
  Loader2,
  RefreshCcw,
  Send,
  Sparkles,
  Trash2,
  Upload,
  Video,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, useTransition } from "react";
import type {
  ChatCitation,
  ChatMessage,
  IngestResponse,
  KnowledgeChunk,
  KnowledgeSource,
} from "@/lib/types";

const STORAGE_KEY = "ai-personal-assistant-nextjs:v1";
const MAX_CHAT_CHUNKS = 400;

type StoredState = {
  sources: KnowledgeSource[];
  chunks: KnowledgeChunk[];
  messages: ChatMessage[];
};

const EMPTY_STATE: StoredState = {
  sources: [],
  chunks: [],
  messages: [],
};

export function AssistantWorkspace() {
  const [state, setState] = useState<StoredState>(EMPTY_STATE);
  const [hasHydrated, setHasHydrated] = useState(false);
  const [question, setQuestion] = useState("");
  const [url, setUrl] = useState("");
  const [mediaUrl, setMediaUrl] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [maxDepth, setMaxDepth] = useState(1);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      setState(loadStoredState());
      setHasHydrated(true);
    });

    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    if (!hasHydrated) return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [hasHydrated, state]);

  const stats = useMemo(() => {
    const characters = state.sources.reduce((sum, source) => sum + source.characterCount, 0);
    return {
      sources: state.sources.length,
      chunks: state.chunks.length,
      characters,
    };
  }, [state.sources, state.chunks.length]);

  const canAsk = state.chunks.length > 0 && question.trim().length > 0 && !isPending;
  const selectedSource = useMemo(
    () => state.sources.find((source) => source.id === selectedSourceId) ?? null,
    [selectedSourceId, state.sources],
  );
  const selectedChunks = useMemo(
    () => state.chunks.filter((chunk) => chunk.sourceId === selectedSourceId),
    [selectedSourceId, state.chunks],
  );
  const selectedText = useMemo(
    () =>
      selectedChunks
        .map((chunk, index) => `--- Chunk ${index + 1} ---\n${chunk.content}`)
        .join("\n\n"),
    [selectedChunks],
  );

  async function ingestFiles(files: FileList | null) {
    if (!files?.length) return;
    setError(null);
    setStatus(`Processing ${files.length} file${files.length === 1 ? "" : "s"}...`);

    try {
      const results: IngestResponse[] = [];
      for (const file of Array.from(files)) {
        const form = new FormData();
        form.append("file", file);
        const response = await fetch("/api/ingest/file", {
          method: "POST",
          body: form,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Unable to ingest ${file.name}.`);
        results.push(payload);
      }

      setState((current) => appendIngestResults(current, results));
      setSelectedSourceId(results[0]?.source.id ?? null);
      setStatus(`Added ${results.reduce((sum, item) => sum + item.chunks.length, 0)} chunks.`);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to process files.");
      setStatus(null);
    }
  }

  async function ingestUrl() {
    const value = url.trim();
    if (!value) return;

    setError(null);
    setStatus("Fetching URL...");

    try {
      const response = await fetch("/api/ingest/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: value, maxDepth }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Unable to ingest URL.");

      setState((current) => appendIngestResults(current, [payload]));
      setSelectedSourceId(payload.source.id);
      setStatus(`Added ${payload.chunks.length} chunks from ${value}.`);
      setUrl("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to ingest URL.");
      setStatus(null);
    }
  }

  async function startMediaFileTranscription(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;

    setError(null);
    setStatus(`Uploading ${file.name} to AssemblyAI...`);

    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch("/api/ingest/media", {
        method: "POST",
        body: form,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Unable to start transcription.");

      if (mediaInputRef.current) mediaInputRef.current.value = "";
      await pollMediaTranscription(payload.transcriptId, payload.name || file.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start media transcription.");
      setStatus(null);
    }
  }

  async function startMediaUrlTranscription() {
    const value = mediaUrl.trim();
    if (!value) return;

    setError(null);
    setStatus("Starting media URL transcription...");

    try {
      const response = await fetch("/api/ingest/media", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mediaUrl: value }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Unable to start transcription.");

      setMediaUrl("");
      await pollMediaTranscription(payload.transcriptId, payload.name || value);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start media transcription.");
      setStatus(null);
    }
  }

  async function pollMediaTranscription(transcriptId: string, name: string) {
    for (let attempt = 0; attempt < 90; attempt += 1) {
      setStatus(`Transcribing ${name}...`);
      await delay(4000);

      const params = new URLSearchParams({ id: transcriptId, name });
      const response = await fetch(`/api/ingest/media?${params.toString()}`);
      const payload = await response.json();

      if (!response.ok) throw new Error(payload.error || "Unable to poll transcription.");

      if (payload.status === "completed") {
        setState((current) => appendIngestResults(current, [payload]));
        setSelectedSourceId(payload.source.id);
        setStatus(`Added transcript from ${name}.`);
        return;
      }

      if (payload.status === "error") {
        throw new Error(payload.error || "Transcription failed.");
      }
    }

    throw new Error("Transcription is still running. Try again later with the transcript id.");
  }

  async function ingestYouTube() {
    const value = youtubeUrl.trim();
    if (!value) return;

    setError(null);
    setStatus("Fetching YouTube captions...");

    try {
      const response = await fetch("/api/ingest/youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: value }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Unable to import YouTube captions.");

      setState((current) => appendIngestResults(current, [payload]));
      setSelectedSourceId(payload.source.id);
      setStatus(`Added ${payload.chunks.length} chunks from YouTube captions.`);
      setYoutubeUrl("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to import YouTube captions.");
      setStatus(null);
    }
  }

  function askQuestion() {
    const trimmed = question.trim();
    if (!trimmed || !state.chunks.length) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      createdAt: new Date().toISOString(),
    };

    setQuestion("");
    setError(null);
    setState((current) => ({
      ...current,
      messages: [...current.messages, userMessage],
    }));

    startTransition(async () => {
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: trimmed,
            chunks: state.chunks.slice(-MAX_CHAT_CHUNKS),
          }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Unable to answer.");

        const assistantMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: payload.answer,
          citations: payload.citations,
          createdAt: new Date().toISOString(),
        };

        setState((current) => ({
          ...current,
          messages: [...current.messages, assistantMessage],
        }));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to generate an answer.");
      }
    });
  }

  function removeSource(sourceId: string) {
    setState((current) => ({
      ...current,
      sources: current.sources.filter((source) => source.id !== sourceId),
      chunks: current.chunks.filter((chunk) => chunk.sourceId !== sourceId),
    }));
    if (selectedSourceId === sourceId) setSelectedSourceId(null);
  }

  function resetWorkspace() {
    setState(EMPTY_STATE);
    setSelectedSourceId(null);
    setStatus(null);
    setError(null);
  }

  function exportKnowledge() {
    const blob = new Blob([JSON.stringify(state, null, 2)], { type: "application/json" });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = `knowledge-assistant-${new Date().toISOString()}.json`;
    anchor.click();
    URL.revokeObjectURL(href);
  }

  async function copySelectedText() {
    if (!selectedText) return;
    await navigator.clipboard.writeText(selectedText);
    setStatus("Copied imported text to clipboard.");
  }

  return (
    <main className="workspace">
      <aside className="sidebar" aria-label="Knowledge base controls">
        <div className="brand">
          <div className="brandMark">
            <Sparkles size={20} aria-hidden="true" />
          </div>
          <div>
            <h1>AI Knowledge Assistant</h1>
            <p>Next.js and Vercel</p>
          </div>
        </div>

        <section className="panel">
          <div className="sectionTitle">
            <Upload size={17} aria-hidden="true" />
            <h2>Add documents</h2>
          </div>
          <label className="dropzone">
            <input
              id="knowledge-files"
              name="knowledge-files"
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.docx,.txt,.md,.markdown,.csv,.json,.html,.htm,.xlsx"
              onChange={(event) => void ingestFiles(event.target.files)}
            />
            <FileText size={26} aria-hidden="true" />
            <span>Upload PDF, Word, Excel, CSV, JSON, Markdown, HTML, or text</span>
          </label>
        </section>

        <section className="panel">
          <div className="sectionTitle">
            <Globe2 size={17} aria-hidden="true" />
            <h2>Crawl URL</h2>
          </div>
          <input
            id="crawl-url"
            name="crawl-url"
            className="input"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://example.com"
            type="url"
          />
          <div className="depthRow">
            <label htmlFor="depth">Depth</label>
            <input
              id="depth"
              name="depth"
              type="range"
              min="0"
              max="3"
              value={maxDepth}
              onChange={(event) => setMaxDepth(Number(event.target.value))}
            />
            <span>{maxDepth}</span>
          </div>
          <button className="button buttonSecondary" onClick={() => void ingestUrl()}>
            <RefreshCcw size={16} aria-hidden="true" />
            Fetch URL
          </button>
        </section>

        <section className="panel">
          <div className="sectionTitle">
            <AudioLines size={17} aria-hidden="true" />
            <h2>Audio and video</h2>
          </div>
          <label className="compactDropzone">
            <input
              id="media-files"
              name="media-files"
              ref={mediaInputRef}
              type="file"
              accept="audio/*,video/*,.mp3,.mp4,.m4a,.wav,.webm,.mov"
              onChange={(event) => void startMediaFileTranscription(event.target.files)}
            />
            <span>Upload audio or video for transcription</span>
          </label>
          <input
            id="media-url"
            name="media-url"
            className="input"
            value={mediaUrl}
            onChange={(event) => setMediaUrl(event.target.value)}
            placeholder="Public audio/video URL"
            type="url"
          />
          <button className="button buttonSecondary" onClick={() => void startMediaUrlTranscription()}>
            <AudioLines size={16} aria-hidden="true" />
            Transcribe media URL
          </button>
        </section>

        <section className="panel">
          <div className="sectionTitle">
            <Video size={17} aria-hidden="true" />
            <h2>YouTube captions</h2>
          </div>
          <input
            id="youtube-url"
            name="youtube-url"
            className="input"
            value={youtubeUrl}
            onChange={(event) => setYoutubeUrl(event.target.value)}
            placeholder="https://youtube.com/watch?v=..."
            type="url"
          />
          <button className="button buttonSecondary" onClick={() => void ingestYouTube()}>
            <Video size={16} aria-hidden="true" />
            Import captions
          </button>
        </section>

        <section className="panel">
          <div className="sectionTitle">
            <Database size={17} aria-hidden="true" />
            <h2>Knowledge base</h2>
          </div>
          <div className="statsGrid">
            <Metric label="Sources" value={stats.sources.toLocaleString()} />
            <Metric label="Chunks" value={stats.chunks.toLocaleString()} />
            <Metric label="Chars" value={stats.characters.toLocaleString()} />
          </div>
          <div className="sourceList">
            {state.sources.length === 0 ? (
              <p className="emptyText">No sources yet.</p>
            ) : (
              state.sources.map((source) => (
                <SourceRow
                  key={source.id}
                  source={source}
                  isSelected={source.id === selectedSourceId}
                  onRemove={removeSource}
                  onSelect={setSelectedSourceId}
                />
              ))
            )}
          </div>
          <div className="actionRow">
            <button className="iconButton" onClick={exportKnowledge} aria-label="Export knowledge base">
              <Download size={16} aria-hidden="true" />
            </button>
            <button className="iconButton danger" onClick={resetWorkspace} aria-label="Clear knowledge base">
              <Trash2 size={16} aria-hidden="true" />
            </button>
          </div>
        </section>
      </aside>

      <section className="chatShell" aria-label="Chat with knowledge base">
        <header className="chatHeader">
          <div>
            <p className="smallLabel">Queryable workspace</p>
            <h2>Ask across your uploaded knowledge</h2>
          </div>
          <div className="runtimeBadge">Gemini via LangChain.js</div>
        </header>

        {(status || error) && (
          <div className={error ? "notice errorNotice" : "notice"}>
            {error ? <AlertTriangle size={16} aria-hidden="true" /> : <Loader2 size={16} aria-hidden="true" />}
            <span>{error ?? status}</span>
            <button className="noticeClose" onClick={() => { setStatus(null); setError(null); }} aria-label="Dismiss notice">
              <X size={14} aria-hidden="true" />
            </button>
          </div>
        )}

        {selectedSource && (
          <section className="previewPanel" aria-label="Imported text preview">
            <div className="previewHeader">
              <div>
                <p className="smallLabel">Imported text</p>
                <h3>{selectedSource.name}</h3>
                <span>
                  {selectedSource.chunkCount.toLocaleString()} chunks,{" "}
                  {selectedSource.characterCount.toLocaleString()} extracted characters
                </span>
              </div>
              <div className="previewActions">
                <button className="iconButton" onClick={() => void copySelectedText()} aria-label="Copy imported text">
                  <Copy size={16} aria-hidden="true" />
                </button>
                <button className="iconButton" onClick={() => setSelectedSourceId(null)} aria-label="Close imported text preview">
                  <X size={16} aria-hidden="true" />
                </button>
              </div>
            </div>
            <textarea
              className="previewText"
              name="imported-text-preview"
              readOnly
              value={selectedText}
              aria-label={`Imported text from ${selectedSource.name}`}
            />
          </section>
        )}

        <div className="messageList">
          {state.messages.length === 0 ? (
            <div className="emptyState">
              <Sparkles size={34} aria-hidden="true" />
              <h3>Build a knowledge base, then ask a question.</h3>
              <p>
                Add documents or crawl a site. The assistant retrieves matching chunks and cites
                the sources it used.
              </p>
            </div>
          ) : (
            state.messages.map((message) => <MessageBubble key={message.id} message={message} />)
          )}
          {isPending && (
            <div className="message assistantMessage">
              <Loader2 className="spin" size={16} aria-hidden="true" />
              Thinking...
            </div>
          )}
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            askQuestion();
          }}
        >
          <textarea
            id="question"
            name="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={
              state.chunks.length
                ? "Ask about the current knowledge base..."
                : "Add a source before asking a question..."
            }
            rows={2}
          />
          <button className="button" disabled={!canAsk} type="submit">
            <Send size={16} aria-hidden="true" />
            Ask
          </button>
        </form>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function SourceRow({
  source,
  isSelected,
  onSelect,
  onRemove,
}: {
  source: KnowledgeSource;
  isSelected: boolean;
  onSelect: (sourceId: string) => void;
  onRemove: (sourceId: string) => void;
}) {
  const Icon =
    source.type === "url" ? Globe2 : source.type === "youtube" ? Video : source.type === "media" ? AudioLines : FileText;
  return (
    <div className={isSelected ? "sourceRow selectedSourceRow" : "sourceRow"}>
      <Icon size={16} aria-hidden="true" />
      <div>
        <strong>{source.name}</strong>
        <span>{source.chunkCount} chunks</span>
      </div>
      <button className="rowButton" onClick={() => onSelect(source.id)} aria-label={`View imported text from ${source.name}`}>
        <Eye size={14} aria-hidden="true" />
      </button>
      <button className="rowButton" onClick={() => onRemove(source.id)} aria-label={`Remove ${source.name}`}>
        <X size={14} aria-hidden="true" />
      </button>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <article className={`message ${isUser ? "userMessage" : "assistantMessage"}`}>
      <p>{message.content}</p>
      {!isUser && message.citations?.length ? <CitationList citations={message.citations} /> : null}
    </article>
  );
}

function CitationList({ citations }: { citations: ChatCitation[] }) {
  return (
    <div className="citations">
      {citations.map((citation) => (
        <details key={citation.chunkId}>
          <summary>
            [{citation.index}] {citation.sourceName}
          </summary>
          <p>{citation.snippet}</p>
        </details>
      ))}
    </div>
  );
}

function appendIngestResults(current: StoredState, results: IngestResponse[]): StoredState {
  return {
    ...current,
    sources: [...results.map((item) => item.source), ...current.sources],
    chunks: [...current.chunks, ...results.flatMap((item) => item.chunks)],
  };
}

function delay(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function loadStoredState(): StoredState {
  if (typeof window === "undefined") return EMPTY_STATE;

  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return EMPTY_STATE;

  try {
    const parsed = JSON.parse(raw) as StoredState;
    return {
      sources: parsed.sources ?? [],
      chunks: parsed.chunks ?? [],
      messages: parsed.messages ?? [],
    };
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return EMPTY_STATE;
  }
}
