import { NextResponse } from "next/server";
import { z } from "zod";
import { createChunks } from "@/lib/text";

export const runtime = "nodejs";
export const maxDuration = 60;

const ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2";

const StartUrlSchema = z.object({
  mediaUrl: z.string().url(),
  name: z.string().optional(),
});

const TranscriptStatusSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
});

type AssemblyTranscript = {
  id: string;
  status: "queued" | "processing" | "completed" | "error";
  text?: string;
  error?: string;
  audio_duration?: number;
};

export async function POST(request: Request) {
  try {
    const apiKey = getAssemblyApiKey();
    const contentType = request.headers.get("content-type") ?? "";

    let audioUrl: string;
    let name: string;

    if (contentType.includes("multipart/form-data")) {
      const form = await request.formData();
      const file = form.get("file");
      if (!(file instanceof File)) {
        return NextResponse.json({ error: "Missing uploaded audio or video file." }, { status: 400 });
      }

      audioUrl = await uploadMedia(apiKey, file);
      name = file.name;
    } else {
      const body = StartUrlSchema.parse(await request.json());
      audioUrl = body.mediaUrl;
      name = body.name || new URL(body.mediaUrl).pathname.split("/").pop() || body.mediaUrl;
    }

    const transcriptId = await startTranscript(apiKey, audioUrl);
    return NextResponse.json({
      transcriptId,
      status: "queued",
      name,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to start transcription." },
      { status: error instanceof MissingAssemblyKeyError ? 500 : 400 },
    );
  }
}

export async function GET(request: Request) {
  try {
    const apiKey = getAssemblyApiKey();
    const params = Object.fromEntries(new URL(request.url).searchParams);
    const query = TranscriptStatusSchema.parse(params);
    const transcript = await getTranscript(apiKey, query.id);

    if (transcript.status === "error") {
      return NextResponse.json(
        { status: "error", error: transcript.error || "Transcription failed." },
        { status: 400 },
      );
    }

    if (transcript.status !== "completed") {
      return NextResponse.json({ status: transcript.status });
    }

    const result = await createChunks({
      text: transcript.text || "",
      sourceName: query.name || `Transcript ${transcript.id}`,
      sourceType: "media",
      metadata: {
        transcriptId: transcript.id,
        audioDurationSeconds: transcript.audio_duration ?? null,
      },
    });

    return NextResponse.json({
      status: "completed",
      ...result,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to read transcription status." },
      { status: error instanceof MissingAssemblyKeyError ? 500 : 400 },
    );
  }
}

async function uploadMedia(apiKey: string, file: File) {
  const response = await fetch(`${ASSEMBLYAI_BASE_URL}/upload`, {
    method: "POST",
    headers: {
      authorization: apiKey,
    },
    body: await file.arrayBuffer(),
  });

  if (!response.ok) {
    throw new Error(`AssemblyAI upload failed with status ${response.status}.`);
  }

  const payload = (await response.json()) as { upload_url?: string };
  if (!payload.upload_url) {
    throw new Error("AssemblyAI upload did not return an upload URL.");
  }

  return payload.upload_url;
}

async function startTranscript(apiKey: string, audioUrl: string) {
  const response = await fetch(`${ASSEMBLYAI_BASE_URL}/transcript`, {
    method: "POST",
    headers: {
      authorization: apiKey,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      audio_url: audioUrl,
      iab_categories: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`AssemblyAI transcription failed to start with status ${response.status}.`);
  }

  const payload = (await response.json()) as { id?: string };
  if (!payload.id) {
    throw new Error("AssemblyAI did not return a transcript id.");
  }

  return payload.id;
}

async function getTranscript(apiKey: string, id: string) {
  const response = await fetch(`${ASSEMBLYAI_BASE_URL}/transcript/${encodeURIComponent(id)}`, {
    headers: {
      authorization: apiKey,
    },
  });

  if (!response.ok) {
    throw new Error(`AssemblyAI status request failed with status ${response.status}.`);
  }

  return (await response.json()) as AssemblyTranscript;
}

function getAssemblyApiKey() {
  const key = process.env.ASSEMBLYAI_API_KEY;
  if (!key) throw new MissingAssemblyKeyError();
  return key;
}

class MissingAssemblyKeyError extends Error {
  constructor() {
    super("ASSEMBLYAI_API_KEY is not configured for this deployment.");
  }
}
