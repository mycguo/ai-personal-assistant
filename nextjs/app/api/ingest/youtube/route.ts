import { NextResponse } from "next/server";
import { z } from "zod";
import { YoutubeTranscript } from "youtube-transcript";
import { createChunks } from "@/lib/text";

export const runtime = "nodejs";
export const maxDuration = 30;

const RequestSchema = z.object({
  url: z.string().url(),
});

export async function POST(request: Request) {
  try {
    const body = RequestSchema.parse(await request.json());
    const videoId = extractYouTubeVideoId(body.url);
    if (!videoId) {
      return NextResponse.json({ error: "The URL is not a supported YouTube video URL." }, { status: 400 });
    }

    const transcript = await YoutubeTranscript.fetchTranscript(videoId);
    const text = transcript.map((part) => part.text).join(" ");
    const result = await createChunks({
      text,
      sourceName: `YouTube ${videoId}`,
      sourceType: "youtube",
      url: body.url,
      metadata: {
        url: body.url,
        videoId,
        captionCount: transcript.length,
      },
    });

    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to import YouTube captions." },
      { status: 400 },
    );
  }
}

function extractYouTubeVideoId(value: string) {
  const url = new URL(value);

  if (url.hostname === "youtu.be") {
    return url.pathname.slice(1) || null;
  }

  if (["youtube.com", "www.youtube.com", "m.youtube.com"].includes(url.hostname)) {
    if (url.pathname === "/watch") return url.searchParams.get("v");
    if (url.pathname.startsWith("/shorts/")) return url.pathname.split("/")[2] || null;
    if (url.pathname.startsWith("/embed/")) return url.pathname.split("/")[2] || null;
  }

  return null;
}
