import { NextResponse } from "next/server";
import { z } from "zod";
import { extractTextFromFile } from "@/lib/file-extractors";
import { createChunks } from "@/lib/text";

export const runtime = "nodejs";
export const maxDuration = 60;

const MetadataSchema = z.object({
  description: z.string().optional(),
});

export async function POST(request: Request) {
  try {
    const form = await request.formData();
    const file = form.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "Missing uploaded file." }, { status: 400 });
    }

    const metadata = MetadataSchema.safeParse({
      description: form.get("description")?.toString(),
    }).data;
    const text = await extractTextFromFile(file);
    const result = await createChunks({
      text,
      sourceName: file.name,
      sourceType: "file",
      metadata: {
        size: file.size,
        mimeType: file.type || null,
        description: metadata?.description || null,
      },
    });

    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to ingest file." },
      { status: 400 },
    );
  }
}
