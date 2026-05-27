import { NextResponse } from "next/server";
import * as cheerio from "cheerio";
import { z } from "zod";
import { createChunks, normalizeWhitespace } from "@/lib/text";

export const runtime = "nodejs";
export const maxDuration = 60;

const RequestSchema = z.object({
  url: z.string().url(),
  maxDepth: z.number().int().min(0).max(3).default(1),
});

type PageText = {
  url: string;
  title: string;
  text: string;
  links: URL[];
};

export async function POST(request: Request) {
  try {
    const body = RequestSchema.parse(await request.json());
    const root = new URL(body.url);
    const pages = await crawl(root, body.maxDepth);
    const combined = pages
      .map((page) => `URL: ${page.url}\nTitle: ${page.title}\n\n${page.text}`)
      .join("\n\n---\n\n");

    const result = await createChunks({
      text: combined,
      sourceName: root.hostname,
      sourceType: "url",
      url: root.toString(),
      metadata: {
        url: root.toString(),
        pageCount: pages.length,
      },
    });

    return NextResponse.json({
      ...result,
      pages: pages.map((page) => ({ url: page.url, title: page.title })),
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to ingest URL." },
      { status: 400 },
    );
  }
}

async function crawl(root: URL, maxDepth: number) {
  const visited = new Set<string>();
  const queue: Array<{ url: URL; depth: number }> = [{ url: root, depth: 0 }];
  const pages: PageText[] = [];

  while (queue.length && visited.size < 12) {
    const current = queue.shift();
    if (!current) break;

    const normalized = normalizeUrl(current.url);
    if (visited.has(normalized)) continue;
    visited.add(normalized);

    const page = await fetchPage(current.url);
    if (!page) continue;
    pages.push(page);

    if (current.depth >= maxDepth) continue;

    for (const link of page.links) {
      const linkKey = normalizeUrl(link);
      if (!visited.has(linkKey)) {
        queue.push({ url: link, depth: current.depth + 1 });
      }
    }
  }

  if (!pages.length) {
    throw new Error("No readable pages were fetched.");
  }

  return pages;
}

async function fetchPage(url: URL): Promise<PageText | null> {
  const response = await fetch(url, {
    headers: {
      "User-Agent": "ai-personal-assistant-nextjs/1.0",
      Accept: "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.2",
    },
    redirect: "follow",
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch ${url.toString()}: ${response.status}`);
  }

  const html = await response.text();
  const $ = cheerio.load(html);
  $("script, style, noscript, svg, nav, footer").remove();
  const title = normalizeWhitespace($("title").first().text() || url.hostname);
  const text = normalizeWhitespace($("body").text() || $.text());

  return {
    url: url.toString(),
    title,
    text,
    links: extractLinks(html, url, url.origin),
  };
}

function extractLinks(htmlText: string, baseUrl: URL, origin: string) {
  const $ = cheerio.load(htmlText);
  const links: URL[] = [];
  $("a[href]").each((_, element) => {
    const href = $(element).attr("href");
    if (!href) return;
    try {
      const next = new URL(href, baseUrl);
      if (next.origin === origin) links.push(next);
    } catch {
      // Ignore malformed URLs from source pages.
    }
  });
  return links;
}

function normalizeUrl(url: URL) {
  const copy = new URL(url);
  copy.hash = "";
  return copy.toString().replace(/\/$/, "");
}
