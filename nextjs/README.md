# AI Personal Assistant - Next.js

This is a Vercel-ready rewrite of the Streamlit knowledge assistant.

## Local development

```bash
npm install
cp .env.example .env.local
npm run dev
```

Set `GOOGLE_API_KEY` in `.env.local`. Set `ASSEMBLYAI_API_KEY` if you want audio/video transcription.

## Vercel deployment

1. Import this repository in Vercel.
2. Set the project root directory to `nextjs`.
3. Add the `GOOGLE_API_KEY` environment variable.
4. Add `ASSEMBLYAI_API_KEY` for audio/video transcription.
5. Optionally set `GEMINI_MODEL`.
6. Deploy with the default Next.js build settings.

## Architecture

- `app/api/ingest/file`: extracts text from uploaded PDF, DOCX, Excel, CSV, JSON, Markdown, HTML, and TXT files.
- `app/api/ingest/url`: fetches and crawls same-origin web pages.
- `app/api/ingest/media`: uploads audio/video to AssemblyAI, polls transcription, and turns completed transcripts into chunks.
- `app/api/ingest/youtube`: imports available YouTube captions.
- `app/api/chat`: retrieves relevant chunks and answers with Gemini through LangChain.js.
- Browser local storage keeps the knowledge base between sessions. This avoids filesystem or vector database persistence requirements in Vercel serverless runtime.
