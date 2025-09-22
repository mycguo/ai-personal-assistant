# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based AI knowledge assistant implementing Retrieval-Augmented Generation (RAG). Users can upload documents, crawl websites, and transcribe audio/video to build a queryable knowledge base.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_app.py

# Run with coverage
pytest --cov=. tests/
```

### Vector Store Testing
```bash
# Test vector store implementation
python test_milvus.py
```

### Container Development
```bash
# Build Docker image
docker build -t ai-assistant .

# Run container locally
docker run -p 8080:8080 ai-assistant
```

## Architecture Overview

### Core Components

**Main Application**: `app.py` - Chat interface for querying the knowledge base
**Admin Interface**: `pages/app_admin.py` - Document upload and processing
**System Admin**: `pages/system_admin.py` - S3 backup/restore operations
**Vector Store**: Multiple implementations for different deployment scenarios

### Vector Store Architecture

The application supports multiple vector store backends:

1. **SimpleVectorStore** (Current Default): File-based storage using JSON + pickle
   - Location: `./vector_store_personal_assistant/`
   - Files: `metadata.json`, `vectors.pkl`
   - Benefits: No external dependencies, Streamlit-compatible, simple deployment

2. **MilvusVectorStore**: Scalable vector database
   - Local: Milvus Lite (`.db` file)
   - Remote: Milvus server or Zilliz Cloud
   - Configuration via `MILVUS_URI` environment variable

3. **Legacy FAISS**: Original implementation (deprecated)
   - Migration tools available in `MilvusVectorStore.migrate_from_faiss()`

### Document Processing Pipeline

**Supported Formats**:
- Text: PDF (PyPDF2), Word (python-docx), Excel (pandas), plain text
- Web: Custom crawler with BeautifulSoup, LangChain WebBaseLoader
- Audio/Video: AssemblyAI transcription, yt-dlp for YouTube

**Processing Flow**:
1. Content extraction with metadata
2. Text chunking (5000 chars, 1000 overlap)
3. Embedding generation (OpenAI text-embedding-3-large)
4. Vector storage with metadata preservation

### LLM Integration

**Primary Stack**:
- Model: Google Gemini 2.0 Flash
- Embeddings: OpenAI text-embedding-3-large
- Framework: LangChain for RAG pipeline

**Alternative Models**: NVIDIA DeepSeek R1 available (currently disabled for performance)

## Configuration Management

### Environment Variables
```bash
# Core AI Services
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
GENAI_API_KEY=your_google_key
NVIDIA_API_KEY=your_nvidia_key
ASSEMBLYAI_API_KEY=your_assemblyai_key

# Vector Store (optional)
MILVUS_URI=./milvus_local.db  # Local file
# MILVUS_URI=http://localhost:19530  # Local server
# MILVUS_URI=https://your-cluster.api.milvus.io  # Cloud
MILVUS_TOKEN=your_token  # For cloud deployments

# AWS S3 (optional backup)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
BUCKET_NAME=your_bucket
```

### Streamlit Secrets
For production deployment, use Streamlit secrets in `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your_key"
GOOGLE_API_KEY = "your_key"
# ... other secrets
```

## Key Development Patterns

### Vector Store Interface
All vector store implementations follow this pattern:
```python
def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]
def similarity_search(self, query: str, k: int = 4) -> List[Document]
def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]
```

### Error Handling for Web Content
```python
# Standard pattern for web requests
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Encoding": "identity"  # Disable compression
})

if response.encoding is None:
    response.encoding = 'utf-8'
```

### Metadata Enrichment
All document types include comprehensive metadata:
- PDFs: Author, title, subject, creator, producer, page count
- Word: Core properties, creation/modification dates
- Excel: Sheet information, row/column counts
- Web: URL, crawling depth, source information

### Streamlit Integration Patterns
- Multi-page architecture using `pages/` directory
- Session state for async operation tracking
- File upload with progress indicators
- Error handling with user-friendly messages

## Deployment Considerations

### Local Development
- Uses file-based vector store by default
- Requires OpenAI API key for embeddings
- Optional services can be mocked for testing

### Production Deployment
- Containerized with Docker (optimized for Cloud Run)
- Supports environment-based configuration
- Can scale to cloud vector databases (Milvus/Zilliz)
- S3 integration for backup/restore

### Vector Store Migration
When upgrading vector store backends:
1. Use `MilvusVectorStore.migrate_from_faiss()` for FAISS migration
2. Ensure consistent `store_path` between admin and query interfaces
3. Test similarity search results after migration

## Testing Strategy

- Unit tests with mocked external services
- Integration tests for web crawler functionality
- Vector store functionality tested via `test_milvus.py`
- Mock patterns available for AI service testing

## External Service Dependencies

**Critical Dependencies**:
- OpenAI API (embeddings)
- Google AI API (primary LLM)

**Optional Dependencies**:
- AssemblyAI (audio transcription)
- AWS S3 (backup storage)
- NVIDIA API (alternative LLM)

All external services should be gracefully degraded when unavailable.