# 🧠 Multimodal Agentic RAG Chatbot — Complete Implementation Plan
### Antigravity 2.0 · 21-File Agent Prompt Structure · 18-Week Roadmap
> **Stack**: NVIDIA NIM (free) · LangGraph · Qdrant Cloud · Supabase · FastAPI · Next.js · Vercel + Render

---

## 🗺️ MASTER ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
│   Next.js 14 Chat UI (Vercel free) ← WebSocket streaming       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST / SSE
┌───────────────────────────▼─────────────────────────────────────┐
│                    FastAPI BACKEND (Render free)                 │
│   /ingest  /chat  /upload  /health                              │
└──────┬─────────────────────────────────────┬────────────────────┘
       │                                     │
┌──────▼──────────┐                 ┌────────▼──────────────────┐
│ INGESTION       │                 │   5-AGENT CRAG GRAPH      │
│ PIPELINE        │                 │   (LangGraph StateGraph)  │
│                 │                 │                           │
│ Docling         │                 │  ┌─────────────────────┐  │
│ → PDF/DOCX/PPTX │                 │  │  Agent 1: Router    │  │
│ → Table→MD      │                 │  │  simple/complex/    │  │
│ → Image→caption │                 │  │  multimodal path    │  │
│   (NVIDIA VLM)  │                 │  └──────────┬──────────┘  │
│ → Audio→text    │                 │             │              │
│   (Parakeet)    │                 │  ┌──────────▼──────────┐  │
│ → Video→frames  │                 │  │  Agent 2: Retriever │  │
│   + captions    │                 │  │  BM25+Dense→RRF     │  │
└──────┬──────────┘                 │  │  →VL Reranker       │  │
       │ embed                      │  └──────────┬──────────┘  │
       │ (NVIDIA llama-             │             │              │
       │  nemotron-vl-embed)        │  ┌──────────▼──────────┐  │
┌──────▼──────────────────────┐    │  │  Agent 3: Grader    │  │
│      KNOWLEDGE LAYER        │    │  │  score 0-1 →        │  │
│                             │    │  │  sufficient?        │  │
│  Qdrant Cloud (free 1GB)    │◄───┘  └──────────┬──────────┘  │
│  ├─ dense_vector (2048d)    │             ↕ loop (max 3)      │
│  └─ bm25_sparse (IDF)       │  ┌──────────▼──────────┐       │
│                             │  │  Agent 4: Rewriter  │       │
│  Supabase (free 500MB)      │  │  HyDE + Step-back   │       │
│  ├─ document_registry       │  │  + Decomposition    │       │
│  ├─ chunk_metadata          │  └──────────┬──────────┘       │
│  └─ sessions                │             │                   │
│                             │  ┌──────────▼──────────┐       │
│  Supabase Storage (1GB)     │  │  Agent 5: Generator │       │
│  └─ raw files + images      │  │  NVIDIA VLM stream  │       │
└─────────────────────────────┘  │  +citations+images  │       │
                                 └─────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘

NVIDIA FREE MODELS (build.nvidia.com)
├─ LLM/VLM:    nvidia/nemotron-nano-12b-v1-vl  (multimodal, free)
├─ Embedding:  nvidia/llama-nemotron-embed-vl-1b-v2 (2048d, free)
├─ Reranker:   nvidia/llama-nemotron-rerank-vl-1b-v2 (free)
├─ Audio:      nvidia/parakeet-ctc-1-1b (free) / Groq Whisper
└─ Safety:     nvidia/nemotron-3.5-content-safety (4B, free)
```

---

## 🎯 KEY REPOS & TAKEAWAYS

| Repo | URL | Key Pattern Used |
|------|-----|-----------------|
| NVIDIA-AI-Blueprints/rag | github.com/NVIDIA-AI-Blueprints/rag | Plan-and-execute agentic pipeline, LangGraph nodes, scope discovery |
| NVIDIA/GenerativeAIExamples | github.com/NVIDIA/GenerativeAIExamples | Query Decomposer, Router node, NeMo Retriever integration |
| NVIDIA/workbench-example-agentic-rag | github.com/NVIDIA/workbench-example-agentic-rag | Free build.nvidia.com endpoint wiring, Gradio → swap to Next.js |
| langchain-ai/langchain-nvidia | github.com/langchain-ai/langchain-nvidia | LangGraph CRAG loops, NVIDIA NIM LangChain connector |
| asinghcsu/AgenticRAG-Survey | github.com/asinghcsu/AgenticRAG-Survey | CRAG + Adaptive RAG + Self-RAG pattern catalogue |

**Critical Takeaways:**
1. **NVIDIA Blueprint** → Use `plan-and-execute` node pattern as your Router logic scaffold
2. **GenerativeAIExamples** → `agentic_rag_with_nemo_retriever_nim.ipynb` is the closest working reference — steal the retry loop pattern
3. **Workbench Agentic RAG** → Wire free endpoints using `langchain-nvidia-ai-endpoints` — don't call NIM raw HTTP
4. **Qdrant Hybrid** → Use native `bm25_sparse_vector` with `Modifier.IDF` + `dense_vector` in one collection, then `FusionQuery(RRF)` — no Elasticsearch needed
5. **CRAG Pattern** → Grader score < 0.5 → Rewriter → loop; score < 0.3 after 3 retries → fallback to web search

---

## 🆓 FREE INFRASTRUCTURE STACK

```
Service              Provider              Free Tier Limits
─────────────────────────────────────────────────────────────
Frontend             Vercel               Unlimited deploys, 100GB BW
Backend (FastAPI)    Render.com           750 hrs/month, 512MB RAM
Vector DB            Qdrant Cloud         1GB storage, unlimited vectors
Metadata / Auth      Supabase             500MB Postgres, 1GB storage
File Storage         Supabase Storage     1GB objects
Cache                Upstash Redis        10,000 cmds/day
Audio Transcription  Groq Whisper-large   Free tier (rate limited)
LLM / VLM            NVIDIA build.nvidia  Unlimited on free models
Embedding            NVIDIA NIM           Free (llama-nemotron-embed-vl)
Reranker             NVIDIA NIM           Free (llama-nemotron-rerank-vl)
Safety               NVIDIA NIM           Free (nemotron-3.5-safety)
CI/CD                GitHub Actions       2000 min/month
Domain / SSL         Vercel               Free subdomain + auto-SSL
```

> ⚠️ **Render free tier sleeps after 15 min inactivity** — add an Upstash-triggered cron ping or upgrade to $7/mo paid for always-on.

---

## 📦 FREE NVIDIA MODELS — SELECTION RATIONALE

```python
NVIDIA_MODELS = {
    # Primary LLM + VLM — handles text AND image inputs
    "generator": "nvidia/nemotron-nano-12b-v1-vl",

    # Multimodal embedding — text + image in unified 2048-dim space
    # Critical for indexing images-in-PDFs with same vector as text
    "embedding": "nvidia/llama-nemotron-embed-vl-1b-v2",

    # Cross-encoder reranker — processes (query, doc_image/text) pairs
    # +6-7% recall over embedding-only on visual doc benchmarks
    "reranker": "nvidia/llama-nemotron-rerank-vl-1b-v2",

    # Image captioning during ingestion (alt: same VLM)
    "captioner": "nvidia/nemotron-nano-12b-v1-vl",

    # Multimodal content safety — input + output guardrails
    "safety": "nvidia/nemotron-3.5-content-safety",
}

BASE_URL = "https://integrate.api.nvidia.com/v1"
```

---

## 🗂️ 21-FILE ANTIGRAVITY 2.0 STRUCTURE

```
📁 antigravity/
├── 01_project_charter.md          ← Architecture + non-negotiables
├── 02_infra_setup.md              ← Vercel + Render + Qdrant + Supabase
├── 03_nvidia_nim_integration.md   ← API client + model wiring + fallbacks
├── 04_docling_parser.md           ← PDF/DOCX/PPTX text extraction
├── 05_image_table_extraction.md   ← Extract images + tables, VLM caption
├── 06_audio_pipeline.md           ← Groq Whisper → transcript chunks
├── 07_video_pipeline.md           ← ffmpeg frames → VLM captions → chunks
├── 08_chunking_strategy.md        ← Multimodal chunk schema + metadata
├── 09_nvidia_vl_embedder.md       ← Embed text + image chunks via NIM
├── 10_qdrant_hybrid_index.md      ← Dense+BM25 collection + RRF wiring
├── 11_supabase_schema.md          ← Tables: docs, chunks, sessions, users
├── 12_agent1_router.md            ← LangGraph Router node + routing logic
├── 13_agent2_retriever.md         ← Hybrid retrieval + RRF + VL reranker
├── 14_agent3_grader.md            ← LLM-as-judge relevance scorer
├── 15_agent4_rewriter.md          ← HyDE + step-back + decomposition
├── 16_agent5_generator.md         ← Streaming multimodal generator
├── 17_langgraph_crag.md           ← Full StateGraph + conditional edges
├── 18_fastapi_backend.md          ← REST endpoints + SSE streaming
├── 19_nextjs_frontend.md          ← Chat UI + file upload + image preview
├── 20_safety_evaluation.md        ← NVIDIA safety + RAGAS eval
└── 21_deployment_roadmap.md       ← 18-week plan + GitHub Actions CI/CD
```

---

## 📋 FILE-BY-FILE AGENT PROMPT SPECS

---

### `01_project_charter.md`
**Objective**: Define the product, constraints, and non-negotiables for all downstream agents.

**Prompt spec**:
```
Product: Multimodal RAG Chatbot (enterprise docs — HR, finance, policy)
Stack hard constraints:
  - All inference: NVIDIA build.nvidia.com free APIs only
  - No paid services except Qdrant Cloud free cluster
  - Deploy: Vercel (frontend) + Render free (backend)
  - Orchestration: LangGraph StateGraph, Python 3.11+
  - Agent pattern: 5-agent CRAG (Router→Retriever→Grader→Rewriter→Generator)
  - Retrieval: Qdrant native BM25 sparse + NVIDIA VL dense, RRF fusion
  
Non-negotiables:
  - Max 3 retrieval retry loops before fallback
  - SSE streaming on all generator responses
  - Every chunk stores: doc_id, chunk_id, page_num, modality, metadata
  - All images encoded as base64 alongside text representation
  - NVIDIA content safety check on every user input and output
```

---

### `02_infra_setup.md`
**Objective**: Bootstrap all free-tier services and generate environment config.

**Services to provision**:
```
1. Qdrant Cloud
   - create account at cloud.qdrant.io
   - create free cluster (1GB)
   - collect: QDRANT_URL, QDRANT_API_KEY

2. Supabase
   - create project at supabase.com
   - collect: SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY

3. NVIDIA build.nvidia.com
   - create account → API Catalog → generate key
   - collect: NVIDIA_API_KEY

4. Groq (audio transcription)
   - create account at console.groq.com
   - collect: GROQ_API_KEY

5. Upstash Redis
   - create free database at upstash.com
   - collect: UPSTASH_REDIS_URL, UPSTASH_REDIS_TOKEN

6. Render.com
   - new Web Service → connect GitHub repo → Python env
   - set env vars from .env.example

7. Vercel
   - new Project → connect GitHub repo → Next.js auto-detect
   - set NEXT_PUBLIC_API_URL=https://your-render-app.onrender.com
```

**`.env.example`**:
```env
# NVIDIA
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Qdrant
QDRANT_URL=https://xxx.cloud.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION=multimodal_rag

# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_KEY=...

# Groq (audio)
GROQ_API_KEY=...

# Upstash
UPSTASH_REDIS_URL=rediss://...
UPSTASH_REDIS_TOKEN=...

# App
CORS_ORIGINS=https://your-app.vercel.app
MAX_RETRIEVAL_RETRIES=3
TOP_K_RETRIEVE=20
TOP_K_RERANK=5
RELEVANCE_THRESHOLD=0.5
```

---

### `03_nvidia_nim_integration.md`
**Objective**: Build a unified NVIDIA NIM client with retry logic and model routing.

**Core client**:
```python
# app/services/nvidia_client.py
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class NVIDIAClient:
    """Unified wrapper for all NVIDIA NIM endpoints."""

    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.NVIDIA_API_KEY,
        )
        self.models = {
            "generator":  "nvidia/nemotron-nano-12b-v1-vl",
            "embedder":   "nvidia/llama-nemotron-embed-vl-1b-v2",
            "reranker":   "nvidia/llama-nemotron-rerank-vl-1b-v2",
            "safety":     "nvidia/nemotron-3.5-content-safety",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, inputs: list[str | dict]) -> list[list[float]]:
        """Embed text or image+text pairs. inputs can be strings or
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}"""
        response = self.client.embeddings.create(
            model=self.models["embedder"],
            input=inputs,
            encoding_format="float",
        )
        return [d.embedding for d in response.data]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(self, messages: list[dict], stream: bool = True):
        return self.client.chat.completions.create(
            model=self.models["generator"],
            messages=messages,
            stream=stream,
            max_tokens=1500,
            temperature=0.1,
        )

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        # Use NIM reranker endpoint
        response = self.client.post(
            "/ranking",
            json={
                "model": self.models["reranker"],
                "query": {"role": "user", "content": query},
                "passages": [{"role": "user", "content": p} for p in passages],
            }
        )
        return [r["logit"] for r in response.json()["rankings"]]
```

---

### `04_docling_parser.md`
**Objective**: Parse enterprise documents (PDF, DOCX, PPTX) into structured output preserving layout.

**Why Docling**: IBM's open-source Docling (2024-2025) is the best-in-class enterprise doc parser. It handles: multi-column PDFs, embedded tables → Markdown, footnotes, headers/footers, and exports clean Markdown. Zero cost, runs locally.

```python
# app/ingestion/document_parser.py
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path

class DocumentParser:
    def __init__(self):
        pipeline_opts = PdfPipelineOptions(
            do_ocr=True,           # OCR for scanned pages
            do_table_structure=True,  # Table detection → Markdown
            generate_page_images=True,  # For downstream VLM captioning
        )
        self.converter = DocumentConverter()

    def parse(self, file_path: str) -> dict:
        result = self.converter.convert(file_path)
        doc = result.document

        return {
            "text_markdown": doc.export_to_markdown(),
            "pages": [
                {
                    "page_num": p.page_no,
                    "text": p.export_to_markdown(),
                    "tables": self._extract_tables(p),
                    "image_refs": self._extract_image_refs(p),
                }
                for p in doc.pages
            ],
            "metadata": {
                "title": doc.name,
                "num_pages": len(doc.pages),
                "source_path": file_path,
            },
        }

    def _extract_tables(self, page) -> list[dict]:
        return [
            {"markdown": t.export_to_markdown(), "caption": t.caption or ""}
            for t in page.tables
        ]

    def _extract_image_refs(self, page) -> list[str]:
        return [img.uri for img in page.images if img.uri]
```

**Packages**:
```
pip install docling pymupdf pdfplumber python-docx python-pptx
```

---

### `05_image_table_extraction.md`
**Objective**: Extract all images from parsed documents, generate VLM captions, produce image chunks.

```python
# app/ingestion/image_extractor.py
import base64, fitz  # PyMuPDF

class ImageExtractor:
    def __init__(self, nvidia_client):
        self.nvidia = nvidia_client

    def extract_and_caption(self, pdf_path: str) -> list[dict]:
        """Extract embedded images from PDF, generate captions via NVIDIA VLM."""
        doc = fitz.open(pdf_path)
        image_chunks = []

        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                media_type = base_image["colorspace"]

                caption = self._caption_image(b64)

                image_chunks.append({
                    "chunk_id": f"img_{page_num}_{img_index}",
                    "modality": "image",
                    "page_num": page_num + 1,
                    "base64": b64,
                    "caption": caption,
                    # Text representation for BM25 + metadata
                    "text_repr": f"[IMAGE page {page_num+1}] {caption}",
                })

        return image_chunks

    def _caption_image(self, b64: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",
                     "text": "Describe this image in detail for document retrieval. "
                             "Focus on: type (chart/diagram/figure), key data points, "
                             "labels, and what concept it illustrates."}
                ]
            }
        ]
        resp = self.nvidia.generate(messages, stream=False)
        return resp.choices[0].message.content
```

---

### `06_audio_pipeline.md`
**Objective**: Transcribe audio files (MP3, WAV, M4A) via Groq Whisper, chunk into overlapping segments.

```python
# app/ingestion/audio_pipeline.py
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter

class AudioPipeline:
    def __init__(self):
        self.groq = Groq(api_key=settings.GROQ_API_KEY)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64
        )

    def process(self, audio_path: str, doc_id: str) -> list[dict]:
        with open(audio_path, "rb") as f:
            transcription = self.groq.audio.transcriptions.create(
                file=(audio_path, f),
                model="whisper-large-v3",
                response_format="verbose_json",   # includes timestamps
                timestamp_granularities=["segment"],
            )

        # Build timestamped text
        full_text = " ".join(s.text for s in transcription.segments)
        chunks = self.splitter.split_text(full_text)

        return [
            {
                "chunk_id": f"audio_{doc_id}_{i}",
                "modality": "audio_transcript",
                "text_repr": chunk,
                "source_type": "audio",
                "doc_id": doc_id,
            }
            for i, chunk in enumerate(chunks)
        ]
```

---

### `07_video_pipeline.md`
**Objective**: Extract keyframes from video, caption each via VLM, create temporal chunks.

```python
# app/ingestion/video_pipeline.py
import subprocess, base64, tempfile
from pathlib import Path

class VideoPipeline:
    def __init__(self, nvidia_client):
        self.nvidia = nvidia_client

    def process(self, video_path: str, doc_id: str,
                frame_interval_sec: int = 30) -> list[dict]:
        frames = self._extract_keyframes(video_path, frame_interval_sec)
        chunks = []
        for i, (timestamp, b64_frame) in enumerate(frames):
            caption = self._caption_frame(b64_frame, timestamp)
            chunks.append({
                "chunk_id": f"video_{doc_id}_{i}",
                "modality": "video_frame",
                "timestamp_sec": timestamp,
                "base64": b64_frame,
                "caption": caption,
                "text_repr": f"[VIDEO at {timestamp}s] {caption}",
                "doc_id": doc_id,
            })
        return chunks

    def _extract_keyframes(self, path: str, interval: int) -> list[tuple]:
        """Use ffmpeg to extract 1 frame every N seconds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pattern = f"{tmpdir}/frame_%04d.jpg"
            subprocess.run([
                "ffmpeg", "-i", path,
                "-vf", f"fps=1/{interval}",
                "-q:v", "2", out_pattern, "-y"
            ], capture_output=True, check=True)

            frames = []
            for frame_file in sorted(Path(tmpdir).glob("*.jpg")):
                idx = int(frame_file.stem.split("_")[1]) - 1
                timestamp = idx * interval
                b64 = base64.b64encode(frame_file.read_bytes()).decode()
                frames.append((timestamp, b64))
            return frames

    def _caption_frame(self, b64: str, timestamp: int) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",
                 "text": f"This is a video frame at {timestamp} seconds. "
                         "Describe what is happening and any visible text/charts."}
            ]
        }]
        resp = self.nvidia.generate(messages, stream=False)
        return resp.choices[0].message.content
```

---

### `08_chunking_strategy.md`
**Objective**: Define the universal chunk schema and chunking logic for all modalities.

**Universal Chunk Schema**:
```python
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    AUDIO_TRANSCRIPT = "audio_transcript"
    VIDEO_FRAME = "video_frame"

class MultimodalChunk(BaseModel):
    chunk_id: str           # unique: {doc_id}_{modality}_{index}
    doc_id: str
    modality: Modality
    text_repr: str          # always present — used for BM25 + LLM context
    base64: Optional[str]   # image/video frames only
    page_num: Optional[int]
    timestamp_sec: Optional[int]
    metadata: dict          # doc_title, dept, date, author, file_type
    embedding: Optional[list[float]]  # populated after embed step
```

**Text Chunking Strategy**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For enterprise docs: 512 tokens, 20% overlap
# Why: Policy/HR docs have dense paragraphs; 512 captures full clauses
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)

# Tables: keep whole table as one chunk (max ~800 tokens)
# Images: one chunk per image (caption as text_repr)
# Audio: 512 token chunks with 64 overlap
# Video: one chunk per keyframe
```

---

### `09_nvidia_vl_embedder.md`
**Objective**: Embed all chunk types via NVIDIA VL embedding model, handle image+text pairs.

```python
# app/ingestion/embedder.py

class MultimodalEmbedder:
    def __init__(self, nvidia_client):
        self.nvidia = nvidia_client

    def embed_chunks(self, chunks: list[MultimodalChunk]) -> list[MultimodalChunk]:
        BATCH_SIZE = 16  # Stay within NIM rate limits
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            inputs = [self._build_input(c) for c in batch]
            embeddings = self.nvidia.embed(inputs)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb
        return chunks

    def _build_input(self, chunk: MultimodalChunk):
        """For images/video: send base64 + caption as multimodal input."""
        if chunk.modality in (Modality.IMAGE, Modality.VIDEO_FRAME) \
                and chunk.base64:
            return [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{chunk.base64}"}},
                {"type": "text", "text": chunk.text_repr},
            ]
        # Text, table, audio: text only
        return chunk.text_repr

    def embed_query(self, query: str,
                    image_b64: str | None = None) -> list[float]:
        """Embed a user query, optionally with an attached image."""
        if image_b64:
            inp = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": query},
            ]
        else:
            inp = query
        return self.nvidia.embed([inp])[0]
```

---

### `10_qdrant_hybrid_index.md`
**Objective**: Create Qdrant collection with dense + BM25 sparse vectors, upsert all chunks, expose search.

```python
# app/retrieval/qdrant_store.py
from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION = "multimodal_rag"
DENSE_DIM = 2048  # NVIDIA llama-nemotron-embed-vl-1b-v2 output dim

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config={
                    "dense": models.VectorParams(
                        size=DENSE_DIM,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF  # IDF weighting enabled
                    )
                },
            )
            # Create payload indexes for metadata filtering
            for field in ["doc_id", "modality", "dept", "doc_type"]:
                self.client.create_payload_index(
                    collection_name=COLLECTION,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def upsert(self, chunks: list[MultimodalChunk]):
        from qdrant_client.http.models import PointStruct, Document
        points = [
            PointStruct(
                id=c.chunk_id,
                vector={
                    "dense": c.embedding,
                    "bm25": models.SparseVector(
                        # Use fastembed BM25 tokenizer for sparse
                        **self._bm25_encode(c.text_repr)
                    ),
                },
                payload={
                    "text_repr": c.text_repr,
                    "modality": c.modality,
                    "doc_id": c.doc_id,
                    "page_num": c.page_num,
                    "base64": c.base64,
                    "metadata": c.metadata,
                },
            )
            for c in chunks
        ]
        self.client.upsert(collection_name=COLLECTION, points=points)

    def hybrid_search(
        self,
        dense_vector: list[float],
        query_text: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[dict]:
        """BM25 sparse + dense semantic → RRF fusion."""
        qdrant_filter = self._build_filter(filters)

        results = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query_text, model="Qdrant/bm25"),
                    using="bm25",
                    limit=top_k,
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_k,
                    filter=qdrant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [
            {"score": p.score, **p.payload}
            for p in results.points
        ]

    def _build_filter(self, filters: dict | None) -> models.Filter | None:
        if not filters:
            return None
        conditions = [
            models.FieldCondition(
                key=k, match=models.MatchValue(value=v)
            )
            for k, v in filters.items()
        ]
        return models.Filter(must=conditions)

    def _bm25_encode(self, text: str) -> dict:
        from fastembed import SparseTextEmbedding
        model = SparseTextEmbedding("Qdrant/bm25")
        result = list(model.embed([text]))[0]
        return {"indices": result.indices.tolist(),
                "values": result.values.tolist()}
```

---

### `11_supabase_schema.md`
**Objective**: Define Supabase PostgreSQL tables for document registry, chunk metadata, and sessions.

```sql
-- Document registry
CREATE TABLE documents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename     TEXT NOT NULL,
    file_type    TEXT NOT NULL,  -- pdf, docx, pptx, mp3, mp4
    dept         TEXT,           -- HR, Finance, Policy...
    uploaded_by  TEXT,
    status       TEXT DEFAULT 'pending',  -- pending/processing/ready/failed
    num_chunks   INT,
    storage_path TEXT,           -- Supabase Storage path
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Chunk metadata (searchable by doc, page, modality)
CREATE TABLE chunks (
    id           TEXT PRIMARY KEY,  -- matches Qdrant point ID
    doc_id       UUID REFERENCES documents(id),
    modality     TEXT NOT NULL,
    page_num     INT,
    timestamp_sec INT,
    text_repr    TEXT NOT NULL,
    has_image    BOOLEAN DEFAULT FALSE,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_chunks_modality ON chunks(modality);

-- Chat sessions
CREATE TABLE sessions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Chat history (for multi-turn context)
CREATE TABLE messages (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   UUID REFERENCES sessions(id),
    role         TEXT NOT NULL,  -- user / assistant
    content      TEXT NOT NULL,
    image_b64    TEXT,           -- if user sent image
    source_chunks JSONB,         -- retrieved chunk IDs + scores
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_messages_session ON messages(session_id);
```

---

### `12_agent1_router.md`
**Objective**: LangGraph Router node classifies queries and sets retrieval strategy.

```python
# app/agents/router.py
from typing import Literal
from langchain_nvidia_ai_endpoints import ChatNVIDIA

class RouterAgent:
    """
    Classifies query into:
    - 'simple'     → single-hop factual, short context needed
    - 'complex'    → multi-hop, requires decomposition
    - 'multimodal' → query contains or asks about an image/chart/table
    - 'direct'     → greeting/chitchat, no retrieval needed
    """

    SYSTEM_PROMPT = """You are a query routing expert for an enterprise RAG system.
Classify the user query into exactly one category:
- simple: single factual question answerable from one document passage
- complex: requires synthesizing multiple sources or multi-step reasoning
- multimodal: asks about a chart, image, diagram, or table, or the user has attached an image
- direct: greetings, meta questions about the chatbot, no document retrieval needed

Return ONLY the category word. Nothing else."""

    def __init__(self):
        self.llm = ChatNVIDIA(
            model="nvidia/nemotron-nano-12b-v1-vl",
            api_key=settings.NVIDIA_API_KEY,
            base_url=settings.NVIDIA_BASE_URL,
        )

    def route(self, state: dict) -> dict:
        query = state["query"]
        has_image = bool(state.get("user_image_b64"))

        if has_image:
            state["route"] = "multimodal"
            return state

        response = self.llm.invoke([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ])
        route = response.content.strip().lower()
        if route not in ("simple", "complex", "multimodal", "direct"):
            route = "simple"  # safe default

        state["route"] = route
        # Extract metadata filters from query if present
        state["filters"] = self._extract_filters(query)
        return state

    def _extract_filters(self, query: str) -> dict:
        """Heuristic: detect dept/doc_type mentions → pre-filter Qdrant."""
        filters = {}
        dept_keywords = {"hr": "HR", "finance": "Finance",
                        "policy": "Policy", "legal": "Legal"}
        for kw, dept in dept_keywords.items():
            if kw in query.lower():
                filters["dept"] = dept
        return filters
```

---

### `13_agent2_retriever.md`
**Objective**: Execute hybrid search (BM25 + dense → RRF) then rerank with NVIDIA VL reranker.

```python
# app/agents/retriever.py

class RetrieverAgent:
    def __init__(self, qdrant_store, embedder, nvidia_client):
        self.store = qdrant_store
        self.embedder = embedder
        self.nvidia = nvidia_client

    def retrieve(self, state: dict) -> dict:
        query = state["rewritten_query"] or state["query"]
        image_b64 = state.get("user_image_b64")
        filters = state.get("filters", {})

        # Step 1: Embed query (multimodal if image present)
        dense_vec = self.embedder.embed_query(query, image_b64)

        # Step 2: Hybrid search → top-20 via RRF
        raw_results = self.store.hybrid_search(
            dense_vector=dense_vec,
            query_text=query,
            top_k=settings.TOP_K_RETRIEVE,
            filters=filters,
        )

        # Step 3: Rerank via NVIDIA VL reranker → top-5
        if len(raw_results) > settings.TOP_K_RERANK:
            passages = [r["text_repr"] for r in raw_results]
            scores = self.nvidia.rerank(query, passages)
            # Sort by reranker score, keep top-k
            ranked = sorted(
                zip(raw_results, scores),
                key=lambda x: x[1], reverse=True
            )[:settings.TOP_K_RERANK]
            raw_results = [r for r, _ in ranked]

        state["retrieved_docs"] = raw_results
        state["retrieval_count"] = state.get("retrieval_count", 0) + 1
        return state
```

---

### `14_agent3_grader.md`
**Objective**: Score retrieved documents for relevance. If insufficient, trigger rewrite loop.

```python
# app/agents/grader.py

GRADER_PROMPT = """You are a relevance grader for an enterprise RAG system.
Given a query and a retrieved document passage, score the relevance from 0.0 to 1.0.

Score guide:
1.0 = Directly answers the query
0.7 = Partially answers, has related information
0.4 = Tangentially related
0.0 = Irrelevant

Return ONLY a JSON: {"score": <float>, "reason": "<one sentence>"}"""

class GraderAgent:
    def __init__(self, nvidia_client):
        self.nvidia = nvidia_client

    def grade(self, state: dict) -> dict:
        query = state["query"]
        docs = state["retrieved_docs"]

        scored_docs = []
        for doc in docs:
            messages = [
                {"role": "system", "content": GRADER_PROMPT},
                {"role": "user", "content":
                    f"Query: {query}\n\nDocument: {doc['text_repr'][:800]}"},
            ]
            resp = self.nvidia.generate(messages, stream=False)
            try:
                import json
                result = json.loads(resp.choices[0].message.content)
                doc["relevance_score"] = result["score"]
            except Exception:
                doc["relevance_score"] = 0.3

            if doc["relevance_score"] >= settings.RELEVANCE_THRESHOLD:
                scored_docs.append(doc)

        state["relevant_docs"] = scored_docs
        state["is_sufficient"] = len(scored_docs) >= 2  # Need at least 2 relevant
        return state

    def should_rewrite(self, state: dict) -> Literal["rewrite", "generate"]:
        """LangGraph conditional edge."""
        if state["is_sufficient"]:
            return "generate"
        if state["retrieval_count"] >= settings.MAX_RETRIEVAL_RETRIES:
            # Hard stop — generate with what we have
            state["relevant_docs"] = state["retrieved_docs"][:2]
            return "generate"
        return "rewrite"
```

---

### `15_agent4_rewriter.md`
**Objective**: Reformulate queries using HyDE, step-back, and decomposition strategies.

```python
# app/agents/rewriter.py

class RewriterAgent:
    """Three-strategy rewriter (cycle through on each retry)."""

    def __init__(self, nvidia_client):
        self.nvidia = nvidia_client

    def rewrite(self, state: dict) -> dict:
        retry_num = state.get("retrieval_count", 1)
        query = state["query"]
        strategy = ["hyde", "stepback", "decompose"][(retry_num - 1) % 3]

        if strategy == "hyde":
            state["rewritten_query"] = self._hyde(query)
        elif strategy == "stepback":
            state["rewritten_query"] = self._stepback(query)
        else:
            # Decompose → use first sub-query for next retrieval
            sub_queries = self._decompose(query)
            state["sub_queries"] = sub_queries
            state["rewritten_query"] = sub_queries[0]

        return state

    def _hyde(self, query: str) -> str:
        """Hypothetical Document Embedding: generate a fake answer,
        embed that as the new query (improves dense recall)."""
        prompt = (f"Write a short expert answer (3-4 sentences) to: '{query}'. "
                  "This is a hypothetical document to improve search recall.")
        resp = self.nvidia.generate(
            [{"role": "user", "content": prompt}], stream=False
        )
        return resp.choices[0].message.content

    def _stepback(self, query: str) -> str:
        prompt = (f"Make this specific query more general to find broader context: "
                  f"'{query}'. Return only the generalized query.")
        resp = self.nvidia.generate(
            [{"role": "user", "content": prompt}], stream=False
        )
        return resp.choices[0].message.content.strip()

    def _decompose(self, query: str) -> list[str]:
        prompt = (f"Break this complex question into 2-3 simpler sub-questions: "
                  f"'{query}'. Return as a JSON list of strings.")
        resp = self.nvidia.generate(
            [{"role": "user", "content": prompt}], stream=False
        )
        import json
        try:
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return [query]
```

---

### `16_agent5_generator.md`
**Objective**: Assemble multimodal context and stream the final answer with citations.

```python
# app/agents/generator.py

SYSTEM_PROMPT = """You are an intelligent enterprise assistant.
Answer the user's question using ONLY the provided context.
Always cite your sources as [Source 1], [Source 2] etc.
If context contains image/chart descriptions, explicitly reference them.
If you cannot find the answer in the context, say so clearly."""

class GeneratorAgent:
    def __init__(self, nvidia_client):
        self.nvidia = nvidia_client

    def generate(self, state: dict):
        query = state["query"]
        docs = state["relevant_docs"]
        user_image = state.get("user_image_b64")
        history = state.get("chat_history", [])[-4:]  # last 2 turns

        # Build context with source numbering
        context_parts = []
        for i, doc in enumerate(docs, 1):
            ctx = f"[Source {i}] (modality: {doc['modality']})\n{doc['text_repr']}"
            context_parts.append(ctx)
        context = "\n\n---\n\n".join(context_parts)

        # Assemble messages (multimodal if image present)
        user_content = [{"type": "text",
                         "text": f"Context:\n{context}\n\nQuestion: {query}"}]
        if user_image:
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{user_image}"}
            })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_content},
        ]

        # Stream response
        stream = self.nvidia.generate(messages, stream=True)
        state["stream"] = stream
        state["source_docs"] = docs
        return state
```

---

### `17_langgraph_crag.md`
**Objective**: Wire all 5 agents into a LangGraph StateGraph with conditional CRAG loop.

```python
# app/graph/crag_graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

class RAGState(TypedDict):
    query: str
    user_image_b64: Optional[str]
    route: Optional[str]
    filters: dict
    rewritten_query: Optional[str]
    sub_queries: list[str]
    retrieved_docs: list[dict]
    relevant_docs: list[dict]
    is_sufficient: bool
    retrieval_count: int
    chat_history: list[dict]
    stream: object
    source_docs: list[dict]

def build_crag_graph(router, retriever, grader, rewriter, generator):
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("router",    router.route)
    graph.add_node("retriever", retriever.retrieve)
    graph.add_node("grader",    grader.grade)
    graph.add_node("rewriter",  rewriter.rewrite)
    graph.add_node("generator", generator.generate)

    # Entry point
    graph.set_entry_point("router")

    # Router → conditional routing
    def route_decision(state: RAGState) -> str:
        if state["route"] == "direct":
            return "generator"
        return "retriever"

    graph.add_conditional_edges("router", route_decision,
                                {"retriever": "retriever",
                                 "generator": "generator"})

    # Retriever → Grader (always)
    graph.add_edge("retriever", "grader")

    # Grader → Rewriter or Generator (CRAG loop)
    graph.add_conditional_edges("grader", grader.should_rewrite,
                                {"rewrite": "rewriter",
                                 "generate": "generator"})

    # Rewriter → back to Retriever
    graph.add_edge("rewriter", "retriever")

    # Generator → END
    graph.add_edge("generator", END)

    return graph.compile()

# CRAG FLOW DIAGRAM
# ┌────────┐     direct    ┌──────────┐
# │ Router │──────────────►│Generator │──►END
# └───┬────┘               └──────────┘
#     │ retrieval                ▲
#     ▼                          │ generate
# ┌──────────┐  ┌────────┐  ┌───┴────┐
# │Retriever │─►│ Grader │─►│Rewriter│
# └──────────┘  └────────┘  └────────┘
#     ▲               │ loop (max 3)
#     └───────────────┘
```

---

### `18_fastapi_backend.md`
**Objective**: REST API with SSE streaming, file upload, and ingestion endpoints.

```python
# app/main.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json

app = FastAPI(title="Multimodal RAG API")
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

# ── Chat endpoint (SSE streaming) ─────────────────────────────────
@app.post("/api/chat")
async def chat(request: ChatRequest):
    async def event_stream():
        state = {
            "query": request.query,
            "user_image_b64": request.image_b64,
            "chat_history": request.history or [],
            "retrieval_count": 0,
            "filters": {},
            "rewritten_query": None,
            "sub_queries": [],
        }
        result = crag_graph.invoke(state)
        stream = result.get("stream")

        if stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'type':'text','content':delta})}\n\n"

        # Send source citations
        sources = result.get("source_docs", [])
        yield f"data: {json.dumps({'type':'sources','sources':sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ── File ingestion endpoint ───────────────────────────────────────
@app.post("/api/ingest")
async def ingest(
    file: UploadFile = File(...),
    dept: str = "General",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    doc_id = await save_to_supabase(file)
    background_tasks.add_task(run_ingestion_pipeline, doc_id, dept)
    return {"doc_id": doc_id, "status": "processing"}

@app.get("/api/ingest/{doc_id}/status")
async def ingest_status(doc_id: str):
    status = await supabase.get_doc_status(doc_id)
    return {"doc_id": doc_id, "status": status}

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Project structure**:
```
backend/
├── app/
│   ├── main.py
│   ├── config.py          (settings from env)
│   ├── agents/
│   │   ├── router.py
│   │   ├── retriever.py
│   │   ├── grader.py
│   │   ├── rewriter.py
│   │   └── generator.py
│   ├── graph/
│   │   └── crag_graph.py
│   ├── ingestion/
│   │   ├── document_parser.py
│   │   ├── image_extractor.py
│   │   ├── audio_pipeline.py
│   │   ├── video_pipeline.py
│   │   ├── chunking.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   └── qdrant_store.py
│   └── services/
│       ├── nvidia_client.py
│       └── supabase_client.py
├── requirements.txt
├── render.yaml
└── .env.example
```

**`render.yaml`**:
```yaml
services:
  - type: web
    name: multimodal-rag-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port 10000
    healthCheckPath: /health
    envVars:
      - fromGroup: rag-env-vars
```

---

### `19_nextjs_frontend.md`
**Objective**: Next.js 14 chat UI with file upload, streaming display, image preview, source accordion.

**Key components**:
```
frontend/
├── app/
│   ├── page.tsx              (main chat route)
│   ├── layout.tsx
│   └── globals.css
├── components/
│   ├── ChatInterface.tsx     (message list + input)
│   ├── MessageBubble.tsx     (renders text + images + sources)
│   ├── FileUpload.tsx        (drag-drop, progress bar)
│   ├── SourceAccordion.tsx   (collapsible source citations)
│   └── ImagePreview.tsx      (query-time image attachment)
└── lib/
    ├── api.ts                (fetch wrapper + SSE parser)
    └── types.ts
```

**SSE Client (`lib/api.ts`)**:
```typescript
export async function streamChat(
  query: string,
  history: Message[],
  imageB64?: string,
  onChunk: (text: string) => void,
  onSources: (sources: Source[]) => void,
) {
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history, image_b64: imageB64 }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const lines = decoder.decode(value).split("\n\n");
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") return;
      const parsed = JSON.parse(data);
      if (parsed.type === "text") onChunk(parsed.content);
      if (parsed.type === "sources") onSources(parsed.sources);
    }
  }
}
```

**File upload** → POST `/api/ingest` → poll `/api/ingest/{id}/status` with 5s interval until `ready`.

---

### `20_safety_evaluation.md`
**Objective**: NVIDIA content safety on all I/O; RAGAS evaluation harness.

**Safety Middleware**:
```python
# app/middleware/safety.py
async def check_safety(text: str, nvidia_client) -> dict:
    resp = nvidia_client.client.chat.completions.create(
        model="nvidia/nemotron-3.5-content-safety",
        messages=[{"role": "user", "content": text}],
        max_tokens=100,
    )
    content = resp.choices[0].message.content
    # Returns: {"safe": true/false, "category": null/"hate"/"violence"/...}
    return parse_safety_response(content)
```

**RAGAS Evaluation** (run offline against a golden test set):
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,         # Is answer grounded in context?
    answer_relevancy,     # Does answer address the question?
    context_precision,    # Are retrieved docs relevant?
    context_recall,       # Are all relevant docs retrieved?
)

# Build test dataset: 50 Q&A pairs from your enterprise docs
dataset = load_golden_dataset("eval/golden_qa.json")

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=nvidia_llm,
    embeddings=nvidia_embeddings,
)
# Target baselines:
# faithfulness    > 0.85
# answer_relevancy > 0.80
# context_precision > 0.75
# context_recall   > 0.70
```

---

### `21_deployment_roadmap.md`
**Objective**: 18-week phased delivery plan and GitHub Actions CI/CD.

```
╔══════════════════════════════════════════════════════════════════╗
║           18-WEEK DELIVERY ROADMAP                              ║
╠══════════════════╦═══════════════════════════════════════════════╣
║ Phase 1 Wk 1-3  ║ FOUNDATION                                   ║
║                  ║ ✓ All free-tier accounts provisioned          ║
║                  ║ ✓ Qdrant collection created & tested          ║
║                  ║ ✓ Supabase schema migrated                    ║
║                  ║ ✓ NVIDIA NIM client with retry + safety       ║
║                  ║ ✓ FastAPI skeleton + health endpoint          ║
║                  ║ ✓ Next.js scaffold on Vercel                  ║
╠══════════════════╬═══════════════════════════════════════════════╣
║ Phase 2 Wk 4-6  ║ INGESTION PIPELINE                           ║
║                  ║ ✓ Docling PDF/DOCX/PPTX parser               ║
║                  ║ ✓ Image extraction + VLM captioning           ║
║                  ║ ✓ Table extraction → Markdown                 ║
║                  ║ ✓ Chunking + MultimodalChunk schema           ║
║                  ║ ✓ NVIDIA VL embedding service                 ║
║                  ║ ✓ Qdrant upsert + metadata indexed            ║
║                  ║ ✓ Supabase document registry working          ║
╠══════════════════╬═══════════════════════════════════════════════╣
║ Phase 3 Wk 7-8  ║ AUDIO / VIDEO PIPELINE                       ║
║                  ║ ✓ Groq Whisper audio transcription            ║
║                  ║ ✓ ffmpeg keyframe extraction                  ║
║                  ║ ✓ VLM video frame captioning                  ║
║                  ║ ✓ All modalities ingesting into same Qdrant   ║
╠══════════════════╬═══════════════════════════════════════════════╣
║ Phase 4 Wk 9-11 ║ HYBRID RETRIEVAL LAYER                       ║
║                  ║ ✓ BM25 sparse + dense → RRF working           ║
║                  ║ ✓ NVIDIA VL reranker integrated               ║
║                  ║ ✓ Metadata filters (dept, file_type)          ║
║                  ║ ✓ RAGAS context_precision baseline measured    ║
╠══════════════════╬═══════════════════════════════════════════════╣
║ Phase 5 Wk 12-14║ 5-AGENT CRAG GRAPH                           ║
║                  ║ ✓ Agent 1: Router with 4 routes               ║
║                  ║ ✓ Agent 2: Hybrid Retriever                   ║
║                  ║ ✓ Agent 3: Grader (LLM-as-judge)             ║
║                  ║ ✓ Agent 4: Rewriter (HyDE + step-back)       ║
║                  ║ ✓ Agent 5: Streaming Generator                ║
║                  ║ ✓ LangGraph StateGraph + retry loop tested     ║
╠══════════════════╬═══════════════════════════════════════════════╣
║ Phase 6 Wk 15-16║ BACKEND + STREAMING                          ║
║                  ║ ✓ FastAPI /chat SSE endpoint                  ║
║                  ║ ✓ /ingest background task + status polling    ║
║                  ║ ✓ Upstash Redis query cache                   ║
║                  ║ ✓ NVIDIA content safety middleware            ║
╠══════════════════╬═══════════════════════════════════════════════╣
║ Phase 7 Wk 17   ║ NEXT.JS FRONTEND                             ║
║                  ║ ✓ Chat UI with SSE stream rendering           ║
║                  ║ ✓ File upload + ingestion status              ║
║                  ║ ✓ Image attachment at query time              ║
║                  ║ ✓ Source accordion with citations             ║
╠══════════════════╬═══════════════════════════════════════════════╣
║ Phase 8 Wk 18   ║ EVALUATION + LAUNCH                          ║
║                  ║ ✓ RAGAS full eval on 50 golden Q&As           ║
║                  ║ ✓ GitHub Actions CI/CD pipeline               ║
║                  ║ ✓ Render auto-deploy on push to main          ║
║                  ║ ✓ Vercel auto-deploy on push to main          ║
║                  ║ ✓ README + API docs (auto-generated)          ║
╚══════════════════╩═══════════════════════════════════════════════╝
```

**GitHub Actions CI/CD (`.github/workflows/deploy.yml`)**:
```yaml
name: Deploy RAG App

on:
  push:
    branches: [main]

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          pip install -r backend/requirements.txt
          pytest backend/tests/ -v
      - name: Deploy to Render
        run: |
          curl -X POST ${{ secrets.RENDER_DEPLOY_HOOK }}

  # Vercel deploys automatically on GitHub push via Vercel GitHub integration
```

---

## 🚀 QUICK START SEQUENCE (Week 1 Day 1)

```bash
# 1. Clone reference repo
git clone https://github.com/NVIDIA/workbench-example-agentic-rag
# Study: how free endpoints are wired via langchain-nvidia-ai-endpoints

# 2. Create your repo
mkdir multimodal-rag && cd multimodal-rag
git init && git remote add origin <your-github>

# 3. Install core stack
pip install langchain-nvidia-ai-endpoints langgraph qdrant-client \
            fastembed supabase docling groq fastapi uvicorn \
            python-multipart tenacity ragas

npm create next-app@latest frontend -- --typescript --tailwind --app

# 4. Verify NVIDIA API
python -c "
from openai import OpenAI
client = OpenAI(base_url='https://integrate.api.nvidia.com/v1',
                api_key='YOUR_KEY')
r = client.chat.completions.create(
    model='nvidia/nemotron-nano-12b-v1-vl',
    messages=[{'role':'user','content':'Hello'}],
    max_tokens=50
)
print(r.choices[0].message.content)
"

# 5. Verify Qdrant Cloud
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='YOUR_URL', api_key='YOUR_KEY')
print(c.get_collections())
"
```

---

## 📊 EVALUATION TARGETS

| Metric | Baseline (Week 9) | Target (Week 18) |
|--------|-------------------|------------------|
| Context Precision | 0.60 | > 0.75 |
| Context Recall | 0.55 | > 0.70 |
| Faithfulness | 0.70 | > 0.85 |
| Answer Relevancy | 0.65 | > 0.80 |
| Avg Latency (TTFT) | — | < 2s |
| Avg Latency (full) | — | < 8s |
| Retry rate (Grader) | — | < 25% |

---

## ⚠️ KNOWN RISKS & MITIGATIONS

| Risk | Mitigation |
|------|------------|
| Render free tier cold start (~30s) | Add Upstash cron ping every 14min |
| NVIDIA NIM rate limits on free tier | Upstash Redis cache for repeated queries; batch embed |
| Qdrant 1GB free tier overflow | Limit base64 storage in Qdrant payload; store in Supabase Storage; only store embedding + text_repr in Qdrant |
| Video ffmpeg not on Render free | Use `ffmpeg-python` with static binary or disable video on free deploy |
| LangGraph retry infinite loop | Hard max_retries=3 enforced in grader.should_rewrite |
| Large PDFs (>100 pages) timing out | Chunk ingestion into background tasks with status polling |
