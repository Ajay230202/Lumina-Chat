import os
import json
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from app.config import settings
from app.services.nvidia_client import NVIDIAClient
from app.retrieval.qdrant_store import QdrantStore
from app.ingestion.embedder import MultimodalEmbedder
from app.ingestion.pipeline import IngestionPipeline

from app.agents.router import RouterAgent
from app.agents.retriever import RetrieverAgent
from app.agents.grader import GraderAgent
from app.agents.rewriter import RewriterAgent
from app.agents.generator import GeneratorAgent
from app.graph.crag_graph import build_crag_graph

from app.services.safety import SafetyGuard

app = FastAPI(title="Lumina RAG API")

from app.mcp_server import mcp
app.mount("/mcp", mcp.sse_app())

# Setup CORS
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",")] if settings.CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services & Agents
nvidia_client = NVIDIAClient()
qdrant_store = QdrantStore()
embedder = MultimodalEmbedder(nvidia_client)
safety_guard = SafetyGuard()

router_agent = RouterAgent(nvidia_client)
retriever_agent = RetrieverAgent(qdrant_store, embedder, nvidia_client)
grader_agent = GraderAgent(nvidia_client)
rewriter_agent = RewriterAgent(nvidia_client)
generator_agent = GeneratorAgent(nvidia_client)

crag_graph = build_crag_graph(
    router=router_agent,
    retriever=retriever_agent,
    grader=grader_agent,
    rewriter=rewriter_agent,
    generator=generator_agent
)

pipeline = IngestionPipeline()

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None
    image_b64: Optional[str] = None
    session_id: Optional[str] = None

# --- Ingestion Background Task ---
def run_ingestion_task(file_path: str, dept: str, doc_id: str):
    try:
        pipeline.run(file_path, dept, doc_id)
    except Exception as e:
        print(f"Background ingestion failed: {e}")
    finally:
        # Delete the temp file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "message": "Multimodal RAG API is healthy"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Perform content safety check on input query
    if not safety_guard.is_safe(request.query):
        async def unsafe_stream():
            yield f"data: {json.dumps({'type': 'error', 'content': 'Flagged: User query violates content safety policies.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(unsafe_stream(), media_type="text/event-stream")

    async def event_stream():
        # Setup initial state for LangGraph StateGraph
        state = {
            "query": request.query,
            "user_image_b64": request.image_b64,
            "chat_history": request.history or [],
            "retrieval_count": 0,
            "filters": {},
            "rewritten_query": None,
            "sub_queries": [],
            "retrieved_docs": [],
            "relevant_docs": [],
            "is_sufficient": False,
            "stream": None,
            "source_docs": []
        }
        
        # Invoke Graph
        try:
            # Save user message to DB
            if request.session_id:
                pipeline.supabase.add_message(
                    session_id=request.session_id,
                    role="user",
                    content=request.query,
                    image_b64=request.image_b64
                )

            result = crag_graph.invoke(state)
            stream = result.get("stream")
            
            full_response = ""
            if stream:
                # Stream the chat completion text token-by-token
                for chunk in stream:
                    # LangChain stream returns BaseMessageChunk
                    content = getattr(chunk, "content", "")
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"
            
            # Send citations metadata
            sources = result.get("source_docs", [])
            formatted_sources = [
                {
                    "chunk_id": doc.get("chunk_id", doc.get("id", "")),
                    "modality": doc.get("modality", "text"),
                    "text_repr": doc.get("text_repr", ""),
                    "page_num": doc.get("page_num"),
                    "score": doc.get("score")
                }
                for doc in sources
            ]

            # Save assistant message to DB
            if request.session_id:
                pipeline.supabase.add_message(
                    session_id=request.session_id,
                    role="assistant",
                    content=full_response,
                    source_chunks=formatted_sources
                )

            yield f"data: {json.dumps({'type': 'sources', 'sources': formatted_sources})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/ingest")
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dept: str = Form("General")
):
    # Save the uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Create metadata registry record first (will fail if SQL setup hasn't run)
        filename = file.filename
        file_ext = Path(filename).suffix.lower().replace(".", "")
        doc_record = pipeline.supabase.create_document(
            filename=filename,
            file_type=file_ext,
            dept=dept
        )
        doc_id = doc_record.get("id")
        
        # Register ingestion pipeline to run in the background
        background_tasks.add_task(run_ingestion_task, temp_file_path, dept, doc_id)
        
        return {"doc_id": doc_id, "status": "processing"}
    except Exception as e:
        # Delete temp file if registry creation fails
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize ingestion. Have you run the Supabase SQL migrations? Error: {e}"
        )

@app.get("/api/ingest/{doc_id}/status")
async def ingest_status(doc_id: str):
    try:
        status = pipeline.supabase.get_document_status(doc_id)
        return {"doc_id": doc_id, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    try:
        docs = pipeline.supabase.get_all_documents()
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions")
async def create_new_session():
    try:
        session = pipeline.supabase.create_session()
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/history")
async def get_session_messages(session_id: str):
    try:
        messages = pipeline.supabase.get_session_history(session_id)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
