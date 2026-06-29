from pydantic import BaseModel
from enum import Enum
from typing import Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils import generate_uuid

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
    text_repr: str          # text used for BM25 + LLM context
    base64: Optional[str] = None   # base64 string for image/video modality
    page_num: Optional[int] = None
    timestamp_sec: Optional[int] = None
    metadata: dict = {}
    embedding: Optional[List[float]] = None

class ChunkingService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
        )

    def chunk_document(self, parsed_doc: dict, doc_id: str, dept: str) -> List[MultimodalChunk]:
        """Convert a parsed document (containing text, pages, tables) into MultimodalChunks."""
        chunks = []
        
        # 1. Text chunking from pages
        for page in parsed_doc.get("pages", []):
            page_num = page.get("page_num", 1)
            page_text = page.get("text", "").strip()
            if not page_text:
                continue
                
            text_segments = self.text_splitter.split_text(page_text)
            for idx, segment in enumerate(text_segments):
                chunks.append(MultimodalChunk(
                    chunk_id=generate_uuid(f"{doc_id}_text_{page_num}_{idx}"),
                    doc_id=doc_id,
                    modality=Modality.TEXT,
                    text_repr=segment,
                    page_num=page_num,
                    metadata={
                        "title": parsed_doc["metadata"].get("title", ""),
                        "dept": dept,
                        "file_type": parsed_doc["metadata"].get("file_type", "pdf"),
                    }
                ))

        # 2. Table chunking
        # Each table is kept intact as its own chunk
        table_idx = 0
        for page in parsed_doc.get("pages", []):
            page_num = page.get("page_num", 1)
            for t in page.get("tables", []):
                t_markdown = t.get("markdown", "").strip()
                t_caption = t.get("caption", "").strip()
                if not t_markdown:
                    continue
                
                text_repr = f"Table on page {page_num}: {t_caption}\n\n{t_markdown}"
                chunks.append(MultimodalChunk(
                    chunk_id=generate_uuid(f"{doc_id}_table_{page_num}_{table_idx}"),
                    doc_id=doc_id,
                    modality=Modality.TABLE,
                    text_repr=text_repr,
                    page_num=page_num,
                    metadata={
                        "title": parsed_doc["metadata"].get("title", ""),
                        "dept": dept,
                        "caption": t_caption,
                        "file_type": parsed_doc["metadata"].get("file_type", "pdf"),
                    }
                ))
                table_idx += 1

        return chunks
