import os
from pathlib import Path
from app.ingestion.document_parser import DocumentParser
from app.ingestion.image_extractor import ImageExtractor
from app.ingestion.audio_pipeline import AudioPipeline
from app.ingestion.video_pipeline import VideoPipeline
from app.ingestion.chunking import ChunkingService, MultimodalChunk
from app.ingestion.embedder import MultimodalEmbedder
from app.retrieval.qdrant_store import QdrantStore
from app.services.supabase_client import SupabaseService
from app.services.nvidia_client import NVIDIAClient

class IngestionPipeline:
    def __init__(self):
        self.nvidia = NVIDIAClient()
        self.doc_parser = DocumentParser()
        self.img_extractor = ImageExtractor(self.nvidia)
        self.audio_pipeline = AudioPipeline()
        self.video_pipeline = VideoPipeline(self.nvidia)
        self.chunker = ChunkingService()
        self.embedder = MultimodalEmbedder(self.nvidia)
        self.qdrant = QdrantStore()
        self.supabase = SupabaseService()

    def run(self, file_path: str, dept: str = "General", doc_id: str = None) -> str:
        filename = os.path.basename(file_path)
        file_ext = Path(file_path).suffix.lower().replace(".", "")
        
        print(f"Ingesting file: {filename} ({file_ext})")
        if not doc_id:
            doc_record = self.supabase.create_document(
                filename=filename,
                file_type=file_ext,
                dept=dept
            )
            doc_id = doc_record.get("id")
        else:
            self.supabase.update_document_status(doc_id, "processing")
        
        try:
            chunks = []
            
            # 1. Parse and chunk based on file extension
            if file_ext in ["pdf", "docx", "pptx"]:
                parsed_doc = self.doc_parser.parse(file_path)
                parsed_doc["metadata"]["file_type"] = file_ext
                
                # Extract text and table chunks
                chunks = self.chunker.chunk_document(parsed_doc, doc_id, dept)
                
                # For PDFs, also extract embedded images
                if file_ext == "pdf":
                    print("Extracting images from PDF...")
                    img_chunks = self.img_extractor.extract_and_caption(file_path, doc_id)
                    for img in img_chunks:
                        img["doc_id"] = doc_id
                        img["metadata"] = {
                            "title": parsed_doc["metadata"].get("title", ""),
                            "dept": dept,
                            "file_type": file_ext
                        }
                        chunks.append(MultimodalChunk(**img))
                        
            elif file_ext in ["mp3", "wav", "m4a"]:
                print("Processing audio transcript...")
                audio_chunks = self.audio_pipeline.process(file_path, doc_id)
                for chunk in audio_chunks:
                    chunk["metadata"] = {"dept": dept, "file_type": file_ext}
                    chunks.append(MultimodalChunk(**chunk))
                    
            elif file_ext in ["mp4", "avi", "mov"]:
                print("Processing video frames...")
                video_chunks = self.video_pipeline.process(file_path, doc_id)
                for chunk in video_chunks:
                    chunk["metadata"] = {"dept": dept, "file_type": file_ext}
                    chunks.append(MultimodalChunk(**chunk))
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            if not chunks:
                print("No chunks generated.")
                self.supabase.update_document_status(doc_id, "failed")
                return doc_id

            # 2. Generate embeddings
            print(f"Generating embeddings for {len(chunks)} chunks...")
            chunks = self.embedder.embed_chunks(chunks)

            # 3. Upsert to Qdrant
            print("Upserting chunks to Qdrant...")
            self.qdrant.upsert(chunks)

            # 4. Save metadata to Supabase
            print("Saving metadata to Supabase...")
            supabase_chunks = [
                {
                    "id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "modality": c.modality.value,
                    "page_num": c.page_num,
                    "timestamp_sec": c.timestamp_sec,
                    "text_repr": c.text_repr,
                    "has_image": c.base64 is not None
                }
                for c in chunks
            ]
            self.supabase.insert_chunks(supabase_chunks)
            
            # Update doc registry status
            self.supabase.update_document_status(doc_id, "ready", num_chunks=len(chunks))
            print(f"Successfully ingested document {filename}!")
            return doc_id

        except Exception as e:
            print(f"Pipeline ingestion failed for {filename}: {e}")
            self.supabase.update_document_status(doc_id, "failed")
            raise e
