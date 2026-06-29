from typing import List
from app.ingestion.chunking import MultimodalChunk
from app.services.nvidia_client import NVIDIAClient

class MultimodalEmbedder:
    def __init__(self, nvidia_client: NVIDIAClient):
        self.nvidia = nvidia_client

    def embed_chunks(self, chunks: List[MultimodalChunk]) -> List[MultimodalChunk]:
        """Embed all chunk types using the text representation to ensure compatibility."""
        if not chunks:
            return chunks

        # Extract text representations
        texts = [chunk.text_repr for chunk in chunks]
        
        # Batch embed (NVIDIA Client wraps NVIDIAEmbeddings)
        embeddings = self.nvidia.embed(texts)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            
        return chunks

    def embed_query(self, query: str, image_b64: str | None = None) -> List[float]:
        """Embed a user query. If an image is present, we embed the query text.
        The image itself will be processed by the VLM in the generation/reasoning phase."""
        # Embed the query text
        embeddings = self.nvidia.embed([query])
        return embeddings[0]
