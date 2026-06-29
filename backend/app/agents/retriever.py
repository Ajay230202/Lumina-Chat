from app.retrieval.qdrant_store import QdrantStore
from app.ingestion.embedder import MultimodalEmbedder
from app.services.nvidia_client import NVIDIAClient
from app.config import settings

class RetrieverAgent:
    def __init__(self, qdrant_store: QdrantStore, embedder: MultimodalEmbedder, nvidia_client: NVIDIAClient):
        self.store = qdrant_store
        self.embedder = embedder
        self.nvidia = nvidia_client

    def retrieve(self, state: dict) -> dict:
        query = state.get("rewritten_query") or state["query"]
        image_b64 = state.get("user_image_b64")
        filters = state.get("filters", {})

        # Step 1: Embed query (multimodal query support)
        dense_vec = self.embedder.embed_query(query, image_b64)

        # Step 2: Hybrid search -> top-K (default 20) via Qdrant RRF
        raw_results = self.store.hybrid_search(
            dense_vector=dense_vec,
            query_text=query,
            top_k=settings.TOP_K_RETRIEVE,
            filters=filters,
        )

        # Step 3: Rerank via NVIDIA Reranker -> top-N (default 5)
        if len(raw_results) > settings.TOP_K_RERANK:
            passages = [r["text_repr"] for r in raw_results]
            scores = self.nvidia.rerank(query, passages)
            
            # Pair results with scores and sort
            ranked_results = []
            for doc, score in zip(raw_results, scores):
                doc["rerank_score"] = score
                ranked_results.append(doc)
                
            ranked_results = sorted(
                ranked_results,
                key=lambda x: x["rerank_score"],
                reverse=True
            )[:settings.TOP_K_RERANK]
            
            raw_results = ranked_results

        state["retrieved_docs"] = raw_results
        state["retrieval_count"] = state.get("retrieval_count", 0) + 1
        return state
