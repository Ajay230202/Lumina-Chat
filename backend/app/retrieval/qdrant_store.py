from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from app.config import settings
from app.ingestion.chunking import MultimodalChunk

DENSE_DIM = 2048  # NVIDIA llama-nemotron-embed-vl-1b-v2 output dimension

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION
        # Initialize FastEmbed BM25 model once to cache it
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=DENSE_DIM,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
            # Create indexes for fast metadata filtering
            for field in ["doc_id", "modality", "dept", "file_type"]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def upsert(self, chunks: list[MultimodalChunk]):
        points = []
        for c in chunks:
            # Generate sparse vector
            sparse_enc = self._bm25_encode(c.text_repr)
            points.append(
                models.PointStruct(
                    id=c.chunk_id,
                    vector={
                        "dense": c.embedding,
                        "bm25": models.SparseVector(
                            indices=sparse_enc["indices"],
                            values=sparse_enc["values"]
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
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def hybrid_search(
        self,
        dense_vector: list[float],
        query_text: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[dict]:
        """BM25 sparse + dense semantic → RRF fusion query."""
        qdrant_filter = self._build_filter(filters)
        sparse_vec = self._bm25_encode(query_text)

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"]
                    ),
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
            {
                "id": p.id,
                "score": p.score,
                **p.payload
            }
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
        result = list(self.sparse_model.embed([text]))[0]
        return {
            "indices": result.indices.tolist(),
            "values": result.values.tolist()
        }
