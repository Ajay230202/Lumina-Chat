import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from app.config import settings

# Set environment variable so the LangChain endpoints pick it up
os.environ["NVIDIA_API_KEY"] = settings.NVIDIA_API_KEY

class NVIDIAClient:
    """Unified wrapper for all NVIDIA NIM endpoints using langchain-nvidia-ai-endpoints."""

    def __init__(self):
        self.models = {
            "generator":  "meta/llama-3.2-11b-vision-instruct",
            "text_llm":   "meta/llama-3.1-8b-instruct",
            "embedder":   "nvidia/llama-nemotron-embed-vl-1b-v2",
            "reranker":   "nvidia/llama-nemotron-rerank-1b-v2",
            "safety":     "nvidia/nemotron-3.5-content-safety",
        }
        
        self.llm = ChatNVIDIA(
            model=self.models["generator"],
            temperature=0.1,
            max_tokens=1500,
        )

        self.text_llm = ChatNVIDIA(
            model=self.models["text_llm"],
            temperature=0.1,
            max_tokens=500,
        )
        
        self.embedder = NVIDIAEmbeddings(
            model=self.models["embedder"]
        )
        
        self.reranker = NVIDIARerank(
            model=self.models["reranker"],
        )

    def embed(self, inputs: list[str]) -> list[list[float]]:
        """Embed a list of text passages."""
        return self.embedder.embed_documents(inputs)

    def generate(self, messages: list[dict], stream: bool = True):
        """Invoke the chat completions endpoint using LangChain messages."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        lc_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                # For multimodal input, LangChain HumanMessage can handle lists of dicts
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
                
        if stream:
            return self.llm.stream(lc_messages)
        else:
            return self.llm.invoke(lc_messages)

    def generate_text(self, messages: list[dict]):
        """Invoke the fast text-only model for grading, routing, etc."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        lc_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        return self.text_llm.invoke(lc_messages)

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Rank a list of passages against a query and return their relevance scores."""
        from langchain_core.documents import Document
        docs = [Document(page_content=p, metadata={"index": i}) for i, p in enumerate(passages)]
        compressed = self.reranker.compress_documents(docs, query)
        
        # Map scores back to original passages index order
        scores = [0.0] * len(passages)
        for doc in compressed:
            idx = doc.metadata["index"]
            # LangChain NVIDIARerank populates relevance score in metadata under 'relevance_score'
            score = doc.metadata.get("relevance_score", 0.0)
            scores[idx] = score
        return scores
