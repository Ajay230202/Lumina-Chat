from typing import Literal
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import settings

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

    def __init__(self, nvidia_client=None):
        self.nvidia = nvidia_client
        if not self.nvidia:
            from app.services.nvidia_client import NVIDIAClient
            self.nvidia = NVIDIAClient()

    def route(self, state: dict) -> dict:
        query = state["query"]
        has_image = bool(state.get("user_image_b64"))

        # If user attached an image, bypass LLM routing and route to multimodal
        if has_image:
            state["route"] = "multimodal"
            return state

        # Invoke model
        response = self.nvidia.generate_text([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ])
        
        route = response.content.strip().lower()
        
        # Guard rails for output
        if route not in ("simple", "complex", "multimodal", "direct"):
            route = "simple"  # safe default

        state["route"] = route
        state["filters"] = self._extract_filters(query)
        return state

    def _extract_filters(self, query: str) -> dict:
        """Heuristic to detect department mentions and construct Qdrant filter."""
        filters = {}
        dept_keywords = {
            "hr": "HR", 
            "finance": "Finance",
            "policy": "Policy", 
            "legal": "Legal"
        }
        for kw, dept in dept_keywords.items():
            if kw in query.lower():
                filters["dept"] = dept
        return filters
