import json
from typing import Literal
from app.services.nvidia_client import NVIDIAClient
from app.config import settings

GRADER_PROMPT = """You are a relevance grader for an enterprise RAG system.
Given a query and a retrieved document passage, score the relevance from 0.0 to 1.0.

Score guide:
1.0 = Directly answers the query
0.7 = Partially answers, has related information
0.4 = Tangentially related
0.0 = Irrelevant

Return ONLY a JSON: {"score": <float>, "reason": "<one sentence>"}"""

class GraderAgent:
    def __init__(self, nvidia_client: NVIDIAClient):
        self.nvidia = nvidia_client

    def grade(self, state: dict) -> dict:
        query = state["query"]
        docs = state.get("retrieved_docs", [])

        scored_docs = []
        for doc in docs:
            messages = [
                {"role": "system", "content": GRADER_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nDocument: {doc['text_repr'][:800]}"},
            ]
            
            try:
                resp = self.nvidia.generate_text(messages)
                content = resp.content.strip()
                
                # Robust markdown code block cleaning
                import re
                if "```" in content:
                    match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                
                try:
                    result = json.loads(content)
                    score = float(result.get("score", 0.0))
                except Exception:
                    # Secondary fallback: search for score in raw content
                    score_match = re.search(r'"score"\s*:\s*([0-9.]+)', content)
                    if score_match:
                        score = float(score_match.group(1))
                    else:
                        # Tertiary fallback: see if the entire response is just a float
                        score = float(content)
                
                doc["relevance_score"] = score
            except Exception as e:
                print(f"Error grading document: {e}. Defaulting score to 0.7 to prevent retrieval drop.")
                doc["relevance_score"] = 0.7

            if doc["relevance_score"] >= settings.RELEVANCE_THRESHOLD:
                scored_docs.append(doc)

        state["relevant_docs"] = scored_docs
        state["is_sufficient"] = len(scored_docs) >= 2  # Need at least 2 relevant docs
        return state

    def should_rewrite(self, state: dict) -> Literal["rewrite", "generate"]:
        """LangGraph conditional edge to decide next state node."""
        # If this is a multimodal query, skip rewriting and go straight to generation
        if state.get("route") == "multimodal":
            return "generate"

        if state.get("is_sufficient"):
            return "generate"
        
        # If we exceeded the max retrievals limit, stop looping and generate
        if state.get("retrieval_count", 0) >= settings.MAX_RETRIEVAL_RETRIES:
            # Safe fallback: take top retrieved docs and proceed
            state["relevant_docs"] = state.get("retrieved_docs", [])[:2]
            return "generate"
            
        return "rewrite"
