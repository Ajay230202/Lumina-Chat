from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END
from app.agents.router import RouterAgent
from app.agents.retriever import RetrieverAgent
from app.agents.grader import GraderAgent
from app.agents.rewriter import RewriterAgent
from app.agents.generator import GeneratorAgent

class RAGState(TypedDict):
    query: str
    user_image_b64: Optional[str]
    route: Optional[str]
    filters: dict
    rewritten_query: Optional[str]
    sub_queries: List[str]
    retrieved_docs: List[dict]
    relevant_docs: List[dict]
    is_sufficient: bool
    retrieval_count: int
    chat_history: List[dict]
    stream: Optional[Any]
    source_docs: List[dict]

def build_crag_graph(
    router: RouterAgent, 
    retriever: RetrieverAgent, 
    grader: GraderAgent, 
    rewriter: RewriterAgent, 
    generator: GeneratorAgent
):
    graph = StateGraph(RAGState)

    # 1. Define Nodes
    graph.add_node("router", router.route)
    graph.add_node("retriever", retriever.retrieve)
    graph.add_node("grader", grader.grade)
    graph.add_node("rewriter", rewriter.rewrite)
    graph.add_node("generator", generator.generate)

    # 2. Set Entry point
    graph.set_entry_point("router")

    # 3. Define Conditional router decisions
    def route_decision(state: RAGState) -> str:
        if state.get("route") == "direct":
            return "generator"
        return "retriever"

    graph.add_conditional_edges(
        "router", 
        route_decision,
        {
            "retriever": "retriever",
            "generator": "generator"
        }
    )

    # 4. retriever always goes to grader
    graph.add_edge("retriever", "grader")

    # 5. grader conditional routing (CRAG loops)
    graph.add_conditional_edges(
        "grader", 
        grader.should_rewrite,
        {
            "rewrite": "rewriter",
            "generate": "generator"
        }
    )

    # 6. rewriter routes back to retriever
    graph.add_edge("rewriter", "retriever")

    # 7. generator goes to end
    graph.add_edge("generator", END)

    return graph.compile()
