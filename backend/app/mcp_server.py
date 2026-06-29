from mcp.server.fastmcp import FastMCP
from app.retrieval.qdrant_store import QdrantStore
from app.ingestion.embedder import MultimodalEmbedder
from app.services.nvidia_client import NVIDIAClient
from app.services.supabase_client import SupabaseService

# Create the FastMCP server
mcp = FastMCP("Lumina RAG Server")

# Initialize RAG components
nvidia_client = NVIDIAClient()
qdrant_store = QdrantStore()
embedder = MultimodalEmbedder(nvidia_client)
supabase = SupabaseService()

@mcp.tool()
def list_documents() -> str:
    """List all uploaded documents in the Lumina RAG knowledge base, including their department and ingestion status."""
    try:
        docs = supabase.get_all_documents()
        formatted_docs = []
        for doc in docs:
            formatted_docs.append(
                f"- {doc['filename']} (Type: {doc['file_type']}, Dept: {doc['dept']}, Status: {doc['status']}, ID: {doc['id']})"
            )
        return "Uploaded Documents:\n" + ("\n".join(formatted_docs) if formatted_docs else "No documents found.")
    except Exception as e:
        return f"Error listing documents: {str(e)}"

@mcp.tool()
def query_knowledge_base(query: str, dept: str = None) -> str:
    """Query the Lumina RAG knowledge base for text segments, tables, and visual captions related to the query.
    
    Args:
        query: The search query to retrieve relevant contexts for.
        dept: Optional department filter (General, HR, Finance, Policy, Legal).
    """
    try:
        # 1. Generate dense query vector
        query_vector = embedder.embed_text(query)
        
        # 2. Extract department filter if specified
        filter_dict = {}
        if dept:
            filter_dict["dept"] = dept
            
        # 3. Query Qdrant
        results = qdrant_store.query(
            query_text=query,
            dense_vector=query_vector,
            filters=filter_dict,
            limit=5
        )
        
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append(
                f"Result {i} (Score: {r['score']:.3f}, Modality: {r['modality']}, Page: {r.get('page_num', 'N/A')}):\n{r['text_repr']}\n"
            )
        
        return "Retrieved Contexts:\n\n" + ("\n---\n\n".join(formatted_results) if formatted_results else "No matches found.")
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"
