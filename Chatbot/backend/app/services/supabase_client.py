from supabase import create_client, Client
from app.config import settings

class SupabaseService:
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )

    # --- Document Registry ---
    def create_document(self, filename: str, file_type: str, dept: str = "General", storage_path: str = None) -> dict:
        data = {
            "filename": filename,
            "file_type": file_type,
            "dept": dept,
            "status": "pending",
            "storage_path": storage_path
        }
        res = self.client.table("documents").insert(data).execute()
        return res.data[0] if res.data else {}

    def update_document_status(self, doc_id: str, status: str, num_chunks: int = None) -> dict:
        data = {"status": status}
        if num_chunks is not None:
            data["num_chunks"] = num_chunks
        res = self.client.table("documents").update(data).eq("id", doc_id).execute()
        return res.data[0] if res.data else {}

    def get_document_status(self, doc_id: str) -> str:
        res = self.client.table("documents").select("status").eq("id", doc_id).execute()
        if res.data:
            return res.data[0]["status"]
        return "not_found"

    def get_all_documents(self) -> list:
        res = self.client.table("documents").select("*").order("created_at", desc=True).execute()
        return res.data or []

    # --- Chunks Metadata ---
    def insert_chunks(self, chunks: list):
        # chunks is a list of dicts matching database columns
        if not chunks:
            return
        self.client.table("chunks").insert(chunks).execute()

    # --- Chat Sessions ---
    def create_session(self, user_id: str = None) -> dict:
        data = {}
        if user_id:
            data["user_id"] = user_id
        res = self.client.table("sessions").insert(data).execute()
        return res.data[0] if res.data else {}

    # --- Messages ---
    def add_message(self, session_id: str, role: str, content: str, image_b64: str = None, source_chunks: list = None) -> dict:
        data = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "image_b64": image_b64,
            "source_chunks": source_chunks or []
        }
        res = self.client.table("messages").insert(data).execute()
        return res.data[0] if res.data else {}

    def get_session_messages(self, session_id: str) -> list:
        res = self.client.table("messages").select("*").eq("session_id", session_id).order("created_at", asc=True).execute()
        return res.data or []

    def get_session_history(self, session_id: str) -> list[dict]:
        res = self.client.table("messages").select("*").eq("session_id", session_id).order("created_at").execute()
        return res.data if res.data else []
