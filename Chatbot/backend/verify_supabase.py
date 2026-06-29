import os
import sys

backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import dotenv
dotenv.load_dotenv(os.path.join(backend_path, ".env"))

from app.services.supabase_client import SupabaseService

def main():
    print("Testing Supabase connection...")
    try:
        service = SupabaseService()
        # Query documents table
        res = service.client.table("documents").select("*").limit(1).execute()
        print("Supabase connection success! Query result:", res.data)
    except Exception as e:
        print("Supabase query failed. This might mean the tables are not created yet.")
        print("Error details:", e)

if __name__ == "__main__":
    main()
