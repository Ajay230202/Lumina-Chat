import os
import sys

# Add backend directory to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import dotenv
dotenv.load_dotenv(os.path.join(backend_path, ".env"))

from app.services.nvidia_client import NVIDIAClient
from app.config import settings

def main():
    print("API Key loaded:", settings.NVIDIA_API_KEY[:8] + "..." + settings.NVIDIA_API_KEY[-8:])
    print("Initializing client...")
    client = NVIDIAClient()
    
    # 1. Test Chat Generation
    print("\n--- Testing Chat Generation ---")
    try:
        response = client.generate([{"role": "user", "content": "Say hello in 3 words."}], stream=False)
        print("Response:", response.content.strip())
    except Exception as e:
        print("Chat generation failed:", e)

    # 2. Test Embedding
    print("\n--- Testing Embedding ---")
    try:
        embs = client.embed(["NVIDIA NIM is awesome."])
        print(f"Embedding success! Vector dimensions: {len(embs[0])}")
    except Exception as e:
        print("Embedding failed:", e)

    # 3. Test Reranking
    print("\n--- Testing Reranking ---")
    try:
        scores = client.rerank(
            "What is Antigravity?", 
            ["Antigravity is an AI agent.", "Pineapples grow on trees.", "Another statement that has nothing to do with gravity."]
        )
        print("Rerank success! Scores:", scores)
    except Exception as e:
        print("Reranking failed:", e)

if __name__ == "__main__":
    main()
