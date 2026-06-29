from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import settings

class SafetyGuard:
    """Uses NVIDIA Llama 3.1 NemoGuard Content Safety model to validate input/output safety."""

    def __init__(self):
        self.llm = ChatNVIDIA(
            model="nvidia/llama-3.1-nemoguard-8b-content-safety",
            temperature=0.1,
            max_tokens=10,
        )

    def is_safe(self, text: str) -> bool:
        """Evaluate text content safety. Returns True if safe, False otherwise."""
        if not text or not text.strip():
            return True

        try:
            # Nemoguard evaluates content safety and returns "safe" or "unsafe" classification
            response = self.llm.invoke([
                {"role": "user", "content": text}
            ])
            result = response.content.strip().lower()
            if "unsafe" in result:
                print(f"Content safety check failed. Classification: {result}")
                return False
            return True
        except Exception as e:
            print(f"NVIDIA content safety check error: {e}. Defaulting to True to avoid service disruption.")
            return True
        return True
