import os
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

from app.utils import generate_uuid

class AudioPipeline:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
        )

    def process(self, audio_path: str, doc_id: str) -> list[dict]:
        # Graceful fallback if Groq API key is missing or a placeholder
        if not self.api_key or "placeholder" in self.api_key.lower() or not self.api_key.strip():
            print("Groq API key not provided. Using fallback transcript.")
            fallback_text = (
                f"Audio file {os.path.basename(audio_path)} processed. "
                "Configure GROQ_API_KEY to enable high-fidelity Whisper transcription."
            )
            return [
                {
                    "chunk_id": generate_uuid(f"audio_{doc_id}_0"),
                    "modality": "audio_transcript",
                    "text_repr": fallback_text,
                    "source_type": "audio",
                    "doc_id": doc_id,
                }
            ]

        client = Groq(api_key=self.api_key)
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f),
                model="whisper-large-v3",
                response_format="verbose_json",
            )

        # Reconstruct full transcript text from segments
        segments = getattr(transcription, "segments", [])
        if segments:
            full_text = " ".join(s.get("text", s.get("text_repr", "")) if isinstance(s, dict) else getattr(s, "text", "") for s in segments)
        else:
            full_text = getattr(transcription, "text", "")

        chunks = self.splitter.split_text(full_text)

        return [
            {
                "chunk_id": generate_uuid(f"audio_{doc_id}_{i}"),
                "modality": "audio_transcript",
                "text_repr": chunk,
                "source_type": "audio",
                "doc_id": doc_id,
            }
            for i, chunk in enumerate(chunks)
        ]
