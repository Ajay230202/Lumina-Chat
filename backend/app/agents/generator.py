from app.services.nvidia_client import NVIDIAClient

SYSTEM_PROMPT = """You are an intelligent, helpful enterprise assistant with advanced vision and document capabilities.

Guidelines:
1. IMAGE ANALYSIS: If the user has attached an image, diagram, chart, or drawing, prioritize describing, analyzing, and answering the question based on the visual contents of the image. 
2. RETRIEVED CONTEXT: Use the provided document context to assist with answering. However, if the retrieved document context is unrelated to the attached image (e.g., resumes vs technical design diagrams), focus purely on the visual content of the image to answer. Do not describe or hallucinate based on the unrelated document context.
3. CITATIONS: When using document context, cite your sources as [Source 1], [Source 2] etc.
4. REPRESENTATION: If the user asks for a chart, table, or visual representation, create a clean, formatted Markdown table or visual text diagram summarizing the key data points, metrics, and comparisons.
5. GENERAL: If referring to 'the person', 'the candidate', or 'the author', relate it directly to the subject of the document context."""

class GeneratorAgent:
    def __init__(self, nvidia_client: NVIDIAClient):
        self.nvidia = nvidia_client

    def generate(self, state: dict) -> dict:
        query = state["query"]
        docs = state.get("relevant_docs") or state.get("retrieved_docs") or []
        user_image = state.get("user_image_b64")
        history = state.get("chat_history", [])[-4:]  # Limit to last 2 turns

        # Assemble sources context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            ctx = f"[Source {i}] (modality: {doc['modality']})\n{doc['text_repr']}"
            context_parts.append(ctx)
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context found."

        # Setup message structure (multimodal if user uploaded an image)
        user_content = [{"type": "text", "text": f"Context:\n{context}\n\nQuestion: {query}"}]
        if user_image:
            image_url = user_image if user_image.startswith("data:") else f"data:image/jpeg;base64,{user_image}"
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        
        # Add history in user/assistant format
        for msg in history:
            messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })

        # Add current query
        messages.append({"role": "user", "content": user_content})

        # Generate stream
        stream = self.nvidia.generate(messages, stream=True)
        state["stream"] = stream
        state["source_docs"] = docs
        return state
