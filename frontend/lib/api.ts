import { Message, Source } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function streamChat(
  query: string,
  history: Message[],
  imageB64?: string,
  onChunk: (text: string) => void,
  onSources: (sources: Source[]) => void,
  onError: (err: string) => void,
  sessionId?: string,
  signal?: AbortSignal
) {
  try {
    const formattedHistory = history.map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    const response = await fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        history: formattedHistory,
        image_b64: imageB64,
        session_id: sessionId
      }),
      signal
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Response body reader not available");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      
      // Save the last incomplete chunk back to the buffer
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        if (!line.startsWith("data: ")) continue;

        const dataStr = line.slice(6).trim();
        if (dataStr === "[DONE]") return;

        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.type === "text") {
            onChunk(parsed.content);
          } else if (parsed.type === "sources") {
            onSources(parsed.sources);
          } else if (parsed.type === "error") {
            onError(parsed.content);
          }
        } catch (e) {
          console.error("Error parsing SSE line:", e, line);
        }
      }
    }
  } catch (error: any) {
    onError(error.message || "Failed to communicate with API server");
  }
}

export async function uploadFile(file: File, dept: string = "General"): Promise<{ doc_id: string; status: string }> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("dept", dept);

  const response = await fetch(`${API_URL}/api/ingest`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorDetails = await response.json().catch(() => ({ detail: 'Failed to upload' }));
    throw new Error(errorDetails.detail || "Upload failed");
  }

  return response.json();
}

export async function getIngestStatus(docId: string): Promise<{ doc_id: string; status: string }> {
  const response = await fetch(`${API_URL}/api/ingest/${docId}/status`);
  if (!response.ok) {
    throw new Error("Failed to fetch ingestion status");
  }
  return response.json();
}

export async function getDocuments(): Promise<any[]> {
  const response = await fetch(`${API_URL}/api/documents`);
  if (!response.ok) {
    throw new Error("Failed to fetch documents");
  }
  return response.json();
}

export async function createSession(): Promise<{ id: string }> {
  const response = await fetch(`${API_URL}/api/sessions`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error("Failed to create session");
  }
  return response.json();
}

export async function getSessionHistory(sessionId: string): Promise<any[]> {
  const response = await fetch(`${API_URL}/api/sessions/${sessionId}/history`);
  if (!response.ok) {
    throw new Error("Failed to fetch session history");
  }
  return response.json();
}
