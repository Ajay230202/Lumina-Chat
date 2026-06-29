export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  image_b64?: string;
  sources?: Source[];
}

export interface Source {
  chunk_id: string;
  modality: string;
  text_repr: string;
  page_num?: number;
  score?: number;
}
