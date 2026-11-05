export interface Citation {
  marker: number;
  chunk_id: string;
  document_id: string;
  title: string;
  source_url: string;
  heading_path: string;
  last_updated: string | null;
  licence: string | null;
  label: string;
  admin_uploaded: boolean;
}

export interface EvidenceItem {
  rank: number;
  score: number;
  chunk_id: string;
  document_id: string;
  title: string;
  source_url: string;
  heading_path: string;
  text: string;
  last_updated?: string | null;
  label?: string;
  admin_uploaded?: boolean;
}

export interface ChatMessage {
  id: string;
  messageId?: string;
  role: "user" | "assistant";
  text: string;
  citations: Citation[];
  evidence?: EvidenceItem[];
  notes: string[];
  status?: string;
  mode?: string;
  rewritten?: string;
  route?: string;
  steps?: string[];
  suggestions?: string[];
  rawReply?: string | null;
  pending?: boolean;
  feedback?: "up" | "down";
}

export interface Session {
  token: string;
  username: string;
  role: "user" | "admin";
}

export interface Passage {
  chunk_id: string;
  title: string;
  source_url: string;
  heading_path: string;
  text: string;
  licence: string | null;
  publisher: string | null;
  last_updated: string | null;
  label: string;
  admin_uploaded: boolean;
}

export interface UploadRow {
  id: number;
  filename: string;
  title: string;
  document_id: string;
  chunks: number | null;
  status: string;
  created_at: string;
}

export interface StreamHandlers {
  onMeta: (meta: { message_id: string; status: string; mode: string; rewritten_question?: string; route?: string; steps?: string[] }) => void;
  onToken: (text: string) => void;
  onCitations: (citations: Citation[], notes: string[], rawReply?: string | null, suggestions?: string[], evidence?: EvidenceItem[]) => void;
}
