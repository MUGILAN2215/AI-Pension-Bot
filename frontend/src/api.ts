import type { Citation, Passage, Session, StreamHandlers, UploadRow } from "./types";

const BASE: string = (import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

const activeControllers = new Set<AbortController>();

/** Immediately aborts all active in-flight network requests (called on logout or session reset). */
export function abortAllRequests(): void {
  for (const c of activeControllers) {
    try {
      c.abort();
    } catch {
      /* ignore */
    }
  }
  activeControllers.clear();
}

function makeTrackingController(externalSignal?: AbortSignal): { controller: AbortController; cleanup: () => void } {
  const controller = new AbortController();
  activeControllers.add(controller);
  const onAbort = () => controller.abort();
  if (externalSignal) {
    if (externalSignal.aborted) controller.abort();
    else externalSignal.addEventListener("abort", onAbort, { once: true });
  }
  const cleanup = () => {
    activeControllers.delete(controller);
    if (externalSignal) externalSignal.removeEventListener("abort", onAbort);
  };
  return { controller, cleanup };
}

async function errorFrom(res: Response): Promise<ApiError> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* keep statusText */
  }
  return new ApiError(res.status, detail);
}

function authHeaders(s: Session): Record<string, string> {
  return { Authorization: `Bearer ${s.token}` };
}

export async function login(username: string, password: string): Promise<Session> {
  const { controller, cleanup } = makeTrackingController();
  try {
    const res = await fetch(`${BASE}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
      signal: controller.signal,
    });
    if (!res.ok) throw await errorFrom(res);
    const d = await res.json();
    return { token: d.access_token, username: d.username, role: d.role };
  } finally {
    cleanup();
  }
}

/** Sends a question and reads the Server-Sent Events reply (fetch + stream; EventSource cannot POST). */
export async function chatStream(
  s: Session,
  question: string,
  history: { role: string; content: string }[],
  h: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const { controller, cleanup } = makeTrackingController(signal);
  try {
    const res = await fetch(`${BASE}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(s) },
      body: JSON.stringify({ question, history }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw await errorFrom(res);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const event = /^event: (.*)$/m.exec(block)?.[1];
        const data = /^data: (.*)$/m.exec(block)?.[1];
        if (!event || data === undefined) continue;
        const payload = JSON.parse(data);
        if (event === "meta") h.onMeta(payload);
        else if (event === "token") h.onToken(payload.text as string);
        else if (event === "citations") h.onCitations(payload.citations as Citation[], payload.notes as string[], (payload.raw_reply as string | null | undefined) ?? null, payload.suggestions as string[] | undefined, payload.evidence as any);
      }
    }
  } catch (err: any) {
    if (err.name === "AbortError") {
      return; // gracefully stopped on logout / cancel
    }
    throw err;
  } finally {
    cleanup();
  }
}

export async function getPassage(s: Session, chunkId: string): Promise<Passage> {
  const { controller, cleanup } = makeTrackingController();
  try {
    const res = await fetch(`${BASE}/api/chunks/${encodeURIComponent(chunkId)}`, {
      headers: authHeaders(s),
      signal: controller.signal,
    });
    if (!res.ok) throw await errorFrom(res);
    return await res.json();
  } catch (err: any) {
    if (err.name === "AbortError") throw new Error("Request cancelled");
    throw err;
  } finally {
    cleanup();
  }
}

export async function sendFeedback(s: Session, messageId: string, rating: "up" | "down", chunkId?: string): Promise<void> {
  const { controller, cleanup } = makeTrackingController();
  try {
    const res = await fetch(`${BASE}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(s) },
      body: JSON.stringify({ message_id: messageId, rating, chunk_id: chunkId ?? null }),
      signal: controller.signal,
    });
    if (!res.ok) throw await errorFrom(res);
  } catch (err: any) {
    if (err.name === "AbortError") return;
    throw err;
  } finally {
    cleanup();
  }
}

export async function uploadDocument(s: Session, form: FormData): Promise<{ status: string; chunks: number; document_id: string }> {
  const { controller, cleanup } = makeTrackingController();
  try {
    const res = await fetch(`${BASE}/api/admin/documents`, {
      method: "POST",
      headers: authHeaders(s),
      body: form,
      signal: controller.signal,
    });
    if (!res.ok) throw await errorFrom(res);
    return await res.json();
  } catch (err: any) {
    if (err.name === "AbortError") throw new Error("Upload cancelled");
    throw err;
  } finally {
    cleanup();
  }
}

export async function listUploads(s: Session): Promise<UploadRow[]> {
  const { controller, cleanup } = makeTrackingController();
  try {
    const res = await fetch(`${BASE}/api/admin/documents`, {
      headers: authHeaders(s),
      signal: controller.signal,
    });
    if (!res.ok) throw await errorFrom(res);
    return await res.json();
  } catch (err: any) {
    if (err.name === "AbortError") return [];
    throw err;
  } finally {
    cleanup();
  }
}
