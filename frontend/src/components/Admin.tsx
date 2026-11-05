import { FormEvent, useState } from "react";
import { ApiError, listUploads, uploadDocument } from "../api";
import type { Session, UploadRow } from "../types";

export default function Admin({
  session,
  rows,
  onUpdateRows,
  onSessionExpired,
}: {
  session: Session;
  rows: UploadRow[];
  onUpdateRows: (rows: UploadRow[]) => void;
  onSessionExpired: () => void;
}) {
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      const data = await listUploads(session);
      onUpdateRows(data);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onSessionExpired();
    }
  }

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    setBusy(true);
    setMessage("");
    setError("");
    try {
      const res = await uploadDocument(session, new FormData(form));
      setMessage(`Document ${res.status === "indexed" ? "indexed" : "already up to date"} (${res.chunks} passages). No model training is needed; it is searchable now.`);
      form.reset();
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="admin">
      <form onSubmit={submit} className="card">
        <h2>Add a document</h2>
        <p className="muted">Text, Markdown, text-based PDF or Word files. Answers will label it "Admin-uploaded document". Only upload material you are allowed to use.</p>
        <label>File<input name="file" type="file" accept=".txt,.md,.pdf,.docx" required /></label>
        <label>Title<input name="title" required maxLength={200} /></label>
        <label>Source URL (optional)<input name="source_url" type="url" /></label>
        <label>Publisher (optional)<input name="publisher" /></label>
        <label>Licence (optional)<input name="licence" /></label>
        <label>Document date (optional)<input name="date" placeholder="2025-04-06" /></label>
        {message && <p className="ok-text" role="status">{message}</p>}
        {error && <p className="error" role="alert">{error}</p>}
        <button type="submit" disabled={busy}>{busy ? "Indexing…" : "Upload and index"}</button>
      </form>
      <div className="history-header">
        <h2>Upload history</h2>
        <button type="button" className="refresh-table-btn" onClick={() => void refresh()} disabled={busy} title="Load or refresh upload history">
          {rows.length > 0 ? "🔄 Refresh" : "📥 Load history"}
        </button>
      </div>
      <table>
        <thead><tr><th>When</th><th>Title</th><th>File</th><th>Passages</th><th>Status</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}><td>{r.created_at}</td><td>{r.title}</td><td>{r.filename}</td><td>{r.chunks ?? "-"}</td><td>{r.status}</td></tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={5} className="muted" style={{ textAlign: "center", padding: "16px" }}>
                Click "📥 Load history" above to view uploaded documents.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
