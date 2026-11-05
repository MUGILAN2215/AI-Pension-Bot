import { useEffect, useState } from "react";
import { getPassage } from "../api";
import type { Citation, Passage, Session } from "../types";

/** Only real web links become clickable (blocks javascript: and other schemes). */
function safeHref(url: string): string | null {
  return /^https?:\/\//i.test(url) ? url : null;
}

export default function CitationDrawer({ session, citation, onClose }: { session: Session; citation: Citation; onClose: () => void }) {
  const [passage, setPassage] = useState<Passage | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setPassage(null);
    setError("");
    getPassage(session, citation.chunk_id).then(setPassage).catch((e: Error) => setError(e.message));
  }, [session, citation.chunk_id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const href = safeHref(citation.source_url);
  return (
    <aside className="drawer" role="dialog" aria-label="Source passage">
      <div className="drawer-head">
        <h2>[{citation.marker}] {citation.title}</h2>
        <button className="link" onClick={onClose} aria-label="Close">✕</button>
      </div>
      <p className="badges">
        <span className={citation.admin_uploaded ? "badge warn" : "badge ok"}>{citation.label}</span>
        <span className="badge">Last updated: {citation.last_updated ?? "not stated"}</span>
        <span className="badge">Licence: {citation.licence ?? "not stated"}</span>
      </p>
      {citation.admin_uploaded && <p className="muted">This passage comes from a document uploaded by an administrator, not from GOV.UK or HMRC.</p>}
      {citation.heading_path && <p className="muted">{citation.heading_path}</p>}
      {error && <p className="error">{error}</p>}
      {!passage && !error && <p className="muted">Loading passage…</p>}
      {passage && <pre className="passage">{passage.text}</pre>}
      {href ? <a href={href} target="_blank" rel="noopener noreferrer">Open the official page ↗</a> : <span className="muted">No public web link for this source.</span>}
    </aside>
  );
}
