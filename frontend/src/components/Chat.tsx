import { FormEvent, Fragment, useRef, useState, useEffect } from "react";
import { ApiError, chatStream, sendFeedback } from "../api";
import type { ChatMessage, Citation, EvidenceItem, Session } from "../types";

const STARTERS = [
  { icon: "🏛️", text: "How many qualifying years do I need for the new State Pension?" },
  { icon: "💰", text: "How does Pension Credit work and who can get it?" },
  { icon: "📋", text: "Can I pay voluntary National Insurance contributions to fill gaps?" },
  { icon: "🔮", text: "How do I check my official State Pension forecast online?" },
];

const STATUS_NOTE: Record<string, string> = {
  ungrounded: "Here are the verified excerpts from official sources regarding your question:",
  retrieval_failed: "The search step failed. Please try asking again.",
};

function newId(): string {
  return crypto.randomUUID();
}

function renderCitations(text: string, citations: Citation[], onOpen: (c: Citation) => void) {
  return text.split(/(\[\d+\])/g).map((part, i) => {
    const m = /^\[(\d+)\]$/.exec(part);
    const cite = m ? citations.find((c) => c.marker === Number(m[1])) : undefined;
    return cite ? (
      <button key={i} className="cite" onClick={() => onOpen(cite)} title={`Click to view source: ${cite.title}`}>
        {part}
      </button>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    );
  });
}

function renderInlineWithCitations(text: string, citations: Citation[], onOpen: (c: Citation) => void) {
  const boldParts = text.split(/(\*\*[^*]+\*\*)/g);
  return boldParts.map((bPart, bIdx) => {
    if (bPart.startsWith("**") && bPart.endsWith("**") && bPart.length > 4) {
      const inner = bPart.slice(2, -2);
      return <strong key={bIdx}>{renderCitations(inner, citations, onOpen)}</strong>;
    }
    return <Fragment key={bIdx}>{renderCitations(bPart, citations, onOpen)}</Fragment>;
  });
}

/** Parses markdown headings, lists, bold text, and citation badges cleanly. */
function FormattedAnswer({ msg, onOpen }: { msg: ChatMessage; onOpen: (c: Citation) => void }) {
  const lines = msg.text.split("\n");
  const elements: JSX.Element[] = [];
  let currentList: string[] = [];

  function flushList() {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`} className="answer-list">
          {currentList.map((item, idx) => (
            <li key={idx}>{renderInlineWithCitations(item, msg.citations, onOpen)}</li>
          ))}
        </ul>
      );
      currentList = [];
    }
  }

  lines.forEach((line, lineIdx) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      return;
    }

    if (trimmed.startsWith("### ")) {
      flushList();
      elements.push(
        <h4 key={lineIdx} className="answer-h4">
          {renderInlineWithCitations(trimmed.slice(4), msg.citations, onOpen)}
        </h4>
      );
      return;
    }

    if (trimmed.startsWith("## ")) {
      flushList();
      elements.push(
        <h3 key={lineIdx} className="answer-h3">
          {renderInlineWithCitations(trimmed.slice(3), msg.citations, onOpen)}
        </h3>
      );
      return;
    }

    if (trimmed === "---" || trimmed === "***") {
      flushList();
      elements.push(<hr key={lineIdx} className="answer-hr" />);
      return;
    }

    if (trimmed.startsWith("* ") || trimmed.startsWith("- ") || trimmed.startsWith("• ")) {
      currentList.push(trimmed.slice(2));
      return;
    }

    flushList();
    elements.push(
      <p key={lineIdx} className="answer-p">
        {renderInlineWithCitations(trimmed, msg.citations, onOpen)}
      </p>
    );
  });

  flushList();

  return <div className="formatted-answer">{elements}</div>;
}

/** Formats raw evidence text by parsing and styling markdown headings (#, ##), bullet lists, and bold text cleanly. */
function FormattedEvidenceText({ text }: { text: string }) {
  // Normalize markdown headers that were squished together:
  // e.g. "# Voluntary National Insurance # Overview If you do not pay..."
  let cleanText = text
    .replace(/(^|\n)#{1,6}\s*/g, "\n### ")
    .replace(/(\s+)#{1,6}\s+/g, "\n\n### ")
    .trim();

  // Normalize list bullets that were squished onto single lines:
  // e.g. ": - item 1 - item 2" -> ":\n- item 1\n- item 2"
  cleanText = cleanText.replace(/:\s+-\s+/g, ":\n- ").replace(/([^\n])\s+-\s+/g, "$1\n- ");

  const lines = cleanText.split("\n");
  const elements: JSX.Element[] = [];
  let currentList: string[] = [];

  function flushList() {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`ev-ul-${elements.length}`} className="evidence-list">
          {currentList.map((item, idx) => (
            <li key={idx}>{renderInline(item)}</li>
          ))}
        </ul>
      );
      currentList = [];
    }
  }

  function renderInline(str: string) {
    const parts = str.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      return <Fragment key={i}>{part}</Fragment>;
    });
  }

  lines.forEach((line, lineIdx) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      return;
    }

    if (trimmed.startsWith("### ") || trimmed.startsWith("## ") || trimmed.startsWith("# ")) {
      flushList();
      const headingText = trimmed.replace(/^#+\s*/, "");
      elements.push(
        <h5 key={lineIdx} className="evidence-h5">
          {renderInline(headingText)}
        </h5>
      );
      return;
    }

    if (trimmed.startsWith("- ") || trimmed.startsWith("* ") || trimmed.startsWith("• ")) {
      currentList.push(trimmed.slice(2));
      return;
    }

    flushList();
    elements.push(
      <p key={lineIdx} className="evidence-p">
        {renderInline(trimmed)}
      </p>
    );
  });

  flushList();

  return <div className="evidence-text">{elements}</div>;
}

/** Format external link or local file label accurately. */
function getSourceLinkInfo(url: string | undefined, adminUploaded?: boolean): { isLink: boolean; url?: string; label: string } | null {
  if (!url) return null;
  const isHttp = url.startsWith("http://") || url.startsWith("https://");
  if (!isHttp) {
    const filename = url.replace(/^uploaded-file:/, "");
    return { isLink: false, label: `📄 Uploaded file: ${filename}` };
  }
  if (adminUploaded) {
    try {
      const hostname = new URL(url).hostname.replace(/^www\./, "");
      return { isLink: true, url, label: `Open source website (${hostname}) ↗` };
    } catch {
      return { isLink: true, url, label: "Open source website ↗" };
    }
  }
  if (url.includes("gov.uk")) {
    return { isLink: true, url, label: "Open official GOV.UK page ↗" };
  }
  if (url.includes("hmrc")) {
    return { isLink: true, url, label: "Open official HMRC page ↗" };
  }
  return { isLink: true, url, label: "Open source link ↗" };
}

/** One line per source page (several passages from the same page are grouped). */
function groupSources(citations: Citation[]): { key: string; first: Citation; markers: number[] }[] {
  const groups = new Map<string, { key: string; first: Citation; markers: number[] }>();
  for (const c of citations) {
    const key = `${c.title}|${c.source_url}`;
    const g = groups.get(key);
    if (g) g.markers.push(c.marker);
    else groups.set(key, { key, first: c, markers: [c.marker] });
  }
  return [...groups.values()];
}

/** Unified Right Panel displaying all Top-K retrieved chunks and highlighting clicked citations. */
function RightEvidencePanel({
  selected,
  highlightedChunkId,
  onClose,
}: {
  selected: { query: string; evidence: EvidenceItem[] } | null;
  highlightedChunkId: string | null;
  onClose: () => void;
}) {
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});

  useEffect(() => {
    if (highlightedChunkId && cardRefs.current[highlightedChunkId]) {
      cardRefs.current[highlightedChunkId]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [highlightedChunkId]);

  if (!selected || selected.evidence.length === 0) return null;

  return (
    <aside className="right-evidence-panel" role="region" aria-label="Retrieved Evidence">
      <div className="panel-head">
        <div>
          <div className="panel-title-row">
            <span className="panel-icon">📑</span>
            <h3>Retrieved Evidence</h3>
            <span className="count-pill">{selected.evidence.length} sources</span>
          </div>
          <p className="panel-sub">
            Query: <em>"{selected.query}"</em>
          </p>
        </div>
        <button className="icon-btn close-btn" onClick={onClose} aria-label="Close evidence panel" title="Close panel">
          ✕
        </button>
      </div>

      <div className="panel-body">
        <div className="evidence-cards">
          {selected.evidence.map((ev) => {
            const isHighlighted = highlightedChunkId === ev.chunk_id;
            return (
              <div
                key={ev.chunk_id}
                ref={(el) => {
                  cardRefs.current[ev.chunk_id] = el;
                }}
                className={`evidence-card${isHighlighted ? " highlighted" : ""}`}
                id={`chunk-${ev.chunk_id}`}
              >
                <div className="evidence-meta">
                  <span className="badge rank-badge">#{ev.rank} Match</span>
                  <span className="badge score-badge" title="Semantic meaning similarity">Semantic: {typeof ev.score === 'number' ? `${Math.round(ev.score * 100)}%` : ev.score}</span>
                  <span className="badge source-badge">{ev.label || "GOV.UK"}</span>
                  {ev.last_updated && <span className="badge date-badge">Updated: {ev.last_updated}</span>}
                </div>
                <h4 className="evidence-title">{ev.title}</h4>
                {ev.heading_path && <p className="evidence-heading">{ev.heading_path}</p>}
                <FormattedEvidenceText text={ev.text} />
                {(() => {
                  const link = getSourceLinkInfo(ev.source_url, ev.admin_uploaded);
                  if (!link) return null;
                  return (
                    <div className="evidence-actions">
                      {link.isLink ? (
                        <a href={link.url} target="_blank" rel="noopener noreferrer" className="govuk-link">
                          {link.label}
                        </a>
                      ) : (
                        <span className="file-source-tag">{link.label}</span>
                      )}
                    </div>
                  );
                })()}
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

const CHAT_STORAGE_KEY = "pension-ai-chat-messages";
const EVIDENCE_STORAGE_KEY = "pension-ai-selected-evidence";

function loadCachedMessages(): ChatMessage[] {
  try {
    const raw = sessionStorage.getItem(CHAT_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function loadCachedEvidence(): { query: string; evidence: EvidenceItem[] } | null {
  try {
    const raw = sessionStorage.getItem(EVIDENCE_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export default function Chat({ session, onSessionExpired }: { session: Session; onSessionExpired: () => void }) {
  const [messages, setMessages] = useState<ChatMessage[]>(loadCachedMessages);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<{ query: string; evidence: EvidenceItem[] } | null>(loadCachedEvidence);
  const [showEvidence, setShowEvidence] = useState<boolean>(() => !!loadCachedEvidence());
  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      const clean = messages.filter((m) => !m.pending);
      sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(clean));
    } catch {
      /* ignore quota errors */
    }
  }, [messages]);

  useEffect(() => {
    try {
      if (selectedEvidence) {
        sessionStorage.setItem(EVIDENCE_STORAGE_KEY, JSON.stringify(selectedEvidence));
      } else {
        sessionStorage.removeItem(EVIDENCE_STORAGE_KEY);
      }
    } catch {
      /* ignore */
    }
  }, [selectedEvidence]);

  function clearChat() {
    setMessages([]);
    setSelectedEvidence(null);
    setShowEvidence(false);
    setHighlightedChunkId(null);
    sessionStorage.removeItem(CHAT_STORAGE_KEY);
    sessionStorage.removeItem(EVIDENCE_STORAGE_KEY);
  }

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleOpenCitation(c: Citation) {
    setHighlightedChunkId(c.chunk_id);
    setShowEvidence(true);
  }

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    const history = messages.filter((m) => !m.pending).slice(-6).map((m) => ({ role: m.role, content: m.text }));
    const reply = newId();
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", text: q, citations: [], notes: [] },
      { id: reply, role: "assistant", text: "", citations: [], notes: [], pending: true },
    ]);
    setInput("");
    setBusy(true);
    const patch = (fn: (m: ChatMessage) => ChatMessage) => setMessages((prev) => prev.map((m) => (m.id === reply ? fn(m) : m)));
    try {
      await chatStream(session, q, history, {
        onMeta: (meta) =>
          patch((m) => ({
            ...m,
            messageId: meta.message_id,
            status: meta.status,
            mode: meta.mode,
            rewritten: meta.rewritten_question,
            route: meta.route,
            steps: meta.steps,
          })),
        onToken: (t) => patch((m) => ({ ...m, text: m.text + t })),
        onCitations: (citations, notes, rawReply, suggestions, evidence) => {
          patch((m) => ({ ...m, citations, notes, rawReply, suggestions, evidence }));
          if (evidence && evidence.length > 0) {
            setSelectedEvidence({ query: q, evidence });
            setShowEvidence(true);
            // Default highlight to first citation chunk if present
            if (citations.length > 0) {
              setHighlightedChunkId(citations[0].chunk_id);
            }
          }
        },
      });
      patch((m) => ({ ...m, pending: false }));
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) return onSessionExpired();
      patch((m) => ({
        ...m,
        pending: false,
        text: err instanceof ApiError ? `Sorry - ${err.message}` : "Sorry - the server could not be reached. Please try again.",
        status: "error",
      }));
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void ask(input);
  }

  async function rate(m: ChatMessage, rating: "up" | "down") {
    try {
      await sendFeedback(session, m.messageId ?? m.id, rating);
      setMessages((prev) => prev.map((x) => (x.id === m.id ? { ...x, feedback: rating } : x)));
    } catch {
      /* feedback is best-effort */
    }
  }

  return (
    <div className="chat-workspace">
      <div className="chat-main">
        {messages.length > 0 && (
          <div className="chat-top-actions">
            {!showEvidence && selectedEvidence && selectedEvidence.evidence.length > 0 ? (
              <button
                type="button"
                className="evidence-toggle-btn"
                onClick={() => setShowEvidence(true)}
                title="Open evidence panel"
              >
                📑 Show Evidence ({selectedEvidence.evidence.length})
              </button>
            ) : <span />}
            <button
              type="button"
              className="new-chat-btn"
              onClick={clearChat}
              title="Clear current messages and start fresh"
            >
              🗑️ Clear Chat
            </button>
          </div>
        )}

        <div className="chat">
          <div className="thread">
            {messages.length === 0 && (
              <section className="starters">
                <div className="hero-banner">
                  <h2>👋 Hello! I'm your UK Pension Helper</h2>
                  <p className="hero-desc">
                    Pensions can feel complicated with confusing rules and jargon. I'm here to explain everything
                    in <strong>simple, friendly English</strong> that anyone can understand — backed by official GOV.UK and HMRC sources.
                  </p>
                </div>
                <p className="starter-heading">Popular questions to get started:</p>
                <div className="cards">
                  {STARTERS.map((s) => (
                    <button key={s.text} className="card starter" onClick={() => void ask(s.text)}>
                      <span className="card-icon">{s.icon}</span>
                      <span>{s.text}</span>
                    </button>
                  ))}
                </div>
              </section>
            )}

            {messages.map((m) => (
              <div key={m.id} className={`msg-row ${m.role}-row`}>
                {m.role === "user" ? (
                  <div className="user-bubble">
                    <p>{m.text}</p>
                  </div>
                ) : (
                  <div className={`assistant-card${m.status === "insufficient_evidence" ? " refusal" : ""}`}>
                    <div className="agent-header">
                      <div className="agent-info">
                        <span className="agent-avatar">🏛️</span>
                        <div>
                          <strong className="agent-name">UK Pension Assistant</strong>
                          <span className="agent-subtitle">
                            {m.citations.some((c) => c.admin_uploaded)
                              ? "Official & Admin-Uploaded Sources"
                              : "Verified GOV.UK & HMRC Sources"}
                          </span>
                        </div>
                      </div>
                      {m.route && (
                        <span className="route-badge">
                          {m.route === "calculator"
                            ? "🧮 NI Calculator"
                            : m.citations.some((c) => c.admin_uploaded)
                            ? "📑 Knowledge Base Source"
                            : "📖 Official Guidance"}
                        </span>
                      )}
                    </div>

                    {m.status && STATUS_NOTE[m.status] && <p className="notice">{STATUS_NOTE[m.status]}</p>}
                    <FormattedAnswer msg={m} onOpen={handleOpenCitation} />

                    {m.pending && (
                      <div className="typing-indicator">
                        <span className="dot"></span>
                        <span className="dot"></span>
                        <span className="dot"></span>
                        <span className="typing-text">Reviewing official guidance…</span>
                      </div>
                    )}

                    {m.citations.length > 0 && (
                      <div className="sources-container">
                        <span className="sources-label">Official Sources Cited:</span>
                        <ul className="sources-list">
                          {groupSources(m.citations).map((g) => (
                            <li key={g.key} className="source-item">
                              <button className="source-btn" onClick={() => handleOpenCitation(g.first)}>
                                <span className="source-markers">{g.markers.map((n) => `[${n}]`).join("")}</span>
                                <span className="source-title">{g.first.title}</span>
                              </button>
                              <span className={g.first.admin_uploaded ? "badge warn" : "badge ok"}>{g.first.label}</span>
                              <span className="badge date">Updated: {g.first.last_updated ?? "not stated"}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {!m.pending && m.evidence && m.evidence.length > 0 && (
                      <div className="evidence-pill-row">
                        <button
                          type="button"
                          className="view-evidence-pill"
                          onClick={() => {
                            setSelectedEvidence({ query: m.rewritten || m.text, evidence: m.evidence || [] });
                            setShowEvidence(true);
                          }}
                        >
                          📑 View {m.evidence.length} verified passages in side panel
                        </button>
                      </div>
                    )}

                    {!m.pending && m.status === "ok" && m.citations.length > 0 && (
                      <div className="quick-actions">
                        <button
                          type="button"
                          className="quick-btn"
                          onClick={() => void ask("Can you explain that in even simpler terms with a simple example?")}
                        >
                          ✨ Explain simpler
                        </button>
                        <button
                          type="button"
                          className="quick-btn"
                          onClick={() => void ask("Can you give me a realistic example of how this works?")}
                        >
                          💡 Give an example
                        </button>
                      </div>
                    )}

                    {!m.pending && m.suggestions && m.suggestions.length > 0 && (
                      <div className="suggestions">
                        <p className="suggestions-title">💡 Helpful follow-ups:</p>
                        <div className="chips">
                          {m.suggestions.map((s) => (
                            <button key={s} type="button" className="chip" onClick={() => void ask(s)}>
                              <span>{s}</span>
                              <span className="chip-arrow">→</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {!m.pending && m.notes.length > 0 && (
                      <details className="details">
                        <summary>Notes about these sources ({m.notes.length})</summary>
                        {m.notes.map((n) => <p key={n} className="muted">{n}</p>)}
                      </details>
                    )}

                    {!m.pending && m.route && (
                      <details className="details">
                        <summary>How this answer was made</summary>
                        <p className="muted">Question used for search: {m.rewritten ?? "-"}</p>
                        <p className="muted">
                          Route: {m.route === "calculator" ? "calculator + search" : m.route} · answer writer:{" "}
                          {m.mode === "generated" ? "language model" : "quoted passages"}
                        </p>
                        {m.steps && <p className="muted">{m.steps.join(" → ")}</p>}
                        {m.rawReply && <p className="muted">Model reply (debug): {m.rawReply}</p>}
                      </details>
                    )}

                    {!m.pending && m.status && m.status !== "error" && (
                      <div className="feedback-row">
                        <span className="feedback-label">Was this explanation helpful?</span>
                        <div className="feedback-buttons">
                          <button
                            className={`feedback-btn${m.feedback === "up" ? " active" : ""}`}
                            disabled={!!m.feedback}
                            onClick={() => void rate(m, "up")}
                            aria-label="Helpful"
                          >
                            👍 Yes
                          </button>
                          <button
                            className={`feedback-btn${m.feedback === "down" ? " active" : ""}`}
                            disabled={!!m.feedback}
                            onClick={() => void rate(m, "down")}
                            aria-label="Not helpful"
                          >
                            👎 No
                          </button>
                        </div>
                        {m.feedback && <span className="feedback-thanks">Thank you for your feedback!</span>}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={endRef} />
          </div>

          <form className="composer-container" onSubmit={onSubmit}>
            <div className="composer-bar">
              <span className="search-icon">💬</span>
              <input
                id="question"
                name="question"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about State Pension, Pension Credit, NI qualifying years…"
                maxLength={1000}
                aria-label="Your question"
                autoComplete="off"
              />
              <button type="submit" className="send-btn" disabled={busy || !input.trim()}>
                <span>Ask</span>
                <span className="send-arrow">→</span>
              </button>
            </div>
            <p className="disclaimer">
              🛡️ Based on official GOV.UK and HMRC data. This is general guidance, not personal financial, tax, or legal advice.
            </p>
          </form>
        </div>
      </div>

      {showEvidence && messages.length > 0 && selectedEvidence && selectedEvidence.evidence.length > 0 && (
        <RightEvidencePanel
          selected={selectedEvidence}
          highlightedChunkId={highlightedChunkId}
          onClose={() => setShowEvidence(false)}
        />
      )}
    </div>
  );
}
