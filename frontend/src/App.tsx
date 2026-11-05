import { useCallback, useState } from "react";
import Admin from "./components/Admin";
import Chat from "./components/Chat";
import Login from "./components/Login";
import type { Session, UploadRow } from "./types";
import { abortAllRequests } from "./api";

const STORAGE_KEY = "pension-ai-session";

function loadSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export default function App() {
  const [session, setSession] = useState<Session | null>(loadSession);
  const [tab, setTab] = useState<"chat" | "admin">("chat");
  const [adminDocs, setAdminDocs] = useState<UploadRow[] | null>(null);

  const logout = useCallback(() => {
    // 1. Immediately abort all in-flight network requests
    abortAllRequests();
    // 2. Clear all cached session and local data completely
    sessionStorage.clear();
    localStorage.clear();
    // 3. Reset application state
    setSession(null);
    setTab("chat");
    setAdminDocs(null);
    // 4. Force a clean browser window reset to guarantee 0 carried-over state or logs
    window.location.replace(window.location.pathname);
  }, []);

  function handleLogin(s: Session) {
    sessionStorage.clear();
    localStorage.clear();
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(s));
    setSession(s);
    setTab("chat");
    setAdminDocs(null);
  }

  if (!session) return <Login onLogin={handleLogin} />;

  return (
    <div className="app" key={session.token}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-logo">🇬🇧</span>
          <div className="brand-text">
            <span className="brand-title">UK Pension AI</span>
            <span className="brand-badge">Official Sources</span>
          </div>
        </div>
        <nav className="nav-tabs">
          <button className={tab === "chat" ? "tab active" : "tab"} onClick={() => setTab("chat")}>
            💬 Assistant
          </button>
          {session.role === "admin" && (
            <button className={tab === "admin" ? "tab active" : "tab"} onClick={() => setTab("admin")}>
              📚 Knowledge Base
            </button>
          )}
        </nav>
        <div className="who">
          <span className="user-pill">
            <span className="status-dot"></span>
            {session.username} ({session.role})
          </span>
          <button className="logout-btn" onClick={logout} title="Sign out">
            Log out
          </button>
        </div>
      </header>
      <main className="main-content">
        <div className={`tab-panel ${tab === "chat" ? "active" : ""}`} role="tabpanel">
          <Chat session={session} onSessionExpired={logout} />
        </div>
        {session.role === "admin" && tab === "admin" && (
          <div className="tab-panel active" role="tabpanel">
            <Admin
              session={session}
              rows={adminDocs ?? []}
              onUpdateRows={setAdminDocs}
              onSessionExpired={logout}
            />
          </div>
        )}
      </main>
    </div>
  );
}
