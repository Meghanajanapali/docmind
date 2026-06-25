import { useEffect, useRef, useState } from "react"
import ReactMarkdown from "react-markdown"
import "./App.css"

const API_BASE = "http://127.0.0.1:8000"

function App() {

  const [currentPage, setCurrentPage] = useState("home")

  // Notes
  const [notes, setNotes] = useState([])
  const [input, setInput] = useState("")

  // Chat
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState([])
  const [isTyping, setIsTyping] = useState(false)
  const [typingSessionId, setTypingSessionId] = useState(null)

  // Upload
  const [uploadStatus, setUploadStatus] = useState("")
  const [isUploading, setIsUploading] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  // Auth
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [loggedIn, setLoggedIn] = useState(!!localStorage.getItem("token"))
  const [isSignup, setIsSignup] = useState(false)

  // Documents (all user docs across all sessions)
  const [allDocuments, setAllDocuments] = useState([])

  // Sessions — each session has a `documents: []` array
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [renamingSessionId, setRenamingSessionId] = useState(null)
  const [renameValue, setRenameValue] = useState("")

  // Sidebar
  const [expandedSections, setExpandedSections] = useState({
    notes: false,
    documents: true,
    chats: true
  })

  const [dashboard, setDashboard] = useState({ notes: 0, documents: 0, chats: 0 })

  const fileInputRef = useRef(null)
  const chatBottomRef = useRef(null)
  const renameInputRef = useRef(null)

  // Derived: documents for the active session
  const activeSession = sessions.find(s => s.id === activeSessionId)
  const sessionDocs = activeSession?.documents || []

  // ─── Helpers ─────────────────────────────────────────────────────────────────

  function getAuthHeaders() {
    return { Authorization: `Bearer ${localStorage.getItem("token")}` }
  }

  useEffect(() => {
    if (loggedIn) {
      fetchNotes()
      fetchAllDocuments()
      fetchDashboard()
      initSessions()
    }
  }, [loggedIn])

  // Auto-scroll on new messages
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isTyping])

  // ─── Auth ─────────────────────────────────────────────────────────────────────

  async function login() {
    try {
      const res = await fetch(`${API_BASE}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      })
      const data = await res.json()
      if (data.token) {
        localStorage.setItem("token", data.token)
        setLoggedIn(true)
      } else {
        alert(data.detail || data.error || "Login failed")
      }
    } catch { alert("Login error") }
  }

  async function signup() {
    try {
      const res = await fetch(`${API_BASE}/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      })
      const data = await res.json()
      alert(data.message || data.detail || data.error || "Signup done")
      setIsSignup(false)
    } catch { alert("Signup error") }
  }

  function logout() {
    localStorage.removeItem("token")
    setLoggedIn(false)
    setNotes([])
    setMessages([])
    setAllDocuments([])
    setSessions([])
    setActiveSessionId(null)
    setDashboard({ notes: 0, documents: 0, chats: 0 })
  }

  // ─── Notes ───────────────────────────────────────────────────────────────────

  async function fetchNotes() {
    try {
      const res = await fetch(`${API_BASE}/notes`, { headers: getAuthHeaders() })
      const data = await res.json()
      setNotes(Array.isArray(data) ? data : [])
    } catch (e) { console.log(e) }
  }

  async function addNote(titleOverride) {
    const title = titleOverride || input.trim()
    if (!title) return
    try {
      await fetch(`${API_BASE}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ title })
      })
      if (!titleOverride) setInput("")
      fetchNotes()
      fetchDashboard()
    } catch (e) { console.log(e) }
  }

  async function deleteNote(id) {
    try {
      await fetch(`${API_BASE}/notes/${id}`, { method: "DELETE", headers: getAuthHeaders() })
      fetchNotes()
      fetchDashboard()
    } catch (e) { console.log(e) }
  }

  // ─── Dashboard ───────────────────────────────────────────────────────────────

  async function fetchDashboard() {
    try {
      const res = await fetch(`${API_BASE}/dashboard`, { headers: getAuthHeaders() })
      const data = await res.json()
      setDashboard({ notes: data.notes || 0, documents: data.documents || 0, chats: data.chats || 0 })
    } catch (e) { console.log(e) }
  }

  // ─── Documents ───────────────────────────────────────────────────────────────

  async function fetchAllDocuments() {
    try {
      const res = await fetch(`${API_BASE}/documents`, { headers: getAuthHeaders() })
      const data = await res.json()
      setAllDocuments(Array.isArray(data) ? data : [])
    } catch (e) { console.log(e) }
  }

  async function uploadFile(file) {
    if (!file) return
    if (file.type !== "application/pdf") {
      setUploadStatus("❌ Only PDF files supported")
      return
    }

    setIsUploading(true)
    setUploadStatus(`Uploading ${file.name}...`)

    const formData = new FormData()
    formData.append("file", file)

    // Pass current session_id so backend links this PDF to the session
    const sessionParam = activeSessionId ? `?session_id=${activeSessionId}` : ""

    try {
      const res = await fetch(`${API_BASE}/upload${sessionParam}`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: formData
      })
      const data = await res.json()

      if (!res.ok) {
        setUploadStatus(`❌ ${data.detail || "Upload failed"}`)
        return
      }

      setUploadStatus(`✅ ${file.name} uploaded`)

      // Update the active session's document list locally
      if (activeSessionId && data.filename) {
        setSessions(prev => prev.map(s =>
          s.id === activeSessionId && !s.documents.includes(data.filename)
            ? { ...s, documents: [...s.documents, data.filename] }
            : s
        ))
      }

      fetchAllDocuments()
      fetchDashboard()

      if (fileInputRef.current) fileInputRef.current.value = ""

    } catch {
      setUploadStatus("❌ Upload failed")
    } finally {
      setIsUploading(false)
    }
  }

  async function removeDocFromSession(filename) {
    if (!activeSessionId) return
    try {
      await fetch(`${API_BASE}/chat-sessions/${activeSessionId}/documents/${encodeURIComponent(filename)}`, {
        method: "DELETE",
        headers: getAuthHeaders()
      })
      setSessions(prev => prev.map(s =>
        s.id === activeSessionId
          ? { ...s, documents: s.documents.filter(d => d !== filename) }
          : s
      ))
    } catch (e) { console.log(e) }
  }

  async function deleteDocument(filename) {
    try {
      await fetch(`${API_BASE}/documents/${filename}`, {
        method: "DELETE", headers: getAuthHeaders()
      })
      // Remove from all sessions locally
      setSessions(prev => prev.map(s => ({
        ...s,
        documents: s.documents.filter(d => d !== filename)
      })))
      fetchAllDocuments()
      fetchDashboard()
    } catch (e) { console.log(e) }
  }

  // Drag and drop
  function handleDragOver(e) { e.preventDefault(); setIsDragging(true) }
  function handleDragLeave() { setIsDragging(false) }
  function handleDrop(e) {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files).filter(f => f.type === "application/pdf")
    if (files.length === 0) { setUploadStatus("❌ Only PDF files supported"); return }
    files.forEach(f => uploadFile(f))
  }

  // ─── Sessions ────────────────────────────────────────────────────────────────

  async function fetchSessions() {
    try {
      const res = await fetch(`${API_BASE}/chat-sessions`, { headers: getAuthHeaders() })
      const data = await res.json()
      return Array.isArray(data) ? data : []
    } catch { return [] }
  }

  async function initSessions() {
    const existing = await fetchSessions()
    if (existing.length > 0) {
      setSessions(existing)
      setActiveSessionId(existing[0].id)
      fetchSessionMessages(existing[0].id)
    } else {
      await createNewChat()
    }
  }

  async function fetchSessionMessages(sessionId) {
    try {
      const res = await fetch(`${API_BASE}/chat-sessions/${sessionId}/messages`, {
        headers: getAuthHeaders()
      })
      const data = await res.json()
      const formatted = []
      if (Array.isArray(data)) {
        data.forEach(chat => {
          formatted.push({ role: "user", text: chat.question })
          formatted.push({ role: "ai", text: chat.answer })
        })
      }
      setMessages(formatted)
    } catch { setMessages([]) }
  }

  async function createNewChat() {
    try {
      const res = await fetch(`${API_BASE}/chat-sessions`, {
        method: "POST", headers: getAuthHeaders()
      })
      const data = await res.json()
      if (data.id) {
        setSessions(prev => [data, ...prev])
        setActiveSessionId(data.id)
        setMessages([])
        setUploadStatus("")
      }
    } catch (e) { console.log(e) }
  }

  function selectSession(sessionId) {
    if (sessionId === activeSessionId) return
    setActiveSessionId(sessionId)
    setUploadStatus("")
    fetchSessionMessages(sessionId)
  }

  async function deleteSession(sessionId) {
    try {
      await fetch(`${API_BASE}/chat-sessions/${sessionId}`, {
        method: "DELETE", headers: getAuthHeaders()
      })
      const remaining = sessions.filter(s => s.id !== sessionId)
      setSessions(remaining)
      if (sessionId === activeSessionId) {
        if (remaining.length > 0) {
          setActiveSessionId(remaining[0].id)
          fetchSessionMessages(remaining[0].id)
        } else {
          await createNewChat()
        }
      }
      fetchDashboard()
    } catch (e) { console.log(e) }
  }

  function startRename(session) {
    setRenamingSessionId(session.id)
    setRenameValue(session.title || "New Chat")
    setTimeout(() => renameInputRef.current?.focus(), 50)
  }

  async function submitRename(sessionId) {
    const newTitle = renameValue.trim()
    if (!newTitle) { setRenamingSessionId(null); return }
    try {
      await fetch(`${API_BASE}/chat-sessions/${sessionId}/rename`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ title: newTitle })
      })
      setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, title: newTitle } : s))
    } catch (e) { console.log(e) }
    setRenamingSessionId(null)
  }

  function toggleSection(section) {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  // ─── Ask AI (streaming) ──────────────────────────────────────────────────────

  async function askAI() {
    if (!question.trim()) return

    let sessionId = activeSessionId
    if (!sessionId) {
      await createNewChat()
      sessionId = activeSessionId
    }

    const currentQuestion = question
    setQuestion("")
    setMessages(prev => [...prev, { role: "user", text: currentQuestion }])
    setIsTyping(true)
    setTypingSessionId(sessionId)
    setMessages(prev => [...prev, { role: "ai", text: "" }])

    try {
      const res = await fetch(
        `${API_BASE}/ask/stream?question=${encodeURIComponent(currentQuestion)}&session_id=${sessionId}`,
        { headers: getAuthHeaders() }
      )

      const data = await res.json()
      const answer = data.answer || "No answer"
      const sourcesText = data.sources?.length > 0
        ? `\n\n📄 Sources:\n${data.sources.join("\n")}`
        : ""

      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = { role: "ai", text: answer + sourcesText }
        return updated
      })

      const updated = await fetchSessions()
      setSessions(updated)
      fetchDashboard()

    } catch (e) {
      console.log(e)
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = { role: "ai", text: "Something went wrong." }
        return updated
      })
    } finally {
      setIsTyping(false)
      setTypingSessionId(null)
    }
  }

  // ─── Render: Login ────────────────────────────────────────────────────────────

  if (!loggedIn) {
    return (
      <div className="loginPage">
        <div className="loginBox">
          <div className="loginLogo">🧠</div>
          <h1 className="loginTitle">DocMind</h1>
          <p className="loginSubtitle">Your AI-powered document assistant</p>
          <h2>{isSignup ? "Create Account" : "Welcome back"}</h2>
          <input className="loginInput" placeholder="Username" value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && (isSignup ? signup() : login())} />
          <input className="loginInput" type="password" placeholder="Password" value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && (isSignup ? signup() : login())} />
          <button className="loginBtn" onClick={isSignup ? signup : login}>
            {isSignup ? "Create Account" : "Login"}
          </button>
          <p className="loginSwitch" onClick={() => setIsSignup(!isSignup)}>
            {isSignup ? "Already have an account? Login" : "New user? Create account"}
          </p>
        </div>
      </div>
    )
  }

  // ─── Render: Dashboard ────────────────────────────────────────────────────────

  if (currentPage === "dashboard") {
    return (
      <div className="dashboardPage">
        <button className="backBtn" onClick={() => setCurrentPage("home")}>← Back</button>
        <h1>Dashboard</h1>
        <div className="dashboard">
          <div className="card"><div className="card-icon">📄</div><div className="card-label">Documents</div><h2>{dashboard.documents}</h2></div>
          <div className="card"><div className="card-icon">📝</div><div className="card-label">Notes</div><h2>{dashboard.notes}</h2></div>
          <div className="card"><div className="card-icon">💬</div><div className="card-label">Chats</div><h2>{dashboard.chats}</h2></div>
        </div>
      </div>
    )
  }

  // ─── Render: Main ─────────────────────────────────────────────────────────────

  return (
    <div className="container">

      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-brand">
          <span className="sidebar-logo">🧠</span>
          <span className="sidebar-title">DocMind</span>
        </div>

        <button className="newChatBtn" onClick={createNewChat}>+ New Chat</button>

        {/* All Documents */}
        <div className="sidebar-section">
          <div className="sidebar-section-header" onClick={() => toggleSection("documents")}>
            <span>📄 All Documents ({allDocuments.length})</span>
            <span>{expandedSections.documents ? "▼" : "▶"}</span>
          </div>
          {expandedSections.documents && (
            <div className="sidebar-section-body">
              {allDocuments.length === 0
                ? <p className="empty-state">No documents yet</p>
                : allDocuments.map((doc, i) => (
                  <div key={i} className="document-item" title={doc}>
                    <span className="document-name">📄 {doc}</span>
                    <button className="icon-btn delete-btn"
                      onClick={() => deleteDocument(doc)} title="Delete from all sessions">✕</button>
                  </div>
                ))
              }
            </div>
          )}
        </div>

        {/* Notes */}
        <div className="sidebar-section">
          <div className="sidebar-section-header" onClick={() => toggleSection("notes")}>
            <span>📝 Notes ({notes.length})</span>
            <span>{expandedSections.notes ? "▼" : "▶"}</span>
          </div>
          {expandedSections.notes && (
            <div className="sidebar-section-body">
              <div className="note-input-row">
                <input placeholder="Add a note..." value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && addNote()} />
                <button className="icon-btn add-btn" onClick={() => addNote()}>+</button>
              </div>
              {notes.length === 0
                ? <p className="empty-state">No notes yet</p>
                : notes.map(note => (
                  <div className="note-item" key={note.id}>
                    <p className="note-title">{note.title}</p>
                    <button className="icon-btn delete-btn" onClick={() => deleteNote(note.id)}>✕</button>
                  </div>
                ))
              }
            </div>
          )}
        </div>

        {/* Chat History */}
        <div className="sidebar-section">
          <div className="sidebar-section-header" onClick={() => toggleSection("chats")}>
            <span>💬 Chats ({sessions.length})</span>
            <span>{expandedSections.chats ? "▼" : "▶"}</span>
          </div>
          {expandedSections.chats && (
            <div className="sidebar-section-body">
              {sessions.length === 0
                ? <p className="empty-state">No chats yet</p>
                : sessions.map(session => (
                  <div key={session.id}
                    className={`chat-session ${session.id === activeSessionId ? "active" : ""}`}
                    onClick={() => selectSession(session.id)}>
                    {renamingSessionId === session.id ? (
                      <input ref={renameInputRef} className="rename-input"
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onBlur={() => submitRename(session.id)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") submitRename(session.id)
                          if (e.key === "Escape") setRenamingSessionId(null)
                        }}
                        onClick={(e) => e.stopPropagation()} />
                    ) : (
                      <>
                        <div className="chat-session-info">
                          <span className="chat-session-title"
                            onDoubleClick={(e) => { e.stopPropagation(); startRename(session) }}>
                            {session.title || "New Chat"}
                          </span>
                          {session.documents?.length > 0 && (
                            <span className="chat-session-doc">
                              📄 {session.documents.length} PDF{session.documents.length > 1 ? "s" : ""}
                            </span>
                          )}
                        </div>
                        <button className="icon-btn delete-btn"
                          onClick={(e) => { e.stopPropagation(); deleteSession(session.id) }}>✕</button>
                      </>
                    )}
                  </div>
                ))
              }
            </div>
          )}
        </div>
      </div>

      {/* Main */}
      <div className="main">

        {/* Topbar */}
        <div className="topbar">
          <div className="topbar-left">
            {sessionDocs.length > 0 ? (
              <div className="session-docs-bar">
                {sessionDocs.map((doc, i) => (
                  <div key={i} className="session-doc-chip">
                    <span>📄 {doc}</span>
                    <button className="chip-remove" onClick={() => removeDocFromSession(doc)}>✕</button>
                  </div>
                ))}
              </div>
            ) : (
              <span className="topbar-hint">No PDFs in this chat — upload below or ask general questions</span>
            )}
          </div>
          <div className="topbar-actions">
            <button className="topbar-btn" onClick={() => { fetchDashboard(); setCurrentPage("dashboard") }}>Dashboard</button>
            <button className="topbar-btn danger" onClick={logout}>Logout</button>
          </div>
        </div>

        {/* Upload */}
        <div className={`upload-box ${isDragging ? "dragging" : ""}`}
          onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
          <div className="upload-inner">
            <span className="upload-icon">📎</span>
            <span className="upload-hint">
              {isDragging ? "Drop PDFs here" : "Drag & drop PDFs, or "}
              {!isDragging && (
                <label className="upload-link">
                  browse
                  <input ref={fileInputRef} type="file" accept="application/pdf" multiple
                    style={{ display: "none" }}
                    onChange={(e) => {
                      Array.from(e.target.files).forEach(f => uploadFile(f))
                    }} />
                </label>
              )}
            </span>
            {isUploading && <span className="upload-spinner">⏳</span>}
            {uploadStatus && <span className="upload-status">{uploadStatus}</span>}
          </div>
        </div>

        {/* Chat */}
        <div className="chat-box">
          {messages.length === 0 && (
            <div className="chat-empty">
              <div className="chat-empty-icon">🧠</div>
              <p>Upload PDFs above and ask questions, or just say hi</p>
            </div>
          )}
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className="message-label">{msg.role === "user" ? "You" : "DocMind"}</div>
              <div className="message-body">
                {msg.role === "ai"
                  ? <ReactMarkdown>{msg.text || " "}</ReactMarkdown>
                  : <p>{msg.text}</p>
                }
              </div>
              {msg.role === "ai" && msg.text && (
                <button className="save-note-btn" title="Save as note"
                  onClick={() => addNote(msg.text.slice(0, 120))}>
                  📌 Save as note
                </button>
              )}
            </div>
          ))}
          {isTyping && typingSessionId === activeSessionId && (
            <div className="message ai">
              <div className="message-label">DocMind</div>
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
          <div ref={chatBottomRef} />
        </div>

        {/* Input */}
        <div className="input-row">
          <input placeholder="Ask anything..." value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && askAI()} />
          <button onClick={askAI} disabled={isTyping}>
            {isTyping ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
