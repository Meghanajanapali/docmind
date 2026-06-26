import os
import re
import bcrypt
from dotenv import load_dotenv

load_dotenv()

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pypdf import PdfReader
from jose import jwt, JWTError
from groq import Groq

import uuid as uuid_lib

from embedding import get_model, client as qdrant_client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from database import SessionLocal, engine
from models import Base, Note, User, Chat, ChatSession, SessionDocument

# -----------------------------------
# SETUP
# -----------------------------------

app = FastAPI()
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set.")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

# -----------------------------------
# AUTH
# -----------------------------------

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload

# -----------------------------------
# TEXT CHUNKING
# -----------------------------------

def chunk_text(text, chunk_size=500, overlap=100):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > chunk_size:
            chunks.append(current.strip())
            current = (current[-overlap:] + " " + sentence) if overlap > 0 else sentence
        else:
            current = (current + " " + sentence).strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks

# -----------------------------------
# HOME
# -----------------------------------

@app.get("/")
def home():
    return {"message": "DocMind backend running"}

# -----------------------------------
# NOTES
# -----------------------------------

@app.get("/notes")
def get_notes(user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        return db.query(Note).filter(Note.user_id == user["user_id"]).all()
    finally:
        db.close()

@app.post("/notes")
def add_note(note: dict, user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        db.add(Note(title=note["title"], user_id=user["user_id"]))
        db.commit()
        return {"message": "Added"}
    finally:
        db.close()

@app.delete("/notes/{note_id}")
def delete_note(note_id: int, user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        note = db.query(Note).filter(Note.id == note_id, Note.user_id == user["user_id"]).first()
        if not note:
            return {"message": "Not found"}
        db.delete(note)
        db.commit()
        return {"message": "Deleted"}
    finally:
        db.close()

# -----------------------------------
# PDF UPLOAD
# -----------------------------------

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: int = None,
    user=Depends(get_current_user)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf = PdfReader(file.file)
    pages_text = []
    for page_num, page in enumerate(pdf.pages, start=1):
        extracted = page.extract_text()
        if extracted and extracted.strip():
            pages_text.append((page_num, extracted))

    if not pages_text:
        return {"message": "No readable text"}

    all_chunks = []
    for page_num, page_text in pages_text:
        for chunk in chunk_text(page_text):
            all_chunks.append((chunk, page_num))

    model = get_model()
    for index, (chunk, page_num) in enumerate(all_chunks):
        embedding = model.encode(chunk).tolist()
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=str(uuid_lib.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "filename": file.filename,
                    "chunk_number": index,
                    "page_number": page_num,
                    "document_id": file.filename,
                    "user_id": int(user["user_id"])
                }
            )]
        )

    if session_id:
        db = SessionLocal()
        try:
            existing = db.query(SessionDocument).filter(
                SessionDocument.session_id == session_id,
                SessionDocument.filename == file.filename
            ).first()
            if not existing:
                db.add(SessionDocument(
                    session_id=session_id,
                    user_id=user["user_id"],
                    filename=file.filename
                ))
                db.commit()
        finally:
            db.close()

    return {"message": "Uploaded", "filename": file.filename, "chunks": len(all_chunks)}

# -----------------------------------
# AUTH ROUTES
# -----------------------------------

@app.post("/signup")
def signup(data: dict):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == data["username"]).first():
            raise HTTPException(status_code=400, detail="Username already taken")
        db.add(User(username=data["username"], password=hash_password(data["password"])))
        db.commit()
        return {"message": "User created"}
    finally:
        db.close()

@app.post("/login")
def login(data: dict):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == data["username"]).first()
        if not user or not verify_password(data["password"], user.password):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        token = jwt.encode({"user_id": user.id}, SECRET_KEY, algorithm=ALGORITHM)
        return {"token": token}
    finally:
        db.close()

# -----------------------------------
# DOCUMENTS
# -----------------------------------

@app.get("/documents")
def get_documents(user=Depends(get_current_user)):
    try:
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=int(user["user_id"])))
            ]),
            with_payload=True,
            limit=1000
        )
        docs = set()
        for point in results[0]:
            fname = point.payload.get("filename")
            if fname:
                docs.add(fname)
        return list(docs)
    except Exception:
        return []

@app.delete("/documents/{filename}")
def delete_document(filename: str, user=Depends(get_current_user)):
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(must=[
                FieldCondition(key="filename", match=MatchValue(value=filename)),
                FieldCondition(key="user_id", match=MatchValue(value=int(user["user_id"])))
            ])
        )
    except Exception:
        pass
    return {"message": "Deleted"}

# -----------------------------------
# CHAT SESSIONS
# -----------------------------------

@app.post("/chat-sessions")
def create_chat_session(user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        session = ChatSession(user_id=user["user_id"], title="New Chat")
        db.add(session)
        db.commit()
        db.refresh(session)
        return {"id": session.id, "title": session.title, "created_at": session.created_at, "documents": []}
    finally:
        db.close()

@app.get("/chat-sessions")
def get_chat_sessions(user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).filter(
            ChatSession.user_id == user["user_id"]
        ).order_by(ChatSession.created_at.desc()).all()
        result = []
        for s in sessions:
            docs = db.query(SessionDocument).filter(
                SessionDocument.session_id == s.id,
                SessionDocument.user_id == user["user_id"]
            ).all()
            result.append({
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at,
                "documents": [d.filename for d in docs]
            })
        return result
    finally:
        db.close()

@app.get("/chat-sessions/{session_id}/messages")
def get_chat_session_messages(session_id: int, user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user["user_id"]
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return db.query(Chat).filter(
            Chat.session_id == session_id,
            Chat.user_id == user["user_id"]
        ).order_by(Chat.id.asc()).all()
    finally:
        db.close()

@app.delete("/chat-sessions/{session_id}")
def delete_chat_session(session_id: int, user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user["user_id"]
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        db.query(Chat).filter(Chat.session_id == session_id).delete()
        db.query(SessionDocument).filter(SessionDocument.session_id == session_id).delete()
        db.delete(session)
        db.commit()
        return {"message": "Deleted"}
    finally:
        db.close()

@app.patch("/chat-sessions/{session_id}/rename")
def rename_chat_session(session_id: int, data: dict, user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user["user_id"]
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session.title = data.get("title", session.title)
        db.commit()
        return {"message": "Renamed"}
    finally:
        db.close()

@app.delete("/chat-sessions/{session_id}/documents/{filename}")
def remove_document_from_session(session_id: int, filename: str, user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        db.query(SessionDocument).filter(
            SessionDocument.session_id == session_id,
            SessionDocument.filename == filename,
            SessionDocument.user_id == user["user_id"]
        ).delete()
        db.commit()
        return {"message": "Removed from session"}
    finally:
        db.close()

# -----------------------------------
# DASHBOARD
# -----------------------------------

@app.get("/dashboard")
def dashboard(user=Depends(get_current_user)):
    db = SessionLocal()
    try:
        notes_count = db.query(Note).filter(Note.user_id == user["user_id"]).count()
        chats_count = db.query(Chat).filter(Chat.user_id == user["user_id"]).count()
        try:
            results = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="user_id", match=MatchValue(value=int(user["user_id"])))
                ]),
                with_payload=True,
                limit=1000
            )
            documents = set(
                point.payload.get("filename")
                for point in results[0]
                if point.payload.get("filename")
            )
        except Exception:
            documents = set()
        return {"notes": notes_count, "documents": len(documents), "chats": chats_count}
    finally:
        db.close()

# -----------------------------------
# ASK
# -----------------------------------

@app.get("/ask/stream")
def ask_ai_stream(
    question: str,
    session_id: int,
    user=Depends(get_current_user)
):
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user["user_id"]
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session_docs = db.query(SessionDocument).filter(
            SessionDocument.session_id == session_id,
            SessionDocument.user_id == user["user_id"]
        ).all()
        session_filenames = [d.filename for d in session_docs]

        recent_chats = list(reversed(
            db.query(Chat).filter(
                Chat.session_id == session_id,
                Chat.user_id == user["user_id"]
            ).order_by(Chat.id.desc()).limit(4).all()
        ))
    finally:
        db.close()

    model = get_model()
    embedding = model.encode(question).tolist()

    context = ""
    sources = []
    RELEVANCE_THRESHOLD = 0.65

    try:
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=int(user["user_id"])))
        ]
        if session_filenames:
            from qdrant_client.models import MatchAny
            must_conditions.append(
                FieldCondition(key="document_id", match=MatchAny(any=session_filenames))
            )

        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            query_filter=Filter(must=must_conditions),
            limit=5,
            with_payload=True,
            score_threshold=RELEVANCE_THRESHOLD
        ).points

        for hit in results:
            payload = hit.payload
            context += payload.get("text", "") + "\n"
            fname = payload.get("filename", "")
            page = payload.get("page_number")
            entry = f"{fname}, page {page}" if page else fname
            if entry not in sources:
                sources.append(entry)

    except Exception as e:
        print(f"Qdrant error: {e}")

    system_prompt = f"""You are DocMind, a helpful AI assistant inside a personal document assistant app.

{"Use the following context from the user's documents to answer the question. Synthesize in your own words." if context else "Answer from your general knowledge."}

{"Context:" + chr(10) + context if context else ""}

Use markdown formatting where helpful."""

    messages_for_llm = [{"role": "system", "content": system_prompt}]
    for chat in recent_chats:
        messages_for_llm.append({"role": "user", "content": chat.question})
        messages_for_llm.append({"role": "assistant", "content": chat.answer})
    messages_for_llm.append({"role": "user", "content": question})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_llm,
            max_tokens=1024
        )
        full_answer = response.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    db = SessionLocal()
    try:
        db.add(Chat(
            user_id=user["user_id"],
            session_id=session_id,
            question=question,
            answer=full_answer
        ))
        sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if sess and sess.title == "New Chat":
            title = question.strip()
            sess.title = (title[:40].rstrip() + "...") if len(title) > 40 else title or "New Chat"
        db.commit()
    finally:
        db.close()

    return {"answer": full_answer, "sources": sources}