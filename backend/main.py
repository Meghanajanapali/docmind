import os
import re
from dotenv import load_dotenv

# Load .env before anything else so all os.getenv() calls below
# can read the values correctly
load_dotenv()

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status
)
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pypdf import PdfReader
from jose import jwt, JWTError
from passlib.context import CryptContext

from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
import uuid

from embedding import get_model, client as qdrant_client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
import uuid as uuid_lib
from database import SessionLocal, engine
from models import Base, Note, User, Chat, ChatSession, SessionDocument


# -----------------------------------
# SETUP
# -----------------------------------

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:5173")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY is not set. Create a .env file based on .env.example "
        "and set SECRET_KEY there."
    )

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


# -----------------------------------
# AUTH
# -----------------------------------

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
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

def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):

    # Split into sentences instead of raw character slices, so chunks
    # don't cut off in the middle of a sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # If adding this sentence would exceed chunk_size, save the
        # current chunk and start a new one.
        if current and len(current) + len(sentence) + 1 > chunk_size:

            chunks.append(current.strip())

            # Carry over the tail of the previous chunk for overlap,
            # so context isn't lost at chunk boundaries.
            if overlap > 0:
                current = current[-overlap:] + " " + sentence
            else:
                current = sentence

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

    return {

        "message": "Backend working"

    }


# -----------------------------------
# NOTES
# -----------------------------------

@app.get("/notes")
def get_notes(

    user=Depends(

        get_current_user

    )

):

    db = SessionLocal()

    try:

        notes = db.query(
            Note
        ).filter(
            Note.user_id == user["user_id"]
        ).all()

        return notes

    finally:

        db.close()


@app.post("/notes")
def add_note(

    note: dict,

    user=Depends(

        get_current_user

    )

):

    db = SessionLocal()

    try:

        new_note = Note(

            title=note["title"],
            user_id=user["user_id"]

        )

        db.add(

            new_note

        )

        db.commit()

        return {

            "message": "Added"

        }

    finally:

        db.close()


@app.delete("/notes/{note_id}")
def delete_note(

    note_id: int,

    user=Depends(

        get_current_user

    )

):

    db = SessionLocal()

    try:

        note = db.query(

            Note

        ).filter(

            Note.id == note_id,
            Note.user_id == user["user_id"]

        ).first()

        if not note:

            return {

                "message": "Not found"

            }

        db.delete(note)

        db.commit()

        return {

            "message": "Deleted"

        }

    finally:

        db.close()


# -----------------------------------
# PDF UPLOAD
# -----------------------------------

@app.post("/upload")
async def upload_pdf(

    file: UploadFile = File(...),
    session_id: int = None,

    user=Depends(

        get_current_user

    )

):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    pdf = PdfReader(file.file)

    # Extract text per page so we can track which page each chunk came from
    pages_text = []
    for page_num, page in enumerate(pdf.pages, start=1):
        extracted = page.extract_text()
        if extracted and extracted.strip():
            pages_text.append((page_num, extracted))

    if not pages_text:
        return {"message": "No readable text"}

    # Chunk each page separately so chunks don't span page boundaries
    all_chunks = []  # list of (chunk_text, page_number)
    for page_num, page_text in pages_text:
        for chunk in chunk_text(page_text):
            all_chunks.append((chunk, page_num))

    for index, (chunk, page_num) in enumerate(all_chunks):

        embedding = get_model().model.encode(chunk).tolist()
        point_id = str(uuid_lib.uuid4())

        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=point_id,
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

    # Link this document to the session if one was provided
    if session_id:
        db = SessionLocal()
        try:
            existing = db.query(SessionDocument).filter(
                SessionDocument.session_id == session_id,
                SessionDocument.filename == file.filename
            ).first()

            if not existing:
                session_doc = SessionDocument(
                    session_id=session_id,
                    user_id=user["user_id"],
                    filename=file.filename
                )
                db.add(session_doc)
                db.commit()
        finally:
            db.close()

    return {
        "message": "Uploaded",
        "filename": file.filename,
        "chunks": len(all_chunks)
    }


# -----------------------------------
# ASK AI
# -----------------------------------


@app.get("/ask")
def ask_ai(
    question: str,
    session_id: int,
    document: str = None,
    user=Depends(get_current_user)
):

    db = SessionLocal()

    try:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user["user_id"]
        ).first()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
    finally:
        db.close()

    embedding = get_model().model.encode(question).tolist()

    context = ""
    sources = []
    RELEVANCE_THRESHOLD = 0.3  # Qdrant cosine: higher = more similar (1.0 max)

    try:
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=int(user["user_id"])))
        ]

        if document:
            must_conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document))
            )

        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            query_filter=Filter(must=must_conditions),
            limit=5,
            with_payload=True,
            score_threshold=0.1
        ).points

        for hit in results:
            payload = hit.payload
            text = payload.get("text", "")
            context += text + "\n"
            fname = payload.get("filename", "")
            page = payload.get("page_number")
            entry = f"{fname}, page {page}" if page else fname
            if entry not in sources:
                sources.append(entry)

    except Exception:
        pass

    prompt = f"""You are a helpful AI assistant inside a personal document assistant app.

The user may ask general conversational questions (greetings, small talk, general
knowledge) or questions about their uploaded documents.

If the context below is relevant to the question, use it to give an accurate,
well-explained answer in your own words — do not copy sentences directly from
the context, synthesize the relevant information into a coherent answer.

If the context is empty or not relevant to the question, ignore it completely
and respond naturally as a helpful conversational assistant.

Context:
{context if context else "(no relevant document content found)"}

Question:
{question}

Answer:"""


    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        answer = response.choices[0].message.content
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is currently unavailable."
        )

    db = SessionLocal()

    try:

        chat = Chat(

            user_id=user["user_id"],

            session_id=session_id,

            question=question,

            answer=answer

        )

        db.add(chat)

        # If this is the session's first message, derive a title from it
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id
        ).first()

        if session and session.title == "New Chat":
            title = question.strip()
            if len(title) > 40:
                title = title[:40].rstrip() + "..."
            session.title = title or "New Chat"

        db.commit()

    finally:

        db.close()
    return {
        "answer": answer,
        "sources": sources
    }

# -----------------------------------
# AUTH ROUTES
# -----------------------------------

@app.post("/signup")
def signup(

    data: dict

):

    db = SessionLocal()

    try:

        existing = db.query(User).filter(
            User.username == data["username"]
        ).first()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )

        hashed = pwd_context.hash(

            data["password"]

        )

        user = User(

            username=data["username"],

            password=hashed

        )

        db.add(user)

        db.commit()

        return {

            "message": "User created"

        }

    finally:

        db.close()


@app.post("/login")
def login(

    data: dict

):

    db = SessionLocal()

    try:

        user = db.query(

            User

        ).filter(

            User.username == data["username"]

        ).first()

        if not user:

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        valid = pwd_context.verify(

            data["password"],

            user.password

        )

        if not valid:

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        token = jwt.encode(

            {

                "user_id": user.id

            },

            SECRET_KEY,

            algorithm=ALGORITHM

        )

        return {

            "token": token

        }

    finally:

        db.close()


# -----------------------------------
# DOCUMENTS
# -----------------------------------

@app.get("/documents")
def get_documents(
    user=Depends(get_current_user)
):

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
def delete_document(
    filename: str,
    user=Depends(get_current_user)
):

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


@app.post("/chat-sessions")
def create_chat_session(
    user=Depends(get_current_user)
):

    db = SessionLocal()

    try:

        session = ChatSession(
            user_id=user["user_id"],
            title="New Chat"
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "documents": []
        }

    finally:

        db.close()


@app.get("/chat-sessions")
def get_chat_sessions(
    user=Depends(get_current_user)
):

    db = SessionLocal()

    try:

        sessions = db.query(ChatSession).filter(
            ChatSession.user_id == user["user_id"]
        ).order_by(ChatSession.created_at.desc()).all()

        session_list = []

        for s in sessions:
            docs = db.query(SessionDocument).filter(
                SessionDocument.session_id == s.id,
                SessionDocument.user_id == user["user_id"]
            ).all()

            session_list.append({
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at,
                "documents": [d.filename for d in docs]
            })

        return session_list

    finally:

        db.close()


@app.get("/chat-sessions/{session_id}/documents")
def get_session_documents(
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )

        docs = db.query(SessionDocument).filter(
            SessionDocument.session_id == session_id,
            SessionDocument.user_id == user["user_id"]
        ).all()

        return [d.filename for d in docs]

    finally:

        db.close()


@app.get("/chat-sessions/{session_id}/messages")
def get_chat_session_messages(
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )

        chats = db.query(Chat).filter(
            Chat.session_id == session_id,
            Chat.user_id == user["user_id"]
        ).order_by(Chat.id.asc()).all()

        return chats

    finally:

        db.close()


@app.delete("/chat-sessions/{session_id}")
def delete_chat_session(
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )

        db.query(Chat).filter(
            Chat.session_id == session_id,
            Chat.user_id == user["user_id"]
        ).delete()

        db.query(SessionDocument).filter(
            SessionDocument.session_id == session_id,
            SessionDocument.user_id == user["user_id"]
        ).delete()

        db.delete(session)
        db.commit()

        return {
            "message": "Deleted"
        }

    finally:

        db.close()



@app.get("/dashboard")
def dashboard(
    user=Depends(get_current_user)
):

    db = SessionLocal()

    try:

        notes_count = db.query(Note).filter(
            Note.user_id == user["user_id"]
        ).count()

        chats_count = db.query(Chat).filter(
            Chat.user_id == user["user_id"]
        ).count()

        try:
            results = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="user_id", match=MatchValue(value=int(user["user_id"])))
                ]),
                with_payload=True,
                limit=1000
            )
            documents = set()
            for point in results[0]:
                fname = point.payload.get("filename")
                if fname:
                    documents.add(fname)
        except Exception:
            documents = set()

        return {
            "notes": notes_count,
            "documents": len(documents),
            "chats": chats_count
        }

    finally:

        db.close()


# -----------------------------------
# ASK (SIMPLE, RELIABLE)
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )

        session_docs = db.query(SessionDocument).filter(
            SessionDocument.session_id == session_id,
            SessionDocument.user_id == user["user_id"]
        ).all()

        session_filenames = [d.filename for d in session_docs]

        recent_chats = db.query(Chat).filter(
            Chat.session_id == session_id,
            Chat.user_id == user["user_id"]
        ).order_by(Chat.id.desc()).limit(4).all()

        recent_chats = list(reversed(recent_chats))

    finally:
        db.close()

    embedding = get_model().model.encode(question).tolist()

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
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=session_filenames)
                )
            )

        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            query_filter=Filter(must=must_conditions),
            limit=5,
            with_payload=True,
            score_threshold=RELEVANCE_THRESHOLD
        ).points

        # print(f"[DEBUG] Hits: {len(results)}")
        # for hit in results:
        #     print(f"[DEBUG] score={hit.score:.4f} file={hit.payload.get('filename')} page={hit.payload.get('page_number')}")

        for hit in results:
            payload = hit.payload
            context += payload.get("text", "") + "\n"
            fname = payload.get("filename", "")
            page = payload.get("page_number")
            entry = f"{fname}, page {page}" if page else fname
            if entry not in sources:
                sources.append(entry)

    except Exception as e:
        print(f"[DEBUG] Qdrant error: {e}")

    system_prompt = f"""You are DocMind, a helpful AI assistant inside a personal document assistant app.

The user may ask general questions or questions about their uploaded documents.

{"Use the following context from the user's documents to answer the question. Synthesize the information in your own words — do not copy directly." if context else "Answer from your general knowledge."}

{"Context:" + chr(10) + context if context else ""}

Use markdown formatting where helpful. Be concise and accurate."""

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

    # Show sources only if we actually found relevant chunks
    final_sources = sources if sources else []

    # Save to DB
    db = SessionLocal()
    try:
        chat = Chat(
            user_id=user["user_id"],
            session_id=session_id,
            question=question,
            answer=full_answer
        )
        db.add(chat)

        sess = db.query(ChatSession).filter(
            ChatSession.id == session_id
        ).first()

        if sess and sess.title == "New Chat":
            title = question.strip()
            sess.title = (title[:40].rstrip() + "...") if len(title) > 40 else title or "New Chat"

        db.commit()
    finally:
        db.close()

    return {
        "answer": full_answer,
        "sources": final_sources
    }


# -----------------------------------
# RENAME CHAT SESSION
# -----------------------------------

@app.patch("/chat-sessions/{session_id}/rename")
def rename_chat_session(
    session_id: int,
    data: dict,
    user=Depends(get_current_user)
):

    db = SessionLocal()

    try:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user["user_id"]
        ).first()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )

        session.title = data.get("title", session.title)
        db.commit()

        return {"message": "Renamed"}

    finally:
        db.close()


# -----------------------------------
# REMOVE DOCUMENT FROM SESSION
# -----------------------------------

@app.delete("/chat-sessions/{session_id}/documents/{filename}")
def remove_document_from_session(
    session_id: int,
    filename: str,
    user=Depends(get_current_user)
):

    db = SessionLocal()

    try:

        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user["user_id"]
        ).first()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        db.query(SessionDocument).filter(
            SessionDocument.session_id == session_id,
            SessionDocument.filename == filename,
            SessionDocument.user_id == user["user_id"]
        ).delete()

        db.commit()

        return {"message": "Removed from session"}

    finally:

        db.close()