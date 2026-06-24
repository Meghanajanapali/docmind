# 🧠 DocMind — AI-Powered Document Assistant

DocMind is a privacy-first, full-stack RAG (Retrieval-Augmented Generation) application that lets you upload PDFs and have intelligent conversations with your documents using a fully local or cloud-hosted AI pipeline.

## ✨ Features

- 📄 **Multi-document RAG** — Upload multiple PDFs per chat session and ask questions across all of them simultaneously
- 🧠 **AI-powered answers** — Powered by Llama 3.1 70B via Groq for fast, high-quality responses
- 💬 **Persistent chat sessions** — Multiple named chat sessions with full conversation history, like ChatGPT
- 🔒 **User isolation** — JWT authentication ensures each user's documents and chats are fully private
- 📌 **Save as note** — Pin any AI response as a note directly from the chat
- 🔍 **Semantic search** — Uses `all-mpnet-base-v2` sentence embeddings with Qdrant vector DB for accurate retrieval
- 📃 **Page-level citations** — Sources show filename and exact page number
- ⚡ **Streaming responses** — Answers stream token by token for a fast, responsive feel
- 🎯 **Relevance filtering** — Irrelevant document chunks are filtered out so general questions get general answers

## 🛠 Tech Stack

**Backend:** FastAPI, PostgreSQL (Neon), Qdrant Cloud, sentence-transformers, Groq (Llama 3.1 70B), JWT auth

**Frontend:** React.js, Vite, react-markdown

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- Accounts: [Neon](https://neon.tech), [Qdrant Cloud](https://cloud.qdrant.io), [Groq](https://console.groq.com)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Run the backend:

```bash
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## 📁 Project Structure

```
docmind/
├── backend/
│   ├── main.py          # FastAPI app, all endpoints
│   ├── embedding.py     # Sentence-transformer + Qdrant setup
│   ├── models.py        # SQLAlchemy models
│   ├── database.py      # DB connection
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx      # Main React component
    │   └── App.css      # Styles
    └── index.html
```

## 🌐 Deployment

- **Backend:** Render (free tier)
- **Frontend:** Vercel (free tier)
- **Database:** Neon (free tier)
- **Vector DB:** Qdrant Cloud (free tier)