import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# Using all-MiniLM-L6-v2 (90MB) instead of all-mpnet-base-v2 (420MB)
# to stay within Render free tier's 512MB RAM limit.
# Produces 384-dimensional vectors.
# BAAI/bge-small-en-v1.5 — 90MB, 384-dim vectors, slightly better than
# MiniLM while staying within Render free tier's 512MB RAM limit.
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
VECTOR_SIZE = 384
COLLECTION_NAME = "docmind"

import warnings

# Qdrant cloud client
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

# Create collection if it doesn't exist
existing = [c.name for c in client.get_collections().collections]

if COLLECTION_NAME not in existing:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

# Create payload indexes for fields used in filters
# (required by Qdrant for filtering to work correctly)
from qdrant_client.models import PayloadSchemaType

try:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="user_id",
        field_schema=PayloadSchemaType.INTEGER
    )
except Exception:
    pass  # Index already exists

try:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="document_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
except Exception:
    pass  # Index already exists

try:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="filename",
        field_schema=PayloadSchemaType.KEYWORD
    )
except Exception:
    pass  # Index already exists