import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# Embedding model — produces 768-dimensional vectors
model = SentenceTransformer("all-mpnet-base-v2")
VECTOR_SIZE = 768
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