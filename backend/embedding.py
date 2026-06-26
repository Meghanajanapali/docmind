import os
import warnings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

load_dotenv()

COLLECTION_NAME = "docmind"
VECTOR_SIZE = 384

# Lazy load model to reduce startup memory spike
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

# Qdrant client
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

# Create collection if needed
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

# Create payload indexes
for field, schema in [
    ("user_id", PayloadSchemaType.INTEGER),
    ("document_id", PayloadSchemaType.KEYWORD),
    ("filename", PayloadSchemaType.KEYWORD),
]:
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema
        )
    except Exception:
        pass