from sentence_transformers import SentenceTransformer
import chromadb

# Upgraded from all-MiniLM-L6-v2 — better semantic similarity at
# the cost of slightly more memory. Re-upload all PDFs after this change
# since old vectors are incompatible with the new model's embedding space.
model = SentenceTransformer("all-mpnet-base-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="docmind")