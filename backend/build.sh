#!/bin/bash
pip install -r requirements.txt
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading embedding model...')
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print('Model downloaded successfully.')
"