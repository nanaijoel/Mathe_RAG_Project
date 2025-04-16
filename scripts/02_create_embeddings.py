import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
# from uuid import uuid4

# KONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_DIR = os.path.join(BASE_DIR, "chunks")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMBED_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY ist nicht gesetzt!")

EMBED_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=OPENAI_API_KEY)


# Embedding-Funktion
def get_embedding(text):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return response.data[0].embedding

# Daten sammeln
texts = []
metadata = []
vectors = []

for fname in sorted(os.listdir(CHUNK_DIR)):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(CHUNK_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            continue
        embedding = get_embedding(content)
        texts.append(content)
        vectors.append(embedding)
        metadata.append({"filename": fname})
        print(f"eingebettet: {fname}")

# FAISS Index erstellen
dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors).astype("float32"))

# Speichern
faiss.write_index(index, os.path.join(EMBED_DIR, "faiss.index"))
with open(os.path.join(EMBED_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump({"texts": texts, "meta": metadata}, f)

