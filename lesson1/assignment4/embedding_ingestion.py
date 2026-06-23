import os 
import pickle 
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_docs(folder):
    docs = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        with open(path, "r") as f:
            text = f.read()
            docs.append(text)

    return docs
            
def chunk_text(text, size):
    chunks = []
    for i in range (0, len(text), size):
        chunks.append(text[i: i+size])
        
    return chunks

documents = load_docs("Documents")
chunks = []

for doc in documents:
    chunk = chunk_text(doc, 300)
    chunks.extend(chunk)

embeddings = embed_model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "vectorstore/faiss.index")

with open("vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Done")