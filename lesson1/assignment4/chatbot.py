import pickle
import faiss
import numpy as np

from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index("vectorstore/faiss.index")

with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def retrieve(query, k=3):
    query_embedding = embed_model.encode([query])

    vector = np.array(query_embedding).astype("float32")

    distances, indices = index.search(vector, k)

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results


def ask_llm(question, context):
    prompt = f"""
             Answer the question using ONLY the context below.

             If the answer is not in the context, say:
             "I don't know."

             Context:{context}

             Question:{question}
             """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    retrieved_chunks = retrieve(question)

    context = "\n".join(retrieved_chunks)

    answer = ask_llm(question, context)

    print("\nBot:", answer)