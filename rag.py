from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGPipeline:
    def __init__(self):
        self.client = Groq(api_key="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")

        self.model_embed = SentenceTransformer("all-MiniLM-L6-v2")

        self.index = faiss.IndexFlatL2(384)
        self.documents = []

    def add_document(self, text):
        chunks = self.chunk_text(text)
        embeddings = self.model_embed.encode(chunks)
        embeddings_np = np.array(embeddings).astype("float32")

        self.index.add(embeddings_np)
        self.documents.extend(chunks)

    def chunk_text(self, text, size=500):
        words = text.split()
        chunks = []
        for i in range(0, len(words), size):
            chunk = " ".join(words[i:i+size])
            chunks.append(chunk)
        return chunks

    def retrieve(self, query, k=3):
        query_embed = self.model_embed.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embed, k)

        results = [self.documents[i] for i in indices[0]]
        return "\n".join(results)

    def answer_question(self, question):
        relevant = self.retrieve(question)

        prompt = f"""
Você é um assistente especialista no conteúdo abaixo:

CONTEÚDO DO PDF:
{relevant}

Pergunta do usuário:
{question}

Resposta:
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
