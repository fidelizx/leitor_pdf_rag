from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class RAGPipeline:
    def __init__(self):
        # Carrega chave secreta
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("❌ ERRO: A variável secreta GROQ_API_KEY não foi configurada.")

        self.client = Groq(api_key=api_key)

        # Modelo de embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # FAISS index
        self.index = faiss.IndexFlatL2(384)
        self.documents = []

    def add_document(self, text):
        embedding = self.embedder.encode([text])
        self.index.add(embedding)
        self.documents.append(text)

    def search(self, query):
        q_emb = self.embedder.encode([query])
        distances, indices = self.index.search(q_emb, 1)

        return self.documents[indices[0][0]]

    def answer_question(self, question):
        contexto = self.search(question)

        prompt = (
            f"Use APENAS essas informações do PDF para responder:\n\n"
            f"{contexto}\n\n"
            f"Pergunta: {question}"
        )

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
s
