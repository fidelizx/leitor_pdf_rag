from groq import Groq
import os

class RAGPipeline:
    def __init__(self):
        # Lê a chave secreta do Streamlit Cloud
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)

        # Armazena o texto dos PDFs enviados
        self.documentos = []

    def add_document(self, texto):
        """Guarda o texto extraído do PDF."""
        self.documentos.append(texto)

    def answer_question(self, pergunta):
        """Envia a pergunta + o conteúdo armazenado para o LLM."""
        contexto = "\n\n".join(self.documentos)

        prompt = f"""
Você é uma IA especialista em responder perguntas com base em documentos.

DOCUMENTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""

        resposta = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return resposta.choices[0].message.content
