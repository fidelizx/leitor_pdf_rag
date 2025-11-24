import streamlit as st
import os
from process_pdf import extract_text_from_pdf
from rag import RAGPipeline

st.title("📚 IA para Ler PDFs (RAG) — Groq Gratuito")

rag = RAGPipeline()

st.sidebar.header("📁 Enviar PDF")
uploaded_pdf = st.sidebar.file_uploader("Escolha um arquivo PDF", type=["pdf"])

if uploaded_pdf:
    pdf_dir = "pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, uploaded_pdf.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    st.sidebar.success("PDF carregado!")

    with st.spinner("Extraindo texto do PDF..."):
        text = extract_text_from_pdf(pdf_path)

    with st.spinner("Criando embeddings..."):
        rag.add_document(text)

    st.success("PDF processado!")

    st.subheader("❓ Pergunte algo sobre o PDF")
    pergunta = st.text_input("Digite sua pergunta:")

    if st.button("Responder"):
        if pergunta.strip():
            resposta = rag.answer_question(pergunta)
            st.write("### 📘 Resposta:")
            st.write(resposta)
        else:
            st.warning("Digite uma pergunta antes de clicar!")
else:
    st.info("Envie um PDF pela barra lateral para começar.")
