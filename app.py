import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from utils import load_pdf, split_documents, create_vector_db, save_vector_db, load_vector_db

import os
from dotenv import load_dotenv
load_dotenv()  # Load your OpenAI API key from .env file

# Set up Streamlit page
st.set_page_config(page_title="Chat With Your PDF", layout="centered")
st.title("📄 Chat With Your PDF (RAG Project)")
st.markdown("Ask any question based on your uploaded PDF!")

# Step 1: Upload PDF
pdf = st.file_uploader("Upload a PDF", type="pdf")

# Session state to store vector DB
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Step 2: When user uploads a PDF
if pdf is not None:
    with st.spinner("Reading and processing PDF..."):
        # Save PDF to temp file
        temp_path = "temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf.read())

        # Load, split, and embed PDF
        docs = load_pdf(temp_path)
        chunks = split_documents(docs)
        vector_store = create_vector_db(chunks)

        # Save vector DB for future use (optional)
        save_vector_db(vector_store)

        st.session_state.vector_store = vector_store
        st.success("PDF processed and stored in vector database!")

# Step 3: Input question
question = st.text_input("Ask a question related to the PDF:")

# Step 4: Answer question using RAG (LLM + Retriever)
if st.session_state.vector_store and question:
    with st.spinner("Searching for answer..."):
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vector_store.as_retriever()
        )
        result = qa_chain.run(question)
        st.markdown(f"### 🧠 Answer:\n{result}")
