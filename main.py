import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"

main_placeholder = st.empty()

# Load a Hugging Face model for language generation
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings using Sentence Transformers
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = [embedding_model.encode(doc.page_content) for doc in docs]
    vectorstore_hf = FAISS.from_embeddings(embeddings, docs)
    
    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_hf, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever()
            
            # Perform retrieval
            retrieved_docs = retriever.get_relevant_documents(query)
            context = " ".join([doc.page_content for doc in retrieved_docs])
            
            # Generate an answer using Hugging Face model
            result = llm_pipeline(f"Answer this question based on the context:\n\nContext: {context}\n\nQuestion: {query}")
            
            st.header("Answer")
            st.write(result[0]['generated_text'])

            # Display sources
            if retrieved_docs:
                st.subheader("Sources:")
                for doc in retrieved_docs:
                    st.write(doc.metadata.get("source", "Unknown Source"))
