# Standard library imports
import os

# Third-party imports
import streamlit as st
import openai
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone
import tempfile
import logging
import warnings

# Local application imports
from utils import read_doc, chunk_data, retrieve_answers, setup_openai_embeddings, setup_pinecone, create_pinecone_index, setup_llm


# Configuration and setup
try:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']
    OPENAI_LLM_MODEL_NAME = os.environ['OPENAI_LLM_MODEL_NAME']
except KeyError as e:
    st.error(f"Environment variable {e} not set. Please check your .env file or environment configuration.")


# Streamlit UI
def streamlit_ui():
    st.title("Document Question Answering System")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx", "doc"])
    if uploaded_file is not None:
        try:
            # Save uploaded file to a temporary file to work around the limitation
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name

            doc = read_doc(temp_file_path)
            documents = chunk_data(docs=doc, chunk_size=800, chunk_overlap=50)

            # Setup OpenAI embeddings and Pinecone
            embeddings = setup_openai_embeddings(OPENAI_API_KEY)
            pc = setup_pinecone(PINECONE_API_KEY)
            index = create_pinecone_index(documents, embeddings, PINECONE_INDEX_NAME)

            # Setup LLM and QA chain
            llm = setup_llm(OPENAI_LLM_MODEL_NAME)
            # Assuming load_qa_chain is defined and implemented correctly elsewhere
            chain = load_qa_chain(llm, chain_type="stuff")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return

        user_query = st.text_input("Enter your question here:")

        if st.button("Get Answer"):
            if user_query:
                try:
                    answer = retrieve_answers(chain, index, user_query)
                    st.write(answer)
                except Exception as e:
                    st.error(f"Failed to retrieve answers: {e}")
            else:
                st.write("Please enter a question to get an answer.")

if __name__ == '__main__':
    streamlit_ui()