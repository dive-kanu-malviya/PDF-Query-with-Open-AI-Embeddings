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
import time
# Local application imports
from utils import read_doc, chunk_data, retrieve_answers, setup_openai_embeddings, setup_pinecone, create_pinecone_index, setup_llm
MAX_RESPONSE_TOKENS = 2046
TEMPERATURE = 0.1
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

     # Added descriptive text below the title
    st.write("Upload a document and ask anything. Get answers tuned for your industry needs.")

    st.write("NOTE : Designed to read text rich documents. Answering quality might drop if your document contains too many images and design formatting")



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
            #add sleep for index to upload
            time.sleep(10)

            # Setup LLM and QA chain
            llm = setup_llm(OPENAI_LLM_MODEL_NAME,temperature=TEMPERATURE,max_tokens=MAX_RESPONSE_TOKENS)
            # Assuming load_qa_chain is defined and implemented correctly elsewhere
            chain = load_qa_chain(llm, chain_type="stuff")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return

        user_query = st.text_input("Enter your question here:")

        mode = st.radio("Choose your answer mode:", ("Precise", "Elaborate"), index=0, horizontal=True)


        if st.button("Get Answer"):
            if user_query:
                try:
                    
                    if mode == "Precise":
                        user_query += " Give a precise short answer."
                    elif mode == "Elaborate":
                        user_query += " Give a long answer with different headings"
                    else:
                        st.error(f"Answer mode not selected")
                    answer = retrieve_answers(chain, index, user_query)
                    logging.info(f'{answer}')
                    print(len(answer),answer)
                    st.write(answer)

                except Exception as e:
                    st.error(f"Failed to retrieve answers: {e}")
            else:
                st.write("Please enter a question to get an answer.")

if __name__ == '__main__':
    streamlit_ui()
