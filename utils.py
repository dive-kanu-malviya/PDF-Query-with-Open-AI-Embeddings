
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone as pinecone_obj
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()




#intialize open ai embeddings
def setup_openai_embeddings(api_key):
    return OpenAIEmbeddings(api_key=api_key)

#initalize pinecode using API key
def setup_pinecone(api_key, environment="gcp-starter"):
    return pinecone_obj(api_key=api_key, environment=environment)

#create index based on document contents
def create_pinecone_index(documents, embeddings, index_name):
    return Pinecone.from_documents(documents, embeddings, index_name=index_name)

# initialize the LLM model from Open API
def setup_llm(model_name, temperature=0.1):
    return OpenAI(model_name=model_name, temperature=temperature)

#read document using langchain pdf reader
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents
    

#split data into chunks
def chunk_data(docs,chunk_size,chunk_overlap):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs


## Cosine Similarity Retreive Results from VectorDB
def retrieve_query(index , query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results


## Search answers from VectorDB
def retrieve_answers(chain, index,query):
    doc_search=retrieve_query(index,query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response