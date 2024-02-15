
# Document Question Answering System

This system leverages the power of OpenAI's language models and Pinecone's vector search to provide accurate answers to questions based on the content of uploaded documents. It uses Langchain for efficient document processing and question answering.

## Features

- Upload PDF, TXT, DOCX, or DOC files for analysis
- Chunk and preprocess documents for efficient querying
- Generate embeddings for document sections using OpenAI's language models
- Use Pinecone's vector database for fast and accurate similarity search
- Provide instant answers to user queries based on document content

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8 or later
- Pip for package installation

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/document-question-answering-system.git
cd document-question-answering-system
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Setup

Before running the application, ensure you have set up the necessary environment variables defined in the .env file in this project directory:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: The name of your Pinecone index
- `OPENAI_LLM_MODEL_NAME`: The name of the OpenAI language model you're using

### Running the Application

To start the Streamlit application, run:

```bash
streamlit run app.py
```

The web interface should now be accessible at `http://localhost:8501`.

## Usage

1. Upload your document through the web interface.
2. Enter your question in the provided text input.
3. Click "Get Answer" to retrieve information based on your query.

## Use Cases
The Document Question Answering System can be useful in a variety of scenarios:

1. Research: Researchers can use the system to quickly find answers in long documents or studies, saving time that would otherwise be spent manually searching through the text.
2. Legal and Compliance: Legal professionals can use the system to extract specific details from large volumes of legal documents, such as contracts or legislation.
3. Education: Students and educators can use the system to find answers in textbooks or literature, making study sessions more efficient.
4. Customer Support: Customer support teams can upload manuals or internal documentation to find answers to customer inquiries quickly.
5. Medical Field: Medical professionals can use the system to find information from medical texts and journals, assisting in research or diagnostics.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- OpenAI for providing the language models
- Pinecone for vector database services
- Langchain for document processing tools
