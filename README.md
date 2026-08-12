# GenAI-PDF-RAG

A Retrieval-Augmented Generation (RAG) application that answers questions about the YOLOv9 research paper using LangChain, Hugging Face, ChromaDB, and Streamlit.

## Architecture

```
YOLOv9 PDF
    |
    v
PyPDFLoader
    |
    v
Text Splitting (RecursiveCharacterTextSplitter)
    |
    v
Hugging Face Embeddings (all-mpnet-base-v2)
    |
    v
ChromaDB
    |
    v
Similarity Retriever (Top 5 Relevant Chunks)
    |
    v
Prompt + Retrieved Context
    |
    v
Hugging Face Inference API (GPT-OSS-120B)
    |
    v
Generated Answer
    |
    v
Streamlit UI
```

## Technologies Used

- Python
- LangChain
- LangChain Classic
- Hugging Face
- Sentence Transformers
- ChromaDB
- GPT-OSS-120B
- Streamlit
- PyPDF

## RAG Pipeline

### 1. Document Loading

The YOLOv9 research paper is loaded using `PyPDFLoader`.

```python
loader = PyPDFLoader("yolov9_paper.pdf")
data = loader.load()
```

### 2. Text Splitting

The extracted document is divided into smaller chunks.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(data)
```

The overlap helps preserve context between neighboring chunks.

### 3. Embeddings

Each text chunk is converted into a numerical vector using the Hugging Face Sentence Transformer model:

```
sentence-transformers/all-mpnet-base-v2
```

These vectors allow the application to perform semantic similarity search.

### 4. Vector Database

The embeddings and document chunks are stored in ChromaDB.

```python
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### 5. Retrieval

When a user asks a question, the retriever searches ChromaDB and returns the five most relevant chunks.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

### 6. Generation

The retrieved context is passed to GPT-OSS-120B through the Hugging Face Inference API. The prompt instructs the model to answer using the retrieved context and to say that it does not know when the answer is not available in the context.

### 7. Streamlit

Streamlit provides a simple web interface where users can enter questions about the YOLOv9 paper.

## Installation

Clone the repository:

```bash
git clone https://github.com/vasudeva-analyst/GenAI-PDF-RAG.git
```

Move into the project directory:

```bash
cd GenAI-PDF-RAG
```

Create and activate the Conda environment:

```bash
conda create -n rag_streamlit python=3.10 -y
conda activate rag_streamlit
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Hugging Face API

The application uses a Hugging Face API token to access the hosted LLM.

> **Note:** Do not place your real API token directly inside the source code. The application accepts the token through the Streamlit interface.

## Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal. For example:

```
http://localhost:8501
```

## Example

**Example question:** *What is new in YOLOv9?*

The application retrieves relevant sections from the YOLOv9 paper and generates an answer using the retrieved context.

## Project Structure

```
GenAI-PDF-RAG/
│
├── app.py
├── Basic_RAG_PDF0.ipynb
├── requirements.txt
├── yolov9_paper.pdf
├── README.md
└── .gitignore
```

The ChromaDB directory is generated locally and is excluded from Git using `.gitignore`.

## Future Improvements

- [ ] Support multiple PDF documents
- [ ] Add chat history
- [ ] Display document sources for each answer
- [ ] Add PDF upload through Streamlit
- [ ] Improve retrieval with hybrid search
- [ ] Add conversation memory
- [ ] Deploy the application to the cloud

## Author

**Vasudeva Reddy**
GitHub: [https://github.com/vasudeva-analyst](https://github.com/vasudeva-analyst)