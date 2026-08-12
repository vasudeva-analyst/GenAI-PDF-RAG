\# GenAI PDF RAG Application



A simple Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the YOLOv9 research paper.



The application uses LangChain to build the RAG pipeline, Hugging Face embeddings for semantic search, ChromaDB as the vector database, GPT-OSS-120B through the Hugging Face Inference API for answer generation, and Streamlit for the user interface.



\## What This Project Does



The application takes the YOLOv9 research paper as a PDF and turns it into a searchable knowledge base.



When a user asks a question, the application:



1\. Loads the PDF.

2\. Splits the document into smaller text chunks.

3\. Converts the chunks into embeddings.

4\. Stores the embeddings in ChromaDB.

5\. Searches for the most relevant chunks based on the user's question.

6\. Sends the retrieved context and question to the LLM.

7\. Displays the generated answer through Streamlit.



\## Architecture



```text

YOLOv9 PDF

&#x20;    |

&#x20;    v

PyPDFLoader

&#x20;    |

&#x20;    v

Text Splitting

RecursiveCharacterTextSplitter

&#x20;    |

&#x20;    v

Hugging Face Embeddings

all-mpnet-base-v2

&#x20;    |

&#x20;    v

ChromaDB

&#x20;    |

&#x20;    v

Similarity Retriever

Top 5 Relevant Chunks

&#x20;    |

&#x20;    v

Prompt + Retrieved Context

&#x20;    |

&#x20;    v

Hugging Face Inference API

GPT-OSS-120B

&#x20;    |

&#x20;    v

Generated Answer

&#x20;    |

&#x20;    v

Streamlit UI

```



\## Technologies Used



\* Python

\* LangChain

\* LangChain Classic

\* Hugging Face

\* Sentence Transformers

\* ChromaDB

\* GPT-OSS-120B

\* Streamlit

\* PyPDF



\## RAG Pipeline



\### 1. Document Loading



The YOLOv9 research paper is loaded using `PyPDFLoader`.



```python

loader = PyPDFLoader("yolov9\_paper.pdf")

data = loader.load()

```



\### 2. Text Splitting



The extracted document is divided into smaller chunks.



```python

text\_splitter = RecursiveCharacterTextSplitter(

&#x20;   chunk\_size=1000,

&#x20;   chunk\_overlap=200

)



docs = text\_splitter.split\_documents(data)

```



The overlap helps preserve context between neighboring chunks.



\### 3. Embeddings



Each text chunk is converted into a numerical vector using the Hugging Face Sentence Transformer model:



```text

sentence-transformers/all-mpnet-base-v2

```



These vectors allow the application to perform semantic similarity search.



\### 4. Vector Database



The embeddings and document chunks are stored in ChromaDB.



```python

vectorstore = Chroma.from\_documents(

&#x20;   documents=docs,

&#x20;   embedding=embeddings,

&#x20;   persist\_directory="./chroma\_db"

)

```



\### 5. Retrieval



When a user asks a question, the retriever searches ChromaDB and returns the five most relevant chunks.



```python

retriever = vectorstore.as\_retriever(

&#x20;   search\_type="similarity",

&#x20;   search\_kwargs={"k": 5}

)

```



\### 6. Generation



The retrieved context is passed to GPT-OSS-120B through the Hugging Face Inference API.



The prompt instructs the model to answer using the retrieved context and to say that it does not know when the answer is not available in the context.



\### 7. Streamlit



Streamlit provides a simple web interface where users can enter questions about the YOLOv9 paper.



\## Installation



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

conda create -n rag\_streamlit python=3.10 -y

conda activate rag\_streamlit

```



Install the required packages:



```bash

pip install -r requirements.txt

```



\## Hugging Face API



The application uses a Hugging Face API token to access the hosted LLM.



Do not place your real API token directly inside the source code.



The application accepts the token through the Streamlit interface.



\## Run the Application



Start Streamlit:



```bash

streamlit run app.py

```



Then open the local Streamlit URL shown in the terminal.



For example:



```text

http://localhost:8501

```



\## Example



Example question:



```text

What is new in YOLOv9?

```



The application retrieves relevant sections from the YOLOv9 paper and generates an answer using the retrieved context.



\## Project Structure



```text

GenAI-PDF-RAG/

|

├── app.py

├── Basic\_RAG\_PDF0.ipynb

├── requirements.txt

├── yolov9\_paper.pdf

├── README.md

└── .gitignore

```



The ChromaDB directory is generated locally and is excluded from Git using `.gitignore`.



\## Future Improvements



\* Support multiple PDF documents

\* Add chat history

\* Display document sources for each answer

\* Add PDF upload through Streamlit

\* Improve retrieval with hybrid search

\* Add conversation memory

\* Deploy the application to the cloud



\## Author



Vasudeva Reddy



GitHub: https://github.com/vasudeva-analyst



