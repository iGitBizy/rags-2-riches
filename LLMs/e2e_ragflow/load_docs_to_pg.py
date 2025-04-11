""" load documents into a postgres vector db """

# Install requried packages
"""
psycopg2: connect to Postgres
pgvector + sqlalchemy: manage the vector table
transformers + sentence-transformers: for embeddings 
"""
pip install psycopg2 langchain transformers sentence-transformers faiss-cpu pgvector sqlalchemy
pip install python-dotenv

# --- Step 1: Import libraries ---
import psycopg2
import numpy as np
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# --- Step 2: Load environment variables ---
load_dotenv()  # This loads variables from the .env file into os.environ

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# --- Step 3: Connect to Postgres using env vars ---
conn = psycopg2.connect(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)
cursor = conn.cursor()

# --- Step 4: Load and Split Documents ---
loader = DirectoryLoader('./documents', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# --- Step 5: Generate Embeddings ---
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = MistralAIEmbeddings(model="mistral-embed")

# --- Step 6: Insert into Postgres ---
for doc in docs:
    text = doc.page_content
    vector = embedding_model.embed_query(text)
    vector = np.array(vector).tolist()
    
    cursor.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (text, vector)
    )

conn.commit()
cursor.close()
conn.close()

print("Successfully loaded documents into Postgres securely")


