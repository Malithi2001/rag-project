import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Load your API Key (We will create the .env file next)
load_dotenv()

# 2. Setup the "Brain" for the database
# This turns text into numbers the database can search
embeddings = OpenAIEmbeddings()

# 3. Load and Split the real data
loader = PyPDFLoader("data.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

# 4. Create the Real Vector Database
# 'persist_directory' saves the data to a folder called 'db'
print("Creating vector database...")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)
print("Database created and saved!")