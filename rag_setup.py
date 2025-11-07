import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "Course Materials/"
CHROMA_PERSIST_DIR = "ChromaDB Store"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" 

def setup_knowledge_base():
    """Loads multiple documents, splits, creates embeddings, and saves to ChromaDB."""
    
    # 0. Ensure the data directory exists and create the files listed above inside it.
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory: {DATA_DIR}. Please place your .txt files inside.")
        os.makedirs(DATA_DIR)
        return # Stop execution until data is placed

    print("--- Starting RAG Knowledge Base Setup for Multiple Files ---")

    # 1. Load Data from Directory
    # Loads all .txt files in the specified directory
    loader = DirectoryLoader(
        path=DATA_DIR, 
        glob="**/*.txt", 
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} source documents from the directory.")

    # 2. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # Use a generous chunk size for textbook-like content
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} text chunks.")

    # 3. Create Embeddings & Store in ChromaDB (Same as before)
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    vectorstore.persist()
    print(f"Knowledge Base successfully created with multiple sources and saved to {CHROMA_PERSIST_DIR}")
    return vectorstore

if __name__ == "__main__":
    setup_knowledge_base()