import chromadb
import yaml
import ollama

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHROMA_PATH = config["chroma_path"]
LLM_MODEL = config["llm_model"]
EMBEDDINGS_MODEL = config["embeddings_model"]

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("rag_collection")

def get_embedding(text):
    """Generate embeddings using Ollama with Nomic."""
    response = ollama.embeddings(model=EMBEDDINGS_MODEL, prompt=text)
    return response["embedding"]

def retrieve_relevant_docs(query):
    """Retrieve top 3 relevant documents from ChromaDB."""
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return [r.get("text", "") for r in results["metadatas"][0]]

def generate_response(query, context):
    """Generate a response using Ollama LLaMA 3.2 3B."""
    prompt = f"Context: {context}\n\nUser Query: {query}\n\nAnswer:"
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def query_rag(user_query):
    """Retrieve relevant documents and generate a response."""
    relevant_docs = retrieve_relevant_docs(user_query)
    context = "\n".join(relevant_docs) if relevant_docs else "No relevant data found."
    
    return generate_response(user_query, context)

def reset_chromadb():
    """Deletes and re-creates ChromaDB collection."""
    chroma_client.delete_collection("rag_collection")
