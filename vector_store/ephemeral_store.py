import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import EMBEDDING_MODEL


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom ChromaDB embedding function using Google's gemini-embedding-001."""
    def __init__(self):
        self.embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedder.embed_documents(input)


class EphemeralCodebaseStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.collection_name = f"repo_session_{session_id}"
        self.embedding_fn = GeminiEmbeddingFunction()
        
        # In-memory ChromaDB client for session-scoped data
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def ingest_code_chunks(self, code_chunks: list[dict]):
        """
        Store code chunks into the session vector collection.
        Each chunk dict must contain 'content', 'file_path', and 'chunk_index'.
        """
        if not code_chunks:
            return

        documents = [c["content"] for c in code_chunks]
        metadatas = [
            {"file_path": c["file_path"], "chunk_index": c["chunk_index"]}
            for c in code_chunks
        ]
        ids = [f"code_{c['file_path']}_{c['chunk_index']}" for c in code_chunks]

        # Ingest in batches of 100 to avoid request size limits
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

        print(f"Session [{self.session_id}]: Ingested {len(documents)} code chunks.")

    def query_codebase(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Query code chunks relevant to an architectural check or question."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        retrieved_chunks = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            meta = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            
            for doc, m in zip(docs, meta):
                retrieved_chunks.append({
                    "content": doc,
                    "file_path": m.get("file_path", "unknown")
                })

        return retrieved_chunks

    def cleanup(self):
        """Purge the dynamic collection from memory when the session terminates."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Session [{self.session_id}]: Ephemeral vector storage purged successfully.")
        except Exception as e:
            print(f"Error purging session storage: {e}")