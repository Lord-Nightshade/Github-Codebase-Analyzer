import chromadb
from chromadb.utils import embedding_functions


class EphemeralCodebaseStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.collection_name = f"repo_session_{session_id}"
        
        # Built-in ChromaDB wrapper for sentence-transformers (downloads ~90MB model on first run)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # In-memory ChromaDB client
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def ingest_code_chunks(self, code_chunks: list[dict]):
        """
        Store code chunks into the session vector collection.
        Uses local embeddings with no API rate limits or costs.
        """
        if not code_chunks:
            return

        # Exclude test files to keep storage clean and fast
        filtered_chunks = [
            c for c in code_chunks
            if "src/test" not in c.get("file_path", "").replace("\\", "/").lower()
            and "/test_" not in c.get("file_path", "").replace("\\", "/").lower()
        ]
        
        chunks_to_ingest = filtered_chunks if filtered_chunks else code_chunks

        documents = [c["content"] for c in chunks_to_ingest]
        metadatas = [
            {"file_path": c["file_path"], "chunk_index": c["chunk_index"]}
            for c in chunks_to_ingest
        ]
        ids = [f"code_{c['file_path']}_{c['chunk_index']}" for c in chunks_to_ingest]

        # Local embeddings allow large batch sizes with zero API delays
        batch_size = 200
        total_documents = len(documents)

        print(f"Session [{self.session_id}]: Ingesting {total_documents} code chunks locally...")

        for i in range(0, total_documents, batch_size):
            self.collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

        print(f"Session [{self.session_id}]: Ingestion complete ({total_documents} chunks stored locally).")

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