import os
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import PERSIST_DIR, GUIDELINES_COLLECTION_NAME, EMBEDDING_MODEL


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom ChromaDB embedding function using Google's gemini-embedding-001."""
    def __init__(self):
        self.embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedder.embed_documents(input)


# Default enterprise architecture guidelines seed data
DEFAULT_GUIDELINES = [
    {
        "id": "rule_solid_srp",
        "category": "SOLID",
        "tech_stack": "general",
        "text": "Single Responsibility Principle: A class or module should have only one reason to change. Business logic, database interactions, and UI/HTTP handling must be strictly separated."
    },
    {
        "id": "rule_spring_layers",
        "category": "Architecture",
        "tech_stack": "java",
        "text": "Spring Boot Layered Architecture: Controllers (@RestController) must only handle request validation and response formatting. All business logic must reside in @Service components. Persistence logic belongs in @Repository classes."
    },
    {
        "id": "rule_microservice_resilience",
        "category": "Microservices",
        "tech_stack": "general",
        "text": "Microservice Resilience & Circuit Breaking: Inter-service HTTP/gRPC communication must implement circuit breakers (e.g., Resilience4j), timeouts, and retries with backoff. Services must not fail silently."
    },
    {
        "id": "rule_security_owasp_sql",
        "category": "Security",
        "tech_stack": "sql",
        "text": "SQL Injection Prevention: All database queries must use parameterized queries or ORM abstractions (JPA/Hibernate, SQLAlchemy). Dynamic string concatenation in SQL execution is strictly forbidden."
    },
    {
        "id": "rule_python_fastapi_async",
        "category": "Performance",
        "tech_stack": "python",
        "text": "FastAPI Async Operations: Database queries and external API calls inside route handlers must use non-blocking async clients. Synchronous I/O operations must be delegated to background thread pools."
    }
]


class GuidelinesStore:
    def __init__(self):
        self.embedding_fn = GeminiEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=GUIDELINES_COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        self._seed_default_guidelines()

    def _seed_default_guidelines(self):
        """Populate default architecture guidelines if collection is empty."""
        if self.collection.count() == 0:
            documents = [g["text"] for g in DEFAULT_GUIDELINES]
            metadatas = [
                {"category": g["category"], "tech_stack": g["tech_stack"]}
                for g in DEFAULT_GUIDELINES
            ]
            ids = [g["id"] for g in DEFAULT_GUIDELINES]

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Seeded {len(documents)} architectural guidelines into ChromaDB.")

    def query_guidelines(self, query_text: str, top_k: int = 3) -> list[str]:
        """Fetch architectural rules relevant to the code analysis task."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        if results and "documents" in results and results["documents"]:
            return results["documents"][0]
        return []