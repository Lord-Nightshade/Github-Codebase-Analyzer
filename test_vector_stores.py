import uuid
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from vector_store.guidelines_store import GuidelinesStore
from vector_store.ephemeral_store import EphemeralCodebaseStore


def test_guidelines_store():
    print("\n--- 1. Testing Persistent GuidelinesStore ---")
    store = GuidelinesStore()

    # Query guidelines
    rules = store.query_guidelines("Spring Boot layered architecture and controllers", top_k=2)
    print(f"Retrieved {len(rules)} rule(s):")
    for idx, rule in enumerate(rules, 1):
        print(f"  [{idx}] {rule[:100]}...")

    assert len(rules) > 0, "Guidelines query returned 0 results!"
    print("✅ GuidelinesStore test passed.")


def test_ephemeral_store():
    print("\n--- 2. Testing Session EphemeralCodebaseStore ---")
    session_id = f"sanity_{uuid.uuid4().hex[:6]}"
    ephemeral = EphemeralCodebaseStore(session_id=session_id)

    dummy_chunks = [
        {
            "content": "// File: UserController.java\n@RestController\npublic class UserController { @GetMapping('/users') public List<User> getUsers() { return userService.findAll(); } }",
            "file_path": "src/main/java/com/example/UserController.java",
            "chunk_index": 0
        },
        {
            "content": "// File: UserRepository.java\npublic interface UserRepository extends JpaRepository<User, Long> {}",
            "file_path": "src/main/java/com/example/UserRepository.java",
            "chunk_index": 0
        }
    ]

    print(f"Ingesting {len(dummy_chunks)} dummy code chunks into session [{session_id}]...")
    ephemeral.ingest_code_chunks(dummy_chunks)

    # Query ephemeral store
    results = ephemeral.query_codebase("Find REST controller endpoints", top_k=1)
    print(f"Retrieved {len(results)} chunk(s):")
    for res in results:
        print(f"  - File: {res['file_path']}")
        print(f"    Snippet: {res['content'][:70]}...")

    assert len(results) > 0, "Ephemeral codebase query returned 0 results!"

    # Test Session Purge
    print(f"Purging ephemeral storage for session [{session_id}]...")
    ephemeral.cleanup()
    print("✅ EphemeralCodebaseStore test passed.")


if __name__ == "__main__":
    print("Starting Vector Store Sanity Checks...")
    try:
        test_guidelines_store()
        test_ephemeral_store()
        print("\n🎉 All vector store sanity checks completed successfully!")
    except Exception as e:
        print(f"\n❌ Sanity check failed with error: {e}")