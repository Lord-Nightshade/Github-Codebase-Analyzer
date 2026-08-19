import uuid
from dotenv import load_dotenv

load_dotenv()

from tools.github_cloner import clone_and_parse_repo
from vector_store.ephemeral_store import EphemeralCodebaseStore
from graph import app


def test_github_cloner():
    print("\n--- 1. Testing GitHub Cloner & Chunking Tool ---")
    # Using a small public repository for quick testing
    test_repo_url = "https://github.com/bottlepy/bottle"
    
    chunks = clone_and_parse_repo(test_repo_url)
    print(f"Cloned and parsed repo. Total chunks generated: {len(chunks)}")
    
    if chunks:
        print(f"Sample chunk path: {chunks[0]['file_path']}")
        print(f"Sample chunk content snippet:\n{chunks[0]['content'][:100]}...")

    assert len(chunks) > 0, "Cloner returned zero chunks!"
    print("✅ GitHub Cloner test passed.")


def test_agent_graph_execution():
    print("\n--- 2. Testing Multi-Agent LangGraph Workflow ---")
    session_id = f"graphtest_{uuid.uuid4().hex[:6]}"
    
    # 1. Seed dummy code into ephemeral store for the agent graph to query
    ephemeral_store = EphemeralCodebaseStore(session_id=session_id)
    dummy_chunks = [
        {
            "content": "// File: OrderController.java\n@RestController\npublic class OrderController {\n"
                       "    @Autowired private OrderService service;\n"
                       "    @PostMapping('/order') public Response createOrder(@RequestBody Order order) {\n"
                       "        return service.processOrder(order);\n    }\n}",
            "file_path": "src/main/java/com/app/OrderController.java",
            "chunk_index": 0
        }
    ]
    ephemeral_store.ingest_code_chunks(dummy_chunks)

    # 2. Invoke the compiled LangGraph workflow
    initial_state = {
        "messages": [],
        "session_id": session_id,
        "repo_url": "https://github.com/example/demo",
        "tech_stack": "Java/Spring Boot",
        "guidelines": [],
        "analysis_draft": "",
        "reviewer_critique": "",
        "review_status": "IN_PROGRESS",
        "iteration_count": 0,
        "final_report": "",
        "next_step": "supervisor"
    }

    print("Executing app.invoke() through Supervisor, Guidelines, Analyzer, and Reviewer nodes...")
    final_state = app.invoke(initial_state)

    print("\nGraph Execution Results:")
    print(f"  - Final Review Status: {final_state.get('review_status')}")
    print(f"  - Total Iterations: {final_state.get('iteration_count')}")
    print(f"  - Final Report Generated: {bool(final_state.get('final_report'))}")

    assert final_state.get("final_report"), "Graph execution failed to generate a final report!"

    # 3. Clean up test session storage
    ephemeral_store.cleanup()
    print("✅ Agent Graph execution test passed.")


if __name__ == "__main__":
    print("Starting System Components Sanity Checks...")
    try:
        test_github_cloner()
        test_agent_graph_execution()
        print("\n🎉 All component tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")