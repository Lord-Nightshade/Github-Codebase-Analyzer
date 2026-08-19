import sys
import uuid
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import MODEL_NAME
from graph import app
from tools.github_cloner import clone_and_parse_repo
from vector_store.ephemeral_store import EphemeralCodebaseStore

load_dotenv()


def main():
    print("==================================================")
    print("   GitHub Code Architecture Reviewer System      ")
    print("==================================================")

    repo_url = input("\nEnter GitHub Repository URL: ").strip()
    tech_stack = input("Enter Target Tech Stack (e.g., Java/Spring Boot, Python/FastAPI): ").strip()

    if not repo_url or not tech_stack:
        print("Error: Both Repository URL and Tech Stack are required.")
        sys.exit(1)

    # Generate unique session token
    session_id = str(uuid.uuid4())[:8]
    print(f"\nInitializing session [{session_id}]...")

    ephemeral_store = EphemeralCodebaseStore(session_id=session_id)

    try:
        # 1. Clone & Chunk Target Codebase
        code_chunks = clone_and_parse_repo(repo_url)
        if not code_chunks:
            print("No supported source files were found in this repository.")
            return

        # 2. Ingest Chunks into Ephemeral Vector Store
        ephemeral_store.ingest_code_chunks(code_chunks)

        # 3. Invoke LangGraph Execution
        print("\nStarting Multi-Agent Architectural Analysis...")
        initial_state = {
            "messages": [],
            "session_id": session_id,
            "repo_url": repo_url,
            "tech_stack": tech_stack,
            "guidelines": [],
            "analysis_draft": "",
            "reviewer_critique": "",
            "review_status": "IN_PROGRESS",
            "iteration_count": 0,
            "final_report": "",
            "next_step": "supervisor"
        }

        result = app.invoke(initial_state)

        print("\n" + "=" * 60)
        print("                FINAL ARCHITECTURAL REPORT              ")
        print("=" * 60 + "\n")
        print(result.get("final_report", "No report was generated."))
        print("\n" + "=" * 60)

        # 4. Interactive Follow-Up Chat Loop
        print("\nYou can now ask follow-up questions about this repository.")
        print("Type 'exit' or 'quit' to terminate session and purge memory.\n")

        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)

        while True:
            user_query = input("\n[Session Active] Ask question > ").strip()
            if user_query.lower() in ["exit", "quit"]:
                print("Terminating chat session...")
                break

            if not user_query:
                continue

            # RAG Retrieval against active ephemeral session vectors
            retrieved_chunks = ephemeral_store.query_codebase(user_query, top_k=4)
            context_str = "\n\n".join(
                [f"// File: {c['file_path']}\n{c['content']}" for c in retrieved_chunks]
            )

            system_prompt = SystemMessage(
                content="You are a Lead Software Architect answering follow-up queries regarding "
                        "the evaluated codebase. Refer to provided source snippets to answer accurately."
            )
            user_prompt = HumanMessage(
                content=f"Source Code Context:\n{context_str}\n\nUser Question: {user_query}"
            )

            response = llm.invoke([system_prompt, user_prompt])
            print(f"\nAnswer:\n{response.content}")

    except Exception as e:
        print(f"\nExecution error encountered: {e}")

    finally:
        # 5. Purge Session Vector Collection
        print("\nCleaning up session vector storage...")
        ephemeral_store.cleanup()
        print("Session memory purged clean.")


if __name__ == "__main__":
    main()