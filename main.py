import sys
import uuid
import warnings
from dotenv import load_dotenv

# Suppress Gemini sampling parameter warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import MODEL_NAME
from graph import app
from tools.github_cloner import clone_and_parse_repo
from vector_store.ephemeral_store import EphemeralCodebaseStore

load_dotenv()


def parse_response_text(content) -> str:
    """Extract clean text string from Gemini response blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def main():
    print("==================================================")
    print("    GitHub Code Architecture Reviewer System      ")
    print("==================================================")

    repo_url = input("\nEnter GitHub Repository URL: ").strip()
    tech_stack = input("Enter Target Tech Stack (e.g., Java/Spring Boot, Python/FastAPI): ").strip()

    if not repo_url or not tech_stack:
        print("Error: Both Repository URL and Tech Stack are required.")
        sys.exit(1)

    session_id = str(uuid.uuid4())[:8]
    print(f"\nInitializing session [{session_id}]...")

    ephemeral_store = EphemeralCodebaseStore(session_id=session_id)

    try:
        # 1. Clone & Chunk
        code_chunks = clone_and_parse_repo(repo_url)
        if not code_chunks:
            print("No supported source files were found in this repository.")
            return

        # 2. Ingest Chunks
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
        print("               FINAL ARCHITECTURAL REPORT              ")
        print("=" * 60 + "\n")
        
        final_report_raw = result.get("final_report", "No report was generated.")
        final_report = parse_response_text(final_report_raw)
        print(final_report)
        print("\n" + "=" * 60)

        # 4. Interactive Follow-Up Chat Loop with Memory
        print("\nYou can now ask follow-up questions about this repository.")
        print("Type 'exit' or 'quit' to terminate session and purge memory.\n")

        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)

        # Base System Prompt containing Defensive Framing + Initial Report Context
        system_instruction = SystemMessage(
            content=(
                "You are an expert Defensive Software Architect helping a developer improve their application's "
                "design, resilience, and coding practices.\n"
                "You are reviewing source code for standard software engineering quality, design patterns, "
                "and architectural improvements.\n\n"
                f"### INITIAL ARCHITECTURAL REPORT FOR THIS REPO:\n{final_report}\n\n"
                "When answering follow-up queries, refer to the provided source code context and the initial report."
            )
        )

        # Maintain conversation history across chat turns
        chat_history = []

        while True:
            user_query = input("\n[Session Active] Ask question > ").strip()
            if user_query.lower() in ["exit", "quit"]:
                print("Terminating chat session...")
                break

            if not user_query:
                continue

            # RAG Retrieval against active vector store
            retrieved_chunks = ephemeral_store.query_codebase(user_query, top_k=4)
            context_str = "\n\n".join(
                [f"// File: {c['file_path']}\n{c['content']}" for c in retrieved_chunks]
            )

            # Construct message history for LLM
            current_user_message = HumanMessage(
                content=f"Source Code Context for Query:\n{context_str}\n\nUser Question: {user_query}"
            )

            # Pass System Instruction + Past History + Current Query
            full_prompt = [system_instruction] + chat_history + [current_user_message]

            response = llm.invoke(full_prompt)
            clean_answer = parse_response_text(response.content)

            print(f"\nAnswer:\n{clean_answer}")

            # Append to history (keep history bounded to last 6 turns to avoid context bloat)
            chat_history.append(HumanMessage(content=user_query))
            chat_history.append(AIMessage(content=clean_answer))
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]

    except Exception as e:
        print(f"\nExecution error encountered: {e}")

    finally:
        print("\nCleaning up session vector storage...")
        ephemeral_store.cleanup()
        print("Session memory purged clean.")


if __name__ == "__main__":
    main()