from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import MODEL_NAME
from agents.state import ReviewState
from vector_store.guidelines_store import GuidelinesStore
from vector_store.ephemeral_store import EphemeralCodebaseStore

# Shared LLM Instance
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)


def supervisor_node(state: ReviewState) -> dict:
    """
    Supervisor Agent that manages workflow state, delegates sub-tasks,
    and synthesizes the final review output when approved.
    """
    status = state.get("review_status", "IN_PROGRESS")
    guidelines = state.get("guidelines", [])
    analysis_draft = state.get("analysis_draft", "")
    iteration_count = state.get("iteration_count", 0)

    # 1. First Pass: Fetch Architecture Rules
    if not guidelines:
        return {"next_step": "retriever"}

    # 2. Second Pass: Run Code Analysis
    if not analysis_draft:
        return {"next_step": "code_analyzer"}

    # 3. Third Pass: Trigger Quality Gate Reviewer
    if status == "IN_PROGRESS":
        return {"next_step": "reviewer"}

    # 4. Handle Revision Loop if Rejected (Max 2 Retries)
    if status == "REJECTED" and iteration_count < 2:
        print(f"Supervisor: Analysis rejected. Triggering revision loop (Attempt {iteration_count + 1})...")
        return {"next_step": "code_analyzer"}

    # 5. Final Synthesis Pass: Build Final Markdown Report
    print("Supervisor: Analysis approved or max retries reached. Synthesizing final report...")
    system_prompt = SystemMessage(
        content="You are a Principal Software Architect. Synthesize the code analysis findings and "
                "reviewer critique into a structured, highly actionable Markdown review report."
    )
    user_prompt = HumanMessage(
        content=f"Tech Stack: {state.get('tech_stack')}\n"
                f"Architecture Rules applied:\n{guidelines}\n\n"
                f"Code Analysis Draft:\n{analysis_draft}\n\n"
                f"Reviewer Critique:\n{state.get('reviewer_critique', 'None')}"
    )

    response = llm.invoke([system_prompt, user_prompt])
    
    return {
        "final_report": response.content,
        "next_step": "END"
    }


def guidelines_retriever_node(state: ReviewState) -> dict:
    """Queries persistent ChromaDB store for architectural guidelines matching tech stack."""
    tech_stack = state.get("tech_stack", "general")
    print(f"Guidelines Agent: Fetching architectural rules for tech stack [{tech_stack}]...")

    store = GuidelinesStore()
    retrieved_rules = store.query_guidelines(
        query_text=f"Architecture best practices and design principles for {tech_stack}",
        top_k=4
    )

    return {
        "guidelines": retrieved_rules,
        "next_step": "supervisor"
    }


def code_analyzer_node(state: ReviewState) -> dict:
    """Queries session ChromaDB store for relevant code chunks and evaluates compliance."""
    session_id = state.get("session_id")
    tech_stack = state.get("tech_stack")
    guidelines = state.get("guidelines", [])
    critique = state.get("reviewer_critique", "")

    print(f"Code Analyzer Agent: Inspecting repository vectors for session [{session_id}]...")

    # Fetch code chunks from session vector store
    ephemeral_store = EphemeralCodebaseStore(session_id=session_id)
    relevant_chunks = ephemeral_store.query_codebase(
        query_text=f"Controllers, Services, Repositories, API endpoints, error handling, and architecture in {tech_stack}",
        top_k=6
    )

    code_context = "\n\n".join(
        [f"--- File: {c['file_path']} ---\n{c['content']}" for c in relevant_chunks]
    )

    system_prompt = SystemMessage(
        content="You are a Senior Code Auditor. Review the provided source code against the target "
                "architecture guidelines. Identify architecture smells, SRP violations, security risks, "
                "and missing design patterns."
    )
    
    revision_context = f"\nPrevious Critique to Fix:\n{critique}" if critique else ""
    user_prompt = HumanMessage(
        content=f"Target Tech Stack: {tech_stack}\n"
                f"Guidelines to Enforce:\n{guidelines}\n\n"
                f"Source Code Samples:\n{code_context}"
                f"{revision_context}"
    )

    response = llm.invoke([system_prompt, user_prompt])

    return {
        "analysis_draft": response.content,
        "next_step": "supervisor"
    }


def final_reviewer_node(state: ReviewState) -> dict:
    """Quality gate node that evaluates the code analysis draft for accuracy and completeness."""
    analysis_draft = state.get("analysis_draft", "")
    guidelines = state.get("guidelines", [])
    current_iterations = state.get("iteration_count", 0)

    print("Final Reviewer Agent: Running Quality Gate evaluation...")

    system_prompt = SystemMessage(
        content="You are a Technical Quality Gate Evaluator. Review the candidate code analysis draft. "
                "Ensure it evaluates code against guidelines accurately without hallucinations. "
                "Respond with status APPROVED if thorough, or REJECTED with constructive critique if incomplete."
    )
    user_prompt = HumanMessage(
        content=f"Guidelines:\n{guidelines}\n\n"
                f"Analysis Draft:\n{analysis_draft}\n\n"
                "Provide your evaluation in format:\n"
                "STATUS: [APPROVED or REJECTED]\n"
                "CRITIQUE: [Your detailed feedback]"
    )

    response = llm.invoke([system_prompt, user_prompt])
    text = response.content

    if "STATUS: APPROVED" in text or "APPROVED" in text.split("\n")[0]:
        status = "APPROVED"
        critique = "Analysis approved by Quality Gate."
    else:
        status = "REJECTED"
        critique = text.replace("STATUS: REJECTED", "").strip()

    return {
        "review_status": status,
        "reviewer_critique": critique,
        "iteration_count": current_iterations + 1,
        "next_step": "supervisor"
    }