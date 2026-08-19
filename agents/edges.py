from agents.state import ReviewState


def route_next_step(state: ReviewState) -> str:
    """Determines the next node execution path based on state['next_step']."""
    next_step = state.get("next_step", "supervisor")
    
    if next_step == "retriever":
        return "guidelines_retriever"
    elif next_step == "code_analyzer":
        return "code_analyzer"
    elif next_step == "reviewer":
        return "final_reviewer"
    elif next_step == "supervisor":
        return "supervisor"
    elif next_step == "END":
        return "END"
    
    return "supervisor"