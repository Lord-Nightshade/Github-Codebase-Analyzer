from langgraph.graph import StateGraph, START, END

from agents.state import ReviewState
from agents.nodes import (
    supervisor_node,
    guidelines_retriever_node,
    code_analyzer_node,
    final_reviewer_node
)
from agents.edges import route_next_step


def build_graph():
    """Assembles and compiles the multi-agent architectural review graph."""
    workflow = StateGraph(ReviewState)

    # 1. Register Graph Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("guidelines_retriever", guidelines_retriever_node)
    workflow.add_node("code_analyzer", code_analyzer_node)
    workflow.add_node("final_reviewer", final_reviewer_node)

    # 2. Define Start Edge
    workflow.add_edge(START, "supervisor")

    # 3. Dynamic Supervisor Routing Edges
    workflow.add_conditional_edges(
        "supervisor",
        route_next_step,
        {
            "guidelines_retriever": "guidelines_retriever",
            "code_analyzer": "code_analyzer",
            "final_reviewer": "final_reviewer",
            "END": END
        }
    )

    # 4. Worker Nodes Route Back to Supervisor
    workflow.add_edge("guidelines_retriever", "supervisor")
    workflow.add_edge("code_analyzer", "supervisor")
    workflow.add_edge("final_reviewer", "supervisor")

    return workflow.compile()


# Export compiled graph executable
app = build_graph()