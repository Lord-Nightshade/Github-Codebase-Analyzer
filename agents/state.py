import operator
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage


class ReviewState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    session_id: str
    repo_url: str
    tech_stack: str
    guidelines: List[str]
    analysis_draft: str
    reviewer_critique: str
    review_status: str  # "IN_PROGRESS", "APPROVED", or "REJECTED"
    iteration_count: int
    final_report: str
    next_step: str  # Used by routing logic ("retriever", "code_analyzer", "reviewer", "END")