"""Persistent sessions: run the paper graph and save the result; answer within a conversation."""
import logging

from backend.agents.paper_agent import run_paper_graph
from backend.agents.qa_agent import ask
from backend.core.exceptions import ConversationNotFound, PipelineError
from backend.models import database as db
from backend.models.schemas import QAAnswer, ReportSession
from backend.services import vector_store
from backend.services.arxiv_client import extract_arxiv_id

logger = logging.getLogger(__name__)


def create_report(user_input: str, refresh: bool = False) -> ReportSession:
    """Brief a paper (or reuse a saved briefing) and start a new conversation about it."""
    arxiv_id = extract_arxiv_id(user_input)

    # Reuse: same paper already briefed and indexed -> no LLM calls at all
    if arxiv_id and not refresh:
        existing = db.latest_report_id_for(arxiv_id)
        if existing and vector_store.is_indexed(arxiv_id):
            logger.info("Reusing saved report %s for %s", existing, arxiv_id)
            return ReportSession(report_id=existing,
                                 conversation_id=db.create_conversation(existing),
                                 briefing=db.get_report(existing), reused=True)

    state = run_paper_graph(user_input)
    if state.get("error"):
        raise PipelineError(f"{state['error']} (failed in: {state.get('failed_node')})")

    db.save_paper(state["paper"], state["parsed"].parse_quality, state["collection_name"])
    report_id = db.save_report(user_input, state["briefing"])
    return ReportSession(report_id=report_id,
                         conversation_id=db.create_conversation(report_id),
                         briefing=state["briefing"],
                         steps=state.get("steps", []),
                         candidates=state.get("candidates", []))


def ask_in_conversation(conversation_id: str, question: str) -> QAAnswer:
    """Answer a question using the conversation's saved history, then save the answer."""
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(f"No conversation with ID {conversation_id}.")
    history = db.get_history(conversation_id)
    answer = ask(conversation.arxiv_id, question, history)
    db.add_message(conversation_id, answer)
    return answer