"""QA conversations about a briefed paper."""
from fastapi import APIRouter

from backend.agents import session
from backend.core.exceptions import ConversationNotFound
from backend.models import database as db
from backend.models.schemas import (ConversationCreate, ConversationCreated, ConversationView,
                                    QAAnswer, QuestionRequest)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationCreated)
def create_conversation(request: ConversationCreate) -> ConversationCreated:
    """Start a new conversation about an existing report."""
    conversation_id = db.create_conversation(request.report_id)
    conversation = db.get_conversation(conversation_id)
    return ConversationCreated(conversation_id=conversation_id, report_id=request.report_id,
                               arxiv_id=conversation.arxiv_id)


@router.post("/{conversation_id}/messages", response_model=QAAnswer)
def ask(conversation_id: str, request: QuestionRequest) -> QAAnswer:
    """Ask a question. Answers are grounded in the paper, with citations."""
    return session.ask_in_conversation(conversation_id, request.question)


@router.get("/{conversation_id}", response_model=ConversationView)
def get_conversation(conversation_id: str) -> ConversationView:
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(f"No conversation with ID {conversation_id}.")
    return ConversationView(conversation_id=conversation.id, report_id=conversation.report_id,
                            arxiv_id=conversation.arxiv_id,
                            messages=db.get_history(conversation_id))