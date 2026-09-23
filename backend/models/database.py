"""SQLite persistence: papers, reports (briefings), conversations and messages."""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from backend.core.exceptions import ReportNotFound
from sqlalchemy import ForeignKey, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from backend.core.config import settings
from backend.models.schemas import Briefing, PaperMeta, QAAnswer


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------- tables ----------

class Base(DeclarativeBase):
    pass


class PaperRow(Base):
    __tablename__ = "papers"
    arxiv_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    meta_json: Mapped[str] = mapped_column(Text)            # PaperMeta as JSON
    parse_quality: Mapped[str] = mapped_column(String(20))
    collection_name: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ReportRow(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=_new_id)
    arxiv_id: Mapped[str] = mapped_column(ForeignKey("papers.arxiv_id"))
    user_input: Mapped[str] = mapped_column(Text)            # what the user typed
    briefing_json: Mapped[str] = mapped_column(Text)         # Briefing as JSON
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ConversationRow(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=_new_id)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"))
    arxiv_id: Mapped[str] = mapped_column(String(32))        # which Chroma collection to query
    created_at: Mapped[datetime] = mapped_column(default=_now)


class MessageRow(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    answer_json: Mapped[str] = mapped_column(Text)           # QAAnswer as JSON (includes the question)
    created_at: Mapped[datetime] = mapped_column(default=_now)


_engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})


def _session() -> Session:
    return Session(_engine, expire_on_commit=False)


def init_db() -> None:
    """Create the data folder and all tables (safe to call every time)."""
    Path("data").mkdir(exist_ok=True)
    Base.metadata.create_all(_engine)


# ---------- papers and reports ----------

def save_paper(paper: PaperMeta, parse_quality: str, collection_name: str) -> None:
    """Insert the paper, or update it if it already exists."""
    with _session() as s:
        s.merge(PaperRow(arxiv_id=paper.arxiv_id, title=paper.title,
                         meta_json=paper.model_dump_json(), parse_quality=parse_quality,
                         collection_name=collection_name))
        s.commit()


def save_report(user_input: str, briefing: Briefing) -> str:
    with _session() as s:
        row = ReportRow(arxiv_id=briefing.arxiv_id, user_input=user_input,
                        briefing_json=briefing.model_dump_json())
        s.add(row)
        s.commit()
        return row.id


def get_report(report_id: str) -> Briefing | None:
    with _session() as s:
        row = s.get(ReportRow, report_id)
        return Briefing.model_validate_json(row.briefing_json) if row else None


def latest_report_id_for(arxiv_id: str) -> str | None:
    with _session() as s:
        return s.scalar(select(ReportRow.id).where(ReportRow.arxiv_id == arxiv_id)
                        .order_by(ReportRow.created_at.desc()).limit(1))


def list_reports() -> list[dict]:
    with _session() as s:
        rows = s.execute(select(ReportRow.id, ReportRow.arxiv_id, PaperRow.title, ReportRow.created_at)
                         .join(PaperRow, PaperRow.arxiv_id == ReportRow.arxiv_id)
                         .order_by(ReportRow.created_at.desc()))
        return [dict(row._mapping) for row in rows]


# ---------- conversations ----------

def create_conversation(report_id: str) -> str:
    with _session() as s:
        report = s.get(ReportRow, report_id)
        if report is None:
            raise ReportNotFound(f"No report with ID {report_id}.")
        conversation = ConversationRow(report_id=report_id, arxiv_id=report.arxiv_id)
        s.add(conversation)
        s.commit()
        return conversation.id

def get_conversation(conversation_id: str) -> ConversationRow | None:
    with _session() as s:
        return s.get(ConversationRow, conversation_id)


def add_message(conversation_id: str, answer: QAAnswer) -> None:
    with _session() as s:
        s.add(MessageRow(conversation_id=conversation_id, answer_json=answer.model_dump_json()))
        s.commit()


def get_history(conversation_id: str) -> list[QAAnswer]:
    """All answers in a conversation, oldest first."""
    with _session() as s:
        rows = s.scalars(select(MessageRow.answer_json)
                         .where(MessageRow.conversation_id == conversation_id)
                         .order_by(MessageRow.id))
        return [QAAnswer.model_validate_json(r) for r in rows]
    
def get_paper(arxiv_id: str) -> PaperRow | None:
    with _session() as s:
        return s.get(PaperRow, arxiv_id)


def report_ids_for(arxiv_id: str) -> list[str]:
    """All report IDs for a paper, newest first."""
    with _session() as s:
        return list(s.scalars(select(ReportRow.id).where(ReportRow.arxiv_id == arxiv_id)
                              .order_by(ReportRow.created_at.desc())))