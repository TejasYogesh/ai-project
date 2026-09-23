"""Command-line interface: brief an arXiv paper, then chat about it. Everything is saved in SQLite."""
import logging

import typer

from backend.agents import session
from backend.core.exceptions import AgentError
from backend.core.llm import init_llm
from backend.models import database as db
from backend.services.summarizer import briefing_to_markdown

app = typer.Typer(help="arXiv paper digest & QA agent", add_completion=False)


def _setup(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    init_llm()
    db.init_db()


def _chat_loop(conversation_id: str) -> None:
    typer.echo(f"\nAsk questions about the paper. Type 'exit' to quit. "
               f"(conversation {conversation_id})\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            answer = session.ask_in_conversation(conversation_id, question)
        except AgentError as e:
            typer.echo(f"Error: {e}\n")
            continue
        typer.echo(f"\nAgent: {answer.answer}")
        for c in answer.citations:
            typer.echo(f"   [{c.number}] {c.section} (p.{c.page})")
        typer.echo("")
    typer.echo(f"Saved. Resume later with: python cli.py chat {conversation_id}")


@app.command()
def brief(user_input: str = typer.Argument(..., help="arXiv ID, arXiv URL, or a research topic"),
          start_chat: bool = typer.Option(True, "--chat/--no-chat", help="Start QA after the briefing"),
          refresh: bool = typer.Option(False, "--refresh", help="Regenerate even if a briefing is saved"),
          verbose: bool = typer.Option(False, "--verbose", "-v", help="Show the agent's steps")):
    """Generate (or reuse) an executive briefing, then optionally chat about the paper."""
    _setup(verbose)
    try:
        result = session.create_report(user_input, refresh=refresh)
    except AgentError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)

    typer.echo("\n" + briefing_to_markdown(result.briefing))
    if result.reused:
        typer.echo("\n(Saved briefing reused. Use --refresh to regenerate.)")
    if len(result.candidates) > 1:
        typer.echo("\nOther relevant papers found:")
        for c in result.candidates[1:]:
            typer.echo(f"   {c.paper.arxiv_id}  {c.paper.title}")
    typer.echo(f"\nreport_id={result.report_id}  conversation_id={result.conversation_id}")
    if start_chat:
        _chat_loop(result.conversation_id)


@app.command()
def chat(conversation_id: str = typer.Argument(..., help="ID printed after a briefing"),
         verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Resume a saved conversation."""
    _setup(verbose)
    if db.get_conversation(conversation_id) is None:
        typer.echo(f"No conversation with ID {conversation_id}.")
        raise typer.Exit(code=1)
    for turn in db.get_history(conversation_id):
        typer.echo(f"You: {turn.question}\nAgent: {turn.answer}\n")
    _chat_loop(conversation_id)


@app.command("list")
def list_reports():
    """List saved briefings."""
    db.init_db()
    for r in db.list_reports():
        typer.echo(f"{r['id']}  {r['created_at']:%Y-%m-%d %H:%M}  {r['arxiv_id']}  {r['title']}")


if __name__ == "__main__":
    app()