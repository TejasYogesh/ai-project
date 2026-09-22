# backend/core/exceptions.py
class AgentError(Exception):
    """Base class for all project errors."""


class ArxivAPIError(AgentError):
    """The arXiv API could not be reached or returned an error."""


class PaperNotFound(AgentError):
    """No paper exists for the given arXiv ID."""