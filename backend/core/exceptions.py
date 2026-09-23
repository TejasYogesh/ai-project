"""Custom exceptions for the project."""


class AgentError(Exception):
    """Base class for all project errors."""


class ArxivAPIError(AgentError):
    """The arXiv API could not be reached or returned an error."""


class PaperNotFound(AgentError):
    """No paper exists for the given arXiv ID."""


class PDFDownloadError(AgentError):
    """The PDF could not be downloaded."""


class PDFParseError(AgentError):
    """The PDF could not be opened or read."""
    

class PaperNotIndexed(AgentError):
    """A question was asked about a paper that has not been indexed yet."""
    
    
class BriefingError(AgentError):
    """The LLM could not produce a valid briefing."""