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
    
class ArxivQueryError(ArxivAPIError):
    """arXiv rejected the search query itself (HTTP 400), e.g. invalid syntax."""
    
    
class QAError(AgentError):
    """The QA agent could not complete (e.g. the LLM was unavailable)."""
    
class PipelineError(AgentError):
    """The paper graph ended with an error."""


class ReportNotFound(AgentError):
    """No saved briefing has this ID."""


class ConversationNotFound(AgentError):
    """No saved conversation has this ID."""