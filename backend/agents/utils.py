"""Helpers shared by all graphs."""
import functools
import logging

from google.genai import errors as genai_errors

from backend.core.exceptions import AgentError

logger = logging.getLogger(__name__)


def graph_node(name: str):
    """Wrap a node so it records its name in `steps` and turns known failures into
    an `error` in the state instead of crashing the whole graph."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            logger.info("-> node: %s", name)
            try:
                update = fn(state) or {}
            except (AgentError, genai_errors.APIError) as e:
                logger.warning("Node %s failed: %s", name, e)
                return {"error": str(e), "failed_node": name, "steps": [name]}
            if update.get("error"):
                update.setdefault("failed_node", name)       # fixes "ERROR in None"
            return {**update, "steps": [name] + update.get("steps", [])}
        return wrapper
    return decorator