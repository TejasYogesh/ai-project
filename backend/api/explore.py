"""Topic search without briefing: see which papers the explorer agent finds."""
from fastapi import APIRouter

from backend.agents.topic_explorer import explorer_graph
from backend.core.exceptions import PipelineError
from backend.models.schemas import ExploreRequest, ExploreResponse

router = APIRouter(tags=["explore"])


@router.post("/explore", response_model=ExploreResponse)
def explore(request: ExploreRequest) -> ExploreResponse:
    """Run the topic explorer agent and return the chosen paper and shortlist."""
    result = explorer_graph.invoke({"topic": request.topic})
    if result.get("error"):
        raise PipelineError(result["error"])
    return ExploreResponse(chosen=result["paper"], shortlist=result["shortlist"],
                           tried_queries=result.get("tried_queries", []),
                           warnings=result.get("warnings", []))