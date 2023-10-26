from anyio import run
from domain.essay import Essay
from domain.essay_report import CompetencyReport, EssayReport
from fastapi import APIRouter
from services.essay_grader_service import get_grade_from_gpt

router = APIRouter(prefix="/essay_report")


@router.post("/essay")
async def submit_essay(essay: Essay) -> EssayReport:
    """Receives an essay and returns a report with the grade for each competency."""

    return await get_grade_from_gpt(essay)
