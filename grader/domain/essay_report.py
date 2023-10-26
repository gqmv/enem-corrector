from pydantic import BaseModel, Field, computed_field


class CompetencyReport(BaseModel):
    competency_name: str
    score: int
    text: str


class EssayReport(BaseModel):
    competency_reports: list[CompetencyReport]

    @computed_field
    def score(self) -> int:
        return sum([cr.score for cr in self.competency_reports])
