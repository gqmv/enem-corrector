from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services import get_grade_from_gpt


class Essay(BaseModel):
    text: str
    title: str


class GradeCategory(BaseModel):
    name: str
    score: int
    description: str


class Grade(BaseModel):
    categories: list[GradeCategory]
    score: int


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/grade")
def get_grade(essay: Essay) -> Grade:
    gpt_grade = get_grade_from_gpt(essay.title, essay.text)

    cat1 = GradeCategory(
        name="Competência 1",
        score=gpt_grade.categories[0].score,
        description=gpt_grade.categories[0].description,
    )
    cat2 = GradeCategory(
        name="Competência 2",
        score=gpt_grade.categories[1].score,
        description=gpt_grade.categories[1].description,
    )
    cat3 = GradeCategory(
        name="Competência 3",
        score=gpt_grade.categories[2].score,
        description=gpt_grade.categories[2].description,
    )
    cat4 = GradeCategory(
        name="Competência 4",
        score=gpt_grade.categories[3].score,
        description=gpt_grade.categories[3].description,
    )
    cat5 = GradeCategory(
        name="Competência 5",
        score=gpt_grade.categories[4].score,
        description=gpt_grade.categories[4].description,
    )

    return Grade(categories=[cat1, cat2, cat3, cat4, cat5], score=gpt_grade.score)
