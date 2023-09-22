import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field

load_dotenv()


class GPTGradeCategory(BaseModel):
    score: int = Field(..., description="A nota do aluno na competência")
    description: str = Field(
        ...,
        description="A descrição da nota do aluno na competência, incluindo o que ele fez de certo e errado. É necessário que o motivo pela subtração de pontos seja explicado.",
    )


class GPTGrade(BaseModel):
    categories: list[GPTGradeCategory] = Field(
        ...,
        description="A lista de competências e suas respectivas notas",
        min_items=5,
        max_items=5,
    )
    score: int = Field(..., description="A nota final do aluno na redação")


def get_grade_from_gpt(title: str, text: str) -> GPTGrade:
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        openai_api_key=os.environ["OPENAI_KEY"],
    )

    PROMPT_FILE = Path(__file__).parent / "prompt.txt"

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            "O arquivo prompt.txt não foi encontrado. Por favor, crie-o e tente novamente."
        )

    with open(PROMPT_FILE, "r") as f:
        prompt_text = f.read()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            ("human", "{text}"),
        ]
    )

    chain = create_structured_output_chain(GPTGrade, llm, prompt, verbose=True)
    return chain.run(title=title, text=text)
