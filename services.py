import os

from dotenv import load_dotenv
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field

load_dotenv()


class GPTGradeCategory(BaseModel):
    score: int = Field(..., gt=0, le=200, description="A nota do aluno na competência")
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
    score: int = Field(
        ..., gt=0, le=1000, description="A nota final do aluno na redação"
    )


def get_grade_from_gpt(title: str, text: str) -> GPTGrade:
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é uma máquina especialista em corrigir redações do Exame Nacional do Ensino Médio Brasileiro (ENEM) seguindo as normas do Ministério da Educação (MEC). Analise a seguinte redação e entregue a nota em cada uma das 5 competências, especificando claramente os motivos pelos quais a pontuação foi reduzida, caso seja.",
            ),
            ("human", "Tema: {title}\n\n{text}"),
        ]
    )

    chain = create_structured_output_chain(GPTGrade, llm, prompt, verbose=True)
    return chain.run(title=title, text=text)
