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
        openai_api_key=os.environ["OPENAI_KEY"],
    )

    CATEGORY_DETAILS = """
Para isso, leve em consideração as seguintes informações:
Competência 1: ter domínio da escrita formal da Língua Portuguesa
Essa competência diz respeito às regras gramaticais da norma culta da Língua Portuguesa, desde pontuação e acentuação até regência e concordância verbal e nominal:
0 ponto — o candidato desconhece a modalidade escrita formal da língua;
40 pontos — o domínio da modalidade escrita formal é precário, com frequentes e diversificados desvios gramaticais;
80 pontos — o domínio da modalidade escrita formal é insuficiente, com muitos desvios;
120 pontos — o domínio da modalidade escrita formal é mediano, com alguns desvios;
160 pontos — o domínio da modalidade escrita formal é bom, com poucos desvios;
200 pontos — o domínio da modalidade escrita formal é excelente, sendo que os poucos desvios observados são aceitos como excepcionalidades.

Competência 2: compreender o tema e não fugir do que é proposto
Aqui, é avaliado se o candidato conseguiu entender a proposta a partir da leitura dos textos de apoio e se produziu seu texto de acordo com as características de um texto dissertativo-argumentativo:
0 ponto — o candidato fugiu do tema e não atendeu à estrutura de texto dissertativo-argumentativo;
40 pontos — o candidato foge um pouco do assunto e demonstra domínio precário do gênero dissertativo-argumentativo;
80 pontos — o tema é desenvolvido com domínio insuficiente da estrutura, e o candidato recorre à cópia de trechos dos textos de apoio;
120 pontos — o tema é atendido, mas com argumentação previsível e domínio mediano da estrutura;
160 pontos — a argumentação é consistente, e a estrutura bem-atendida;
200 pontos — o candidato tem um excelente domínio do gênero e desenvolve o tema a partir de um repertório sociocultural produtivo.

Competência 3: saber selecionar e organizar informações e argumentos em defesa de um ponto de vista
Na competência 3, é avaliada a coerência na construção do texto, isto é, na forma como o candidato organiza suas ideias e as apresenta para o leitor:
0 ponto — as informações apresentadas não se relacionam e não há defesa de ponto de vista;
40 pontos — os argumentos são incoerentes ou se relacionam pouco ao tema, sem defesa de ponto de vista;
80 pontos — os argumentos se relacionam ao tema e defendem um ponto de vista, mas são apresentados de forma desorganizada ou contraditória;
120 pontos — os argumentos são ligados a um ponto de vista, mas são limitados e pouco organizados;
160 pontos — os argumentos são apresentados de forma organizada e em favor de um ponto de vista;
200 pontos — o candidato tem uma argumentação consistente e organizada e demonstra ter opiniões bem-fundamentadas.

Competência 4: ter conhecimento dos mecanismos linguísticos necessários para a construção da argumentação
Aqui, os corretores avaliam se o candidato sabe empregar mecanismos que garantem a estruturação lógica do texto, como conectivos responsáveis pela coesão na redação:
0 ponto — as informações não são articuladas;
40 pontos — a articulação do texto é precária;
80 pontos — a articulação é insuficiente, com repertório limitado de recursos coesivos;
120 pontos — as partes do texto são articuladas de forma mediana;
160 pontos — a articulação é boa e realizada com um repertório diversificado de recursos coesivos;
200 pontos — as partes do texto são articuladas com bastante domínio.

Competência 5: elaborar uma proposta de intervenção com respeito aos direitos humanos
Por fim, é esperado que o candidato saiba apresentar uma proposta de intervenção ao tema abordado. Ou seja, consiga ter uma ideia que minimamente enfrente aquele problema. Isso indica o preparo para o exercício da cidadania de forma ativa, sempre respeitando os direitos humanos.
0 ponto — não há proposta de intervenção ou ela não se relaciona ao tema;
40 pontos — a proposta de intervenção é vaga;
80 pontos — a proposta é apresentada, mas de forma insuficiente;
120 pontos — a proposta é mediana e se articula razoavelmente ao problema abordado;
160 pontos — há uma boa proposta de intervenção bem articulada ao tema;
200 pontos — a proposta é muito bem elaborada e articulada à discussão presente no texto.

Note ainda que, a fuga ao tema, a não elaboração de uma proposta de intervenção ou a não obediência à estrutura de texto dissertativo-argumentativo são motivos para zerar a redação. Nesse caso, o candidato recebe a nota 0 em todas as competências, mesmo que tenha escrito um texto com qualidade em relação à gramática e à argumentação.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é uma máquina especialista em corrigir redações do Exame Nacional do Ensino Médio Brasileiro (ENEM) seguindo as normas do Ministério da Educação (MEC). Analise a seguinte redação sobre o tema '{title}' e entregue a nota em cada uma das 5 competências, especificando claramente os motivos pelos quais a pontuação foi reduzida, caso seja. Procure ser o mais detalhado possível, sempre realizando críticas construtivas."
                + CATEGORY_DETAILS,
            ),
            ("human", "{text}"),
        ]
    )

    chain = create_structured_output_chain(GPTGrade, llm, prompt, verbose=True)
    return chain.run(title=title, text=text)
