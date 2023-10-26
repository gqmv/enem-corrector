from pydantic import BaseModel


class Essay(BaseModel):
    theme: str
    text: str
