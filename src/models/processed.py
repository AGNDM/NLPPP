from pydantic import BaseModel

class ProcessedExample(BaseModel):
    question: str
    context: str
    answer: str
    source: str

    class Config:
        extra = "ignore"
