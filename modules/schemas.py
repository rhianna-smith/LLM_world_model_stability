from enum import IntEnum
from pydantic import BaseModel, Field

class LLMAnswer(BaseModel):
    final_answer: str = Field(
        description="The exact option text selected by the model.",
    )
    reason: str = Field(
        description="A brief explanation for the selected option.",
    )


class EventLikelihoodRating(IntEnum):
    impossible = 1
    technically_possible = 2
    plausible = 3
    likely = 4
    very_likely = 5


class EventLikelihoodAnnotation(BaseModel):
    rating: EventLikelihoodRating = Field(
        description="Ordinal likelihood rating for the causal relationship.",
    )
    reason: str = Field(
        description="A brief explanation for the likelihood rating.",
    )
