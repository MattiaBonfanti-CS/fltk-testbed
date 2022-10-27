from pydantic import BaseModel
from typing import List


class Results(BaseModel):
    """
    Experiments results class.
    """
    start_time: str
    end_time: str
    loss: float
    accuracy: List[float]


class ResultsQueue(BaseModel):
    """
    Experiments queue setup class.
    """
    start_time: List[str]
    end_time: List[str]
    epochs: List[int]
    learning_rate: float
