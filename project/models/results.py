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
