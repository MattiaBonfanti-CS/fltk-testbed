from datetime import datetime
from pydantic import BaseModel
from typing import List


class Results(BaseModel):
    """
    Experiments results class.
    """
    start_time: datetime
    end_time: datetime
    loss: float
    accuracy: List[float]
