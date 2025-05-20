
from pydantic import BaseModel
from typing import Any

class JobModel(BaseModel):
    submission_id: str
    task_id: str
    flow: str
    buid: str
    callback: str
    resource: str
    request: str
    response: str


job = JobModel()
