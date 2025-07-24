
from pydantic import BaseModel, Field
from typing import Optional

class FileRemoteModel(BaseModel):
    url: str
    md5: Optional[str] = Field(default_factory=str)
    size: Optional[int] = 0

print(FileRemoteModel.model_validate({
    'url': 'https://live_455842841_1415061_20250721153035.jpg'
}))

