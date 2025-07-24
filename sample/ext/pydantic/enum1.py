import json
from enum import Enum
from pydantic import BaseModel, ConfigDict


class Label(Enum):
    NONE = -1
    CLEAR = 0
    BLUR = 1
    BLACK = 2


class Response(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    label: Label = Label.NONE
    md5: str = None
    size: int = None

    model_config = ConfigDict(
        json_encoders={Enum: lambda e: e.name.lower()}
    )


r = Response(label=Label.CLEAR)
print(r.model_dump(
    exclude_none=True,
))
print(r.model_dump_json(
    exclude_none=True,
))


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name.lower()
        elif isinstance(obj, BaseModel):
            return obj.model_dump_json(exclude_none=True)
        return super().default(obj)


j = json.dumps(dict(
    label=Label.CLEAR,
    md5='1234567890abcdef',
    size=1024,
), cls=JSONEncoder)
print(j)
