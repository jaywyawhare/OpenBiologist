from pydantic import BaseModel

class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None 