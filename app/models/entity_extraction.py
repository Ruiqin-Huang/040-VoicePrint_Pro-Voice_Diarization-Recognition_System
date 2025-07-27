from pydantic import BaseModel
from typing import List, Optional

class EntityExtractionRequest(BaseModel):
    text: str
    entity_types: Optional[List[str]] = None

class EntityExtractionResponse(BaseModel):
    result: str