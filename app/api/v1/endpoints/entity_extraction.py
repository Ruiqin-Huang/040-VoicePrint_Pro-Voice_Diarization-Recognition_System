from fastapi import APIRouter
from app.models.entity_extraction import EntityExtractionRequest, EntityExtractionResponse
from app.services.entity_extraction import extract_entities

router = APIRouter()

@router.post("/entity_extraction", response_model=EntityExtractionResponse, tags=["实体抽取"])
async def entity_extraction_api(req: EntityExtractionRequest):
    result = extract_entities(req.text, req.entity_types)
    return EntityExtractionResponse(result=result)