from fastapi import APIRouter

from app.api.v1.endpoints import speech_recognition, speech_segmentation, entity_extraction

api_router = APIRouter()

# 添加各个子路由
api_router.include_router(speech_recognition.router, tags=["语音识别"])
api_router.include_router(speech_segmentation.router, tags=["语音分割"])
api_router.include_router(entity_extraction.router, tags=["实体抽取"])