from fastapi import APIRouter

from app.api.v1.endpoints import speech_recognition, speech_segmentation

api_router = APIRouter()

# 添加各个子路由
api_router.include_router(speech_recognition.router, tags=["语音识别"])
api_router.include_router(speech_segmentation.router, tags=["语音分割"])