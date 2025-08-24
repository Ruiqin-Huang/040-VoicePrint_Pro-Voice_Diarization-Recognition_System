from fastapi import APIRouter

from app.api.v1.endpoints import speech_recognition, speech_segmentation, entity_extraction, event_argument_extraction, audio_diarization_cluster, language_detection, image_ocr, translation, audio_identification_registration, audio_registration, diarization_comparison

api_router = APIRouter()

# 添加各个子路由
api_router.include_router(speech_recognition.router, tags=["语音识别"])
api_router.include_router(speech_segmentation.router, tags=["语音分割"])
api_router.include_router(entity_extraction.router, tags=["实体抽取"])
api_router.include_router(event_argument_extraction.router, tags=["事件论元抽取"])
api_router.include_router(audio_diarization_cluster.router, tags=["说话人分割与聚类"])
api_router.include_router(language_detection.router, tags=["语种检测"])
api_router.include_router(image_ocr.router, tags=["图像OCR"])
api_router.include_router(translation.router, tags=["翻译"])
api_router.include_router(audio_identification_registration.router, tags=["说话人识别与注册"])
api_router.include_router(audio_registration.router, tags=["说话人声纹注册"])
api_router.include_router(diarization_comparison.router, tags=["说话人切分与比对"])