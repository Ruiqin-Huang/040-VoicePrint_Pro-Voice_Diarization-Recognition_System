import os
from pydantic import BaseSettings
# from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 应用基本配置
    APP_NAME: str = "语音处理服务"
    DEBUG: bool = True
    PORT: int = 8000
    
    # 模型路径配置
    WHISPER_CACHE_DIR: str = "./pretrained_models/whisper"
    WHISPER_MODEL_SIZE: str = "medium"
    DIARIZATION_MODEL_PATH: str = "./pretrained_models/iic/speech_campplus_speaker-diarization_common"
    DIARIZATION_MODEL_REVISION: str = "v1.0.0"
    
    
    # 输出目录配置
    SEGMENTATION_OUTPUT_DIR: str = "./data/audio_segmentation"
    
    # 默认语言设置
    DEFAULT_LANGUAGE: str = "zh"
    
    # GPU设置
    USE_GPU: bool = True
    GPU_ID: int = 0
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()