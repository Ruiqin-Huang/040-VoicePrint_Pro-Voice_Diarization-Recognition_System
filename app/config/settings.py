import os
from pydantic import BaseSettings
# from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 应用基本配置
    APP_NAME: str = "语音处理服务"
    DEBUG: bool = True
    PORT: int = 8000
    
    # 模型路径配置
    MODEL_DIR: str = "./pretrained_models"
    WHISPER_CACHE_DIR: str = "./pretrained_models/whisper"
    WHISPER_MODEL_SIZE: str = "medium"
    DIARIZATION_MODEL_PATH: str = "./pretrained_models/iic/speech_campplus_speaker-diarization_common"
    DIARIZATION_MODEL_REVISION: str = "v1.0.0"
    
    # 输入目录配置
    INPUT_DIR: str = "./data/input"

    # 输出目录配置
    OUTPUT_DIR: str = "./data/output"
    SEGMENTATION_OUTPUT_DIR: str = "speech_segmentation"
    RECOGNITION_OUTPUT_DIR: str = "speech_recognition"
    
    # 默认语言设置
    DEFAULT_LANGUAGE: str = "zh"
    
    # GPU设置
    USE_GPU: bool = True
    GPU_ID: int = 0
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()