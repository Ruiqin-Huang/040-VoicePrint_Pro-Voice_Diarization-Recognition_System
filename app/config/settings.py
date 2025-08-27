import os
# from pydantic import BaseSettings
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 应用基本配置
    APP_NAME: str = "语音处理服务"
    DEBUG: bool = True
    PORT: int = 8765
    
    # 模型路径配置
    MODEL_DIR: str = "./pretrained_models"
    WHISPER_CACHE_DIR: str = "./pretrained_models/whisper"
    WHISPER_MODEL_SIZE: str = "medium"
    DIARIZATION_MODEL_PATH: str = "./pretrained_models/iic/speech_campplus_speaker-diarization_common"
    DIARIZATION_MODEL_REVISION: str = "v1.0.0"
    VAD_MODEL_PATH: str = "./pretrained_models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    VAD_MODEL_REVISION: str = "v2.0.4"
    SPEAKER_EMBEDDING_MODEL_PATH: str = "./pretrained_models/iic/speech_campplus_sv_zh_en_16k-common_advanced"
    SPEAKER_EMBEDDING_MODEL_FILE: str = "campplus_cn_en_common.pt"
    ENTITY_EXTRACTION_MODEL_PATH: str = "./pretrained_models/iic/nlp_seqgpt-560m"
    FASTTEXT_CACHE_DIR: str = "./pretrained_models/fasttext/lid.176.bin"
    PADDLEOCR_CACHE_DIR: str = "./pretrained_models/paddleocr/"
    OCR_DETECTION_CACHE_DIR: str = "PP-OCRv5_server_det"
    OCR_RECOGNITION_CACHE_DIR: str = "PP-OCRv5_server_rec"
    TRANSLATION_M2M100_CACHE_DIR: str = "./pretrained_models/m2m100"
    TRANSLATION_SMALL100_CACHE_DIR: str = "./pretrained_models/small100"
    
    # 输入目录配置
    INPUT_DIR: str = "./data/input"

    # 输出目录配置
    OUTPUT_DIR: str = "./data/output"
    SEGMENTATION_OUTPUT_DIR: str = "speech_segmentation"
    RECOGNITION_OUTPUT_DIR: str = "speech_recognition"
    DIARIZATION_CLUSTER_OUTPUT_DIR: str = "diarization_cluster"
    
    # 默认语言设置
    DEFAULT_LANGUAGE: str = "zh"
    # 语言列表
    LANG_DICT : dict = {
        'zh': "中文",
        'en': "英文",
        'ru': "俄文",
        'ja': "日文",
        'mn': "蒙文"
    }
    
    # GPU设置
    USE_GPU: bool = True
    GPU_ID: int = 0
    
    # Milvus配置
    MILVUS_COLLECTION = "voiceprint_db" # 默认插入的Milvus集合名称
    
    # 说话人聚类配置（使用谱聚类）
    DIAR_CLUSTER_CONFIG_CONTENT: str = """
fbank_dim: 80
embedding_size: 192

feature_extractor:
  obj: speakerlab.process.processor.FBank
  args:
    n_mels: <fbank_dim>
    sample_rate: 16000
    mean_nor: True

embedding_model:
  obj: speakerlab.models.campplus.DTDNN.CAMPPlus
  args:
    feat_dim: <fbank_dim>
    embedding_size: <embedding_size>
"""
    
    MILVUS_HOST: str = "10.108.17.241"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION: str = "voiceprint_db"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()