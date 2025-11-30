import os
# from pydantic import BaseSettings
from pydantic_settings import BaseSettings
from typing import List, Literal, Optional

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
    GPU_ID: int = 2

    OCR_GPU_ID: List[int] = [0, 1, 3]
    
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
    
    # --- LLM 配置 ---
    # llm_mode: 'local_hf' 使用本地HuggingFace模型, 'api' 使用Ollama API
    llm_mode: Literal['local_hf', 'ollama_api'] = 'ollama_api'
    
    # local_hf 模式配置
    llm_hf_path: Optional[str] = "./pretrained_models/qwen1.5-7b-chat-hf" # 本地huggingface格式模型目录路径
    llm_device: str = "auto" # 'auto', 'cpu', 'cuda', 'cuda:0' 等
    
    # api 模式配置 (Ollama)
    llm_api_endpoint: Optional[str] = "http://localhost:11434/api/generate" # Ollama API 服务地址
    llm_model_name: Optional[str] = "deepseek-r1:7b" # Ollama 中部署的模型名称
    
    # 实体抽取配置
    ENTITY_EXTRACTION_MAX_TEXT_LENGTH: int = 10000  # 最大文本长度（字符数）
    ENTITY_EXTRACTION_CHUNK_SIZE: int = 2000  # 文本分块大小（字符数）
    ENTITY_EXTRACTION_CHUNK_OVERLAP: int = 200  # 分块重叠大小（字符数）
    ENTITY_EXTRACTION_MAX_CONCURRENT_TASKS: int = 4  # 最大并发任务数
    ENTITY_EXTRACTION_TIMEOUT: int = 60  # LLM调用超时时间（秒）
    ENTITY_EXTRACTION_MAX_RETRIES: int = 2  # 最大重试次数

    MILVUS_HOST: str = "10.108.17.241"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION: str = "voiceprint_db"

    PADDLEOCR_PYTHON_EXEC: str = "~/.conda/envs/paddleocr/bin/python"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()