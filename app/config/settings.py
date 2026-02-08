"""
应用配置管理模块

使用Pydantic BaseSettings管理语音处理系统的所有配置项。
支持环境变量覆盖和.env文件配置。

主要功能：
- 应用基本配置（名称、调试模式、端口）
- 预训练模型路径配置
- 输入输出目录配置
- GPU资源分配配置
- 数据库连接配置
- LLM模型配置
- 实体抽取参数配置

配置来源优先级：
1. 环境变量
2. .env文件
3. 默认值

依赖：
- pydantic_settings: 配置管理
- typing: 类型注解
"""

import os
# from pydantic import BaseSettings  # 旧版本pydantic
from pydantic_settings import BaseSettings
from typing import List, Literal, Optional

class Settings(BaseSettings):
    """
    应用配置类

    继承自Pydantic BaseSettings，自动读取环境变量和.env文件。
    包含语音处理系统的所有配置参数。

    Attributes:
        应用配置: APP_NAME, DEBUG, PORT
        模型配置: 各种预训练模型的路径和参数
        目录配置: 输入输出目录设置
        GPU配置: GPU使用和分配设置
        数据库配置: Milvus向量数据库连接
        LLM配置: 大语言模型相关设置
        实体抽取配置: 文本处理参数
    """

    # === 应用基本配置 ===
    APP_NAME: str = "语音处理服务"  # 应用名称，用于API文档和日志
    DEBUG: bool = True  # 调试模式开关，影响日志级别和错误处理
    PORT: int = 8765  # FastAPI服务监听端口

    # === 模型路径配置 ===
    MODEL_DIR: str = "./pretrained_models"  # 预训练模型根目录
    WHISPER_CACHE_DIR: str = "./pretrained_models/whisper"  # Whisper语音识别模型缓存目录
    WHISPER_MODEL_SIZE: str = "turbo"  # Whisper模型大小：tiny/base/small/medium/large/turbo
    DIARIZATION_MODEL_PATH: str = "./pretrained_models/iic/speech_campplus_speaker-diarization_common"  # 说话人分离模型路径
    DIARIZATION_MODEL_REVISION: str = "v1.0.0"  # 说话人分离模型版本
    VAD_MODEL_PATH: str = "./pretrained_models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"  # 语音活动检测模型路径
    VAD_MODEL_REVISION: str = "v2.0.4"  # 语音活动检测模型版本
    SPEAKER_EMBEDDING_MODEL_PATH: str = "./pretrained_models/iic/speech_campplus_sv_zh_en_16k-common_advanced"  # 说话人嵌入模型路径
    SPEAKER_EMBEDDING_MODEL_FILE: str = "campplus_cn_en_common.pt"  # 说话人嵌入模型文件名
    ENTITY_EXTRACTION_MODEL_PATH: str = "./pretrained_models/iic/nlp_seqgpt-560m"  # 实体抽取模型路径
    FASTTEXT_CACHE_DIR: str = "./pretrained_models/fasttext/lid.176.bin"  # FastText语言检测模型路径
    PADDLEOCR_CACHE_DIR: str = "./pretrained_models/paddleocr/"  # PaddleOCR模型缓存目录
    OCR_DETECTION_CACHE_DIR: str = "PP-OCRv5_server_det"  # OCR检测模型目录名
    OCR_RECOGNITION_CACHE_DIR: str = "PP-OCRv5_server_rec"  # OCR识别模型目录名
    TRANSLATION_M2M100_CACHE_DIR: str = "./pretrained_models/m2m100_1.2B"  # M2M100翻译模型缓存目录
    TRANSLATION_SMALL100_CACHE_DIR: str = "./pretrained_models/small100"  # Small100翻译模型缓存目录
    TRANSLATION_HY_MT15_CACHE_DIR: str = "./pretrained_models/HY-MT1___5-1___8B"  # Hunyuan MT1.5翻译模型缓存目录

    # === 混元翻译微服务配置 ===
    HY_TRANSLATION_PORT: int = 8766  # 混元翻译服务端口
    HY_TRANSLATION_LOCAL_PORT: int = 8901  # 混元翻译模型本地端口
    HY_TRANSLATION_SERVICE_URL: str = f'http://localhost:{HY_TRANSLATION_LOCAL_PORT}/api/translation'  # 混元翻译服务URL

    # === 输入输出目录配置 ===
    INPUT_DIR: str = "./data/input"  # 输入根目录

    # 输出目录配置
    OUTPUT_DIR: str = "./data/output"  # 输出根目录
    SEGMENTATION_OUTPUT_DIR: str = "speech_segmentation"  # 语音切分输出子目录
    RECOGNITION_OUTPUT_DIR: str = "speech_recognition"  # 语音识别输出子目录
    DIARIZATION_CLUSTER_OUTPUT_DIR: str = "diarization_cluster"  # 说话人聚类输出子目录

    # === 默认语言设置 ===
    DEFAULT_LANGUAGE: str = "zh"  # 默认处理语言代码
    # 支持的语言列表映射
    LANG_DICT: dict = {
        'zh': "中文",
        'en': "英文",
        'ru': "俄文",
        'ja': "日文",
        'mn': "蒙文"
    }

    # === GPU设置 ===
    USE_GPU: bool = True  # 是否使用GPU加速
    GPU_ID: int = 0  # 默认GPU设备ID

    OCR_GPU_ID: List[int] = [1, 2]  # OCR服务可用的GPU设备ID列表

    # === Milvus向量数据库配置 ===
    MILVUS_HOST: str = "10.108.17.241"  # Milvus服务器主机地址
    MILVUS_PORT: str = "19530"  # Milvus服务器端口
    MILVUS_COLLECTION: str = "voiceprint_db"  # 默认集合名称，用于声纹数据存储

    # === 说话人聚类配置 ===
    # 使用谱聚类的YAML配置内容，定义特征提取和嵌入模型参数
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

    # === LLM（大语言模型）配置 ===
    # llm_mode: 'local_hf' 使用本地HuggingFace模型, 'ollama_api' 使用Ollama API, 'vllm' 使用vLLM API
    llm_mode: Literal['local_hf', 'ollama_api', 'vllm'] = 'ollama_api'

    # 本地HuggingFace模型模式配置
    llm_hf_path: Optional[str] = "./pretrained_models/qwen1.5-7b-chat-hf"  # 本地HF模型目录路径
    llm_device: str = "auto"  # 模型运行设备：'auto', 'cpu', 'cuda', 'cuda:0' 等

    # Ollama API模式配置
    llm_api_endpoint: Optional[str] = "http://localhost:11434/api/generate"  # Ollama API服务地址
    llm_model_name: Optional[str] = "deepseek-r1:7b"  # Ollama中部署的模型名称

    # vLLM API模式配置
    vllm_api_endpoint: Optional[str] = "http://localhost:8000/v1/chat/completions"  # vLLM API服务地址
    vllm_model_name: Optional[str] = "qwen1.5-7b-chat"  # vLLM中部署的模型名称

    # === 实体抽取配置 ===
    ENTITY_EXTRACTION_MAX_TEXT_LENGTH: int = 10000  # 最大文本长度限制（字符数）
    ENTITY_EXTRACTION_CHUNK_SIZE: int = 2000  # 文本分块大小，用于长文本处理
    ENTITY_EXTRACTION_CHUNK_OVERLAP: int = 200  # 分块重叠大小，确保实体不被截断
    ENTITY_EXTRACTION_MAX_CONCURRENT_TASKS: int = 4  # 最大并发任务数，控制资源使用
    ENTITY_EXTRACTION_TIMEOUT: int = 60  # LLM调用超时时间（秒）
    ENTITY_EXTRACTION_MAX_RETRIES: int = 2  # 最大重试次数，处理临时失败

    # === PaddleOCR环境配置 ===
    PADDLEOCR_PYTHON_EXEC: str = "/opt/conda/envs/paddleocr/bin/python"  # PaddleOCR专用Python环境路径

    class Config:
        """
        Pydantic配置类

        定义BaseSettings的行为配置，包括环境变量处理规则。
        """
        case_sensitive = True  # 环境变量名称大小写敏感
        env_file = ".env"  # 环境变量文件路径

# 创建全局配置实例，所有模块都可以导入使用
settings = Settings()