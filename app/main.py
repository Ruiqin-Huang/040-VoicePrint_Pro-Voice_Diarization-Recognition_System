from fastapi import FastAPI
from pathlib import Path
import os
import json
import subprocess

from app.api.v1.router import api_router
from app.config.settings import settings
from app.config.path_mapper import PathMapper
from app.worker.ocr_worker import ocr_worker_pool

from contextlib import asynccontextmanager
from app.llm import llm_client
from app.api.v1.router import api_router

def auto_detect_mappings() -> tuple:
    """自动识别输入输出路径映射"""
    host_input = os.getenv('HOST_INPUT_PATH')
    host_output = os.getenv('HOST_OUTPUT_PATH')
    
    return host_input, host_output

async def lifespan(app: FastAPI):
    """
    应用生命周期管理，在启动时预加载模型。
    """
    print(f"Application starting in '{settings.llm_mode}' mode.")
    if settings.llm_mode == 'local_hf':
        print("Pre-loading local Hugging Face model...")
        llm_client.get_local_hf_pipeline() # 显式调用HF pipeline加载
        print("Model loaded.")

    # 开启 OCR 子进程
    global worker_process
    print("Pre-loading OCR subprocess in paddleocr...")
    ocr_worker_pool.start()

    yield
    print("Shutting down OCR subprocess...")
    ocr_worker_pool.stop()

    print("Application shutting down.")
app = FastAPI(
    title="VoicePrintPro_API",
    description="提供语音分割和语音识别功能的API服务",
    version="1.0.0",
    lifespan=lifespan
)

# # 自动初始化PathMapper
# host_input, host_output = auto_detect_mappings()
# if not host_input or not host_output:
#     raise RuntimeError("无法自动检测路径映射，请确保已正确挂载卷")

# app.state.path_mapper = PathMapper(
#     host_input_dir=host_input,
#     host_output_dir=host_output
# )

# 注册路由
app.include_router(api_router)

@app.get("/health")
async def health_check():
    # mapper = app.state.path_mapper
    return {
        "status": "ok",
        "config": {
            "auto_detected": {
                # "host_input": mapper.host_input_dir,
                # "host_output": mapper.host_output_dir,
                "container_input": settings.INPUT_DIR,
                "container_output_root": str(Path(settings.SEGMENTATION_OUTPUT_DIR).parent.resolve())
            },
            "models": {
                "whisper_cache": settings.WHISPER_CACHE_DIR,
                "diarization_model": settings.DIARIZATION_MODEL_PATH
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)