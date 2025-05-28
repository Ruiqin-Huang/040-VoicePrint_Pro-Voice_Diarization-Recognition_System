from fastapi import FastAPI
from pathlib import Path
import os
import json
import subprocess

from app.api.v1.router import api_router
from app.config.settings import settings
from app.config.path_mapper import PathMapper

def get_docker_mounts() -> dict:
    """自动获取当前容器的挂载点映射"""
    try:
        cmd = "docker inspect --format='{{json .Mounts}}' $(hostname)"
        result = subprocess.check_output(cmd, shell=True).decode()
        return {m['Destination']: m['Source'] for m in json.loads(result)}
    except Exception as e:
        if settings.DEBUG:
            print(f"无法获取Docker挂载点: {str(e)}")
        return {}

def auto_detect_mappings() -> tuple:
    """自动识别输入输出路径映射"""
    mounts = get_docker_mounts()
    
    # 从settings.py获取容器内路径
    container_input = str(Path(settings.INPUT_DIR).resolve())
    container_output = str(Path(settings.SEGMENTATION_OUTPUT_DIR).parent.resolve())
    
    # 尝试自动映射
    host_input = mounts.get(container_input, None)
    host_output = mounts.get(container_output, None)
    
    return host_input, host_output

app = FastAPI(
    title="VoicePrintPro_API",
    description="提供语音分割和语音识别功能的API服务",
    version="1.0.0",
)

# 自动初始化PathMapper
host_input, host_output = auto_detect_mappings()
if not host_input or not host_output:
    raise RuntimeError("无法自动检测路径映射，请确保已正确挂载卷")

app.state.path_mapper = PathMapper(
    host_input_dir=host_input,
    host_output_dir=host_output
)

def get_path_mapper():
    return app.state.path_mapper

# 注册路由
app.include_router(api_router)

@app.get("/health")
async def health_check():
    mapper = app.state.path_mapper
    return {
        "status": "ok",
        "config": {
            "auto_detected": {
                "host_input": mapper.host_input_dir,
                "host_output": mapper.host_output_dir,
                "container_input": mapper.container_input_dir,
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
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)