"""
混元MT1.5翻译微服务入口

该模块负责：
- FastAPI应用初始化
- CORS中间件配置
- 应用生命周期事件（启动/关闭）
- Uvicorn服务启动

可通过以下方式启动：
- 直接运行：python -m app.hy_main
- Docker中运行：CMD ["python", "-m", "app.hy_main"]
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.services.hy_translation import load_model, clear_models
from app.api.v1.endpoints import hy_translation

# 创建FastAPI应用
app = FastAPI(
    title="Hunyuan MT1.5 Translation Service",
    version="1.0.0",
    description="混元MT1.5机器翻译微服务"
)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含路由
app.include_router(
    hy_translation.router,
    tags=["混元大模型机器翻译"]
)


@app.on_event("startup")
async def startup_event():
    """
    应用启动时的回调函数
    - 设置CUDA可见设备
    - 加载混元模型
    """
    print(f"Starting Hunyuan MT1.5 Translation Service...")
    
    # 加载模型
    try:
        load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭时的回调函数
    - 清理模型资源
    - 释放GPU显存
    """
    print("Shutting down Hunyuan MT1.5 Translation Service...")
    clear_models()
    print("✓ Resources cleaned up")


@app.get("/health")
async def health():
    """
    健康检查端点
    
    Returns:
        dict: 服务状态信息
    """
    return {
        "status": "healthy",
        "service": "Hunyuan MT1.5 Translation Service",
        "version": "1.0.0"
    }


def main():
    """
    应用入口点
    
    启动Uvicorn服务器：
    - 主机：0.0.0.0
    - 端口：从settings.HY_TRANSLATION_PORT读取（默认8766）
    - Workers：1（防止GPU显存冲突）
    """
    # 设置GPU环境
    gpu_id = getattr(settings, 'GPU_ID', 0)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 获取服务端口
    port = getattr(settings, 'HY_TRANSLATION_PORT', 8766)
    
    print(f"\n{'='*60}")
    print(f"启动 Hunyuan MT1.5 Translation Service")
    # print(f"{'='*60}")
    # print(f"Host: 0.0.0.0")
    # print(f"Port: {port}")
    print(f"GPU ID: {gpu_id}")
    print(f"{'='*60}\n")
    
    # 启动服务
    uvicorn.run(
        "app.hy_main:app",
        host="0.0.0.0",
        port=port,
        # workers=1  # 混元模型需要单worker防止显存冲突
    )

if __name__ == "__main__":
    main()
