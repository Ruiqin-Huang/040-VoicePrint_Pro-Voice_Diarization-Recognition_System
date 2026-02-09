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
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.services.hy_translation import load_model, clear_models
from app.api.v1.endpoints import hy_translation

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器
    - 启动时：设置CUDA可见设备，加载混元模型
    - 关闭时：清理模型资源，释放GPU显存
    """
    # 启动逻辑
    print(f"Starting Hunyuan MT1.5 Translation Service...")
    
    # 加载模型
    try:
        load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise
    
    yield  # 应用运行期
    
    # 关闭逻辑
    print("Shutting down Hunyuan MT1.5 Translation Service...")
    clear_models()
    print("✓ Resources cleaned up")

# 创建FastAPI应用
app = FastAPI(
    title="Hunyuan MT1.5 Translation Service",
    version="1.0.0",
    description="混元MT1.5机器翻译微服务",
    lifespan=lifespan
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

@app.get("/health")
async def health():
    """
    健康检查端点
    
    检查服务是否正常运行，模型是否已加载
    
    Returns:
        dict: 包含状态、模型加载情况、设备信息
    """
    return {
        "status": "healthy",
        "service": "Hunyuan MT1.5 Translation Service",
        "version": "1.0.0",
        "model": "Tencent-Hunyuan/HY-MT1.5-1.8B",
        "device": f"cuda:{settings.GPU_ID}"
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
        reload=settings.DEBUG,
        # workers=1  # 混元模型需要单worker防止显存冲突
    )

if __name__ == "__main__":
    main()
