from fastapi import FastAPI
import os

from app.api.v1.router import api_router
from app.config.settings import settings
from app.core.error_codes import ResponseCode

app = FastAPI(
    title="VoicePrintPro_API",
    description="提供语音分割和语音识别功能的API服务",
    version="1.0.0",
)

# 注册API路由
app.include_router(api_router)

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)