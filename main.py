from fastapi import FastAPI, APIRouter
from api.api_speech_recognition import router as asr_router

app = FastAPI(title="API Gateway")

@app.get('/')
async def homepage() -> dict:
	return {"message": "Homepage"}

# 挂载子路由
api_router = APIRouter(prefix="/api")
api_router.include_router(asr_router)
app.include_router(api_router)

# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "healthy"}