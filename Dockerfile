# 阶段1：构建环境
FROM python:3.8-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 阶段2：运行环境
FROM python:3.8-slim
WORKDIR /app

# 复制已安装的依赖
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV WHISPER_CACHE_DIR=/app/pretrained_models

# 启动网关
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]