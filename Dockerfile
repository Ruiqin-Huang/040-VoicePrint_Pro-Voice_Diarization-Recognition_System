# 使用NVIDIA CUDA基础镜像
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# 设置工作目录
WORKDIR /app

# 避免交互式提示和Python缓存
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# 添加conda到PATH
ENV PATH /opt/conda/bin:$PATH

# 创建并激活conda环境
RUN conda create -n VoicePrint_Pro python=3.8 -y && \
    conda clean --all -y

# 安装SOX (音频处理库)
RUN conda install -n VoicePrint_Pro -c conda-forge sox -y && \
    conda clean --all -y

# 指定conda环境执行命令
SHELL ["conda", "run", "-n", "VoicePrint_Pro", "/bin/bash", "-c"]

# 复制优化的依赖文件
COPY requirements.txt .

# 配置pip使用清华源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 分批安装依赖，增加可靠性
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建数据目录结构
RUN mkdir -p workspace/dataset \
    workspace/vad \
    workspace/emb \
    workspace/result/text \
    workspace/result/voiceprintlib

# 暴露API端口
EXPOSE 8000

# 设置启动命令
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "VoicePrint_Pro", "python", "-m", "app.main"]