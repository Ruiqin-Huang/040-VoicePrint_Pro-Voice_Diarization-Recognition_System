#!/bin/bash

# 设置镜像站和环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1  # 启用高效下载

# 创建模型保存目录
mkdir -p ./pretrained_models/m2m100
mkdir -p ./pretrained_models/small100

# 下载m2m100模型
echo "正在下载 m2m100-418M 模型..."
huggingface-cli download \
    --resume-download \
    --local-dir ./pretrained_models/m2m100 \
    --local-dir-use-symlinks False \
    facebook/m2m100_418M

# 下载small100模型
echo "正在下载 small100 模型..."
huggingface-cli download \
    --resume-download \
    --local-dir ./pretrained_models/small100 \
    --local-dir-use-symlinks False \
    alirezamsh/small100

echo "所有模型下载完成！"