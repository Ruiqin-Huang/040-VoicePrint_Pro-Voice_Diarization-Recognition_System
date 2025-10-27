# 自动检测conda路径
if [ -z "$CONDA_EXE" ]; then
    echo "未检测到conda命令，请确认已安装Miniconda/Anaconda"
    exit 1
fi

# 推导出profile脚本路径
CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n paddleocr python=3.10 -y
conda activate paddleocr

python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl