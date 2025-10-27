conda create -n paddleocr python=3.10 -y

conda init

conda activate paddleocr

python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu118/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl