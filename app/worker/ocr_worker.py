import json
import os
import subprocess
import asyncio

from app.config.settings import settings

class OCRWorker:
    def __init__(self, conda_env="paddleocr", ocr_module="app.services.image_ocr"):
        self.conda_env = conda_env
        self.ocr_module = ocr_module
        self.process = None
        self.python_exec = os.path.expanduser(settings.PADDLEOCR_PYTHON_EXEC)
        
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = str(settings.GPU_ID)

    def start(self):
        if self.process is None:
            self.process = subprocess.Popen(
                [self.python_exec, "-m", self.ocr_module],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self.env
            )
            # 等待子进程启动信号
            print("OCR Worker response:", self.process.stdout.readline())
            # print("stderr:", self.process.stderr.readline())
            # print("poll:", self.process.poll())

    async def send(self, data):
        """发送数据到 OCR 子进程并等待响应"""
        if self.process is None:
            raise RuntimeError("OCR worker is not running")
        if self.process.poll() is not None:
            raise RuntimeError("OCR worker process has exited")

        # 发送请求
        # print(json.dumps([item.dict() for item in data]))
        self.process.stdin.write(json.dumps([item.dict() for item in data], ensure_ascii=False) + "\n")
        self.process.stdin.flush()

        # 等待返回
        loop = asyncio.get_event_loop()
        response_line = await loop.run_in_executor(None, self.process.stdout.readline)
        # print("OCR Worker response:", response_line)
        response = json.loads(response_line)

        # 拆解成两个部分
        processed_files = response.get("processed_files", [])
        invalid_files = response.get("invalid_files", [])

        return processed_files, invalid_files

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("OCR worker stopped")
            self.process = None

# 创建全局单例
ocr_worker = OCRWorker()
