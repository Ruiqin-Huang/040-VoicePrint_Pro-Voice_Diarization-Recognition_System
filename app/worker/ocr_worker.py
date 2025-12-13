import json
import os
import subprocess
import asyncio
from typing import Any, Dict, List, Tuple

from app.config.settings import settings

class OCRWorker:
    def __init__(self, gpu_id, conda_env="paddleocr", ocr_module="app.services.image_ocr"):
        self.gpu_id = gpu_id
        self.conda_env = conda_env
        self.ocr_module = ocr_module
        self.process = None
        self.python_exec = os.path.expanduser(settings.PADDLEOCR_PYTHON_EXEC)
        
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.env["FLAGS_allocator_strategy"] = "naive_best_fit"
        
        self.loop = None
        self.lock = asyncio.Lock()

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
            self.loop = asyncio.get_running_loop()
            print("OCR Worker response:", self.process.stdout.readline())
            # print("stderr:", self.process.stderr.readline())
            # print("poll:", self.process.poll())

    async def send(self, data):
        async with self.lock:
            return await self._send_impl(data)
    
    async def _send_impl(self, data):
        """发送数据到 OCR 子进程并等待响应"""
        if self.process is None:
            raise RuntimeError("OCR worker is not running")
        if self.process.poll() is not None:
            raise RuntimeError("OCR worker process has exited")

        # 发送请求
        print("Current OCR Files:", json.dumps([item.dict() for item in data]))
        self.process.stdin.write(json.dumps([item.dict() for item in data], ensure_ascii=False) + "\n")
        self.process.stdin.flush()

        # 等待返回
        response_line = await self.loop.run_in_executor(None, self.process.stdout.readline)
        print("OCR Worker response:", response_line)
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

class OCRWorkerPool:
    def __init__(self):
        """
        根据 OCR_GPU_ID 配置自动确定worker数量
        OCR_GPU_ID 是 List[int] 类型，如 [6, 7]
        """
        self.gpu_ids = self._get_gpu_ids()
        self.num_workers = len(self.gpu_ids)
        self.workers: List[OCRWorker] = []
        # self.available_workers = asyncio.Queue()
        # 延迟创建队列，防止绑定到错误的时间循环
        self.available_workers = None
        self._started = False
        
    def _get_gpu_ids(self) -> List[int]:
        """获取OCR_GPU_ID配置"""
        gpu_ids = getattr(settings, 'OCR_GPU_ID', [0])
        if not isinstance(gpu_ids, list):
            # 如果配置不是list，转换为list
            if isinstance(gpu_ids, int):
                gpu_ids = [gpu_ids]
            elif isinstance(gpu_ids, str):
                # 处理字符串格式
                gpu_ids = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',') if gpu_id.strip()]
            else:
                gpu_ids = [0]  # 默认值
                
        # print(f"Using GPU IDs: {gpu_ids}")
        return gpu_ids
        
    def start(self):
        """启动工作进程池"""
        if self._started:
            return
            
        # 在Start中创建队列，确保在正确的时间循环中创建
        # try:
            # loop = asyncio.get_running_loop()
        # except RuntimeError:
            # loop = asyncio.get_event_loop()
            
        # self.available_workers = asyncio.Queue()
        print(f"Starting {self.num_workers} OCR workers on GPUs: {self.gpu_ids}...")
        
        
        for gpu_id in self.gpu_ids:
            worker = OCRWorker(gpu_id=gpu_id)
            worker.start()
            self.workers.append(worker)
            # self.available_workers.put_nowait(worker)
            
        self._started = True
        print(f"OCR worker pool started with {self.num_workers} workers")
    
    async def send(self, data: Any) -> Tuple[List, List]:
        """
        发送单条请求到可用的worker
        
        :param data: 单条请求数据
        :return: (processed_files, invalid_files)
        """
        if not self._started:
            raise RuntimeError("OCR worker pool is not started")
            
        # 延迟到中创建队列
        if self.available_workers is None:
            self.available_workers = asyncio.Queue()
            for worker in self.workers:
                self.available_workers.put_nowait(worker)
            
        # 排队等待可用worker（无限等待）
        worker = await self.available_workers.get()
        
        try:
            # 发送单条请求到worker
            result = await worker.send(data)
            return result
        except Exception as e:
            # 如果worker出错，重新创建worker
            print(f"Worker on GPU {worker.gpu_id} error: {e}, restarting...")
            await self._restart_worker(worker)
            raise
        finally:
            # 将worker放回可用队列
            if worker.process and worker.process.poll() is None:
                self.available_workers.put_nowait(worker)
    
    async def send_concurrent(self, requests: List, timeout_per_request: float = 60.0) -> List[Tuple[List, List]]:
        """
        并发发送多个单条请求（每个请求独立处理）
        
        :param requests: 单条请求列表
        :param timeout_per_request: 每个请求的超时时间
        :return: 结果列表
        """
        tasks = []
        for request in requests:
            task = self.send(request, timeout_per_request)
            tasks.append(task)
        
        # 并发执行，但最大并发数受worker数量限制
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Request {i} failed: {result}")
                processed_results.append(([], [f"Request failed: {str(result)}"]))
            else:
                processed_results.append(result)
                
        return processed_results
    
    async def _restart_worker(self, worker: OCRWorker):
        """重启出错的worker"""
        try:
            worker.stop()
            worker.start()
            print(f"Worker on GPU {worker.gpu_id} restarted successfully")
        except Exception as e:
            print(f"Failed to restart worker on GPU {worker.gpu_id}: {e}")
            # 如果重启失败，创建新的worker替换
            new_worker = OCRWorker(gpu_id=worker.gpu_id)
            new_worker.start()
            self.workers.remove(worker)
            self.workers.append(new_worker)
            # self.available_workers.put_nowait(new_worker)
            if self.available_workers is not None:
                self.available_workers.put_nowait(new_worker)
    
    def get_pool_status(self) -> Dict:
        """获取进程池状态"""
        if not self._started:
            return {"status": "not_started"}
            
        available_count = self.available_workers.qsize() if self.available_workers is not None else len(self.workers)
        total_count = len(self.workers)
        busy_count = total_count - available_count
        
        # 检查worker健康状态
        worker_status = []
        for worker in self.workers:
            is_healthy = worker.process and worker.process.poll() is None
            worker_status.append({
                "gpu_id": worker.gpu_id,
                "healthy": is_healthy,
                "process_alive": is_healthy
            })
        
        healthy_workers = sum(1 for w in worker_status if w["healthy"])
        
        return {
            "status": "running",
            "gpu_ids": self.gpu_ids,
            "total_workers": total_count,
            "available_workers": available_count,
            "busy_workers": busy_count,
            "healthy_workers": healthy_workers,
            "worker_details": worker_status
        }
    
    def stop(self):
        """停止所有worker"""
        if not self._started:
            return
            
        print("Stopping OCR worker pool...")
        
        # 清空队列
        # while not self.available_workers.empty():
        #     try:
        #         self.available_workers.get_nowait()
        #     except:
        #         break
        
        if self.available_workers is not None:
            while not self.available_workers.empty():
                try:
                    self.available_workers.get_nowait()
                except:
                    break
                
        # 停止所有worker
        for worker in self.workers:
            worker.stop()
            
        self.workers.clear()
        self.available_workers = None
        self._started = False
        print("OCR worker pool stopped")

# 创建全局进程池实例 - 自动根据OCR_GPU_ID配置
ocr_worker_pool = OCRWorkerPool()
