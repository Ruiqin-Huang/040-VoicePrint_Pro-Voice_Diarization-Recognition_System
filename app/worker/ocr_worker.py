"""
OCR工作进程池模块

该模块实现了基于GPU的OCR工作进程池，用于异步处理图像OCR识别任务。
通过子进程方式运行OCR服务，支持多GPU并发处理，提高系统性能和稳定性。

主要组件：
- OCRWorker: 单个GPU上的OCR工作进程
- OCRWorkerPool: 工作进程池管理器，支持负载均衡和故障恢复

特性：
- 多GPU支持：自动检测和使用配置的GPU
- 异步处理：基于asyncio的非阻塞I/O
- 进程隔离：每个GPU运行独立子进程，避免内存冲突
- 自动重启：检测到进程异常时自动重启工作进程

依赖：
- subprocess: 子进程管理
- asyncio: 异步编程支持
- app.config.settings: 配置管理
"""

import json
import os
import subprocess
import asyncio
from typing import Any, Dict, List, Tuple

from app.config.settings import settings

class OCRWorker:
    """
    OCR工作进程类
    
    管理单个GPU上的OCR子进程，负责启动、通信和停止OCR处理服务。
    每个实例绑定到一个特定的GPU，通过子进程方式运行OCR模块。
    
    Attributes:
        gpu_id: GPU设备ID
        conda_env: Conda环境名称（保留参数）
        ocr_module: OCR模块路径
        process: 子进程对象
        python_exec: Python执行器路径
        env: 环境变量字典
        lock: 异步锁，保护并发访问
    """

    def __init__(self, gpu_id, conda_env="paddleocr", ocr_module="app.services.image_ocr"):
        """
        初始化OCR工作进程
        
        Args:
            gpu_id: 要使用的GPU设备ID
            conda_env: Conda环境名称（当前未使用）
            ocr_module: OCR处理模块的导入路径
        """
        self.gpu_id = gpu_id
        self.conda_env = conda_env
        self.ocr_module = ocr_module
        self.process = None
        # 获取Python执行器路径
        self.python_exec = os.path.expanduser(settings.PADDLEOCR_PYTHON_EXEC)
        
        # 复制环境变量并设置GPU相关配置
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 指定使用的GPU
        self.env["FLAGS_allocator_strategy"] = "naive_best_fit"  # PaddlePaddle内存分配策略
        self.env["OCR_WORKER_GPU_ID"] = str(gpu_id)  # 传递GPU ID给子进程
        
        # 创建异步锁保护并发访问
        self.lock = asyncio.Lock()

    def start(self):
        """
        启动OCR子进程
        
        创建子进程运行OCR模块，设置标准输入输出管道。
        等待子进程发送启动确认信号。
        """
        if self.process is None:
            # 启动子进程
            self.process = subprocess.Popen(
                [self.python_exec, "-m", self.ocr_module],  # 运行OCR模块
                stdin=subprocess.PIPE,   # 标准输入管道
                stdout=subprocess.PIPE,  # 标准输出管道
                # stderr=subprocess.PIPE,  # 标准错误管道（已注释）
                text=True,               # 文本模式
                bufsize=1,               # 行缓冲
                env=self.env             # 自定义环境变量
            )
            # 等待子进程启动信号
            print(f"OCR Worker on GPU {self.gpu_id}:", self.process.stdout.readline())
            # print("stderr:", self.process.stderr.readline())
            # print("poll:", self.process.poll())

    async def send(self, data):
        """
        发送数据到OCR子进程（线程安全）
        
        使用异步锁确保同一时间只有一个请求在处理。
        
        Args:
            data: 要发送的数据（通常是文件请求列表）
            
        Returns:
            Tuple[List, List]: (处理成功的文件列表, 处理失败的文件列表)
        """
        async with self.lock:
            return await self._send_impl(data)
    
    async def _send_impl(self, data):
        """
        发送数据的实际实现
        
        将数据序列化为JSON发送给子进程，然后等待并解析响应。
        
        Args:
            data: 要发送的数据
            
        Returns:
            Tuple[List, List]: (processed_files, invalid_files)
            
        Raises:
            RuntimeError: 当进程未运行或已退出时抛出
        """
        # 检查进程状态
        if self.process is None:
            raise RuntimeError("OCR worker on GPU {self.gpu_id} is not running")
        if self.process.poll() is not None:
            raise RuntimeError("OCR worker on GPU {self.gpu_id} process has exited")

        # 发送请求数据
        # print("Current OCR Files:", json.dumps([item.dict() for item in data]))
        self.process.stdin.write(json.dumps([item.dict() for item in data], ensure_ascii=False) + "\n")
        self.process.stdin.flush()

        # 等待并读取响应
        loop = asyncio.get_running_loop()
        response_line = await loop.run_in_executor(None, self.process.stdout.readline)
        # print("OCR Worker response:", response_line)
        response = json.loads(response_line)

        # 解析响应数据
        processed_files = response.get("processed_files", [])
        invalid_files = response.get("invalid_files", [])

        return processed_files, invalid_files

    def stop(self):
        """
        停止OCR子进程
        
        终止子进程并等待其完全退出。
        """
        if self.process:
            self.process.terminate()  # 发送终止信号
            self.process.wait()       # 等待进程退出
            print(f"OCR worker on GPU {self.gpu_id} stopped")
            self.process = None

class OCRWorkerPool:
    """
    OCR工作进程池管理器
    
    管理多个OCRWorker实例，实现负载均衡和故障恢复。
    根据配置文件自动创建对应数量的工作进程，每个绑定到一个GPU。
    
    Attributes:
        gpu_ids: 可用的GPU ID列表
        num_workers: 工作进程数量
        workers: OCRWorker实例列表
        available_workers: 可用工作进程队列
        _started: 池是否已启动的标志
    """

    def __init__(self):
        """
        初始化工作进程池
        
        根据配置文件中的OCR_GPU_ID自动确定要创建的工作进程数量。
        支持多种配置格式：列表、单个整数或逗号分隔的字符串。
        """
        self.gpu_ids = self._get_gpu_ids()
        self.num_workers = len(self.gpu_ids)
        self.workers: List[OCRWorker] = []
        # 延迟创建队列，防止绑定到错误的事件循环
        self.available_workers = None
        self._started = False
        
    def _get_gpu_ids(self) -> List[int]:
        """
        获取OCR_GPU_ID配置并解析为GPU ID列表
        
        支持多种配置格式：
        - List[int]: [0, 1, 2]
        - int: 0 (转换为 [0])
        - str: "0,1,2" (转换为 [0, 1, 2])
        
        Returns:
            List[int]: GPU ID列表
        """
        gpu_ids = getattr(settings, 'OCR_GPU_ID', [0])
        if not isinstance(gpu_ids, list):
            # 如果配置不是list，转换为list
            if isinstance(gpu_ids, int):
                gpu_ids = [gpu_ids]
            elif isinstance(gpu_ids, str):
                # 处理字符串格式，如"0,1,2"
                gpu_ids = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',') if gpu_id.strip()]
            else:
                gpu_ids = [0]  # 默认值
                
        # print(f"Using GPU IDs: {gpu_ids}")
        return gpu_ids
        
    def start(self):
        """
        启动工作进程池
        
        为每个配置的GPU创建并启动OCRWorker实例。
        确保在正确的事件循环中创建异步队列。
        """
        if self._started:
            return
            
        # 在start方法中创建队列，确保在正确的事件循环中创建
        print(f"Starting {self.num_workers} OCR workers on GPUs: {self.gpu_ids}...")
        
        # 创建并启动所有worker
        for gpu_id in self.gpu_ids:
            worker = OCRWorker(gpu_id=gpu_id)
            worker.start()
            self.workers.append(worker)
            
        self._started = True
        print(f"OCR worker pool started with {self.num_workers} workers")
    
    async def send(self, data: Any) -> Tuple[List, List]:
        """
        发送单条请求到可用的worker
        
        从可用worker队列中获取一个worker，发送请求并等待结果。
        处理worker异常情况，支持自动重启故障worker。
        
        Args:
            data: 单条请求数据（文件请求列表）
            
        Returns:
            Tuple[List, List]: (处理成功的文件列表, 处理失败的文件列表)
            
        Raises:
            RuntimeError: 当进程池未启动时抛出
        """
        if not self._started:
            raise RuntimeError("OCR worker pool is not started")
            
        # 延迟创建可用worker队列，确保在正确的异步上下文中
        if self.available_workers is None:
            self.available_workers = asyncio.Queue()
            for worker in self.workers:
                self.available_workers.put_nowait(worker)
            
        # 从队列获取可用worker（如果没有可用worker会等待）
        worker = await self.available_workers.get()
        
        try:
            # 发送请求到worker并获取结果
            result = await worker.send(data)
            return result
        except Exception as e:
            # 如果worker出错，记录错误并尝试重启
            print(f"Worker on GPU {worker.gpu_id} error: {e}, restarting...")
            await self._restart_worker(worker)
            raise
        finally:
            # 无论成功还是失败，都将worker放回队列（如果进程仍然存活）
            if worker.process and worker.process.poll() is None:
                self.available_workers.put_nowait(worker)
    
    async def send_concurrent(self, requests: List, timeout_per_request: float = 60.0) -> List[Tuple[List, List]]:
        """
        并发发送多个单条请求
        
        为每个请求创建独立的任务并发执行，最大并发数受可用worker数量限制。
        处理异常情况，将异常转换为标准错误格式。
        
        Args:
            requests: 单条请求列表
            timeout_per_request: 每个请求的超时时间（秒），默认60秒
            
        Returns:
            List[Tuple[List, List]]: 每个请求的结果列表，格式为(成功文件列表, 失败文件列表)
        """
        # 为每个请求创建异步任务
        tasks = []
        for request in requests:
            task = self.send(request)  # 注意：这里没有使用timeout参数，可能需要调整
            tasks.append(task)
        
        # 并发执行所有任务，收集结果（包括异常）
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果，将异常转换为标准格式
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 如果是异常，记录错误并返回空成功列表和错误信息
                print(f"Request {i} failed: {result}")
                processed_results.append(([], [f"Request failed: {str(result)}"]))
            else:
                # 正常结果直接添加
                processed_results.append(result)
                
        return processed_results
    
    async def _restart_worker(self, worker: OCRWorker):
        """
        重启出现故障的worker
        
        尝试停止并重启指定的worker，如果重启失败则创建新的worker替换。
        
        Args:
            worker: 需要重启的OCRWorker实例
        """
        try:
            # 停止当前worker
            worker.stop()
            # 重启worker
            worker.start()
            print(f"Worker on GPU {worker.gpu_id} restarted successfully")
        except Exception as e:
            # 重启失败，创建新的worker替换
            print(f"Failed to restart worker on GPU {worker.gpu_id}: {e}")
            new_worker = OCRWorker(gpu_id=worker.gpu_id)
            new_worker.start()
            # 从列表中移除旧worker，添加新worker
            self.workers.remove(worker)
            self.workers.append(new_worker)
            # 如果可用队列已创建，将新worker加入队列
            if self.available_workers is not None:
                self.available_workers.put_nowait(new_worker)
    
    def get_pool_status(self) -> Dict:
        """
        获取进程池的当前状态信息
        
        返回详细的状态信息，包括worker数量、健康状态、队列状态等。
        用于监控和调试目的。
        
        Returns:
            Dict: 包含以下字段的状态字典：
                - status: "not_started" 或 "running"
                - gpu_ids: GPU ID列表
                - total_workers: 总worker数量
                - available_workers: 可用worker数量
                - busy_workers: 忙碌worker数量
                - healthy_workers: 健康worker数量
                - worker_details: 每个worker的详细信息列表
        """
        if not self._started:
            return {"status": "not_started"}
            
        # 计算可用worker数量
        available_count = self.available_workers.qsize() if self.available_workers is not None else len(self.workers)
        total_count = len(self.workers)
        busy_count = total_count - available_count
        
        # 检查每个worker的健康状态
        worker_status = []
        for worker in self.workers:
            # 检查进程是否存在且未退出
            is_healthy = worker.process and worker.process.poll() is None
            worker_status.append({
                "gpu_id": worker.gpu_id,
                "healthy": is_healthy,
                "process_alive": is_healthy
            })
        
        # 统计健康worker数量
        healthy_workers = sum(1 for w in worker_status if w["healthy"])
        
        # 返回完整的状态信息
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
        """
        停止所有worker进程
        
        终止所有OCRWorker子进程，清空队列并重置状态。
        确保资源被正确释放。
        """
        if not self._started:
            return
            
        print("Stopping OCR worker pool...")
        
        # 清空可用worker队列
        if self.available_workers is not None:
            while not self.available_workers.empty():
                try:
                    self.available_workers.get_nowait()
                except:
                    break
        
        # 停止所有worker
        for worker in self.workers:
            worker.stop()
            
        # 清空worker列表并重置状态
        self.workers.clear()
        self.available_workers = None
        self._started = False
        print("OCR worker pool stopped")

# 创建全局进程池实例 - 自动根据OCR_GPU_ID配置
ocr_worker_pool = OCRWorkerPool()
