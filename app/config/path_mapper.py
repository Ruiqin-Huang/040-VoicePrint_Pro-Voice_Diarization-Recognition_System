import os
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
import json
from typing import Dict, Tuple, Optional, List

from app.config.settings import settings

class PathMapper:
    def __init__(self, host_input_dir: str, host_output_dir: str):
        """
        :param host_input_dir: 宿主机输入目录绝对路径
        :param host_output_dir: 宿主机输出目录绝对路径
        """
        # 规范化宿主机路径
        self.host_input_dir = str(host_input_dir)
        self.host_output_dir = str(host_output_dir)
        
        # 预定义的容器内路径（来自settings）
        self.container_input_dir = str(Path(settings.INPUT_DIR).resolve())
        self.container_output_dir = str(Path(settings.OUTPUT_DIR).resolve())
        self.container_segmentation_dir = str(Path(settings.SEGMENTATION_OUTPUT_DIR).resolve())
        self.container_recognition_dir = str(Path(settings.RECOGNITION_OUTPUT_DIR).resolve())
        
        # 验证路径映射完整性
        self._validate_mappings()

    def _to_pure_path(self, path_str):
        # 如果包含盘符（如 D:\）或反斜杠，视为 Windows 路径
        if ":" in path_str or "\\" in path_str:
            return PureWindowsPath(path_str)
        else:
            return PurePosixPath(path_str)

    def _validate_mappings(self):
        """验证必要路径是否可访问"""
        required_dirs = [
            self.host_input_dir,
            self.host_output_dir
        ]
        
        # for dir_path in required_dirs:
        #     if not os.path.exists(dir_path):
        #         raise RuntimeError(f"宿主机目录不存在: {dir_path}")
            
        #     if not os.access(dir_path, os.R_OK | os.W_OK):
        #         raise RuntimeError(f"宿主机目录无读写权限: {dir_path}")

        if settings.DEBUG:
            print(f"路径映射配置：")
            print(f"  宿主机输入: {self.host_input_dir} → 容器输入: {self.container_input_dir}")
            print(f"  宿主机输出: {self.host_output_dir}")
            print(f"    → 容器分段输出: {self.container_segmentation_dir}")
            print(f"    → 容器识别输出: {self.container_recognition_dir}")

    def host_to_container(self, host_path: str) -> str:
        """宿主机路径 → 容器路径"""
        host_path = str(host_path)
        
        # 输入路径映射
        if host_path.startswith(self.host_input_dir):
            # 获取文件名
            filename = self._to_pure_path(host_path).relative_to(self._to_pure_path(self.host_input_dir))
            # 组合容器路径
            container_path = Path(self.container_input_dir) / filename
            return str(container_path)
        
        # 输出路径映射（自动识别子目录）
        elif host_path.startswith(self.host_output_dir):
            relative_path = os.path.relpath(host_path, self.host_output_dir)
            
            if relative_path.startswith("audio_segmentation"):
                return os.path.join(
                    self.container_segmentation_dir,
                    relative_path[len("audio_segmentation")+1:]
                )
            elif relative_path.startswith("audio_recognition"):
                return os.path.join(
                    self.container_recognition_dir,
                    relative_path[len("audio_recognition")+1:]
                )
            else:
                # 默认映射到分段输出目录
                return os.path.join(
                    self.container_segmentation_dir,
                    relative_path
                )
        
        raise ValueError(f"路径未映射到容器: {host_path} (仅允许: {self.host_input_dir} 或 {self.host_output_dir} 下的路径)")

    def container_to_host(self, container_path: str) -> str:
        """容器路径 → 宿主机路径"""
        container_path = str(Path(container_path).resolve())
        
        # 输入路径反向映射
        if container_path.startswith(self.container_input_dir):
            # 获取文件名
            filename = self._to_pure_path(container_path).relative_to(self._to_pure_path(self.container_input_dir))
            # 组合容器路径
            host_path = self._to_pure_path(self.host_input_dir) + filename
            return str(host_path)
        
        # 输出路径反向映射
        elif container_path.startswith(self.container_output_dir):
            print(container_path, self.container_output_dir)
            # 获取文件名
            file = self._to_pure_path(container_path).relative_to(self._to_pure_path(self.container_output_dir))

            print(file)
            # 组合容器路径
            host_path = self._to_pure_path(self.host_output_dir + '/' + str(file))
            print(host_path)
            return str(host_path)
        
        raise ValueError(f"容器路径未映射到宿主机: {container_path}")

    def validate_host_path(self, host_path: str) -> bool:
        """验证宿主机路径是否允许访问"""
        try:
            self.host_to_container(host_path)
            return True
        except ValueError:
            return False

    def get_container_input_dir(self) -> str:
        """获取容器内输入目录"""
        return self.container_input_dir

    def get_container_output_dirs(self) -> Dict[str, str]:
        """获取容器内所有输出目录"""
        return {
            "segmentation": self.container_segmentation_dir,
            "recognition": self.container_recognition_dir
        }