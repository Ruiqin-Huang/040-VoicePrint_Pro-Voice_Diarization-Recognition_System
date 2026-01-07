"""
路径映射管理模块

提供宿主机和Docker容器之间路径映射的功能。
用于在容器化环境中正确处理文件路径转换。

主要功能：
- 宿主机路径到容器路径的转换
- 容器路径到宿主机路径的反向转换
- 路径有效性验证
- 支持音频分段和识别输出的子目录映射

设计背景：
- Docker容器内外的路径表示不同
- 需要在不同操作系统间保持路径兼容性
- 支持复杂的输出目录结构（分段/识别子目录）

依赖：
- pathlib: 跨平台路径处理
- app.config.settings: 应用配置信息

注意：
由于项目已改为相对路径输入方式，该模块暂时未被使用，
但保留以备不时之需。
"""

import os
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
import json
from typing import Dict, Tuple, Optional, List

from app.config.settings import settings

class PathMapper:
    """
    路径映射器类

    负责管理宿主机和Docker容器之间的路径映射关系。
    支持双向路径转换和有效性验证。

    Attributes:
        host_input_dir (str): 宿主机输入目录的绝对路径
        host_output_dir (str): 宿主机输出目录的绝对路径
        container_input_dir (str): 容器内输入目录的绝对路径
        container_output_dir (str): 容器内输出目录的绝对路径
        container_segmentation_dir (str): 容器内音频分段输出目录
        container_recognition_dir (str): 容器内音频识别输出目录
    """

    def __init__(self, host_input_dir: str, host_output_dir: str):
        """
        初始化路径映射器

        Args:
            host_input_dir: 宿主机输入目录的绝对路径
            host_output_dir: 宿主机输出目录的绝对路径

        Note:
            容器内路径通过settings配置获取，确保路径映射的准确性
        """
        # 规范化宿主机路径，确保路径格式统一
        self.host_input_dir = str(host_input_dir)
        self.host_output_dir = str(host_output_dir)

        # 预定义的容器内路径（来自settings配置）
        # 使用resolve()获取绝对路径，避免相对路径问题
        self.container_input_dir = str(Path(settings.INPUT_DIR).resolve())
        self.container_output_dir = str(Path(settings.OUTPUT_DIR).resolve())
        self.container_segmentation_dir = str(Path(settings.SEGMENTATION_OUTPUT_DIR).resolve())
        self.container_recognition_dir = str(Path(settings.RECOGNITION_OUTPUT_DIR).resolve())

        # 验证路径映射配置的完整性
        self._validate_mappings()

    def _to_pure_path(self, path_str):
        """
        将路径字符串转换为纯路径对象

        根据路径特征自动识别是Windows还是POSIX路径格式。
        用于跨平台路径处理。

        Args:
            path_str: 待转换的路径字符串

        Returns:
            PurePath: 对应的纯路径对象（PureWindowsPath或PurePosixPath）

        Note:
            - 包含盘符（如D:）或反斜杠的路径被识别为Windows路径
            - 其他路径默认为POSIX格式
        """
        # 如果包含盘符（如 D:\）或反斜杠，视为 Windows 路径
        if ":" in path_str or "\\" in path_str:
            return PureWindowsPath(path_str)
        else:
            return PurePosixPath(path_str)

    def _validate_mappings(self):
        """
        验证路径映射配置的完整性

        检查必要的宿主机目录是否存在和可访问。
        在调试模式下输出路径映射配置信息。

        Note:
            目前路径验证代码被注释掉，可能由于容器环境
            或权限控制的原因。保留框架以备将来启用。
        """
        required_dirs = [
            self.host_input_dir,
            self.host_output_dir
        ]

        # 路径存在性和权限检查（目前被注释）
        # 在容器环境中可能不需要或无法进行这些检查
        # for dir_path in required_dirs:
        #     if not os.path.exists(dir_path):
        #         raise RuntimeError(f"宿主机目录不存在: {dir_path}")
        #
        #     if not os.access(dir_path, os.R_OK | os.W_OK):
        #         raise RuntimeError(f"宿主机目录无读写权限: {dir_path}")

        # 在调试模式下输出路径映射配置，便于问题排查
        if settings.DEBUG:
            print(f"路径映射配置：")
            print(f"  宿主机输入: {self.host_input_dir} → 容器输入: {self.container_input_dir}")
            print(f"  宿主机输出: {self.host_output_dir}")
            print(f"    → 容器分段输出: {self.container_segmentation_dir}")
            print(f"    → 容器识别输出: {self.container_recognition_dir}")

    def host_to_container(self, host_path: str) -> str:
        """
        将宿主机路径转换为容器内路径

        支持输入和输出路径的转换，包括子目录的智能识别。
        输出路径会根据子目录名称自动映射到对应的容器目录。

        Args:
            host_path: 宿主机上的绝对路径

        Returns:
            str: 对应的容器内路径

        Raises:
            ValueError: 当路径不在允许的映射范围内时抛出

        Note:
            - 输入路径直接映射到容器输入目录
            - 输出路径根据子目录名称智能路由：
              * audio_segmentation/* → 分段输出目录
              * audio_recognition/* → 识别输出目录
              * 其他 → 默认分段输出目录
        """
        host_path = str(host_path)

        # 处理输入路径映射：宿主机输入目录下的文件映射到容器输入目录
        if host_path.startswith(self.host_input_dir):
            # 计算相对路径（文件名部分）
            filename = self._to_pure_path(host_path).relative_to(self._to_pure_path(self.host_input_dir))
            # 组合为容器内完整路径
            container_path = Path(self.container_input_dir) / filename
            return str(container_path)

        # 处理输出路径映射：支持子目录的智能识别和路由
        elif host_path.startswith(self.host_output_dir):
            # 计算相对于输出目录的相对路径
            relative_path = os.path.relpath(host_path, self.host_output_dir)

            # 根据子目录名称路由到对应的容器目录
            if relative_path.startswith("audio_segmentation"):
                # 音频分段结果映射到分段输出目录
                return os.path.join(
                    self.container_segmentation_dir,
                    relative_path[len("audio_segmentation")+1:]  # 去掉目录名前缀和分隔符
                )
            elif relative_path.startswith("audio_recognition"):
                # 音频识别结果映射到识别输出目录
                return os.path.join(
                    self.container_recognition_dir,
                    relative_path[len("audio_recognition")+1:]  # 去掉目录名前缀和分隔符
                )
            else:
                # 默认情况下映射到分段输出目录
                return os.path.join(
                    self.container_segmentation_dir,
                    relative_path
                )

        # 路径不在允许的映射范围内，抛出异常
        raise ValueError(f"路径未映射到容器: {host_path} (仅允许: {self.host_input_dir} 或 {self.host_output_dir} 下的路径)")

    def container_to_host(self, container_path: str) -> str:
        """
        将容器内路径转换为宿主机路径

        执行host_to_container的反向操作，将容器路径映射回宿主机路径。

        Args:
            container_path: 容器内的绝对路径

        Returns:
            str: 对应的宿主机路径

        Raises:
            ValueError: 当容器路径不在映射范围内时抛出

        Note:
            - 容器输入目录下的路径映射回宿主机输入目录
            - 容器输出目录下的路径映射回宿主机输出目录
            - 使用resolve()确保路径规范化
        """
        # 规范化容器路径，确保是绝对路径
        container_path = str(Path(container_path).resolve())

        # 处理输入路径的反向映射
        if container_path.startswith(self.container_input_dir):
            # 计算相对于容器输入目录的文件名
            filename = self._to_pure_path(container_path).relative_to(self._to_pure_path(self.container_input_dir))
            # 组合为宿主机完整路径
            host_path = self._to_pure_path(self.host_input_dir) / filename
            return str(host_path)

        # 处理输出路径的反向映射
        elif container_path.startswith(self.container_output_dir):
            # 计算相对于容器输出目录的文件路径
            file = self._to_pure_path(container_path).relative_to(self._to_pure_path(self.container_output_dir))

            # 组合为宿主机输出目录下的完整路径
            host_path = self._to_pure_path(self.host_output_dir) / file
            return str(host_path)

        # 容器路径不在映射范围内
        raise ValueError(f"容器路径未映射到宿主机: {container_path}")

    def validate_host_path(self, host_path: str) -> bool:
        """
        验证宿主机路径是否在允许的映射范围内

        通过尝试执行路径转换来验证路径的有效性。
        如果转换成功则路径有效，否则无效。

        Args:
            host_path: 待验证的宿主机路径

        Returns:
            bool: 路径是否有效（True为有效，False为无效）
        """
        try:
            # 尝试转换为容器路径，如果成功则路径有效
            self.host_to_container(host_path)
            return True
        except ValueError:
            # 转换失败，路径无效
            return False

    def get_container_input_dir(self) -> str:
        """
        获取容器内输入目录路径

        Returns:
            str: 容器内输入目录的绝对路径

        Note:
            该路径用于容器内文件输入操作
        """
        return self.container_input_dir

    def get_container_output_dirs(self) -> Dict[str, str]:
        """
        获取容器内所有输出目录路径

        Returns:
            Dict[str, str]: 包含所有输出目录的字典
                - "segmentation": 音频分段输出目录
                - "recognition": 音频识别输出目录

        Note:
            方便外部代码获取不同类型的输出目录路径
        """
        return {
            "segmentation": self.container_segmentation_dir,
            "recognition": self.container_recognition_dir
        }