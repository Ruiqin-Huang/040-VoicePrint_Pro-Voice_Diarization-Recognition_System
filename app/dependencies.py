"""
FastAPI依赖注入模块

提供FastAPI应用的依赖注入函数，用于在请求处理过程中
获取共享的服务实例和配置对象。

主要功能：
- 提供PathMapper实例的依赖注入
- 支持从FastAPI应用状态中获取共享对象

依赖：
- FastAPI: Web框架和Request对象
- app.config.path_mapper: 路径映射服务
"""

from fastapi import Request
from app.config.path_mapper import PathMapper

def get_path_mapper(request: Request) -> PathMapper:
    """
    获取PathMapper实例的依赖注入函数

    从FastAPI应用的state中获取预先配置的PathMapper实例，
    用于处理宿主机和容器之间的路径映射。

    Args:
        request: FastAPI请求对象，包含应用状态信息

    Returns:
        PathMapper: 配置好的路径映射器实例

    Note:
        该函数作为FastAPI的依赖注入使用，确保每个请求都能
        访问到正确的路径映射配置。由于项目已改为相对路径
        输入方式，该依赖暂时未被使用，但保留以备不时之需。
    """
    return request.app.state.path_mapper