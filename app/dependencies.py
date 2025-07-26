from fastapi import Request
from app.config.path_mapper import PathMapper

def get_path_mapper(request: Request) -> PathMapper:
    """从app.state获取PathMapper实例"""
    return request.app.state.path_mapper