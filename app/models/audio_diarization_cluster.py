from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator

class AudioDiarizationClusterRequest(BaseModel):
    audio_files: List[str] = Field(..., description="待处理的音频文件绝对路径列表")
    num_speakers_per_audio: Optional[int] = Field(2, description="每个音频文件中预期的说话人数量")

    @validator('audio_files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("音频文件列表不能为空")
        return v

class ClusterResult(BaseModel):
    speaker_id: str
    audio_files: List[str]

class AudioDiarizationClusterResponse(BaseModel):
    total_clusters: int = Field(..., description="聚类出的说话人总数")
    clusters: List[ClusterResult] = Field(..., description="每个聚类的详细信息")
    workspace: str = Field(..., description="本次任务的工作目录，包含所有中间和最终结果")