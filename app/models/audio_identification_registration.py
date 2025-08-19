from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator

class AudioIdentificationRequest(BaseModel):
    audio_files: List[str] = Field(..., description="待处理的音频文件绝对路径列表")
    num_speakers_per_audio: Optional[int] = Field(2, description="每个输入音频文件中预期的说话人数量")
    update_voiceprintlib: bool = Field(False, description="是否更新声纹库。True则注册新声纹并更新已有声纹，False则仅识别不更新。")
    threshold: float = Field(0.65, description="声纹识别的余弦距离阈值，小于等于该值则认为是同一人。取值范围[0, 2]。")

    @validator('audio_files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("音频文件列表不能为空")
        return v

class IdentificationResult(BaseModel):
    source_segment: str = Field(..., description="用于识别的音频片段路径（相对工作区）")
    identified_speaker: str = Field(..., description="识别出的说话人ID，'unknown'代表未识别或新说话人")
    is_new_speaker: bool = Field(..., description="是否被判定为新说话人")
    min_distance: Optional[float] = Field(None, description="与最相似的库中说话人的距离")
    distances: Dict[str, float] = Field(..., description="与库中所有说话人的距离映射")

class IdentificationResponseData(BaseModel):
    identification_results: List[IdentificationResult]
    newly_registered_speakers: Optional[Dict[str, List[str]]] = Field(None, description="本次任务新注册的说话人及其音频片段")
    updated_speakers: Optional[List[str]] = Field(None, description="本次任务更新了声纹的已有说话人ID列表")
    library_updated: bool