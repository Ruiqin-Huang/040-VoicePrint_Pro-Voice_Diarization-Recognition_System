from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator

class DiarizationComparisonRequest(BaseModel):
    audio_files: List[str] = Field(..., description="待处理的音频文件绝对路径列表")
    collection_name: str = Field(..., description="用于比对的目标声纹库集合名称")
    accept_threshold: float = Field(0.85, ge=0, le=1, description="可接受的相似度阈值，范围[0, 1]，默认为0.85")

    @validator('audio_files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("音频文件列表不能为空")
        return v

class ComparisonResultDetail(BaseModel):
    person_id: str
    similarity: float

class DiarizationComparisonResult(BaseModel):
    origin_audio_file: str = Field(..., description="原始音频文件名")
    segment_audio_file: str = Field(..., description="切分后的音频文件名")
    calling_called: str = Field(..., description="主叫或被叫 ('calling' 或 'called')")
    identified_speaker: str = Field(..., description="对比分析结果（距离最小的person_id或'unknown_person'）")
    max_similarity: float = Field(..., description="与最相似说话人的余弦相似度（0-1）")
    is_accepted: bool = Field(..., description="是否通过相似度阈值判断")
    compare_result: List[ComparisonResultDetail] = Field(..., description="与数据库中所有person_id的相似度计算结果")

class DiarizationComparisonResponseData(BaseModel):
    collection_name: str = Field(..., description="参与比较的目标说话人声纹库")
    comparison_results: List[DiarizationComparisonResult]
    inserted_count: int = Field(..., description="比对通过后，成功补充入库的声纹数量")
    inserted_result: List[Dict[str, str]] = Field(
        ..., 
        description="成功补充入库的记录详情列表，每条记录包含音频文件路径、人员ID和生成的主键ID"
    )