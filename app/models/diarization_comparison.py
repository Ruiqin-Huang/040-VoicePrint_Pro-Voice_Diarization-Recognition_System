"""
说话人切分及声纹比对数据模型模块

定义说话人切分和声纹比对相关的Pydantic数据模型，用于API请求和响应的数据验证和序列化。

包含以下模型：
- DiarizationComparisonRequest: 说话人切分及声纹比对请求模型
- ComparisonResultDetail: 单个说话人的比对结果详情
- ClusterResultItem: 聚类结果项
- DiarizationComparisonResult: 单个音频片段的切分和比对结果
- DiarizationComparisonResponseData: 说话人切分及声纹比对响应数据模型

依赖：
- pydantic: 数据验证和序列化
- typing: 类型注解
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator

class DiarizationComparisonRequest(BaseModel):
    """
    说话人切分及声纹比对请求模型
    
    用于接收音频文件列表，进行主被叫切分，并将切分后的声纹与目标声纹库进行比对。
    
    Attributes:
        audio_files: 待处理的音频文件绝对路径列表
        collection_name: 用于比对的目标声纹库集合名称
        accept_threshold: (已注释) 可接受的相似度阈值，范围[0, 1]，默认为0.85
    """
    audio_files: List[str] = Field(..., description="待处理的音频文件绝对路径列表")
    collection_name: str = Field(..., description="用于比对的目标声纹库集合名称")
    # accept_threshold: float = Field(0.85, ge=0, le=1, description="可接受的相似度阈值，范围[0, 1]，默认为0.85")

    @validator('audio_files')
    def files_must_not_be_empty(cls, v):
        """
        验证器：确保音频文件列表不为空
        
        Args:
            v: 音频文件列表值
            
        Returns:
            List[str]: 验证后的音频文件列表
            
        Raises:
            ValueError: 当音频文件列表为空时抛出
        """
        if not v:
            raise ValueError("音频文件列表不能为空")
        return v

class ComparisonResultDetail(BaseModel):
    """
    单个说话人的比对结果详情模型
    
    表示音频片段与声纹库中某个人员的相似度比对结果。
    
    Attributes:
        person_id: 人员ID
        similarity: 相似度分数（0-1之间的浮点数）
    """
    person_id: str
    similarity: float

class ClusterResultItem(BaseModel):
    """
    聚类结果项模型
    
    表示单个音频片段在t-SNE降维空间中的位置和所属聚类信息。
    
    Attributes:
        segment_audio_file: 分割后的音频文件名
        x_coordinate: t-SNE降维后的X坐标
        y_coordinate: t-SNE降维后的Y坐标
        cluster_id: 所属聚类ID
    """
    segment_audio_file: str = Field(..., description="分割后的音频文件名")
    x_coordinate: float = Field(..., description="t-SNE降维后的X坐标")
    y_coordinate: float = Field(..., description="t-SNE降维后的Y坐标")
    cluster_id: int = Field(..., description="所属聚类ID")

class DiarizationComparisonResult(BaseModel):
    """
    单个音频片段的切分和比对结果模型
    
    包含音频片段的主被叫信息、聚类结果以及与声纹库的比对结果。
    
    Attributes:
        origin_audio_file: 原始音频文件名
        segment_audio_file: 切分后的音频文件名
        calling_called: 主叫或被叫 ('calling' 或 'called')
        cluster_id: 该音频片段所属的聚类ID
        top_match_speaker: 相似度最高的person_id
        top_match_similarity: 与最相似说话人的余弦相似度（0-1）
        compare_result: 与数据库中所有person_id的相似度计算结果列表
        is_accepted: (已注释) 是否通过相似度阈值判断
    """
    origin_audio_file: str = Field(..., description="原始音频文件名")
    segment_audio_file: str = Field(..., description="切分后的音频文件名")
    calling_called: str = Field(..., description="主叫或被叫 ('calling' 或 'called')")
    cluster_id: int = Field(..., description="该音频片段所属的聚类ID")
    top_match_speaker: Optional[str] = Field(..., description="相似度最高的person_id") 
    top_match_similarity: Optional[float] = Field(..., description="与最相似说话人的余弦相似度（0-1）")  
    compare_result: List[ComparisonResultDetail] = Field(..., description="与数据库中所有person_id的相似度计算结果")
    # is_accepted: bool = Field(..., description="是否通过相似度阈值判断") 

class DiarizationComparisonResponseData(BaseModel):
    """
    说话人切分及声纹比对响应数据模型
    
    包含完整的切分、聚类和比对结果信息。
    
    Attributes:
        collection_name: 参与比较的目标说话人声纹库集合名称
        comparison_results: 所有音频片段的切分和比对结果列表
        cluster_results: 所有分割音频的聚类结果，包含2D坐标和聚类编号
        inserted_count: (已注释) 比对通过后，成功补充入库的声纹数量
        inserted_result: (已注释) 成功补充入库的记录详情列表
    """
    collection_name: str = Field(..., description="参与比较的目标说话人声纹库")
    comparison_results: List[DiarizationComparisonResult]
    cluster_results: List[ClusterResultItem] = Field(..., description="所有分割音频的聚类结果，包含2D坐标和聚类编号")
    # inserted_count: int = Field(..., description="比对通过后，成功补充入库的声纹数量")  
    # inserted_result: List[Dict[str, str]] = Field(..., description="成功补充入库的记录详情列表，每条记录包含音频文件路径、人员ID和生成的主键ID")  