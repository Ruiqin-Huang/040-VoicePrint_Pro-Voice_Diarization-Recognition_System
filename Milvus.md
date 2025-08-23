# MilvusClient 接口说明

## 数据结构

| 字段名 | 类型 | 描述 | 约束 |
|--------|------|------|------|
| id | INT64 | 自增主键 | 自动生成，唯一标识 |
| person_id | VARCHAR | 人员唯一标识 | max_length=64 |
| file_id | VARCHAR | 源音频文件唯一标识 | max_length=64 |
| embedding | FLOAT_VECTOR | 声纹特征向量 | dim=192 |

表格已创建，且插入时有自动建表，直接使用接口即可

## insert - 数据插入接口

### 方法
```python
def insert(self, collection_name, person_ids, file_ids, embeddings)
```

### 功能描述
批量插入声纹数据到指定集合，每条记录包含人员ID、文件ID和192维声纹向量。

### 参数说明

| 参数名 | 类型 | 必须 | 描述 | 示例 |
|--------|------|------|------|------|
| `collection_name` | str | 是 | 目标集合名称 | `"voiceprint_db"` |
| `person_ids` | List[str] | 是 | 人员ID列表 | `["user_001", "user_002"]` |
| `file_ids` | List[str] | 是 | 文件ID列表 | `["file_001.wav", "file_002.wav"]` |
| `embeddings` | List[List[float]] | 是 | 192维声纹向量列表 | `[[0.1]*192, [0.2]*192]` |

### 返回值
- **成功**: List[int] - 插入记录的主键ID列表
- **失败**: None

### 使用示例
```python
from app.config.settings import settings
from utils.Milvus import MilvusClient

# 初始化客户端
config = {"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT}
mc = MilvusClient(config=config)

# 准备批量数据
person_ids = ["user_001", "user_001", "user_002"]
file_ids = ["1111", "1112", "1113"]
embeddings = [
    [0.1] * 192,  # user_001 的第一个声纹
    [0.15] * 192, # user_001 的第二个声纹
    [0.2] * 192   # user_002 的声纹
]

# 执行批量插入
result_ids = mc.insert(settings.MILVUS_COLLECTION, person_ids, file_ids, embeddings)
print(f"插入成功，生成ID: {result_ids}")
```

---

## search_embedding_cosine - 余弦相似度匹配接口

### 方法签名
```python
def search_embedding_cosine(self, collection_name, query_vectors, threshold=0.9, top_k=1)
```

### 功能描述
基于余弦相似度搜索最相似的声纹记录，支持批量查询和阈值过滤。

### 参数说明

| 参数名 | 类型 | 必须 | 描述 | 默认值 | 示例 |
|--------|------|------|------|--------|------|
| `collection_name` | str | 是 | 目标集合名称 | - | `"voiceprint_db"` |
| `query_vectors` | List[List[float]] | 是 | 查询向量列表 | - | `[[0.1]*192, [0.2]*192]` |
| `threshold` | float | 否 | 相似度阈值 | `0.9` | `0.85` |
| `top_k` | int | 否 | 每个查询返回的最大结果数 | `1` | `1` |

### 返回值
- **成功**: List[Dict] - 每个查询向量的搜索结果列表
- **失败**: List[Dict] - 保持结构一致的错误结果

### 返回数据结构
```python
[
    {
        "query_index": 0,                   # 查询向量索引
        "compare_result": "158xxxxx",       # 最佳匹配人员ID（没找到为 None）
        "compare_similarity": 0.0,          # 最佳匹配相似度（余弦相似度）
        "object_file_id": "1234",           # 最佳匹配源音频文件ID（没找到为 None）
        "all_matches": [...]                # 所有匹配结果（调试用）
    }
]
```

### 使用示例
```python
from app.config.settings import settings
from utils.Milvus import MilvusClient

# 初始化客户端
config = {"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT}
mc = MilvusClient(config=config)

# 准备查询向量
query_vectors = [
    [0.12] * 192,  # 查询向量1
    [0.18] * 192   # 查询向量2
]

# 执行余弦相似度搜索
results = mc.search_embedding_cosine(collection_name=settings.MILVUS_COLLECTION, query_vectors=query_vectors, threshold=0.8)

# 处理搜索结果
response = []
for i, result in enumerate(results):
    if result['compare_result']:
        response.append({
                "info_person_voice_clustering_id": "",
                "calling_called": "",
                "compare_result": result["compare_result"],
                "compare_similarity": result["compare_similarity"],
                "subject_file_id": file_id,
                "object_file_id": result["object_file_id"]
            }
        )
    else:
        response.append({
                "info_person_voice_clustering_id": "",
                "calling_called": "",
                "compare_result": "unknown",
                "compare_similarity": 0.0,
                "subject_file_id": file_id,
                "object_file_id": ""
            }
        )
```
