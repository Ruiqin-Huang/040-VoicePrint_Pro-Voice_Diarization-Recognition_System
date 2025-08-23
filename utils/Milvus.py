import pymilvus
from datetime import datetime
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

VECTOR_DIMENSION = 192
METRIC_TYPE = "IP"


class MilvusClient:
    def __init__(self, config):
        self.host = config.get('host', None)
        self.port = config.get('port', None)
        self.get_connection()
        self.collection = None

    def get_connection(self):
        try:
            """
            链接 milvus 数据库
            """
            # self.collection = None
            connections.connect(host=self.host, port=self.port)
            print(f"Successfully connect to Milvus with IP:{self.host} and PORT:{self.port}")
        except Exception as e:
            print(f"Failed to connect Milvus: {e}")

    def change_collection(self, collection_name):
        """
        查询 指定表（集合）是否存在
        """
        try:
            if self.has_collection(collection_name):
                self.collection = Collection(name=collection_name)
            else:
                raise Exception(f"There has no collection named:{collection_name}")
        except Exception as e:
            print(f"Error: {e}")

    def has_collection(self, collection_name):
        """
        查询 指定表（集合）是否存在
        """
        try:
            status = utility.has_collection(collection_name)
            print(f"Find collection {collection_name} {status}")
            return status
        except Exception as e:
            print(f"Failed to check collection: {e}")

    def create_collection(self, collection_name):
        """
        创建表（集合）
        自定义 FieldSchema（字段类型及格式）
        field1，field2，field3 ，field4为实例字段信息
        https://milvus.io/docs/v2.1.x/create_collection.md 根据数据格式定义FieldSchema类型
        """
        try:
            if not self.has_collection(collection_name):
                # 1. 自增主键
                field_id = FieldSchema(
                    name="id", 
                    dtype=DataType.INT64, 
                    is_primary=True, 
                    auto_id=True  # 自动生成ID
                )

                # 2. 人员ID
                field_person_id = FieldSchema(
                    name="person_id", 
                    dtype=DataType.VARCHAR, 
                    max_length=64,
                    description="人员唯一标识"
                )

                # 3. 源音频文件ID
                field_file = FieldSchema(
                    name="file_id", 
                    dtype=DataType.VARCHAR, 
                    max_length=64,
                    description="源音频文件唯一标识"
                )

                # 4. 声纹向量
                field_embedding = FieldSchema(
                    name="embedding", 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=VECTOR_DIMENSION,
                    description="声纹特征向量"
                )

                # field1 = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, descrition="DataType.VARCHAR",
                #                      is_primary=True, auto_id=False)

                # field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector",
                #                      dim=VECTOR_DIMENSION, is_primary=False)

                # field3 = FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=200, description="_id")
                # field3 = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000, description="text")

                schema = CollectionSchema(fields=[field_id, field_person_id, field_file, field_embedding], description="new test")
                self.collection = Collection(name=collection_name, schema=schema)
                # print(f"Create Milvus collection: {self.collection}")
            return f"Collection {collection_name} OK"
        except Exception as e:
            print(f"Failed to create collection: {e}")

    def insert(self, collection_name, person_ids, file_ids, embeddings):
        """
        插入数据（根据 FieldSchema 定义）
        参数:
            collection_name: 集合名称
            data: 批量数据，格式为 [(person_id, file_id, embedding), ...]
        """
        try:
            self.create_collection(collection_name)
            self.collection = Collection(name=collection_name)
            # data = [vectors]
            data = [person_ids, file_ids, embeddings]
            mr = self.collection.insert(data)
            ids = mr.primary_keys
            self.collection.load()
            # collection加载到内存中
            print(f"Insert vectors to Milvus in collection: {collection_name} with {len(embeddings)} rows")
            return ids
        except Exception as e:
            print(f"Failed to insert data into Milvus: {e}")

    def create_index(self, collection_name, field_name):
        """
        创建索引
        https://milvus.io/docs/v2.1.x/build_index.md
        """
        try:
            self.change_collection(collection_name)
            default_index = {"index_type": "IVF_FLAT", "metric_type": METRIC_TYPE, "params": {"nlist": 16384}}
            status = self.collection.create_index(field_name=field_name, index_params=default_index)
            if not status.code:
                print(
                    f"Successfully create index in collection:{collection_name} with param:{default_index}")
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            print(f"Failed to create index: {e}")

    def drop_index(self, collection_name):
        """
        删除索引
        """
        try:
            self.change_collection(collection_name)
            self.collection.drop_index()
            print("Successfully drop index!")
            return True
        except Exception as e:
            print(f"Failed to drop index: {e}")
            return False

    def delete_collection(self, collection_name):
        """
        删除表（集合）
        """
        try:
            self.change_collection(collection_name)
            self.collection.drop()
            print("Successfully drop collection!")
            return True
        except Exception as e:
            print(f"Failed to drop collection: {e}")
            return False

    def delete_data(self, collection_name, id):
        """
        删除一条指定id的数据
        """
        try:
            self.change_collection(collection_name)
            expr = "id in[" + id + "]"
            self.collection.delete(expr)
            return True
        except Exception as e:
            print(f"Failed to drop collection data: {e}")
            return False

    def search_embedding_cosine(self, collection_name, query_vectors, threshold=0.9, top_k=1):
        """
        根据向量查询 top_k 个相近的数据，使用余弦相似度搜索（Milvus内置支持）
        注意：余弦相似度范围是[-1, 1]，1表示完全相似
        """
        try:
            self.change_collection(collection_name)
            
            # 使用余弦相似度参数
            search_params = {
                "metric_type": "IP",  # 内积，用于余弦相似度
                "params": {"nprobe": 16}
            }
            
            results = self.collection.search(
                query_vectors, 
                anns_field="embedding", 
                param=search_params, 
                limit=top_k,
                output_fields=["person_id", "file_id"]
            )

            # 为每个查询向量处理结果
            all_results = []
            
            for i, query_result in enumerate(results):
                # 初始化结果结构
                result = {
                    "query_index": i,
                    "compare_result": None,
                    "compare_similarity": 0.0,
                    "object_file_id": None,
                    "all_matches": []         # 所有结果（用于调试）
                }
                
                # 处理所有匹配结果
                for hit in query_result:
                    similarity = hit.distance
                    match_info = {
                        "person_id": hit.entity.get("person_id"),
                        "file_id": hit.entity.get("file_id"),
                        "similarity": similarity,
                        "id": hit.id
                    }
                    
                    # 记录所有结果
                    result["all_matches"].append(match_info)
                    
                    # 筛选符合阈值的结果，更新最佳匹配
                    if similarity >= threshold and similarity > result["compare_similarity"]:
                        result["compare_similarity"] = similarity
                        result["compare_result"] = hit.entity.get("person_id")
                        result["object_file_id"] = hit.entity.get("file_id")
                
                all_results.append(result)
            
            return all_results
            
        except Exception as e:
            print(f"搜索失败: {e}")
            # 返回空结果，保持结构一致
            return [{
                "query_index": i,
                "compare_result": None,
                "compare_similarity": 0.0,
                "object_file_id": None,
                "all_matches": []
            } for i in range(len(query_vectors))]
    
    def search_file_id(self, collection_name, file_ids):
        """
        根据 file_id 查询数据
        """
        try:
            self.change_collection(collection_name)
            res = self.collection.query(expr=f"file_id in {file_ids}", output_fields=["person_id", "file_id", "embedding"])
            return res
        except Exception as e:
            print(f"Failed to search in Milvus: {e}")
            return False

    def search_varchar_list_embedding(self, collection_name, id_name, id_list,output_fields):
        """
        根据 指定 id_list（多个id） 查询数据
        输入：
        id_name：查询字段的name，注意，字段类型 是 VARCHAR  ！！！！！！！
        id_list: 查询字段列表
        """
        try:
            self.change_collection(collection_name)
            ql_str = str(id_list).replace("'", '"')
            exprr = id_name + f' in {ql_str}'
            # 'id in ["13c02b05ca1f149bcbc18db61402ad60"]'
            # res1 = self.collection.query(expr=exprr, output_fields=[id_name, "embeddings"])
            res1 = self.collection.query(expr=exprr, output_fields=output_fields)
            # res1 = self.collection.query(expr="id in[436181647246025113]", output_fields=["doc_id", "embedding"])
            # print(res1)
            return res1
        except Exception as e:
            print(f"Failed to search in Milvus: {e}")
            return False

    def search_Int_list_embedding(self, collection_name, id_name, id_list):
        """
        根据 指定 id_list（多个id） 查询数据
        输入：
        id_name：查询字段的name，注意，字段类型 是 Int  ！！！！！！！
        id_list: 查询字段列表
        """
        try:
            self.change_collection(collection_name)
            # exprr = "id in "+str(id_list)
            exprr = id_name + " in " + str(id_list)
            res1 = self.collection.query(expr=exprr, output_fields=["text", "embedding"])
            # res1 = self.collection.query(expr="id in[436181647246025113]", output_fields=["doc_id", "embedding"])
            # print(res1)
            return res1
        except Exception as e:
            print(f"Failed to search in Milvus: {e}")
            return False

    def search_one_id_embedding(self, collection_name, id):
        """
       根据 指定 id  查询数据
       """
        try:
            self.change_collection(collection_name)

            # expr = "film_id in [ 0, 1 ]"
            id = '5e7eceeb33fedd77fac10d8b4c3e3d65'
            exprr = 'news_id in ' + '[' + str(id) + ']'
            # exprr = f"news_id in {[id]}"
            # exprr = f"id in [436656467004420717]"
            # res1 = self.collection.query(expr=str(exprr), output_fields=["id", "embedding"])
            res1 = self.collection.query(expr=exprr, output_fields=["news_id", "embeddings"])
            # print(res1)
            return res1
        except Exception as e:
            print(f"Failed to search in Milvus: {e}")
            return False

    def count(self, collection_name):
        """
        获取表（集合）数量
        """
        try:
            self.change_collection(collection_name)
            num = self.collection.num_entities
            print(f"Successfully get the num:{num} of the collection:{collection_name}")
            return num
        except Exception as e:
            print(f"Failed to count vectors in Milvus: {e}")
            return False

    def time_travel(self, collection_name, vectors, top_k):
        """
        根据 实时时间戳查询数据
        """
        # 时间旅行搜索
        # 为所有数据插入或删除操作维护一个时间线，并采用时间戳机制。可以在搜索或查询中指定时间戳，以检索过去特定时间点的数据
        # https://milvus.io/docs/v2.1.x/timetravel.md
        self.change_collection(collection_name)
        now_datetime = datetime.datetime.now()
        now_timestamp = utility.mkts_from_datetime(now_datetime)
        print(now_timestamp)
        search_param = {
            "data": vectors,
            "anns_field": "embedding",
            "param": {"metric_type": METRIC_TYPE},
            "limit": top_k,
            "travel_timestamp": now_timestamp,
        }
        res = self.collection.search(**search_param)
        # print(res)
        return res


if __name__ == '__main__':
    config = {"host": "10.108.17.241", "port": "19530"}
    mc = MilvusClient(config=config)
    b = mc.has_collection('new_milvus')
    print(b)
    if b:
        # mc.change_collection('news_test')
        # m = mc.count('new_milvus')
        # print(m)
        # m = mc.search('news_test',**{"1":2})
        # m = mc.search_one_id_embedding('test_search',"5e7eceeb33fedd77fac10d8b4c3e3d65")
        m = mc.search_varchar_list_embedding('news_test', 'news_id',["5e7eceeb33fedd77fac10d8b4c3e3d65"],["news_id", "embeddings"])
        print(m)