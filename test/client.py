from pymilvus import MilvusClient

def get_client():
    # return MilvusClient("milvus.db")
    return MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus"
    )