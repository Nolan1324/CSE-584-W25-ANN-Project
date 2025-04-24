from pymilvus import MilvusClient


def get_client() -> MilvusClient:
    """Return the current MilbusClient."""
    return MilvusClient(uri="http://localhost:19530", token="root:Milvus")
