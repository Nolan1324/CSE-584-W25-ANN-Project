import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
import time

from sift import SiftDataset
from client import get_client

client = get_client()

def create_collection_schema(name):
    if client.has_collection(name):
        client.drop_collection(name)

    id_schema = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
    )
    attrib_schema = FieldSchema(
        name="attribute",
        dtype=DataType.INT32,
    )
    vector_schema = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=128,
    )
    schema = CollectionSchema(
        fields=[id_schema, vector_schema, attrib_schema],
        description="Test attributes on SIF",
        enable_dynamic_field=False,
    )

    client.create_collection(
        collection_name=name,
        schema=schema
    )

    index_params = MilvusClient.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        metric_type="L2",
        index_type="FLAT",
        index_name="vector_index",
    )

    client.create_index(
        collection_name=name,
        index_params=index_params,
        sync=True
    )

    client.load_collection(name)


def create_and_populate_collection(name):
    print(f'Creating collection {name}')

    np.random.seed(584)

    create_collection_schema(name)

    dataset = SiftDataset(name, name)

    CHUNK_SIZE = 10000
    for i in range(0, dataset.num_base_vecs, CHUNK_SIZE):
        print(f'{i}/{dataset.num_base_vecs}')
        
        attrs = np.random.randint(0, 1000, size=CHUNK_SIZE)
        data = [{'id': i + offset, 'vector': list(vector), 'attribute': attrs[offset]} 
                for offset, vector in enumerate(iter(dataset.base[i:i+CHUNK_SIZE]))]
        res = client.insert(collection_name=name, data=data)

if __name__ == '__main__':
    create_and_populate_collection('siftsmall')
    # create_and_populate_collection('sift')