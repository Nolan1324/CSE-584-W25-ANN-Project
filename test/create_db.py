import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
import time
from tqdm import tqdm

from sift import SiftDataset
from client import get_client
from partitioner import Partitioner


class Creator():
    def __init__(self, partitioner=None):
        self.client = get_client()
        self.partitioner = partitioner

    def create_collection_schema(self, name):
        if self.client.has_collection(name):
            self.client.drop_collection(name)

        id_schema = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
        )
        attrib_schema = FieldSchema(
            name="attribute",
            dtype=DataType.INT64,
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

        self.client.create_collection(
            collection_name=name,
            schema=schema
        )

        self.partitioner.add_partitions_to_collection(self.client, name)

        index_params = MilvusClient.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            metric_type="L2",
            index_type="HNSW",
            index_name="vector_index",
        )

        self.client.create_index(
            collection_name=name,
            index_params=index_params,
            sync=True
        )

        self.client.load_collection(name)


    def create_and_populate_collection(self, name):
        print(f'Creating collection {name}')

        np.random.seed(584)

        self.create_collection_schema(name)

        dataset = SiftDataset(name, name)

        # CHUNK_SIZE = 10000
        # for i in range(0, dataset.num_base_vecs, CHUNK_SIZE):
        #     print(f'{i}/{dataset.num_base_vecs}')
            
        #     attrs = np.random.randint(0, 1000, size=CHUNK_SIZE)
        #     data = [{'id': i + offset, 'vector': list(vector), 'attribute': attrs[offset]} 
        #             for offset, vector in enumerate(iter(dataset.base[i:i+CHUNK_SIZE]))]
        #     res = client.insert(collection_name=name, data=data)

        # attrs = np.random.randint(0, 1000, size=dataset.num_base_vecs)
        for i in tqdm(range(0, dataset.num_base_vecs)):
            attr = int(1000 * (i / dataset.num_base_vecs))
            data = {'id': i, 'vector': list(dataset.base[i]), 'attribute': attr}
            res = self.client.insert(
                collection_name=name,
                data=data,
                partition_name=self.partitioner.query_partition(attr)
                    if self.partitioner is not None else None  
            )

if __name__ == '__main__':
    Creator(Partitioner([(0, 100), (101, 1000)])) \
        .create_and_populate_collection('siftsmall')
    # create_and_populate_collection('sift')