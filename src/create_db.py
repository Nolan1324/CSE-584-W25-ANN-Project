from typing import List, Optional
import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from tqdm import tqdm

from sift import Dataset, load_sift_1m
from client import get_client
from partitioner import Partitioner, RangePartitioner
from attributes import uniform_attributes, uniform_attributes_example


class Creator():
    def __init__(self, partitioner: Optional[Partitioner] = None, num_auto_partitions: Optional[int] = None, datatype: DataType = DataType.FLOAT_VECTOR):
        self.client = get_client()
        self.partitioner = partitioner
        self.num_auto_partitions = num_auto_partitions
        self.datatype = datatype

        if partitioner is not None and num_auto_partitions is not None:
            raise ValueError('Cannot specify both custom partitioner and auto partitioner at the same time')

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
            is_partition_key=self.num_auto_partitions is not None
        )
        vector_schema = FieldSchema(
            name="vector",
            dtype=self.datatype,
            dim=128,
        )
        schema = CollectionSchema(
            fields=[id_schema, vector_schema, attrib_schema],
            description="Test attributes on SIFT",
            enable_dynamic_field=False,
        )

        self.client.create_collection(
            collection_name=name,
            schema=schema,
            num_partitions=self.num_auto_partitions,
        )

        if self.partitioner is not None:
            self.partitioner.add_partitions_to_collection(self.client, name)


    def populate_collection(self, name: str, dataset: Dataset, attributes: 'np.ndarray[np.int32]', index_type: str = 'HNSW'):
        assert(dataset.num_base_vecs == attributes.shape[0])
        assert(dataset.d == 128)
        
        if self.partitioner is None:
            CHUNK_SIZE = 10000

            for i in tqdm(range(0, dataset.num_base_vecs, CHUNK_SIZE)):
                print(f'{i}/{dataset.num_base_vecs}')
                
                data = [{'id': i + offset, 'vector': list(vector), 'attribute': attributes[i + offset]} 
                        for offset, vector in enumerate(iter(dataset.base[i:i+CHUNK_SIZE]))]
                res = self.client.insert(collection_name=name, data=data)
        else:
            CHUNK_SIZE = 10_000

            def _insert_batch(partition_name: str, batch: List[int]):
                data = [
                    {
                        'id': index,
                        'vector': dataset.base[index],
                        'attribute': attributes[index],
                    }
                    for index in batch
                ]
                res = self.client.insert(collection_name=name, data=data, partition_name=partition_name)

            partition_batches = {partition_name: [] for partition_name in self.partitioner.partition_names}
            for i in tqdm(range(0, dataset.num_base_vecs)):
                partition_name = self.partitioner.get_partition(attributes[i])
                batch = partition_batches[partition_name]
                batch.append(i)

                if len(batch) >= CHUNK_SIZE:
                    _insert_batch(partition_name, batch)
                    batch.clear()
            for partition_name, batch in partition_batches.items():
                if batch:
                    _insert_batch(partition_name, batch)
                    batch.clear()

        index_params = MilvusClient.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            metric_type="L2",
            index_type=index_type,
            index_name="vector_index",
        )

        self.client.create_index(
            collection_name=name,
            index_params=index_params,
            sync=True
        )

        self.client.load_collection(name)
        self.client.flush(name)
        self.client.load_partitions(name, self.partitioner.partition_names)
        

if __name__ == '__main__':
    creator = Creator(RangePartitioner([(0, 100), (101, 1000)]))
    
    name = 'sift'
    dataset = load_sift_1m('../data/sift')
    creator.create_collection_schema(name)
    creator.populate_collection(name, dataset, uniform_attributes_example(dataset.num_base_vecs))
