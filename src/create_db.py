import time
from typing import List, Optional
from logging import Logger

import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus import utility
from tqdm import tqdm

from sift import Dataset, load_sift_1m
from client import get_client
from partitioner import Partitioner, RangePartitioner
from attributes import uniform_attributes, uniform_attributes_basic
from utils import Timer


class Creator():
    def __init__(self, partitioner: Optional[Partitioner] = None, num_auto_partitions: Optional[int] = None, datatype: DataType = DataType.FLOAT_VECTOR, logger: Logger = None):
        self.client = get_client()
        self.partitioner = partitioner
        self.num_auto_partitions = num_auto_partitions
        self.datatype = datatype
        self.logger: Logger = logger.getChild('Creator') if logger else None

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
        
        self.logger.info(f'Flushing {name}')
        with Timer() as t:
            self.client.flush(name)
        self.logger.info(f'Flushed {name} in {t.duration:.2f}s')

        index_params = MilvusClient.prepare_index_params()

        self.logger.info(f'Adding index {index_type} for {name}')
        with Timer() as t:
            index_params.add_index(
                field_name="vector",
                metric_type="L2",
                index_type=index_type,
                index_name="vector_index",
            )
        self.logger.info(f'Index {index_type} added in {t.duration:.2f}s')

        self.logger.info(f'Creating index {index_type} for {name}')
        with Timer() as t:
            self.client.create_index(
                collection_name=name,
                index_params=index_params,
                sync=True
            )
        self.logger.info(f'Index {index_type} created in {t.duration:.2f}s')
        
        # self.logger.info(f'Waiting for index building to complete for {name}')
        # utility.wait_for_index_building_complete(
        #     collection_name=name,
        #     index_name="vector_index",
        # )
        
        # def wait_index():
        #     while True:
        #         index_state = utility.index_building_progress(
        #             collection_name=name,
        #             index_name="vector_index",
        #         )
        #         if index_state.get("pending_index_rows", -1) == 0:
        #             break
        #         self.logger.info(f'Waiting for index building to complete: {index_state}')
        #         time.sleep(2)

        # wait_index()

        self.logger.info(f'Loading {name}')
        with Timer() as t:
            self.client.load_collection(name)
        self.logger.info(f'Loaded {name} in {t.duration:.2f}s')
        
        self.logger.info(f'Flushing {name}')
        with Timer() as t:
            self.client.flush(name)
        self.logger.info(f'Flushed {name} in {t.duration:.2f}s')
        
        if self.partitioner is not None:
            for partition_name in self.partitioner.partition_names:
                is_loaded = self.client.get_load_state(name, partition_name)
                self.logger.debug(f'Partition {partition_name}: {is_loaded["state"]}')
        # self.client.load_partitions(name, self.partitioner.partition_names)
        

if __name__ == '__main__':
    creator = Creator(RangePartitioner([(0, 100), (101, 1000)]))
    
    name = 'sift'
    dataset = load_sift_1m('../data/sift')
    creator.create_collection_schema(name)
    creator.populate_collection(name, dataset, uniform_attributes_basic(dataset.num_base_vecs))
