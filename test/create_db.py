import numpy as np
from pymilvus import MilvusClient
import time

from sift import SiftDataset

client = MilvusClient("milvus.db")

def create_collection(name):
    print(f'Creating collection {name}')

    dataset = SiftDataset(name, name)

    client.create_collection(
        collection_name=name,
        dimension=dataset.dim,
    )

    CHUNK_SIZE = 10000

    for i in range(0, dataset.num_base_vecs, CHUNK_SIZE):
        print(f'{i}/{dataset.num_base_vecs}')
        data = [{'id': i + offset, 'vector': list(vector)} for offset, vector in enumerate(iter(dataset.base[i:i+CHUNK_SIZE]))]
        res = client.insert(collection_name=name, data=data)

if __name__ == '__main__':
    create_collection('siftsmall')
    create_collection('sift')