import numpy as np
from pymilvus import MilvusClient
import time

from sift import SiftDataset
from client import get_client

client = get_client()

collection_name = 'sift'

dataset = SiftDataset(collection_name, collection_name, with_base=False)

search_vector_id = 0

start_time = time.time()
res = client.search(
    collection_name=collection_name,
    data=[dataset.query[search_vector_id,:]],
    limit=100,
    output_fields=["id"],
    filter="attribute < 10"
)
end_time = time.time()

print(f'Search time: {end_time - start_time}')
print(dataset.ground_truth[search_vector_id])
print([x['id'] for x in res[0]])