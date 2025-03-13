import numpy as np
from pymilvus import MilvusClient
import time

from sift import SiftDataset

client = MilvusClient("milvus.db")

dataset = SiftDataset('siftsmall', 'siftsmall', with_base=False)

search_vector_id = 10

start_time = time.time()
res = client.search(
    collection_name="siftsmall",
    data=[dataset.query[search_vector_id,:]],
    limit=1,
    output_fields=["vector"],
)
end_time = time.time()

print(f'Search time: {end_time - start_time}')
print(dataset.ground_truth[search_vector_id][0])
print(res[0][0]['id'])