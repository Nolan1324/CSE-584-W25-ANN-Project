from client import get_client

client = get_client()

res = client.list_partitions(
    collection_name="siftsmall"
)
print(res)

res = client.get(
    ids=[1100],
    collection_name="siftsmall",
    partition_names=['range_101_1000'],
    output_fields=["attribute"],
)

print(res)
