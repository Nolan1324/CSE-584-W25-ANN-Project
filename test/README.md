## Milvus setup

- `pip install pymilvus`
- Run `./standalone_embed.sh` to start Docker container. Check it is running at `http://127.0.0.1:9091/webui/`

## Use

- Download datasets with `download_dataset.sh`
- Create the database collections with `create_db.py` (takes a while for SIFT, so only run once ideally)
- Conduct test search with `search.py`

