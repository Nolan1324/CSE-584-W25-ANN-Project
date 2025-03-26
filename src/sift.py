from os import PathLike
from pathlib import Path
from typing import Literal

from fvecs import fvecs_read, ivecs_read


class SiftDataset():
    def __init__(self, directory: PathLike, prefix: str, with_base=True):
        path = Path(directory)
        def read(suffix: str, ext: Literal['fvecs', 'ivecs']):
            file_path = path / f'{prefix}_{suffix}.{ext}'
            if ext == 'ivecs':
                data = ivecs_read(file_path)
            else:
                data = fvecs_read(file_path)
            return data

        if with_base:
            self.base = read('base', 'fvecs')
            self.num_base_vecs = self.base.shape[0]
        self.ground_truth = read('groundtruth', 'ivecs')
        self.learn = read('learn', 'fvecs')
        self.query = read('query', 'fvecs')
        self.dim = self.query.shape[1]
    
    
if __name__ == '__main__':
    dataset = SiftDataset('siftsmall', 'siftsmall')
