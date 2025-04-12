import json
from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt
from utils import WORKLOAD_PATH
from pymilvus import DataType
from predicates import Not, Predicate, And, Atomic, Operator

class QueryEntry:
    def __init__(self, query, predicates, closest_ids, closest_scores):
        self.query = query
        self.conditions = predicates
        self.closest_ids = closest_ids
        self.closest_scores = closest_scores

    # def has_range_condition(self):
    #     """Recursively check for presence of a range condition."""
    #     def check_condition(cond):
    #         if isinstance(cond, dict):
    #             if "range" in cond:
    #                 return True
    #             for key, value in cond.items():
    #                 if isinstance(value, list):
    #                     return any(check_condition(item) for item in value)
    #                 elif isinstance(value, dict):
    #                     if check_condition(value):
    #                         return True
    #         return False

    #     return check_condition(self.conditions)


class Workload:
    def __init__(self):
        self.entries = []

    def load_from_jsonl(self, filepath: PathLike, num_queries: int):
        """Load and filter queries from a JSONL file based on 'range' condition."""
        predicates = []
        with open(Path(filepath), "r", encoding="utf-8") as file:
            for line in file:
                if (num_queries == 0):
                    break
                num_queries -= 1
                try:
                    data = json.loads(line)
                    
                    for attr in data['conditions']['and'][0].keys():
                        op = list(data['conditions']['and'][0][attr].keys())[0]
                        if (op != 'range'):
                            continue
                        range = data['conditions']['and'][0][attr][op]
                        GTE_pred = Atomic(attr, Operator.GTE, range['gt'] + 1)
                        LTE_pred = Not(Atomic(attr, Operator.LTE, range['lt'] - 1))
                        predicates.append(GTE_pred)
                        predicates.append(LTE_pred)
                        entry = QueryEntry(
                            query=data["query"],
                            predicates=[GTE_pred, LTE_pred],
                            closest_ids=data["closest_ids"],
                            closest_scores=data["closest_scores"]
                        )
                        self.entries.append(entry)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Skipping line due to error: {e}")
            return predicates

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]
    
if __name__ == "__main__":
    workload = Workload()
    predicates = workload.load_from_jsonl(WORKLOAD_PATH / "tests.jsonl", 1000000)

    print(f"Loaded {len(workload)} queries with range conditions.")
    # for entry in workload:
    #     print(entry.query, entry.conditions)