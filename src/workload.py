from __future__ import annotations

from dataclasses import dataclass, field
import json
from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt
from utils import WORKLOAD_PATH
from predicates import Not, Predicate, And, Atomic, Operator, Or


def generate_discrete_exponential(n: int, ratio: float, shuffled: bool = True) -> npt.NDArray[np.float32]:
    p = np.exp(-ratio * np.arange(n))
    p /= np.sum(p)
    return np.random.default_rng().permutation(p) if shuffled else p


def sample_discrete_exponential(n: int, mean: float) -> npt.NDArray[np.float32]:
    ratio = np.random.default_rng().normal(loc=mean, scale=0.1)
    ratio = np.clip(ratio, 0, 1)
    return generate_discrete_exponential(n, ratio, shuffled=True)


@dataclass
class QueryEntry:
    query: npt.NDArray = field(default=None)
    predicates: list[Predicate] = field(default_factory=list)
    closest_ids: npt.NDArray[np.int64] = field(default=None)
    closest_scores: npt.NDArray[np.float32] = field(default=None)

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
    def __init__(self, entries: list[QueryEntry] = None):
        self.entries = entries or []
    
    @classmethod
    def create_synthetic_workload(cls, attributes: list[str]) -> list[Predicate]:
        n_attributes = len(attributes)
        query_attribute_distribution = sample_discrete_exponential(n_attributes, 0.25)
        query_selectivity_distributions = {attribute: sample_discrete_exponential(9, 0.25) for attribute in attributes}
        query_length_distribution = sample_discrete_exponential(n_attributes, 0.25)
        query_type_distribution = [0, 1/3, 1/3, 1/3]
        query_type_distribution = np.array(query_type_distribution) / np.sum(query_type_distribution)
        return (
            attributes,
            query_attribute_distribution,
            query_selectivity_distributions,
            query_length_distribution,
            query_type_distribution,
        )
    
    @classmethod
    def sample_synthetic_workload(
        cls,
        n_samples: int,
        attributes: list[str],
        query_attribute_distribution: npt.NDArray[np.float32],
        query_selectivity_distributions: dict[str, npt.NDArray[np.float32]],
        query_length_distribution: npt.NDArray[np.float32],
        query_type_distribution: npt.NDArray[np.float32],
    ) -> list[Predicate]:
        entries: list[Predicate] = []
        rng = np.random.default_rng()
        
        percentiles = {attribute: np.arange(100, 1000, 100) for attribute in attributes}
        
        def sampe_atomic_predicate(attribute: str) -> Atomic:
            selectivity = rng.choice(percentiles[attribute], p=query_selectivity_distributions[attribute])
            return Atomic(attr=attribute, op=Operator.GTE, value=int(selectivity))
        
        for _ in range(n_samples):
            query_type = rng.choice(["none", "single", "and", "or"], p=query_type_distribution)
            match query_type:
                case "none":
                    predicate = None
                case "single":
                    predicate = sampe_atomic_predicate(rng.choice(attributes, p=query_attribute_distribution))
                case "and":
                    num_predicates = rng.choice(len(query_attribute_distribution), p=query_length_distribution)
                    selected_attributes = rng.choice(attributes, p=query_attribute_distribution, size=num_predicates, replace=False)
                    predicate = And(*[sampe_atomic_predicate(attr) for attr in selected_attributes])
                case "or":
                    num_predicates = rng.choice(len(query_attribute_distribution), p=query_length_distribution)
                    selected_attributes = rng.choice(attributes, p=query_attribute_distribution, size=num_predicates, replace=False)
                    predicate = Or(*[sampe_atomic_predicate(attr) for attr in selected_attributes])
            if predicate is not None:
                entries.append(predicate)
        return entries

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
    print(workload.entries[0].predicates)
    # for entry in workload:
    #     print(entry.query, entry.conditions)
