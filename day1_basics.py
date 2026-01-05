from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Sequence, Tuple


def normalize(xs: Sequence[float]) -> List[float]:
    """
    Z-score normalize a 1D list: z_i = (x_i - mean) / std.

    Use *population* std:
        var = sum((x-mean)^2)/n
        std = sqrt(var)

    Edge case: if std == 0 (all values identical), return a list of zeros.
    """
    if len(xs) == 0:
        return []
    n=len(xs)
    mean=sum(xs)/n
    var=sum((x-mean)**2 for x in xs)/n
    std=math.sqrt(var)
    if std==0:
        return [0.0 for _ in xs]
    return [(x - mean) / std for x in xs]


def train_test_split_indices(
    n: int, #nr of samples in dataset
    test_ratio: float, #proportion of dataset to include in test split
    seed: int = 0, #The seed is the starting state of that algorithm. Same seed → same shuffle/split → reproducible, Different seed → different shuffle/split → different train/test sets
) ->Tuple[List[int], List[int]]: #returns two lists of indices: train_indices and test_indices
    """
    Return (train_indices, test_indices) for dataset size n.

    Requirements:
    - deterministic given seed
    - shuffled indices
    - no duplicates
    - lengths add up to n
    - test set size = int(round(n * test_ratio)) but clamped to [1, n-1]
    """
    if n <= 1:
        raise ValueError("n must be >= 2")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1 (exclusive)")

    # TODO: create indices [0..n-1]
    # TODO: shuffle with random.Random(seed)
    # TODO: compute test_size with rounding and clamp
    # TODO: split into test and train
    indices = list(range(n)) #create indices [0..n-1]
    rng = random.Random(seed)
    rng.shuffle(indices)
    test_size = int(round(n * test_ratio)) #compute test_size with rounding
    test_size = max(1, min(n - 1, test_size)) #clamp test_size to [1, n-1]

    test_idx = indices[:test_size] #from the start to test_size without including test_size
    train_idx = indices[test_size:]
    return train_idx, test_idx


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Root Mean Squared Error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Inputs must be non-empty")

    # TODO: compute mean squared error then sqrt
    n = len(y_true) #number of samples
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n 
    return math.sqrt(mse)


def _demo() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    zs = normalize(xs)
    print("xs:", xs)
    print("zs:", [round(z, 4) for z in zs])

    train_idx, test_idx = train_test_split_indices(n=10, test_ratio=0.2, seed=42)
    print("train:", train_idx)
    print("test:", test_idx)

    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.5, 2.0]
    print("rmse:", round(rmse(y_true, y_pred), 6))


if __name__ == "__main__":
    _demo()
