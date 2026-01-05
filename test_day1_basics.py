import math
import pytest

from day1_basics import normalize, train_test_split_indices, rmse

def test_normalize_empty():
    assert normalize([]) == []

def test_normalize_constant_returns_zeros():
    zs = normalize([5.0, 5.0, 5.0])
    assert zs == [0.0, 0.0, 0.0]

def test_normalize_has_zero_mean_and_unit_variance_population():
    xs = [1.0, 2.0, 3.0, 4.0]
    zs = normalize(xs)
    mean = sum(zs) / len(zs)
    var = sum((z - mean) ** 2 for z in zs) / len(zs)
    std=math.sqrt(var)
    
    assert abs(mean) < 1e-9
    assert abs(var - 1.0) < 1e-9
    assert abs(std - 1.0) < 1e-9

def test_split_indices_sizes_and_no_overlap():
    n = 10
    test_ratio = 0.2
    seed = 0

    train_idx, test_idx = train_test_split_indices(n, test_ratio, seed)

    assert len(train_idx) + len(test_idx) == n # Ensure all indices are accounted for
    assert len(test_idx) == int(round(n * test_ratio)) # Ensure test set size is correct
    assert set(train_idx).isdisjoint(set(test_idx)) # Ensure no overlap
    assert len(set(train_idx)) == len(train_idx) # Ensure no duplicates
    assert len(set(test_idx)) == len(test_idx) # Ensure no duplicates

def test_split_indices_deterministic_with_seed():
    a = train_test_split_indices(n=10, test_ratio=0.3, seed=123)
    b = train_test_split_indices(n=10, test_ratio=0.3, seed=123)
    assert a == b

def test_rmse_basic():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    result = rmse(y_true, y_pred)
    expected = math.sqrt(((0.0)**2 + (0.5)**2 + (-1.0)**2) / 3)
    assert abs(result - expected) < 1e-12

def test_rmse_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        rmse([1.0], [1.0, 2.0])
