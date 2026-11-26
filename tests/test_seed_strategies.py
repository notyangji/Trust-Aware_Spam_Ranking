import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.seed_strategies import high_pagerank_seeds, random_pool_seeds, svd_components, svd_seeds


def two_blocks():
    """서로 링크하는 무리 두 개(0–3번, 4–6번)와 고립된 7번 host."""
    A = np.zeros((8, 8))
    A[:4, :4] = 1
    A[4:7, 4:7] = 1
    np.fill_diagonal(A, 0)
    return csr_matrix(A)


def test_random_pool_seeds_is_reproducible_and_nested():
    pool = np.arange(10, 30)
    a = random_pool_seeds(pool, 5, random_seed=3)
    assert np.array_equal(a, random_pool_seeds(pool, 5, random_seed=3))
    assert set(a) <= set(pool) and len(set(a)) == 5
    assert set(a) <= set(random_pool_seeds(pool, 10, random_seed=3))   # n이 커져도 작은 집합 포함
    with pytest.raises(ValueError):
        random_pool_seeds(pool, 21, random_seed=0)


def test_high_pagerank_seeds_picks_top_scores_in_pool():
    pr = np.array([0.5, 0.1, 0.3, 0.05, 0.05])
    assert list(high_pagerank_seeds(np.array([1, 2, 3, 4]), pr, 2)) == [1, 2]   # 0번은 후보 밖
    assert list(high_pagerank_seeds(np.array([3, 4]), pr, 1)) == [3]           # 동점이면 작은 ID


def test_svd_components_sorted_and_seeds_cover_both_blocks():
    U, s = svd_components(two_blocks(), k=2)
    assert U.shape == (8, 2) and s[0] >= s[1]
    seeds = svd_seeds(U, np.arange(8), 2)
    # 축 0은 큰 무리(0–3), 축 1은 작은 무리(4–6)를 대표 → 무리마다 seed 하나
    assert len(set(seeds) & {0, 1, 2, 3}) == 1 and len(set(seeds) & {4, 5, 6}) == 1


def test_svd_seeds_respects_pool_nesting_and_limits():
    U, _ = svd_components(two_blocks(), k=3)
    pool = np.array([1, 2, 5, 6, 7])
    one, two = svd_seeds(U, pool, 1), svd_seeds(U, pool, 2)
    assert set(two) <= set(pool) and len(set(two)) == 2 and set(one) <= set(two)
    with pytest.raises(ValueError):
        svd_seeds(U, pool, 4)              # 축이 3개뿐
