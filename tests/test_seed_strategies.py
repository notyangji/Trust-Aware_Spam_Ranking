import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.common.schema import NONSPAM, SPAM, UNLABELED, Dataset, HostGraph
from src.seed_strategies import high_pagerank_seeds, random_pool_seeds, seed_pool, svd_components, svd_seeds
from src.seeds import select_trusted_seeds


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


def test_random_pool_seeds_matches_select_trusted_seeds():
    """기본 random 전략(B의 select_trusted_seeds)과 후보 지정 추첨(random_pool_seeds)이 같은 seed를 고르는지.

    E2에서 random은 select_trusted_seeds로, domain·commercial은 random_pool_seeds로 뽑으므로
    두 추첨 방식이 같아야 전략 간 차이가 후보 차이에서만 나온다.
    """
    rng = np.random.default_rng(0)
    n = 300
    labels = np.full(n, UNLABELED, dtype=np.int8)
    labeled = rng.permutation(n)[:240]
    labels[labeled] = np.where(rng.random(240) < 0.2, SPAM, NONSPAM)
    train, val, test = np.sort(labeled[:160]), np.sort(labeled[160:200]), np.sort(labeled[200:])
    in_set = np.zeros(n, dtype=np.int8)
    in_set[labeled[:200]], in_set[labeled[200:]] = 1, 2           # train·val = SET1, test = SET2
    ds = Dataset(HostGraph([f"h{i}.co.uk" for i in range(n)], csr_matrix((n, n))), labels, in_set, train, val, test)
    pool = seed_pool(ds, "random", domains=[])
    for r in range(3):
        for k in (5, 30, 100):
            assert np.array_equal(select_trusted_seeds(ds, n_seeds=k, random_seed=r), random_pool_seeds(pool, k, r))
        # noise가 있으면 NONSPAM 쪽은 noise 0일 때 순서의 앞부분과 같다
        mixed = select_trusted_seeds(ds, n_seeds=20, random_seed=r, contamination_rate=0.25)
        assert set(mixed[labels[mixed] == NONSPAM]) == set(random_pool_seeds(pool, 15, r))
