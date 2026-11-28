import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.graph import build_transition
from src.rank import pagerank
from src.trust import hop_distance, inverse_pagerank, spam_mass, trustrank


def test_trustrank_propagates_scores_from_trusted_seed():
    """신뢰 점수가 링크를 따라 줄어들고 고립된 호스트에는 도달하지 않는지 확인한다."""
    A = csr_matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    M = build_transition(A)

    scores, info = trustrank(M, [0], beta=0.5, tol=1e-12, return_info=True)

    # 링크를 한 번 지날 때마다 점수가 절반이므로 0~2번의 점수 비는 4:2:1이다.
    np.testing.assert_allclose(scores, [4 / 7, 2 / 7, 1 / 7, 0.0])
    assert info["converged"]


def test_spam_mass_matches_definition():
    """(PR − TR) / PR 정의대로 계산하고, seed는 음수·seed에서 먼 호스트는 큰 값이 되는지 확인한다."""
    A = csr_matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])
    M = build_transition(A)
    pr = pagerank(M, beta=0.85, tol=1e-12)
    tr = trustrank(M, [0], beta=0.85, tol=1e-12)

    mass = spam_mass(pr, tr)

    np.testing.assert_allclose(mass, (pr - tr) / pr)
    assert mass[0] < 0            # seed는 TR이 PR보다 큼
    assert mass[3] == 1.0         # seed에서 닿지 않는 호스트는 TR = 0 → 전부 spam mass


def test_spam_mass_rejects_bad_input():
    with pytest.raises(ValueError):
        spam_mass([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        spam_mass([0.0, 1.0], [0.5, 0.5])


def test_hop_distance_counts_links_from_nearest_seed():
    """가장 가까운 seed 기준으로 hop을 세고, 닿지 않는 호스트는 최댓값 + 1인지 확인한다."""
    A = csr_matrix([
        [0, 5, 0, 0, 0],   # 0 → 1 (링크 수 5여도 1 hop)
        [0, 0, 1, 0, 0],   # 1 → 2
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],   # 3 → 2
        [0, 0, 0, 0, 0],   # 4: 고립
    ])

    np.testing.assert_array_equal(hop_distance(A, [0]), [0, 1, 2, 3, 3])
    np.testing.assert_array_equal(hop_distance(A, [0, 3]), [0, 1, 1, 0, 2])


def test_inverse_pagerank_ranks_hosts_by_out_links():
    """뒤집은 그래프의 PageRank와 같고, 링크를 거는 쪽(출발)이 높은 점수를 받는지 확인한다."""
    A = csr_matrix([
        [0, 1, 1, 1],   # 0은 세 곳에 링크를 건다
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    scores, info = inverse_pagerank(A, beta=0.85, tol=1e-12, return_info=True)

    np.testing.assert_allclose(scores, pagerank(build_transition(A.T.tocsr()), beta=0.85, tol=1e-12))
    assert scores[0] == scores.max()
    assert info["converged"]
