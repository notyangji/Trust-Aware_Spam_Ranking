import numpy as np
from scipy.sparse import csr_matrix

from src.graph import build_transition
from src.trust import trustrank


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
