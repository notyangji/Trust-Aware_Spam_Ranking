import numpy as np
from scipy.sparse import csr_matrix

from src.graph import build_transition
from src.ppr import personalized_pagerank
from src.rank import pagerank


def test_ppr_with_all_hosts_matches_pagerank():
    A = csr_matrix([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])
    M = build_transition(A)

    ppr_scores = personalized_pagerank(M, [0, 1, 2])
    pr_scores = pagerank(M)

    np.testing.assert_allclose(ppr_scores, pr_scores)


def test_ppr_gives_zero_score_to_unreachable_host():
    A = csr_matrix([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    M = build_transition(A)

    scores = personalized_pagerank(M, [0])

    assert scores[0] > 0
    assert scores[1] > 0
    assert scores[2] == 0.0
