"""비교용 baseline 점수. PageRank 계열과 같은 형식(길이 N float64)으로 만들어 C가 같은 방식으로 평가한다.

점수 방향 (C와 합의 전, file_spec "합의 필요" 1번):
- random_scores: 방향 없음 (무작위 기준선)
- indegree_scores: 클수록 spam 의심 (들어오는 링크가 적은 host일수록 높음)
- 참고로 pagerank는 클수록 중요(= spam 아님 쪽)라 방향이 반대다.
"""

from __future__ import annotations

import numpy as np


def random_scores(n: int, seed: int) -> np.ndarray:
    """길이 n, [0, 1) 균등 난수 (float64). seed는 cfg["baselines"]["random_seed"]."""
    return np.random.default_rng(seed).random(n)


def indegree_scores(A) -> np.ndarray:
    """in-degree baseline: −log1p(in_deg), 길이 N float64. 클수록 spam 의심.

    in_deg = host j로 링크를 거는 host 수 (링크 수가 아니라 host 수, 이진).
    A[i, j] = host i → host j의 링크 수 (행 = 출발, 열 = 도착)이므로 열 방향으로 센다.
    """
    in_deg = np.asarray((A > 0).sum(axis=0)).ravel()
    return -np.log1p(in_deg).astype(np.float64)
