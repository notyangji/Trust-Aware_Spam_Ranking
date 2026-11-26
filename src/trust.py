import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.ppr import personalized_pagerank
from src.seeds import validate_seed_ids


def trustrank(M, seed_ids, **rank_options):
    """Trusted seed에 같은 비중을 주고 링크를 따라 신뢰 점수를 전파한다.

    M은 build_transition()이 만든 전이 행렬이다.
    seed_ids는 select_trusted_seeds()가 고른 원본 호스트 ID다.
    rank_options는 beta, tol, return_info 같은 공통 랭킹 옵션이다.
    호스트 ID 순서로 신뢰 점수를 반환하고, return_info=True이면 수렴 정보도 반환한다.
    """
    return personalized_pagerank(M, seed_ids, **rank_options)


def spam_mass(pagerank_scores, trust_scores):
    """PageRank 중 trusted seed에서 오지 않은 몫의 비율 (PR − TR) / PR을 계산한다.

    pagerank_scores는 균등 teleport PageRank, trust_scores는 같은 β로 계산한 TrustRank다.
    두 점수는 모두 합이 1인 길이 N 배열이어야 한다.
    값이 클수록 spam 의심이다 (higher_is_spam=True). seed 근처처럼 TR이 PR보다 크면 음수가 된다.
    균등 teleport PageRank는 모든 호스트에서 (1 − β) / N 이상이므로 0으로 나누는 경우는 없다.
    """
    pr = np.asarray(pagerank_scores, dtype=np.float64)
    tr = np.asarray(trust_scores, dtype=np.float64)
    if pr.shape != tr.shape or pr.ndim != 1:
        raise ValueError("PageRank와 TrustRank 점수는 길이가 같은 1차원 배열이어야 합니다.")
    if np.any(pr <= 0):
        raise ValueError("PageRank 점수가 0 이하인 호스트가 있습니다. 균등 teleport PageRank를 넣어야 합니다.")
    return (pr - tr) / pr


def hop_distance(adjacency, seed_ids):
    """seed에서 링크(출발 → 도착)를 몇 번 따라가야 각 호스트에 닿는지 계산한다.

    adjacency는 HostGraph.adjacency (A[i, j] = i → j 링크 수)이고, 링크 수와 상관없이 한 번에 1 hop으로 센다.
    seed 자신은 0이다. 어떤 seed에서도 닿지 않는 호스트는 (닿은 호스트 중 최댓값 + 1)로 둔다.
    값이 클수록 spam 의심이다 (higher_is_spam=True). 반환값은 길이 N의 float64 배열이다.
    """
    n = adjacency.shape[0]
    seed_ids = validate_seed_ids(seed_ids, n)
    hops = csr_matrix(adjacency, dtype=np.float64, copy=True)
    hops.data[:] = 1.0
    dist = dijkstra(hops, directed=True, indices=seed_ids, unweighted=True, min_only=True)
    reached = np.isfinite(dist)
    dist[~reached] = dist[reached].max() + 1
    return dist
