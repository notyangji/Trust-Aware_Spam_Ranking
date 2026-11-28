"""TrustRank seed 선정 전략 (실험 E2·E6용). 모든 전략은 학습용 NONSPAM 안에서만 고른다.

기본 무작위 전략은 src/seeds.py의 select_trusted_seeds를 그대로 쓰고, 여기에는 그 밖의 전략만 둔다.

    seed_pool(ds, strategy, domains)        후보 거르기: domain·domain_high_pr(신뢰 도메인) / commercial(co.uk) / 전체
    random_pool_seeds(pool, n, random_seed) 후보를 섞고 앞에서 n개 (select_trusted_seeds와 같은 추첨 방식)
    high_pagerank_seeds(pool, pr, n)        PageRank 상위 n개 (강의: top k pages by PageRank)
    svd_components(adjacency, k)            인접 행렬의 truncated SVD 왼쪽 특이벡터 k개 (특이값 큰 순)
    svd_seeds(U, pool, n)                   SVD 축마다 대표 host 하나씩 → seed n개

domain_high_pr(신뢰 도메인 후보 중 PageRank 상위)는 미리 정한 전략이 아니라, validation 결과(E2·E6)를
보고 추가한 탐색 결과다. 보고서·final test에서 사전에 정한 비교와 구분해 표시한다.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from src.common.schema import NONSPAM
from src.graph import domain_of

STRATEGIES = ("random", "domain", "commercial", "high_pr", "svd", "domain_high_pr")


def seed_pool(ds, strategy, domains):
    """학습용 NONSPAM 중 전략에 맞는 host ID를 오름차순으로 돌려준다.

    domain·domain_high_pr: domain_of(host)가 domains(예: ac.uk, gov.uk)에 속하는 host
    commercial: co.uk host
    random·high_pr·svd: 학습용 NONSPAM 전체
    """
    ids = ds.train_idx[ds.labels[ds.train_idx] == NONSPAM]
    if strategy in ("random", "high_pr", "svd"):
        return np.sort(ids)
    dom = np.array([domain_of(ds.graph.hostnames[i]) for i in ids])
    if strategy in ("domain", "domain_high_pr"):
        return np.sort(ids[np.isin(dom, list(domains))])
    if strategy == "commercial":
        return np.sort(ids[dom == "co.uk"])
    raise ValueError(f"알 수 없는 seed 전략: {strategy} (가능: {STRATEGIES})")


def _check_n(pool, n, strategy):
    if n > len(pool):
        raise ValueError(f"{strategy} 후보 {len(pool)}개에서 {n}개를 고를 수 없습니다.")


def random_pool_seeds(pool, n, random_seed):
    """후보 순서를 먼저 섞고 앞에서 n개 (n이 커져도 작은 집합을 포함). 오름차순 ID."""
    _check_n(pool, n, "random")
    return np.sort(np.random.default_rng(int(random_seed)).permutation(pool)[:n])


def high_pagerank_seeds(pool, pagerank_scores, n):
    """후보 중 PageRank가 높은 n개. 같은 값이면 ID가 작은 쪽을 먼저 (결정적).

    host ID는 hostname 알파벳 순이므로, 동점이 경계에 걸리면 알파벳이 앞선 host가 뽑힌다.
    실제 데이터에서는 train NONSPAM 전체 후보의 상위 1000개, domain 후보의 상위 100개 안에서
    경계 동점이 없음을 확인했다. PageRank 동점은 링크를 받지 않는 host(최솟값)처럼 아래쪽에 몰린다.
    """
    _check_n(pool, n, "high_pr")
    pool = np.sort(pool)
    order = np.argsort(-np.asarray(pagerank_scores)[pool], kind="stable")
    return np.sort(pool[order[:n]])


def svd_components(adjacency, k, random_state=0):
    """링크 유무(0/1) 인접 행렬의 truncated SVD에서 왼쪽 특이벡터 U (N × k), 특이값 큰 순.

    A[i, j] = i → j 이므로 왼쪽 특이벡터는 "어떤 host들에 링크를 거는가"(out-link 패턴)를 요약한다.
    trust는 out-link를 따라 퍼지므로 seed 선정에는 왼쪽 벡터를 쓴다.
    random_state로 ARPACK 시작 벡터를 고정해 재현한다.
    """
    A = csr_matrix(adjacency, dtype=np.float64, copy=True)
    A.data[:] = 1.0
    U, s, _ = svds(A, k=k, random_state=random_state)
    order = np.argsort(-s)                      # svds는 특이값을 오름차순으로 준다
    return U[:, order], s[order]


def svd_seeds(U, pool, n):
    """SVD 축 0, 1, 2, … 순서로, 축마다 후보 중 |U[:, j]|가 가장 큰 host를 하나씩 고른다.

    이미 고른 host는 건너뛰고 그 축의 다음 host를 고른다. 축 j개에서 seed j개가 나오므로
    n은 U의 열 수 이하여야 한다. 앞쪽 축부터 고르므로 n이 커져도 작은 집합을 포함한다.
    """
    _check_n(pool, n, "svd")
    if n > U.shape[1]:
        raise ValueError(f"SVD 축 {U.shape[1]}개로 seed {n}개를 고를 수 없습니다.")
    pool = np.sort(pool)
    chosen, taken = [], set()
    for j in range(n):
        order = np.argsort(-np.abs(U[pool, j]), kind="stable")
        pick = next(int(pool[i]) for i in order if int(pool[i]) not in taken)
        chosen.append(pick)
        taken.add(pick)
    return np.sort(np.array(chosen, dtype=np.int64))
