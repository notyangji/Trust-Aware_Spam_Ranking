"""엔진(build_transition, power_iteration) 검증. 데이터·네트워크 없이 수 초 안에 끝난다.

test 1–6: 강의 예제 (GUIDE 7절 표), test 7–8: networkx와 교차 확인, 그 밖에 불변식·예외.
node 번호는 0부터. 강의의 y, a, m은 0, 1, 2 / 강의의 1–4번 node는 0–3.
"""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from src.graph import build_transition
from src.rank import pagerank, power_iteration


def make_A(edges, n):
    """edge 목록 → 인접 행렬 A (A[i, j] = i → j). 테스트 예제는 self-loop를 포함할 수 있다."""
    rows, cols = zip(*edges)
    return sp.csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))


def seed_vector(n, seeds):
    s = np.zeros(n)
    s[list(seeds)] = 1.0
    return s


Y, A_, M_ = 0, 1, 2
FLOW = [(Y, Y), (Y, A_), (A_, Y), (A_, M_), (M_, A_)]
SPIDER_TRAP = [(Y, Y), (Y, A_), (A_, Y), (A_, M_), (M_, M_)]
DEAD_END = [(Y, Y), (Y, A_), (A_, Y), (A_, M_)]
TOPIC = [(0, 1), (0, 2), (1, 0), (2, 3), (3, 2)]   # 강의 1→2, 1→3, 2→1, 3→4, 4→3


# ---------- build_transition ----------

def test_transition_lecture_matrix():
    M = build_transition(make_A(FLOW, 3)).toarray()
    np.testing.assert_allclose(M, [[0.5, 0.5, 0.0],
                                   [0.5, 0.0, 1.0],
                                   [0.0, 0.5, 0.0]])


def test_transition_dead_end_column_zero_and_input_unchanged():
    A = sp.csr_matrix(np.array([[0, 3, 1], [0, 0, 0], [2, 0, 0]], dtype=np.float64))
    before = A.toarray().copy()
    Mb = build_transition(A).toarray()
    Mw = build_transition(A, weighted=True).toarray()
    np.testing.assert_allclose(Mb[:, 0], [0, 0.5, 0.5])     # 이진: 1/d_i
    np.testing.assert_allclose(Mw[:, 0], [0, 0.75, 0.25])   # 가중: 링크 수 비례
    assert not Mb[:, 1].any()                               # dead end 열 = 0
    np.testing.assert_array_equal(A.toarray(), before)      # 원본 A 그대로


# ---------- test 1–6: 강의 예제 ----------

def test1_flow_beta1():
    r = pagerank(build_transition(make_A(FLOW, 3)), beta=1.0)
    np.testing.assert_allclose(r, [2 / 5, 2 / 5, 1 / 5], atol=1e-3)


def test2_spider_trap_teleport():
    # 행렬 방향이 뒤집히면 m이 가장 크지 않게 됨 → 방향 오류 검출
    r = pagerank(build_transition(make_A(SPIDER_TRAP, 3)), beta=0.8)
    np.testing.assert_allclose(r, [7 / 33, 5 / 33, 21 / 33], atol=1e-3)
    assert r.argmax() == M_


def test3_dead_end():
    r = pagerank(build_transition(make_A(DEAD_END, 3)), beta=0.8)
    np.testing.assert_allclose(r, [35 / 81, 25 / 81, 21 / 81], atol=1e-3)
    assert r.sum() == pytest.approx(1.0)


def test4_topic_specific():
    r = power_iteration(build_transition(make_A(TOPIC, 4)), teleport=seed_vector(4, {0}), beta=0.8)
    np.testing.assert_allclose(r, [0.294, 0.118, 0.327, 0.261], atol=1e-3)


@pytest.mark.parametrize("seeds, expected", [
    ({0, 1, 2, 3}, [0.13, 0.10, 0.39, 0.36]),
    ({0, 1, 2}, [0.17, 0.13, 0.38, 0.30]),
    ({0, 1}, [0.26, 0.20, 0.29, 0.23]),
])
def test5_topic_seed_sets(seeds, expected):
    # 강의 표는 반올림·버림이 섞여 있어 0.01까지 허용 (GUIDE 7절)
    r = power_iteration(build_transition(make_A(TOPIC, 4)), teleport=seed_vector(4, seeds), beta=0.8)
    np.testing.assert_allclose(r, expected, atol=0.01)


@pytest.mark.parametrize("beta, expected", [
    (0.9, [0.17, 0.07, 0.40, 0.36]),
    (0.7, [0.39, 0.14, 0.27, 0.19]),
])
def test6_topic_beta(beta, expected):
    r = power_iteration(build_transition(make_A(TOPIC, 4)), teleport=seed_vector(4, {0}), beta=beta)
    np.testing.assert_allclose(r, expected, atol=0.01)


# ---------- test 7–8: networkx 교차 확인 ----------

N_RANDOM = 300


@pytest.fixture(scope="module")
def random_graph():
    """무작위 300-node 방향 그래프 (dead end 약 20%, self-loop 없음)."""
    rng = np.random.default_rng(0)
    n = N_RANDOM
    dead = rng.random(n) < 0.2
    edges = set()
    for i in np.flatnonzero(~dead):
        for j in rng.choice(n, size=rng.integers(1, 8), replace=False):
            if j != i:
                edges.add((int(i), int(j)))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    A = make_A(sorted(edges), n)
    seeds = rng.choice(n, size=10, replace=False)
    return G, build_transition(A), seeds


def nx_vector(d, n):
    return np.array([d[i] for i in range(n)])


def test7_networkx_teleport(random_graph):
    G, M, seeds = random_graph
    n = N_RANDOM
    # PageRank: networkx는 dangling=None이면 dead end 질량을 personalization(균등)으로 보냄
    r = pagerank(M, beta=0.85, tol=1e-14, max_iter=1000)
    ref = nx_vector(nx.pagerank(G, alpha=0.85, weight=None, tol=1e-14, max_iter=1000), n)
    np.testing.assert_allclose(r, ref, rtol=0, atol=1e-12)
    # PPR: dead end 질량을 seed로 → dangling="teleport"와 대응
    s = seed_vector(n, seeds)
    r = power_iteration(M, teleport=s, beta=0.85, tol=1e-14, max_iter=1000)
    pers = {int(i): 1.0 for i in seeds}
    ref = nx_vector(nx.pagerank(G, alpha=0.85, personalization=pers, weight=None,
                                tol=1e-14, max_iter=1000), n)
    np.testing.assert_allclose(r, ref, rtol=0, atol=1e-12)


def test8_networkx_uniform(random_graph):
    G, M, seeds = random_graph
    n = N_RANDOM
    s = seed_vector(n, seeds)
    r = power_iteration(M, teleport=s, beta=0.85, tol=1e-14, max_iter=1000, dangling="uniform")
    pers = {int(i): 1.0 for i in seeds}
    ref = nx_vector(nx.pagerank(G, alpha=0.85, personalization=pers, weight=None,
                                dangling={i: 1.0 for i in range(n)}, tol=1e-14, max_iter=1000), n)
    np.testing.assert_allclose(r, ref, rtol=0, atol=1e-12)


# ---------- 불변식 ----------

@pytest.mark.parametrize("dangling", ["teleport", "uniform"])
@pytest.mark.parametrize("personalized", [False, True])
def test_sum_one_nonnegative(random_graph, dangling, personalized):
    _, M, seeds = random_graph
    s = seed_vector(N_RANDOM, seeds) if personalized else None
    r = power_iteration(M, teleport=s, dangling=dangling)
    assert r.shape == (N_RANDOM,) and r.dtype == np.float64
    assert r.sum() == pytest.approx(1.0, abs=1e-12)
    assert (r >= 0).all()


def test_unreachable_exactly_zero():
    # seed 0 → 1 → 2(dead end). 3 ↔ 4는 seed에서 도달 불가능, 5는 고립
    A = make_A([(0, 1), (1, 2), (3, 4), (4, 3), (3, 0)], 6)
    M = build_transition(A)
    r = power_iteration(M, teleport=seed_vector(6, {0}), dangling="teleport")
    assert (r[[0, 1, 2]] > 0).all()
    assert r[3] == 0 and r[4] == 0 and r[5] == 0   # r(0) = s 덕분에 정확히 0
    # 같은 그래프의 PageRank는 모든 host > 0
    assert (pagerank(M) > 0).all()


def test_return_info(random_graph):
    _, M, _ = random_graph
    r, info = pagerank(M, return_info=True)
    assert set(info) == {"iterations", "residual", "converged", "time_sec"}
    assert info["converged"] and info["residual"] < 1e-8 and 1 <= info["iterations"] <= 300
    assert r.sum() == pytest.approx(1.0)


# ---------- 예외·경고 ----------

def test_invalid_dangling_raises(random_graph):
    _, M, _ = random_graph
    with pytest.raises(ValueError):
        power_iteration(M, dangling="unifrom")


def test_not_converged_warns(random_graph):
    _, M, _ = random_graph
    with pytest.warns(RuntimeWarning):
        r = power_iteration(M, max_iter=2, tol=1e-14)
    assert r.sum() == pytest.approx(1.0)                 # 마지막 r은 반환됨


def test_empty_seed_fails(random_graph):
    _, M, _ = random_graph
    with pytest.raises(AssertionError):
        power_iteration(M, teleport=np.zeros(N_RANDOM))
