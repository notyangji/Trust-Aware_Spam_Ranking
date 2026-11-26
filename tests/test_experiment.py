import numpy as np
import pytest
import scipy.sparse as sp

from scripts.run_experiment import expand_grid, resolve_n_seeds, run_experiment
from src.common.schema import NONSPAM, SPAM, UNLABELED, Dataset, HostGraph


def make_ds(n=400, seed=0):
    """학습·검증·테스트 라벨이 있는 작은 무작위 그래프. 앞 40개 host는 ac.uk (PPR seed 도메인)."""
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.02, random_state=seed, format="csr")
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    hosts = [f"h{i}.ac.uk" if i < 40 else f"h{i}.co.uk" for i in range(n)]
    labels = np.full(n, UNLABELED, dtype=np.int8)
    idx = rng.permutation(n)[:300]
    labels[idx] = np.where(rng.random(300) < 0.15, SPAM, NONSPAM)
    in_set = np.zeros(n, dtype=np.int8)
    in_set[idx[:200]] = 1
    in_set[idx[200:]] = 2
    return Dataset(HostGraph(hosts, A), labels, in_set,
                   np.sort(idx[:140]), np.sort(idx[140:200]), np.sort(idx[200:]))


class FakeCfg(dict):
    def require(self, section, key):
        return self[section][key]


CFG = FakeCfg(rank={"beta": 0.85, "tol": 1e-10, "max_iter": 300, "dangling": "teleport"},
              domains={"ppr_seed_domains": ["ac.uk"]})
SEED_CFG = {"trusted": {"n_seeds": 20, "contamination_rate": 0.0, "random_seed": 42}}


def test_expand_grid_overrides_defaults_and_rejects_unknown_keys():
    runs = expand_grid({"fixed": {"beta": 0.9}, "grid": {"n_seeds": [10, 20], "random_seed": [0, 1]}},
                       {"beta": 0.85, "n_seeds": 100, "contamination_rate": 0.0, "random_seed": 42})
    assert len(runs) == 4 and all(r["beta"] == 0.9 for r in runs)
    assert {(r["n_seeds"], r["random_seed"]) for r in runs} == {(10, 0), (10, 1), (20, 0), (20, 1)}
    with pytest.raises(ValueError):
        expand_grid({"grid": {"alpha": [1]}}, {"beta": 0.85})


def test_resolve_all_uses_every_train_nonspam():
    ds = make_ds()
    n_nonspam = int((ds.labels[ds.train_idx] == NONSPAM).sum())
    assert resolve_n_seeds(ds, "all", 0.0) == n_nonspam
    assert resolve_n_seeds(ds, 30, 0.0) == 30


def test_run_experiment_rows_directions_and_reference():
    ds = make_ds()
    exp = {"name": "t", "methods": ["random", "pagerank", "ppr", "trustrank", "spam_mass", "hop"],
           "grid": {"n_seeds": [10, 20], "random_seed": [0, 1]},
           "reference": {"method": "trustrank", "n_seeds": 20}}
    rows = run_experiment(exp, ds, CFG, SEED_CFG, n_boot=50, log=lambda *_: None)

    # 조합 4개 × 방법 6개, PPR은 seed 제외 평가가 한 줄씩 더
    assert len(rows) == 4 * 6 + 4
    assert {r["eval_subset"] for r in rows if r["method"] == "ppr"} == {"val", "val_no_seed"}
    # 정해진 방향
    assert all(r["direction"] == "-" for r in rows if r["method"] == "trustrank")
    assert all(r["direction"] == "+" for r in rows if r["method"] in ("spam_mass", "hop", "random"))
    # 평가는 validation만 (test host 수가 아니라 val host 수)
    assert all(r["n_eval"] == len(ds.val_idx) for r in rows if r["eval_subset"] == "val")
    # 기준 조건 자신과의 차이는 0, 다른 조합에는 차이 CI가 붙음
    ref = [r for r in rows if r["method"] == "trustrank" and r["n_seeds"] == 20]
    assert all(r["diff_vs_ref"] == 0 for r in ref)
    other = [r for r in rows if r["method"] == "trustrank" and r["n_seeds"] == 10]
    assert all(r["diff_lo"] <= r["diff_vs_ref"] <= r["diff_hi"] for r in other)
    # 지표 범위와 CI
    assert all(0 <= r["pr_auc"] <= 1 and r["pr_auc_lo"] <= r["pr_auc_hi"] for r in rows)


def test_seed_strategies_pick_from_their_pools():
    """전략별 seed가 학습용 NONSPAM 안에서, 해당 후보에서만 뽑히는지 확인한다."""
    from scripts.run_experiment import ScoreCache
    from src.graph import domain_of

    ds = make_ds()
    hosts = list(ds.graph.hostnames)
    for i in range(40, 80):                       # co.uk 외 도메인도 섞는다
        hosts[i] = f"h{i}.org.uk"
    ds = Dataset(HostGraph(hosts, ds.graph.adjacency), ds.labels, ds.in_set,
                 ds.train_idx, ds.val_idx, ds.test_idx)
    cache = ScoreCache(ds, CFG, {k: CFG["rank"][k] for k in ("tol", "max_iter", "dangling")})
    train_nonspam = set(ds.train_idx[ds.labels[ds.train_idx] == NONSPAM])
    base = {"beta": 0.85, "contamination_rate": 0.0, "random_seed": 0}

    for strategy, allowed in [("domain", {"ac.uk"}), ("commercial", {"co.uk"})]:
        seeds = cache.trusted({**base, "seed_strategy": strategy, "n_seeds": 5})
        assert len(seeds) == 5 and set(seeds) <= train_nonspam
        assert {domain_of(hosts[i]) for i in seeds} <= allowed

    seeds = cache.trusted({**base, "seed_strategy": "high_pr", "n_seeds": 5})
    pr, _ = cache.score("pagerank", base)
    pool = np.array(sorted(train_nonspam))
    assert set(seeds) == set(pool[np.argsort(-pr[pool], kind="stable")[:5]])

    with pytest.raises(ValueError):               # 후보보다 많이 요청
        cache.trusted({**base, "seed_strategy": "domain", "n_seeds": 10_000})
    with pytest.raises(ValueError):               # random 외 전략은 noise 미지원
        cache.trusted({**base, "seed_strategy": "domain", "n_seeds": 5, "contamination_rate": 0.1})
