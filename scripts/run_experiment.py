"""실험 설정 파일의 조합마다 점수를 계산하고 validation에서 평가해 CSV로 저장한다.

사용법:
    python scripts/run_experiment.py configs/experiments/e3.yaml
    python scripts/run_experiment.py configs/experiments/e3.yaml --n-boot 200   # 빠른 확인용

출력: results/metrics/<name>.csv (조합 × 방법 하나 = 한 줄)
- validation만 사용한다. test 평가는 scripts/run_final_test.py에서 마지막에 한 번만 한다.
- 점수는 메모리에서 바로 평가하고 .npy로 저장하지 않는다 (점수 파일 저장은 run_seeded_rankings.py).

설정 파일 형식:
    name: e3
    methods: [trustrank, spam_mass, hop]
    fixed: {beta: 0.85, contamination_rate: 0.0}          # 기본값과 다른 고정값
    grid: {n_seeds: [10, 100, all], random_seed: [0, 1]}   # 바꿔 가며 돌릴 값 (곱집합)
    reference: {method: trustrank, n_seeds: 100}           # paired 차이 CI의 기준 (선택)
지정하지 않은 값은 기본 설정(default.yaml의 rank, seeds.yaml의 trusted)을 따른다.
n_seeds: all은 학습용 NONSPAM 전체(혼입이 있으면 그만큼 뺀 수)를 뜻한다.

seed_strategy (trusted seed를 어디서 어떻게 고를지, 기본 random). 후보는 모두 학습용 NONSPAM이다.
    random      전체 후보에서 무작위 (src/seeds.py의 select_trusted_seeds 그대로)
    domain      domains.ppr_seed_domains(ac.uk·gov.uk) 도메인 후보에서 무작위
    commercial  co.uk 후보에서 무작위
    high_pr     전체 후보 중 PageRank 상위 n개 (난수 안 씀)
    svd         인접 행렬 truncated SVD의 축마다 대표 host 하나씩 (난수 안 씀, 축 수 = n)
    domain_high_pr  domain 후보 중 PageRank 상위 n개 (난수 안 씀). val을 보고 추가한 탐색 전략
random 외의 전략은 contamination_rate 0만 지원한다. 전략 함수는 src/seed_strategies.py에 있다.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import indegree_scores, random_scores  # noqa: E402
from src.common.schema import NONSPAM  # noqa: E402
from src.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.graph import build_transition  # noqa: E402
from src.metrics import (bootstrap, choose_direction, eval_arrays, paired_diff_ci,  # noqa: E402
                         pr_auc, precision_at_k, roc_auc, tie_fraction)
from src.ppr import personalized_pagerank  # noqa: E402
from src.rank import pagerank  # noqa: E402
from src.seed_strategies import (STRATEGIES as SEED_STRATEGIES, high_pagerank_seeds,  # noqa: E402
                                 random_pool_seeds, seed_pool, svd_components, svd_seeds)
from src.seeds import select_topic_seeds, select_trusted_seeds  # noqa: E402
from src.trust import hop_distance, inverse_pagerank, spam_mass, trustrank  # noqa: E402

# 방법별 점수 방향. "val"은 validation PR-AUC로 +/−를 정한다 (choose_direction).
# random은 방향을 고르지 않는다 (고르면 운 좋은 쪽이 뽑혀 부풀려짐).
DIRECTION = {
    "pagerank": "val", "pagerank_weighted": "val", "inverse_pagerank": "val", "ppr": "val",
    "indegree": +1, "random": +1,
    "trustrank": -1, "spam_mass": +1, "hop": +1,
}
SEEDED = {"trustrank", "spam_mass", "hop"}       # trusted seed에 따라 달라지는 방법
PARAMS = ("beta", "n_seeds", "contamination_rate", "random_seed", "seed_strategy")
SVD_MAX_K = 100                                 # svd 전략은 SVD를 한 번(축 100개) 계산해 앞쪽 축만 쓴다
KS = (100, 500)
COLUMNS = ["exp", "method", *PARAMS, "direction", "eval_subset", "n_eval", "n_pos", "prevalence",
           "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", *[f"p@{k}" for k in KS], "tie_frac",
           "ref", "diff_vs_ref", "diff_lo", "diff_hi", "iterations", "converged", "time_sec", "commit"]


def expand_grid(exp_cfg, defaults):
    """fixed와 grid를 기본값 위에 덮어 실행할 조합 목록을 만든다."""
    base = {**defaults, **(exp_cfg.get("fixed") or {})}
    grid = exp_cfg.get("grid") or {}
    unknown = (set(base) | set(grid)) - set(PARAMS)
    if unknown:
        raise ValueError(f"알 수 없는 설정 키: {sorted(unknown)} (가능: {PARAMS})")
    keys = list(grid)
    return [{**base, **dict(zip(keys, values))} for values in itertools.product(*(grid[k] for k in keys))]


def resolve_n_seeds(ds, n_seeds, contamination_rate, pool_size=None):
    """n_seeds: all → 후보 전체(기본: 학습용 NONSPAM 전체)를 쓰도록 총개수를 계산한다."""
    if n_seeds != "all":
        return int(n_seeds)
    n_nonspam = pool_size if pool_size is not None else int((ds.labels[ds.train_idx] == NONSPAM).sum())
    return int(n_nonspam / (1 - contamination_rate)) if contamination_rate < 1 else n_nonspam


class ScoreCache:
    """같은 β·seed 조합의 점수를 한 번만 계산한다."""

    def __init__(self, ds, cfg, rank_options):
        self.ds, self.cfg, self.rank = ds, cfg, rank_options
        self.A = ds.graph.adjacency
        self.M = build_transition(self.A)
        self._M_w = None
        self._memo = {}
        self.seed_domains = list(cfg.require("domains", "ppr_seed_domains"))
        self.topic_ids = select_topic_seeds(ds.graph.hostnames, self.seed_domains)

    def _get(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def _rank(self, beta):
        return {**self.rank, "beta": float(beta), "return_info": True}

    def seed_pool(self, strategy):
        return self._get(("pool", strategy), lambda: seed_pool(self.ds, strategy, self.seed_domains))

    def svd_U(self, k):
        if k > SVD_MAX_K:
            raise ValueError(f"svd 전략은 n_seeds {SVD_MAX_K} 이하만 지원합니다.")
        return self._get(("svd",), lambda: svd_components(self.A, SVD_MAX_K)[0])

    def trusted(self, run):
        strategy = run.get("seed_strategy", "random")
        rate = float(run["contamination_rate"])
        if strategy == "random":
            n = resolve_n_seeds(self.ds, run["n_seeds"], rate)
            key = ("seeds", strategy, n, rate, run["random_seed"])
            return self._get(key, lambda: select_trusted_seeds(
                self.ds, n_seeds=n, random_seed=int(run["random_seed"]), contamination_rate=rate))
        if rate != 0:
            raise ValueError(f"seed_strategy={strategy}는 contamination_rate 0만 지원합니다.")
        pool = self.seed_pool(strategy)
        n = resolve_n_seeds(self.ds, run["n_seeds"], 0.0, pool_size=len(pool))
        if strategy in ("high_pr", "domain_high_pr"):
            pr, _ = self.score("pagerank", run)
            return self._get(("seeds", strategy, n, run["beta"]), lambda: high_pagerank_seeds(pool, pr, n))
        if strategy == "svd":
            return self._get(("seeds", strategy, n), lambda: svd_seeds(self.svd_U(n), pool, n))
        return self._get(("seeds", strategy, n, run["random_seed"]),
                         lambda: random_pool_seeds(pool, n, run["random_seed"]))

    def score(self, method, run):
        """(점수, 수렴 정보) — 수렴 정보가 없는 방법은 None."""
        b = run["beta"]
        if method == "pagerank":
            return self._get(("pr", b), lambda: pagerank(self.M, **self._rank(b)))
        if method == "pagerank_weighted":
            if self._M_w is None:
                self._M_w = build_transition(self.A, weighted=True)
            return self._get(("prw", b), lambda: pagerank(self._M_w, **self._rank(b)))
        if method == "inverse_pagerank":
            return self._get(("inv", b), lambda: inverse_pagerank(self.A, **self._rank(b)))
        if method == "ppr":
            return self._get(("ppr", b), lambda: personalized_pagerank(self.M, self.topic_ids, **self._rank(b)))
        if method == "indegree":
            return self._get(("indeg",), lambda: (indegree_scores(self.A), None))
        if method == "random":
            r = int(run["random_seed"])
            return self._get(("rand", r), lambda: (random_scores(self.ds.graph.n, r), None))
        seeds = self.trusted(run)
        key = (seeds.tobytes(), b)
        if method == "trustrank":
            return self._get(("tr",) + key, lambda: trustrank(self.M, seeds, **self._rank(b)))
        if method == "spam_mass":
            pr, _ = self.score("pagerank", run)
            tr, info = self.score("trustrank", run)
            return self._get(("mass",) + key, lambda: (spam_mass(pr, tr), info))
        if method == "hop":
            return self._get(("hop", seeds.tobytes()), lambda: (hop_distance(self.A, seeds), None))
        raise ValueError(f"알 수 없는 방법: {method} (가능: {sorted(DIRECTION)})")


def metrics_row(risk_eval, y, n_boot, boot_seed):
    """평가 대상 host만의 risk·라벨로 지표와 bootstrap 표본을 계산한다."""
    ci, stats = bootstrap({"m": risk_eval}, y, n_boot=n_boot, seed=boot_seed)
    row = {"n_eval": len(y), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
           "pr_auc": pr_auc(risk_eval, y), "pr_auc_lo": ci["m"][0], "pr_auc_hi": ci["m"][1],
           "roc_auc": roc_auc(risk_eval, y), "tie_frac": tie_fraction(risk_eval)}
    for k in KS:
        row[f"p@{k}"] = precision_at_k(risk_eval, y, k)
    return row, stats["m"]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_experiment(exp_cfg, ds, cfg, seed_cfg, n_boot=1000, boot_seed=0, log=print, splits=("val",)):
    """exp_cfg의 조합 × 방법마다 점수를 계산해 평가한다.

    splits: 평가할 분할. 실험(E1–E6)은 ("val",)만 쓴다. ("val", "test")는 run_final_test.py 전용.
    점수 방향("val" 방법)은 splits와 관계없이 항상 validation으로 정한다.
    exp_cfg에 groups가 있으면 묶음마다 methods·fixed·grid를 따로 두고, 같은 점수 캐시와 기준(reference)을 공유한다.
    """
    rank = {k: cfg.require("rank", k) for k in ("tol", "max_iter", "dangling")}
    defaults = {"beta": float(cfg.require("rank", "beta")), **{
        k: seed_cfg["trusted"][k] for k in ("n_seeds", "contamination_rate", "random_seed")},
        "seed_strategy": "random"}
    groups = exp_cfg.get("groups") or [{**exp_cfg, "name": ""}]
    plan = []
    for g in groups:
        runs = expand_grid(g, defaults)
        bad = {r["seed_strategy"] for r in runs} - set(SEED_STRATEGIES)
        if bad:
            raise ValueError(f"알 수 없는 seed_strategy: {sorted(bad)} (가능: {SEED_STRATEGIES})")
        methods = list(g["methods"])
        unknown = set(methods) - set(DIRECTION)
        if unknown:
            raise ValueError(f"알 수 없는 방법: {sorted(unknown)} (가능: {sorted(DIRECTION)})")
        plan.append((g.get("name", ""), runs, methods))
    if set(splits) - {"val", "test"}:
        raise ValueError(f"splits는 val·test만 가능: {splits}")

    cache = ScoreCache(ds, cfg, rank)
    evals = {sp: eval_arrays(ds, sp) for sp in splits}
    commit = git_commit()
    rows, stats = [], []

    for group, runs, methods in plan:
        for i, run in enumerate(runs, 1):
            for method in methods:
                t0 = time.perf_counter()
                score, info = cache.score(method, run)
                d = DIRECTION[method]
                sign = choose_direction(score, ds)[0] if d == "val" else d   # 방향은 항상 val로
                risk = sign * np.asarray(score)
                for sp, (idx, y) in evals.items():
                    subsets = [(sp, np.ones(len(idx), bool))]
                    if method == "ppr":                  # seed host 자신을 뺀 평가도 남긴다
                        subsets.append((f"{sp}_no_seed", ~np.isin(idx, cache.topic_ids)))
                    for subset, mask in subsets:
                        m, st = metrics_row(risk[idx][mask], y[mask], n_boot, boot_seed)
                        rows.append({"exp": exp_cfg["name"], "group": group, "method": method, **run,
                                     "direction": "+" if sign > 0 else "-", "eval_subset": subset, **m,
                                     "iterations": (info or {}).get("iterations", ""),
                                     "converged": (info or {}).get("converged", ""),
                                     "time_sec": round(time.perf_counter() - t0, 3), "commit": commit})
                        stats.append(st)
            log(f"[{group or exp_cfg['name']} {i}/{len(runs)}] " + ", ".join(f"{k}={run[k]}" for k in PARAMS))

    add_reference_diffs(rows, stats, exp_cfg.get("reference"), subsets=splits)
    return rows


def add_reference_diffs(rows, stats, reference, subsets=("val",)):
    """기준 조건과의 PR-AUC 차이 CI (같은 bootstrap 표본이라 paired). random_seed는 같은 값끼리 짝짓는다.

    분할(val·test)마다 같은 분할의 기준 줄과 짝짓는다. *_no_seed 줄에는 붙이지 않는다.
    """
    if not reference:
        return
    ref_method = reference["method"]
    ref_params = {k: v for k, v in reference.items() if k != "method"}
    for row, st in zip(rows, stats):
        if row["eval_subset"] not in subsets:
            continue
        for ref_row, ref_st in zip(rows, stats):
            if (ref_row["method"] == ref_method and ref_row["eval_subset"] == row["eval_subset"]
                    and ref_row["random_seed"] == row["random_seed"]
                    and all(ref_row[k] == v for k, v in ref_params.items())
                    and all(ref_row[k] == row[k] for k in PARAMS if k not in ref_params and k != "random_seed")):
                mean, lo, hi = paired_diff_ci({"a": st, "b": ref_st}, "a", "b")
                row.update({"ref": ref_method + "".join(f",{k}={v}" for k, v in ref_params.items()),
                            "diff_vs_ref": mean, "diff_lo": lo, "diff_hi": hi})
                break


def write_csv(rows, path, columns=COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description="실험 조합 실행 → validation 평가 → CSV")
    parser.add_argument("experiment", help="실험 설정 파일 (configs/experiments/e*.yaml)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="공통 설정 파일 경로")
    parser.add_argument("--seed-config", default="configs/seeds.yaml", help="seed 기본 설정 파일 경로")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap 반복 수")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_cfg = load_config(args.seed_config)
    exp_cfg = load_config(args.experiment)
    ds = load_dataset(cfg.path("processed_dir"))

    t0 = time.perf_counter()
    rows = run_experiment(exp_cfg, ds, cfg, seed_cfg, n_boot=args.n_boot)
    out = ROOT / "results" / "metrics" / f"{exp_cfg['name']}.csv"
    write_csv(rows, out)
    print(f"{len(rows)}줄 저장: {out.relative_to(ROOT)} ({time.perf_counter() - t0:.1f}초)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
