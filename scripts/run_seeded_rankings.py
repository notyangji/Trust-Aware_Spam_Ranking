"""전처리 그래프에서 PPR·TrustRank를 계산해 점수와 실행 기록을 저장한다.

사용법:
    python scripts/run_seeded_rankings.py
    python scripts/run_seeded_rankings.py --config configs/default.yaml --seed-config configs/seeds.yaml

기본 출력은 results/scores/의 ppr_binary_b085, trustrank_binary_b085다.
같은 출력 경로에서 β가 같은 실행은 파일을 덮어쓴다. 결과를 따로
보관하려면 공통 설정에서 paths.scores_dir를 다르게 지정한다.
PPR의 higher_is_spam은 validation에서 점수 방향을 확정할 때까지 null이다.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baselines import save_scores  # noqa: E402
from src.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.graph import build_transition  # noqa: E402
from src.ppr import personalized_pagerank  # noqa: E402
from src.seeds import select_topic_seeds, select_trusted_seeds, validate_seed_config  # noqa: E402
from src.trust import trustrank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="PPR·TrustRank 점수 계산 및 저장")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="공통 설정 파일 경로")
    parser.add_argument("--seed-config", default="configs/seeds.yaml", help="Seed 설정 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_cfg = load_config(args.seed_config)
    validate_seed_config(seed_cfg)

    rank = {
        "beta": float(cfg.require("rank", "beta")),
        "tol": float(cfg.require("rank", "tol")),
        "max_iter": int(cfg.require("rank", "max_iter")),
        "dangling": str(cfg.require("rank", "dangling")),
    }
    ds = load_dataset(cfg.path("processed_dir"))
    M = build_transition(ds.graph.adjacency, weighted=False)

    topic_ids, topic_info = select_topic_seeds(
        ds.graph.hostnames,
        cfg.require("domains", "ppr_seed_domains"),
        return_info=True,
        **seed_cfg["topic"],
    )
    trusted = seed_cfg["trusted"]
    trusted_ids, trusted_info = select_trusted_seeds(
        ds,
        n_seeds=trusted["n_seeds"],
        random_seed=trusted["random_seed"],
        contamination_rate=trusted["contamination_rate"],
        return_info=True,
    )

    out_dir = cfg.path("scores_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    beta_tag = f"b{round(rank['beta'] * 100):03d}"
    runs = (
        ("ppr", personalized_pagerank, topic_ids, topic_info, None),
        ("trustrank", trustrank, trusted_ids, trusted_info, False),
    )
    for method, calculate, seed_ids, seed_info, higher_is_spam in runs:
        scores, info = calculate(M, seed_ids, return_info=True, **rank)
        name = f"{method}_binary_{beta_tag}"
        save_scores(out_dir, name, scores, {
            "method": method,
            "higher_is_spam": higher_is_spam,
            "config": {**rank, "weighted": False},
            "seed_selection": seed_info,
            "info": info,
        })
        print(f"{name}: seed {seed_info['n_selected']}개, {info['iterations']}회 "
              f"수렴={info['converged']}, 합 {scores.sum():.6f}")

    print(f"저장 완료: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
