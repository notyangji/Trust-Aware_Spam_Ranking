"""전처리 그래프에서 PPR·TrustRank와 보조 점수를 계산해 점수와 실행 기록을 저장한다.

사용법:
    python scripts/run_seeded_rankings.py
    python scripts/run_seeded_rankings.py --config configs/default.yaml --seed-config configs/seeds.yaml
    python scripts/run_seeded_rankings.py --seed-config configs/experiments/e3_n050.yaml --name e3_n050

저장하는 점수 (results/scores/<method>_<tag>.npy + .json):
    ppr, trustrank, spam_mass, hop, inverse_pagerank
tag는 --name을 주면 그 값, 주지 않으면 binary_b085처럼 β로 정한다.
--name 없이 β가 같은 실행을 반복하면 파일을 덮어쓰므로, 조건을 바꿔 여러 번 돌릴 때는 --name을 준다.
PPR과 Inverse PageRank의 higher_is_spam은 validation에서 점수 방향을 확정할 때까지 null이다.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baselines import save_scores  # noqa: E402
from src.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.graph import build_transition  # noqa: E402
from src.ppr import personalized_pagerank  # noqa: E402
from src.rank import pagerank  # noqa: E402
from src.seeds import select_topic_seeds, select_trusted_seeds, validate_seed_config  # noqa: E402
from src.trust import hop_distance, inverse_pagerank, spam_mass, trustrank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="PPR·TrustRank 점수 계산 및 저장")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="공통 설정 파일 경로")
    parser.add_argument("--seed-config", default="configs/seeds.yaml", help="Seed 설정 파일 경로")
    parser.add_argument("--name", default=None,
                        help="결과 파일 이름의 뒷부분 (예: e3_n050 → trustrank_e3_n050). 없으면 binary_b085처럼 β로 정함")
    args = parser.parse_args()
    if args.name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error("--name에는 영문, 숫자, _, ., -만 쓸 수 있습니다.")

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
    tag = args.name or f"binary_b{round(rank['beta'] * 100):03d}"
    config_record = {**rank, "weighted": False}

    def save(method, scores, higher_is_spam, seed_info, info):
        name = f"{method}_{tag}"
        save_scores(out_dir, name, scores, {
            "method": method,
            "higher_is_spam": higher_is_spam,
            "config": config_record,
            "seed_selection": seed_info,
            "info": info,
        })
        return name

    results = {}
    runs = (
        ("ppr", personalized_pagerank, topic_ids, topic_info, None),
        ("trustrank", trustrank, trusted_ids, trusted_info, False),
    )
    for method, calculate, seed_ids, seed_info, higher_is_spam in runs:
        scores, info = calculate(M, seed_ids, return_info=True, **rank)
        results[method] = scores
        name = save(method, scores, higher_is_spam, seed_info, info)
        print(f"{name}: seed {seed_info['n_selected']}개, {info['iterations']}회 "
              f"수렴={info['converged']}, 합 {scores.sum():.6f}")

    # spam mass: 같은 β의 균등 teleport PageRank와 TrustRank로 계산한다.
    pr, pr_info = pagerank(M, return_info=True, **rank)
    name = save("spam_mass", spam_mass(pr, results["trustrank"]), True, trusted_info, {"pagerank": pr_info})
    print(f"{name}: PageRank {pr_info['iterations']}회 수렴={pr_info['converged']}")

    # hop 거리: TrustRank와 같은 trusted seed에서 링크를 따라간 거리.
    hops = hop_distance(ds.graph.adjacency, trusted_ids)
    name = save("hop", hops, True, trusted_info, {"max_hop": float(hops.max())})
    print(f"{name}: 최댓값 {hops.max():.0f} (seed에서 닿지 않은 호스트는 이 값)")

    # Inverse PageRank: seed와 무관하다. 방향은 validation에서 정하도록 null.
    inv, inv_info = inverse_pagerank(ds.graph.adjacency, return_info=True, **rank)
    name = save("inverse_pagerank", inv, None, None, inv_info)
    print(f"{name}: {inv_info['iterations']}회 수렴={inv_info['converged']}")

    print(f"저장 완료: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
