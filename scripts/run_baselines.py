"""PageRank·baseline 점수 계산 → results/scores/ (build_processed.py 실행 후).

사용법:
    python scripts/run_baselines.py [--config configs/default.yaml]

출력 (paths.scores_dir), 각 점수마다 .npy와 같은 이름의 .json:
    pagerank_binary_b085    PageRank, 이진 그래프 (B의 spam mass, C의 E1)
    pagerank_weighted_b085  PageRank, 링크 수 가중 그래프 (E5)
    random_seed42           [0, 1) 균등 난수
    indegree                −log1p(in_deg)

.npy  길이 N float64 (index = host id)
.json method, higher_is_spam(점수 방향), 사용한 설정값, 수렴 정보(PageRank만), 생성일
      higher_is_spam: true = 클수록 spam 의심, false = 클수록 spam 아님, null = 방향 없음.
      C는 이 값을 보고 평가할 때 점수를 뒤집는다. B의 PPR·TrustRank 점수도 같은 형식으로 저장한다.
β 등을 바꾸려면 config를 복사해 --config로 지정한다 (파일 이름의 b085 등으로 구분).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # `python scripts/...`로 실행해도 src를 import할 수 있게

from src.baselines import indegree_scores, random_scores  # noqa: E402
from src.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.graph import build_transition  # noqa: E402
from src.rank import pagerank  # noqa: E402


def save_scores(out_dir: Path, name: str, scores: np.ndarray, record: dict) -> None:
    """scores를 <name>.npy로, record(+ 생성일, N)를 <name>.json으로 저장."""
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.ndim == 1 and np.isfinite(scores).all(), f"{name}: 점수가 1차원 유한값이 아님"
    np.save(out_dir / f"{name}.npy", scores)
    record = {**record, "n": int(scores.size), "created": datetime.date.today().isoformat()}
    (out_dir / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="PageRank·baseline 점수 계산")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="설정 파일 경로")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rank = {"beta": float(cfg.require("rank", "beta")),
            "tol": float(cfg.require("rank", "tol")),
            "max_iter": int(cfg.require("rank", "max_iter")),
            "dangling": str(cfg.require("rank", "dangling"))}
    seed = int(cfg.require("baselines", "random_seed"))
    out = cfg.path("scores_dir")
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(cfg.path("processed_dir"))
    A = ds.graph.adjacency
    tag = f"b{round(rank['beta'] * 100):03d}"           # β 0.85 → b085

    for weighted in (False, True):
        kind = "weighted" if weighted else "binary"
        r, info = pagerank(build_transition(A, weighted=weighted), return_info=True, **rank)
        save_scores(out, f"pagerank_{kind}_{tag}", r, {
            "method": "pagerank", "higher_is_spam": False,
            "config": {**rank, "weighted": weighted}, "info": info})
        print(f"pagerank_{kind}_{tag}: {info['iterations']}회 수렴={info['converged']} "
              f"({info['time_sec']:.2f}초), 합 {r.sum():.6f}")

    save_scores(out, f"random_seed{seed}", random_scores(ds.graph.n, seed), {
        "method": "random", "higher_is_spam": None, "config": {"seed": seed}})
    save_scores(out, "indegree", indegree_scores(A), {
        "method": "indegree", "higher_is_spam": True, "config": {"formula": "-log1p(in_deg)"}})
    print(f"random_seed{seed}, indegree 저장")
    print(f"완료: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
