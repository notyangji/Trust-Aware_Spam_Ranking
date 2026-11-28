"""최종 test 평가 (한 번만 실행).

사용법:
    python scripts/run_final_test.py              # configs/final.yaml → results/metrics/final_test.csv

- val(E1–E6)에서 방법·seed 전략·설정을 모두 정한 뒤 configs/final.yaml에 고정하고, 이 스크립트로 test를 한 번 본다.
- 같은 목록을 val에서도 다시 평가해 한 파일에 둔다 (eval_subset = val / test). val과 test를 나란히 비교하기 위해서다.
- 점수 방향은 항상 validation으로 정한다 (run_experiment와 같은 규칙).
- 결과 파일이 이미 있으면 실행하지 않는다. test를 여러 번 보지 않기 위한 장치다.
  다시 만들어야 하는 사정(예: 실행 중 오류)이 있으면 --overwrite를 쓰고, 그 이유를 커밋 메시지에 남긴다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import COLUMNS, run_experiment, write_csv  # noqa: E402
from src.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.data import load_dataset  # noqa: E402

FINAL_COLUMNS = ["exp", "group", *COLUMNS[1:]]          # baselines / selected 구분 열 추가


def main() -> int:
    parser = argparse.ArgumentParser(description="최종 test 평가 (한 번만)")
    parser.add_argument("--final", default="configs/final.yaml", help="최종 평가 목록")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="공통 설정 파일 경로")
    parser.add_argument("--seed-config", default="configs/seeds.yaml", help="seed 기본 설정 파일 경로")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap 반복 수")
    parser.add_argument("--overwrite", action="store_true", help="이미 있는 결과를 덮어씀 (이유를 기록할 것)")
    args = parser.parse_args()

    final_cfg = load_config(args.final)
    out = ROOT / "results" / "metrics" / f"{final_cfg['name']}.csv"
    if out.exists() and not args.overwrite:
        print(f"{out.relative_to(ROOT)}가 이미 있습니다. test는 한 번만 봅니다. "
              "다시 만들어야 하면 --overwrite (이유를 커밋 메시지에 남길 것).")
        return 1

    cfg = load_config(args.config)
    seed_cfg = load_config(args.seed_config)
    ds = load_dataset(cfg.path("processed_dir"))

    t0 = time.perf_counter()
    rows = run_experiment(final_cfg, ds, cfg, seed_cfg, n_boot=args.n_boot, splits=("val", "test"))
    write_csv(rows, out, columns=FINAL_COLUMNS)
    print(f"{len(rows)}줄 저장: {out.relative_to(ROOT)} ({time.perf_counter() - t0:.1f}초)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
