"""raw → data/processed/ (다운로드 후 한 번 실행).

사용법:
    python scripts/build_processed.py [--config configs/default.yaml]

출력 (paths.processed_dir):
    graph.npz      인접 행렬 A (self-loop 제거, scipy.sparse.save_npz)
    hostnames.txt  한 줄에 hostname 하나, 줄 번호 = host id
    labels.csv     host_id, label, set, spamicity, assessments
    split.npz      train_idx, val_idx, test_idx (int64 host id, 사이트 단위 분할)
    meta.json      원본 SHA-256, 그래프·라벨 점검 통계, 분할 설정과 집합별 spam/nonspam 수

- 점검 수치가 기대값과 다르면 경고만 출력하고 계속 진행한다 (판단은 사람이).
- 다시 실행해 분할이 바뀌면 알린다 → B·C에게 공유할 것.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # `python scripts/...`로 실행해도 src를 import할 수 있게

from src.data import load_dataset, load_graph, load_hostnames, load_labels, make_split  # noqa: E402
from src.graph import site_of  # noqa: E402

GRAPH_FILE = "uk-2007-05.hostgraph_weighted.graph-txt.gz"
HOSTNAMES_FILE = "uk-2007-05.hostnames.txt.gz"
LABEL_ARCHIVES = ("webspam-uk2007-set1-1.0.tar.gz", "webspam-uk2007-set2-1.0.tar.gz")

# 문헌·공식 안내의 값 (GUIDE 3.5절). 다르면 경고.
EXPECTED = {"n": 114_529, "edges": 1_836_441, "labeled": 6_479}


def load_config(path: Path) -> dict:
    """설정 파일에서 paths.raw_dir, paths.processed_dir, split을 읽는다. 파일이나 키가 없으면 오류."""
    if not path.exists():
        raise FileNotFoundError(f"설정 파일 없음: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    def get(section: str, key: str):
        value = (cfg.get(section) or {}).get(key)
        if value is None:
            raise ValueError(f"{path}에 설정 키 {section}.{key} 없음")
        return value

    return {"raw_dir": ROOT / get("paths", "raw_dir"),
            "processed_dir": ROOT / get("paths", "processed_dir"),
            "val_frac": float(get("split", "val_frac")),
            "random_seed": int(get("split", "random_seed"))}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def graph_stats(A: sp.csr_matrix, parse_stats: dict) -> dict:
    """GUIDE 3.5절 점검표용 그래프 통계."""
    n = parse_stats["n"]                        # 파일 첫 줄의 host 수 = A의 행·열 수
    out_deg = np.diff(A.indptr)
    in_deg = np.bincount(A.indices, minlength=n)
    return {
        "n": n,
        "n_lines": parse_stats["n_lines"],
        "nnz": int(A.nnz),
        "self_loops": parse_stats["self_loops"],
        "duplicates": parse_stats["entries"] - int(A.nnz),
        "dead_ends": int((out_deg == 0).sum()),
        "in_degree_zero": int((in_deg == 0).sum()),
        "isolated": int(((out_deg == 0) & (in_deg == 0)).sum()),
        "weight_min": float(A.data.min()),
        "weight_median": float(np.median(A.data)),
        "weight_max": float(A.data.max()),
    }


def count_labels(labels_df, idx) -> dict:
    """host id 배열 → {"spam": .., "nonspam": ..}"""
    lab = labels_df.set_index("host_id").loc[idx, "label"]
    return {k: int((lab == k).sum()) for k in ("spam", "nonspam")}


def warn_if_unexpected(meta: dict) -> None:
    checks = [("N", meta["graph"]["n"], EXPECTED["n"]),
              ("edge 수 (nnz)", meta["graph"]["nnz"], EXPECTED["edges"]),
              ("라벨 있는 host 수", meta["labels"]["total"], EXPECTED["labeled"])]
    for name, got, want in checks:
        status = "OK" if got == want else "경고: 기대값과 다름 → parser 문제인지 정의 차이인지 확인"
        print(f"  {name}: {got:,} (기대 {want:,})  {status}")
    if meta["graph"]["n_lines"] not in (meta["graph"]["n"], meta["graph"]["n"] - 1):
        print("  경고: 파일 줄 수가 N과 맞지 않음")


def report_split_change(out_dir: Path, new: dict) -> None:
    """기존 split.npz와 비교해 분할이 바뀌었는지 알린다."""
    old_path = out_dir / "split.npz"
    if not old_path.exists():
        print("  분할: 새로 만듦")
        return
    with np.load(old_path) as old:
        same = all(k in old and np.array_equal(old[k], new[k]) for k in new)
    print("  분할: 기존과 동일" if same else "  분할: 기존과 달라짐 → B·C에게 공유할 것")


def main() -> int:
    parser = argparse.ArgumentParser(description="raw → data/processed/")
    parser.add_argument("--config", default="configs/default.yaml", help="설정 파일 경로")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    raw, out = cfg["raw_dir"], cfg["processed_dir"]

    print("[읽기]")
    A, parse_stats = load_graph(raw / GRAPH_FILE, return_stats=True)
    hostnames = load_hostnames(raw / HOSTNAMES_FILE)
    labels_df = load_labels(raw / "set1", raw / "set2")
    n = parse_stats["n"]                        # 파일 첫 줄의 host 수 = A의 행·열 수
    assert len(hostnames) == n, f"hostname 수 {len(hostnames):,}가 N={n:,}과 다름"

    print("[분할]")
    tr, va, te = make_split(labels_df, n, cfg["val_frac"], cfg["random_seed"], hostnames)
    shared = {site_of(hostnames[i]) for i in tr} & {site_of(hostnames[i]) for i in va}
    assert not shared, f"train·val에 걸친 사이트 {len(shared)}개"
    split = {"train_idx": tr, "val_idx": va, "test_idx": te}

    meta = {
        "created": datetime.date.today().isoformat(),
        "source_sha256": {name: sha256_of(raw / name)
                          for name in (GRAPH_FILE, HOSTNAMES_FILE, *LABEL_ARCHIVES)},
        "graph": graph_stats(A, parse_stats),
        "labels": {
            "total": len(labels_df),
            **{s: {k: int(((labels_df["set"] == s) & (labels_df["label"] == k)).sum())
                   for k in ("spam", "nonspam", "undecided")} for s in ("SET1", "SET2")},
        },
        "split": {
            "unit": "site",
            "val_frac": cfg["val_frac"],
            "random_seed": cfg["random_seed"],
            "train": count_labels(labels_df, tr),
            "val": count_labels(labels_df, va),
            "test": count_labels(labels_df, te),
            "train_val_shared_sites": len(shared),
        },
    }

    print("[점검]")
    warn_if_unexpected(meta)
    out.mkdir(parents=True, exist_ok=True)
    report_split_change(out, split)

    print(f"[저장] {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    sp.save_npz(out / "graph.npz", A)
    (out / "hostnames.txt").write_text("\n".join(hostnames) + "\n", encoding="utf-8")
    labels_df.to_csv(out / "labels.csv", index=False)
    np.savez(out / "split.npz", **split)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ds = load_dataset(out)                      # 저장한 파일이 Dataset 검사를 통과하는지 확인
    for name in ("train", "val", "test"):
        c = meta["split"][name]
        print(f"  {name:5} {c['spam'] + c['nonspam']:5,}  (spam {c['spam']}, nonspam {c['nonspam']})")
    print(f"완료: load_dataset() 검사 통과 (N = {ds.graph.n:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
