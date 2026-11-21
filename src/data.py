"""데이터 로딩: raw 파일 parsing, train/val/test 분할, processed → Dataset.

- scripts/build_processed.py가 raw에 대해 load_graph / load_hostnames / load_labels / make_split을 한 번 호출해
  data/processed/를 만든다.
- B·C는 load_dataset()만 호출한다.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.schema import NONSPAM, SPAM, UNDECIDED, UNLABELED, Dataset, HostGraph
from src.graph import site_of

ROOT = Path(__file__).resolve().parents[1]

LABEL_CODE = {"spam": SPAM, "nonspam": NONSPAM, "undecided": UNDECIDED}
SET_CODE = {"SET1": 1, "SET2": 2}
PROCESSED_FILES = ("graph.npz", "hostnames.txt", "labels.csv", "split.npz")


def _open_text(path: str | Path):
    """.gz면 압축을 풀면서, 아니면 그대로 텍스트로 연다."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def load_graph(path, return_stats=False):
    """host graph 파일 → 인접 행렬 A (csr_matrix, float64, shape (N, N)).

    A[i, j] = host i → host j의 link 수 (행 = 출발, 열 = 도착).
    파일 형식: 1행 = N, 이후 i+2행 = host i의 out-link `dest:nlinks` (빈 줄 = dead end).
    self-loop는 여기서만 제거한다. 중복 (src, dst)는 csr_matrix 생성 시 합산된다.
    return_stats=True면 (A, stats)를 돌려준다. stats: n, n_lines, self_loops, entries, nnz.
    """
    rows, cols, vals = [], [], []
    n_lines = n_self = 0
    with _open_text(path) as f:
        n = int(f.readline())
        for src, line in enumerate(f):
            n_lines += 1
            for tok in line.replace(",", " ").split():
                dst, w = tok.split(":")
                dst = int(dst)
                if dst == src:
                    n_self += 1
                    continue
                rows.append(src)
                cols.append(dst)
                vals.append(int(w))
    # 마지막 host가 dead end이고 파일이 줄바꿈 없이 끝나면 n_lines == n - 1일 수 있음
    assert n_lines in (n, n - 1), f"줄 수 {n_lines}가 N={n}과 맞지 않음"
    if cols:
        lo, hi = min(cols), max(cols)
        assert 0 <= lo and hi < n, f"dest 범위 [{lo}, {hi}]가 [0, {n})을 벗어남"
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    if return_stats:
        return A, {"n": n, "n_lines": n_lines, "self_loops": n_self,
                   "entries": len(vals), "nnz": A.nnz}
    return A


def load_hostnames(path) -> list[str]:
    """hostname 목록 (index = host id).

    두 형식을 모두 읽는다.
    - 원본 `uk-2007-05.hostnames.txt.gz`: 한 줄에 `id hostname`
    - processed `hostnames.txt`: 한 줄에 hostname 하나, 줄 번호 = host id
    """
    names = []
    with _open_text(path) as f:
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) == 2:
                assert int(parts[0]) == i, f"{i}번째 줄의 id가 {parts[0]}임 (0부터 순서대로여야 함)"
                names.append(parts[1])
            elif len(parts) == 1:
                names.append(parts[0])
            else:
                raise ValueError(f"{path}: {i}번째 줄 형식을 알 수 없음: {line!r}")
    return names


def _find_label_file(set_dir: Path, set_name: str) -> Path:
    """압축 해제 폴더 안에서 `*-SET1-labels.txt`를 찾는다 (tar 안에 상위 폴더가 한 단계 더 있음)."""
    found = sorted(Path(set_dir).rglob(f"*-{set_name}-labels.txt"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"{set_dir}에서 {set_name} label 파일을 하나로 찾지 못함 (찾은 수: {len(found)}). "
            "scripts/download_data.py를 먼저 실행")
    return found[0]


def load_labels(set1_dir, set2_dir) -> pd.DataFrame:
    """label 표 (약 6,479행).

    columns: host_id(int64), label(str), set(str), spamicity(float, 판정 없음 = NaN), assessments(str)
    label ∈ {"spam", "nonspam", "undecided"}, set ∈ {"SET1", "SET2"}
    원본 형식: `hostid label spamicity assessments` (공백 구분, spamicity가 '-'이면 판정 없음)
    """
    frames = []
    for set_name, set_dir in (("SET1", set1_dir), ("SET2", set2_dir)):
        df = pd.read_csv(_find_label_file(set_dir, set_name), sep=r"\s+", header=None,
                         names=["host_id", "label", "spamicity", "assessments"],
                         dtype={"host_id": np.int64, "label": str, "assessments": str},
                         na_values={"spamicity": ["-"]}, keep_default_na=False)
        df["spamicity"] = df["spamicity"].astype(np.float64)
        df.insert(2, "set", set_name)
        assert not df["host_id"].duplicated().any(), f"{set_name} 안에 같은 host가 두 번 있음"
        frames.append(df)
    labels_df = pd.concat(frames, ignore_index=True)

    unknown = set(labels_df["label"]) - set(LABEL_CODE)
    assert not unknown, f"알 수 없는 label 값: {unknown}"
    both = set(frames[0]["host_id"]) & set(frames[1]["host_id"])
    assert not both, f"SET1·SET2 양쪽에 있는 host {len(both)}개"
    return labels_df


def make_split(labels_df, n, val_frac, seed, hostnames):
    """SET1의 spam/nonspam을 train·val로 사이트 단위 층화 분할, SET2의 spam/nonspam은 test.

    같은 사이트의 host는 항상 train·val 중 한쪽에만 들어간다 (공식 SET1/SET2 분할과 같은 방식,
    seed와 같은 사이트의 host가 val에 들어가 성능이 부풀려지는 데이터 누수 방지).
    사이트 묶음을 seed로 섞은 뒤, val의 spam·nonspam 수가 각각 val_frac 비율에 이를 때까지 채운다.
    undecided는 세 집합 모두에서 제외. 반환값은 정렬된 int64 host id 배열 3개.
    """
    assert len(hostnames) == n, "hostnames 수가 N과 다름"
    df = labels_df[labels_df["label"].isin(["spam", "nonspam"])]
    s1, s2 = df[df["set"] == "SET1"], df[df["set"] == "SET2"]

    ids = s1["host_id"].to_numpy(np.int64)
    is_spam = (s1["label"] == "spam").to_numpy()
    sites = np.array([site_of(hostnames[i]) for i in ids])
    _, group = np.unique(sites, return_inverse=True)      # host → 사이트 묶음 번호
    g_spam = np.bincount(group, weights=is_spam).astype(np.int64)
    g_non = np.bincount(group, weights=~is_spam).astype(np.int64)

    target_spam = round(val_frac * is_spam.sum())
    target_non = round(val_frac * (~is_spam).sum())
    in_val = np.zeros(len(g_spam), dtype=bool)
    n_spam = n_non = 0
    for g in np.random.default_rng(seed).permutation(len(g_spam)):
        if n_spam + g_spam[g] <= target_spam and n_non + g_non[g] <= target_non:
            in_val[g] = True
            n_spam += g_spam[g]
            n_non += g_non[g]

    va_mask = in_val[group]
    tr, va = ids[~va_mask], ids[va_mask]
    te = s2["host_id"].to_numpy(np.int64)
    assert max(tr.max(), va.max(), te.max()) < n, "host id가 N 이상"
    return np.sort(tr), np.sort(va), np.sort(te)


def load_dataset(processed_dir="data/processed") -> Dataset:
    """data/processed/ → Dataset. B·C는 이 함수만 호출한다.

    상대 경로는 프로젝트 루트 기준으로 해석한다 (notebooks/ 등 어디서 호출해도 같음).
    """
    d = Path(processed_dir)
    if not d.is_absolute():
        d = ROOT / d
    missing = [name for name in PROCESSED_FILES if not (d / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{d}에 {', '.join(missing)} 없음. scripts/build_processed.py를 먼저 실행")

    A = sp.load_npz(d / "graph.npz").tocsr().astype(np.float64)
    graph = HostGraph(hostnames=load_hostnames(d / "hostnames.txt"), adjacency=A)
    n = graph.n

    labels_df = pd.read_csv(d / "labels.csv", usecols=["host_id", "label", "set"])
    for col, codes in (("label", LABEL_CODE), ("set", SET_CODE)):
        unknown = set(labels_df[col]) - set(codes)
        if unknown:
            raise ValueError(f"{d / 'labels.csv'}의 {col}에 알 수 없는 값: {unknown}")
    ids = labels_df["host_id"].to_numpy(np.int64)
    labels = np.full(n, UNLABELED, dtype=np.int8)
    labels[ids] = labels_df["label"].map(LABEL_CODE).to_numpy(np.int8)
    in_set = np.zeros(n, dtype=np.int8)
    in_set[ids] = labels_df["set"].map(SET_CODE).to_numpy(np.int8)

    with np.load(d / "split.npz") as s:
        train_idx, val_idx, test_idx = (s[k].astype(np.int64) for k in ("train_idx", "val_idx", "test_idx"))

    return Dataset(graph=graph, labels=labels, in_set=in_set,
                   train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
