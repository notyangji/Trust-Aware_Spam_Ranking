"""팀 공통 자료형: 라벨 상수, HostGraph, Dataset.

- 파일 입출력은 없다. src/data.py의 load_dataset()이 이 자료형으로 만들어 B·C에게 넘긴다.
- 잘못된 데이터(크기 불일치, self-loop, 분할 겹침 등)는 객체를 만드는 순간 AssertionError로 막는다.
- 파일 이름을 types.py로 하지 않는다: 표준 라이브러리 types를 가려 import가 깨진다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

# labels 배열 값. undecided(평가했지만 판정 불가)와 unlabeled(평가 안 함)를 섞지 않는다.
SPAM, NONSPAM, UNDECIDED, UNLABELED = 1, 0, -1, -2


@dataclass(frozen=True)
class HostGraph:
    """호스트 그래프. host id = hostnames의 index (원본 번호 0..N-1, 재번호 금지)."""

    hostnames: list[str]    # index = host id
    adjacency: csr_matrix   # A[i, j] = host i → host j의 link 수, float64, shape (N, N), self-loop 제거됨

    def __post_init__(self):
        n = len(self.hostnames)
        assert self.adjacency.shape == (n, n), "hostnames 수와 행렬 크기가 다름"
        assert not self.adjacency.diagonal().any(), "self-loop가 남아 있음"

    @property
    def n(self) -> int:
        return len(self.hostnames)


@dataclass(frozen=True)
class Dataset:
    """그래프 + 라벨 + 분할. B·C는 load_dataset()으로 받은 이 객체만 쓴다.

    그래프는 전체를 그대로 쓴다(transductive). test host도 그래프에 남아 있고,
    label만 seed 선택·조정에 쓰지 않는다.
    """

    graph: HostGraph
    labels: np.ndarray      # shape (N,), int8: SPAM / NONSPAM / UNDECIDED / UNLABELED
    in_set: np.ndarray      # shape (N,), int8: 1 = SET1, 2 = SET2, 0 = label 없음
    train_idx: np.ndarray   # host id(int64). SET1의 spam/nonspam 일부 (seed 선택·학습)
    val_idx: np.ndarray     # host id(int64). SET1의 나머지 (β 등 조정)
    test_idx: np.ndarray    # host id(int64). SET2의 spam/nonspam — 최종 평가에만 사용

    def __post_init__(self):
        n = self.graph.n
        assert self.labels.shape == (n,) and self.in_set.shape == (n,), "labels·in_set 길이가 N과 다름"
        tr, va, te = map(set, (self.train_idx, self.val_idx, self.test_idx))
        assert not (tr & va or tr & te or va & te), "분할이 겹침"
        assert (self.in_set[self.train_idx] == 1).all() and (self.in_set[self.val_idx] == 1).all(), \
            "train·val은 SET1만"
        assert (self.in_set[self.test_idx] == 2).all(), "test는 SET2만"
        for idx in (self.train_idx, self.val_idx, self.test_idx):
            assert np.isin(self.labels[idx], (SPAM, NONSPAM)).all(), "undecided·미표시 포함됨"
