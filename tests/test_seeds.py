"""작은 가상 데이터로 seed 선정과 개인화 벡터를 확인한다."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.common.schema import NONSPAM, SPAM, Dataset, HostGraph
from src.seeds import make_personalization, select_topic_seeds, select_trusted_seeds, validate_seed_ids


HOSTNAMES = [
    "shop.example.co.uk",
    "www.first.ac.uk",
    "www.office.gov.uk",
    "www.second.ac.uk",
    "www.third.ac.uk",
]


@pytest.fixture
def dataset():
    # train의 2번은 spam이고, val의 1번과 test의 4번은 nonspam이다.
    n = len(HOSTNAMES)
    return Dataset(
        graph=HostGraph(hostnames=HOSTNAMES, adjacency=csr_matrix((n, n))),
        labels=np.array([NONSPAM, NONSPAM, SPAM, NONSPAM, NONSPAM], dtype=np.int8),
        in_set=np.array([1, 1, 1, 1, 2], dtype=np.int8),
        train_idx=np.array([0, 2, 3], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
        test_idx=np.array([4], dtype=np.int64),
    )


def test_topic_all_returns_original_host_ids():
    # 지정한 도메인의 호스트를 모두 골라 원본 ID를 반환하는지 확인한다.
    seed_ids = select_topic_seeds(HOSTNAMES, ["ac.uk", "gov.uk"])

    np.testing.assert_array_equal(seed_ids, [1, 2, 3, 4])
    assert seed_ids.dtype == np.int64


def test_topic_random_is_reproducible_and_keeps_existing_seeds():
    # 같은 난수값으로 다시 뽑거나 개수를 늘렸을 때 기존 seed가 유지되는지 확인한다.
    options = {"strategy": "random", "random_seed": 42}
    small = select_topic_seeds(HOSTNAMES, ["ac.uk"], n_seeds=1, **options)
    repeated = select_topic_seeds(HOSTNAMES, ["ac.uk"], n_seeds=1, **options)
    large = select_topic_seeds(HOSTNAMES, ["ac.uk"], n_seeds=2, **options)

    np.testing.assert_array_equal(small, repeated)
    assert small.size == 1
    assert large.size == 2
    assert len(set(large)) == large.size
    assert set(small).issubset(large)
    assert set(large).issubset({1, 3, 4})


def test_topic_random_rejects_request_larger_than_pool():
    # Topic 후보보다 많이 요청하면 ValueError가 발생하는지 확인한다.
    with pytest.raises(ValueError, match="topic seed 후보"):
        select_topic_seeds(HOSTNAMES, ["gov.uk"], strategy="random", n_seeds=2, random_seed=42)


def test_trusted_seeds_use_only_training_nonspam(dataset):
    # 학습용 호스트 중 NONSPAM인 호스트만 선택하는지 확인한다.
    seed_ids = select_trusted_seeds(dataset, n_seeds=2, random_seed=42)

    np.testing.assert_array_equal(seed_ids, [0, 3])
    assert seed_ids.dtype == np.int64


def test_trusted_seeds_reject_request_larger_than_training_nonspam_pool(dataset):
    # 학습용 NONSPAM 후보보다 많이 요청하면 ValueError가 발생하는지 확인한다.
    with pytest.raises(ValueError, match="NONSPAM"):
        select_trusted_seeds(dataset, n_seeds=3, random_seed=42)


def test_trusted_seeds_round_half_up_and_keep_total_count(dataset):
    # 2개의 25%는 0.5이므로 SPAM 1개를 섞고 같은 추첨 순서의 NONSPAM 1개를 유지한다.
    clean = select_trusted_seeds(dataset, n_seeds=1, random_seed=42)
    mixed = select_trusted_seeds(dataset, n_seeds=2, random_seed=42, contamination_rate=0.25)

    assert mixed.size == 2
    np.testing.assert_array_equal(mixed[dataset.labels[mixed] == NONSPAM], clean)
    np.testing.assert_array_equal(mixed[dataset.labels[mixed] == SPAM], [2])


def test_trusted_seeds_exclude_validation_and_test_spam(dataset):
    # Val과 test에 SPAM이 있어도 학습용 SPAM 후보가 부족하면 오류를 낸다.
    labels = dataset.labels.copy()
    labels[dataset.val_idx] = SPAM
    labels[dataset.test_idx] = SPAM
    dataset = replace(dataset, labels=labels)

    with pytest.raises(ValueError, match=r"\bSPAM\b"):
        select_trusted_seeds(dataset, n_seeds=2, random_seed=42, contamination_rate=1.0)


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), True])
def test_trusted_seeds_reject_invalid_contamination_rate(dataset, rate):
    # 범위를 벗어난 수, NaN, 불리언은 혼입률로 받을 수 없다.
    with pytest.raises(ValueError, match="contamination_rate"):
        select_trusted_seeds(dataset, n_seeds=2, random_seed=42, contamination_rate=rate)


def test_validate_seed_ids_preserves_input_order():
    # 유효한 ID를 원래 순서의 int64 배열로 돌려준다.
    ids = validate_seed_ids([3, 1], n_nodes=5)

    np.testing.assert_array_equal(ids, [3, 1])
    assert ids.dtype == np.int64


@pytest.mark.parametrize("seed_ids", [[], [1, 1], [-1], [5], [1, True]])
def test_validate_seed_ids_rejects_invalid_ids(seed_ids):
    # 빈 목록, 중복, 범위 밖 ID, 불리언을 거부한다.
    with pytest.raises(ValueError):
        validate_seed_ids(seed_ids, n_nodes=5)


def test_topic_ids_can_be_used_to_make_personalization():
    # 선택한 정수 ID가 균등 비중을 담은 float64 벡터로 변환되는지 확인한다.
    seed_ids = select_topic_seeds(HOSTNAMES, ["ac.uk", "gov.uk"])
    weights = make_personalization(len(HOSTNAMES), seed_ids)

    np.testing.assert_array_equal(weights, [0.0, 0.25, 0.25, 0.25, 0.25])
    assert weights.dtype == np.float64
