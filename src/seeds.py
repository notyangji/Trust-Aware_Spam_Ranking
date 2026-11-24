from collections.abc import Mapping
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

from src.common.schema import NONSPAM, SPAM
from src.graph import UK_SLD, domain_of


def validate_seed_ids(seed_ids, n_nodes):
    """호스트 ID를 검사하고 원래 순서의 1차원 int64 배열로 반환한다.

    seed_ids는 비어 있지 않은 정수 ID 목록이고 n_nodes는 전체 호스트 수다.
    호스트 수가 양의 정수가 아니거나 ID에 중복 또는 범위 밖 값이 있으면 ValueError를 낸다.
    """
    if (isinstance(n_nodes, (bool, np.bool_))
            or not isinstance(n_nodes, (int, np.integer))
            or n_nodes <= 0):
        raise ValueError("n_nodes는 0보다 큰 정수여야 합니다.")

    # 원래 타입을 보존해서 [1, True]의 True가 정수 1로 바뀌는 것을 막는다.
    seed_ids = np.asarray(seed_ids, dtype=object)
    if seed_ids.ndim != 1:
        raise ValueError("seed_ids는 1차원 목록이어야 합니다.")
    if len(seed_ids) == 0:
        raise ValueError("seed를 하나 이상 선택해야 합니다.")

    if any(isinstance(i, (bool, np.bool_))
           or not isinstance(i, (int, np.integer)) for i in seed_ids):
        raise ValueError("seed ID는 bool이나 소수가 아닌 정수여야 합니다.")
    if any(i < 0 or i >= n_nodes for i in seed_ids):
        raise ValueError(f"seed ID는 0 이상 {n_nodes} 미만이어야 합니다.")
    if np.unique(seed_ids).size != len(seed_ids):
        raise ValueError("seed ID가 중복되어 있습니다.")

    return np.asarray(seed_ids, dtype=np.int64)


def make_personalization(n_nodes, seed_ids):
    """선택한 seed에 같은 비중을 주는 개인화 벡터를 만든다.

    n_nodes는 전체 호스트 수이고 seed_ids는 검증할 호스트 ID 목록이다.
    길이가 n_nodes이고 합이 1인 float64 배열을 반환한다.
    """
    seed_ids = validate_seed_ids(seed_ids, n_nodes)
    s = np.zeros(n_nodes, dtype=np.float64)
    s[seed_ids] = 1.0 / len(seed_ids)
    return s


def _validate_random_seed_options(name, n_seeds, random_seed):
    # YAML에서 읽은 정수만 받는다. bool은 정수로 취급하지 않는다.
    if type(n_seeds) is not int or n_seeds <= 0:
        raise ValueError(f"{name}.n_seeds는 0보다 큰 정수여야 합니다.")
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError(f"{name}.random_seed는 0 이상의 정수여야 합니다.")


def _validate_contamination_rate(rate):
    if type(rate) not in (int, float) or not 0 <= rate <= 1:
        raise ValueError("trusted.contamination_rate는 0~1 사이의 수여야 합니다.")


def validate_seed_config(seed_cfg):
    """읽어 온 seed 설정의 구조와 값을 검사한다. 파일을 직접 읽지는 않는다.

    topic은 all 또는 random, trusted는 random을 지원한다.
    random에는 양의 정수 n_seeds와 0 이상의 정수 random_seed가 필요하다.
    trusted.contamination_rate는 0~1 사이의 수여야 한다.
    잘못된 설정이면 ValueError를 내고, 통과하면 반환값 없이 끝난다.
    실제 후보 수가 충분한지는 이후 seed 선정 함수에서 검사한다.
    """
    if not isinstance(seed_cfg, Mapping):
        raise ValueError("seed 설정은 key: value 형식이어야 합니다.")
    for name in ("topic", "trusted"):
        if not isinstance(seed_cfg.get(name), Mapping):
            raise ValueError(f"{name} 설정이 필요하며 key: value 형식이어야 합니다.")

    topic = seed_cfg["topic"]
    trusted = seed_cfg["trusted"]
    unknown_topic_keys = set(topic) - {"strategy", "n_seeds", "random_seed"}
    if unknown_topic_keys:
        keys = ", ".join(sorted(map(repr, unknown_topic_keys)))
        raise ValueError(f"topic에 지원하지 않는 설정 키가 있습니다: {keys}")

    if topic.get("strategy") not in ("all", "random"):
        raise ValueError("topic.strategy는 all 또는 random이어야 합니다.")
    if trusted.get("strategy") != "random":
        raise ValueError("trusted.strategy는 현재 random만 지원합니다.")

    if topic["strategy"] == "all":
        if topic.get("n_seeds") is not None or topic.get("random_seed") is not None:
            raise ValueError("topic.strategy가 all이면 n_seeds와 random_seed를 지정하지 않습니다.")

    for name, settings in (("topic", topic), ("trusted", trusted)):
        if settings["strategy"] == "random":
            _validate_random_seed_options(name, settings.get("n_seeds"), settings.get("random_seed"))

    _validate_contamination_rate(trusted.get("contamination_rate"))


def select_topic_seeds(
    hostnames, domains, *, strategy="all", n_seeds=None, random_seed=None, return_info=False
):
    """지정한 도메인에서 topic seed를 고른다.

    매개변수
    - hostnames (list[str]) — 전체 호스트의 주소 목록. 배열의 위치가 원본 호스트 ID다.
    - domains (list[str]) — 후보로 삼을 도메인 목록. 예를 들어 ["ac.uk", "gov.uk"].
    - strategy (str, 기본값 "all") — "all"은 후보 전체, "random"은 무작위로 일부를 고른다.
    - n_seeds (int | None, 기본값 None) — "random"에서 고를 개수. 양의 정수를 지정한다.
    - random_seed (int | None, 기본값 None) — "random"의 추첨 순서를 재현하는 0 이상의 정수.
    - return_info (bool, 기본값 False) — True이면 ID 배열과 선정 기록을 함께 반환한다.

    strategy가 "all"이면 n_seeds와 random_seed는 기본값 None으로 둔다.
    선택한 원본 호스트 ID를 오름차순의 1차원 int64 배열로 반환한다.
    지원하지 않는 도메인이 있거나 후보 수가 부족하면 ValueError를 낸다.
    """
    if strategy not in ("all", "random"):
        raise ValueError("topic.strategy는 all 또는 random이어야 합니다.")
    if strategy == "random":
        _validate_random_seed_options("topic", n_seeds, random_seed)
    elif n_seeds is not None or random_seed is not None:
        raise ValueError("topic.strategy가 all이면 n_seeds와 random_seed를 지정하지 않습니다.")

    wanted_domains = set(domains)
    supported_domains = {f"{sld}.uk" for sld in UK_SLD} | {"other"}
    unsupported_domains = wanted_domains - supported_domains
    if unsupported_domains:
        names = ", ".join(sorted(map(repr, unsupported_domains)))
        raise ValueError(f"지원하지 않는 topic 도메인이 있습니다: {names}")

    seed_ids = []
    for host_id, hostname in enumerate(hostnames):
        if domain_of(hostname) in wanted_domains:
            seed_ids.append(host_id)

    if not seed_ids:
        raise ValueError("지정한 도메인에 해당하는 topic seed 후보가 없습니다.")

    candidates = np.array(seed_ids, dtype=np.int64)
    if strategy == "all":
        selected_ids = candidates
    else:
        assert n_seeds is not None
        if n_seeds > candidates.size:
            raise ValueError(f"topic seed 후보 {candidates.size}개에서 {n_seeds}개를 고를 수 없습니다.")

        rng = np.random.default_rng(random_seed)
        shuffled = rng.permutation(candidates)
        selected_ids = np.sort(shuffled[:n_seeds])

    if not return_info:
        return selected_ids

    info = {
        "kind": "topic",
        "strategy": strategy,
        "domains": sorted(wanted_domains),
        "random_seed": random_seed,
        "candidate_count": int(candidates.size),
        "n_requested": n_seeds,
        "n_selected": int(selected_ids.size),
        "seed_ids": selected_ids.tolist(),
    }
    return selected_ids, info


def select_trusted_seeds(ds, *, n_seeds, random_seed, contamination_rate=0.0, return_info=False):
    """학습용 호스트에서 TrustRank seed를 고르고 지정한 비율만큼 SPAM을 섞는다.

    매개변수
    - ds (Dataset) — A의 load_dataset()으로 읽은 데이터. train_idx에 해당하는 라벨을 읽는다.
    - n_seeds (int) — 고를 seed의 총개수. 1 이상의 정수다.
    - random_seed (int) — 추첨 순서를 재현하는 0 이상의 정수.
    - contamination_rate (int | float, 기본값 0.0) — 전체 seed 중 SPAM의 비율. 0~1이다.
    - return_info (bool, 기본값 False) — True이면 ID 배열과 요청·실제 혼입률 등을 함께 반환한다.

    SPAM 개수는 n_seeds에 혼입률을 곱해 반올림한다. 정확히 .5이면 올린다.
    나머지는 NONSPAM에서 고르며, 어느 후보군이든 요청한 수보다 적으면 ValueError를 낸다.
    원본 호스트 ID를 오름차순의 1차원 int64 배열로 반환한다.
    """
    _validate_random_seed_options("trusted", n_seeds, random_seed)
    _validate_contamination_rate(contamination_rate)

    train_ids = ds.train_idx
    if np.any((train_ids < 0) | (train_ids >= ds.graph.n)):
        raise ValueError("train_idx에 범위 밖 호스트 ID가 있습니다.")

    train_labels = ds.labels[train_ids]
    nonspam_candidates = np.unique(train_ids[train_labels == NONSPAM])
    spam_candidates = np.unique(train_ids[train_labels == SPAM])

    requested_spam = Decimal(n_seeds) * Decimal(str(contamination_rate))
    n_spam = int(requested_spam.to_integral_value(rounding=ROUND_HALF_UP))
    n_nonspam = n_seeds - n_spam

    if n_nonspam > nonspam_candidates.size:
        raise ValueError(f"trusted seed 후보 중 NONSPAM {nonspam_candidates.size}개에서 "
                         f"{n_nonspam}개를 고를 수 없습니다.")
    if n_spam > spam_candidates.size:
        raise ValueError(f"trusted seed 후보 중 SPAM {spam_candidates.size}개에서 "
                         f"{n_spam}개를 고를 수 없습니다.")

    rng = np.random.default_rng(random_seed)
    # 후보군마다 전체 순서를 먼저 정해, 뽑는 개수를 바꿔도 추첨 순서를 유지한다.
    nonspam_ids = rng.permutation(nonspam_candidates)[:n_nonspam]
    spam_ids = rng.permutation(spam_candidates)[:n_spam]
    selected_ids = np.sort(np.concatenate((nonspam_ids, spam_ids)))

    if not return_info:
        return selected_ids

    info = {
        "kind": "trusted",
        "strategy": "random",
        "random_seed": random_seed,
        "candidate_count": int(nonspam_candidates.size + spam_candidates.size),
        "candidate_counts": {"nonspam": int(nonspam_candidates.size), "spam": int(spam_candidates.size)},
        "n_requested": n_seeds,
        "n_selected": int(selected_ids.size),
        "selected_counts": {"nonspam": n_nonspam, "spam": n_spam},
        "requested_contamination_rate": float(contamination_rate),
        "actual_contamination_rate": n_spam / selected_ids.size,
        "seed_ids": selected_ids.tolist(),
    }
    return selected_ids, info
