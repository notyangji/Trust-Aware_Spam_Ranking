from collections.abc import Mapping

import numpy as np


def make_personalization(n_nodes, seed_ids):
    """선택한 seed에 같은 비중을 주는 개인화 벡터를 만든다.
    n_nodes: 호스트 수, seed_ids: seed 호스트 ID 목록
    return: 길이가 n_nodes이고 합이 1인 float64 배열

    n_nodes가 양의 정수인지, seed_ids가 비어 있지 않은 1차원 정수 목록인지 검사한다.
    범위 밖 ID와 중복 ID는 허용하지 않는다.
    """

    # 오류 체크
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

    # 실제 계산
    seed_ids = np.asarray(seed_ids, dtype=np.int64)
    s = np.zeros(n_nodes, dtype=np.float64)
    s[seed_ids] = 1.0 / len(seed_ids)
    return s


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
    if topic.get("strategy") not in ("all", "random"):
        raise ValueError("topic.strategy는 all 또는 random이어야 합니다.")
    if trusted.get("strategy") != "random":
        raise ValueError("trusted.strategy는 현재 random만 지원합니다.")

    if topic["strategy"] == "all":
        if topic.get("n_seeds") is not None or topic.get("random_seed") is not None:
            raise ValueError("topic.strategy가 all이면 n_seeds와 random_seed를 지정하지 않습니다.")

    for name, settings in (("topic", topic), ("trusted", trusted)):
        if settings["strategy"] == "random":
            count = settings.get("n_seeds")
            seed = settings.get("random_seed")
            # YAML에서 읽은 정수만 받는다. bool은 정수로 취급하지 않는다.
            if type(count) is not int or count <= 0:
                raise ValueError(f"{name}.n_seeds는 0보다 큰 정수여야 합니다.")
            if type(seed) is not int or seed < 0:
                raise ValueError(f"{name}.random_seed는 0 이상의 정수여야 합니다.")

    rate = trusted.get("contamination_rate")
    if type(rate) not in (int, float) or not 0 <= rate <= 1:
        raise ValueError("trusted.contamination_rate는 0~1 사이의 수여야 합니다.")
