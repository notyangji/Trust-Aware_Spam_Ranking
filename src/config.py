"""설정 파일(configs/*.yaml) 읽기. 모든 script·notebook이 이 함수로 설정을 읽는다.

    from src.config import load_config

    cfg = load_config()                      # configs/default.yaml
    cfg = load_config("configs/beta090.yaml")
    beta = cfg.require("rank", "beta")       # 키가 없으면 어떤 키가 없는지 알려주는 ValueError
    scores_dir = cfg.path("scores_dir")      # paths.* → 프로젝트 루트 기준 절대 경로

상대 경로(설정 파일 경로, paths.*)는 모두 프로젝트 루트 기준으로 해석한다.
그래서 scripts/, notebooks/ 등 어디서 실행해도 같은 파일을 가리킨다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/default.yaml"


def resolve_path(path: str | Path) -> Path:
    """상대 경로는 프로젝트 루트 기준으로, 절대 경로는 그대로."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class Config(dict):
    """yaml 내용을 담은 dict. cfg["rank"]["beta"]처럼 그대로 쓸 수 있고, 필수 키 확인·경로 해석을 더한다."""

    def __init__(self, data: dict, source: Path):
        super().__init__(data)
        self.source = source                 # 읽어 온 설정 파일 경로 (오류 메시지용)

    def require(self, section: str, key: str) -> Any:
        """section.key 값. 없거나 비어 있으면 ValueError."""
        value = (self.get(section) or {}).get(key)
        if value is None:
            raise ValueError(f"{self.source}에 설정 키 {section}.{key} 없음")
        return value

    def path(self, key: str) -> Path:
        """paths.key 값을 프로젝트 루트 기준 절대 경로로."""
        return resolve_path(self.require("paths", key))


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """설정 파일을 읽어 Config로 돌려준다. 파일이 없거나 최상위가 dict가 아니면 오류."""
    source = resolve_path(path)
    if not source.exists():
        raise FileNotFoundError(f"설정 파일 없음: {source}")
    with open(source, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{source}: 최상위가 key: value 형식이 아님")
    return Config(data, source)
