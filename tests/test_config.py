"""src/config.py 검증. 데이터 없이 실행된다."""

import pytest

from src.config import ROOT, load_config, resolve_path


def test_default_config_loads():
    cfg = load_config()
    assert cfg.require("rank", "beta") == 0.85
    assert isinstance(cfg.require("rank", "tol"), float)      # "1e-8"처럼 써서 문자열이 되면 실패
    assert cfg.path("scores_dir") == ROOT / "results" / "scores"
    assert cfg["split"]["val_frac"] == 0.3                    # dict처럼도 쓸 수 있음


def test_relative_path_is_from_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                               # 현재 폴더가 달라도
    assert load_config("configs/default.yaml").source == ROOT / "configs" / "default.yaml"
    assert resolve_path(tmp_path) == tmp_path                 # 절대 경로는 그대로


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/없는파일.yaml")


@pytest.mark.parametrize("text, missing", [
    ("", "rank.beta"),                                        # 빈 파일
    ("rank:\n", "rank.beta"),                                 # 섹션만 있음
    ("rank:\n  betta: 0.85\n", "rank.beta"),                  # 키 오타
])
def test_missing_key_raises(tmp_path, text, missing):
    p = tmp_path / "bad.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=missing):
        load_config(p).require("rank", "beta")


def test_non_mapping_raises(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
