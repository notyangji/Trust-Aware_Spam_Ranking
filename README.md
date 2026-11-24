# Trust-Aware_Spam_Ranking
- 사용 데이터셋: [WEBSPAM-UK2007](https://chato.cl/webspam/datasets/uk2007/) (`.uk` 웹 호스트 그래프 + 호스트 단위 스팸 라벨). 파일 형식·라벨·통계·라이선스는 [data/README.md](data/README.md) 참고.
- 프로젝트 목표: 

# 코드 구성

| 파일 | 내용 | 상태 |
|---|---|---|
| `configs/default.yaml` | 공통 설정 (경로, β, tol, 분할 비율, PPR 후보 도메인 등) | 작성 |
| `configs/seeds.yaml` | Topic·trusted seed 선정 방식, 개수, 난수값, spam 혼입률 | 작성 |
| `src/config.py` | 설정 파일 읽기 `load_config` (모든 script·notebook 공통) | 작성 |
| `src/common/schema.py` | 공통 자료형 `HostGraph`, `Dataset`, 라벨 상수 | 작성 |
| `src/data.py` | raw parsing, train/val/test 분할(사이트 단위), `load_dataset()` | 작성 |
| `src/graph.py` | 전이 행렬 `build_transition`, 도메인 판정 `domain_of`·`site_of` | 작성 |
| `src/rank.py` | 공통 `power_iteration`, `pagerank` | 작성 |
| `src/seeds.py` | Seed 설정·ID 검증, topic·trusted seed 선정, 개인화 벡터 구성 | 작성 |
| `src/baselines.py` | random·in-degree baseline 점수 | 작성 |
| `scripts/download_data.py` | 원본 4개 다운로드, 라벨 압축 해제 | 작성 |
| `scripts/build_processed.py` | raw → `data/processed/` (분할 포함) | 작성 |
| `scripts/run_baselines.py` | PageRank·baseline 점수 → `results/scores/` | 작성 |
| `tests/test_rank.py`, `tests/test_config.py` | 엔진·설정 테스트 | 작성 |
| `tests/test_seeds.py` | Seed 선정·검증 테스트 | 작성 |
| `notebooks/01_eda.ipynb` | EDA: 통계표, degree 분포, spam/nonspam 비교, TrustRank 가정, 도메인 집계, seed 도달 범위 | 작성 |

# 실행 순서

## 환경 설정
- Python 3.12
- 패키지: `pip install -r requirements.txt`

## 실행

프로젝트 루트에서 실행한다. 모든 script는 `--config`로 다른 설정 파일을 지정할 수 있다 (기본 `configs/default.yaml`).

```
python scripts/download_data.py      # 1. 원본 → data/raw/
python scripts/build_processed.py    # 2. data/raw/ → data/processed/
python scripts/run_baselines.py      # 3. PageRank·baseline 점수 → results/scores/
pytest                               # 엔진·설정·seed 테스트 (데이터 불필요)
```

| 단계 | 출력 | 시간 |
|---|---|---|
| 1. `download_data.py` | 원본 4개(약 8.6 MB), 라벨 압축 해제. 이미 받은 파일은 건너뜀 | 네트워크에 따라 |
| 2. `build_processed.py` | `graph.npz`, `hostnames.txt`, `labels.csv`, `split.npz`, `meta.json` | 약 3초 |
| 3. `run_baselines.py` | 점수 4개 (`.npy` + `.json`) | 약 1초 |

- 2단계는 점검 수치(N, edge 수, 라벨 수)가 기대값과 다르면 경고를 출력한다. 원본 SHA-256과 점검 통계는 `data/processed/meta.json`에 기록된다.
- 2단계를 다시 실행해 **분할이 바뀌면 "기존과 달라짐"이 출력된다 → B·C에게 공유**한다 (점수·평가를 다시 내야 함).
- `data/raw/`, `data/processed/`, `results/scores/`의 파일은 Git에 올리지 않는다. 각자 위 순서로 만든다.
- 원본 파일 상세와 SHA-256은 [data/README.md](data/README.md) 참고.

# 함수 사용법

함수별 사용 예시, 공통 자료형, 그래프·랭킹 함수의 인자와 반환값은 [함수와 공통 자료형](docs/function_reference.md)을 참고한다.

# 설정 (`configs/default.yaml`)

모든 script·notebook은 이 파일에서 설정값을 읽는다 (코드에 숫자를 직접 쓰지 않음). 실험 설정을 바꿀 때는 파일을 복사해 `configs/beta090.yaml`처럼 만들고 `--config`로 지정한다.

| 섹션 | 키 | 값 | 비고 |
|---|---|---|---|
| `paths` | `raw_dir` / `processed_dir` / `scores_dir` / `figures_dir` | `data/raw` / `data/processed` / `results/scores` / `results/figures` | 프로젝트 루트 기준 |
| `download` | `base_url` | WEBSPAM-UK2007 공식 주소 | |
| `rank` | `beta` | 0.85 | 링크를 따라갈 확률 |
| | `tol` | 1.0e-8 | `1e-8`로 쓰면 문자열로 읽히므로 반드시 `1.0e-8` |
| | `max_iter` | 300 | |
| | `dangling` | `teleport` | dead end 처리 (`teleport` / `uniform`) |
| `split` | `val_frac` / `random_seed` | 0.3 / 42 | 팀 합의값. val spam이 부족하면 더 늘려 봄 |
| `baselines` | `random_seed` | 42 | |
| `domains` | `ppr_seed_domains` | `[ac.uk, gov.uk]` | B와 결정 |

Seed 선정 조건은 [configs/seeds.yaml](configs/seeds.yaml)에 둔다. Topic은 기본 도메인의 후보 전체를 사용하고, trusted는 학습용 nonspam에서 난수값 42로 100개를 고른다. 이 값은 첫 실험의 기준 조건이다.

# 점수 파일 (`results/scores/`)

모든 방법(PageRank, baseline, B의 PPR·TrustRank 등)의 점수는 같은 형식으로 저장해 C가 같은 방식으로 평가한다.

| 파일 | 형식 |
|---|---|
| `<method>_<config>.npy` | 길이 N float64, index = host id |
| `<method>_<config>.json` | `method`, `higher_is_spam`, 사용한 설정값(`config`), 수렴 정보(`info`, power iteration만), `n`, `created` |

**점수 방향 `higher_is_spam`**: 방법마다 원래 점수를 그대로 저장하고, 방향은 `.json`에 기록한다.

| 값 | 의미 | 예 |
|---|---|---|
| `false` | 클수록 spam 아님 | PageRank, TrustRank |
| `true` | 클수록 spam 의심 | in-degree baseline, spam mass |
| `null` | 방향 없음 | random |

현재 `run_baselines.py`가 만드는 파일 (β = 0.85):

| 파일 | `higher_is_spam` | 비고 |
|---|---|---|
| `pagerank_binary_b085` | `false` | 이진 그래프, 81회 수렴. B의 spam mass, C의 E1에 사용 |
| `pagerank_weighted_b085` | `false` | 링크 수 가중 그래프, 83회 수렴 (E5) |
| `random_seed42` | `null` | [0, 1) 균등 난수 |
| `indegree` | `true` | `−log1p(in_deg)`, in_deg = 링크를 거는 host 수 |

- β를 바꾼 config로 실행하면 `pagerank_binary_b090`처럼 파일 이름으로 구분된다.
- 저장 형식은 `scripts/run_baselines.py`의 `save_scores()`를 참고한다.

# 테스트 (`tests/`)

```
pytest
```

프로젝트 루트에서 실행한다. 데이터·네트워크 없이 1초 안에 끝난다. 엔진(`graph.py`, `rank.py`)이나 설정 코드를 고친 뒤에는 merge 전에 반드시 실행한다. `pytest.ini`가 루트를 import 경로에 넣어 주므로 `src`를 바로 import할 수 있다 (테스트 파일을 `python tests/...`로 직접 실행하면 `src`를 찾지 못함).

**`tests/test_rank.py`** (22개)

| 그룹 | 내용 | 허용 오차 |
|---|---|---|
| build_transition | 강의 예제 행렬, dead end 열 = 0, 이진·가중 차이, 원본 보존 | — |
| test 1–4 | 강의 예제 (Flow, Spider trap, Dead end, Topic-Specific) | 1e-3 |
| test 5–6 | 강의 표 (seed 집합·β 변화) | 0.01 |
| test 7–8 | 무작위 300-node 그래프에서 `networkx.pagerank`와 비교 | 1e-12 |
| 불변식 | 합 = 1, ≥ 0, seed에서 도달 불가능한 host `== 0` | 정확히 |
| 예외·경고 | `ValueError`, `RuntimeWarning`, 빈 seed `AssertionError` | — |

- `networkx`는 결과 검증(test 7–8)에만 쓰며, 이 프로젝트의 알고리즘이 아니다. test 7은 `networkx`의 기본 dead end 처리(personalization으로 보냄)를 `dangling="teleport"`와, test 8은 균등 `dangling`을 `"uniform"`과 대응시킨다.

**`tests/test_config.py`** (7개): 기본 설정 읽기, 현재 폴더와 무관한 경로 해석, 없는 파일·없는 키·잘못된 형식의 오류.

**`tests/test_seeds.py`**는 가상 데이터로 잘못된 입력과 후보 부족 처리, 선정 재현성, 학습 분할 제한, 혼입률 기록, 개인화 벡터 생성을 검증한다.
