# Trust-Aware_Spam_Ranking
- 사용 데이터셋: [WEBSPAM-UK2007](https://chato.cl/webspam/datasets/uk2007/) (`.uk` 웹 호스트 그래프 + 호스트 단위 스팸 라벨). 파일 형식·라벨·통계·라이선스는 [data/README.md](data/README.md) 참고.
- 프로젝트 목표: 

# 코드 구성

| 파일 | 내용 | 상태 |
|---|---|---|
| `configs/default.yaml` | 모든 설정값 (경로, β, tol, 분할 비율, seed 등) | 작성 |
| `scripts/download_data.py` | 원본 4개 다운로드, 라벨 압축 해제 | 작성 |
| `src/common/schema.py` | 공통 자료형 `HostGraph`, `Dataset`, 라벨 상수 | 작성 |
| `src/data.py` | raw parsing, train/val/test 분할, `load_dataset()` | 작성 예정 |
| `src/graph.py` | 전이 행렬 `build_transition`, 도메인 판정 `domain_of` | 작성 |
| `src/rank.py` | 공통 `power_iteration`, `pagerank` | 작성 |
| `tests/test_rank.py` | 강의 예제·networkx 교차 확인·불변식·예외 테스트 | 작성 |
| `scripts/build_processed.py` | raw → `data/processed/` | 작성 예정 |
| `src/baselines.py`, `scripts/run_baselines.py` | random·in-degree baseline, PageRank 점수 저장 | 작성 예정 |
| `notebooks/01_eda.ipynb` | EDA | 작성 예정 |

# 사용 예시

프로젝트 루트에서 실행하는 기준이다. 설정값은 `configs/default.yaml`에서 읽는다.

```python
import numpy as np
import yaml

cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
```

아래 예시의 `ds`는 `src/data.py`의 `load_dataset()`(작성 예정)으로 받는 `Dataset`이다 (자료형은 아래 "공통 자료형" 참고).

**전이 행렬·도메인 (`src/graph.py`)**

```python
from src.graph import build_transition, domain_of

M = build_transition(ds.graph.adjacency)     # column-stochastic 전이 행렬
domain_of("www.ox.ac.uk")                    # "ac.uk"
```

**PageRank·TrustRank (`src/rank.py`)**: `cfg["rank"]`를 그대로 넘기면 β, tol, max_iter, dangling이 적용된다.

```python
from src.common.schema import NONSPAM
from src.rank import pagerank, power_iteration

r = pagerank(M, **cfg["rank"])               # PageRank
top10 = [ds.graph.hostnames[i] for i in np.argsort(-r)[:10]]

s = np.zeros(ds.graph.n)                     # TrustRank: train의 nonspam host를 seed로
s[ds.train_idx[ds.labels[ds.train_idx] == NONSPAM]] = 1.0
t = power_iteration(M, teleport=s, **cfg["rank"])   # seed에서 도달 불가능한 host는 0
```

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
| `split` | `val_frac` / `random_seed` | 0.2 / 42 | B·C와 합의 후 확정 |
| `baselines` | `random_seed` | 42 | |
| `domains` | `ppr_seed_domains` | `[ac.uk, gov.uk]` | B와 결정 |

# 공통 자료형 (`src/common/schema.py`)

팀 전원이 같이 쓰는 자료형을 정의한다. `src/data.py`의 `load_dataset()`(작성 예정)이 `data/processed/`를 읽어 `Dataset`을 만들고, B·C는 이 객체만 받아 쓴다.

## 라벨 상수

| 상수 | 값 | 의미 |
|---|---:|---|
| `SPAM` | 1 | spam |
| `NONSPAM` | 0 | nonspam |
| `UNDECIDED` | −1 | 평가했지만 판정 불가 |
| `UNLABELED` | −2 | 평가하지 않음 (라벨 없음) |

## `HostGraph`

| 필드 | 타입 | 내용 |
|---|---|---|
| `hostnames` | `list[str]`, 길이 N | index = host id (원본 번호 0 ~ 114,528, 재번호 금지) |
| `adjacency` | `csr_matrix`, float64, (N, N) | `A[i, j]` = host i → host j의 링크 수 (행 = 출발, 열 = 도착). self-loop 제거됨 |
| `n` (속성) | `int` | 호스트 수 N |

- `adjacency`는 0이 아닌 칸만 저장하는 희소 행렬(CSR)이다. 약 22 MB.

## `Dataset`

| 필드 | 타입 | 내용 |
|---|---|---|
| `graph` | `HostGraph` | 전체 그래프 |
| `labels` | `np.ndarray`, int8, (N,) | 라벨 상수 값 |
| `in_set` | `np.ndarray`, int8, (N,) | 1 = SET1, 2 = SET2, 0 = 라벨 없음 |
| `train_idx` | `np.ndarray`, int64 | SET1의 spam·nonspam 일부 (seed 선택·학습) |
| `val_idx` | `np.ndarray`, int64 | SET1의 나머지 (β 등 조정) |
| `test_idx` | `np.ndarray`, int64 | SET2의 spam·nonspam (최종 평가에만 사용) |

- 분할 배열의 값은 **host id**이므로 `ds.labels[ds.train_idx]`처럼 바로 쓸 수 있다.
- 세 분할 모두 undecided와 라벨 없는 호스트는 제외한다.
- 그래프는 전체를 그대로 쓴다. test 호스트도 그래프에 남아 있고, 라벨만 seed 선택·조정에 쓰지 않는다.

## 생성 시 검사

두 클래스 모두 `frozen=True`라 필드를 다른 값으로 바꿔 끼울 수 없고, 데이터가 잘못되면 객체를 만드는 순간 `AssertionError`가 난다.

| 클래스 | 검사 |
|---|---|
| `HostGraph` | 행렬 크기 = (hostname 수, hostname 수), self-loop 없음 |
| `Dataset` | `labels`·`in_set` 길이 = N, 분할끼리 겹치지 않음, train·val은 SET1만, test는 SET2만, 분할에 undecided·라벨 없음 미포함 |

- `frozen=True`는 배열 안의 값 수정(`ds.labels[0] = 1`)까지 막지는 않으므로, 받은 배열을 직접 고치지 않는다.

# 그래프·랭킹 (`src/graph.py`, `src/rank.py`)

## 행렬 방향

행렬 방향은 `build_transition` 한 곳에서만 정한다. B는 `build_transition`이 준 `M`만 쓰고 직접 전치하지 않는다.

| 행렬 | 방향 | 의미 |
|---|---|---|
| `A` (adjacency) | 행 = 출발, 열 = 도착 | `A[i, j]` = host i → host j의 링크 수 |
| `M` (전이 행렬) | column-stochastic (강의와 같음) | `M[j, i]` = i에서 j로 갈 확률. dead end 열은 0 |

```python
from src.graph import build_transition, domain_of
from src.rank import pagerank, power_iteration

A = ds.graph.adjacency
M     = build_transition(A)                  # 이진 (기본, 강의식 1/d_i)
M_w   = build_transition(A, weighted=True)   # 링크 수 가중
M_rev = build_transition(A.T.tocsr())        # 뒤집은 그래프 (Inverse PR)
dom   = [domain_of(h) for h in ds.graph.hostnames]   # "ac.uk", "gov.uk", ..., 목록 밖은 "other"
```

- `build_transition`은 원본 `A`를 바꾸지 않는다 (내부에서 복사).
- `domain_of`는 대소문자, 끝의 `.`, 포트(`:8080`)를 무시한다. `UK_SLD`(ac, co, gov, org, ltd, plc, me, net, nhs, police, sch, mod, nic) 밖은 `"other"`이며, 실제 데이터에서 `other`는 28개(`bl.uk`, `parliament.uk` 등)뿐이다.

## `power_iteration`

```python
r = pagerank(M)                                  # PageRank (teleport 균등)
r = power_iteration(M, teleport=s)               # PPR·TrustRank (s: 길이 N, seed 위치만 0이 아님)
r, info = pagerank(M, return_info=True)          # info: iterations, residual, converged, time_sec
```

| 인자 | 기본값 | 내용 |
|---|---|---|
| `teleport` | `None` (균등) | 길이 N, ≥ 0, 합 > 0. 함수 안에서 합 1로 정규화 |
| `beta` / `tol` / `max_iter` | 0.85 / 1e-8 / 300 | 실제 값은 `configs/default.yaml`의 `rank`에서 읽어 넘긴다 |
| `dangling` | `"teleport"` | `"teleport"`: dead end 누출분을 s로 → seed에서 도달 불가능한 host는 **정확히 0**. `"uniform"`: dead end 누출분을 모든 host에 균등 분배 |

- 결과 `r`은 길이 N float64, 합 = 1, 모두 ≥ 0이다.
- 초기값은 `r(0) = s`이다. 1/N으로 시작하면 도달 불가능한 host 점수가 0이 되지 않으므로 바꾸지 않는다.
- `max_iter` 안에 수렴하지 못하면 `RuntimeWarning`을 내고 마지막 `r`을 돌려준다. `dangling`에 다른 값이 오면 `ValueError`.

실제 그래프에서의 수렴 (`tol` 1e-8):

| | β = 0.85 | β = 0.90 |
|---|---|---|
| PageRank (이진) | 81회, 0.16초 | 123회, 0.30초 |
| PageRank (가중) | 83회, 0.20초 | 126회, 0.27초 |
| Inverse PR | 69회, 0.15초 | 99회, 0.20초 |

# 테스트 (`tests/test_rank.py`)

데이터·네트워크 없이 1초 안에 끝난다. 엔진(`graph.py`, `rank.py`)을 고친 뒤에는 merge 전에 반드시 실행한다.

| 그룹 | 내용 | 허용 오차 |
|---|---|---|
| build_transition | 강의 예제 행렬, dead end 열 = 0, 이진·가중 차이, 원본 보존 | — |
| test 1–4 | 강의 예제 (Flow, Spider trap, Dead end, Topic-Specific) | 1e-3 |
| test 5–6 | 강의 표 (seed 집합·β 변화) | 0.01 |
| test 7–8 | 무작위 300-node 그래프에서 `networkx.pagerank`와 비교 | 1e-12 |
| 불변식 | 합 = 1, ≥ 0, seed에서 도달 불가능한 host `== 0` | 정확히 |
| 예외·경고 | `ValueError`, `RuntimeWarning`, 빈 seed `AssertionError` | — |

- `networkx`는 결과 검증(test 7–8)에만 쓰며, 이 프로젝트의 알고리즘이 아니다. test 7은 `networkx`의 기본 dead end 처리(personalization으로 보냄)를 `dangling="teleport"`와, test 8은 균등 `dangling`을 `"uniform"`과 대응시킨다.

# 프로젝트 실행 방법

## 환경 설정
- Python 3.12
- 패키지: `pip install -r requirements.txt`

## 데이터 준비

```
python scripts/download_data.py
```

원본 4개(약 8.6 MB)를 `data/raw/`에 받고 라벨 압축 파일을 푼다. 자세한 동작과 파일별 SHA-256은 [data/README.md](data/README.md) 참고. `data/raw/`, `data/processed/`의 데이터 파일은 Git에 올리지 않는다.

## 테스트

```
pytest
```

프로젝트 루트에서 실행한다. `pytest.ini`가 루트를 import 경로에 넣어 주므로 `src`를 바로 import할 수 있다. 테스트 파일을 `python tests/test_rank.py`로 직접 실행하면 `src`를 찾지 못한다.
