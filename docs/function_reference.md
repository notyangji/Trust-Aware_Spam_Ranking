# 함수와 공통 자료형

설치와 실행 순서는 [프로젝트 README](../README.md)를 참고한다.

## 사용 예시

**설정 (`src/config.py`)**

```python
import numpy as np
from src.config import load_config

cfg = load_config()                          # configs/default.yaml
cfg.require("rank", "beta")                  # 0.85 (키가 없으면 어떤 키가 없는지 알려주는 ValueError)
cfg.path("scores_dir")                       # 프로젝트 루트 기준 절대 경로
```

**데이터 (`src/data.py`)**: B·C는 `load_dataset()`만 쓴다. 반환값의 필드는 [공통 자료형](#공통-자료형)을 참고한다.

```python
from src.data import load_dataset

ds = load_dataset()                          # data/processed/ → Dataset
y_train = ds.labels[ds.train_idx]            # train host의 라벨 (1 = spam, 0 = nonspam)
```

**전이 행렬·도메인 (`src/graph.py`)**

```python
from src.graph import build_transition, domain_of, site_of

M = build_transition(ds.graph.adjacency)     # column-stochastic 전이 행렬
domain_of("www.ox.ac.uk")                    # "ac.uk"
site_of("www1.example.co.uk")                # "example.co.uk"
```

**Seed 설정·선정·개인화 벡터 (`src/seeds.py`)**

`load_config()`로 읽은 설정을 `validate_seed_config()`에 전달한다. 설정은 `topic`과 `trusted`를 담은 중첩 딕셔너리 형태다. 검사에 통과하면 `None`을 반환하고, 잘못된 설정이면 `ValueError`가 발생한다.

`select_topic_seeds()`는 지정한 도메인에 속하는 원본 호스트 ID를 반환한다. `make_personalization()`은 전체 호스트 수와 중복 없는 정수 ID 목록을 받아, seed에 같은 비중을 주는 길이 N의 `float64` 배열을 만든다.

```python
from src.seeds import make_personalization, select_topic_seeds, validate_seed_config

seed_cfg = load_config("configs/seeds.yaml")
validate_seed_config(seed_cfg)

domains = cfg.require("domains", "ppr_seed_domains")
seed_ids = select_topic_seeds(ds.graph.hostnames, domains)
s = make_personalization(ds.graph.n, seed_ids)  # power_iteration()의 teleport에 전달
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

**Baseline (`src/baselines.py`)**

```python
from src.baselines import indegree_scores, random_scores

rnd = random_scores(ds.graph.n, cfg.require("baselines", "random_seed"))   # [0, 1) 균등 난수
deg = indegree_scores(ds.graph.adjacency)    # −log1p(in_deg), 클수록 spam 의심
```

**점수 파일 읽기 (C)**: `.json`의 `higher_is_spam`으로 방향을 맞춘 뒤 평가한다.

```python
import json

name = "pagerank_binary_b085"
scores = np.load(cfg.path("scores_dir") / f"{name}.npy")
meta = json.loads((cfg.path("scores_dir") / f"{name}.json").read_text(encoding="utf-8"))
risk = -scores if meta["higher_is_spam"] is False else scores   # 클수록 spam 의심으로 통일
```

## 설정 읽기

설정은 `src/config.py`의 `load_config()`로 읽는다. 각자 yaml을 직접 열지 않는다.

| 기능 | 사용법 | 내용 |
|---|---|---|
| 읽기 | `load_config()` / `load_config("configs/beta090.yaml")` | 결과는 dict처럼 쓸 수 있음 (`cfg["rank"]["beta"]`) |
| 필수 키 | `cfg.require("rank", "beta")` | 키가 없거나 비어 있으면 `ValueError` (예: `...에 설정 키 rank.beta 없음`) |
| 경로 | `cfg.path("scores_dir")` | `paths.*`를 프로젝트 루트 기준 절대 경로로 |

- 상대 경로는 설정 파일 경로와 `paths.*` 모두 **프로젝트 루트 기준**이다. `scripts/`, `notebooks/` 어디서 실행해도 같은 파일을 가리킨다.

## 공통 자료형

팀 전원이 같이 쓰는 자료형을 정의한다. `src/data.py`의 `load_dataset()`이 `data/processed/`를 읽어 `Dataset`을 만들고, B·C는 이 객체만 받아 쓴다.

### 라벨 상수

| 상수 | 값 | 의미 |
|---|---:|---|
| `SPAM` | 1 | spam |
| `NONSPAM` | 0 | nonspam |
| `UNDECIDED` | −1 | 평가했지만 판정 불가 |
| `UNLABELED` | −2 | 평가하지 않음 (라벨 없음) |

### `HostGraph`

| 필드 | 타입 | 내용 |
|---|---|---|
| `hostnames` | `list[str]`, 길이 N | index = host id (원본 번호 0 ~ 114,528, 재번호 금지) |
| `adjacency` | `csr_matrix`, float64, (N, N) | `A[i, j]` = host i → host j의 링크 수 (행 = 출발, 열 = 도착). self-loop 제거됨 |
| `n` (속성) | `int` | 호스트 수 N |

- `adjacency`는 0이 아닌 칸만 저장하는 희소 행렬(CSR)이다. 약 22 MB.

### `Dataset`

| 필드 | 타입 | 내용 |
|---|---|---|
| `graph` | `HostGraph` | 전체 그래프 |
| `labels` | `np.ndarray`, int8, (N,) | 라벨 상수 값 |
| `in_set` | `np.ndarray`, int8, (N,) | 1 = SET1, 2 = SET2, 0 = 라벨 없음 |
| `train_idx` | `np.ndarray`, int64 | SET1의 spam·nonspam 일부 (seed 선택·학습) |
| `val_idx` | `np.ndarray`, int64 | SET1의 나머지 (β 등 조정) |
| `test_idx` | `np.ndarray`, int64 | SET2의 spam·nonspam (최종 평가에만 사용) |

현재 분할 (val 30%, seed 42):

| 분할 | host 수 | spam | nonspam |
|---|---:|---:|---:|
| train | 2,798 | 155 | 2,643 |
| val | 1,200 | 67 | 1,133 |
| test | 2,055 | 122 | 1,933 |

- 분할 배열의 값은 **host id**이므로 `ds.labels[ds.train_idx]`처럼 바로 쓸 수 있다.
- 세 분할 모두 undecided와 라벨 없는 호스트는 제외한다.
- train·val은 SET1을 **사이트(등록 도메인, `site_of`) 단위**로 나눈다. `www1.example.co.uk`와 `www2.example.co.uk`처럼 같은 사이트의 호스트는 항상 같은 쪽에 들어가, seed와 같은 사이트의 호스트가 val에 섞이는 데이터 누수를 막는다 (공식 SET1/SET2 분할과 같은 방식).
- 그래프는 전체를 그대로 쓴다. test 호스트도 그래프에 남아 있고, 라벨만 seed 선택·조정에 쓰지 않는다.

### 생성 시 검사

두 클래스 모두 `frozen=True`라 필드를 다른 값으로 바꿔 끼울 수 없고, 데이터가 잘못되면 객체를 만드는 순간 `AssertionError`가 난다.

| 클래스 | 검사 |
|---|---|
| `HostGraph` | 행렬 크기 = (hostname 수, hostname 수), self-loop 없음 |
| `Dataset` | `labels`·`in_set` 길이 = N, 분할끼리 겹치지 않음, train·val은 SET1만, test는 SET2만, 분할에 undecided·라벨 없음 미포함 |

- `frozen=True`는 배열 안의 값 수정(`ds.labels[0] = 1`)까지 막지는 않으므로, 받은 배열을 직접 고치지 않는다.

## 그래프·랭킹

### 행렬 방향

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
- `site_of`는 사이트(등록 도메인)를 돌려준다. `UK_SLD` 아래는 3단계까지(`www1.example.co.uk` → `example.co.uk`), 그 밖은 2단계까지(`molcat1.bl.uk` → `bl.uk`). train·val 분할과 사이트별 집계에 쓴다.

### `power_iteration`

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
