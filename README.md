# Trust-Aware_Spam_Ranking
- 사용 데이터셋: [WEBSPAM-UK2007](https://chato.cl/webspam/datasets/uk2007/) (`.uk` 웹 호스트 그래프 + 호스트 단위 스팸 라벨)
- 프로젝트 목표: 

# 데이터셋

WEBSPAM-UK2007은 2007년 5월에 수집한 `.uk` 도메인 웹 데이터(호스트 114,529개)에 스팸 여부 라벨을 붙인 데이터셋이다. 이 프로젝트에서는 페이지 본문은 쓰지 않고 **호스트 그래프와 라벨**만 사용한다. 데이터셋 배경은 [data/README.md](data/README.md)를 참고한다.

## 사용 파일

| 파일 | 내용 | 형식 |
|---|---|---|
| `uk-2007-05.hostgraph_weighted.graph-txt.gz` | 호스트 간 링크 그래프 | 1행 = 호스트 수 N, 2행부터 한 줄에 호스트 하나(0번부터)의 나가는 링크를 `도착id:링크수`로 나열 (공백 구분, 빈 줄 = 나가는 링크 없음) |
| `uk-2007-05.hostnames.txt.gz` | 호스트 id ↔ 이름 대응표 | `id hostname` (기본 포트가 아니면 `:포트`가 붙음. 예: `afis.postoffice.co.uk:8080`) |
| `webspam-uk2007-set1-1.0.tar.gz` | SET1 라벨 (학습용) | `hostid label spamicity assessments` |
| `webspam-uk2007-set2-1.0.tar.gz` | SET2 라벨 (테스트용) | SET1과 같음 |

- 네 파일 모두 같은 호스트 id(0 ~ 114,528)를 쓴다.
- `.gz` 두 파일은 압축을 풀지 않고 `gzip.open()`으로 바로 읽는다. 라벨 `.tar.gz`는 `data/raw/set1/`, `data/raw/set2/`에 풀린다.
- 그래프의 링크 수(weight)는 두 호스트의 페이지 사이에 있는 하이퍼링크 개수이다. 예: `70094:293` = 이 호스트에서 70094번 호스트로 가는 링크 293개.
- 라벨 파일의 `label`은 `nonspam` / `spam` / `undecided`, `spamicity`는 평가자 판정의 평균(N = 0, B = 0.5, S = 1, U는 제외)이다.

## 라벨

| | SET1 (학습용) | SET2 (테스트용) |
|---|---:|---:|
| nonspam | 3,776 | 1,933 |
| spam | 222 | 122 |
| undecided | 277 | 149 |
| **합계** | **4,275** | **2,204** |

- 라벨이 있는 호스트는 6,479개로 전체의 약 5.7%이다.
- SET1과 SET2에 겹치는 호스트는 없다. 공식 분할은 3단계 도메인 단위로 되어 있어, `www1.example.co.uk`와 `www2.example.co.uk`는 항상 같은 세트에 들어간다.
- 검증(val) 데이터는 SET1을 다시 나눠 만든다. SET2는 최종 평가에만 쓴다.

## 호스트 그래프 통계

원본 파일에서 직접 집계한 값이다.

| 항목 | 값 |
|---|---:|
| 호스트 수 N | 114,529 |
| edge 수 | 1,836,441 |
| self-loop | 0 |
| dead end (나가는 링크 0) | 49,379 (43.1%) |
| 들어오는 링크 0 | 4,766 |
| 고립 호스트 (나가는 링크·들어오는 링크 모두 0) | 3,380 |
| weight (링크 수) | 최소 1 · 중앙값 1 · 최대 2,579,857 |
| 평균 out-degree | 16.0 |

## 라이선스

라벨은 [CC BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/)이다 (라벨 압축 파일의 `README.txt` 기준). 연구 목적으로 사용할 수 있으며, 공개 재배포 시 비상업·동일조건 조항이 적용된다.

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

# 프로젝트 실행 방법

## 환경 설정
- Python 3.12
- 패키지: `pip install -r requirements.txt`

## 데이터 준비

```
python scripts/download_data.py
```

- 원본 4개(약 8.6 MB)를 `data/raw/`에 받고, 라벨 압축 파일 2개를 풀어 둔다.
- 이미 받은 파일은 건너뛰며, 마지막에 파일별 크기와 SHA-256을 출력한다.
- `data/raw/`, `data/processed/`의 데이터 파일은 Git에 올리지 않는다.
