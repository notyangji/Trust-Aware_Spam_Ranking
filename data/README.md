# WEBSPAM-UK2007 데이터

WEBSPAM-UK2007은 웹 스팸 탐지 연구를 위한 데이터셋으로, 2007년 5월 밀라노대학교 Laboratory of Web Algorithmics가 수집한 `.uk` 도메인 웹 데이터를 기반으로 한다.

- **규모:** 전체 수집 데이터는 114,529개 호스트와 105,896,555개 웹 페이지로 구성된다. 스팸 라벨은 이 중 평가된 호스트에 제공되며, 모든 호스트에 라벨이 있는 것은 아니다.
- **구성:** 호스트 단위 스팸/비스팸 라벨, URL·하이퍼링크 그래프, HTML 페이지 본문으로 구성되며, 사전 계산된 호스트 특성도 별도로 제공된다.
- **이 프로젝트에서 쓰는 것:** 페이지 본문은 쓰지 않고 **호스트 그래프와 라벨**만 사용한다. 이를 이용해 PageRank, Personalized PageRank, TrustRank를 비교하고, seed 선정 방식·개수·품질에 따른 스팸 탐지 성능을 분석한다.

## 받는 방법

```
python scripts/download_data.py [--config configs/default.yaml]
```

- 설정 파일의 `download.base_url`(기준 URL)과 `paths.raw_dir`(저장 위치)을 읽는다. 둘 중 하나라도 없으면 오류로 멈춘다.
- 원본 4개(약 8.6 MB)를 `raw/`에 받고, 라벨 압축 파일 2개를 `raw/set1/`, `raw/set2/`에 푼다.
- 이미 받은 파일과 이미 푼 폴더는 건너뛴다. 마지막에 파일별 크기와 SHA-256을 출력한다.
- 받는 중에는 `<파일명>.part`로 저장하고 끝까지 받은 뒤에만 원래 이름으로 바꾸므로, 중간에 끊겨도 깨진 파일이 남지 않는다.
- 압축 해제는 `tarfile`의 `filter="data"`를 써서 폴더 밖으로 풀리는 항목을 막는다 (Python 3.12+).
- 서버가 `python`이 들어간 User-Agent를 406으로 거부하므로, 스크립트는 `webspam-downloader/1.0`을 쓴다.

받은 뒤 폴더 구조:

```
data/
├── raw/                                         # 원본 (Git 제외)
│   ├── uk-2007-05.hostgraph_weighted.graph-txt.gz
│   ├── uk-2007-05.hostnames.txt.gz
│   ├── webspam-uk2007-set1-1.0.tar.gz
│   ├── webspam-uk2007-set2-1.0.tar.gz
│   ├── set1/webspam-uk2007-set1-1.0/            # tar 안에 상위 폴더가 한 단계 더 있음
│   │   ├── WEBSPAM-UK2007-SET1-labels.txt
│   │   ├── WEBSPAM-UK2007-SET1-raw-assessments.txt
│   │   ├── WEBSPAM-UK2007-hostnames.txt
│   │   ├── README.txt
│   │   └── CREDITS.txt
│   └── set2/webspam-uk2007-set2-1.0/
│       ├── WEBSPAM-UK2007-SET2-labels.txt
│       ├── WEBSPAM-UK2007-SET2-raw-assessments.txt
│       └── WEBSPAM-UK2007-hostnames.txt
└── processed/                                   # 전처리 결과 (Git 제외)
```

## 원본 파일

| 파일 | 내용 | 형식 |
|---|---|---|
| `uk-2007-05.hostgraph_weighted.graph-txt.gz` | 호스트 간 링크 그래프 | 1행 = 호스트 수 N, 2행부터 한 줄에 호스트 하나(0번부터)의 나가는 링크를 `도착id:링크수`로 나열 (공백 구분, 빈 줄 = 나가는 링크 없음) |
| `uk-2007-05.hostnames.txt.gz` | 호스트 id ↔ 이름 대응표 | `id hostname` (기본 포트가 아니면 `:포트`가 붙음. 예: `afis.postoffice.co.uk:8080`) |
| `webspam-uk2007-set1-1.0.tar.gz` | SET1 라벨 (학습용) | `hostid label spamicity assessments` |
| `webspam-uk2007-set2-1.0.tar.gz` | SET2 라벨 (테스트용) | SET1과 같음 |

- 네 파일 모두 같은 호스트 id(0 ~ 114,528)를 쓴다. 라벨 압축 파일 안의 `WEBSPAM-UK2007-hostnames.txt`는 `uk-2007-05.hostnames.txt.gz`와 내용이 같다.
- `.gz` 두 파일은 압축을 풀지 않고 `gzip.open()`으로 바로 읽는다.
- 그래프는 페이지 단위 링크를 호스트 단위로 묶은 것이고, 링크 수(weight)는 두 호스트의 페이지 사이에 있는 하이퍼링크 개수이다. 예: `70094:293` = 이 호스트에서 70094번 호스트로 가는 링크 293개.
- 라벨 파일의 `label`은 `nonspam` / `spam` / `undecided`, `spamicity`는 평가자 판정의 평균(N = 0, B = 0.5, S = 1, U는 제외, 유효한 판정이 없으면 `-`)이다. `assessments`는 평가자별 판정 목록이다 (예: `j6:N,j9:N`).

### 크기·SHA-256

접근일 2026-11-20 기준으로 받은 파일이다. 다시 받았을 때 값이 다르면 원본이 바뀐 것이다.

| 파일 | 크기 (bytes) | SHA-256 |
|---|---:|---|
| `webspam-uk2007-set1-1.0.tar.gz` | 1,170,426 | `c4dbf6898582554dd1fe4ad1c3500578aa18b4b4795354262c002c530c30f982` |
| `webspam-uk2007-set2-1.0.tar.gz` | 1,074,409 | `5e9b87c3be438fcd2c04036803369cd1f6e4e7f2b6a69a0e01d5dea5e1515e9e` |
| `uk-2007-05.hostgraph_weighted.graph-txt.gz` | 5,394,352 | `dc29d04ae027cda134bdf65ffafca18029db59f05a5be8a7ff90eb88ae8c1ab2` |
| `uk-2007-05.hostnames.txt.gz` | 971,436 | `04956c6f78fcf23d11019c586d952d479cbd4169ec710f6b6d32b8a94ce092b9` |

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

## 라이선스·인용

- 라벨과 그래프는 [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)이다. 연구 목적으로 사용할 수 있으며, 공개 재배포 시 비상업·동일조건 조항이 적용된다. 라벨은 연구용이며 검색엔진 순위나 필터링에 직접 쓰지 않도록 권고한다.
- 페이지 본문은 별도의 연구용 데이터 이용 동의가 필요하다 (이 프로젝트에서는 쓰지 않음).
- 공식 페이지가 안내하는 인용 형식:

  > "Web Spam Collections". http://chato.cl/webspam/datasets/ Crawled by the Laboratory of Web Algorithmics, University of Milan, http://law.di.unimi.it/. URLs retrieved 05 2007.

출처: [WEBSPAM-UK2007 공식 안내](https://chato.cl/webspam/datasets/uk2007/), [라이선스](https://chato.cl/webspam/datasets/license.php) · 확인일: 2026-11-20
