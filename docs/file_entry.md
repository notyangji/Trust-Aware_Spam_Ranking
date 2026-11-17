# 파일 구조 예상도

GUIDE_A_graph_core.md(v2) 기준. 작성일 2026-11-15. 담당: A = 데이터·그래프·PageRank, B = PPR·TrustRank·Inverse PR, C = 실험·평가.
대괄호 안 숫자는 GUIDE의 절 번호. (B의 역할은 그냥 [B]로 적었음. 이건 절 번호랑 별개로 구분한거임.)

```
team_project/
├── README.md                      # 설치, 데이터 준비 방법 (A는 데이터·설치 부분 작성)
├── requirements.txt               # numpy, scipy, pandas, scikit-learn, networkx, pytest, matplotlib, pyyaml
├── .gitignore                     # data/raw/, data/processed/, results/scores/
├── configs/
│   ├── default.yaml               # beta, tol, max_iter, val_frac, random_seed, 파일 경로
│   └── seeds.yaml                 # [B] seed 선정 방식·개수·품질·난수 설정
│
├── scripts/                       # 실행용 (python scripts/xxx.py)
│   ├── download_data.py           # [3.1] 원본 4개 다운로드, SHA-256 출력
│   ├── build_processed.py         # [3.2–3.6] raw → processed (점검 통계, split.npz 포함)
│   ├── run_baselines.py           # [6] PageRank·Random·in-degree 점수 계산 → results/scores/
│   └── run_seeded_rankings.py      # [B] seed 선정 → 점수 계산 → 결과 저장
│
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   └── schema.py              # [2.1] HostGraph, Dataset, label 상수
│   ├── data.py                    # [3] load_graph, load_hostnames, load_labels,
│   │                              #     make_split, load_dataset
│   ├── graph.py                   # [4] build_transition, domain_of, UK_SLD
│   ├── rank.py                    # [5–6] power_iteration, pagerank
│   ├── baselines.py               # [6] random_scores, indegree_scores
│   ├── seeds.py                   # [B] topic/trusted seed 선정·개인화 벡터 구성
│   ├── ppr.py                     # [B] Personalized PageRank
│   └── trust.py                   # [B] TrustRank·Inverse PR·spam mass·hop 거리 baseline
│
├── tests/
│   ├── test_rank.py               # [7] test 1–8 + 불변식 + 예외·경고
│   ├── test_seeds.py              # [B] seed 선정·개인화 벡터 검증
│   ├── test_ppr.py                # [B] PPR 검증
│   └── test_trust.py              # [B] TR 및 관련 점수·baseline 검증
│
├── notebooks/
│   └── 01_eda.ipynb               # [8] EDA, 도메인 집계
│
├── data/
│   ├── README.md                  # 출처, 접근일, SHA-256, license, 인용 (git에 포함)
│   ├── raw/                       # 원본 (git 제외)
│   └── processed/                 # graph.npz, hostnames.txt, labels.csv, split.npz, meta.json (git 제외)
│
├── results/
│   ├── scores/                    # <method>_<config>.npy, 길이 N float64
│   └── figures/                   # EDA 그림
│
└── docs/
    └── AI_USE.md                  # AI 사용 기록 [10]
```

## 파일별 내용

| 파일 | 담당 | 주요 내용 | 사용하는 사람 |
|---|---|---|---|
| `src/common/schema.py` | A | `HostGraph`, `Dataset`, `SPAM / NONSPAM / UNDECIDED / UNLABELED` | 전원 |
| `src/data.py` | A | `load_graph`, `load_hostnames`, `load_labels`, `make_split`, `load_dataset` | B, C (`load_dataset`만) |
| `src/graph.py` | A | `build_transition`, `domain_of`, `UK_SLD` | B |
| `src/rank.py` | A | `power_iteration`, `pagerank` | B |
| `src/baselines.py` | A | `random_scores`, `indegree_scores` | C |
| `tests/test_rank.py` | A | 강의 예제 test 1–6, networkx 교차 확인 test 7–8 | 전원 |
| `scripts/download_data.py` | A | 원본 다운로드, 이미 있으면 건너뜀 | 전원 |
| `scripts/build_processed.py` | A | raw → processed, `meta.json`에 점검 통계 기록 | A (한 번 실행) |
| `scripts/run_baselines.py` | A | `pagerank_binary_b085.npy`, random·in-degree 점수 저장 | A |
| `configs/default.yaml` | A | 모든 설정값 (코드에 숫자를 직접 쓰지 않음) | 전원 |
| `notebooks/01_eda.ipynb` | A | 통계표, degree 분포, 도메인 집계, TrustRank 가정 검증 | 보고서 |
| `src/seeds.py` | B | topic/trusted seed 선정, seed ID 검증, 개인화 벡터 구성, seed 개수·품질 조건 적용 | B |
| `src/ppr.py` | B | A의 `power_iteration`을 활용한 Personalized PageRank 계산 | B, C |
| `src/trust.py` | B | TrustRank, Inverse PR, spam mass, hop 거리 baseline 구현(적혀있는 순서대로 구현) | B, C |
| `configs/seeds.yaml` | B | topic proxy, seed 선정 방식·개수·품질 조건·난수 설정 | B, C |
| `scripts/run_seeded_rankings.py` | B | 공통 데이터 로딩, seed 선정, 알고리즘 실행, 점수·실행 조건 저장 | B, C |
| `tests/test_seeds.py` | B | seed 후보 범위, 중복·빈 집합·잘못된 ID, 재현성, 개인화 벡터 검증 | 전원 |
| `tests/test_ppr.py` | B | 균등 개인화 시 PR과의 일치, 점수 합·비음수성 검증 | 전원 |
| `tests/test_trust.py` | B | TR의 PPR 연계, seed 도달성·역방향 그래프·관련 점수 검증 | 전원 |

## 작성 순서

| 순서 | 파일 | 데이터 필요 여부 |
|---|---|---|
| 1 | `requirements.txt`, `.gitignore`, `configs/default.yaml`, `src/common/schema.py` | 필요 없음 |
| 2 | `src/graph.py`, `src/rank.py`, `tests/test_rank.py` | 필요 없음 |
| 3 | `scripts/download_data.py`, `data/README.md` | 다운로드 |
| 4 | `src/data.py`, `scripts/build_processed.py` | raw 필요 |
| 5 | `src/baselines.py`, `scripts/run_baselines.py` | processed 필요 |
| 6 | `notebooks/01_eda.ipynb` | processed 필요 |

## 참고

- `configs/default.yaml`, `scripts/build_processed.py`, `scripts/run_baselines.py`는 GUIDE 본문에 파일 이름이 없던 것을 이 문서에서 정함.
- B·C의 파일 이름은 예상이며 각자 정함. 단, `src/common/`, `src/data.py`, `src/graph.py`, `src/rank.py`의 함수 형식은 GUIDE 2절에서 고정.
