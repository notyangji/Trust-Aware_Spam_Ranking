# AI 사용 기록

AI 활용 내역을 작업별로 아래 표에 한 행씩 추가한다.

| 날짜 | AI 도구 | 대상 파일 | 활용 내용 | 검증 내용 |
|---|---|---|---|---|
| 2026-11-17 | Codex | `.gitignore` | 데이터·점수 산출물, Python 캐시·가상환경, Jupyter 임시 파일, 환경변수 파일, macOS 메타데이터 등의 기본 제외 규칙 작성 | `git check-ignore`로 제외·유지 규칙 확인, `git diff --check` 통과 |
| 2026-11-17 | Codex | `data/README.md` | WEBSPAM-UK2007 데이터셋 개요, 프로젝트 활용 방향, 로컬 저장 위치 및 출처 작성 | 공식 안내 페이지에서 수집 시기·규모·구성·라벨 분할 확인, `git diff --check` 통과 |
| 2026-11-20 | Claude Code (Opus 5.5) | `scripts/download_data.py` | 원본 4개 다운로드, `.part` 임시 저장 후 이름 변경, `filter="data"` 압축 해제, SHA-256 출력 작성. 서버의 User-Agent 거부(406) 수정, 설정 파일만 읽도록 수정 | 실제 다운로드·압축 해제 성공, 재실행 시 건너뛰기, 설정 키 누락 시 오류 메시지 확인 |
| 2026-11-20 | Claude Code (Opus 5.5) | `src/common/schema.py` | GUIDE 2.1절 참고 구현을 옮기고 assert 메시지 추가 | 정상 데이터 통과, 잘못된 데이터 8가지 경우 차단 확인, 실제 그래프로 `HostGraph` 생성 |
| 2026-11-20 | Claude Code (Opus 5.5) | `configs/default.yaml` | file_spec 2절 제안 구조로 설정 작성 | PyYAML로 읽어 값 타입 확인 (`tol`이 float) |
| 2026-11-20 | Claude Code (Opus 5.5) | `src/graph.py` | GUIDE 4절 `build_transition`, 3.3절 `domain_of` 작성, `site_of` 추가 | 강의 예제 전이 행렬과 일치, 실제 그래프 열 합 검사, 도메인 빈도표로 `UK_SLD` 범위 확인 (`other` 28개) |
| 2026-11-20 | Claude Code (Opus 5.5) | `src/rank.py`, `tests/test_rank.py`, `pytest.ini` | GUIDE 5–7절 power iteration·테스트 작성, 반환 타입 `@overload` 추가 | `pytest` 22개 통과, 일부러 버그(행렬 방향 반전, 초기값 1/N)를 넣으면 테스트가 실패하는지 확인, 실제 그래프 수렴 확인 |
| 2026-11-20 | Claude Code (Opus 5.5) | `README.md`, `data/README.md` | 데이터셋 파일 형식·라벨·그래프 통계·SHA-256·라이선스, 코드 사용법 정리 | 통계는 원본 파일 직접 집계로 확인, 라이선스·인용은 공식 페이지에서 확인, 예시 코드 실행 확인 |
| 2026-11-21 | Claude Code (Opus 5.5) | `src/data.py` | GUIDE 3절 로더 작성, 팀 합의에 따라 SET1을 사이트 단위로 분할(val 30%) | 실제 데이터 분할 수치 확인, train·val에 걸친 사이트 0개, 사이트 기준이 공식 SET1/SET2 분할과 어긋나지 않음 확인 |
| 2026-11-21 | Claude Code (Opus 5.5) | `scripts/build_processed.py` | raw → processed 변환, `meta.json` 점검 통계, 분할 변경 감지 작성 | 점검 수치(N, edge, 라벨 수)가 기대값과 일치, 재실행 시 분할 동일·변경 감지 확인 |
| 2026-11-21 | Claude Code (Opus 5.5) | `src/config.py`, `tests/test_config.py` | 스크립트마다 있던 설정 읽기를 하나로 통합 | 리팩토링 전후 출력 파일 바이트 단위 동일, `pytest` 29개 통과 |
| 2026-11-21 | Claude Code (Opus 5.5) | `src/baselines.py`, `scripts/run_baselines.py` | random·in-degree baseline, 점수 파일(`.npy` + `.json` 방향 기록) 저장 작성 | 작은 예제로 in-degree 계산 확인, 실제 그래프 in-degree 0 host 수가 통계와 일치 |
| 2026-11-21 | Claude Code (Opus 5.5) | `notebooks/01_eda.ipynb`, `results/figures/` | GUIDE 8절 EDA 6개 항목과 그림 작성 | Jupyter 커널로 전체 실행 시 오류 없음, 그림 직접 확인, 수치를 별도 계산과 대조 |
| 2026-11-20 | Claude (Anthropic, Claude Opus 5.5) | `requirements.txt`, `README.md` | 패키지 목록과 버전 고정, Python 버전·설치 방법 표기 | 새 venv(Python 3.12)에 설치 후 전체 import 확인 |
| 2026-11-21 | Claude (Anthropic, Claude Opus 5.5) | `docs/file_entry.md`, `.gitignore` | 역할 C 파일 트리·표 초안 작성 | 팀 실험 설계(E1–E6)와 대조, `git diff`로 B 내용 보존 확인 |
| 2026-11-21 | Claude (Anthropic, Claude Opus 5.5) | `src/metrics.py`, `tests/test_metrics.py` | 평가 지표(PR-AUC, ROC-AUC, P@K 동점 기대값, 방향 결정, paired bootstrap CI) 구현 및 테스트 작성 | `pytest` 9개 통과, sklearn 결과와 교차 확인 |
| 2026-11-23 | Codex | `configs/seeds.yaml`, `src/seeds.py`, 관련 Markdown 문서 | #13 seed 설정·선정 로직 설계와 구현 보조, 함수 사용법과 점수 저장 규약 정리 | 구현·테스트 코드와 문서 대조, `git diff --check` 확인 |
| 2026-11-27 | Claude (Anthropic, Claude Opus 5.5) | `src/trust.py`, `tests/test_trust.py`, `scripts/run_seeded_rankings.py`, `docs/function_reference.md` | spam mass·hop 거리·Inverse PageRank 함수와 테스트 4개, 저장 스크립트에 `--name` 옵션과 보조 점수 저장 추가, 사용법 문서 작성 | 강의 자료(spam mass 정의)와 대조, 작은 예제 그래프 테스트로 정의·경계값 확인, `pytest` 69개 통과 |
| 2026-11-27 | Claude (Anthropic, Claude Opus 5.5) | `scripts/run_experiment.py`, `tests/test_experiment.py`, `configs/experiments/e1–e5.yaml` | 실험 runner(설정 조합 실행, seed 선정 전략, val 평가, bootstrap CI, paired 차이, CSV 저장)와 실험 설정 작성 | 작은 예제 데이터 테스트 4개(조합 수, 방향, val만 사용, 기준 차이 0, 전략별 seed 후보), 저장 점수 파일 평가값과 대조, `pytest` 73개 통과 |
| 2026-11-27 | Claude (Anthropic, Claude Opus 5.5) | `notebooks/02_results.ipynb`, `results/metrics/summary_*.md`, `results/figures/e*.png` | 결과 요약 표, 그림, 오류 분석 셀 작성 | 요약 표 수치를 CSV 직접 집계와 대조, 그림 육안 확인 |

