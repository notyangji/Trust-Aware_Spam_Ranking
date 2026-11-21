# AI 사용 기록

AI 활용 내역을 작업별로 아래 표에 한 행씩 추가한다.

| 날짜 | AI 도구 | 대상 파일 | 활용 내용 | 검증 내용 |
|---|---|---|---|---|
| 2026-11-17 | Codex | `.gitignore` | 데이터·점수 산출물, Python 캐시·가상환경, Jupyter 임시 파일, 환경변수 파일, macOS 메타데이터 등의 기본 제외 규칙 작성 | `git check-ignore`로 제외·유지 규칙 확인, `git diff --check` 통과 |
| 2026-11-17 | Codex | `data/README.md` | WEBSPAM-UK2007 데이터셋 개요, 프로젝트 활용 방향, 로컬 저장 위치 및 출처 작성 | 공식 안내 페이지에서 수집 시기·규모·구성·라벨 분할 확인, `git diff --check` 통과 |
| 2026-11-20 | Claude (Anthropic, Claude Opus 5.5) | `requirements.txt`, `README.md` | 패키지 목록과 버전 고정, Python 버전·설치 방법 표기 | 새 venv(Python 3.12)에 설치 후 전체 import 확인 |
| 2026-11-21 | Claude (Anthropic, Claude Opus 5.5) | `docs/file_entry.md`, `.gitignore` | 역할 C 파일 트리·표 초안 작성 | 팀 실험 설계(E1–E6)와 대조, `git diff`로 B 내용 보존 확인 |
| 2026-11-21 | Claude (Anthropic, Claude Opus 5.5) | `src/metrics.py`, `tests/test_metrics.py` | 평가 지표(PR-AUC, ROC-AUC, P@K 동점 기대값, 방향 결정, paired bootstrap CI) 구현 및 테스트 작성 | `pytest` 9개 통과, sklearn 결과와 교차 확인 |
