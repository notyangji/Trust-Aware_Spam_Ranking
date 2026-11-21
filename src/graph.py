"""그래프 변환: 전이 행렬(build_transition), 도메인 판정(domain_of, site_of).

행렬 방향은 여기서 한 번만 정한다.
- A: 행 = 출발, 열 = 도착. A[i, j] = host i → host j의 link 수.
- M: column-stochastic (강의와 같음). M[j, i] = P(i → j), r_new = β · M @ r.
B는 build_transition이 준 M만 쓰고 직접 전치하지 않는다.
뒤집은 그래프(Inverse PR)는 build_transition(A.T.tocsr())로 만든다.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# .uk 아래 2단계 도메인 목록. 실제 hostname의 마지막 두 조각 빈도표를 보고 보강한다.
UK_SLD = {"ac", "co", "gov", "org", "ltd", "plc", "me", "net",
          "nhs", "police", "sch", "mod", "nic"}


def build_transition(A, weighted=False):
    """인접 행렬 A → column-stochastic 전이 행렬 M (CSR, float64).

    M[j, i] = P(i → j). dead end(나가는 링크 0) 열은 0으로 남기고, 보정은 power iteration에서 한다.
    weighted=False(기본, 강의식): 링크 수와 상관없이 나가는 host마다 1/d_i.
    weighted=True: 링크 수에 비례 (E5 비교용).
    원본 A는 바뀌지 않는다 (내부에서 복사).
    """
    A = A.tocsr().astype(np.float64)           # astype은 복사본을 반환 → 원본 A는 그대로
    A.sum_duplicates()                         # 중복 항목·명시적 0 정리 (이진화 전에 필수)
    A.eliminate_zeros()
    if not weighted:
        A.data[:] = 1.0                        # 강의식 1/d_i
    out = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
    P = sp.diags(inv) @ A                      # row-stochastic, dead end 행은 0
    M = P.T.tocsr()

    col_sum = np.asarray(M.sum(axis=0)).ravel()
    alive = out > 0
    assert np.allclose(col_sum[alive], 1.0), "dead end가 아닌 열의 합이 1이 아님"
    assert not col_sum[~alive].any(), "dead end 열에 0이 아닌 값이 있음"
    return M


def _host_parts(hostname: str) -> list[str]:
    """hostname을 소문자·끝의 '.'·포트(:8080) 제거 후 '.'로 나눈다. 예: "WWW.X.ac.uk.:8080" → [www, x, ac, uk]"""
    return hostname.strip().lower().rstrip(".").split(":")[0].split(".")


def _is_uk_sld(parts: list[str]) -> bool:
    """마지막 두 조각이 .uk 2단계 도메인(ac.uk, co.uk, ...)인지."""
    return len(parts) >= 2 and parts[-1] == "uk" and parts[-2] in UK_SLD


def domain_of(hostname: str) -> str:
    """hostname → .uk 2단계 도메인 ("ac.uk", "gov.uk", ...). 목록 밖은 "other".

    대소문자, 끝의 '.', 포트(:8080)는 무시한다.
    예: www.x.ac.uk → ac.uk, x.county.sch.uk → sch.uk, parliament.uk → other
    """
    parts = _host_parts(hostname)
    return parts[-2] + ".uk" if _is_uk_sld(parts) else "other"


def site_of(hostname: str) -> str:
    """hostname → 사이트(등록 도메인). train·val 분할과 EDA에서 같은 사이트의 host를 묶을 때 쓴다.

    .uk 2단계 도메인(UK_SLD) 아래는 3단계까지, 그 밖은 2단계까지.
    예: www1.example.co.uk → example.co.uk, molcat1.bl.uk → bl.uk
    """
    parts = _host_parts(hostname)
    k = 3 if len(parts) >= 3 and _is_uk_sld(parts) else 2
    return ".".join(parts[-k:])
