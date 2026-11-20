"""공통 power iteration과 PageRank.

PageRank, PPR, TrustRank, Inverse PR이 모두 이 power_iteration 하나를 쓴다.
M은 src.graph.build_transition이 만든 column-stochastic 행렬 (M[j, i] = P(i → j)).
설정값(beta, tol, max_iter, dangling)은 configs/default.yaml의 rank 섹션에서 읽어 넘긴다.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

# return_info 값에 따라 반환 타입을 편집기(Pylance 등)에 알려주는 선언. 실행 동작과는 무관.
@overload
def power_iteration(M, teleport=..., beta: float = ..., tol: float = ..., max_iter: int = ...,
                    dangling: str = ..., return_info: Literal[False] = ...) -> NDArray[np.float64]: ...
@overload
def power_iteration(M, teleport=..., beta: float = ..., tol: float = ..., max_iter: int = ...,
                    dangling: str = ..., *, return_info: Literal[True]
                    ) -> tuple[NDArray[np.float64], dict[str, Any]]: ...


def power_iteration(M, teleport=None, beta=0.85, tol=1e-8, max_iter=300,
                    dangling="teleport", return_info=False):
    """r = β · M r + (누출분) · s 를 수렴할 때까지 반복한다.

    teleport: 길이 N, ≥ 0, 합 > 0 (함수 안에서 합 1로 정규화). None이면 균등 (= PageRank).
    dangling:
      "teleport" (기본) — teleport분과 dead end 누출분을 모두 s로 보냄.
                          s에서 도달 불가능한 host의 점수는 정확히 0.
      "uniform"         — teleport분은 s로, dead end 누출분만 모든 host에 균등 분배.
    초기값 r(0) = s. 수렴 판정은 L1 오차 < tol. 수렴하지 못하면 RuntimeWarning.
    반환: r (N,) float64, 합 = 1, ≥ 0.
          return_info=True면 (r, {"iterations", "residual", "converged", "time_sec"}).
    """
    if dangling not in ("teleport", "uniform"):
        raise ValueError(f"dangling은 'teleport' 또는 'uniform'이어야 함: {dangling!r}")
    assert max_iter >= 1
    N = M.shape[0]
    assert M.shape == (N, N)
    s = np.full(N, 1.0 / N) if teleport is None else np.asarray(teleport, dtype=np.float64)
    assert s.shape == (N,) and (s >= 0).all() and s.sum() > 0
    s = s / s.sum()
    u = np.full(N, 1.0 / N)
    r = s.copy()                               # PageRank면 강의의 r(0) = 1/N, PPR이면 seed에서 시작
    it, err = 0, np.inf                        # 반복 전 초기값 (max_iter >= 1이라 아래에서 항상 갱신됨)
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        r_new = beta * (M @ r)
        if dangling == "teleport":             # 누출분(teleport + dead end)을 모두 s로
            r_new += (1.0 - r_new.sum()) * s
        else:                                  # "uniform": dead end 누출분만 균등 분배
            r_new += (1.0 - beta) * s
            r_new += (1.0 - r_new.sum()) * u
        err = np.abs(r_new - r).sum()
        r = r_new
        if err < tol:
            break
    elapsed = time.perf_counter() - t0
    converged = err < tol
    if not converged:
        warnings.warn(f"power_iteration: {max_iter}회 안에 수렴하지 않음 "
                      f"(residual={err:.2e}, tol={tol:.0e})", RuntimeWarning)
    if return_info:
        return r, {"iterations": it, "residual": float(err),
                   "converged": bool(converged), "time_sec": elapsed}
    return r


@overload
def pagerank(M, *, return_info: Literal[False] = ..., **kw) -> NDArray[np.float64]: ...
@overload
def pagerank(M, *, return_info: Literal[True], **kw) -> tuple[NDArray[np.float64], dict[str, Any]]: ...


def pagerank(M, *, return_info=False, **kw):
    """PageRank = teleport가 균등인 power_iteration."""
    return power_iteration(M, teleport=None, return_info=return_info, **kw)
