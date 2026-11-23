import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from src.common.schema import SPAM, NONSPAM


def _check(risk, y):
    risk = np.asarray(risk, dtype=np.float64)
    y = np.asarray(y)
    assert risk.ndim == 1 and risk.shape == y.shape, "risk와 y의 길이가 다름"
    assert np.isfinite(risk).all(), "risk에 NaN 또는 inf가 있음"
    assert np.isin(y, (0, 1)).all(), "y는 0/1만"
    assert 0 < y.sum() < len(y), "spam과 nonspam이 모두 있어야 함"
    return risk, y.astype(np.int8)


def pr_auc(risk, y):
    """PR-AUC = average precision (sklearn). 동점은 하나의 threshold로 묶임."""
    risk, y = _check(risk, y)
    return float(average_precision_score(y, risk))


def roc_auc(risk, y):
    risk, y = _check(risk, y)
    return float(roc_auc_score(y, risk))


def precision_at_k(risk, y, k):
    """상위 K개 중 spam 비율. K번째 경계에 동점이 걸치면 무작위로 깨는 경우의 기댓값."""
    risk, y = _check(risk, y)
    k = int(min(k, len(y)))
    thr = np.sort(risk)[::-1][k - 1]
    above, tied = risk > thr, risk == thr
    need = k - above.sum()
    pos = y[above].sum() + need * y[tied].sum() / tied.sum()
    return float(pos / k)


def tie_fraction(risk):
    """가장 큰 동점 묶음이 차지하는 비율 (예: TR = 0인 host)."""
    _, counts = np.unique(np.asarray(risk), return_counts=True)
    return float(counts.max() / len(risk))


def eval_arrays(ds, split):
    idx = {"val": ds.val_idx, "test": ds.test_idx}[split]
    y = ds.labels[idx]
    assert np.isin(y, (SPAM, NONSPAM)).all()
    return idx, (y == SPAM).astype(np.int8)


def evaluate(risk, ds, split, ks=(50, 100, 500)):
    """risk: 길이 N (높을수록 위험). split: "val" 또는 "test"."""
    risk = np.asarray(risk, dtype=np.float64)
    assert risk.shape == (ds.graph.n,), "risk는 전체 host 길이 N"
    idx, y = eval_arrays(ds, split)
    r = risk[idx]
    out = {"split": split, "n_eval": len(idx), "n_pos": int(y.sum()),
           "prevalence": float(y.mean()), "pr_auc": pr_auc(r, y),
           "roc_auc": roc_auc(r, y), "tie_frac": tie_fraction(r)}
    for k in ks:
        out[f"p@{k}"] = precision_at_k(r, y, k)
    return out


def choose_direction(score, ds):
    """원점수의 방향을 val PR-AUC로 결정. test는 보지 않음."""
    a = evaluate(score, ds, "val")["pr_auc"]
    b = evaluate(-np.asarray(score), ds, "val")["pr_auc"]
    return (1 if a >= b else -1), a, b


def bootstrap(risks, y, n_boot=1000, seed=0, metric=pr_auc):
    """risks: {이름: 평가 대상 host만의 risk 배열}. 층화 bootstrap, 모든 method에 같은 표본 사용(paired)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = {m: np.empty(n_boot) for m in risks}
    for b in range(n_boot):
        s = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for m, r in risks.items():
            stats[m][b] = metric(np.asarray(r)[s], y[s])
    ci = {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in stats.items()}
    return ci, stats


def paired_diff_ci(stats, a, b):
    d = stats[a] - stats[b]
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
