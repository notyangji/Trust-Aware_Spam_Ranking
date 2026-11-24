import numpy as np, pytest
from types import SimpleNamespace
from src.metrics import (pr_auc, roc_auc, precision_at_k, tie_fraction, evaluate,
                         choose_direction, bootstrap, paired_diff_ci)
from src.common.schema import SPAM, NONSPAM, UNLABELED

y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])

def test_perfect():
    r = y.astype(float)
    assert pr_auc(r, y) == 1 and roc_auc(r, y) == 1
    assert precision_at_k(r + np.arange(10) * 1e-3, y, 3) == 1

def test_reversed():
    assert roc_auc(-y.astype(float), y) == 0

def test_constant_equals_random():
    r = np.zeros(10)
    assert roc_auc(r, y) == 0.5
    assert pr_auc(r, y) == pytest.approx(y.mean())
    assert precision_at_k(r, y, 4) == pytest.approx(y.mean())

def test_tie_at_boundary():
    r = np.array([3, 2, 2, 2, 1.]); yy = np.array([1, 1, 0, 0, 1])
    assert precision_at_k(r, yy, 2) == pytest.approx((1 + 1/3) / 2)

def test_no_ties_matches_sort():
    rng = np.random.default_rng(0)
    r = rng.random(200); yy = (rng.random(200) < 0.1).astype(int)
    for k in (1, 10, 50):
        assert precision_at_k(r, yy, k) == yy[np.argsort(-r)[:k]].mean()

def test_tie_fraction():
    assert tie_fraction(np.array([0, 0, 0, 1.])) == 0.75

def test_bad_input():
    with pytest.raises(AssertionError):
        pr_auc(np.array([np.nan, 1.]), np.array([0, 1]))
    with pytest.raises(AssertionError):
        pr_auc(np.array([0.1, 0.2]), np.array([1, 1]))

def make_ds():
    n = 12
    labels = np.full(n, UNLABELED, dtype=np.int8)
    labels[[0, 1, 2, 3, 4, 5]] = [SPAM, NONSPAM, NONSPAM, SPAM, NONSPAM, NONSPAM]
    labels[[6, 7, 8, 9]] = [SPAM, NONSPAM, NONSPAM, NONSPAM]
    return SimpleNamespace(graph=SimpleNamespace(n=n), labels=labels,
                           val_idx=np.arange(6), test_idx=np.arange(6, 10))

def test_evaluate_and_direction():
    ds = make_ds()
    score = np.zeros(12); score[[0, 3, 6]] = -5      # spam의 원점수가 낮음
    d, a, b = choose_direction(score, ds)
    assert d == -1 and b > a
    out = evaluate(-score, ds, "test", ks=(1,))
    assert out["pr_auc"] == 1 and out["n_pos"] == 1 and out["p@1"] == 1

def test_bootstrap_paired():
    rng = np.random.default_rng(1)
    yy = (rng.random(300) < 0.1).astype(int)
    good = yy + rng.normal(0, 0.5, 300); bad = rng.random(300)
    ci, st = bootstrap({"good": good, "bad": bad}, yy, n_boot=200)
    m, lo, hi = paired_diff_ci(st, "good", "bad")
    assert lo > 0 and ci["good"][0] < pr_auc(good, yy) < ci["good"][1]
