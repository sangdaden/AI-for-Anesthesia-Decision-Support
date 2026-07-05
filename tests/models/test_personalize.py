import numpy as np
from adaptivedose.models.personalize import RequirementModel, leave_one_out, FEATURES

def _synth(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    y = 20 + 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.5, size=n)
    return X, y

def test_fit_recovers_known_coefficients():
    X, y = _synth()
    m = RequirementModel().fit(X, y)
    coef = m.coefficients(FEATURES)
    assert abs(coef["age"] - 3.0) < 0.3
    assert abs(coef["weight"] - (-2.0)) < 0.3
    assert abs(coef["intercept"] - 20.0) < 0.5

def test_predict_shape_and_values():
    X, y = _synth()
    m = RequirementModel().fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (len(y),)
    assert np.mean(np.abs(preds - y)) < 1.0

def test_leave_one_out_beats_mean_baseline_when_signal_exists():
    X, y = _synth()
    preds = leave_one_out(X, y)
    assert preds.shape == (len(y),)
    n = len(y)
    baseline = np.array([y[np.arange(n) != i].mean() for i in range(n)])
    model_mae = np.mean(np.abs(y - preds))
    base_mae = np.mean(np.abs(y - baseline))
    assert model_mae < base_mae
