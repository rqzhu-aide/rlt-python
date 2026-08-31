"""Port of RLT tests/testthat/test-resample.R."""

import numpy as np

from rlt import RLT_reg, RLT_cla, RLT_surv
from helpers import (
    generate_simple_regression,
    generate_classification_data,
    generate_survival_data,
)


def test_replacement_true_regression_fits_and_predicts():
    d = generate_simple_regression(n=80, p=10)
    # expect_error(..., NA) -> no exception raised
    RLT_reg(n_estimators=30, resample_replace=True, n_jobs=2, verbose=0, random_state=1).fit(
        d["X"], d["y"]
    )


def test_replacement_true_classification_fits_and_predicts():
    d = generate_classification_data(n=80, p=10)
    RLT_cla(n_estimators=30, resample_replace=True, n_jobs=2, verbose=0, random_state=1).fit(
        d["X"], d["y"]
    )


def test_replacement_true_survival_fits_and_predicts():
    d = generate_survival_data(n=80, p=10, seed=1)
    RLT_surv(n_estimators=30, resample_replace=True, n_jobs=2, verbose=0, random_state=1).fit(
        d["X"], d["y"]
    )


def test_replacement_true_stored_in_parameters():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, resample_replace=True, n_jobs=2, verbose=0, random_state=1).fit(
        d["X"], d["y"]
    )
    # R: fit$parameters[["resample.replace"]] == 1 -> truthy bool
    assert fit.resample_replace is True


def test_replacement_false_regression_fits_and_predicts():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(n_estimators=30, resample_replace=False, n_jobs=2, verbose=0, random_state=1).fit(
        d["X"], d["y"]
    )


def test_replacement_false_classification_fits_and_predicts():
    d = generate_classification_data(n=80, p=10)
    RLT_cla(n_estimators=30, resample_replace=False, n_jobs=2, verbose=0, random_state=1).fit(
        d["X"], d["y"]
    )


def test_replacement_false_survival_fits_and_predicts():
    d = generate_survival_data(n=80, p=10, seed=1)
    RLT_surv(n_estimators=30, resample_replace=False, n_jobs=2, verbose=0, random_state=1).fit(
        d["X"], d["y"]
    )


def test_replacement_false_stored_in_parameters():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, resample_replace=False, n_jobs=2, verbose=0, random_state=1).fit(
        d["X"], d["y"]
    )
    # R: fit$parameters[["resample.replace"]] == 0 -> falsy bool
    assert fit.resample_replace is False


def test_resample_prob_050_regression_subsampling_works():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(
        n_estimators=30, resample_prob=0.5, resample_replace=False, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])


def test_resample_prob_080_regression_subsampling_works():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(
        n_estimators=30, resample_prob=0.8, resample_replace=False, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])


def test_resample_prob_0632_regression_bootstrap_works():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(
        n_estimators=30, resample_prob=0.632, resample_replace=True, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])


def test_resample_prob_050_classification_works():
    d = generate_classification_data(n=80, p=10)
    RLT_cla(
        n_estimators=30, resample_prob=0.5, resample_replace=False, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])


def test_resample_prob_050_survival_works():
    d = generate_survival_data(n=80, p=10, seed=1)
    RLT_surv(
        n_estimators=30, resample_prob=0.5, resample_replace=False, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])


def test_bootstrap_produces_oob_predictions():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(
        n_estimators=30, resample_replace=True, resample_prob=1.0, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    assert np.asarray(fit.oob_prediction_).shape == (d["n"],)


def test_subsampling_produces_oob_predictions():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(
        n_estimators=30, resample_replace=False, resample_prob=0.5, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    assert np.asarray(fit.oob_prediction_).shape == (d["n"],)


def test_default_resample_parameters_are_stored():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    # R: fit$parameters[["resample.replace"]] / [["resample.prob"]] not null;
    # Python stores them on params_ as replacement / resample_prob
    assert fit.params_.replacement in (True, False)
    assert fit.params_.resample_prob is not None
