"""Descriptive retention metrics from a censored customer snapshot.

For each customer and tenure k ("active after k months"), the answer is one of:
  known active    : k <= window and (months_observed > k or churned == 0)
  known churned   : k <= window and not known active
  not observable  : k > window   (signed up too recently; NOT churned)
where window = cutoff - signup_month.
"""

import numpy as np
import pandas as pd


def status_matrix(customers, cutoff=24, horizon=24):
    """Customer x tenure matrix: 1.0 = active, 0.0 = churned, NaN = not observable yet."""
    window = cutoff - customers["signup_month"]
    cols = {}
    for k in range(horizon + 1):
        active = (customers["months_observed"] > k) | (customers["churned"] == 0)
        cols[k] = active.astype(float).where(window >= k)
    return pd.DataFrame(cols, index=customers.index)


def cohort_triangle(customers, cutoff=24, horizon=24):
    """Share of each signup cohort active after k months (NaN where not yet observable)."""
    status = status_matrix(customers, cutoff, horizon)
    return status.groupby(customers["signup_month"]).mean()


def person_month(customers):
    """Expand to one row per customer per month at risk (month = 1 .. months_observed).

    event = 1 only in the month the customer churned. This is the input for
    discrete-time hazard models (one binary outcome per customer-month).
    """
    pm = customers.loc[customers.index.repeat(customers["months_observed"])].copy()
    pm["month"] = pm.groupby("customer_id").cumcount() + 1
    pm["event"] = ((pm["month"] == pm["months_observed"]) & (pm["churned"] == 1)).astype(int)
    return pm.reset_index(drop=True)


def retention_curve(customers, cutoff=24, horizon=24, method="pooled"):
    """Blended retention by tenure with counts and a 95% binomial interval.

    method="pooled"   : denominator = customers observable at k (censoring-aware)
    method="snapshot" : denominator = all customers (counts not-yet-observable as churned)
    """
    status = status_matrix(customers, cutoff, horizon)
    n_active = (status == 1).sum()
    n_observable = status.notna().sum()
    denom = n_observable if method == "pooled" else pd.Series(len(customers), index=status.columns)
    rate = n_active / denom
    se = np.sqrt(rate * (1 - rate) / denom)
    return pd.DataFrame({
        "n_observable": n_observable,
        "n_active": n_active,
        "retention": rate,
        "ci_low": (rate - 1.96 * se).clip(lower=0),
        "ci_high": (rate + 1.96 * se).clip(upper=1),
    }).rename_axis("k")
