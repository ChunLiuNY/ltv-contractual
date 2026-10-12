"""Simulate a subscription customer base where we know the true answer.

Data-generating story
---------------------
1. Each customer is secretly "loyal" or "flaky" (unobserved type).
2. Their type sets a baseline monthly churn probability.
3. Observed features (channel, plan, discount) multiply that baseline.
4. Monthly churn is constant for each individual, so their true lifetime is
   Geometric(p): the number of months they pay before cancelling.
5. Customers sign up across `n_signup_months` months; we look at the data at
   `cutoff_month`, so anyone still active then is right-censored.

Time convention
---------------
- signup_month = 0 .. n_signup_months - 1
- A customer who signs up in month s can be watched for (cutoff_month - s) months.
- true_lifetime = number of paid months (>= 1). If true_lifetime <= window,
  we saw them churn; otherwise we only know they lasted at least `window` months.
"""

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    # Hidden customer types
    "share_flaky": 0.40,
    "base_churn": {"loyal": 0.07, "flaky": 0.45},
    # Observed feature mix
    "channel_mix": {"paid_social": 0.45, "search": 0.35, "referral": 0.20},
    "share_annual": 0.25,
    "share_discount": 0.30,
    # True churn multipliers (1.0 = no effect). These are the "true effects"
    # we will later compare Cox hazard ratios against.
    "channel_mult": {"paid_social": 1.30, "search": 1.00, "referral": 0.70},
    "plan_mult": {"monthly": 1.00, "annual": 0.60},
    "discount_mult": 1.40,
    # Monthly margin (not revenue): $20/mo at 70% margin; annual is discounted.
    "margin": {"monthly": 14.0, "annual": 11.0},
}


def simulate_customers(n_customers=4000, n_signup_months=24, cutoff_month=24,
                       params=None, seed=42):
    """Return (customers, truth).

    customers : what an analyst would actually see.
    truth     : hidden ground truth, for evaluation only.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    rng = np.random.default_rng(seed)
    n = n_customers

    # --- Observed features ---------------------------------------------------
    signup_month = rng.integers(0, n_signup_months, size=n)
    channel = rng.choice(list(p["channel_mix"]), size=n,
                         p=list(p["channel_mix"].values()))
    plan = np.where(rng.random(n) < p["share_annual"], "annual", "monthly")
    discount = (rng.random(n) < p["share_discount"]).astype(int)

    # --- Hidden type and true monthly churn ----------------------------------
    ctype = np.where(rng.random(n) < p["share_flaky"], "flaky", "loyal")
    base = np.array([p["base_churn"][t] for t in ctype])
    mult = (np.array([p["channel_mult"][c] for c in channel])
            * np.array([p["plan_mult"][pl] for pl in plan])
            * np.where(discount == 1, p["discount_mult"], 1.0))
    true_churn = np.clip(base * mult, 0.001, 0.95)

    # --- True lifetime and what we actually observe --------------------------
    true_lifetime = rng.geometric(true_churn)          # paid months, >= 1
    window = cutoff_month - signup_month               # months we can watch
    churned = (true_lifetime <= window).astype(int)
    months_observed = np.minimum(true_lifetime, window)

    customers = pd.DataFrame({
        "customer_id": np.arange(n),
        "signup_month": signup_month,
        "channel": channel,
        "plan": plan,
        "discount_at_signup": discount,
        "months_observed": months_observed,
        "churned": churned,
        "monthly_margin": np.array([p["margin"][pl] for pl in plan]),
    })
    truth = pd.DataFrame({
        "customer_id": np.arange(n),
        "customer_type": ctype,
        "true_monthly_churn": true_churn,
        "true_lifetime": true_lifetime,
    })
    return customers, truth


def true_retention_curve(truth, horizon=24):
    """Expected share of customers still active after t months, t = 0..horizon.

    Uses each customer's true churn rate: S(t) = mean_i (1 - p_i)^t.
    This is the exact population curve (no sampling noise), which is what
    every method will be graded against.
    """
    t = np.arange(horizon + 1)
    p_i = truth["true_monthly_churn"].to_numpy()[:, None]
    return pd.Series(((1 - p_i) ** t).mean(axis=0), index=t, name="true_retention")
