"""Turn a retention curve into LTV.

Convention: retention[t] = share still active after t months (retention[0] = 1).
A customer active after t months pays for month t + 1, so month t + 1's margin is
weighted by retention[t] and discounted by (1 + d)^t. Over a 24-month horizon this
sums t = 0 .. 23.
"""

import numpy as np


def ltv(retention, monthly_margin, d=0.01, horizon=24):
    """Expected discounted margin per customer over `horizon` months."""
    s = np.asarray(retention, dtype=float)[:horizon]
    t = np.arange(len(s))
    return float(np.sum(s * monthly_margin / (1 + d) ** t))
