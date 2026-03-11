"""CHPED (Combined Heat and Power Economic Dispatch) instance data.

Parameters from literature (Basu 2011, Vasebi et al. 2007).
Cost function: F_i(P_i) = a_i + b_i*P_i + c_i*P_i^2 + |d_i*sin(e_i*(P_min_i - P_i))|
"""

# 4-unit instance (Basu 2011)
CHPED_4UNIT = {
    "name": "4-unit",
    "n_units": 4,
    "n_periods": 1,
    # Cost coefficients: a, b, c, d, e (valve-point effect)
    "a": [25.0, 60.0, 100.0, 120.0],
    "b": [2.0, 1.8, 2.1, 2.0],
    "c": [0.008, 0.006, 0.009, 0.007],
    "d": [100.0, 140.0, 160.0, 180.0],
    "e": [0.042, 0.040, 0.038, 0.037],
    "P_min": [10.0, 20.0, 30.0, 40.0],
    "P_max": [75.0, 125.0, 175.0, 250.0],
    "demand": [400.0],
    "reserve": [0.0],  # no spinning reserve for simple case
}

# 7-unit instance
CHPED_7UNIT = {
    "name": "7-unit",
    "n_units": 7,
    "n_periods": 1,
    "a": [25.0, 60.0, 100.0, 120.0, 40.0, 70.0, 110.0],
    "b": [2.0, 1.8, 2.1, 2.0, 1.9, 2.2, 1.7],
    "c": [0.008, 0.006, 0.009, 0.007, 0.008, 0.005, 0.006],
    "d": [100.0, 140.0, 160.0, 180.0, 120.0, 150.0, 130.0],
    "e": [0.042, 0.040, 0.038, 0.037, 0.041, 0.039, 0.043],
    "P_min": [10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0],
    "P_max": [75.0, 125.0, 175.0, 250.0, 100.0, 150.0, 200.0],
    "demand": [800.0],
    "reserve": [0.0],
}

# 24-unit instance (extended from 4-unit by repeating with perturbations)
def make_24unit():
    """Generate 24-unit instance by replicating and perturbing base units."""
    import numpy as np
    rng = np.random.default_rng(123)
    base = CHPED_4UNIT
    n = 24
    inst = {
        "name": "24-unit",
        "n_units": n,
        "n_periods": 1,
        "a": [], "b": [], "c": [], "d": [], "e": [],
        "P_min": [], "P_max": [],
        "demand": [2000.0],
        "reserve": [100.0],
    }
    for i in range(n):
        j = i % 4
        scale = 0.8 + rng.random() * 0.4  # 0.8 to 1.2
        inst["a"].append(base["a"][j] * scale)
        inst["b"].append(base["b"][j] * scale)
        inst["c"].append(base["c"][j] * scale)
        inst["d"].append(base["d"][j] * scale)
        inst["e"].append(base["e"][j] * scale)
        inst["P_min"].append(base["P_min"][j] * scale)
        inst["P_max"].append(base["P_max"][j] * max(scale, 1.0))
    return inst

CHPED_24UNIT = make_24unit()
