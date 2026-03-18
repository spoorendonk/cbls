"""UC-CHPED (Unit Commitment with Valve-Point Effects) instance data.

Data from Pedroso, Kubo & Viana (2014), "Pricing and unit commitment in combined
energy and reserve markets using valve-point effects", originally at
http://www.dcc.fc.up.pt/~jpp/code/valve/ucp_data.py (GPL).

The 10-unit base UC parameters (min_on, min_off, t_cold, startup costs, initial
state) come from the Kazarlis 10-unit system. For ucp13/ucp40, cost coefficients
(a,b,c,d,e) and power limits come from the corresponding CHPED instances (Sinha
13-unit, Taipower 40-unit), while UC parameters are mapped from Kazarlis.

Cost function: F_i(P_i) = a_i + b_i*P_i + c_i*P_i^2 + |d_i*sin(e_i*(Pmin_i - P_i))|
Startup cost: hot if off < t_cold periods, cold otherwise.
"""

import importlib.util
import os

_chped_path = os.path.join(os.path.dirname(__file__), "..", "..", "chped", "data.py")
_spec = importlib.util.spec_from_file_location("chped_data", _chped_path)
_chped_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chped_data)
CHPED_13UNIT = _chped_data.CHPED_13UNIT
CHPED_40UNIT = _chped_data.CHPED_40UNIT

# ---------------------------------------------------------------------------
# Kazarlis 10-unit base UC parameters
# ---------------------------------------------------------------------------
_KAZARLIS_MIN_ON = [8, 8, 5, 5, 6, 3, 3, 1, 1, 1]
_KAZARLIS_MIN_OFF = [8, 8, 5, 5, 6, 3, 3, 1, 1, 1]
_KAZARLIS_T_COLD = [5, 5, 4, 4, 4, 2, 2, 0, 0, 0]
_KAZARLIS_A_HOT = [4500, 5000, 550, 560, 900, 170, 260, 30, 30, 30]
_KAZARLIS_A_COLD = [9000, 10000, 1100, 1120, 1800, 340, 520, 60, 60, 60]
# Initial state: 1=ON, 0=OFF. Units 1-2 ON, rest OFF.
_KAZARLIS_Y_PREV = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# n_init = min_on for ON units, min_off for OFF units (free to switch at t=1)
_KAZARLIS_N_INIT = [8, 8, 5, 5, 6, 3, 3, 1, 1, 1]

# ---------------------------------------------------------------------------
# UCP_10UNIT — Kazarlis 10-unit system (quadratic cost, no valve-point)
# ---------------------------------------------------------------------------
UCP_10UNIT = {
    "name": "ucp10",
    "n_units": 10,
    "n_periods": 24,
    # Kazarlis quadratic cost coefficients (no valve-point: d=0, e=0)
    "a": [1000.0, 970.0, 700.0, 680.0, 450.0, 370.0, 480.0, 660.0, 665.0, 670.0],
    "b": [16.19, 17.26, 16.60, 16.50, 19.70, 22.26, 27.74, 25.92, 27.27, 27.79],
    "c": [0.00048, 0.00031, 0.00200, 0.00211, 0.00398, 0.00712, 0.00079, 0.00413, 0.00222, 0.00173],
    "d": [0.0] * 10,
    "e": [0.0] * 10,
    "P_min": [150.0, 150.0, 20.0, 20.0, 25.0, 20.0, 25.0, 10.0, 10.0, 10.0],
    "P_max": [455.0, 455.0, 130.0, 130.0, 162.0, 80.0, 85.0, 55.0, 55.0, 55.0],
    # UC parameters (direct from Kazarlis)
    "min_on": list(_KAZARLIS_MIN_ON),
    "min_off": list(_KAZARLIS_MIN_OFF),
    "t_cold": list(_KAZARLIS_T_COLD),
    "n_init": list(_KAZARLIS_N_INIT),
    "y_prev": list(_KAZARLIS_Y_PREV),
    "a_hot": list(_KAZARLIS_A_HOT),
    "a_cold": list(_KAZARLIS_A_COLD),
    # 24-hour demand profile
    "demand": [
        700, 750, 850, 950, 1000, 1100, 1150, 1200,
        1300, 1400, 1450, 1500, 1400, 1300, 1200, 1050,
        1000, 1100, 1200, 1400, 1300, 1100, 900, 800,
    ],
    # 10% spinning reserve
    "reserve": [
        70, 75, 85, 95, 100, 110, 115, 120,
        130, 140, 145, 150, 140, 130, 120, 105,
        100, 110, 120, 140, 130, 110, 90, 80,
    ],
    # No known bounds for 10-unit with valve-point (Pedroso Table 2 is for 13/40)
    "known_bounds": {},
}

# ---------------------------------------------------------------------------
# UCP_13UNIT — 13-unit system with valve-point effects
# Cost coefficients from CHPED_13UNIT (Sinha et al. 2003)
# UC parameters mapped from Kazarlis: [1,2,3, 1,2,3,4,5,6,7,8,9,10] (1-indexed)
# ---------------------------------------------------------------------------
_UCP13_MAP = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 0-indexed into Kazarlis


def _build_ucp13():
    n = 13
    kmap = _UCP13_MAP
    return {
        "name": "ucp13",
        "n_units": n,
        "n_periods": 24,
        # Cost coefficients and power limits from CHPED_13UNIT
        "a": list(CHPED_13UNIT["a"]),
        "b": list(CHPED_13UNIT["b"]),
        "c": list(CHPED_13UNIT["c"]),
        "d": list(CHPED_13UNIT["d"]),
        "e": list(CHPED_13UNIT["e"]),
        "P_min": list(CHPED_13UNIT["P_min"]),
        "P_max": list(CHPED_13UNIT["P_max"]),
        # UC parameters mapped from Kazarlis
        "min_on": [_KAZARLIS_MIN_ON[kmap[i]] for i in range(n)],
        "min_off": [_KAZARLIS_MIN_OFF[kmap[i]] for i in range(n)],
        "t_cold": [_KAZARLIS_T_COLD[kmap[i]] for i in range(n)],
        "n_init": [_KAZARLIS_N_INIT[kmap[i]] for i in range(n)],
        "y_prev": [_KAZARLIS_Y_PREV[kmap[i]] for i in range(n)],
        "a_hot": [_KAZARLIS_A_HOT[kmap[i]] for i in range(n)],
        "a_cold": [_KAZARLIS_A_COLD[kmap[i]] for i in range(n)],
        # 24-hour demand profile
        "demand": [
            1250, 1340, 1510, 1690, 1780, 1960, 2050, 2140,
            2320, 2490, 2580, 2670, 2490, 2320, 2140, 1870,
            1780, 1960, 2140, 2490, 2320, 1960, 1600, 1420,
        ],
        # 10% spinning reserve
        "reserve": [
            125, 134, 151, 169, 178, 196, 205, 214,
            232, 249, 258, 267, 249, 232, 214, 187,
            178, 196, 214, 249, 232, 196, 160, 142,
        ],
        # Known bounds from Pedroso Table 2 (1hr MIP): {periods: (LB, UB)}
        "known_bounds": {
            1: (11701, 11701),
            3: (38850, 38850),
            6: (91406, 91784),
            12: (231587, 232537),
            24: (464053, 466187),
        },
    }


UCP_13UNIT = _build_ucp13()

# ---------------------------------------------------------------------------
# UCP_40UNIT — 40-unit system with valve-point effects
# Cost coefficients from CHPED_40UNIT (Taipower system)
# UC parameters mapped from Kazarlis: units i -> Kazarlis (i % 10)
# ---------------------------------------------------------------------------


def _build_ucp40():
    n = 40
    return {
        "name": "ucp40",
        "n_units": n,
        "n_periods": 24,
        # Cost coefficients and power limits from CHPED_40UNIT
        "a": list(CHPED_40UNIT["a"]),
        "b": list(CHPED_40UNIT["b"]),
        "c": list(CHPED_40UNIT["c"]),
        "d": list(CHPED_40UNIT["d"]),
        "e": list(CHPED_40UNIT["e"]),
        "P_min": list(CHPED_40UNIT["P_min"]),
        "P_max": list(CHPED_40UNIT["P_max"]),
        # UC parameters mapped from Kazarlis (repeating every 10 units)
        "min_on": [_KAZARLIS_MIN_ON[i % 10] for i in range(n)],
        "min_off": [_KAZARLIS_MIN_OFF[i % 10] for i in range(n)],
        "t_cold": [_KAZARLIS_T_COLD[i % 10] for i in range(n)],
        "n_init": [_KAZARLIS_N_INIT[i % 10] for i in range(n)],
        "y_prev": [_KAZARLIS_Y_PREV[i % 10] for i in range(n)],
        "a_hot": [_KAZARLIS_A_HOT[i % 10] for i in range(n)],
        "a_cold": [_KAZARLIS_A_COLD[i % 10] for i in range(n)],
        # 24-hour demand profile
        "demand": [
            5360, 5740, 6510, 7270, 7650, 8420, 8800, 9190,
            9950, 10720, 11100, 11480, 10720, 9950, 9190, 8040,
            7650, 8420, 9190, 10720, 9950, 8420, 6890, 6120,
        ],
        # 10% spinning reserve
        "reserve": [
            536, 574, 651, 727, 765, 842, 880, 919,
            995, 1072, 1110, 1148, 1072, 995, 919, 804,
            765, 842, 919, 1072, 995, 842, 689, 612,
        ],
        # Known bounds from Pedroso Table 2 (1hr MIP): {periods: (LB, UB)}
        "known_bounds": {
            1: (55645, 55645),
            3: (178396, 178547),
            6: (416108, 416606),
            12: (1112371, 1113801),
            24: (2235971, 2238504),
        },
    }


UCP_40UNIT = _build_ucp40()

# ---------------------------------------------------------------------------
# UCP_100UNIT — 100-unit system (10× Kazarlis base, valve-point from 40-unit)
# Cost coefficients cycle from CHPED_40UNIT; UC parameters cycle from Kazarlis.
# Demand scaled proportionally: 100/40 = 2.5× the 40-unit demand.
# ---------------------------------------------------------------------------


def _build_ucp100():
    n = 100
    base40 = CHPED_40UNIT
    return {
        "name": "ucp100",
        "n_units": n,
        "n_periods": 24,
        "a": [base40["a"][i % 40] for i in range(n)],
        "b": [base40["b"][i % 40] for i in range(n)],
        "c": [base40["c"][i % 40] for i in range(n)],
        "d": [base40["d"][i % 40] for i in range(n)],
        "e": [base40["e"][i % 40] for i in range(n)],
        "P_min": [base40["P_min"][i % 40] for i in range(n)],
        "P_max": [base40["P_max"][i % 40] for i in range(n)],
        "min_on": [_KAZARLIS_MIN_ON[i % 10] for i in range(n)],
        "min_off": [_KAZARLIS_MIN_OFF[i % 10] for i in range(n)],
        "t_cold": [_KAZARLIS_T_COLD[i % 10] for i in range(n)],
        "n_init": [_KAZARLIS_N_INIT[i % 10] for i in range(n)],
        "y_prev": [_KAZARLIS_Y_PREV[i % 10] for i in range(n)],
        "a_hot": [_KAZARLIS_A_HOT[i % 10] for i in range(n)],
        "a_cold": [_KAZARLIS_A_COLD[i % 10] for i in range(n)],
        # 2.5× the 40-unit demand
        "demand": [round(d * 2.5) for d in UCP_40UNIT["demand"]],
        "reserve": [round(r * 2.5) for r in UCP_40UNIT["reserve"]],
        "known_bounds": {},
    }


UCP_100UNIT = _build_ucp100()

# ---------------------------------------------------------------------------
# UCP_200UNIT — 200-unit system (20× Kazarlis base)
# Demand scaled: 200/40 = 5× the 40-unit demand.
# ---------------------------------------------------------------------------


def _build_ucp200():
    n = 200
    base40 = CHPED_40UNIT
    return {
        "name": "ucp200",
        "n_units": n,
        "n_periods": 24,
        "a": [base40["a"][i % 40] for i in range(n)],
        "b": [base40["b"][i % 40] for i in range(n)],
        "c": [base40["c"][i % 40] for i in range(n)],
        "d": [base40["d"][i % 40] for i in range(n)],
        "e": [base40["e"][i % 40] for i in range(n)],
        "P_min": [base40["P_min"][i % 40] for i in range(n)],
        "P_max": [base40["P_max"][i % 40] for i in range(n)],
        "min_on": [_KAZARLIS_MIN_ON[i % 10] for i in range(n)],
        "min_off": [_KAZARLIS_MIN_OFF[i % 10] for i in range(n)],
        "t_cold": [_KAZARLIS_T_COLD[i % 10] for i in range(n)],
        "n_init": [_KAZARLIS_N_INIT[i % 10] for i in range(n)],
        "y_prev": [_KAZARLIS_Y_PREV[i % 10] for i in range(n)],
        "a_hot": [_KAZARLIS_A_HOT[i % 10] for i in range(n)],
        "a_cold": [_KAZARLIS_A_COLD[i % 10] for i in range(n)],
        # 5× the 40-unit demand
        "demand": [round(d * 5.0) for d in UCP_40UNIT["demand"]],
        "reserve": [round(r * 5.0) for r in UCP_40UNIT["reserve"]],
        "known_bounds": {},
    }


UCP_200UNIT = _build_ucp200()

# ---------------------------------------------------------------------------
# Convenience: sub-instance with fewer periods
# ---------------------------------------------------------------------------


def make_subinstance(inst, n_periods):
    """Create a sub-instance using the first n_periods of the demand profile."""
    assert 1 <= n_periods <= inst["n_periods"], f"n_periods must be 1..{inst['n_periods']}"
    sub = dict(inst)
    sub["n_periods"] = n_periods
    sub["demand"] = inst["demand"][:n_periods]
    sub["reserve"] = inst["reserve"][:n_periods]
    sub["name"] = f"{inst['name']}-{n_periods}p"
    # Look up bounds for this period count
    bounds = inst["known_bounds"].get(n_periods)
    sub["known_bounds"] = {n_periods: bounds} if bounds else {}
    return sub


def extend_horizon(inst, n_periods):
    """Extend an instance to n_periods by repeating the 24h demand profile.

    Each repeated day gets a slight variation (±3% sinusoidal) to avoid
    perfect periodicity that could be trivially exploited.
    """
    import math as _math

    base_T = inst["n_periods"]
    assert n_periods > base_T, f"n_periods must be > {base_T}"
    sub = dict(inst)
    sub["n_periods"] = n_periods
    demand = []
    reserve = []
    for t in range(n_periods):
        day = t // base_T
        hour = t % base_T
        # Slight daily variation: ±3% sinusoidal over the week
        variation = 1.0 + 0.03 * _math.sin(2 * _math.pi * day / 7)
        demand.append(round(inst["demand"][hour] * variation))
        reserve.append(round(inst["reserve"][hour] * variation))
    sub["demand"] = demand
    sub["reserve"] = reserve
    sub["name"] = f"{inst['name']}-{n_periods}p"
    sub["known_bounds"] = {}
    return sub
