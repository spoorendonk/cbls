"""Maritime fleet bunker-ECA benchmark instance data.

Four instance sizes matching the C++ make_small(), make_medium(), make_large(),
make_xlarge() factories in benchmarks/bunker-eca/data.h.

Usage:
    python data.py          # saves all instances as JSONL files
"""

import json
import math
import os


# ---------------------------------------------------------------------------
# Small instance: 3 ships, 10 cargoes, 7 regions, 60 days
# ---------------------------------------------------------------------------
def make_small():
    inst = {
        "name": "small-3s-10c",
        "planning_horizon_days": 60.0,
        "bonus_price": 300.0,
        "regions": [
            {"name": "Rotterdam",  "is_eca": True,  "hfo_price": 450.0, "mgo_price": 750.0, "port_cost": 5000.0, "bunker_available": True},
            {"name": "Hamburg",    "is_eca": True,  "hfo_price": 460.0, "mgo_price": 760.0, "port_cost": 4500.0, "bunker_available": True},
            {"name": "Singapore",  "is_eca": False, "hfo_price": 400.0, "mgo_price": 700.0, "port_cost": 3000.0, "bunker_available": True},
            {"name": "Houston",    "is_eca": True,  "hfo_price": 420.0, "mgo_price": 720.0, "port_cost": 4000.0, "bunker_available": True},
            {"name": "Dubai",      "is_eca": False, "hfo_price": 380.0, "mgo_price": 680.0, "port_cost": 3500.0, "bunker_available": True},
            {"name": "Shanghai",   "is_eca": False, "hfo_price": 410.0, "mgo_price": 710.0, "port_cost": 3200.0, "bunker_available": True},
            {"name": "Santos",     "is_eca": False, "hfo_price": 430.0, "mgo_price": 730.0, "port_cost": 3800.0, "bunker_available": False},
        ],
        "ships": [
            {"name": "Vessel-A", "v_min_laden": 11.0, "v_max_laden": 14.5, "v_min_ballast": 12.0, "v_max_ballast": 14.5,
             "fuel_coeff_laden": 0.0035, "fuel_coeff_ballast": 0.0028,
             "hfo_tank_max": 2500.0, "mgo_tank_max": 500.0, "hfo_safety": 125.0, "mgo_safety": 25.0,
             "min_bunkering": 100.0, "initial_hfo": 1800.0, "initial_mgo": 200.0, "origin_region": 0, "available_day": 0.0},
            {"name": "Vessel-B", "v_min_laden": 11.0, "v_max_laden": 14.5, "v_min_ballast": 12.0, "v_max_ballast": 14.5,
             "fuel_coeff_laden": 0.0038, "fuel_coeff_ballast": 0.0030,
             "hfo_tank_max": 2500.0, "mgo_tank_max": 500.0, "hfo_safety": 125.0, "mgo_safety": 25.0,
             "min_bunkering": 100.0, "initial_hfo": 2000.0, "initial_mgo": 300.0, "origin_region": 2, "available_day": 2.0},
            {"name": "Vessel-C", "v_min_laden": 11.0, "v_max_laden": 14.5, "v_min_ballast": 12.0, "v_max_ballast": 14.5,
             "fuel_coeff_laden": 0.0033, "fuel_coeff_ballast": 0.0026,
             "hfo_tank_max": 2500.0, "mgo_tank_max": 500.0, "hfo_safety": 125.0, "mgo_safety": 25.0,
             "min_bunkering": 100.0, "initial_hfo": 1500.0, "initial_mgo": 150.0, "origin_region": 4, "available_day": 1.0},
        ],
        "cargoes": [
            # Contract cargoes (6)
            {"pickup_region": 0, "delivery_region": 1, "quantity": 25000, "revenue": 80000,
             "pickup_tw_start": 0.0, "pickup_tw_end": 5.0, "delivery_tw_start": 3.0, "delivery_tw_end": 12.0,
             "service_time_load": 1.5, "service_time_discharge": 1.0, "is_contract": True},
            {"pickup_region": 2, "delivery_region": 5, "quantity": 30000, "revenue": 120000,
             "pickup_tw_start": 0.0, "pickup_tw_end": 8.0, "delivery_tw_start": 10.0, "delivery_tw_end": 22.0,
             "service_time_load": 1.0, "service_time_discharge": 1.0, "is_contract": True},
            {"pickup_region": 4, "delivery_region": 2, "quantity": 20000, "revenue": 140000,
             "pickup_tw_start": 0.0, "pickup_tw_end": 5.0, "delivery_tw_start": 14.0, "delivery_tw_end": 28.0,
             "service_time_load": 1.0, "service_time_discharge": 1.5, "is_contract": True},
            {"pickup_region": 3, "delivery_region": 6, "quantity": 28000, "revenue": 180000,
             "pickup_tw_start": 0.0, "pickup_tw_end": 8.0, "delivery_tw_start": 18.0, "delivery_tw_end": 35.0,
             "service_time_load": 1.5, "service_time_discharge": 1.0, "is_contract": True},
            {"pickup_region": 5, "delivery_region": 4, "quantity": 22000, "revenue": 160000,
             "pickup_tw_start": 0.0, "pickup_tw_end": 5.0, "delivery_tw_start": 16.0, "delivery_tw_end": 30.0,
             "service_time_load": 1.0, "service_time_discharge": 1.5, "is_contract": True},
            {"pickup_region": 2, "delivery_region": 4, "quantity": 18000, "revenue": 100000,
             "pickup_tw_start": 5.0, "pickup_tw_end": 15.0, "delivery_tw_start": 20.0, "delivery_tw_end": 38.0,
             "service_time_load": 1.0, "service_time_discharge": 1.0, "is_contract": True},
            # Spot cargoes (4)
            {"pickup_region": 0, "delivery_region": 3, "quantity": 15000, "revenue": 200000,
             "pickup_tw_start": 0.0, "pickup_tw_end": 8.0, "delivery_tw_start": 18.0, "delivery_tw_end": 35.0,
             "service_time_load": 1.0, "service_time_discharge": 1.0, "is_contract": False},
            {"pickup_region": 6, "delivery_region": 3, "quantity": 20000, "revenue": 170000,
             "pickup_tw_start": 0.0, "pickup_tw_end": 8.0, "delivery_tw_start": 20.0, "delivery_tw_end": 38.0,
             "service_time_load": 1.5, "service_time_discharge": 1.5, "is_contract": False},
            {"pickup_region": 4, "delivery_region": 1, "quantity": 12000, "revenue": 190000,
             "pickup_tw_start": 0.0, "pickup_tw_end": 8.0, "delivery_tw_start": 22.0, "delivery_tw_end": 42.0,
             "service_time_load": 1.0, "service_time_discharge": 1.0, "is_contract": False},
            {"pickup_region": 5, "delivery_region": 2, "quantity": 25000, "revenue": 110000,
             "pickup_tw_start": 0.0, "pickup_tw_end": 8.0, "delivery_tw_start": 10.0, "delivery_tw_end": 24.0,
             "service_time_load": 1.0, "service_time_discharge": 1.5, "is_contract": False},
        ],
        "legs": [],
        "bunker_options": [
            {"region": 0, "day": 0.0, "hfo_price": 450.0, "mgo_price": 750.0},
            {"region": 2, "day": 0.0, "hfo_price": 400.0, "mgo_price": 700.0},
            {"region": 4, "day": 0.0, "hfo_price": 380.0, "mgo_price": 680.0},
            {"region": 3, "day": 0.0, "hfo_price": 420.0, "mgo_price": 720.0},
            {"region": 5, "day": 0.0, "hfo_price": 410.0, "mgo_price": 710.0},
        ],
    }

    # Legs: symmetric pairs
    leg_data = [
        (0, 1,   400, 1.0),
        (0, 2,  8300, 0.05),
        (0, 3,  5000, 0.15),
        (0, 4,  6200, 0.03),
        (0, 5,  9800, 0.03),
        (0, 6,  5800, 0.03),
        (1, 2,  8600, 0.05),
        (1, 3,  5300, 0.12),
        (1, 4,  6500, 0.03),
        (1, 5, 10100, 0.03),
        (2, 3, 12500, 0.02),
        (2, 4,  3500, 0.0),
        (2, 5,  2500, 0.0),
        (2, 6,  9200, 0.0),
        (3, 4, 10800, 0.04),
        (3, 5, 12000, 0.04),
        (3, 6,  5000, 0.03),
        (4, 5,  4200, 0.0),
        (4, 6,  7500, 0.0),
        (5, 6, 10500, 0.0),
    ]
    for a, b, dist, eca in leg_data:
        inst["legs"].append({"from_region": a, "to_region": b, "distance": dist, "eca_fraction": eca})
        inst["legs"].append({"from_region": b, "to_region": a, "distance": dist, "eca_fraction": eca})

    return inst


# ---------------------------------------------------------------------------
# Medium instance: 7 ships, 30 cargoes, 15 regions, 90 days
# ---------------------------------------------------------------------------
def make_medium():
    regions = [
        {"name": "Rotterdam",    "is_eca": True,  "hfo_price": 450.0, "mgo_price": 750.0, "port_cost": 5000.0, "bunker_available": True},
        {"name": "Hamburg",      "is_eca": True,  "hfo_price": 460.0, "mgo_price": 760.0, "port_cost": 4500.0, "bunker_available": True},
        {"name": "Antwerp",      "is_eca": True,  "hfo_price": 455.0, "mgo_price": 755.0, "port_cost": 4800.0, "bunker_available": True},
        {"name": "Singapore",    "is_eca": False, "hfo_price": 400.0, "mgo_price": 700.0, "port_cost": 3000.0, "bunker_available": True},
        {"name": "Houston",      "is_eca": True,  "hfo_price": 420.0, "mgo_price": 720.0, "port_cost": 4000.0, "bunker_available": True},
        {"name": "New_York",     "is_eca": True,  "hfo_price": 430.0, "mgo_price": 730.0, "port_cost": 4500.0, "bunker_available": True},
        {"name": "Dubai",        "is_eca": False, "hfo_price": 380.0, "mgo_price": 680.0, "port_cost": 3500.0, "bunker_available": True},
        {"name": "Shanghai",     "is_eca": False, "hfo_price": 410.0, "mgo_price": 710.0, "port_cost": 3200.0, "bunker_available": True},
        {"name": "Busan",        "is_eca": False, "hfo_price": 415.0, "mgo_price": 715.0, "port_cost": 3100.0, "bunker_available": True},
        {"name": "Santos",       "is_eca": False, "hfo_price": 430.0, "mgo_price": 730.0, "port_cost": 3800.0, "bunker_available": False},
        {"name": "Durban",       "is_eca": False, "hfo_price": 390.0, "mgo_price": 690.0, "port_cost": 3600.0, "bunker_available": True},
        {"name": "Mumbai",       "is_eca": False, "hfo_price": 385.0, "mgo_price": 685.0, "port_cost": 3400.0, "bunker_available": True},
        {"name": "Gothenburg",   "is_eca": True,  "hfo_price": 465.0, "mgo_price": 765.0, "port_cost": 4200.0, "bunker_available": True},
        {"name": "Los_Angeles",  "is_eca": True,  "hfo_price": 425.0, "mgo_price": 725.0, "port_cost": 3900.0, "bunker_available": True},
        {"name": "Tokyo",        "is_eca": False, "hfo_price": 420.0, "mgo_price": 720.0, "port_cost": 3300.0, "bunker_available": True},
    ]

    inst = {
        "name": "medium-7s-30c",
        "planning_horizon_days": 90.0,
        "bonus_price": 300.0,
        "regions": regions,
        "ships": [
            {"name": "Vessel-A", "v_min_laden": 11.0, "v_max_laden": 14.5, "v_min_ballast": 12.0, "v_max_ballast": 14.5,
             "fuel_coeff_laden": 0.0035, "fuel_coeff_ballast": 0.0028,
             "hfo_tank_max": 2500.0, "mgo_tank_max": 500.0, "hfo_safety": 125.0, "mgo_safety": 25.0,
             "min_bunkering": 100.0, "initial_hfo": 1800.0, "initial_mgo": 200.0, "origin_region": 0, "available_day": 0.0},
            {"name": "Vessel-B", "v_min_laden": 11.0, "v_max_laden": 14.5, "v_min_ballast": 12.0, "v_max_ballast": 14.5,
             "fuel_coeff_laden": 0.0038, "fuel_coeff_ballast": 0.0030,
             "hfo_tank_max": 2500.0, "mgo_tank_max": 500.0, "hfo_safety": 125.0, "mgo_safety": 25.0,
             "min_bunkering": 100.0, "initial_hfo": 2000.0, "initial_mgo": 300.0, "origin_region": 3, "available_day": 2.0},
            {"name": "Vessel-C", "v_min_laden": 11.0, "v_max_laden": 14.5, "v_min_ballast": 12.0, "v_max_ballast": 14.5,
             "fuel_coeff_laden": 0.0033, "fuel_coeff_ballast": 0.0026,
             "hfo_tank_max": 2500.0, "mgo_tank_max": 500.0, "hfo_safety": 125.0, "mgo_safety": 25.0,
             "min_bunkering": 100.0, "initial_hfo": 1500.0, "initial_mgo": 150.0, "origin_region": 6, "available_day": 1.0},
            {"name": "Vessel-D", "v_min_laden": 11.5, "v_max_laden": 14.0, "v_min_ballast": 12.5, "v_max_ballast": 14.0,
             "fuel_coeff_laden": 0.0036, "fuel_coeff_ballast": 0.0029,
             "hfo_tank_max": 2200.0, "mgo_tank_max": 450.0, "hfo_safety": 110.0, "mgo_safety": 22.5,
             "min_bunkering": 100.0, "initial_hfo": 1600.0, "initial_mgo": 250.0, "origin_region": 7, "available_day": 0.0},
            {"name": "Vessel-E", "v_min_laden": 10.5, "v_max_laden": 14.5, "v_min_ballast": 11.5, "v_max_ballast": 14.5,
             "fuel_coeff_laden": 0.0040, "fuel_coeff_ballast": 0.0032,
             "hfo_tank_max": 2800.0, "mgo_tank_max": 550.0, "hfo_safety": 140.0, "mgo_safety": 27.5,
             "min_bunkering": 100.0, "initial_hfo": 2200.0, "initial_mgo": 350.0, "origin_region": 4, "available_day": 3.0},
            {"name": "Vessel-F", "v_min_laden": 11.0, "v_max_laden": 14.0, "v_min_ballast": 12.0, "v_max_ballast": 14.0,
             "fuel_coeff_laden": 0.0034, "fuel_coeff_ballast": 0.0027,
             "hfo_tank_max": 2400.0, "mgo_tank_max": 480.0, "hfo_safety": 120.0, "mgo_safety": 24.0,
             "min_bunkering": 100.0, "initial_hfo": 1700.0, "initial_mgo": 180.0, "origin_region": 0, "available_day": 5.0},
            {"name": "Vessel-G", "v_min_laden": 11.5, "v_max_laden": 14.5, "v_min_ballast": 12.5, "v_max_ballast": 14.5,
             "fuel_coeff_laden": 0.0032, "fuel_coeff_ballast": 0.0025,
             "hfo_tank_max": 2600.0, "mgo_tank_max": 520.0, "hfo_safety": 130.0, "mgo_safety": 26.0,
             "min_bunkering": 100.0, "initial_hfo": 1900.0, "initial_mgo": 280.0, "origin_region": 3, "available_day": 1.0},
        ],
        "cargoes": [],
        "legs": [],
        "bunker_options": [],
    }

    # Legs: key routes (matching C++ set_dist calls)
    known_legs = [
        (0, 1,    400, 1.0),
        (0, 2,    200, 1.0),
        (0, 3,   8300, 0.05),
        (0, 4,   5000, 0.15),
        (0, 5,   3500, 0.12),
        (0, 6,   6200, 0.03),
        (0, 7,   9800, 0.03),
        (0, 8,  10200, 0.03),
        (0, 9,   5800, 0.03),
        (0, 10,  6900, 0.02),
        (0, 11,  6400, 0.02),
        (0, 12,   500, 1.0),
        (0, 13,  8000, 0.04),
        (0, 14, 10500, 0.03),
        (1, 3,   8600, 0.05),
        (1, 4,   5300, 0.12),
        (1, 12,   300, 1.0),
        (2, 3,   8400, 0.05),
        (2, 5,   3600, 0.12),
        (3, 4,  12500, 0.02),
        (3, 6,   3500, 0.0),
        (3, 7,   2500, 0.0),
        (3, 8,   2800, 0.0),
        (3, 9,   9200, 0.0),
        (3, 10,  4600, 0.0),
        (3, 11,  2800, 0.0),
        (3, 13,  7800, 0.02),
        (3, 14,  3100, 0.0),
        (4, 5,   1800, 0.15),
        (4, 6,  10800, 0.04),
        (4, 7,  12000, 0.04),
        (4, 9,   5000, 0.03),
        (4, 13,  4500, 0.08),
        (5, 9,   5200, 0.06),
        (5, 13,  4800, 0.08),
        (6, 7,   4200, 0.0),
        (6, 8,   4500, 0.0),
        (6, 10,  3200, 0.0),
        (6, 11,  1200, 0.0),
        (7, 8,    500, 0.0),
        (7, 14,  1100, 0.0),
        (8, 14,   700, 0.0),
        (9, 10,  4000, 0.0),
        (11, 10, 2600, 0.0),
        (13, 14, 5500, 0.02),
    ]

    # Track which pairs have explicit distances
    has_dist = set()
    for a, b, dist, eca in known_legs:
        inst["legs"].append({"from_region": a, "to_region": b, "distance": dist, "eca_fraction": eca})
        inst["legs"].append({"from_region": b, "to_region": a, "distance": dist, "eca_fraction": eca})
        has_dist.add((a, b))
        has_dist.add((b, a))

    # Fill missing pairs with estimates (matching C++ logic)
    for i in range(15):
        for j in range(i + 1, 15):
            if (i, j) not in has_dist:
                d = 6000.0
                ri = regions[i]["is_eca"]
                rj = regions[j]["is_eca"]
                if ri and rj:
                    eca = 0.3
                elif ri or rj:
                    eca = 0.05
                else:
                    eca = 0.0
                inst["legs"].append({"from_region": i, "to_region": j, "distance": d, "eca_fraction": eca})
                inst["legs"].append({"from_region": j, "to_region": i, "distance": d, "eca_fraction": eca})

    # 30 cargoes: 18 contract + 12 spot (matching C++ add_cargo calls)
    cargo_data = [
        # Contract cargoes (18)
        (0, 3,  25000, 180000,  0, 10, 28, 45, 1.5, 1.0, True),
        (3, 7,  30000, 120000,  0, 10,  9, 22, 1.0, 1.0, True),
        (6, 0,  20000, 200000,  0, 10, 22, 38, 1.0, 1.5, True),
        (4, 3,  28000, 160000,  0, 10, 40, 60, 1.5, 1.0, True),
        (7, 4,  22000, 190000,  0, 10, 14, 28, 1.0, 1.5, True),
        (3, 6,  18000, 100000,  0, 14, 12, 28, 1.0, 1.0, True),
        (0, 7,  24000, 210000,  5, 15, 35, 55, 1.5, 1.0, True),
        (7, 0,  26000, 195000, 10, 20, 42, 62, 1.0, 1.5, True),
        (6, 3,  21000, 140000,  0, 12, 12, 28, 1.0, 1.0, True),
        (4, 7,  27000, 170000,  0, 10, 20, 35, 1.5, 1.0, True),
        (3, 0,  19000, 185000,  5, 15, 32, 50, 1.0, 1.5, True),
        (11, 7, 23000, 115000,  0, 10,  8, 22, 1.0, 1.0, True),
        (10, 0, 20000, 175000,  0, 12, 24, 42, 1.5, 1.0, True),
        (8, 4,  25000, 165000,  0, 10, 20, 35, 1.0, 1.5, True),
        (0, 6,  22000, 155000,  0, 10, 22, 38, 1.0, 1.0, True),
        (7, 6,  18000, 105000, 10, 20, 22, 38, 1.0, 1.0, True),
        (6, 7,  24000, 130000, 15, 25, 28, 45, 1.0, 1.0, True),
        (3, 4,  20000, 175000,  5, 15, 45, 65, 1.5, 1.5, True),
        # Spot cargoes (12)
        (0, 7,  15000, 250000,  0, 10, 35, 55, 1.0, 1.0, False),
        (9, 4,  20000, 170000,  0, 10, 18, 32, 1.5, 1.5, False),
        (6, 1,  12000, 150000,  0, 14, 24, 42, 1.0, 1.0, False),
        (7, 9,  25000, 130000,  0, 10, 32, 50, 1.0, 1.5, False),
        (5, 3,  18000, 200000,  0, 10, 12, 26, 1.0, 1.0, False),
        (14, 6, 16000, 145000,  5, 15, 20, 35, 1.0, 1.0, False),
        (11, 0, 22000, 220000,  0, 10, 22, 40, 1.5, 1.0, False),
        (8, 3,  19000, 125000,  5, 15, 12, 28, 1.0, 1.0, False),
        (13, 7, 21000, 180000,  0, 10, 20, 35, 1.0, 1.5, False),
        (0, 9,  17000, 160000,  0, 14, 20, 38, 1.0, 1.0, False),
        (3, 10, 23000, 140000,  5, 15, 20, 35, 1.5, 1.0, False),
        (4, 8,  20000, 155000, 10, 20, 28, 42, 1.0, 1.0, False),
    ]
    for p, d, q, r, ps, pe, ds, de, sl, sd, contract in cargo_data:
        inst["cargoes"].append({
            "pickup_region": p, "delivery_region": d, "quantity": q, "revenue": r,
            "pickup_tw_start": float(ps), "pickup_tw_end": float(pe),
            "delivery_tw_start": float(ds), "delivery_tw_end": float(de),
            "service_time_load": sl, "service_time_discharge": sd,
            "is_contract": contract,
        })

    # Bunker options: at major hubs, every 10 days (matching C++ logic)
    for r in [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13]:
        d = 0.0
        while d < inst["planning_horizon_days"]:
            price_var = 1.0 + 0.02 * math.sin(d * 0.3 + r)
            inst["bunker_options"].append({
                "region": r,
                "day": d,
                "hfo_price": regions[r]["hfo_price"] * price_var,
                "mgo_price": regions[r]["mgo_price"] * price_var,
            })
            d += 10.0

    return inst


# ---------------------------------------------------------------------------
# Large instance: 15 ships, 60 cargoes (doubled medium + extra ship)
# ---------------------------------------------------------------------------
def make_large():
    base = make_medium()
    base["name"] = "large-15s-60c"
    base["planning_horizon_days"] = 90.0
    n_regions = len(base["regions"])

    # Double ships with modifications
    orig_ships = list(base["ships"])
    for i, s in enumerate(orig_ships):
        s2 = dict(s)
        s2["name"] = s["name"] + "-2"
        s2["available_day"] = s["available_day"] + 5.0
        s2["origin_region"] = (s["origin_region"] + 3) % n_regions
        s2["fuel_coeff_laden"] = s["fuel_coeff_laden"] * (1.0 + 0.05 * (i % 3))
        base["ships"].append(s2)

    # Add one more
    base["ships"].append({
        "name": "Vessel-H",
        "v_min_laden": 11.0, "v_max_laden": 14.0,
        "v_min_ballast": 12.0, "v_max_ballast": 14.0,
        "fuel_coeff_laden": 0.0037, "fuel_coeff_ballast": 0.0029,
        "hfo_tank_max": 2300.0, "mgo_tank_max": 460.0,
        "hfo_safety": 115.0, "mgo_safety": 23.0,
        "min_bunkering": 100.0,
        "initial_hfo": 1700.0, "initial_mgo": 220.0,
        "origin_region": 10, "available_day": 4.0,
    })

    # Double cargoes with shifted time windows
    orig_cargoes = list(base["cargoes"])
    for i, c in enumerate(orig_cargoes):
        c2 = dict(c)
        c2["pickup_tw_start"] = c["pickup_tw_start"] + 30.0
        c2["pickup_tw_end"] = c["pickup_tw_end"] + 30.0
        c2["delivery_tw_start"] = c["delivery_tw_start"] + 30.0
        c2["delivery_tw_end"] = c["delivery_tw_end"] + 30.0
        c2["revenue"] = c["revenue"] * (0.9 + 0.2 * ((i % 5) / 5.0))
        base["cargoes"].append(c2)

    # More bunker options for extended horizon
    regions = base["regions"]
    for r in [0, 3, 6, 7]:
        d = 60.0
        while d < 90.0:
            price_var = 1.0 + 0.02 * math.sin(d * 0.3 + r)
            base["bunker_options"].append({
                "region": r,
                "day": d,
                "hfo_price": regions[r]["hfo_price"] * price_var,
                "mgo_price": regions[r]["mgo_price"] * price_var,
            })
            d += 10.0

    return base


# ---------------------------------------------------------------------------
# XLarge instance: 30 ships, 120 cargoes (doubled large)
# ---------------------------------------------------------------------------
def make_xlarge():
    base = make_large()
    base["name"] = "xlarge-30s-120c"
    base["planning_horizon_days"] = 120.0
    n_regions = len(base["regions"])

    # Double ships
    orig_ships = list(base["ships"])
    for s in orig_ships:
        s2 = dict(s)
        s2["name"] = s["name"] + "-3"
        s2["available_day"] = s["available_day"] + 8.0
        s2["origin_region"] = (s["origin_region"] + 5) % n_regions
        base["ships"].append(s2)

    # Double cargoes
    orig_cargoes = list(base["cargoes"])
    for i, c in enumerate(orig_cargoes):
        c2 = dict(c)
        c2["pickup_tw_start"] = c["pickup_tw_start"] + 60.0
        c2["pickup_tw_end"] = c["pickup_tw_end"] + 60.0
        c2["delivery_tw_start"] = c["delivery_tw_start"] + 60.0
        c2["delivery_tw_end"] = c["delivery_tw_end"] + 60.0
        c2["revenue"] = c["revenue"] * (0.85 + 0.3 * ((i % 7) / 7.0))
        base["cargoes"].append(c2)

    return base


# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------
def save_jsonl(inst, path):
    """Serialize instance to a single-line JSON file (JSONL format)."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(inst, f)
        f.write("\n")


ALL_INSTANCES = {
    "small": make_small,
    "medium": make_medium,
    "large": make_large,
    "xlarge": make_xlarge,
}


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for key, factory in ALL_INSTANCES.items():
        inst = factory()
        path = os.path.join(out_dir, f"{inst['name']}.jsonl")
        save_jsonl(inst, path)
        print(f"Saved {inst['name']}: {len(inst['ships'])} ships, "
              f"{len(inst['cargoes'])} cargoes, {len(inst['regions'])} regions, "
              f"{len(inst['legs'])} legs, {len(inst['bunker_options'])} bunker opts "
              f"-> {path}")
