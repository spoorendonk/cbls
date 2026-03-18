#pragma once

#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>

namespace cbls {
namespace bunker_eca {

struct Region {
    std::string name;
    bool is_eca;
    double hfo_price;   // $/MT
    double mgo_price;   // $/MT
    double port_cost;   // $ per port call
    bool bunker_available;
};

struct Cargo {
    int pickup_region;
    int delivery_region;
    double quantity;        // MT
    double revenue;         // $
    double pickup_tw_start; // days from t=0
    double pickup_tw_end;
    double delivery_tw_start;
    double delivery_tw_end;
    double service_time_load;      // days
    double service_time_discharge; // days
    bool is_contract;  // must be carried
};

struct Ship {
    std::string name;
    double v_min_laden;     // knots
    double v_max_laden;
    double v_min_ballast;
    double v_max_ballast;
    double fuel_coeff_laden;    // k: daily consumption = k * v^3
    double fuel_coeff_ballast;
    double hfo_tank_max;    // MT
    double mgo_tank_max;
    double hfo_safety;      // MT minimum tank level
    double mgo_safety;
    double min_bunkering;   // MT minimum purchase quantity
    double initial_hfo;
    double initial_mgo;
    int origin_region;
    double available_day;   // day ship becomes available
};

struct Leg {
    int from_region;
    int to_region;
    double distance;        // nautical miles
    double eca_fraction;    // 0-1, fraction of leg in ECA
};

struct BunkerOption {
    int region;
    double day;
    double hfo_price;   // $/MT
    double mgo_price;   // $/MT
};

struct Instance {
    std::string name;
    std::vector<Region> regions;
    std::vector<Cargo> cargoes;
    std::vector<Ship> ships;
    std::vector<Leg> legs;            // all region-pair legs
    std::vector<BunkerOption> bunker_options;
    double planning_horizon_days;
    double bonus_price;               // $/MT for final tank value
    double known_optimum;             // -1 = unknown

    // Convenience: get leg between two regions (-1 if not found)
    int find_leg(int from, int to) const {
        for (int i = 0; i < (int)legs.size(); ++i) {
            if (legs[i].from_region == from && legs[i].to_region == to)
                return i;
        }
        return -1;
    }

    // Get leg distance, 0 if same region
    double leg_distance(int from, int to) const {
        if (from == to) return 0.0;
        int idx = find_leg(from, to);
        return idx >= 0 ? legs[idx].distance : 9999.0;
    }

    // Get ECA fraction for a leg
    double leg_eca_fraction(int from, int to) const {
        if (from == to) return 0.0;
        int idx = find_leg(from, to);
        return idx >= 0 ? legs[idx].eca_fraction : 0.0;
    }
};

// ---------------------------------------------------------------------------
// Small instance: 3 ships, 10 cargoes, 7 regions, 60 days
// Uses shorter routes within feasible speed/time window constraints.
// At v_max=14.5kn: sailing_days = dist / (24*14.5) = dist / 348
// ---------------------------------------------------------------------------
inline Instance make_small() {
    Instance inst;
    inst.name = "small-3s-10c";
    inst.planning_horizon_days = 60.0;
    inst.bonus_price = 300.0;  // $/MT
    inst.known_optimum = -1.0;

    // Regions: 0=Rotterdam(ECA), 1=Hamburg(ECA), 2=Singapore, 3=Houston(ECA),
    //          4=Dubai, 5=Shanghai, 6=Santos
    inst.regions = {
        {"Rotterdam", true,  450.0, 750.0, 5000.0, true},
        {"Hamburg",   true,  460.0, 760.0, 4500.0, true},
        {"Singapore", false, 400.0, 700.0, 3000.0, true},
        {"Houston",  true,  420.0, 720.0, 4000.0, true},
        {"Dubai",    false, 380.0, 680.0, 3500.0, true},
        {"Shanghai", false, 410.0, 710.0, 3200.0, true},
        {"Santos",   false, 430.0, 730.0, 3800.0, false},
    };

    // Legs: distances in nautical miles, eca_fraction
    // Symmetric: add both directions
    struct LegData { int a, b; double dist; double eca; };
    std::vector<LegData> ld = {
        {0, 1,  400, 1.0},   // Rotterdam-Hamburg: fully ECA
        {0, 2, 8300, 0.05},  // Rotterdam-Singapore
        {0, 3, 5000, 0.15},  // Rotterdam-Houston: ECA at both ends
        {0, 4, 6200, 0.03},  // Rotterdam-Dubai
        {0, 5, 9800, 0.03},  // Rotterdam-Shanghai
        {0, 6, 5800, 0.03},  // Rotterdam-Santos
        {1, 2, 8600, 0.05},  // Hamburg-Singapore
        {1, 3, 5300, 0.12},  // Hamburg-Houston
        {1, 4, 6500, 0.03},  // Hamburg-Dubai
        {1, 5,10100, 0.03},  // Hamburg-Shanghai
        {2, 3,12500, 0.02},  // Singapore-Houston
        {2, 4, 3500, 0.0},   // Singapore-Dubai
        {2, 5, 2500, 0.0},   // Singapore-Shanghai
        {2, 6, 9200, 0.0},   // Singapore-Santos
        {3, 4,10800, 0.04},  // Houston-Dubai
        {3, 5,12000, 0.04},  // Houston-Shanghai
        {3, 6, 5000, 0.03},  // Houston-Santos
        {4, 5, 4200, 0.0},   // Dubai-Shanghai
        {4, 6, 7500, 0.0},   // Dubai-Santos
        {5, 6,10500, 0.0},   // Shanghai-Santos
    };
    for (auto& l : ld) {
        inst.legs.push_back({l.a, l.b, l.dist, l.eca});
        inst.legs.push_back({l.b, l.a, l.dist, l.eca});
    }

    // Ships
    inst.ships = {
        {"Vessel-A", 11.0, 14.5, 12.0, 14.5, 0.0035, 0.0028,
         2500.0, 500.0, 125.0, 25.0, 100.0, 1800.0, 200.0, 0, 0.0},
        {"Vessel-B", 11.0, 14.5, 12.0, 14.5, 0.0038, 0.0030,
         2500.0, 500.0, 125.0, 25.0, 100.0, 2000.0, 300.0, 2, 2.0},
        {"Vessel-C", 11.0, 14.5, 12.0, 14.5, 0.0033, 0.0026,
         2500.0, 500.0, 125.0, 25.0, 100.0, 1500.0, 150.0, 4, 1.0},
    };

    // Cargoes: 6 contract + 4 spot
    // Time windows computed with generous slack vs min sailing time
    // Min sailing time at 14.5kn = dist/348 days
    inst.cargoes = {
        // Contract cargoes
        // Rotterdam(0)->Hamburg(1): 400nm, ~1.2d sailing
        {0, 1, 25000, 80000, 0.0, 5.0, 3.0, 12.0, 1.5, 1.0, true},
        // Singapore(2)->Shanghai(5): 2500nm, ~7.2d sailing
        {2, 5, 30000, 120000, 0.0, 8.0, 10.0, 22.0, 1.0, 1.0, true},
        // Dubai(4)->Singapore(2): 3500nm, ~10.1d sailing
        {4, 2, 20000, 140000, 0.0, 5.0, 14.0, 28.0, 1.0, 1.5, true},
        // Houston(3)->Santos(6): 5000nm, ~14.4d sailing
        {3, 6, 28000, 180000, 0.0, 8.0, 18.0, 35.0, 1.5, 1.0, true},
        // Shanghai(5)->Dubai(4): 4200nm, ~12.1d sailing
        {5, 4, 22000, 160000, 0.0, 5.0, 16.0, 30.0, 1.0, 1.5, true},
        // Singapore(2)->Dubai(4): 3500nm, ~10.1d sailing
        {2, 4, 18000, 100000, 5.0, 15.0, 20.0, 38.0, 1.0, 1.0, true},
        // Spot cargoes
        // Rotterdam(0)->Houston(3): 5000nm, ~14.4d sailing, ECA at both ends
        {0, 3, 15000, 200000, 0.0, 8.0, 18.0, 35.0, 1.0, 1.0, false},
        // Santos(6)->Houston(3): 5000nm, ~14.4d sailing
        {6, 3, 20000, 170000, 0.0, 8.0, 20.0, 38.0, 1.5, 1.5, false},
        // Dubai(4)->Hamburg(1): 6500nm, ~18.7d sailing
        {4, 1, 12000, 190000, 0.0, 8.0, 22.0, 42.0, 1.0, 1.0, false},
        // Shanghai(5)->Singapore(2): 2500nm, ~7.2d sailing
        {5, 2, 25000, 110000, 0.0, 8.0, 10.0, 24.0, 1.0, 1.5, false},
    };

    // Bunker options: at major hubs (fewer options to keep model tractable)
    inst.bunker_options = {
        {0, 0.0,  450.0, 750.0},  // Rotterdam, day 0
        {2, 0.0,  400.0, 700.0},  // Singapore, day 0
        {4, 0.0,  380.0, 680.0},  // Dubai, day 0
        {3, 0.0,  420.0, 720.0},  // Houston, day 0
        {5, 0.0,  410.0, 710.0},  // Shanghai, day 0
    };

    return inst;
}

// ---------------------------------------------------------------------------
// Medium instance: 7 ships, 30 cargoes, 15 regions, 60 days
// ---------------------------------------------------------------------------
inline Instance make_medium() {
    Instance inst;
    inst.name = "medium-7s-30c";
    inst.planning_horizon_days = 90.0;
    inst.bonus_price = 300.0;
    inst.known_optimum = -1.0;

    // 15 regions
    inst.regions = {
        {"Rotterdam",    true,  450.0, 750.0, 5000.0, true},   // 0
        {"Hamburg",      true,  460.0, 760.0, 4500.0, true},   // 1
        {"Antwerp",     true,  455.0, 755.0, 4800.0, true},    // 2
        {"Singapore",   false, 400.0, 700.0, 3000.0, true},    // 3
        {"Houston",     true,  420.0, 720.0, 4000.0, true},    // 4
        {"New_York",    true,  430.0, 730.0, 4500.0, true},    // 5
        {"Dubai",       false, 380.0, 680.0, 3500.0, true},    // 6
        {"Shanghai",    false, 410.0, 710.0, 3200.0, true},    // 7
        {"Busan",       false, 415.0, 715.0, 3100.0, true},    // 8
        {"Santos",      false, 430.0, 730.0, 3800.0, false},   // 9
        {"Durban",      false, 390.0, 690.0, 3600.0, true},    // 10
        {"Mumbai",      false, 385.0, 685.0, 3400.0, true},    // 11
        {"Gothenburg",  true,  465.0, 765.0, 4200.0, true},    // 12
        {"Los_Angeles", true,  425.0, 725.0, 3900.0, true},    // 13
        {"Tokyo",       false, 420.0, 720.0, 3300.0, true},    // 14
    };

    // Generate legs between all region pairs
    // Approximate distances (nm)
    double dist_matrix[15][15] = {};
    auto set_dist = [&](int a, int b, double d, double eca) {
        dist_matrix[a][b] = d;
        dist_matrix[b][a] = d;
        inst.legs.push_back({a, b, d, eca});
        inst.legs.push_back({b, a, d, eca});
    };

    // Key routes (simplified — not all pairs, add remaining as estimates)
    set_dist(0, 1,   400, 1.0);
    set_dist(0, 2,   200, 1.0);
    set_dist(0, 3,  8300, 0.05);
    set_dist(0, 4,  5000, 0.15);
    set_dist(0, 5,  3500, 0.12);
    set_dist(0, 6,  6200, 0.03);
    set_dist(0, 7,  9800, 0.03);
    set_dist(0, 8, 10200, 0.03);
    set_dist(0, 9,  5800, 0.03);
    set_dist(0, 10, 6900, 0.02);
    set_dist(0, 11, 6400, 0.02);
    set_dist(0, 12,  500, 1.0);
    set_dist(0, 13, 8000, 0.04);
    set_dist(0, 14,10500, 0.03);
    set_dist(1, 3,  8600, 0.05);
    set_dist(1, 4,  5300, 0.12);
    set_dist(1, 12,  300, 1.0);
    set_dist(2, 3,  8400, 0.05);
    set_dist(2, 5,  3600, 0.12);
    set_dist(3, 4, 12500, 0.02);
    set_dist(3, 6,  3500, 0.0);
    set_dist(3, 7,  2500, 0.0);
    set_dist(3, 8,  2800, 0.0);
    set_dist(3, 9,  9200, 0.0);
    set_dist(3, 10, 4600, 0.0);
    set_dist(3, 11, 2800, 0.0);
    set_dist(3, 13, 7800, 0.02);
    set_dist(3, 14, 3100, 0.0);
    set_dist(4, 5,  1800, 0.15);
    set_dist(4, 6, 10800, 0.04);
    set_dist(4, 7, 12000, 0.04);
    set_dist(4, 9,  5000, 0.03);
    set_dist(4, 13, 4500, 0.08);
    set_dist(5, 9,  5200, 0.06);
    set_dist(5, 13, 4800, 0.08);
    set_dist(6, 7,  4200, 0.0);
    set_dist(6, 8,  4500, 0.0);
    set_dist(6, 10, 3200, 0.0);
    set_dist(6, 11, 1200, 0.0);
    set_dist(7, 8,   500, 0.0);
    set_dist(7, 14, 1100, 0.0);
    set_dist(8, 14,  700, 0.0);
    set_dist(9, 10, 4000, 0.0);
    set_dist(11, 10, 2600, 0.0);
    set_dist(13, 14, 5500, 0.02);

    // Fill missing pairs with Euclidean estimates
    for (int i = 0; i < 15; ++i) {
        for (int j = i + 1; j < 15; ++j) {
            if (dist_matrix[i][j] == 0.0) {
                // Rough estimate: average of known distances
                double d = 6000.0;
                double eca = (inst.regions[i].is_eca && inst.regions[j].is_eca) ? 0.3
                           : (inst.regions[i].is_eca || inst.regions[j].is_eca) ? 0.05
                           : 0.0;
                inst.legs.push_back({i, j, d, eca});
                inst.legs.push_back({j, i, d, eca});
            }
        }
    }

    // 7 ships
    inst.ships = {
        {"Vessel-A", 11.0, 14.5, 12.0, 14.5, 0.0035, 0.0028, 2500.0, 500.0, 125.0, 25.0, 100.0, 1800.0, 200.0, 0, 0.0},
        {"Vessel-B", 11.0, 14.5, 12.0, 14.5, 0.0038, 0.0030, 2500.0, 500.0, 125.0, 25.0, 100.0, 2000.0, 300.0, 3, 2.0},
        {"Vessel-C", 11.0, 14.5, 12.0, 14.5, 0.0033, 0.0026, 2500.0, 500.0, 125.0, 25.0, 100.0, 1500.0, 150.0, 6, 1.0},
        {"Vessel-D", 11.5, 14.0, 12.5, 14.0, 0.0036, 0.0029, 2200.0, 450.0, 110.0, 22.5, 100.0, 1600.0, 250.0, 7, 0.0},
        {"Vessel-E", 10.5, 14.5, 11.5, 14.5, 0.0040, 0.0032, 2800.0, 550.0, 140.0, 27.5, 100.0, 2200.0, 350.0, 4, 3.0},
        {"Vessel-F", 11.0, 14.0, 12.0, 14.0, 0.0034, 0.0027, 2400.0, 480.0, 120.0, 24.0, 100.0, 1700.0, 180.0, 0, 5.0},
        {"Vessel-G", 11.5, 14.5, 12.5, 14.5, 0.0032, 0.0025, 2600.0, 520.0, 130.0, 26.0, 100.0, 1900.0, 280.0, 3, 1.0},
    };

    // 30 cargoes: 18 contract + 12 spot
    auto add_cargo = [&](int p, int d, double q, double r,
                         double ps, double pe, double ds, double de,
                         double sl, double sd, bool contract) {
        inst.cargoes.push_back({p, d, q, r, ps, pe, ds, de, sl, sd, contract});
    };

    // Contract cargoes (18) — time windows sized for distance at v_max=14.5kn
    // sailing_days = dist/348. Available = del_end - pick_start - service >= sailing + 2d slack
    add_cargo(0, 3,  25000, 180000,  0, 10, 28, 45, 1.5, 1.0, true);  // Rot->Sing 8300nm,24d
    add_cargo(3, 7,  30000, 120000,  0, 10,  9, 22, 1.0, 1.0, true);  // Sing->Sha 2500nm,7d
    add_cargo(6, 0,  20000, 200000,  0, 10, 22, 38, 1.0, 1.5, true);  // Dubai->Rot 6200nm,18d
    add_cargo(4, 3,  28000, 160000,  0, 10, 40, 60, 1.5, 1.0, true);  // Hou->Sing 12500nm,36d
    add_cargo(7, 4,  22000, 190000,  0, 10, 38, 55, 1.0, 1.5, true);  // Sha->Hou 12000nm,35d
    add_cargo(3, 6,  18000, 100000,  0, 14, 12, 28, 1.0, 1.0, true);  // Sing->Dubai 3500nm,10d
    add_cargo(0, 7,  24000, 210000,  5, 15, 35, 55, 1.5, 1.0, true);  // Rot->Sha 9800nm,28d
    add_cargo(7, 0,  26000, 195000, 10, 20, 42, 62, 1.0, 1.5, true);  // Sha->Rot 9800nm,28d
    add_cargo(6, 3,  21000, 140000,  0, 12, 12, 28, 1.0, 1.0, true);  // Dubai->Sing 3500nm,10d
    add_cargo(4, 7,  27000, 170000,  0, 10, 38, 55, 1.5, 1.0, true);  // Hou->Sha 12000nm,35d
    add_cargo(3, 0,  19000, 185000,  5, 15, 32, 50, 1.0, 1.5, true);  // Sing->Rot 8300nm,24d
    add_cargo(11,7,  23000, 115000,  0, 10, 12, 30, 1.0, 1.0, true);  // Mumbai->Sha ~6000nm,17d
    add_cargo(10,0,  20000, 175000,  0, 12, 24, 42, 1.5, 1.0, true);  // Durban->Rot 6900nm,20d
    add_cargo(8, 4,  25000, 165000,  0, 10, 20, 35, 1.0, 1.5, true);  // Busan->Hou ~6000nm,17d
    add_cargo(0, 6,  22000, 155000,  0, 10, 22, 38, 1.0, 1.0, true);  // Rot->Dubai 6200nm,18d
    add_cargo(7, 6,  18000, 105000, 10, 20, 22, 38, 1.0, 1.0, true);  // Sha->Dubai 4200nm,12d
    add_cargo(6, 7,  24000, 130000, 15, 25, 28, 45, 1.0, 1.0, true);  // Dubai->Sha 4200nm,12d
    add_cargo(3, 4,  20000, 175000,  5, 15, 45, 65, 1.5, 1.5, true);  // Sing->Hou 12500nm,36d

    // Spot cargoes (12) — wider time windows
    add_cargo(0, 7,  15000, 250000,  0, 10, 35, 55, 1.0, 1.0, false); // Rot->Sha 9800nm
    add_cargo(9, 4,  20000, 170000,  0, 10, 18, 32, 1.5, 1.5, false); // Santos->Hou 5000nm
    add_cargo(6, 1,  12000, 150000,  0, 14, 24, 42, 1.0, 1.0, false); // Dubai->Ham 6500nm
    add_cargo(7, 9,  25000, 130000,  0, 10, 32, 50, 1.0, 1.5, false); // Sha->Santos 10500nm
    add_cargo(5, 3,  18000, 200000,  0, 10, 12, 26, 1.0, 1.0, false); // NY->Sing ~6000nm
    add_cargo(14,6,  16000, 145000,  5, 15, 20, 35, 1.0, 1.0, false); // Tokyo->Dubai ~6000nm
    add_cargo(11,0,  22000, 220000,  0, 10, 22, 40, 1.5, 1.0, false); // Mumbai->Rot 6400nm
    add_cargo(8, 3,  19000, 125000,  5, 15, 12, 28, 1.0, 1.0, false); // Busan->Sing 2800nm
    add_cargo(13,7,  21000, 180000,  0, 10, 20, 35, 1.0, 1.5, false); // LA->Sha ~6000nm
    add_cargo(0, 9,  17000, 160000,  0, 14, 20, 38, 1.0, 1.0, false); // Rot->Santos 5800nm
    add_cargo(3, 10, 23000, 140000,  5, 15, 20, 35, 1.5, 1.0, false); // Sing->Durban 4600nm
    add_cargo(4, 8,  20000, 155000, 10, 20, 28, 42, 1.0, 1.0, false); // Hou->Busan ~6000nm

    // Bunker options: at major hubs, every 10 days
    for (int r : {0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13}) {
        for (double d = 0.0; d < inst.planning_horizon_days; d += 10.0) {
            double price_var = 1.0 + 0.02 * std::sin(d * 0.3 + r);
            inst.bunker_options.push_back({r, d,
                inst.regions[r].hfo_price * price_var,
                inst.regions[r].mgo_price * price_var});
        }
    }

    return inst;
}

// ---------------------------------------------------------------------------
// Large and XLarge instances use scaled versions of medium
// ---------------------------------------------------------------------------
inline Instance make_large() {
    auto base = make_medium();
    base.name = "large-15s-60c";
    base.planning_horizon_days = 90.0;

    // Double the ships
    int orig_ships = (int)base.ships.size();
    for (int i = 0; i < orig_ships; ++i) {
        auto s = base.ships[i];
        s.name = s.name + "-2";
        s.available_day += 5.0;
        s.origin_region = (s.origin_region + 3) % (int)base.regions.size();
        s.fuel_coeff_laden *= (1.0 + 0.05 * (i % 3));
        base.ships.push_back(s);
    }
    // Add one more
    base.ships.push_back({"Vessel-H", 11.0, 14.0, 12.0, 14.0, 0.0037, 0.0029,
                           2300.0, 460.0, 115.0, 23.0, 100.0, 1700.0, 220.0, 10, 4.0});

    // Double cargoes with shifted time windows
    int orig_cargoes = (int)base.cargoes.size();
    for (int i = 0; i < orig_cargoes; ++i) {
        auto c = base.cargoes[i];
        c.pickup_tw_start += 30.0;
        c.pickup_tw_end += 30.0;
        c.delivery_tw_start += 30.0;
        c.delivery_tw_end += 30.0;
        c.revenue *= (0.9 + 0.2 * ((i % 5) / 5.0));
        base.cargoes.push_back(c);
    }

    // More bunker options for extended horizon
    for (int r : {0, 3, 6, 7}) {
        for (double d = 60.0; d < 90.0; d += 10.0) {
            double price_var = 1.0 + 0.02 * std::sin(d * 0.3 + r);
            base.bunker_options.push_back({r, d,
                base.regions[r].hfo_price * price_var,
                base.regions[r].mgo_price * price_var});
        }
    }

    return base;
}

inline Instance make_xlarge() {
    auto base = make_large();
    base.name = "xlarge-30s-120c";
    base.planning_horizon_days = 120.0;

    // Double ships again
    int orig_ships = (int)base.ships.size();
    for (int i = 0; i < orig_ships; ++i) {
        auto s = base.ships[i];
        s.name = s.name + "-3";
        s.available_day += 8.0;
        s.origin_region = (s.origin_region + 5) % (int)base.regions.size();
        base.ships.push_back(s);
    }

    // Double cargoes again
    int orig_cargoes = (int)base.cargoes.size();
    for (int i = 0; i < orig_cargoes; ++i) {
        auto c = base.cargoes[i];
        c.pickup_tw_start += 60.0;
        c.pickup_tw_end += 60.0;
        c.delivery_tw_start += 60.0;
        c.delivery_tw_end += 60.0;
        c.revenue *= (0.85 + 0.3 * ((i % 7) / 7.0));
        base.cargoes.push_back(c);
    }

    return base;
}

// ---------------------------------------------------------------------------
// JSONL loader
// ---------------------------------------------------------------------------
inline Instance load_jsonl(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open instance file: " + path);
    }
    nlohmann::json j;
    f >> j;

    Instance inst;
    inst.name = j["name"];
    inst.planning_horizon_days = j["planning_horizon_days"];
    inst.bonus_price = j["bonus_price"];
    inst.known_optimum = j.value("known_optimum", -1.0);

    for (auto& jr : j["regions"]) {
        inst.regions.push_back({
            jr["name"], jr["is_eca"], jr["hfo_price"], jr["mgo_price"],
            jr["port_cost"], jr["bunker_available"]
        });
    }
    for (auto& jc : j["cargoes"]) {
        inst.cargoes.push_back({
            jc["pickup_region"], jc["delivery_region"],
            jc["quantity"], jc["revenue"],
            jc["pickup_tw_start"], jc["pickup_tw_end"],
            jc["delivery_tw_start"], jc["delivery_tw_end"],
            jc["service_time_load"], jc["service_time_discharge"],
            jc["is_contract"]
        });
    }
    for (auto& js : j["ships"]) {
        inst.ships.push_back({
            js["name"],
            js["v_min_laden"], js["v_max_laden"],
            js["v_min_ballast"], js["v_max_ballast"],
            js["fuel_coeff_laden"], js["fuel_coeff_ballast"],
            js["hfo_tank_max"], js["mgo_tank_max"],
            js["hfo_safety"], js["mgo_safety"],
            js["min_bunkering"],
            js["initial_hfo"], js["initial_mgo"],
            js["origin_region"], js["available_day"]
        });
    }
    for (auto& jl : j["legs"]) {
        inst.legs.push_back({
            jl["from_region"], jl["to_region"],
            jl["distance"], jl["eca_fraction"]
        });
    }
    for (auto& jb : j["bunker_options"]) {
        inst.bunker_options.push_back({
            jb["region"], jb["day"],
            jb["hfo_price"], jb["mgo_price"]
        });
    }

    return inst;
}

}  // namespace bunker_eca
}  // namespace cbls
