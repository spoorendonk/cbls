#pragma once

#include <nlohmann/json.hpp>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace cbls {
namespace nuclear_outage {

// ===========================================================================
// Legacy data structures for synthetic instances (JSONL format)
// ===========================================================================

struct NuclearInstance {
    std::string name;
    int n_units, n_periods, n_scenarios, n_outages, n_sites;

    std::vector<double> capacity;
    std::vector<double> min_power;
    std::vector<double> fuel_cost;
    std::vector<int> site;

    std::vector<int> outage_unit;
    std::vector<int> outage_duration;
    std::vector<int> outage_earliest;
    std::vector<int> outage_latest;

    int min_spacing_same_site;
    std::vector<int> max_outages_per_site;

    std::vector<std::vector<double>> demand;
    double penalty_unserved;
    std::map<std::string, double> known_bounds;
};

inline NuclearInstance load_jsonl(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open instance file: " + path);
    }
    nlohmann::json j;
    f >> j;

    NuclearInstance inst;
    inst.name = j["name"];
    inst.n_units = j["n_units"];
    inst.n_periods = j["n_periods"];
    inst.n_scenarios = j["n_scenarios"];
    inst.n_outages = j["n_outages"];
    inst.n_sites = j["n_sites"];

    inst.capacity = j["capacity"].get<std::vector<double>>();
    inst.min_power = j["min_power"].get<std::vector<double>>();
    inst.fuel_cost = j["fuel_cost"].get<std::vector<double>>();
    inst.site = j["site"].get<std::vector<int>>();

    inst.outage_unit = j["outage_unit"].get<std::vector<int>>();
    inst.outage_duration = j["outage_duration"].get<std::vector<int>>();
    inst.outage_earliest = j["outage_earliest"].get<std::vector<int>>();
    inst.outage_latest = j["outage_latest"].get<std::vector<int>>();

    inst.min_spacing_same_site = j["min_spacing_same_site"];
    inst.max_outages_per_site = j["max_outages_per_site"].get<std::vector<int>>();

    inst.demand.resize(inst.n_scenarios);
    for (int s = 0; s < inst.n_scenarios; ++s) {
        inst.demand[s] = j["demand"][s].get<std::vector<double>>();
    }
    inst.penalty_unserved = j["penalty_unserved"];
    for (auto& [key, val] : j["known_bounds"].items()) {
        inst.known_bounds[key] = val.get<double>();
    }
    return inst;
}

inline NuclearInstance make_mini(const NuclearInstance& inst, int n_scenarios) {
    if (n_scenarios < 1 || n_scenarios > inst.n_scenarios) {
        throw std::invalid_argument("n_scenarios out of range");
    }
    NuclearInstance sub = inst;
    sub.n_scenarios = n_scenarios;
    sub.demand.resize(n_scenarios);
    sub.name = inst.name + "-" + std::to_string(n_scenarios) + "sc";
    sub.known_bounds.clear();
    return sub;
}

// ===========================================================================
// ROADEF 2010 competition data structures
// ===========================================================================

struct DecreaseProfile {
    // Piecewise linear mapping: fuel_level -> power_fraction_of_pmax
    // Points sorted by decreasing fuel_level
    std::vector<std::pair<double, double>> points;

    double evaluate(double fuel_stock) const {
        if (points.empty()) return 1.0;
        if (fuel_stock >= points.front().first) return points.front().second;
        if (fuel_stock <= points.back().first) return points.back().second;
        for (size_t i = 0; i + 1 < points.size(); ++i) {
            if (fuel_stock <= points[i].first && fuel_stock >= points[i + 1].first) {
                double range = points[i].first - points[i + 1].first;
                if (range < 1e-12) return points[i].second;
                double frac = (fuel_stock - points[i + 1].first) / range;
                return points[i + 1].second + frac * (points[i].second - points[i + 1].second);
            }
        }
        return points.back().second;
    }
};

struct Type1Plant {
    std::string name;
    int index;
    // Indexed [scenario][timestep]
    std::vector<std::vector<double>> pmin, pmax, cost;
};

struct Type2Plant {
    std::string name;
    int index;
    double initial_stock;  // XI_i
    int n_cycles;          // K (number of schedulable cycles)

    // Outage durations in weeks: DA[k] for k=0..K-1
    std::vector<int> durations;

    // Per-cycle fuel parameters (K+1 entries: index 0 = current campaign k=-1)
    // mmax[0] = current_campaign_max_modulus, mmax[1..K] = max_modulus
    std::vector<double> mmax;
    // bo[0] = current BO, bo[1..K] = stock_threshold for cycles 0..K-1
    std::vector<double> bo;
    // Decrease profiles: profiles[0] = current, profiles[1..K] = per-cycle
    std::vector<DecreaseProfile> profiles;

    // Per-cycle refueling parameters (K entries: index i = cycle i)
    std::vector<double> rmax, rmin;
    std::vector<double> q;            // refueling coefficient Q_{i,k}
    std::vector<double> amax, smax;   // stock bounds at refueling
    std::vector<double> refuel_cost;  // C_{i,k}

    // Per-timestep max power: PMAX_i^t [T]
    std::vector<double> pmax_t;

    // End-of-horizon fuel price: C_{i,T+1}
    double fuel_price_end;
};

// CT13: time window for outage start (one per schedulable outage)
struct CT13Window {
    int plant_idx;  // Type 2 plant index i
    int cycle;      // cycle index k (0-based)
    int TO;         // earliest decoupling week
    int TA;         // latest decoupling week (-1 if unscheduled)
};

// CT14-CT18: spacing/overlap between outage sets
struct SpacingConstraint {
    int type;   // 14, 15, 16, 17, or 18
    int index;
    std::vector<int> plant_set;  // C_m: Type 2 plant indices
    double spacing;              // Se_m (negative = max overlap for CT14/CT15)
    int period_start = -1;       // ID_m (CT15 only)
    int period_end = -1;         // IF_m (CT15 only)
};

// CT19: resource constraints
struct CT19Resource {
    int index;
    double quantity;  // Q_m
    std::vector<int> plant_set;
    struct Usage {
        int plant_idx;
        std::vector<int> start;     // L_{i,k,m} per cycle
        std::vector<int> duration;  // TU_{i,k,m} per cycle
    };
    std::vector<Usage> usages;
};

// CT20: max overlapping outages per week
struct CT20MaxOverlap {
    int index;
    int week;                     // h_m
    std::vector<int> plant_set;   // C_m
    int max_allowed;              // N_m
};

// CT21: max offline power capacity
struct CT21OfflineCap {
    int index;
    std::vector<int> plant_set;   // C_m
    int time_start, time_end;     // IT_m range (weeks)
    double imax;                  // IMAX_m
};

struct ROADEFInstance {
    std::string name;
    int T;           // number of timesteps
    int H;           // number of weeks
    int K;           // max number of campaigns
    int S;           // number of scenarios
    double epsilon;  // tolerance for power profile
    int n_type1;     // J (number of Type 1 plants)
    int n_type2;     // I (number of Type 2 plants)

    int timesteps_per_week;  // T / H

    std::vector<double> timestep_durations;   // D^t [T]
    std::vector<std::vector<double>> demand;  // [S][T]

    std::vector<Type1Plant> type1_plants;
    std::vector<Type2Plant> type2_plants;

    // Scheduling constraints
    std::vector<CT13Window> ct13;
    std::vector<SpacingConstraint> spacing_constraints;
    std::vector<CT19Resource> ct19;
    std::vector<CT20MaxOverlap> ct20;
    std::vector<CT21OfflineCap> ct21;

    int n_outages() const { return static_cast<int>(ct13.size()); }
};

// ===========================================================================
// ROADEF competition format parser
// ===========================================================================

namespace detail {

// Simple tokenizer: reads a file into lines, provides next_line/expect helpers
class ROADEFParser {
public:
    explicit ROADEFParser(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open ROADEF data file: " + path);
        }
        std::string line;
        while (std::getline(f, line)) {
            // Trim trailing whitespace
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
                line.pop_back();
            lines_.push_back(std::move(line));
        }
    }

    bool has_next() const { return pos_ < lines_.size(); }

    const std::string& peek() const { return lines_[pos_]; }

    std::string next_line() {
        if (pos_ >= lines_.size()) throw std::runtime_error("Unexpected end of file");
        return lines_[pos_++];
    }

    // Read a line, split into tokens by whitespace
    std::vector<std::string> next_tokens() {
        auto line = next_line();
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string tok;
        while (iss >> tok) tokens.push_back(tok);
        return tokens;
    }

    // Expect a line starting with keyword, return remaining tokens
    std::vector<std::string> expect(const std::string& keyword) {
        auto tokens = next_tokens();
        if (tokens.empty() || tokens[0] != keyword) {
            throw std::runtime_error("Expected '" + keyword + "' but got: '" +
                                     (tokens.empty() ? "" : tokens[0]) + "' at line " +
                                     std::to_string(pos_));
        }
        return std::vector<std::string>(tokens.begin() + 1, tokens.end());
    }

    // Read keyword + single int value
    int expect_int(const std::string& keyword) {
        auto toks = expect(keyword);
        if (toks.empty()) throw std::runtime_error("Missing value for " + keyword);
        return std::stoi(toks[0]);
    }

    // Read keyword + single double value
    double expect_double(const std::string& keyword) {
        auto toks = expect(keyword);
        if (toks.empty()) throw std::runtime_error("Missing value for " + keyword);
        return std::stod(toks[0]);
    }

    // Read keyword + multiple doubles on same line
    std::vector<double> expect_doubles(const std::string& keyword) {
        auto toks = expect(keyword);
        std::vector<double> vals;
        for (auto& t : toks) vals.push_back(std::stod(t));
        return vals;
    }

    // Read keyword + multiple ints on same line
    std::vector<int> expect_ints(const std::string& keyword) {
        auto toks = expect(keyword);
        std::vector<int> vals;
        for (auto& t : toks) vals.push_back(std::stoi(t));
        return vals;
    }

    size_t line_num() const { return pos_; }

private:
    std::vector<std::string> lines_;
    size_t pos_ = 0;
};

inline DecreaseProfile parse_profile(ROADEFParser& p) {
    int n_points = p.expect_int("profile_points");
    auto toks = p.expect("decrease_profile");
    DecreaseProfile prof;
    for (int i = 0; i < n_points && 2 * i + 1 < (int)toks.size(); ++i) {
        double fuel = std::stod(toks[2 * i]);
        double power = std::stod(toks[2 * i + 1]);
        prof.points.emplace_back(fuel, power);
    }
    return prof;
}

}  // namespace detail

inline ROADEFInstance load_roadef(const std::string& path) {
    detail::ROADEFParser p(path);
    ROADEFInstance inst;

    // Extract instance name from path
    auto slash = path.rfind('/');
    auto dot = path.rfind('.');
    inst.name = path.substr(slash == std::string::npos ? 0 : slash + 1,
                            dot == std::string::npos ? std::string::npos : dot - (slash == std::string::npos ? 0 : slash + 1));

    // ---- Main section ----
    p.expect("begin");  // "begin main"
    inst.T = p.expect_int("timesteps");
    inst.H = p.expect_int("weeks");
    inst.K = p.expect_int("campaigns");
    inst.S = p.expect_int("scenario");
    inst.epsilon = p.expect_double("epsilon");
    inst.n_type1 = p.expect_int("powerplant1");
    inst.n_type2 = p.expect_int("powerplant2");

    int n_ct13 = p.expect_int("constraint13");
    int n_ct14 = p.expect_int("constraint14");
    int n_ct15 = p.expect_int("constraint15");
    int n_ct16 = p.expect_int("constraint16");
    int n_ct17 = p.expect_int("constraint17");
    int n_ct18 = p.expect_int("constraint18");
    int n_ct19 = p.expect_int("constraint19");
    int n_ct20 = p.expect_int("constraint20");
    int n_ct21 = p.expect_int("constraint21");

    inst.timestep_durations = p.expect_doubles("durations");
    inst.timesteps_per_week = inst.T / inst.H;

    inst.demand.resize(inst.S);
    for (int s = 0; s < inst.S; ++s) {
        inst.demand[s] = p.expect_doubles("demand");
    }
    p.expect("end");  // "end main"

    // ---- Power plants ----
    int total_plants = inst.n_type1 + inst.n_type2;
    for (int pp = 0; pp < total_plants; ++pp) {
        p.expect("begin");  // "begin powerplant"

        auto name_toks = p.expect("name");
        std::string plant_name = name_toks.empty() ? "" : name_toks[0];

        auto type_toks = p.expect("type");
        int plant_type = std::stoi(type_toks[0]);

        if (plant_type == 1) {
            Type1Plant t1;
            t1.name = plant_name;
            t1.index = p.expect_int("index");
            int s_count = p.expect_int("scenario");
            int t_count = p.expect_int("timesteps");
            (void)t_count;

            t1.pmin.resize(s_count);
            t1.pmax.resize(s_count);
            t1.cost.resize(s_count);

            for (int s = 0; s < s_count; ++s) {
                t1.pmin[s] = p.expect_doubles("pmin");
                t1.pmax[s] = p.expect_doubles("pmax");
                t1.cost[s] = p.expect_doubles("cost");
            }
            p.expect("end");  // "end powerplant"
            inst.type1_plants.push_back(std::move(t1));

        } else if (plant_type == 2) {
            Type2Plant t2;
            t2.name = plant_name;
            t2.index = p.expect_int("index");
            t2.initial_stock = p.expect_double("stock");
            t2.n_cycles = p.expect_int("campaigns");

            auto dur_toks = p.expect_ints("durations");
            t2.durations = dur_toks;

            // MMAX: current + per-cycle
            double cur_mmax = p.expect_double("current_campaign_max_modulus");
            auto cycle_mmax = p.expect_doubles("max_modulus");
            t2.mmax.resize(t2.n_cycles + 1);
            t2.mmax[0] = cur_mmax;
            for (int k = 0; k < t2.n_cycles; ++k) t2.mmax[k + 1] = cycle_mmax[k];

            // Refueling params (K entries each)
            auto rmax_v = p.expect_doubles("max_refuel");
            auto rmin_v = p.expect_doubles("min_refuel");
            auto q_v = p.expect_doubles("refuel_ratio");
            t2.rmax = rmax_v;
            t2.rmin = rmin_v;
            t2.q.resize(q_v.size());
            for (size_t i = 0; i < q_v.size(); ++i) t2.q[i] = q_v[i];

            // BO: current + stock_threshold (K+1 values including current)
            double cur_bo = p.expect_double("current_campaign_stock_threshold");
            auto bo_vals = p.expect_doubles("stock_threshold");
            // stock_threshold has K+1 values; first should match cur_bo
            t2.bo.resize(t2.n_cycles + 1);
            if ((int)bo_vals.size() >= t2.n_cycles + 1) {
                // stock_threshold already includes current
                for (int k = 0; k <= t2.n_cycles; ++k) t2.bo[k] = bo_vals[k];
            } else {
                // Fallback: combine current + threshold
                t2.bo[0] = cur_bo;
                for (int k = 0; k < t2.n_cycles && k < (int)bo_vals.size(); ++k)
                    t2.bo[k + 1] = bo_vals[k];
            }

            // Per-timestep pmax
            t2.pmax_t = p.expect_doubles("pmax");

            // Stock bounds at refueling (K entries)
            t2.amax = p.expect_doubles("max_stock_before_refueling");
            t2.smax = p.expect_doubles("max_stock_after_refueling");

            // Refueling cost and fuel price
            t2.refuel_cost = p.expect_doubles("refueling_cost");
            {
                auto fp_toks = p.expect("fuel_price");
                // fuel_price line has: price_of_the_remaining_fuel_at_end value
                // The second token is the actual value
                if (fp_toks.size() >= 2) {
                    t2.fuel_price_end = std::stod(fp_toks[1]);
                } else if (fp_toks.size() == 1) {
                    t2.fuel_price_end = std::stod(fp_toks[0]);
                } else {
                    t2.fuel_price_end = 0.0;
                }
            }

            // Profiles: current campaign profile + K cycle profiles
            t2.profiles.resize(t2.n_cycles + 1);

            // Current campaign profile
            p.expect("begin");  // "begin current_campaign_profile"
            t2.profiles[0] = detail::parse_profile(p);
            p.expect("end");  // "end current_campaign_profile"

            // Per-cycle profiles
            for (int k = 0; k < t2.n_cycles; ++k) {
                p.expect("begin");  // "begin profile"
                p.expect_int("campaign_profile");  // index
                t2.profiles[k + 1] = detail::parse_profile(p);
                p.expect("end");  // "end profile"
            }

            p.expect("end");  // "end powerplant"
            inst.type2_plants.push_back(std::move(t2));
        }
    }

    // ---- Constraints ----
    int total_constraints = n_ct13 + n_ct14 + n_ct15 + n_ct16 + n_ct17 + n_ct18 +
                            n_ct19 + n_ct20 + n_ct21;

    for (int c = 0; c < total_constraints; ++c) {
        p.expect("begin");  // "begin constraint"
        int ctype = p.expect_int("type");

        if (ctype == 13) {
            CT13Window w;
            w.plant_idx = -1;
            w.cycle = -1;
            w.TO = -1;
            w.TA = -1;

            // Parse fields in any order until "end constraint"
            while (p.has_next()) {
                auto toks = p.next_tokens();
                if (toks.empty()) continue;
                if (toks[0] == "end") break;
                if (toks[0] == "index") { /* constraint index, not needed */ }
                else if (toks[0] == "powerplant") w.plant_idx = std::stoi(toks[1]);
                else if (toks[0] == "campaign") w.cycle = std::stoi(toks[1]);
                else if (toks[0] == "earliest_stop_time") w.TO = std::stoi(toks[1]);
                else if (toks[0] == "latest_stop_time") w.TA = std::stoi(toks[1]);
            }
            inst.ct13.push_back(w);

        } else if (ctype >= 14 && ctype <= 18) {
            SpacingConstraint sc;
            sc.type = ctype;
            sc.index = -1;

            while (p.has_next()) {
                auto toks = p.next_tokens();
                if (toks.empty()) continue;
                if (toks[0] == "end") {
                    if (toks.size() >= 2 && toks[1] != "constraint") {
                        // CT15: "end <number>" = IF_m (period end)
                        try { sc.period_end = std::stoi(toks[1]); }
                        catch (...) { break; }
                    } else {
                        break;  // "end constraint"
                    }
                }
                else if (toks[0] == "index") sc.index = std::stoi(toks[1]);
                else if (toks[0] == "set") {
                    for (size_t i = 1; i < toks.size(); ++i)
                        sc.plant_set.push_back(std::stoi(toks[i]));
                }
                else if (toks[0] == "spacing") sc.spacing = std::stod(toks[1]);
                else if (toks[0] == "start") sc.period_start = std::stoi(toks[1]);
            }
            inst.spacing_constraints.push_back(sc);

        } else if (ctype == 19) {
            CT19Resource res;
            while (p.has_next()) {
                auto toks = p.next_tokens();
                if (toks.empty()) continue;
                if (toks[0] == "end" && (toks.size() < 2 || toks[1] == "constraint")) break;
                if (toks[0] == "index") res.index = std::stoi(toks[1]);
                else if (toks[0] == "quantity") res.quantity = std::stod(toks[1]);
                else if (toks[0] == "set") {
                    for (size_t i = 1; i < toks.size(); ++i)
                        res.plant_set.push_back(std::stoi(toks[i]));
                }
                else if (toks[0] == "begin" && toks.size() >= 2 && toks[1] == "period") {
                    CT19Resource::Usage usage;
                    while (p.has_next()) {
                        auto ptoks = p.next_tokens();
                        if (ptoks.empty()) continue;
                        if (ptoks[0] == "end" && ptoks.size() >= 2 && ptoks[1] == "period") break;
                        if (ptoks[0] == "powerplant") usage.plant_idx = std::stoi(ptoks[1]);
                        else if (ptoks[0] == "start") {
                            for (size_t i = 1; i < ptoks.size(); ++i)
                                usage.start.push_back(std::stoi(ptoks[i]));
                        }
                        else if (ptoks[0] == "duration") {
                            for (size_t i = 1; i < ptoks.size(); ++i)
                                usage.duration.push_back(std::stoi(ptoks[i]));
                        }
                    }
                    res.usages.push_back(std::move(usage));
                }
            }
            inst.ct19.push_back(std::move(res));

        } else if (ctype == 20) {
            CT20MaxOverlap ct;
            while (p.has_next()) {
                auto toks = p.next_tokens();
                if (toks.empty()) continue;
                if (toks[0] == "end") break;
                if (toks[0] == "index") ct.index = std::stoi(toks[1]);
                else if (toks[0] == "week") ct.week = std::stoi(toks[1]);
                else if (toks[0] == "set") {
                    for (size_t i = 1; i < toks.size(); ++i)
                        ct.plant_set.push_back(std::stoi(toks[i]));
                }
                else if (toks[0] == "max") ct.max_allowed = std::stoi(toks[1]);
            }
            inst.ct20.push_back(ct);

        } else if (ctype == 21) {
            CT21OfflineCap ct;
            while (p.has_next()) {
                auto toks = p.next_tokens();
                if (toks.empty()) continue;
                if (toks[0] == "end") break;
                if (toks[0] == "index") ct.index = std::stoi(toks[1]);
                else if (toks[0] == "set") {
                    for (size_t i = 1; i < toks.size(); ++i)
                        ct.plant_set.push_back(std::stoi(toks[i]));
                }
                else if (toks[0] == "startend") {
                    ct.time_start = std::stoi(toks[1]);
                    if (toks.size() >= 3) ct.time_end = std::stoi(toks[2]);
                    else ct.time_end = ct.time_start;
                }
                else if (toks[0] == "max") ct.imax = std::stod(toks[1]);
            }
            inst.ct21.push_back(ct);

        } else {
            // Unknown constraint type — skip to end
            while (p.has_next()) {
                auto toks = p.next_tokens();
                if (!toks.empty() && toks[0] == "end") break;
            }
        }
    }

    // Sort CT13 by (plant_idx, cycle) for consistent outage ordering
    std::sort(inst.ct13.begin(), inst.ct13.end(),
              [](const CT13Window& a, const CT13Window& b) {
                  if (a.plant_idx != b.plant_idx) return a.plant_idx < b.plant_idx;
                  return a.cycle < b.cycle;
              });

    return inst;
}

}  // namespace nuclear_outage
}  // namespace cbls
