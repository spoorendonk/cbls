#pragma once

// CVRP instance data and the TSPLIB `.vrp` reader (#197).
//
// Only what CVRPLIB's X set uses is accepted (benchmarks/instances/cvrp/README.md):
// `TYPE: CVRP`, `EDGE_WEIGHT_TYPE: EUC_2D`, one depot, and that depot at TSPLIB
// node 1 -- the numbering CVRPLIB's `.sol` files assume, where customer id `c`
// is TSPLIB node `c + 1`. Anything else throws rather than being guessed at,
// because the BKS are costs in exactly that metric and numbering.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cbls::cvrp {

/// One instance. Index 0 of the node arrays is the depot; customer `e`
/// (0-based, the List element) is node `e + 1`, TSPLIB node `e + 2`, and
/// CVRPLIB `.sol` id `e + 1`.
struct CvrpInstance {
    std::string name;
    int capacity = 0;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<int> demand;  // demand[0] == 0, the depot

    [[nodiscard]] int customers() const { return static_cast<int>(x.size()) - 1; }
};

/// TSPLIB's EUC_2D distance: nint of the Euclidean distance, rounded per edge.
/// `floor(d + 0.5)` is TSPLIB's `nint`; std::nearbyint would round halves to
/// even, which differs on an exact half.
inline double tsplib_euc_2d(double x1, double y1, double x2, double y2) {
    return std::floor(std::hypot(x1 - x2, y1 - y2) + 0.5);
}

/// The `k` of an X name (`X-n101-k25` -> 25): the minimum vehicle count, not a
/// fleet limit. -1 when the name does not carry one.
inline int name_k(const std::string& name) {
    const auto pos = name.rfind("-k");
    if (pos == std::string::npos || pos + 2 >= name.size()) {
        return -1;
    }
    int k = 0;
    for (std::size_t i = pos + 2; i < name.size(); ++i) {
        const char c = name[i];
        if (c < '0' || c > '9' || k > 100000000) {
            return -1;
        }
        k = (k * 10) + (c - '0');
    }
    return k;
}

namespace detail {

inline std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return {};
    }
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

[[noreturn]] inline void fail(const std::string& what) {
    throw std::runtime_error("vrp: " + what);
}

/// A header line `KEY : value`; false when the line is not one.
inline bool header_field(const std::string& line, std::string& key, std::string& value) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
        return false;
    }
    key = trim(line.substr(0, colon));
    value = trim(line.substr(colon + 1));
    return true;
}

/// One `id a [b]` line of a section, 1-based id in [1, n].
inline int section_id(std::istringstream& ss, int n, const char* section) {
    int64_t id = 0;
    if (!(ss >> id) || id < 1 || id > n) {
        fail(std::string("bad node id in ") + section);
    }
    return static_cast<int>(id - 1);
}

struct Header {
    std::string name;
    int dimension = 0;
    int capacity = 0;
};

inline void check_header(const Header& h, const std::string& type, const std::string& weight) {
    if (type != "CVRP") {
        fail("TYPE is '" + type + "', expected CVRP");
    }
    if (weight != "EUC_2D") {
        fail("EDGE_WEIGHT_TYPE is '" + weight + "', expected EUC_2D");
    }
    if (h.dimension < 2) {
        fail("DIMENSION must be at least 2 (a depot and a customer)");
    }
    if (h.capacity <= 0) {
        fail("CAPACITY must be positive");
    }
}

}  // namespace detail

namespace detail {

/// The reader's state across lines. Three stages, each its own method: header
/// fields until the first section keyword, then one row shape per section, then
/// the completeness checks a file can only fail once it has ended.
class VrpParser {
public:
    void line(const std::string& raw) {
        const std::string t = trim(raw);
        if (t.empty()) {
            return;
        }
        if (t == "NODE_COORD_SECTION" || t == "DEMAND_SECTION" || t == "DEPOT_SECTION") {
            start_section(t);
        } else if (section_.empty()) {
            header_line(t);
        } else {
            section_line(t);
        }
    }

    CvrpInstance finish() {
        if (section_.empty()) {
            fail("no NODE_COORD_SECTION");
        }
        for (std::size_t i = 0; i < have_coord_.size(); ++i) {
            if (have_coord_[i] == 0 || have_demand_[i] == 0) {
                fail("node " + std::to_string(i + 1) + " has no coordinates or no demand");
            }
        }
        if (depots_.size() != 1) {
            fail("expected exactly one depot, found " + std::to_string(depots_.size()));
        }
        if (depots_[0] != 1) {
            fail("the depot must be node 1 (CVRPLIB's .sol numbering assumes it)");
        }
        if (inst_.demand[0] != 0) {
            fail("the depot has a non-zero demand");
        }
        for (std::size_t i = 1; i < inst_.demand.size(); ++i) {
            if (inst_.demand[i] > inst_.capacity) {
                fail("customer " + std::to_string(i) + " alone exceeds the capacity");
            }
        }
        return inst_;
    }

private:
    void header_line(const std::string& t) {
        std::string key;
        std::string value;
        if (!header_field(t, key, value)) {
            fail("unexpected header line '" + t + "'");
        }
        if (key == "NAME") {
            header_.name = value;
        } else if (key == "TYPE") {
            type_ = value;
        } else if (key == "EDGE_WEIGHT_TYPE") {
            weight_ = value;
        } else if (key == "DIMENSION") {
            header_.dimension = std::stoi(value);
        } else if (key == "CAPACITY") {
            header_.capacity = std::stoi(value);
        }
    }

    void start_section(const std::string& t) {
        if (section_.empty()) {
            check_header(header_, type_, weight_);
            inst_.name = header_.name;
            inst_.capacity = header_.capacity;
            const auto n = static_cast<std::size_t>(header_.dimension);
            inst_.x.assign(n, 0.0);
            inst_.y.assign(n, 0.0);
            inst_.demand.assign(n, 0);
            have_coord_.assign(n, 0);
            have_demand_.assign(n, 0);
        }
        section_ = t;
    }

    void section_line(const std::string& t) {
        std::istringstream ss(t);
        if (section_ == "DEPOT_SECTION") {
            int64_t id = 0;
            if (!(ss >> id)) {
                fail("bad DEPOT_SECTION line '" + t + "'");
            }
            if (id != -1) {
                depots_.push_back(id);
            }
            return;
        }
        const auto i =
            static_cast<std::size_t>(section_id(ss, header_.dimension, section_.c_str()));
        if (section_ == "NODE_COORD_SECTION") {
            if (!(ss >> inst_.x[i] >> inst_.y[i])) {
                fail("bad coordinates for node " + std::to_string(i + 1));
            }
            have_coord_[i] = 1;
        } else {
            int d = 0;
            if (!(ss >> d) || d < 0) {
                fail("bad demand for node " + std::to_string(i + 1));
            }
            inst_.demand[i] = d;
            have_demand_[i] = 1;
        }
    }

    Header header_;
    std::string type_;
    std::string weight_;
    std::string section_;
    CvrpInstance inst_;
    std::vector<uint8_t> have_coord_;
    std::vector<uint8_t> have_demand_;
    std::vector<int64_t> depots_;
};

}  // namespace detail

/// Parses a TSPLIB CVRP file from a stream. Throws std::runtime_error naming the
/// first thing it refuses.
inline CvrpInstance read_vrp(std::istream& in) {
    detail::VrpParser parser;
    std::string line;
    while (std::getline(in, line)) {
        if (detail::trim(line) == "EOF") {
            break;
        }
        parser.line(line);
    }
    return parser.finish();
}

inline CvrpInstance read_vrp_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("vrp: cannot open " + path);
    }
    return read_vrp(in);
}

}  // namespace cbls::cvrp
