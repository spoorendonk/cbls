#pragma once

// OR-Library set-covering instances (Beasley, https://people.brunel.ac.uk/
// ~mastjjb/jeb/orlib/scpinfo.html). File layout, whitespace-separated:
//
//     m n
//     c(1) ... c(n)
//     for each row i: k(i) followed by k(i) 1-based column indices
//
// See benchmarks/instances/setcover/ for the vendored roster and download.py.

#include <cstdint>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cbls {
namespace setcover {

struct SetCoverInstance {
    std::string name;
    int rows = 0;
    int cols = 0;
    std::vector<double> cost;               // per column
    std::vector<std::vector<int>> row_cols; // 0-based columns covering each row
    // Row-major rows x cols membership, so a per-row lambda over the chosen
    // columns is an O(1) lookup instead of a search through row_cols.
    std::vector<uint8_t> covers;

    bool covers_row(int row, int col) const {
        return covers[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                      static_cast<size_t>(col)] != 0;
    }
    int nonzeros() const {
        int n = 0;
        for (const auto& r : row_cols) {
            n += static_cast<int>(r.size());
        }
        return n;
    }
};

// Parses the format above. Throws std::runtime_error on anything malformed —
// truncation, an out-of-range column index, or a row no column covers (which
// would make the instance trivially infeasible and is never true of the
// OR-Library files).
inline SetCoverInstance parse_setcover(std::istream& in, const std::string& name = "") {
    SetCoverInstance inst;
    inst.name = name;
    auto take = [&](const char* what) {
        double v = 0.0;
        if (!(in >> v)) {
            throw std::runtime_error("setcover: " + name + ": file ended while reading " + what);
        }
        return v;
    };

    inst.rows = static_cast<int>(take("row count"));
    inst.cols = static_cast<int>(take("column count"));
    if (inst.rows <= 0 || inst.cols <= 0) {
        throw std::runtime_error("setcover: " + name + ": nonsensical dimensions");
    }
    inst.cost.resize(static_cast<size_t>(inst.cols));
    for (int j = 0; j < inst.cols; ++j) {
        inst.cost[static_cast<size_t>(j)] = take("column costs");
    }

    inst.row_cols.resize(static_cast<size_t>(inst.rows));
    inst.covers.assign(static_cast<size_t>(inst.rows) * static_cast<size_t>(inst.cols), 0);
    for (int i = 0; i < inst.rows; ++i) {
        int count = static_cast<int>(take("a row's column count"));
        if (count <= 0) {
            throw std::runtime_error("setcover: " + name + ": row " + std::to_string(i) +
                                     " is covered by no column");
        }
        auto& covering = inst.row_cols[static_cast<size_t>(i)];
        covering.resize(static_cast<size_t>(count));
        for (int t = 0; t < count; ++t) {
            int col = static_cast<int>(take("a row's column list"));
            if (col < 1 || col > inst.cols) {
                throw std::runtime_error("setcover: " + name + ": row " + std::to_string(i) +
                                         " references column " + std::to_string(col) +
                                         " outside 1.." + std::to_string(inst.cols));
            }
            covering[static_cast<size_t>(t)] = col - 1;
            inst.covers[static_cast<size_t>(i) * static_cast<size_t>(inst.cols) +
                        static_cast<size_t>(col - 1)] = 1;
        }
    }
    return inst;
}

inline SetCoverInstance load_setcover(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("setcover: cannot open " + path);
    }
    // Instance name = file stem, so results tables key on `scp41`, not a path.
    size_t slash = path.find_last_of('/');
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) {
        name = name.substr(0, dot);
    }
    return parse_setcover(in, name);
}

}  // namespace setcover
}  // namespace cbls
