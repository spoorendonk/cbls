// Vendored from spoorendonk/mipx: src/io/solu_reader.cpp
// Source: https://github.com/spoorendonk/mipx
// When mipx upstream improves the reader, port the diff here and bump
// the "last synced commit" below.
// Last synced: 7be105f9276ddfd809c2528034c3724c6389eb84
//
// Local adaptations:
//   * Output type is `cbls::SoluEntry` (see io_mps.h) which adds an
//     `is_optimal` flag to distinguish `=opt=` lines from `=best=` /
//     `=feas=` / unknown tags. mipx's reader only stored `=opt=` and
//     `=inf=` and silently dropped the rest; we keep the same shape but
//     the extra flag helps consumers reason about uncertainty.

#include "cbls/io_mps.h"

#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cbls {

std::vector<SoluEntry> read_solu(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<SoluEntry> entries;
    std::string line;

    while (std::getline(in, line)) {
        // Skip empty lines and comments.
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string tag, name;
        iss >> tag >> name;
        if (tag.empty() || name.empty()) {
            continue;
        }

        SoluEntry entry;
        entry.name = std::move(name);

        if (tag == "=inf=") {
            entry.is_infeasible = true;
            entries.push_back(std::move(entry));
        } else if (tag == "=opt=" || tag == "=best=" || tag == "=feas=") {
            std::string val_str;
            iss >> val_str;
            if (!val_str.empty()) {
                double val = 0.0;
                auto [ptr, ec] =
                    std::from_chars(val_str.data(), val_str.data() + val_str.size(), val);
                if (ec == std::errc{}) {
                    entry.value = val;
                }
            }
            entry.is_optimal = (tag == "=opt=");
            entries.push_back(std::move(entry));
        }
        // Skip lines that don't match the expected format.
    }

    return entries;
}

}  // namespace cbls
