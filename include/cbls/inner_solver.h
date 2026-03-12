#pragma once

#include <string>
#include <unordered_map>

namespace cbls {

class InnerSolverHook {
public:
    virtual ~InnerSolverHook() = default;

    struct Result {
        std::unordered_map<std::string, double> float_values;
        double objective = 0.0;
    };

    virtual Result solve(const std::unordered_map<std::string, double>& fixed_state) = 0;
};

}  // namespace cbls
