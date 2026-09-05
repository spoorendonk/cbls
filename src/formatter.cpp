#include "cbls/formatter.h"

#include "cbls/cbls.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>

namespace cbls {

using json = nlohmann::json;

// --- HumanFormatter ---

static std::string format_count(int64_t n) {
    if (n >= 1000000) {
        return std::to_string(n / 1000000) + "." + std::to_string((n % 1000000) / 100000) + "M";
    }
    if (n >= 1000) {
        return std::to_string(n / 1000) + "." + std::to_string((n % 1000) / 100) + "k";
    }
    return std::to_string(n);
}

void HumanFormatter::print_header(const std::string& model_path, const Model& model, uint64_t seed,
                                  double time_limit) {
    out_ << "cbls " << kVersion << " — Constraint-Based Local Search\n";
    out_ << "Model: " << model_path << " | " << model.num_vars() << " vars"
         << " | " << model.constraint_ids().size() << " constraints"
         << " | "
         << (model.objective_id() >= 0 ? (model.is_maximizing() ? "maximize obj" : "minimize obj")
                                       : "feasibility")
         << "\n";
    out_ << "Seed: " << seed << " | Time limit: " << std::fixed << std::setprecision(1)
         << time_limit << "s\n\n";
    out_ << std::right << std::setw(8) << "Time" << std::setw(11) << "Iter" << std::setw(16)
         << "Objective" << std::setw(11) << "Violation" << std::setw(13) << "Perturbs"
         << "\n";
}

void HumanFormatter::on_progress(const SolveProgress& p) {
    out_ << std::fixed << std::setprecision(2) << std::setw(7) << p.time_seconds << "s"
         << std::setw(11) << format_count(p.iteration);

    if (p.feasible && p.objective < std::numeric_limits<double>::infinity()) {
        out_ << std::setw(16) << std::setprecision(6) << p.objective;
    } else {
        out_ << std::setw(16) << "-";
    }

    if (p.feasible) {
        out_ << std::setw(11) << "";
    } else {
        out_ << std::setw(11) << std::setprecision(2) << p.total_violation;
    }

    out_ << std::setw(13) << p.perturbations;

    if (p.new_best) {
        out_ << "  *";
    }
    out_ << "\n";
}

void HumanFormatter::print_result(const SearchResult& result, const Model& model) {
    out_ << "\n";
    out_ << "Status:     " << (result.feasible ? "feasible" : "infeasible") << "\n";
    if (model.objective_id() < 0) {
        out_ << "Objective:  -\n";  // pure feasibility model
    } else if (std::isfinite(result.objective)) {
        out_ << "Objective:  " << std::fixed << std::setprecision(6) << result.objective << "\n";
    } else {
        // The objective exists but is not a number here. On a feasible
        // assignment that is worth saying (issue #100): the point is real, the
        // objective at it is not, and printing "inf" would read as a value. On
        // an infeasible run there is no assignment to report one at.
        out_ << "Objective:  " << (result.feasible ? "no finite objective at this assignment" : "-")
             << "\n";
    }
    out_ << "Time:       " << std::fixed << std::setprecision(2) << result.time_seconds << "s ("
         << result.iterations << " iterations, stopped on "
         << termination_reason_name(result.termination) << ")\n";
    out_ << "Solution:\n";
    for (const auto& var : model.variables()) {
        out_ << "  " << (var.name.empty() ? "v" + std::to_string(var.id) : var.name) << " = ";
        if (is_structured(var.type)) {
            out_ << "[";
            for (size_t i = 0; i < var.elements.size(); ++i) {
                if (i > 0) {
                    out_ << ", ";
                }
                out_ << var.elements[i];
            }
            out_ << "]";
        } else {
            out_ << var.value;
        }
        out_ << "\n";
    }
}

// --- JsonlFormatter ---

void JsonlFormatter::print_header(const std::string& model_path, const Model& model, uint64_t seed,
                                  double time_limit) {
    json j;
    j["event"] = "start";
    j["version"] = kVersion;
    j["model"] = model_path;
    j["vars"] = model.num_vars();
    j["constraints"] = model.constraint_ids().size();
    j["has_objective"] = model.objective_id() >= 0;
    j["seed"] = seed;
    j["time_limit"] = time_limit;
    out_ << j.dump() << "\n";
}

void JsonlFormatter::on_progress(const SolveProgress& p) {
    json j;
    j["event"] = "progress";
    j["time"] = std::round(p.time_seconds * 1000.0) / 1000.0;
    j["iteration"] = p.iteration;
    if (p.feasible && p.objective < std::numeric_limits<double>::infinity()) {
        j["objective"] = p.objective;
    } else {
        j["objective"] = nullptr;
    }
    j["violation"] = p.total_violation;
    j["feasible"] = p.feasible;
    j["perturbations"] = p.perturbations;
    j["new_best"] = p.new_best;
    out_ << j.dump() << "\n";
}

void JsonlFormatter::print_result(const SearchResult& result, const Model& model) {
    json j;
    j["event"] = "result";
    j["time"] = std::round(result.time_seconds * 1000.0) / 1000.0;
    j["iterations"] = result.iterations;
    // Qualifies the two above: "time_limit" means the run was cut off by its
    // budget, anything else that it finished inside it. Without this a published
    // wall time cannot be read correctly (#104, epic #87).
    j["termination"] = termination_reason_name(result.termination);
    // Same rule as HumanFormatter above, and for the same two reasons: a pure
    // feasibility model has no objective to report (result.objective carries the
    // internal 0.0 placeholder, which is not a value of the model), and a
    // feasible point whose objective is +inf/NaN has none either (#100).
    if (model.objective_id() >= 0 && std::isfinite(result.objective)) {
        j["objective"] = result.objective;
    } else {
        j["objective"] = nullptr;
    }
    j["feasible"] = result.feasible;
    j["status"] = result.feasible ? "feasible" : "infeasible";

    json sol = json::object();
    for (const auto& var : model.variables()) {
        std::string name = var.name.empty() ? "v" + std::to_string(var.id) : var.name;
        if (is_structured(var.type)) {
            sol[name] = var.elements;
        } else {
            sol[name] = var.value;
        }
    }
    j["solution"] = sol;
    out_ << j.dump() << "\n";
}

}  // namespace cbls
