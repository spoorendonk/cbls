#include "cbls/formatter.h"
#include <nlohmann/json.hpp>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>

namespace cbls {

using json = nlohmann::json;

// --- HumanFormatter ---

static std::string format_count(int64_t n) {
    if (n >= 1000000) {
        return std::to_string(n / 1000) + "." + std::to_string((n % 1000) / 100) + "k";
    }
    if (n >= 1000) {
        return std::to_string(n / 1000) + "." + std::to_string((n % 1000) / 100) + "k";
    }
    return std::to_string(n);
}

void HumanFormatter::print_header(const std::string& model_path, const Model& model,
                                   uint64_t seed, double time_limit) {
    out_ << "cbls 0.1.0 — Constraint-Based Local Search\n";
    out_ << "Model: " << model_path
         << " | " << model.num_vars() << " vars"
         << " | " << model.constraint_ids().size() << " constraints"
         << " | " << (model.objective_id() >= 0 ? "minimize obj" : "feasibility")
         << "\n";
    out_ << "Seed: " << seed << " | Time limit: "
         << std::fixed << std::setprecision(1) << time_limit << "s\n\n";
    out_ << std::right
         << std::setw(8) << "Time"
         << std::setw(11) << "Iter"
         << std::setw(16) << "Objective"
         << std::setw(11) << "Violation"
         << std::setw(13) << "Temperature"
         << "\n";
    header_printed_ = true;
}

void HumanFormatter::on_progress(const SolveProgress& p) {
    out_ << std::fixed << std::setprecision(2)
         << std::setw(7) << p.time_seconds << "s"
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

    out_ << std::setw(13) << std::setprecision(3) << p.temperature;

    if (p.new_best) out_ << "  *";
    out_ << "\n";
}

void HumanFormatter::print_result(const SearchResult& result, const Model& model) {
    out_ << "\n";
    out_ << "Status:     " << (result.feasible ? "feasible" : "infeasible") << "\n";
    if (result.feasible && result.objective < std::numeric_limits<double>::infinity()) {
        out_ << "Objective:  " << std::fixed << std::setprecision(6)
             << result.objective << "\n";
    } else if (model.objective_id() < 0) {
        out_ << "Objective:  -\n";
    } else {
        out_ << "Objective:  " << std::fixed << std::setprecision(6)
             << result.objective << "\n";
    }
    out_ << "Time:       " << std::fixed << std::setprecision(2)
         << result.time_seconds << "s ("
         << result.iterations << " iterations)\n";
    out_ << "Solution:\n";
    for (const auto& var : model.variables()) {
        out_ << "  " << (var.name.empty() ? "v" + std::to_string(var.id) : var.name)
             << " = ";
        if (var.type == VarType::List || var.type == VarType::Set) {
            out_ << "[";
            for (size_t i = 0; i < var.elements.size(); ++i) {
                if (i > 0) out_ << ", ";
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

void JsonlFormatter::print_header(const std::string& model_path, const Model& model,
                                   uint64_t seed, double time_limit) {
    json j;
    j["event"] = "start";
    j["version"] = "0.1.0";
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
    j["temperature"] = p.temperature;
    j["new_best"] = p.new_best;
    out_ << j.dump() << "\n";
}

void JsonlFormatter::print_result(const SearchResult& result, const Model& model) {
    json j;
    j["event"] = "result";
    j["time"] = std::round(result.time_seconds * 1000.0) / 1000.0;
    j["iterations"] = result.iterations;
    if (result.feasible && result.objective < std::numeric_limits<double>::infinity()) {
        j["objective"] = result.objective;
    } else {
        j["objective"] = nullptr;
    }
    j["feasible"] = result.feasible;
    j["status"] = result.feasible ? "feasible" : "infeasible";

    json sol = json::object();
    for (const auto& var : model.variables()) {
        std::string name = var.name.empty() ? "v" + std::to_string(var.id) : var.name;
        if (var.type == VarType::List || var.type == VarType::Set) {
            sol[name] = var.elements;
        } else {
            sol[name] = var.value;
        }
    }
    j["solution"] = sol;
    out_ << j.dump() << "\n";
}

}  // namespace cbls
