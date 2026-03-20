#pragma once

#include "model.h"
#include <vector>
#include <string>
#include <cstdio>

namespace cbls {

struct VerifyError {
    enum class Kind { VarBounds, VarIntegrality, ConstraintViolation, ObjectiveMismatch, DagConsistency, Custom };
    Kind kind;
    std::string entity;
    double expected = 0.0;
    double actual = 0.0;
    std::string message;
};

struct VerifyResult {
    bool ok = true;
    std::vector<VerifyError> errors;

    void add_error(VerifyError e) {
        ok = false;
        errors.push_back(std::move(e));
    }

    void merge(const VerifyResult& other) {
        for (auto& e : other.errors) {
            add_error(e);
        }
    }

    void print_diagnostics(FILE* out = stderr) const {
        if (ok) {
            fprintf(out, "verify: PASS\n");
            return;
        }
        fprintf(out, "verify: FAIL (%d errors)\n", (int)errors.size());
        for (auto& e : errors) {
            const char* kind_str = "?";
            switch (e.kind) {
                case VerifyError::Kind::VarBounds:            kind_str = "VarBounds"; break;
                case VerifyError::Kind::VarIntegrality:       kind_str = "VarIntegrality"; break;
                case VerifyError::Kind::ConstraintViolation:  kind_str = "ConstraintViolation"; break;
                case VerifyError::Kind::ObjectiveMismatch:    kind_str = "ObjectiveMismatch"; break;
                case VerifyError::Kind::DagConsistency:       kind_str = "DagConsistency"; break;
                case VerifyError::Kind::Custom:               kind_str = "Custom"; break;
            }
            fprintf(out, "  [%s] %s: expected=%.6f actual=%.6f  %s\n",
                    kind_str, e.entity.c_str(), e.expected, e.actual, e.message.c_str());
        }
    }

    operator bool() const { return ok; }
};

VerifyResult verify_model(const Model& model, double tol = 1e-6);

}  // namespace cbls
