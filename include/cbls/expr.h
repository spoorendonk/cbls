#pragma once

#include "model.h"
#include <vector>

namespace cbls {

class Expr {
public:
    Model* model;
    int32_t handle;  // negative = var, non-negative = node

    Expr(Model* m, int32_t h) : model(m), handle(h) {}

    // Query
    bool is_var() const { return handle < 0; }

    // Convenience accessors (only valid when is_var() == true)
    int32_t var_id() const { return -(handle + 1); }
    Variable& var_mut() { return model->var_mut(var_id()); }
    const Variable& var() const { return model->var(var_id()); }

    // Arithmetic operators
    Expr operator+(const Expr& rhs) const;
    Expr operator-(const Expr& rhs) const;
    Expr operator*(const Expr& rhs) const;
    Expr operator/(const Expr& rhs) const;
    Expr operator-() const;

    // Mixed scalar operators (friends)
    friend Expr operator+(double lhs, const Expr& rhs);
    friend Expr operator+(const Expr& lhs, double rhs);
    friend Expr operator*(double lhs, const Expr& rhs);
    friend Expr operator*(const Expr& lhs, double rhs);
    friend Expr operator-(double lhs, const Expr& rhs);
    friend Expr operator-(const Expr& lhs, double rhs);
    friend Expr operator/(double lhs, const Expr& rhs);
    friend Expr operator/(const Expr& lhs, double rhs);

    // Comparison operators (return Expr constraint, NOT bool)
    Expr operator<=(const Expr& rhs) const;
    Expr operator>=(const Expr& rhs) const;
    Expr operator<(const Expr& rhs) const;
    Expr operator>(const Expr& rhs) const;
    Expr eq(const Expr& rhs) const;    // NOT operator== (would break containers)
    Expr neq(const Expr& rhs) const;

    // Scalar comparison (friends)
    friend Expr operator<=(const Expr& lhs, double rhs);
    friend Expr operator<=(double lhs, const Expr& rhs);
    friend Expr operator>=(const Expr& lhs, double rhs);
    friend Expr operator>=(double lhs, const Expr& rhs);
    friend Expr operator<(const Expr& lhs, double rhs);
    friend Expr operator<(double lhs, const Expr& rhs);
    friend Expr operator>(const Expr& lhs, double rhs);
    friend Expr operator>(double lhs, const Expr& rhs);

    // Power
    Expr pow(const Expr& exp) const;
};

// Free math functions
Expr sin(const Expr& x);
Expr cos(const Expr& x);
Expr tan(const Expr& x);
Expr exp(const Expr& x);
Expr log(const Expr& x);
Expr sqrt(const Expr& x);
Expr abs(const Expr& x);
Expr pow(const Expr& base, const Expr& exp);
Expr min(const std::vector<Expr>& args);
Expr max(const std::vector<Expr>& args);
Expr if_then_else(const Expr& cond, const Expr& then_, const Expr& else_);

}  // namespace cbls
