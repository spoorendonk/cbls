#include "cbls/expr.h"

namespace cbls {

// Arithmetic operators
Expr Expr::operator+(const Expr& rhs) const {
    return {model, model->sum({handle, rhs.handle})};
}

Expr Expr::operator-(const Expr& rhs) const {
    return {model, model->sum({handle, model->neg(rhs.handle)})};
}

Expr Expr::operator*(const Expr& rhs) const {
    return {model, model->prod(handle, rhs.handle)};
}

Expr Expr::operator/(const Expr& rhs) const {
    return {model, model->div_expr(handle, rhs.handle)};
}

Expr Expr::operator-() const {
    return {model, model->neg(handle)};
}

// Mixed scalar operators
Expr operator+(double lhs, const Expr& rhs) {
    return {rhs.model, rhs.model->sum({rhs.model->constant(lhs), rhs.handle})};
}

Expr operator+(const Expr& lhs, double rhs) {
    return {lhs.model, lhs.model->sum({lhs.handle, lhs.model->constant(rhs)})};
}

Expr operator*(double lhs, const Expr& rhs) {
    return {rhs.model, rhs.model->prod(rhs.model->constant(lhs), rhs.handle)};
}

Expr operator*(const Expr& lhs, double rhs) {
    return {lhs.model, lhs.model->prod(lhs.handle, lhs.model->constant(rhs))};
}

Expr operator-(double lhs, const Expr& rhs) {
    return {rhs.model, rhs.model->sum({rhs.model->constant(lhs), rhs.model->neg(rhs.handle)})};
}

Expr operator-(const Expr& lhs, double rhs) {
    return {lhs.model, lhs.model->sum({lhs.handle, lhs.model->constant(-rhs)})};
}

Expr operator/(double lhs, const Expr& rhs) {
    return {rhs.model, rhs.model->div_expr(rhs.model->constant(lhs), rhs.handle)};
}

Expr operator/(const Expr& lhs, double rhs) {
    return {lhs.model, lhs.model->div_expr(lhs.handle, lhs.model->constant(rhs))};
}

// Comparison operators
Expr Expr::operator<=(const Expr& rhs) const {
    return {model, model->leq(handle, rhs.handle)};
}

Expr Expr::operator>=(const Expr& rhs) const {
    return {model, model->geq(handle, rhs.handle)};
}

Expr Expr::operator<(const Expr& rhs) const {
    return {model, model->lt(handle, rhs.handle)};
}

Expr Expr::operator>(const Expr& rhs) const {
    return {model, model->gt(handle, rhs.handle)};
}

Expr Expr::eq(const Expr& rhs) const {
    return {model, model->eq_expr(handle, rhs.handle)};
}

Expr Expr::neq(const Expr& rhs) const {
    return {model, model->neq(handle, rhs.handle)};
}

// Scalar comparison operators
Expr operator<=(const Expr& lhs, double rhs) {
    return lhs <= Expr{lhs.model, lhs.model->constant(rhs)};
}
Expr operator<=(double lhs, const Expr& rhs) {
    return Expr{rhs.model, rhs.model->constant(lhs)} <= rhs;
}
Expr operator>=(const Expr& lhs, double rhs) {
    return lhs >= Expr{lhs.model, lhs.model->constant(rhs)};
}
Expr operator>=(double lhs, const Expr& rhs) {
    return Expr{rhs.model, rhs.model->constant(lhs)} >= rhs;
}
Expr operator<(const Expr& lhs, double rhs) {
    return lhs < Expr{lhs.model, lhs.model->constant(rhs)};
}
Expr operator<(double lhs, const Expr& rhs) {
    return Expr{rhs.model, rhs.model->constant(lhs)} < rhs;
}
Expr operator>(const Expr& lhs, double rhs) {
    return lhs > Expr{lhs.model, lhs.model->constant(rhs)};
}
Expr operator>(double lhs, const Expr& rhs) {
    return Expr{rhs.model, rhs.model->constant(lhs)} > rhs;
}

// Power
Expr Expr::pow(const Expr& exp) const {
    return {model, model->pow_expr(handle, exp.handle)};
}

// Free math functions
Expr sin(const Expr& x) {
    return {x.model, x.model->sin_expr(x.handle)};
}

Expr cos(const Expr& x) {
    return {x.model, x.model->cos_expr(x.handle)};
}

Expr tan(const Expr& x) {
    return {x.model, x.model->tan_expr(x.handle)};
}

Expr exp(const Expr& x) {
    return {x.model, x.model->exp_expr(x.handle)};
}

Expr log(const Expr& x) {
    return {x.model, x.model->log_expr(x.handle)};
}

Expr sqrt(const Expr& x) {
    return {x.model, x.model->sqrt_expr(x.handle)};
}

Expr abs(const Expr& x) {
    return {x.model, x.model->abs_expr(x.handle)};
}

Expr pow(const Expr& base, const Expr& exp) {
    return {base.model, base.model->pow_expr(base.handle, exp.handle)};
}

Expr min(const std::vector<Expr>& args) {
    std::vector<int32_t> handles;
    handles.reserve(args.size());
    for (const auto& a : args) handles.push_back(a.handle);
    return {args[0].model, args[0].model->min_expr(handles)};
}

Expr max(const std::vector<Expr>& args) {
    std::vector<int32_t> handles;
    handles.reserve(args.size());
    for (const auto& a : args) handles.push_back(a.handle);
    return {args[0].model, args[0].model->max_expr(handles)};
}

Expr if_then_else(const Expr& cond, const Expr& then_, const Expr& else_) {
    return {cond.model, cond.model->if_then_else(cond.handle, then_.handle, else_.handle)};
}

}  // namespace cbls
