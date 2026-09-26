#include "cbls/column_generator.h"

#include "cbls/model_extension.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace cbls {

// Key function: emits the vtable here, as SolveCallback's destructor does.
ColumnGenerator::~ColumnGenerator() = default;

namespace {
// Bitwise, so the hash agrees with Equal below: +0.0 and -0.0 are different
// signatures, and so is every NaN payload. That is the conservative direction --
// two signatures that differ only there are admitted as distinct columns rather
// than one being refused as a duplicate.
uint64_t bits_of(double d) {
    uint64_t b = 0;
    std::memcpy(&b, &d, sizeof(b));
    return b;
}

// boost::hash_combine's mixing step, widened to 64 bits.
void mix(size_t& seed, uint64_t v) {
    seed ^= std::hash<uint64_t>{}(v) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}
}  // namespace

ColumnSignatureSet::Signature ColumnSignatureSet::canonical(
    std::vector<std::pair<int32_t, double>> coefficients, double cost) {
    std::sort(coefficients.begin(), coefficients.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    Signature sig;
    sig.cost = cost;
    sig.coefficients.reserve(coefficients.size());
    for (const auto& [row, coef] : coefficients) {
        if (!sig.coefficients.empty() && sig.coefficients.back().first == row) {
            sig.coefficients.back().second += coef;
        } else {
            sig.coefficients.emplace_back(row, coef);
        }
    }
    sig.coefficients.erase(std::remove_if(sig.coefficients.begin(), sig.coefficients.end(),
                                          [](const auto& rc) { return rc.second == 0.0; }),
                           sig.coefficients.end());
    return sig;
}

size_t ColumnSignatureSet::Hash::operator()(const Signature& s) const noexcept {
    size_t seed = s.coefficients.size();
    mix(seed, bits_of(s.cost));
    for (const auto& [row, coef] : s.coefficients) {
        mix(seed, static_cast<uint64_t>(static_cast<uint32_t>(row)));
        mix(seed, bits_of(coef));
    }
    return seed;
}

bool ColumnSignatureSet::Equal::operator()(const Signature& a, const Signature& b) const noexcept {
    if (bits_of(a.cost) != bits_of(b.cost) || a.coefficients.size() != b.coefficients.size()) {
        return false;
    }
    for (size_t i = 0; i < a.coefficients.size(); ++i) {
        if (a.coefficients[i].first != b.coefficients[i].first ||
            bits_of(a.coefficients[i].second) != bits_of(b.coefficients[i].second)) {
            return false;
        }
    }
    return true;
}

bool ColumnSignatureSet::insert(std::vector<std::pair<int32_t, double>> coefficients, double cost) {
    return set_.insert(canonical(std::move(coefficients), cost)).second;
}

bool ColumnSignatureSet::contains(std::vector<std::pair<int32_t, double>> coefficients,
                                  double cost) const {
    return set_.count(canonical(std::move(coefficients), cost)) != 0;
}

ColumnPool::ColumnPool(int64_t cap, int retire_age)
    : cap_(std::max<int64_t>(0, cap)), retire_age_(std::max(0, retire_age)) {}

void ColumnPool::on_added(const ExtensionResult& ext) {
    for (int32_t v = ext.first_new_var; v < ext.end_var(); ++v) {
        entries_.push_back(Entry{v, 0, false});
    }
    added_ += ext.num_new_vars;
}

std::vector<int32_t> ColumnPool::age(const Model& model,
                                     const std::vector<const Model::State*>& keep) {
    std::vector<int32_t> due;
    if (retire_age_ <= 0) {
        return due;
    }
    for (Entry& e : entries_) {
        if (e.retired) {
            continue;
        }
        const Variable& var = model.var(e.var);
        bool at_lb = var.value == var.lb;
        for (const Model::State* s : keep) {
            if (!at_lb) {
                break;
            }
            // A state narrower than the model predates this column; the engine
            // pads every state it keeps, so this is a caller's state that it did
            // not -- refuse rather than read past its end.
            if (static_cast<size_t>(e.var) >= s->values.size()) {
                throw std::invalid_argument(
                    "ColumnPool::age: a kept state is narrower than the model; pad it after "
                    "every extension");
            }
            at_lb = s->values[static_cast<size_t>(e.var)] == var.lb;
        }
        if (!at_lb) {
            e.age = 0;
            continue;
        }
        if (++e.age >= retire_age_) {
            e.retired = true;
            ++retired_;
            due.push_back(e.var);
        }
    }
    return due;
}

}  // namespace cbls
