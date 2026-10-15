#pragma once

#include "model.h"
#include "search.h"

#include <iostream>
#include <string>

namespace cbls {

class HumanFormatter : public SolveCallback {
public:
    explicit HumanFormatter(std::ostream& out = std::cout) : out_(out) {}

    void print_header(const std::string& model_path, const Model& model, uint64_t seed,
                      double time_limit);
    void on_progress(const SolveProgress& p) override;
    void print_result(const SearchResult& result, const Model& model);

private:
    std::ostream& out_;
    // The model's sense, read at print_header: the engine reports a maximised
    // objective negated (it minimises -f), and this prints f (#225).
    bool maximizing_ = false;
};

class JsonlFormatter : public SolveCallback {
public:
    explicit JsonlFormatter(std::ostream& out = std::cout) : out_(out) {}

    void print_header(const std::string& model_path, const Model& model, uint64_t seed,
                      double time_limit);
    void on_progress(const SolveProgress& p) override;
    void print_result(const SearchResult& result, const Model& model);

private:
    std::ostream& out_;
    // The model's sense, read at print_header: the engine reports a maximised
    // objective negated (it minimises -f), and this prints f (#225).
    bool maximizing_ = false;
};

}  // namespace cbls
