#pragma once

#include "search.h"
#include "model.h"
#include <iostream>
#include <string>

namespace cbls {

class HumanFormatter : public SolveCallback {
public:
    explicit HumanFormatter(std::ostream& out = std::cout) : out_(out) {}

    void print_header(const std::string& model_path, const Model& model,
                      uint64_t seed, double time_limit);
    void on_progress(const SolveProgress& p) override;
    void print_result(const SearchResult& result, const Model& model);

private:
    std::ostream& out_;
};

class JsonlFormatter : public SolveCallback {
public:
    explicit JsonlFormatter(std::ostream& out = std::cout) : out_(out) {}

    void print_header(const std::string& model_path, const Model& model,
                      uint64_t seed, double time_limit);
    void on_progress(const SolveProgress& p) override;
    void print_result(const SearchResult& result, const Model& model);

private:
    std::ostream& out_;
};

}  // namespace cbls
