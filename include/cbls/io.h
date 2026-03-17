#pragma once

#include "model.h"
#include <string>
#include <iostream>

namespace cbls {

Model load_model(const std::string& path);
Model load_model(std::istream& input);

void save_model(const Model& model, const std::string& path);
void save_model(const Model& model, std::ostream& out);

}  // namespace cbls
