#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <sstream>

using namespace cbls;

TEST_CASE("load_model with valid JSONL", "[io]") {
    std::string input =
        R"({"var":"x","type":"Float","lb":0,"ub":10})"  "\n"
        R"({"var":"y","type":"Float","lb":0,"ub":10})"  "\n"
        R"({"node":"sum_xy","op":"Sum","children":["x","y"]})"  "\n"
        R"({"node":"c5","op":"Const","value":5.0})"  "\n"
        R"({"node":"geq1","op":"Geq","children":["sum_xy","c5"]})"  "\n"
        R"({"constraint":"geq1"})"  "\n"
        R"({"minimize":"sum_xy"})"  "\n";

    std::istringstream ss(input);
    Model m = load_model(ss);

    REQUIRE(m.num_vars() == 2);
    REQUIRE(m.constraint_ids().size() == 1);
    REQUIRE(m.objective_id() >= 0);
    REQUIRE(m.is_closed());
}

TEST_CASE("load_model with Bool and Int vars", "[io]") {
    std::string input =
        R"({"var":"b","type":"Bool"})"  "\n"
        R"({"var":"i","type":"Int","lb":-5,"ub":5})"  "\n"
        R"({"node":"sum","op":"Sum","children":["b","i"]})"  "\n"
        R"({"minimize":"sum"})"  "\n";

    std::istringstream ss(input);
    Model m = load_model(ss);

    REQUIRE(m.num_vars() == 2);
    REQUIRE(m.var(0).type == VarType::Bool);
    REQUIRE(m.var(1).type == VarType::Int);
    REQUIRE(m.var(1).lb == -5);
    REQUIRE(m.var(1).ub == 5);
}

TEST_CASE("load_model error: unknown op", "[io]") {
    std::string input =
        R"({"var":"x","type":"Float","lb":0,"ub":1})"  "\n"
        R"({"node":"bad","op":"Foo","children":["x"]})"  "\n";

    std::istringstream ss(input);
    REQUIRE_THROWS_AS(load_model(ss), std::invalid_argument);
}

TEST_CASE("load_model error: missing reference", "[io]") {
    std::string input =
        R"({"node":"bad","op":"Sum","children":["nonexistent"]})"  "\n";

    std::istringstream ss(input);
    REQUIRE_THROWS_AS(load_model(ss), std::invalid_argument);
}

TEST_CASE("load_model error: bad JSON", "[io]") {
    std::string input = "not valid json\n";
    std::istringstream ss(input);
    REQUIRE_THROWS_AS(load_model(ss), std::invalid_argument);
}

TEST_CASE("save_model writes valid JSONL", "[io]") {
    Model m;
    m.float_var(0, 10, "x");
    m.float_var(0, 10, "y");
    auto x_handle = -(0 + 1);  // var 0
    auto y_handle = -(1 + 1);  // var 1
    auto sum_id = m.sum({x_handle, y_handle});
    m.minimize(sum_id);
    m.close();

    std::ostringstream out;
    save_model(m, out);

    std::string output = out.str();
    REQUIRE(!output.empty());

    // Verify it can be re-loaded
    std::istringstream ss(output);
    Model m2 = load_model(ss);
    REQUIRE(m2.num_vars() == 2);
    REQUIRE(m2.objective_id() >= 0);
}

TEST_CASE("round-trip save(load(file))", "[io]") {
    std::string input =
        R"({"var":"x","type":"Float","lb":0,"ub":10})"  "\n"
        R"({"var":"y","type":"Float","lb":0,"ub":10})"  "\n"
        R"({"node":"x_sq","op":"Prod","children":["x","x"]})"  "\n"
        R"({"node":"y_sq","op":"Prod","children":["y","y"]})"  "\n"
        R"({"node":"obj","op":"Sum","children":["x_sq","y_sq"]})"  "\n"
        R"({"node":"xy_sum","op":"Sum","children":["x","y"]})"  "\n"
        R"({"node":"c5","op":"Const","value":5.0})"  "\n"
        R"({"node":"geq1","op":"Geq","children":["xy_sum","c5"]})"  "\n"
        R"({"constraint":"geq1"})"  "\n"
        R"({"minimize":"obj"})"  "\n";

    std::istringstream ss1(input);
    Model m1 = load_model(ss1);

    std::ostringstream out1;
    save_model(m1, out1);

    // Re-load and re-save
    std::istringstream ss2(out1.str());
    Model m2 = load_model(ss2);

    REQUIRE(m2.num_vars() == m1.num_vars());
    REQUIRE(m2.num_nodes() == m1.num_nodes());
    REQUIRE(m2.constraint_ids().size() == m1.constraint_ids().size());
    REQUIRE((m2.objective_id() >= 0) == (m1.objective_id() >= 0));
}

TEST_CASE("idempotent save(load(save(load(file))))", "[io]") {
    std::string input =
        R"({"var":"a","type":"Int","lb":1,"ub":100})"  "\n"
        R"({"var":"b","type":"Float","lb":-5,"ub":5})"  "\n"
        R"({"node":"s","op":"Sum","children":["a","b"]})"  "\n"
        R"({"node":"c0","op":"Const","value":0.0})"  "\n"
        R"({"node":"geq","op":"Geq","children":["s","c0"]})"  "\n"
        R"({"constraint":"geq"})"  "\n"
        R"({"minimize":"s"})"  "\n";

    std::istringstream ss1(input);
    Model m1 = load_model(ss1);
    std::ostringstream out1;
    save_model(m1, out1);
    std::string saved1 = out1.str();

    std::istringstream ss2(saved1);
    Model m2 = load_model(ss2);
    std::ostringstream out2;
    save_model(m2, out2);
    std::string saved2 = out2.str();

    REQUIRE(saved1 == saved2);
}

TEST_CASE("load_model from file", "[io]") {
    // Test with the example file (tests run from build dir)
    Model m = load_model("examples/simple.cbls");
    REQUIRE(m.num_vars() == 2);
    REQUIRE(m.constraint_ids().size() == 1);
    REQUIRE(m.objective_id() >= 0);
}

TEST_CASE("load_model skips empty lines", "[io]") {
    std::string input =
        "\n"
        R"({"var":"x","type":"Float","lb":0,"ub":1})"  "\n"
        "\n"
        R"({"node":"c0","op":"Const","value":0.0})"  "\n"
        R"({"node":"geq","op":"Geq","children":["x","c0"]})"  "\n"
        R"({"constraint":"geq"})"  "\n";

    std::istringstream ss(input);
    Model m = load_model(ss);
    REQUIRE(m.num_vars() == 1);
}

TEST_CASE("round-trip model with lambda_sum", "[io]") {
    Model m;
    auto lv = m.list_var(5, "perm");
    auto ls = m.lambda_sum(lv, [](int e) { return static_cast<double>(e * e); });
    m.minimize(ls);
    m.close();

    // Save
    std::ostringstream out;
    save_model(m, out);
    std::string saved = out.str();

    // Verify table appears in output
    REQUIRE(saved.find("\"table\"") != std::string::npos);

    // Reload and verify behavior
    std::istringstream ss(saved);
    Model m2 = load_model(ss);
    REQUIRE(m2.num_vars() == 1);
    REQUIRE(m2.objective_id() >= 0);

    // Set same elements and evaluate
    auto& v2 = m2.var_mut(0);
    v2.elements = {0, 1, 2, 3, 4};
    full_evaluate(m2);
    // 0 + 1 + 4 + 9 + 16 = 30
    REQUIRE(m2.node(m2.objective_id()).value == 30.0);
}

TEST_CASE("idempotent lambda_sum round-trip", "[io]") {
    Model m;
    auto lv = m.list_var(5, "perm");
    auto ls = m.lambda_sum(lv, [](int e) { return e * 1.1; });
    m.minimize(ls);
    m.close();

    std::ostringstream out1;
    save_model(m, out1);
    std::string saved1 = out1.str();

    std::istringstream ss1(saved1);
    Model m2 = load_model(ss1);
    std::ostringstream out2;
    save_model(m2, out2);
    std::string saved2 = out2.str();

    REQUIRE(saved1 == saved2);
}

TEST_CASE("round-trip maximize model", "[io]") {
    Model m;
    auto x = m.float_var(0, 10, "x");
    auto y = m.float_var(0, 10, "y");
    auto obj = m.sum({x, y});
    m.maximize(obj);
    m.close();

    REQUIRE(m.is_maximizing());

    std::ostringstream out;
    save_model(m, out);
    std::string saved = out.str();

    // Must contain "maximize", not "minimize"
    REQUIRE(saved.find("\"maximize\"") != std::string::npos);
    REQUIRE(saved.find("\"minimize\"") == std::string::npos);

    // Reload
    std::istringstream ss(saved);
    Model m2 = load_model(ss);
    REQUIRE(m2.num_vars() == 2);
    REQUIRE(m2.objective_id() >= 0);
    REQUIRE(m2.is_maximizing());
}

TEST_CASE("load_model all comparison ops", "[io]") {
    std::string input =
        R"({"var":"a","type":"Float","lb":0,"ub":10})"  "\n"
        R"({"var":"b","type":"Float","lb":0,"ub":10})"  "\n"
        R"({"node":"leq","op":"Leq","children":["a","b"]})"  "\n"
        R"({"node":"eq","op":"Eq","children":["a","b"]})"  "\n"
        R"({"node":"geq","op":"Geq","children":["a","b"]})"  "\n"
        R"({"node":"neq","op":"Neq","children":["a","b"]})"  "\n"
        R"({"node":"lt","op":"Lt","children":["a","b"]})"  "\n"
        R"({"node":"gt","op":"Gt","children":["a","b"]})"  "\n"
        R"({"constraint":"leq"})"  "\n";

    std::istringstream ss(input);
    Model m = load_model(ss);
    REQUIRE(m.num_vars() == 2);
    REQUIRE(m.num_nodes() == 6);
}
