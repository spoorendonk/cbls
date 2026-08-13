add_executable(cbls_tests
    test_dag.cpp
    test_expr.cpp
    test_moves.cpp
    test_search.cpp
    test_initialization.cpp
    test_violation_delta.cpp
    test_nonfinite_guard.cpp
    test_feasibility_jump.cpp
    test_perturb.cpp
    test_unbounded_domain.cpp
    test_chped.cpp
    test_uc_chped.cpp
    test_inner_solver.cpp
    test_callback.cpp
    test_io.cpp
    test_nuclear_outage.cpp
    test_bunker_eca.cpp
    test_pharma_glsp.cpp
    test_io_mps.cpp
    test_mipfeas.cpp
    test_minlplib.cpp
)
target_link_libraries(cbls_tests PRIVATE cbls_lib Catch2::Catch2WithMain)
target_include_directories(cbls_tests PRIVATE ${CMAKE_SOURCE_DIR} ${json_SOURCE_DIR}/include)

include(Catch)
# Everything except the [timing] class, which is registered separately below so
# it can carry ctest properties (catch_discover_tests registers at build time,
# so set_tests_properties cannot reach the tests it creates).
catch_discover_tests(cbls_tests
    TEST_SPEC "~[timing]"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES SKIP_RETURN_CODE 4)

# Issue #104: [timing] is the suite's ONLY wall-clock-duration assertion —
# everything else asserts on iteration counts or values. Registered on its own so
# it is visible in `ctest -L timing`, carries an explicit TIMEOUT, and is easy to
# quarantine if it ever starts flaking. Don't add tests to this class without a
# reason as concrete as #105's.
add_test(NAME timing_structural_batch_deadline
    COMMAND cbls_tests "[timing]"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(timing_structural_batch_deadline
    PROPERTIES LABELS "timing" TIMEOUT 60 SKIP_RETURN_CODE 4)
