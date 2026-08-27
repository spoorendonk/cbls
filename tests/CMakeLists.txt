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
    test_io_mps.cpp
    test_mipfeas.cpp
    test_minlplib.cpp
    test_setcover.cpp
)
target_link_libraries(cbls_tests PRIVATE cbls_lib Catch2::Catch2WithMain)
target_include_directories(cbls_tests PRIVATE ${CMAKE_SOURCE_DIR} ${json_SOURCE_DIR}/include)

include(Catch)
# Everything except the [timing] class, which is registered separately below so
# it can carry ctest properties (catch_discover_tests registers at build time,
# so set_tests_properties cannot reach the tests it creates).
#
# [slow] is discovered by a second call rather than excluded, because that is
# the only way to attach a LABELS property to a subset: PROPERTIES applies to
# the whole call, and Catch2 3.5.2 has no tag-to-label mapping. Two calls do not
# collide -- Catch.cmake derives its generated filenames from a hash of the
# arguments. Discovering them (rather than lumping them into one add_test) keeps
# per-test granularity and lets ctest -j run them in parallel; collapsing 22
# multi-minute solves into one serial invocation would make the full suite far
# slower than it is.
#
# Measured: 22 of 268 tests take >10s and account for ~1470s of the suite's
# aggregate (summed per-test) time; with `-j$(nproc)` the full run is ~304s of
# wall clock. The remaining 246 finish in ~7s with `-j`, which is what
# pre-commit runs via `ctest -LE slow`. pre-push still runs everything.
#
# SKIP_RETURN_CODE 4 maps Catch2's SKIP() onto ctest's "Skipped" rather than a
# failure. No test uses SKIP() today -- the data-gated ROADEF tests that needed
# it went with the nuclear-outage benchmark -- but it stays so the next
# instance-gated test reports honestly instead of a confusing red at push.
catch_discover_tests(cbls_tests
    TEST_SPEC "~[timing]~[slow]"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES SKIP_RETURN_CODE 4)

catch_discover_tests(cbls_tests
    TEST_SPEC "[slow]~[timing]"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES SKIP_RETURN_CODE 4 LABELS "slow")

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
