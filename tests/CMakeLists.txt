add_executable(cbls_tests
    test_dag.cpp
    test_custom_invariant.cpp
    test_structured_trajectory.cpp
    test_list_partition.cpp
    test_pair_lambda.cpp
    test_expr.cpp
    test_moves.cpp
    test_search.cpp
    test_structural_equivalence.cpp
    test_structural_batch.cpp
    test_parallel.cpp
    test_initialization.cpp
    test_violation_delta.cpp
    test_nonfinite_guard.cpp
    test_feasibility_jump.cpp
    test_perturb.cpp
    test_unbounded_domain.cpp
    test_chped.cpp
    test_uc_chped.cpp
    test_uc_chped_trace.cpp
    test_inner_solver.cpp
    test_callback.cpp
    test_io.cpp
    test_io_mps.cpp
    test_bound_propagation.cpp
    test_mipfeas.cpp
    test_bench_flags.cpp
    test_minlplib.cpp
    test_setcover.cpp
    test_model_share.cpp
    test_model_extend.cpp
    test_throughput.cpp
    test_stop.cpp
    test_counters.cpp
    test_tracer.cpp
    test_executor.cpp
)
target_link_libraries(cbls_tests PRIVATE cbls_lib Catch2::Catch2WithMain)
target_include_directories(cbls_tests PRIVATE ${CMAKE_SOURCE_DIR} ${json_SOURCE_DIR}/include)

include(Catch)
# Everything except the [timing] class, which is registered separately below so
# each of its members can carry ctest properties (catch_discover_tests registers
# at build time, so set_tests_properties cannot reach the tests it creates).
#
# [slow] is discovered by a second call rather than excluded, because that is
# the only way to attach a LABELS property to a subset: PROPERTIES applies to
# the whole call, and Catch2 3.5.2 has no tag-to-label mapping. Two calls do not
# collide -- Catch.cmake derives its generated filenames from a hash of the
# arguments. Discovering them (rather than lumping them into one add_test) keeps
# per-test granularity and lets ctest -j run them in parallel; collapsing the
# slow solves into one serial invocation would make the full suite far slower
# than it is.
#
# Measured at Release: the 6 [slow]-tagged tests account for ~46s of the suite's
# aggregate (summed per-test) time; with `-j$(nproc)` the full run is ~25s of
# wall clock. The remaining 544 (542 discovered, plus the
# two hand-registered tests that carry no slow label --
# timing_structural_batch_deadline and hang_guard_iteration_only_portfolio)
# finish in ~12s with `-j`, which is what pre-commit runs via `ctest -LE slow`.
# pre-push still runs everything.
#
# Re-measured 2026-09-26 at 553 tests: only ONE of
# the six still exceeds the
# ~10s tagging threshold (CHPED 40-unit, 17.2s alone / 17.7s under -j12); the
# other five are 2.9-6.6s on either basis. The split still buys ~13s, but most
# of it is that one test. Re-derive before assuming the other five still earn
# the tag -- "measure both ways before tagging" applies to KEEPING a tag as
# much as to adding one.
#
# The roster is therefore 553 ctest tests: 548 discovered here plus the 5
# hand-registered ones below. That is one more than the 552 TEST_CASEs in
# tests/*.cpp, because hang_guard_iteration_only_portfolio is both discovered
# and hand-registered -- see its note below. `ctest -N -L slow` reports 9, not 6 --
# the three #125 throughput floors carry the slow LABEL without the Catch2
# [slow] TAG. Re-derive with `ctest -N`, never by incrementing.
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

# The [timing] class: tests whose subject is the clock. Each is hand-registered
# by its exact Catch2 test-case name rather than discovered, so it can carry a
# TIMEOUT and be quarantined individually (catch_discover_tests registers at
# build time, so set_tests_properties cannot reach the tests it creates). The
# [timing] tag is also what keeps every one of them out of BOTH
# catch_discover_tests calls above — tag them [slow] instead and they would be
# discovered *and* hand-registered, i.e. run twice.
#
# Issue #104: timing_structural_batch_deadline is the suite's ONLY
# wall-clock-duration assertion — everything else asserts on iteration counts or
# values. Don't add another without a reason as concrete as #105's.
add_test(NAME timing_structural_batch_deadline
    COMMAND cbls_tests "structural batch respects the wall-clock deadline"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(timing_structural_batch_deadline
    PROPERTIES LABELS "timing" TIMEOUT 60 SKIP_RETURN_CODE 4)

# A hang regression needs a TIMEOUT to report itself, so this one is
# hand-registered too (catch_discover_tests registers at build time, so
# set_tests_properties cannot reach the tests it creates). It is tagged
# [parallel] rather than [timing] so it stays OUT of the hand-registration
# collision the [timing] note above describes -- and, unlike [timing], it is
# also discovered by the first catch_discover_tests call above, which is
# deliberate: the property being pinned is cheap and belongs in the fast set.
# The duplicate registration costs one extra sub-second run.
add_test(NAME hang_guard_iteration_only_portfolio
    COMMAND cbls_tests "an iteration-only portfolio returns instead of restarting forever"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(hang_guard_iteration_only_portfolio
    PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 4)

# Issue #125: one throughput floor per benchmarked problem class — a floor on
# GLS iterations completed inside a fixed wall-clock budget, not a ceiling on
# duration. See tests/test_throughput.cpp for the measurement provenance behind
# each floor.
#
# LABELS "timing;slow": `timing` keeps the class greppable, `slow` keeps their
# three wall-clock seconds out of the `ctest -LE slow` pre-commit set (the issue
# asks for that explicitly). They are therefore counted by `ctest -L slow` while
# carrying no Catch2 [slow] tag — the tag and the ctest label are different
# things here, which is why the two roster counts differ.
#
# TIMEOUT 60 against a 1s budget: generous on purpose. A cripple or a regression
# should fail on the iteration floor, which names the problem, rather than on a
# timeout, which does not.
add_test(NAME timing_throughput_milp_feasibility
    COMMAND cbls_tests "MILP feasibility throughput floor"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME timing_throughput_nonconvex_minlp
    COMMAND cbls_tests "non-convex MINLP throughput floor"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME timing_throughput_unit_commitment
    COMMAND cbls_tests "unit-commitment throughput floor"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(
    timing_throughput_milp_feasibility
    timing_throughput_nonconvex_minlp
    timing_throughput_unit_commitment
    PROPERTIES LABELS "timing;slow" TIMEOUT 60 SKIP_RETURN_CODE 4)
