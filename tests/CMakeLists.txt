add_executable(cbls_tests
    test_dag.cpp
    test_expr.cpp
    test_moves.cpp
    test_search.cpp
    test_chped.cpp
    test_inner_solver.cpp
    test_callback.cpp
    test_io.cpp
    test_bunker_eca.cpp
    test_pharma_glsp.cpp
)
target_link_libraries(cbls_tests PRIVATE cbls_lib Catch2::Catch2WithMain)
target_include_directories(cbls_tests PRIVATE ${CMAKE_SOURCE_DIR} ${json_SOURCE_DIR}/include)

include(Catch)
catch_discover_tests(cbls_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
