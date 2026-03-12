add_executable(cbls_tests
    test_dag.cpp
    test_moves.cpp
    test_search.cpp
    test_chped.cpp
)
target_link_libraries(cbls_tests PRIVATE cbls Catch2::Catch2WithMain)
target_include_directories(cbls_tests PRIVATE ${CMAKE_SOURCE_DIR})

include(Catch)
catch_discover_tests(cbls_tests)
