cmake_minimum_required(VERSION 3.15)
project(cbls VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# FetchContent (shared by library, tests, benchmarks, CLI)
include(FetchContent)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Library
add_library(cbls_lib
    src/dag.cpp
    src/dag_ops.cpp
    src/model.cpp
    src/expr.cpp
    src/violation.cpp
    src/moves.cpp
    src/search.cpp
    src/lns.cpp
    src/pool.cpp
    src/inner_solver.cpp
    src/io.cpp
    src/formatter.cpp
)
target_include_directories(cbls_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(cbls_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME cbls
)
# Link json headers privately (interface lib, so use target_include_directories instead)
target_include_directories(cbls_lib PRIVATE ${json_SOURCE_DIR}/include)
target_link_libraries(cbls_lib PRIVATE pthread)

# CLI executable
add_executable(cbls_cli src/cli.cpp)
target_link_libraries(cbls_cli PRIVATE cbls_lib)
set_target_properties(cbls_cli PROPERTIES OUTPUT_NAME cbls)

# Install
include(GNUInstallDirs)
install(TARGETS cbls_lib EXPORT cblsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS cbls_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/cbls DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT cblsTargets
    FILE cblsTargets.cmake
    NAMESPACE cbls::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cbls
)

configure_file(cmake/cblsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cblsConfig.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cblsConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cbls)

# C++ tests
option(CBLS_BUILD_TESTS "Build C++ tests" ON)
if(CBLS_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    add_subdirectory(tests)
endif()

# Examples
option(CBLS_BUILD_EXAMPLES "Build examples" ON)
if(CBLS_BUILD_EXAMPLES)
    add_executable(cbls_simple examples/simple.cpp)
    target_link_libraries(cbls_simple PRIVATE cbls_lib)

    add_executable(cbls_chped examples/chped.cpp)
    target_link_libraries(cbls_chped PRIVATE cbls_lib)
    target_include_directories(cbls_chped PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Benchmarks
option(CBLS_BUILD_BENCHMARKS "Build benchmarks" ON)
if(CBLS_BUILD_BENCHMARKS)
    add_executable(cbls_uc_chped benchmarks/uc-chped/uc_chped.cpp)
    target_link_libraries(cbls_uc_chped PRIVATE cbls_lib)
    target_include_directories(cbls_uc_chped PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${json_SOURCE_DIR}/include)

    add_executable(cbls_nuclear_outage benchmarks/nuclear-outage/nuclear_outage.cpp)
    target_link_libraries(cbls_nuclear_outage PRIVATE cbls_lib)
    target_include_directories(cbls_nuclear_outage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${json_SOURCE_DIR}/include)

    add_executable(cbls_bunker_eca benchmarks/bunker-eca/bunker_eca.cpp)
    target_link_libraries(cbls_bunker_eca PRIVATE cbls_lib)
    target_include_directories(cbls_bunker_eca PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${json_SOURCE_DIR}/include)
endif()

# Python bindings (optional)
option(CBLS_BUILD_PYTHON "Build Python bindings" OFF)
if(CBLS_BUILD_PYTHON)
    add_subdirectory(python)
endif()
