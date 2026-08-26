cmake_minimum_required(VERSION 3.15)
project(cbls VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CBLS_INSTALL_GIT_HOOKS
       "Point this checkout's core.hooksPath at the tracked .githooks/" ON)

# The git hooks live in .githooks/ under version control, so a fresh clone gets
# them for free -- but git only runs hooks from core.hooksPath, which is
# per-checkout local config and therefore cannot be committed. Configuring the
# project sets it, which is the one step every contributor already performs.
# Only an unset hooksPath, or one already pointing at .githooks, is touched.
# The top-level guard matters: without it, a project that vendors cbls via
# add_subdirectory or FetchContent would get core.hooksPath written into *its*
# repository. (PROJECT_IS_TOP_LEVEL would read better but needs CMake 3.21.)
if(CBLS_INSTALL_GIT_HOOKS
   AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" config --get core.hooksPath
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE CBLS_HOOKS_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(NOT CBLS_HOOKS_PATH)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" config core.hooksPath .githooks
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                RESULT_VARIABLE CBLS_HOOKS_RC)
            # Reporting success unconditionally would hide a failed write (a
            # read-only .git, a locked config) behind a green configure, and
            # every gate would then be silently absent.
            if(CBLS_HOOKS_RC EQUAL 0)
                message(STATUS "git hooks: core.hooksPath -> .githooks")
            else()
                message(WARNING
                    "git hooks: could not set core.hooksPath (git exited "
                    "${CBLS_HOOKS_RC}). The tracked hooks under .githooks/ are "
                    "NOT running. Fix with: git config core.hooksPath .githooks")
            endif()
        elseif(NOT CBLS_HOOKS_PATH STREQUAL ".githooks")
            message(WARNING
                "git hooks: core.hooksPath is '${CBLS_HOOKS_PATH}', not '.githooks'. "
                "The tracked hooks under .githooks/ are NOT running. "
                "Fix with: git config core.hooksPath .githooks")
        endif()
    endif()
endif()

# Compression libraries (required for MPS .gz reader; bzip2 optional).
find_package(ZLIB REQUIRED)
option(CBLS_USE_BZIP2 "Enable bzip2 (.bz2) support in the MPS reader" OFF)
if(CBLS_USE_BZIP2)
    find_package(BZip2 REQUIRED)
endif()

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
    src/randomize.cpp
    src/feasibility_jump.cpp
    src/search.cpp
    src/lns.cpp
    src/pool.cpp
    src/inner_solver.cpp
    src/io.cpp
    src/io/mps_reader.cpp
    src/io/solu_reader.cpp
    src/io/mps_to_model.cpp
    src/io/nl_reader.cpp
    src/io/nl_to_model.cpp
    src/formatter.cpp
    src/verify.cpp
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
target_link_libraries(cbls_lib PRIVATE pthread ZLIB::ZLIB)
if(CBLS_USE_BZIP2)
    target_compile_definitions(cbls_lib PRIVATE CBLS_HAS_BZIP2)
    target_link_libraries(cbls_lib PRIVATE BZip2::BZip2)
endif()

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

    add_executable(cbls_minlplib benchmarks/minlplib/minlplib.cpp)
    target_link_libraries(cbls_minlplib PRIVATE cbls_lib)
    target_include_directories(cbls_minlplib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(cbls_setcover benchmarks/setcover/setcover.cpp)
    target_link_libraries(cbls_setcover PRIVATE cbls_lib)
    target_include_directories(cbls_setcover PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(cbls_mipfeas benchmarks/mipfeas/mipfeas.cpp)
    target_link_libraries(cbls_mipfeas PRIVATE cbls_lib)
    target_include_directories(cbls_mipfeas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${json_SOURCE_DIR}/include)
endif()

# Python bindings (optional)
option(CBLS_BUILD_PYTHON "Build Python bindings" OFF)
if(CBLS_BUILD_PYTHON)
    add_subdirectory(python)
endif()
