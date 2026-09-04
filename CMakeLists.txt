cmake_minimum_required(VERSION 3.15)
project(cbls VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Most of this project's test suite is real solver runs, so an unoptimized build
# costs the full ctest cycle roughly 8x. Nothing set a build type, which means the
# documented `cmake -B build` produced -O0 binaries and every push paid for it.
# Only default it: an explicit -DCMAKE_BUILD_TYPE still wins, and multi-config
# generators (which ignore the variable and set CMAKE_CONFIGURATION_TYPES instead)
# are left alone.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type; defaults to Release because the suite is solver runs" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

# ccache, when the machine has it. pre-push runs `rm -rf build`, so without a
# compile cache every push recompiles from scratch; with one the object tree is
# restored from cache instead. Optional by design -- it is a system package that
# cannot be committed, so a machine without it just builds normally.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache: ${CCACHE_PROGRAM}")
endif()

# Measurement builds: sanitizers and profiling symbols. Both default to off, so
# a *fresh* gated build (`cmake -B build ...` per CLAUDE.md, passing neither
# flag) is untouched by them. Both are cache entries, though, so configuring
# `build/` with one once keeps it there through every later flag-less
# `cmake -B build`: pre-push drops it (`rm -rf build`), but pre-commit gates on
# whatever `build/` already is. Each therefore wants its OWN build directory --
# `build-asan/`, `build-profile/` -- and never `build/`, which the git hooks
# rebuild and gate on. `.gitignore` covers `build*/`. Recipes: docs/profiling.md.
#
# Declared here rather than beside the CBLS_BUILD_* options at the bottom of
# this file because add_compile_options()/add_link_options() reach only targets
# created *after* the call, and that has to include Catch2 and the benchmark
# runners -- a sanitizer applied to some translation units and not others
# links, then reports garbage.
set(CBLS_SANITIZE "" CACHE STRING
    "Sanitizers to build with, e.g. address,undefined,float-cast-overflow (empty: off)")
option(CBLS_PROFILE
       "Add debug symbols and frame pointers on top of the build type" OFF)

if(CBLS_SANITIZE OR CBLS_PROFILE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR
            "CBLS_SANITIZE/CBLS_PROFILE pass GCC/Clang flags, and this is "
            "'${CMAKE_CXX_COMPILER_ID}'. Configure without them.")
    endif()
endif()

if(CBLS_SANITIZE)
    # -g and frame pointers are not cosmetic here: without them a sanitizer
    # report is a list of addresses. The build type stays whatever the caller
    # asked for (Release by default) -- ASan works at -O2 and an -O0 sanitizer
    # run of this suite is unusably slow.
    # A CMake list (`-DCBLS_SANITIZE="address;undefined"`) would expand into two
    # arguments, the second of which GCC reads as an input filename and fails on
    # at link. Normalise to the comma form the compiler wants.
    string(REPLACE ";" "," CBLS_SANITIZE_FLAGS "${CBLS_SANITIZE}")
    add_compile_options(-fsanitize=${CBLS_SANITIZE_FLAGS} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${CBLS_SANITIZE_FLAGS})
    message(STATUS "sanitizers: -fsanitize=${CBLS_SANITIZE_FLAGS}")
endif()

if(CBLS_PROFILE)
    # For perf/jemalloc-heap symbolisation on an otherwise optimized binary.
    # Release omits frame pointers, which costs a profiler its call graph.
    add_compile_options(-fno-omit-frame-pointer -g)
    # Multi-config generators leave CMAKE_BUILD_TYPE empty; say so rather than
    # trailing off after "on top of ".
    set(CBLS_PROFILE_ON "${CMAKE_BUILD_TYPE}")
    if(NOT CBLS_PROFILE_ON)
        set(CBLS_PROFILE_ON "the selected configuration")
    endif()
    message(STATUS "profiling: frame pointers + debug symbols on top of "
                   "${CBLS_PROFILE_ON}")
endif()

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

# GIT_SHALLOW is safe on both dependencies below because each is pinned to a tag
# rather than a moving branch. json is header-only and its 4500-commit history is
# never read. Measured: a cold configure drops 32s -> 21s and _deps/ 359MB ->
# 214MB. The remainder is json's working tree, which shallow cloning cannot help.
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
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
    src/bound_propagation.cpp
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
        GIT_SHALLOW TRUE
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
