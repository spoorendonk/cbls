# Not REQUIRED: the ```build fence in CLAUDE.md turns CBLS_BUILD_PYTHON on for
# every gated build, so REQUIRED here would hard-fail `cmake -B build` on any
# machine without the Python development headers -- including someone who only
# wants the C++ library. Degrading is safe because the skip is no longer silent:
# tests/python/conftest.py announces the dropped binding tests, and the gate sets
# CBLS_REQUIRE_BINDINGS=1 so a build that quietly produced no module fails there.
find_package(Python 3.11 COMPONENTS Interpreter Development.Module)

if(NOT Python_FOUND)
    message(WARNING "Python 3.11+ with development headers not found, skipping Python bindings.")
    return()
endif()

# Try to find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
    RESULT_VARIABLE NB_DIR_RESULT
)

if(NB_DIR_RESULT EQUAL 0 AND NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_cbls_core bindings.cpp)
    target_link_libraries(_cbls_core PRIVATE cbls_lib)

    # nanobind ships its headers to consumers as a plain -I, so every diagnostic
    # its macros generate is attributed to the expansion site in bindings.cpp --
    # code we did not write and cannot change. NB_MODULE alone expands to a
    # C-style array, which clang-tidy's modernize-avoid-c-arrays reports at
    # `NB_MODULE(_cbls_core, m)`. clang-tidy is a hard block at push (#121), so
    # re-declaring the same directory SYSTEM is what keeps third-party findings
    # out of our ratchet; CMake de-duplicates and emits it once, as -isystem.
    # Our own code in bindings.cpp is still fully checked. Spelled here rather
    # than via nanobind's NB_SUPPRESS_WARNINGS option because pyproject.toml
    # only requires nanobind>=1.8 and that option is much newer.
    target_include_directories(_cbls_core SYSTEM PRIVATE "${NB_DIR}/include")

    install(TARGETS _cbls_core LIBRARY DESTINATION cbls)
else()
    message(WARNING "nanobind not found, skipping Python bindings. Install with: pip install nanobind")
endif()
