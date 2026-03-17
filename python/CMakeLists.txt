find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

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

    install(TARGETS _cbls_core LIBRARY DESTINATION cbls)
else()
    message(WARNING "nanobind not found, skipping Python bindings. Install with: pip install nanobind")
endif()
