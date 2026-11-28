# A downstream project (#447), configured by tests/consumer/run.cmake in one of
# two modes: find_package against `cmake --install` output, or add_subdirectory
# of the source tree. Both link the one name, cbls::cbls.
cmake_minimum_required(VERSION 3.15)
project(cbls_consumer LANGUAGES CXX)

# Below cbls's C++17 on purpose: cbls::cbls must raise it (cxx_std_17).
set(CMAKE_CXX_STANDARD 11)
# For clang-tidy over main.cpp (CLAUDE.md, Git Hooks).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CBLS_CONSUMER_MODE "" CACHE STRING "find_package or add_subdirectory")
if(CBLS_CONSUMER_MODE STREQUAL "find_package")
    find_package(cbls 0.1 CONFIG REQUIRED)
elseif(CBLS_CONSUMER_MODE STREQUAL "add_subdirectory")
    set(CBLS_SOURCE_DIR "" CACHE PATH "The cbls source tree")
    # A parent with nlohmann/json of its own: cbls needs only json's headers and
    # must not define a second nlohmann_json target.
    add_library(nlohmann_json INTERFACE)
    add_subdirectory("${CBLS_SOURCE_DIR}" cbls)
    # A subproject builds the library only: no Catch2, tests, examples,
    # benchmark runners or install rules, and the parent's build type left alone.
    foreach(_t Catch2 cbls_tests cbls_simple cbls_mipfeas)
        if(TARGET ${_t})
            message(FATAL_ERROR "cbls as a subproject defined target ${_t}")
        endif()
    endforeach()
    if(CBLS_INSTALL)
        message(FATAL_ERROR "cbls as a subproject turned CBLS_INSTALL on")
    endif()
    if(CMAKE_BUILD_TYPE)
        message(FATAL_ERROR
            "cbls as a subproject set CMAKE_BUILD_TYPE to ${CMAKE_BUILD_TYPE}")
    endif()
else()
    message(FATAL_ERROR "CBLS_CONSUMER_MODE: '${CBLS_CONSUMER_MODE}'")
endif()

add_executable(cbls_consumer main.cpp)
target_link_libraries(cbls_consumer PRIVATE cbls::cbls)
# One path under any generator: the genex stops a multi-config one appending
# the configuration, so run.cmake finds the binary.
set_target_properties(cbls_consumer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
# cbls_enable_warnings() is cbls's own and absent here, so its flags by hand.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cbls_consumer PRIVATE -Wall -Wextra)
endif()
