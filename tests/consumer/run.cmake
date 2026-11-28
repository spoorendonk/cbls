# Builds and runs tests/consumer in one mode (#447); registered as ctest's
# consumer_find_package and consumer_add_subdirectory (root CMakeLists.txt).
#
#   cmake -DMODE=find_package|add_subdirectory -DSOURCE_DIR=<cbls source>
#         -DBUILD_DIR=<cbls build> -DWORK_DIR=<scratch> -DCXX=<compiler>
#         -DJSON_SOURCE_DIR=<fetched nlohmann/json> -DJOBS=<n>
#         -DCONFIG=<cbls's configuration> [-DLINK_FLAGS=<flags>] -P run.cmake
#
# LINK_FLAGS reaches the find_package consumer's link: an installed libcbls.a
# built under CBLS_SANITIZE needs the sanitizer runtime there.
foreach(_v MODE SOURCE_DIR BUILD_DIR WORK_DIR CXX JSON_SOURCE_DIR JOBS CONFIG)
    if(NOT DEFINED ${_v})
        message(FATAL_ERROR "run.cmake: -D${_v}= is required")
    endif()
endforeach()

function(run_step)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "failed (${_rc}): ${ARGN}")
    endif()
endfunction()

# CMake >= 3.22 seeds CMAKE_BUILD_TYPE from the environment, which would trip the
# subproject's "build type left alone" check.
unset(ENV{CMAKE_BUILD_TYPE})
file(REMOVE_RECURSE "${WORK_DIR}")
set(_configure "${CMAKE_COMMAND}" -S "${SOURCE_DIR}/tests/consumer"
    -B "${WORK_DIR}/build" -DCMAKE_CXX_COMPILER=${CXX}
    -DCBLS_CONSUMER_MODE=${MODE})

if(MODE STREQUAL "find_package")
    # The installed tree only: no registry entry or build tree can stand in.
    run_step("${CMAKE_COMMAND}" --install "${BUILD_DIR}" --config "${CONFIG}"
             --prefix "${WORK_DIR}/prefix")
    list(APPEND _configure "-DCMAKE_PREFIX_PATH=${WORK_DIR}/prefix"
         -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF)
    if(LINK_FLAGS)
        list(APPEND _configure "-DCMAKE_EXE_LINKER_FLAGS=${LINK_FLAGS}")
    endif()
elseif(MODE STREQUAL "add_subdirectory")
    # The source cbls's own configure already fetched, so no network here.
    list(APPEND _configure "-DCBLS_SOURCE_DIR=${SOURCE_DIR}"
         "-DFETCHCONTENT_SOURCE_DIR_JSON=${JSON_SOURCE_DIR}")
else()
    message(FATAL_ERROR "run.cmake: MODE '${MODE}'")
endif()

run_step(${_configure})
run_step("${CMAKE_COMMAND}" --build "${WORK_DIR}/build" --config "${CONFIG}"
         --parallel ${JOBS})
run_step("${WORK_DIR}/build/bin/cbls_consumer"
         "${SOURCE_DIR}/benchmarks/instances/miplib-fj/flugpl.mps.gz")
