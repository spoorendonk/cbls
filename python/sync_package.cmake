# Mirrors python/cbls (SRC) into the build tree's package (DST) at build time
# (#464): copies what changed and deletes any file the source no longer has, so
# pytest cannot import a module the wheel lacks. The compiled _cbls_core (.so,
# .pyd) and bytecode caches are the build's own and are left alone.
#   cmake -DSRC=<dir> -DDST=<dir> -P sync_package.cmake
file(GLOB _sources RELATIVE "${SRC}" LIST_DIRECTORIES false "${SRC}/*")
file(GLOB _present RELATIVE "${DST}" LIST_DIRECTORIES false "${DST}/*")
foreach(_name IN LISTS _present)
    if(NOT _name IN_LIST _sources AND NOT _name MATCHES "^_cbls_core\\..*\\.(so|pyd)$")
        file(REMOVE "${DST}/${_name}")
    endif()
endforeach()
file(MAKE_DIRECTORY "${DST}")
foreach(_name IN LISTS _sources)
    # Not file(COPY_FILE): that needs CMake 3.21, the root asks for 3.15.
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                            "${SRC}/${_name}" "${DST}/${_name}"
                    RESULT_VARIABLE _status)
    if(NOT _status EQUAL 0)
        message(FATAL_ERROR "could not copy ${SRC}/${_name} to ${DST}")
    endif()
endforeach()
