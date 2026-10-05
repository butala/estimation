set(ESTIMATION_SOURCE kalman_filter.cpp square_root_kf.cpp)
set(ESTIMATION_HEADER filter.hpp kalman_filter.hpp square_root_kf.hpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# No -ffast-math: it flushes NaN/Inf and reassociates, which is unsafe for a
# Kalman filter (NaN is the filter-divergence alarm; Cholesky must preserve SPD)
# and makes results compiler-dependent.
# -mtune=native is always safe (scheduling only). -march=native is added only for
# local builds -- it would make a redistributable (SKBUILD) wheel non-portable.
# NDEBUG is NOT set here: it comes from CMAKE_CXX_FLAGS_<CONFIG> for release
# targets, and is deliberately undefined for the test binary so that the
# contract tests can prove the asserts fire (see test.cpp [contracts]).
set(CMAKE_CXX_FLAGS "-Wall -O3 -mtune=native")
if (NOT DEFINED SKBUILD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

option(ESTIMATION_SANITIZE "Build the test binary with ASan + UBSan + LSan" OFF)

# Any Eigen 3.4+ or 5.x. The 3.3-style version request is rejected by Eigen 5
# config files (same-major compatibility), so the floor is stated here instead.
find_package(Eigen3 REQUIRED NO_MODULE)

# Two build flavours:
#   * the C++ library + tests                 (default: cmake -B build)
#   * the Python extension module             (-DESTIMATION_BUILD_PYTHON=ON)
# The Python flavour is also selected automatically when scikit-build-core is
# building a wheel (it defines SKBUILD).
if (DEFINED SKBUILD OR ESTIMATION_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(lib NB_STATIC module.cpp ${ESTIMATION_SOURCE})
  # target_compile_definitions (not add_compile_definitions): the directory
  # form only applies to targets created after it, so it would miss `lib`.
  target_compile_definitions(lib PRIVATE PYTHON_MODULE NDEBUG EIGEN_NO_DEBUG)
  target_link_libraries(lib PRIVATE Eigen3::Eigen)

  install(TARGETS lib LIBRARY DESTINATION .)
else()
  add_library(estimation SHARED ${ESTIMATION_SOURCE})
  target_compile_definitions(estimation PRIVATE NDEBUG EIGEN_NO_DEBUG)
  target_link_libraries(estimation PRIVATE Eigen3::Eigen)
  target_sources(estimation PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${ESTIMATION_HEADER})

  # The test compiles the sources itself rather than linking libestimation, so
  # that the asserts are live: the [contracts] cases fork and require SIGABRT.
  add_executable(test test.cpp ${ESTIMATION_SOURCE})
  target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test PRIVATE Eigen3::Eigen)
  # Undefine NDEBUG after the per-config flags have defined it (Release does).
  # Eigen's own eigen_assert is left live too -- it catches misuse of Eigen.
  target_compile_options(test PRIVATE -UNDEBUG)

  if (ESTIMATION_SANITIZE)
    target_compile_options(test PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(test PRIVATE -fsanitize=address,undefined)
  endif()

  install(TARGETS estimation test
    LIBRARY DESTINATION lib
    FILE_SET public_headers DESTINATION include/estimation)

  enable_testing()
  add_test(NAME unit COMMAND test)
endif()
