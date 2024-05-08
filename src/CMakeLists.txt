set(ESTIMATION_SOURCE kalman_filter.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -ffast-math")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if (DEFINED SKBUILD)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(lib NB_STATIC module.cpp ${ESTIMATION_SOURCE})
  add_compile_definitions(PYTHON_MODULE)
  target_link_libraries(lib PRIVATE Eigen3::Eigen)

  install(TARGETS lib LIBRARY DESTINATION .)
else()
  add_library(estimation SHARED ${ESTIMATION_SOURCE})
  target_link_libraries(estimation PRIVATE Eigen3::Eigen)
  target_sources(estimation PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    FILES kalman_filter.hpp)

  add_executable(test test.cpp)
  target_link_libraries(test PRIVATE Eigen3::Eigen estimation)

  install(TARGETS estimation test
    LIBRARY DESTINATION lib
    FILE_SET public_headers DESTINATION include/estimation)
endif()
