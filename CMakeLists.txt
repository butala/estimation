cmake_minimum_required(VERSION 3.20)

# scikit-build-core sets SKBUILD_PROJECT_* when building the Python wheel. Fall
# back to a local C++ build when the project is configured directly with cmake,
# so that `cmake -B build` works with no extra arguments.
if (NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME estimation)
endif()
if (NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 0.0.1)
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

add_subdirectory(src)
