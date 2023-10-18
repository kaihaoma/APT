cmake_minimum_required(VERSION 3.20)
project(npc CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Pytorch 
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_torch.py FIND_TORCH_PY)
execute_process(
  COMMAND python ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
list(APPEND NPC_LINKER_LIBS ${TORCH_LIBRARIES})

# GLOG
find_path(glog_INCLUDE_DIR NAMES glog/logging.h)
find_library(glog_LIBRARY NAMES glog)
if(glog_INCLUDE_DIR AND glog_LIBRARY)
  message(STATUS "Found glog:")
  message(STATUS "  (Headers)       ${glog_INCLUDE_DIR}")
  message(STATUS "  (Library)       ${glog_LIBRARY}")
  list(APPEND NPC_INCLUDES ${glog_INCLUDE_DIR})
  list(APPEND NPC_LINKER_LIBS ${glog_LIBRARY})
else()
  message(FATAL_ERROR "glog not found")
endif()

#cmake_policy(SET CMP0148 OLD)
#find_package(PythonLibs REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python include dir: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
#include_directories(${PYTHON_INCLUDE_DIRS})
#target_link_libraries(<your exe or lib> ${PYTHON_LIBRARIES})
list(APPEND NPC_INCLUDES ${Python_INCLUDE_DIRS})
list(APPEND NPC_LINKER_LIBS ${Python_LIBRARIES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")
message(STATUS "Torch dir: ${Torch_DIR}")
message(STATUS "NPC linker: ${NPC_LINKER_LIBS}")

file(GLOB npc_src
          "src/ops/*.cc"
          "src/*.cc"
          "src/cuda/*cu")

include_directories(src ${NPC_INCLUDES})

add_library(npc SHARED ${npc_src})
set_target_properties(npc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(npc PROPERTIES CUDA_STANDARD 14)
set_target_properties(npc PROPERTIES CXX_STANDARD 14)
set_target_properties(npc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(npc ${NPC_LINKER_LIBS})

find_package(MPI REQUIRED)
message(STATUS "MPI_INCLUDE_PATH -- ${MPI_INCLUDE_PATH}")
message(STATUS "MPI_CXX_LIBRARIES -- ${MPI_CXX_LIBRARIES}")
add_subdirectory(third_party/googletest)

enable_testing()
include_directories(third_party/googletest/googletest/include)
include_directories("src")
include_directories(${MPI_INCLUDE_PATH})
file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/*.cc)
add_executable(runUnitTests ${TEST_SRC_FILES})
target_link_libraries(runUnitTests npc gtest gtest_main ${MPI_CXX_LIBRARIES})
add_test(UnitTests runUnitTests)
