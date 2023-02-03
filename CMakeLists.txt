cmake_minimum_required(VERSION 3.5)
project(npc CXX CUDA)

# Pytorch 
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_torch.py FIND_TORCH_PY)
execute_process(
  COMMAND python3 ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)

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

list(APPEND NPC_LINKER_LIBS ${TORCH_LIBRARIES})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")
message(STATUS "Torch dir: ${Torch_DIR}")
message(STATUS "NPC linker: ${NPC_LINKER_LIBS}")




file(GLOB npc_src
          "src/ops/*.cc"
          "src/*.cc")

include_directories(src ${NPC_INCLUDES})

add_library(npc SHARED ${npc_src})
set_target_properties(npc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(npc PROPERTIES CUDA_STANDARD 11)
set_target_properties(npc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(npc PRIVATE ${NPC_LINKER_LIBS})