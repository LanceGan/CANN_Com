# f:/Projects/CANN_Com/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(CANN_Com LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Static linking for MinGW on Windows (avoids DLL issues with AV/Defender)
if(MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
endif()

# Source files (will grow as we add components)
add_library(cann_sim STATIC
    src/simulator/topology/topology_builder.cpp
    src/simulator/network/link_model.cpp
    src/simulator/channel/pure_sim_channel.cpp
    src/simulator/channel/fault_channel.cpp
    src/simulator/simulator.cpp
    src/algorithm/allreduce/allreduce_ring.cpp
    src/algorithm/allreduce/allreduce_rhd.cpp
    src/algorithm/allgather/allgather_ring.cpp
    src/algorithm/allgather/allgather_butterfly.cpp
    src/algorithm/reduce_scatter/reduce_scatter_ring.cpp
    src/algorithm/reduce_scatter/reduce_scatter_butterfly.cpp
    src/algorithm/alltoall/alltoall_direct.cpp
    src/algorithm/selector/algorithm_selector.cpp
)

target_include_directories(cann_sim PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Tests
enable_testing()
add_subdirectory(tests)
