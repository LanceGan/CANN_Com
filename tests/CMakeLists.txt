# f:/Projects/CANN_Com/tests/CMakeLists.txt
find_package(GTest REQUIRED)

# Helper function to add tests
function(add_cann_test TEST_NAME TEST_SRC)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE cann_sim GTest::gtest_main)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

add_cann_test(test_types       unit/test_types.cpp)
add_cann_test(test_topology    unit/test_topology.cpp)
add_cann_test(test_link_model  unit/test_link_model.cpp)
add_cann_test(test_channel     unit/test_pure_sim_channel.cpp)
add_cann_test(test_simulator   unit/test_simulator.cpp)
add_cann_test(test_allreduce       unit/test_allreduce.cpp)
add_cann_test(test_allgather       unit/test_allgather.cpp)
add_cann_test(test_allgather_butterfly unit/test_allgather_butterfly.cpp)
add_cann_test(test_reduce_scatter  unit/test_reduce_scatter.cpp)
add_cann_test(test_reduce_scatter_butterfly  unit/test_reduce_scatter_butterfly.cpp)
add_cann_test(test_alltoall        unit/test_alltoall.cpp)
add_cann_test(test_broadcast       unit/test_broadcast.cpp)
add_cann_test(test_algorithm_selector unit/test_algorithm_selector.cpp)
add_cann_test(test_allreduce_rhd    unit/test_allreduce_rhd.cpp)
add_cann_test(test_allreduce_pipeline unit/test_allreduce_pipeline.cpp)
add_cann_test(test_fault_channel    unit/test_fault_channel.cpp)
add_cann_test(test_multinode        unit/test_multinode.cpp)
add_cann_test(test_reliability      fault/test_reliability.cpp)
add_cann_test(test_mixed_precision  unit/test_mixed_precision.cpp)

# Benchmark
add_executable(bench_comm benchmark/bench_comm.cpp)
target_link_libraries(bench_comm PRIVATE cann_sim)
