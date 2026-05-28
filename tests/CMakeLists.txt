# f:/Projects/CANN_Com/tests/CMakeLists.txt
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

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
