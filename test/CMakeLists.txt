cmake_minimum_required(VERSION 3.19)

project(tinyai-tests)

set(CMAKE_CXX_STANDARD 20)

enable_testing()

add_executable(
  NeuralNetFast
  ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetFast.cpp
)
target_link_libraries(NeuralNetFast tinyai)
target_link_libraries(NeuralNetFast GTest::gmock GTest::gtest_main)
# target_include_directories(NeuralNetFast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(NeuralNetFast)
