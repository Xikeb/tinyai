cmake_minimum_required(VERSION 3.19)

project(tinyai-experiments-xor)

add_executable(
        experiment-xor
        ${CMAKE_CURRENT_SOURCE_DIR}/xor.cpp
)
target_link_libraries(experiment-xor tinyai)