cmake_minimum_required(VERSION 3.19)

project(tinyai-experiments-single-pole)

add_executable(
	single-pole
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(single-pole tinyai sfml-graphics sfml-audio sfml-window sfml-system)
