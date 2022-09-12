cmake_minimum_required(VERSION 3.19)

project(tinyai-experiments-racing)

add_executable(
	racing
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/level.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/level.h
	${CMAKE_CURRENT_SOURCE_DIR}/physics.h

	${CMAKE_CURRENT_SOURCE_DIR}/TINY/tinyxmlerror.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/TINY/tinyxml.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/TINY/tinystr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/TINY/tinyxmlparser.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/TINY/tinyxml2.cpp
)

#todo: gather the assets into a single directory, and link it to a place closer to the binary
target_link_libraries(racing tinyai sfml-graphics sfml-window sfml-system)
