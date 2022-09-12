cmake_minimum_required(VERSION 3.19)
project(tinyai)

set(CMAKE_CXX_STANDARD 20)

add_library(tinyai OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/strong_typedef.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/tinyneat.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/tinyann.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/MutationRateContainer.hpp        ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/MutationRateContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/SpeciatingParameterContainer.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/SpeciatingParameterContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/NetworkInfoContainer.hpp         ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/NetworkInfoContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Gene.hpp                         ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/Gene.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Genotype.hpp                     ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/Genotype.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Specie.hpp                       ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/Specie.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/InnovationContainer.hpp          ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/InnovationContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/NeatPool.hpp                     ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/NeatPool.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/NeuralNet.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/NeuralNet.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/Neuron.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Phenotypes/Neuron.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/NeuralNetFast.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/NeuralNetFast.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyai/InitializationKit.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Phenotypes/InitializationKit.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/detail.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Phenotypes/IPhenotype.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Phenotypes/NeuralNetNaive.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Phenotypes/NeuralNetNaive.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Phenotypes/NeuralNetNaiveRecurrent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Phenotypes/NeuralNetNaiveRecurrent.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Phenotypes/NeuralNetFastSerialRecurrent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyai/Phenotypes/NeuralNetFastSerialRecurrent.hpp
)

target_include_directories(tinyai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(tinyai PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(tinyai PUBLIC -Wall -Wextra)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/googletest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/SFML)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/experiments/xor)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/experiments/SinglePole)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/experiments/racing)