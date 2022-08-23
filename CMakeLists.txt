cmake_minimum_required(VERSION 3.19)
project(tinyai)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_COMPILER /usr/bin/g++)
#set(CMAKE_C_COMPILER  /usr/bin/gcc)

add_library(tinyai OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/include/strong_typedef.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyneat.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyann.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/MutationRateContainer.hpp        ${CMAKE_CURRENT_SOURCE_DIR}/src/MutationRateContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SpeciatingParameterContainer.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/SpeciatingParameterContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/NetworkInfoContainer.hpp         ${CMAKE_CURRENT_SOURCE_DIR}/src/NetworkInfoContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Gene.hpp                         ${CMAKE_CURRENT_SOURCE_DIR}/src/Gene.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Genotype.hpp                     ${CMAKE_CURRENT_SOURCE_DIR}/src/Genotype.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Specie.hpp                       ${CMAKE_CURRENT_SOURCE_DIR}/src/Specie.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/InnovationContainer.hpp          ${CMAKE_CURRENT_SOURCE_DIR}/src/InnovationContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/NeatPool.hpp                     ${CMAKE_CURRENT_SOURCE_DIR}/src/NeatPool.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/NeuralNet.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/NeuralNet.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Neuron.cpp include/Phenotypes/Neuron.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NeuralNetFast.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/NeuralNetFast.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/InitializationKit.cpp include/Phenotypes/InitializationKit.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/detail.hpp
        include/Phenotypes/IPhenotype.hpp
        include/Phenotypes/NeuralNetNaive.cpp include/Phenotypes/NeuralNetNaive.hpp include/Phenotypes/NeuralNetNaiveRecurrent.cpp include/Phenotypes/NeuralNetNaiveRecurrent.hpp include/Phenotypes/NeuralNetFastSerialRecurrent.cpp include/Phenotypes/NeuralNetFastSerialRecurrent.hpp)


if (COMMAND easyLink) # Only for development purposes, defined elsewhere
        easyLink(
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_BINARY_DIR}/include/vendor/${PROJECT_NAME}
                ${PROJECT_NAME}_INCLUDE
        )
        
        add_dependencies(tinyai ${PROJECT_NAME}_INCLUDE)
        target_include_directories(tinyai INTERFACE ${CMAKE_BINARY_DIR}/include/vendor/)
        target_include_directories(tinyai PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
else()
        target_include_directories(tinyai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()


set_target_properties(tinyai PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(tinyai PUBLIC -Wall -Wextra)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/googletest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/experiments/xor)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/experiments/SinglePole)