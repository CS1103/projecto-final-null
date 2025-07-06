cmake_minimum_required(VERSION 3.18)
project(ProyectoFinalNeuralNetwork)

# Establecer el estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuraciones del compilador
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

# Configuración de Debug/Release
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Incluir directorios de headers
include_directories(${CMAKE_SOURCE_DIR})

# Ejecutable principal
add_executable(neural_net_demo 
    main.cpp
)

# Agregar headers como dependencias (para IDEs)
set(HEADER_FILES
    src/tensor.h
    src/neural_network.h
    src/layers/nn_interfaces.h
    src/layers/nn_dense.h
    src/layers/nn_activation.h
    src/layers/nn_loss.h
    src/optimizers/nn_optimizer.h
)

target_sources(neural_net_demo PRIVATE ${HEADER_FILES})

# Propiedades del target
set_target_properties(neural_net_demo PROPERTIES
    OUTPUT_NAME "neural_net_demo"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Tests (opcional)
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    
    # Buscar o agregar framework de testing (por ejemplo, Catch2)
    find_package(Catch2 QUIET)
    
    if(Catch2_FOUND)
        add_executable(run_tests
            tests/test_tensor.cpp
            tests/test_neural_network.cpp
        )
        target_link_libraries(run_tests Catch2::Catch2)
        add_test(NAME tensor_tests COMMAND run_tests)
    else()
        message(WARNING "Catch2 not found. Tests will not be built.")
    endif()
endif()

# Información de compilación
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output Directory: ${CMAKE_BINARY_DIR}/bin")
