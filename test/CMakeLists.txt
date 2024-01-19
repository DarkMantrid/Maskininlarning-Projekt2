cmake_minimum_required(VERSION 3.20)
project(gpio_cpp_tests)

# Locate GTest package
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include directories for your project's source and header files
include_directories(../inc)

# Add a test executable
add_executable(run_gpio_tests 
    test_neural_network.cpp
    ../src/node.cpp
    ../src/gpio_utils.cpp
)

# Link test executable against gtest & your project's libraries
target_link_libraries(run_gpio_tests ${GTEST_BOTH_LIBRARIES} gpiod pthread)

# Set compiler options
target_compile_options(run_gpio_tests PRIVATE -Wall -Werror)

# Set output directory for the test executable
set_target_properties(run_gpio_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../)
