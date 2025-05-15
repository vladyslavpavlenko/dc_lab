cmake_minimum_required(VERSION 3.30)
project(brutalist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

find_package(OpenMP REQUIRED)

add_executable(brutalist main.cpp)

target_link_libraries(brutalist PRIVATE OpenMP::OpenMP_CXX)