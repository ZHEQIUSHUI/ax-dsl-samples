cmake_minimum_required(VERSION 3.13)

project(ax_dsl_demo_add LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT AX8860_BSP AND DEFINED ENV{AX8860_BSP})
    set(AX8860_BSP "$ENV{AX8860_BSP}")
endif()

if(NOT AX8860_BSP)
    message(FATAL_ERROR "Set AX8860_BSP to the AX8860 arm64_glibc BSP directory")
endif()

get_filename_component(AX8860_BSP "${AX8860_BSP}" ABSOLUTE)
if(NOT EXISTS "${AX8860_BSP}/include/ax_engine_api.h")
    message(FATAL_ERROR "AX8860_BSP does not contain include/ax_engine_api.h: ${AX8860_BSP}")
endif()

find_package(Threads REQUIRED)

add_executable(ax_demo_add src/ax_demo_add.cc)
target_include_directories(ax_demo_add PRIVATE "${AX8860_BSP}/include")
target_link_directories(ax_demo_add PRIVATE "${AX8860_BSP}/lib")
target_link_options(ax_demo_add PRIVATE "-Wl,-rpath-link,${AX8860_BSP}/lib")
target_link_libraries(ax_demo_add PRIVATE ax_engine ax_sys Threads::Threads)

