cmake_minimum_required(VERSION 3.24)
project(mlayer VERSION 0.1.0 DESCRIPTION "mcnn concrete layer kernels" LANGUAGES CXX)

include(GNUInstallDirs)
add_library(mlayer INTERFACE)
add_library(mlayer::mlayer ALIAS mlayer)
target_include_directories(mlayer INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
