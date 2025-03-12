# project
cmake_minimum_required(VERSION 3.16)
project (GCGE)
set(GCGE_VERSION_MAJOR 1)
set(GCGE_VERSION_MINOR 0)
set(CMAKE_VERBOSE_MAKEFILE ON)
# build type (set defualt type: RelWithDebInfo)
if(NOT CMAKE_BUILD_TYPE)
  message("Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# 添加 CMake 目录
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/config)

include(linux_mpicc)
# 载入 PETSc 配置
include(PETScConfig)

add_subdirectory(gcge)  # 添加 gcge 库的子目录
add_subdirectory(src)   # 添加 src 可执行文件的子目录
