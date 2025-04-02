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

add_definitions(-DUSE_SLEPC)

if(UNIX)
    set(3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/3rd/ubuntu)
elseif(WIN32)
    set(3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/3rd/windows)
else()
    message(FATAL_ERROR "unspported platform!!!")
endif()
# 添加 CMake查找库 目录
list(APPEND CMAKE_PREFIX_PATH ${3RD_LIB_PATH} ${3RD_LIB_PATH}/lib/cmake)
# 添加 CMake模块文件 目录
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# 载入 MPI 编译器配置
include(linux_mpicc)
# 载入 PETSc 配置
include(PETScConfig)
# 载入 SLEPc 配置
include(SLEPcConfig)

add_subdirectory(gcge)  # 添加 gcge 库的子目录
add_subdirectory(src)   # 添加 src 可执行文件的子目录
