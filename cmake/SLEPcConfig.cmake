# 定义 SLEPc 作为 INTERFACE 库
add_library(SLEPC INTERFACE)

# 判断当前系统
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(SLEPC_DIR ${PROJECT_SOURCE_DIR}/3rd/slepc/windows)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(SLEPC_DIR ${PROJECT_SOURCE_DIR}/3rd/slepc/linux)
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

set(SLEPC_INCLUDE_DIR ${SLEPC_DIR}/include)
set(SLEPC_LIB_DIR ${SLEPC_DIR}/lib)
set(SLEPC_LIB_NAME slepc)

# 设置头文件路径
target_include_directories(SLEPC INTERFACE ${SLEPC_INCLUDE_DIR})

# 设置库文件搜索路径
target_link_directories(SLEPC INTERFACE ${SLEPC_LIB_DIR})

# 链接 SLEPc 库
target_link_libraries(SLEPC INTERFACE ${SLEPC_LIB_NAME})
