# 定义 PETSc 作为 INTERFACE 库
add_library(petsc INTERFACE)

# 判断当前系统
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PETSC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd/petsc/windows/include)
    set(PETSC_LIB_DIR ${PROJECT_SOURCE_DIR}/3rd/petsc/windows/bin)
    set(PETSC_LIB_NAME petsc-dmo)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PETSC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd/petsc/linux/include)
    set(PETSC_LIB_DIR ${PROJECT_SOURCE_DIR}/3rd/petsc/linux/lib)
    set(PETSC_LIB_NAME petsc)
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# 设置头文件路径
target_include_directories(petsc INTERFACE ${PETSC_INCLUDE_DIR})

# 设置库文件搜索路径
target_link_directories(petsc INTERFACE ${PETSC_LIB_DIR})

# 链接 PETSc 库
target_link_libraries(petsc INTERFACE ${PETSC_LIB_NAME})
