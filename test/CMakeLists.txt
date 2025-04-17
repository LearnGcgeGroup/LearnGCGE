
set(GOOGLETEST_LIB_DIR "${CMAKE_SOURCE_DIR}/test/infra/googletest")  # 替换为你安装的实际路径

# 找到 GoogleTest 的头文件和库文件
link_directories(${GOOGLETEST_DIR}/build/lib)  # 如果 GoogleTest 是用 CMake 构建的，这里是编译后的库路径

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)

# 如果找到 OpenMP，将其链接到目标
if(OpenMP_FOUND)
    message(STATUS "OpenMP found, enabling support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  # 添加 OpenMP 编译选项
endif()
find_package(OpenBLAS REQUIRED)

# 创建测试可执行文件
file(GLOB_RECURSE ALL_SOURCES "*.cpp" "*.c" "*.h")
add_executable(my_test ${ALL_SOURCES})  # 你的测试文件

# 链接 GoogleTest 和 pthread 库
target_link_libraries(my_test PRIVATE 
    "${SLEPC_DIR}/lib/libslepc.a"     # SLEPC需要放在PETSC之前
    "${PETSC_DIR}/lib/libpetsc.a"     # PETSC # todo: 这样写存在链接错误，为什么
    GCGE
    MPI::MPI_C          # MPI C 库， # todo: 采用${MPI_C_LIBRARIES}会失败，为什么
    OpenBLAS::OpenBLAS  # OpenBLAS
    gfortran            # GNU Fortran 运行时库
    stdc++              # C++ 标准库
    quadmath            # 高精度数学库（Fortran 相关）
    OpenMP::OpenMP_C    # OpenMP C 库, todo: 是否要包含OpenMP::OpenMP_CXX
    gtest 
    gtest_main 
    pthread)

include_directories(
    ${PROJECT_SOURCE_DIR}/gcge/include
    ${PROJECT_SOURCE_DIR}/src
    ${PETSC_INCLUDE_DIR} # PETSc 头文件路径
    ${SLEPC_INCLUDE_DIR} # SLEPc 头文件路径
    ${GOOGLETEST_LIB_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MPI_C_INCLUDE_PATH}
    ${PETSC_INCLUDE_DIR} # PETSc 头文件路径
    ${SLEPC_INCLUDE_DIR} # SLEPc 头文件路径
)
# 设置库文件搜索路径
link_directories(
    ${GOOGLETEST_LIB_DIR}/lib
    ${PETSC_LIB_DIR}
    ${SLEPC_LIB_DIR}
)

# 注册测试
add_test(NAME MyTest COMMAND my_test)
