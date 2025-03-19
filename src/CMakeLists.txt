SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 解决C++与C语言混合编译问题
add_definitions(-DMPICH_SKIP_MPICXX)

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)

# 如果找到 OpenMP，将其链接到目标
if(OpenMP_FOUND)
    message(STATUS "OpenMP found, enabling support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  # 添加 OpenMP 编译选项
endif()
find_package(OpenBLAS REQUIRED)

# 暂时解决mingw下OpenBLAS_LIBRARIES未定义问题。之后统一编写cmake文件
# 定义宏来根据操作系统选择链接方式
macro(link_openblas target)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # 在 Ubuntu 上使用 ${OpenBLAS_LIBRARIES}
        target_link_libraries(${target} PRIVATE ${OpenBLAS_LIBRARIES})
        message(STATUS "Linking OpenBLAS on Linux: ${OpenBLAS_LIBRARIES}")
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # 在 MinGW 上使用 OpenBLAS::OpenBLAS
        target_link_libraries(${target} PRIVATE OpenBLAS::OpenBLAS)
        message(STATUS "Linking OpenBLAS on Windows (MinGW): OpenBLAS::OpenBLAS")
    else()
        message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}")
    endif()
endmacro()

# test
include_directories(
    ${PROJECT_SOURCE_DIR}/gcge/include
    ${PROJECT_SOURCE_DIR}/src
    ${PETSC_INCLUDE_DIR} # PETSc 头文件路径
    ${SLEPC_INCLUDE_DIR} # SLEPc 头文件路径
)
# 设置库文件搜索路径
link_directories(
    ${PETSC_LIB_DIR}
    ${SLEPC_LIB_DIR}
)

file(GLOB_RECURSE ALL_SOURCES "*.cpp" "*.c" "*.h")
add_executable(test     ${ALL_SOURCES})

if (OpenBLAS_FOUND)
    message(STATUS "OpenBLAS found, enabling support.")
    include_directories(${OpenBLAS_INCLUDE_DIRS})
    link_openblas(test)
else()
    message(FATAL_ERROR "OpenBLAS not found!")
endif()

if(UNIX)
    target_link_libraries(test PRIVATE 
        "${SLEPC_DIR}/lib/libslepc.a"     # SLEPC需要放在PETSC之前
        "${PETSC_DIR}/lib/libpetsc.a"     # PETSC # todo: 这样写存在链接错误，为什么
        GCGE
        MPI::MPI_C          # MPI C 库， # todo: 采用${MPI_C_LIBRARIES}会失败，为什么
        openblas            # OpenBLAS
        m                   # petsc需要的库
        mpifort
        mpi
        gfortran            # GNU Fortran 运行时库
        gcc_s
        stdc++              # C++ 标准库
        quadmath            # 高精度数学库（Fortran 相关）
        OpenMP::OpenMP_C    # OpenMP C 库, todo: 是否要包含OpenMP::OpenMP_CXX
    )
endif()

if(WIN32)
    target_link_libraries(test PRIVATE 
        "${PETSC_DIR}/lib/libpetsc.a"     # PETSC # todo: 这样写存在链接错误，为什么
        GCGE
        MPI::MPI_C          # MPI C 库， # todo: 采用${MPI_C_LIBRARIES}会失败，为什么
        OpenBLAS::OpenBLAS  # OpenBLAS
        regex               # 正则表达式库
        gfortran            # GNU Fortran 运行时库
        stdc++              # C++ 标准库
        quadmath            # 高精度数学库（Fortran 相关）
        OpenMP::OpenMP_C    # OpenMP C 库, todo: 是否要包含OpenMP::OpenMP_CXX
    )
endif()

if(WIN32)
    target_link_libraries(test PRIVATE
        gdi32               # Windows GDI（图形设备接口）
        user32              # Windows 用户界面库
        advapi32            # Windows 高级 API
        kernel32            # Windows 核心 API
    )
endif()

# 添加自定义命令来拷贝 GCGE 库到 test 可执行文件所在的目录
add_custom_command(TARGET test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/shared/lib/ ${CMAKE_BINARY_DIR}
    COMMENT "Copying GCGE library to test output directory"
)
