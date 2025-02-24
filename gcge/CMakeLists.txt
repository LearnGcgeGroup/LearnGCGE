# src
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/app SRC_APP)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)

# this project
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# MPI include directories
include_directories(${MPI_C_INCLUDE_PATH})

#UMFPACK
# 问题3：将umfpack的头文件路径放置在了环境变量中，通过添加环境变量将其添加到include_directories中，方式不好。
#        可以直接建立子文件夹，将umfpack的头文件放置在其中，然后通过include_directories添加。
include_directories($ENV{SUITESPARSE_HEADER_DIR})
# shared library
add_library(GCGE SHARED
    ${SRC_APP}
    ${SRCS}
)

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)

# 如果找到 OpenMP，将其链接到目标
if(OpenMP_FOUND)
    message(STATUS "OpenMP found, enabling support.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  # 添加 OpenMP 编译选项
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

if (OpenBLAS_FOUND)
    message(STATUS "OpenBLAS found, enabling support.")
    include_directories(${OpenBLAS_INCLUDE_DIRS})
    link_openblas(GCGE)
else()
    message(FATAL_ERROR "OpenBLAS not found!")
endif()

# 问题1：没有指定openblas发布件和头文件的位置，因此需要保证openblas的so和头文件均在系统路径中，这样才找得到。
# 问题2：没有指定umfpack发布件的位置，因此需要保证umfpack的so在系统路径中，这样才找得到。
target_link_libraries(GCGE
    PUBLIC
        ${MPI_C_LIBRARIES}  # Link MPI libraries
        OpenMP::OpenMP_C

)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/shared/lib)
set(INCLUDE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/shared/include)
file(GLOB_RECURSE HEADER_ALL
    ${PROJECT_SOURCE_DIR}/include/*.h
)
file(COPY ${HEADER_ALL} DESTINATION ${INCLUDE_OUTPUT_PATH})
