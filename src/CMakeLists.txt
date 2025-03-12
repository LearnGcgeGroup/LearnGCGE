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

# test
include_directories(${PROJECT_SOURCE_DIR}/gcge/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
file(GLOB_RECURSE ALL_SOURCES "*.cpp" "*.c" "*.h")
add_executable(test     ${ALL_SOURCES})

target_link_libraries(test
    PRIVATE 
        GCGE
        ${MPI_C_LIBRARIES}  # Link MPI libraries for the test executable
        OpenMP::OpenMP_CXX
        petsc
)

# 添加自定义命令来拷贝 GCGE 库到 test 可执行文件所在的目录
add_custom_command(TARGET test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/shared/lib/ ${CMAKE_BINARY_DIR}
    COMMENT "Copying GCGE library to test output directory"
)
