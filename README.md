### 一、GCGE配置编译

#### 1、Windows系统
##### 1.1 下载外部依赖包
GCGE当前依赖的外部包如下：
- mingw-w64-x86_64-msmpi 10.1.1-11
- mingw-w64-x86_64-openblas 0.3.28-1
- mingw-w64-x86_64-gcc 14.2.0-1(其中自带openmp)

windows平台下msys2环境：采用`pacman -S mingw-w64-x86_64-msmpi 10.1.1-11`进行安装，其他包类似

##### 1.2 编译
```bash
cd LearnGCGE
mkdir build
cd build
cmake ..
make 
```

##### 1.3 配置编译过程遇见的问题
编译时不能链接libgomp.a和libmingwthrd.a

--原因：测试用电脑已安装有mingw64(并设置系统变量)

--措施：卸载已有的mingw64，重新按上述流程编译


#### 2、Ubuntu18.04系统
##### 2.1 下载外部依赖包
- sudo apt install build-essential openssl libssl-dev 
- 下载编译安装 cmake-3.16.5
- 下载编译安装 OpenBLAS-0.3.24    
- 下载编译安装 mpich-4.2.3

##### 2.2 编译
```bash
cd LearnGCGE
mkdir build
cd build
cmake ..
make 
```

##### 2.3 配置编译过程遇见的问题
1.当cmake..时，find_package(OpenBLAS REQUIRED)失败

--原因：使用sudo apt install libopenblas-dev安装了低版本openblas

--措施：下载编译安装 OpenBLAS-0.3.24 


2.当make时，出现"error: conflicting declaration of C function 'void MPI::Init(int&, char**&)' extern void Init(int&, char**&);"

--原因：使用sudo apt install openmpi-bin libopenmpi-dev安装了低版本openmpi

--措施：下载编译安装 mpich-4.2.3


### 二、运行

#### 1、采用MPI运行方式
编译前(默认设置为MPI形式，因此无需修改)：
 - 修改`gcge\include\ops_config.h`中`#define  OPS_USE_MPI       0`值为1
 - 取消`CMakeLists.txt`中注释` # include(linux_mpicc)`

运行时：
 - 在Mingw64命令行窗口中执行如下命令, 其中<num>为期望采用的进程数量
    ```bash
    mpiexec -n <num> ./test.exe K.mtx M.mtx usrParam.txt
    ```

#### 2、非MPI形成运行(需参照2.1内容做反向修改)
在Mingw64命令行窗口中执行:

```bash
./test.exe K.mtx M.mtx usrParam.txt
```

#### 3、运行过程遇见的问题
1.windows系统运行时,不能使用mpiexec命令

--原因：mingw-w64-x86_64-msmpi安装包的部分版本不含msmpi

--措施：在windows系统下载安装msmpisetup.exe

2.ubuntu系统运行时,出现"OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option"

--措施：export OMP_NUM_THREADS=1

# GCGE文件结构

app

config

src

include

# GCGE配置编译

## 外部包

1. 基础线性代数包 BLAS和LAPACK

目前调用的是openblas，如果调用其他库自行修改CMakeLists.txt

2. UMFPACK

直接法解法器，可以加速收敛

## 编译

```bash
cd GCGE1.0_linux/
mkdir build
cd build
cmake ..
make 
```

生成的库文件及头文件见GCGE1.0_linux/shared

## 调用

示例程序路径 GCGE1.0_linux/example

在CMakeLists.txt中已包含对该算例的编译，生成的可执行文件为GCGE1.0_linux/build/test

执行

```bash
./test test1.mtx
```

其中test1.mtx 即为提供的测试算例，补充了上三角部分（标准稀疏矩阵库mtx格式）


## 备注

该算例条件数较大，且矩阵比较病态，目前设置的参数并不适合一般矩阵的特征值问题（会影响效率）；

GCGE1.0_linux/include/ops_config.h 中包含该软件包涉及的一些计算环境的宏定义，注意同步修改这些宏定义；

目前包里提供的mtx转CSC的程序是通用的，注意矩阵文件需要是完整的；

计算结果已与主流特征值解法器SLEPC进行对比，即便是后面较大量级的特征值，在对应精度范围内小数点都是能对上的，且1的代数重数也是一样的


