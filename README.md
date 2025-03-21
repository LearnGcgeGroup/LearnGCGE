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
- 下载编译安装 OpenBLAS-0.3.28
    ```bash
    # 下载 OpenBLAS
    wget https://github.com/xianyi/OpenBLAS/archive/v0.3.28.tar.gz -O OpenBLAS-0.3.28.tar.gz
    tar -xzf OpenBLAS-0.3.28.tar.gz
    cd OpenBLAS-0.3.28
    # 编译并启用 OpenMP
    make -j4 USE_OPENMP=1 PREFIX=$HOME/deps/OpenBLAS-0.3.28 # 这个路径指定好像没生效，生成在了/opt/OpenBLAS/下
    # 安装
    sudo make install
    echo 'export LD_LIBRARY_PATH=$HOME/deps/OpenBLAS-0.3.28/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export CPATH=/deps/OpenBLAS-0.3.28/include:$CPATH' >> ~/.bashrc
    source ~/.bashrc
    ```
- 下载编译安装 mpich-4.2.3
    ```bash
    wget https://www.mpich.org/static/downloads/4.2.3/mpich-4.2.3.tar.gz
    tar -xzf mpich-4.2.3.tar.gz
    cd mpich-4.2.3
    ./configure --prefix=$HOME/deps/mpich-4.2.3 # --prefix=后面是安装路径，可不填则默认安装到系统路径
    make -j4
    make install
    echo 'export PATH=$HOME/deps/mpich-4.2.3/bin:$PATH' >> ~/.bashrc # 路径要根据自己的安装路径而定
    echo 'export LD_LIBRARY_PATH=$HOME/deps/mpich-4.2.3/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```

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
    mpiexec -n <num> ./test.exe ../example/K.mtx ../example/M.mtx ../example/usrParam.txt
    ```
 - 在Ubuntu bash命令行窗口中执行如下命令, 其中<num>为期望采用的进程数量
    ```bash
    mpiexec -n <num> ./test ../example/K.mtx ../example/M.mtx ../example/usrParam.txt
    ```
#### 2、非MPI形成运行(需参照2.1内容做反向修改)
 - 在Mingw64命令行窗口中执行:

```bash
./test.exe K.mtx M.mtx usrParam.txt
```
 - 在Ubuntu bash命令行窗口中执行:

```bash
./test K.mtx M.mtx usrParam.txt
```
#### 3、运行过程遇见的问题
1. windows系统运行时,不能使用mpiexec命令

--原因：mingw-w64-x86_64-msmpi安装包的部分版本不含msmpi

--措施：在windows系统下载安装msmpisetup.exe

2. ubuntu系统运行时,出现"OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option"

--措施：export OMP_NUM_THREADS=1

3. 编译成功，运行时报错：YOUR APPLICATION TERMINATED WITH THE EXIT STRING: Illegal instruction (signal 4)，加日志定位到PetscInitialize初始化失败。
--原因：因系统差异导致。当前打包的petsc.a是在ubuntu22.04下编译的，不同版本系统需要使用自己系统下编译的petsc.a。
--措施：自己编译petsc.a

### 三、当前软件运行逻辑
    读入MTX矩阵到petsc格式的Mat中，再转换为CCS_MAT格式，调用串行版GCGE求解器
    目的是：验证MTX矩阵读入为petsc的Mat矩阵正确。


### 四、待进一步明确事项
1. libpetsc.a的编译，使用“./configure --with-mpiexec=xx --with-blas-lib=/xxx/lib/libopenblas.a --with-lapack-lib=/xxx/lib/libopenblas.a  xxx”编译的libpetsc.a仅58M大小(上传了)，但是要依赖libmpifort.a等库，这些依赖库是按照另一种“./configure --with xxx”编译出来库(也有一个大小约229M的libpetsc.a)，这些库共同作用才能将petsc作为第三方库使用

2. GCGE编译成动态库时与petsc链接失败
    当前暂时采用GCGE编译成静态库来解决程序跑通的问题，之后修改与petsc的链接问题。

### 附件：原始readme内容
#### 1、外部包

1. 基础线性代数包 BLAS和LAPACK

目前调用的是openblas，如果调用其他库自行修改CMakeLists.txt

2. UMFPACK

直接法解法器，可以加速收敛

#### 2、备注

该算例条件数较大，且矩阵比较病态，目前设置的参数并不适合一般矩阵的特征值问题（会影响效率）；

GCGE1.0_linux/include/ops_config.h 中包含该软件包涉及的一些计算环境的宏定义，注意同步修改这些宏定义；

#### 3、安装方法备份

- 下载编译安装 petsc-3.22.3
    ```bash
    # 网站上下载指定安装包: https://web.cels.anl.gov/projects/petsc/download/release-snapshots/
    tar -xzf petsc-3.22.3.tar.gz
    cd petsc-3.22.3
    ./configure --prefix=$HOME/deps/mpich-4.2.3 # --prefix=后面是安装路径，可不填则默认安装到系统路径
    make -j4
    make install
    echo 'export PATH=$HOME/deps/mpich-4.2.3/bin:$PATH' >> ~/.bashrc # 路径要根据自己的安装路径而定
    echo 'export LD_LIBRARY_PATH=$HOME/deps/mpich-4.2.3/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```

