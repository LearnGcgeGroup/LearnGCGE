/**
 *    @file  app_slepc.c
 *   @brief  app of slecp 
 *
 *  不支持单向量操作 
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/9/13
 *      Revision:  none
 */

#include "app_slepc.h"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#if OPS_USE_SLEPC
#define DEBUG 0

/* 进程分组, 主要用于 AMG, 默认最大层数是16 */
int MG_COMM_COLOR[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
/* 能否这样赋初值 尤其时 MPI_COMM_WORLD 
 * 另外, 这些创建出来的通讯域可以 MPI_Comm_free 吗? 何时 */
MPI_Comm MG_COMM[16][2] = {
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL},
    {MPI_COMM_NULL, MPI_COMM_NULL}
};
MPI_Comm MG_INTERCOMM[16] = {
    MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL,
    MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL,
    MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL,
    MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL};

/* multi-vec */
/**
 * @brief 基于 Mat 类型的矩阵创建 BV（Block Vector）对象，并初始化其大小和设置随机值
 * 
 * @param des_bv 要创建BV对象的存储地址
 * @param num_vec BV对象的列数
 * @param src_mat BV对象长度与src_mat的行数一致
 * @param ops 未使用
 */
static void MultiVecCreateByMat(BV *des_bv, int num_vec, Mat src_mat, struct OPS_ *ops) {
    Vec vector; // 用于存储矩阵向量乘法结构的向量
    /**
     * @brief 为给定的矩阵 src_mat 创建两个向量
     * @param src_mat 表示你要为其创建向量的矩阵
     * @param right: 输出参数，这个向量的维度与矩阵的列数相同，通常用于矩阵-向量乘法操作中的输入向量。
     * @param left: 输出参数，这个向量的维度与矩阵的行数相同，通常用于矩阵-向量乘法操作中的输出向量。
     */
    MatCreateVecs(src_mat, NULL, &vector);
    /**
     * @brief 创建一个空的 BV 对象
     * @param MPI_Comm 用于指定并行计算的通信环境 PETSC_COMM_WORLD，表示所有进程参与计算
     * @param des_bv 输出参数，用于存储创建的 BV 对象
     */
    BVCreate(PETSC_COMM_WORLD, des_bv);
    /**
     * @brief 指定 BV 对象的内部存储类型
     * @param BV 要设置类型的 BV 对象
     * @param BVMAT 将基向量存储在一个密集矩阵中
     */
    BVSetType(*des_bv, BVMAT);
    /**
     * @brief 根据给定的向量vector设置 BV 对象的大小：BV 中每个基向量的长度（行数）将与 vector 的维度相同；BV 中基向量的数量（列数）由参数 num_vec 指定
     * @param des_bv 要设置大小的 BV 对象
     * @param vector BV 的向量大小（即每个基向量的长度）将根据该向量的维度设置
     * @param num_vec BV 中基向量的数量（即 BV 的列数）
     */
    BVSetSizesFromVec(*des_bv, vector, num_vec);

    // 备选设置方案 start
    // PetscInt n, N;
    // VecGetLocalSize(vector, &n);
    // VecGetSize(vector, &N);
    // BVSetSizes(*des_bv, n, N, num_vec);
    // 备选设置方案 end

    /**
     * @brief 释放内存
     * @param Vec * 要释放的向量
     */
    VecDestroy(&vector);
    /**
     * @brief 设置 BV 对象中活跃列的范围，活跃列是指当前正在使用的列。可以限制 BV 的操作仅作用于指定的列。
     * @param des_bv 要设置活跃列的 BV 对象
     * @param PetscInt 0 活跃列的起始列索引
     * @param PetscInt num_vec 活跃列的数量
     */
    BVSetActiveColumns(*des_bv, 0, num_vec);
    /**
     * @brief 将 BV 对象中的所有基向量设置为随机值
     * @param BV *des_bv 要设置随机值的 BV 对象
     */
    BVSetRandom(*des_bv);
    return;
}

/**
 * @brief 析构多维向量
 * 
 * @param des_bv 要析构对象的指针
 * @param num_vec 未用到
 * @param ops 未用到
 */
static void MultiVecDestroy(BV *des_bv, int num_vec, struct OPS_ *ops) {
    BVDestroy(des_bv);
    return;
}

/**
 * @brief 查看多维向量BV的内容
 * 
 * @param x 要查看的多维向量BV
 * @param start 要查看的起始列索引
 * @param end 查看的列数
 * @param ops 未用到
 * 
 * @note MultiVecView 及类似的 BVView、MatView、VecView 等视图（View）函数默认只会在主进程（通常是 rank 0）执行一次，而不会在每个进程执行一次
 */
static void MultiVecView(BV x, int start, int end, struct OPS_ *ops) {
    BVSetActiveColumns(x, start, end);
    /**
     * @brief 将 BV 对象的内容输出到指定的查看器中
     * @param x 要查看的 BV 对象
     * @param viewer 用于输出 BV 内容的查看器。可以是标准输出、文件或其他输出目标。
     */
    BVView(x, PETSC_VIEWER_STDOUT_WORLD);
    return;
}

/**
 * @brief 计算多个向量的局部内积（或块内积），结果存储在指定矩阵中
 * 输入为slepc的BV结构，将其转换为LAPACKVEC结构，调用LAPACK的多向量内积计算函数计算。
 * 由于并行程序中每个进程的BV结构存储部分数据，因此多进程运行时，效率高于单一进程下LAPACKVEC求解所有数据
 * 
 * @remark 此函数并行效率高
 * @param[in] nsdIP     字符参数，指定内积存储方式（例如'S'表示对称存储）
 * @param[in] x         输入矩阵/向量x（对应运算中的Q矩阵）
 * @param[in] y         输入矩阵/向量y（对应运算中的P矩阵）
 * @param[in] is_vec    向量模式标识
 * @param[in] start     二维数组起始索引[start[0], start[1]]
 * @param[in] end       二维数组结束索引[end[0], end[1]]
 * @param[out] inner_prod 结果存储矩阵指针（存放内积计算结果）
 * @param[in] ldIP      inner_prod矩阵的leading dimension
 * @param[in] ops       运算选项结构体
 * 
 * @return 无返回值，结果存储在inner_prod矩阵中
 */
static void MultiVecLocalInnerProd(char nsdIP,
                                   BV x, BV y, int is_vec, int *start, int *end,
                                   double *inner_prod, int ldIP, struct OPS_ *ops) {
    assert(is_vec == 0);

    const PetscScalar *x_array, *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;
    PetscInt x_ld, y_ld; // BV对象转换为array后的主维度
    /**
     * @brief 获取 BV 对象中基向量的只读数组的函数（const 限制）
     * @param x 要获取数组的 BV 对象
     * @param const PetscScalar ** array 输出参数，指向 BV 中基向量的只读数组
     */
    BVGetArrayRead(x, &x_array);
    /**
     * @brief 获取 BV 对象的维度信息
     * @param x 要获取维度信息的 BV 对象
     * @param PetscInt *x_nrows 输出参数，指向 BV 中每个基向量的本地长度（行数）。如果不需要此信息，可以传入 NULL（在并行程序中，表示本地进程的向量长度）。。
     * @param PetscInt *NULL 输出参数，指向 BV 中基向量的全局长度（行数）。如果不需要此信息，可以传入 NULL。
     * @param PetscInt *x_ncols 输出参数，指向 BV 的全局列数
     */
    BVGetSizes(x, &x_nrows, NULL, &x_ncols);
    BVGetLeadingDimension(x, &x_ld);
    if (is_vec == 0) {
        BVGetArrayRead(y, &y_array);
        BVGetSizes(y, &y_nrows, NULL, &y_ncols);
        BVGetLeadingDimension(y, &y_ld);
        LAPACKVEC x_vec, y_vec;
        x_vec.nrows = x_nrows;
        y_vec.nrows = y_nrows;
        x_vec.ncols = x_ncols;
        y_vec.ncols = y_ncols;
        x_vec.ldd = x_ld;
        y_vec.ldd = y_ld;
        x_vec.data = (double *)x_array;
        y_vec.data = (double *)y_array;
        ops->lapack_ops->MultiVecLocalInnerProd(nsdIP,
                                                (void **)&x_vec, (void **)&y_vec, is_vec,
                                                start, end, inner_prod, ldIP, ops->lapack_ops);
        /**
         * @brief 用于释放通过 BVGetArrayRead 获取的只读数组
         * @param BV y 要释放数组的 BV 对象
         * @param const PetscScalar ** y_array 指向通过 BVGetArrayRead 获取的只读数组的指针
         */
        BVRestoreArrayRead(y, &y_array);
    }
    BVRestoreArrayRead(x, &x_array);
    return;
}

/**
 * @brief 给BV向量集合的[start, end)列范围设置随机值
 * 
 * @param x 目标向量集
 * @param start 起始列索引（包含）
 * @param end   结束列索引（不包含）
 * @param ops   操作接口
 */
static void MultiVecSetRandomValue(BV x, int start, int end, struct OPS_ *ops) {
    BVSetActiveColumns(x, start, end);

    // // 调试时使用的固定随机种子方法 start
    // PetscRandom rand;
    // PetscRandomCreate(PETSC_COMM_WORLD, &rand);
    // // 设置固定的随机种子（例如1234）
    // PetscRandomSetSeed(rand, 1234);
    // // 重新初始化随机数生成器
    // PetscRandomSeed(rand);
    // // 调试时使用的固定随机种子方法 end

    // 假设 bv 已经创建
    BVSetRandom(x);

    // // 调试时使用的固定随机种子方法 start
    // PetscRandomDestroy(&rand);
    // // 调试时使用的固定随机种子方法 end
    return;
}

/**
 * @brief 计算多列向量的线性组合：y = alpha * x + beta * y
 *
 * 对指定列范围内的向量进行BLAS风格的axpby操作，支持矩阵列向量批量处理。
 * 当beta=0时执行向量初始化+累加操作，当x=NULL时仅执行向量缩放操作。
 * 
 * @param alpha 标量系数，作用于x向量
 * @param x 输入向量/矩阵。当为NULL时仅对y执行beta缩放操作
 * @param beta 标量系数，作用于y向量
 * @param y 输入输出向量/矩阵。操作结果将直接修改此数据
 * @param start 二维数组，start[0]表示x的起始列，start[1]表示y的起始列
 * @param end 二维数组，end[0]表示x的结束列，end[1]表示y的结束列
 * @param ops 操作接口
 */
static void MultiVecAxpby(double alpha, BV x,
                          double beta, BV y, int *start, int *end, struct OPS_ *ops) {
    assert(end[0] - start[0] == end[1] - start[1]);
    PetscScalar *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;
    PetscInt x_ld, y_ld; // BV对象转换为array后的主维度
    /**
     * @brief 获取 BV 对象中基向量的可读写数组
     * @param BV 要获取数组的 BV 对象
     * @param PetscScalar ** 输出参数，指向 BV 中基向量的可读写数组 获取的数据格式是 列主序（column-major）
     */
    BVGetArray(y, &y_array);
    BVGetSizes(y, &y_nrows, NULL, &y_ncols);
    BVGetLeadingDimension(y, &y_ld);
    LAPACKVEC y_vec;
    y_vec.nrows = y_nrows;
    y_vec.ncols = y_ncols;
    y_vec.ldd = y_ld;
    y_vec.data = y_array;
    if (x == NULL) {
#if 0
       ops->lapack_ops->MultiVecAxpby(alpha,
	     NULL,beta,(void**)&y_vec,start,end,ops->lapack_ops);
#else
        BVSetActiveColumns(y, start[1], end[1]);
        /**
        * @brief 让BV 中的每个基向量的每个元素乘以 标量值beta
        * @param BV y 要缩放的 BV 对象
        * @param PetscScalar beta 缩放因子，所有基向量将乘以该标量值。
        */
        BVScale(y, beta);
#endif
    } else {
        if (x != y) {
            BVSetActiveColumns(x, start[0], end[0]);
            BVSetActiveColumns(y, start[1], end[1]);
            /**
             * @brief 对 BV 对象进行线性组合和矩阵-向量乘法的函数。
             * 将 V 和 W 进行线性组合，结果为 alpha * V + beta * W。
             * 如果 Mat A 不为 NULL，则对线性组合的结果执行矩阵-向量乘法，结果为 A * (alpha * V + beta * W)
             * @param BV y 输入的 BV 对象，包含一组基向量
             * @param alpha 线性组合的系数，作用于 y
             * @param beta 线性组合的系数，作用于 x
             * @param BV x 输入的 BV 对象，包含另一组基向量
             * @param Mat NULL 矩阵对象，用于执行矩阵-向量乘法。如果不需要此操作，可以传入 NULL
             * 
             */
            BVMult(y, alpha, beta, x, NULL);
        } else if (start[0] == start[1]) {
            // x == y 且 start[0] == start[1]场景
            BVSetActiveColumns(y, start[1], end[1]);
            BVScale(y, (alpha + beta));
        } else {
            // x == y 且 start[0] != start[1]场景
            // 确保 x 和 y 的列范围不重叠
            assert(end[0] <= start[1] || end[1] <= start[0]);
            const PetscScalar *x_array;
            LAPACKVEC x_vec;
            BVGetArrayRead(x, &x_array);
            BVGetSizes(x, &x_nrows, NULL, &x_ncols);
            BVGetLeadingDimension(x, &x_ld);
            x_vec.nrows = x_nrows;
            x_vec.ncols = x_ncols;
            x_vec.ldd = x_ld;
            x_vec.data = (double *)x_array;
            ops->lapack_ops->MultiVecAxpby(alpha,
                                           (void **)&x_vec, beta, (void **)&y_vec, start, end, ops->lapack_ops);
            BVRestoreArrayRead(x, &x_array);
        }
    }
    BVRestoreArray(y, &y_array);

    return;
}

/**
 * @brief 计算矩阵块与向量块的乘积，并将结果累加到目标向量块中 y = mat * x
 *
 * @param mat   输入矩阵指针。若为NULL，则执行向量块复制操作而非矩阵乘法
 * @param x     输入向量指针，要求数据连续存储（nrows == ldd）
 * @param y     输出向量指针，要求数据连续存储（nrows == ldd）
 * @param start 起始索引数组，start[0]为x向量的起始列，start[1]为y向量的起始列
 * @param end   结束索引数组，end[0]为x向量的结束列，end[1]为y向量的结束列
 * @param ops   操作接口
 */
static void MatDotMultiVec(Mat mat, BV x,
                           BV y, int *start, int *end, struct OPS_ *ops) {
#if DEBUG
    int n, N, m;
    if (mat != NULL) {
        MatGetSize(mat, &N, &m);
        PetscPrintf(PETSC_COMM_WORLD, "mat global, N = %d, m = %d\n", N, m);
        MatGetLocalSize(mat, &n, &m);
        PetscPrintf(PETSC_COMM_WORLD, "mat local , n = %d, m = %d\n", n, m);
    }
    BVGetSizes(x, &n, &N, &m);
    PetscPrintf(PETSC_COMM_WORLD, "x local n = %d, global N = %d, ncols = %d\n", n, N, m);
    BVGetSizes(y, &n, &N, &m);
    PetscPrintf(PETSC_COMM_WORLD, "y local n = %d, global N = %d, ncols = %d\n", n, N, m);
    ops->Printf("%d,%d, %d,%d\n", start[0], end[0], start[1], end[1]);
#endif

    assert(end[0] - start[0] == end[1] - start[1]);
    int nrows_x, nrows_y;
    BVGetSizes(x, &nrows_x, NULL, NULL);
    BVGetSizes(y, &nrows_y, NULL, NULL);

    if (nrows_x == nrows_y) {
        if (mat == NULL) {
            MultiVecAxpby(1.0, x, 0.0, y, start, end, ops);
        } else {
            // 此时mat是方阵
            /* sometimes Active does not work */
            assert(x != y);
            if (end[0] - start[0] < 5) {
                // todo：是否有bug?
                int ncols = end[1] - start[1], col;
                Vec vec_x, vec_y;
                for (col = 0; col < ncols; ++col) {
                    /**
                     * @brief 从 BV 对象中获取指定列的基向量：不会创建新的 Vec 对象，而是返回 BV 内部存储的第 j 列的 引用
                     * @param BV x 要获取列的 BV 对象
                     * @param PetscInt start[0] + col 要获取的列的索引（从 0 开始）
                     * @param Vec* &vec_x 输出参数，指向获取的基向量(引用)
                     */
                    BVGetColumn(x, start[0] + col, &vec_x);
                    BVGetColumn(y, start[1] + col, &vec_y);
                    /**
                     * @brief 执行矩阵-向量乘法操作y = A ⋅ x
                     * @param Mat A 要使用的矩阵
                     * @param Vec x 输入向量
                     * @param Vec y 输出向量
                     */
                    MatMult(mat, vec_x, vec_y);
                    /**
                     * @brief 释放通过 BVGetColumn 获取的某一列基向量
                     * @param BV x 要释放列的 BV 对象
                     * @param PetscInt start[0] + col 要释放的列的索引（从 0 开始）
                     * @param Vec* &vec_x指向通过 BVGetColumn 获取的基向量的指针
                     */
                    BVRestoreColumn(x, start[0] + col, &vec_x);
                    BVRestoreColumn(y, start[1] + col, &vec_y);
                }
            } else {
                BVSetActiveColumns(x, start[0], end[0]);
                BVSetActiveColumns(y, start[1], end[1]);
                /**
                 * @brief 执行矩阵与 BV 对象的乘法操作 W = A ⋅ V
                 * @param BV V 输入的 BV 对象，包含一组基向量
                 * @param Mat A 输入的矩阵对象
                 * @param BV W 输出的 BV 对象，存储矩阵-向量乘法的结果
                 */
                BVMatMult(x, mat, y);
            }
        }
    } else {
        assert(mat != NULL);
        Vec vec_x, vec_y;
        int ncols = end[1] - start[1], col;
        for (col = 0; col < ncols; ++col) {
            BVGetColumn(x, start[0] + col, &vec_x);
            BVGetColumn(y, start[1] + col, &vec_y);
            MatMult(mat, vec_x, vec_y);
            BVRestoreColumn(x, start[0] + col, &vec_x);
            BVRestoreColumn(y, start[1] + col, &vec_y);
        }
    }
    return;
}

/**
 * @brief 计算矩阵转置与向量的乘积，并将结果累加到目标向量中 W = A^T ⋅ V
 * 
 * @param[in] mat    输入稠密矩阵
 * @param[in] x      输入BV向量集
 * @param[out] y     输出BV向量集结果会被累加到此向量
 * @param[in] start  二维起始索引数组，start[0]为x向量的起始列，start[1]为y向量的起始列
 * @param[in] end    二维结束索引数组，end[0]为x向量的结束列，end[1]为y向量的结束列
 * @param[in] ops    操作接口
 * 
 * @warning 输入约束：
 * - 操作范围必须满足 end[0]-start[0] == end[1]-start[1]
 * - 输入/输出向量的长度必须与矩阵维度匹配（通过assert验证）
 */
static void MatTransDotMultiVec(Mat mat, BV x,
                                BV y, int *start, int *end, struct OPS_ *ops) {
    assert(end[0] - start[0] == end[1] - start[1]);
    int nrows_x, nrows_y;
    BVGetSizes(x, &nrows_x, NULL, NULL);
    BVGetSizes(y, &nrows_y, NULL, NULL);
    if (nrows_x == nrows_y) {
        if (mat == NULL) {
            MultiVecAxpby(1.0, x, 0.0, y, start, end, ops);
        } else {
            BVSetActiveColumns(x, start[0], end[0]);
            BVSetActiveColumns(y, start[1], end[1]);
            /**
             * @brief 用于执行矩阵的转置与 BV 对象的乘法操作 W=A^T ⋅ V
             * @param BV V 输入的 BV 对象，包含一组基向量
             * @param Mat A 输入的矩阵对象
             * @param BV W 输出的 BV 对象，存储矩阵转置-向量乘法的结果
             */
            BVMatMultTranspose(x, mat, y);
        }
    } else {
        Vec vec_x, vec_y;
        assert(end[0] - start[0] == end[1] - start[1]);
        int ncols = end[1] - start[1], col;
        for (col = 0; col < ncols; ++col) {
            BVGetColumn(x, start[0] + col, &vec_x);
            BVGetColumn(y, start[1] + col, &vec_y);
            /**
             * @brief 执行矩阵转置与向量的乘积操作 y = A^T ⋅ x
             * @param Mat 输入的矩阵对象
             * @param Vec 输入的向量对象
             * @param Vec 输出的向量对象，存储矩阵转置-向量乘法的结果
             */
            MatMultTranspose(mat, vec_x, vec_y);
            BVRestoreColumn(x, start[0] + col, &vec_x);
            BVRestoreColumn(y, start[1] + col, &vec_y);
        }
    }
    return;
}

/**
* @brief 执行向量/矩阵的线性组合计算 y = beta*y + x*coef
* 输入为slepc的BV结构，将其转换为LAPACKVEC结构，调用LAPACK的多向量内积计算函数计算。
* 由于并行程序中每个进程的BV结构存储部分数据，因此多进程运行时，效率高于单一进程下LAPACKVEC求解所有数据
* 
* @param[in] x       输入矩阵/向量，NULL表示不参与计算
* @param[in,out] y   输入输出矩阵/向量，存储计算结果
* @param[in] is_vec  保留参数
* @param[in] start   二维起始坐标数组，start[0]为x向量的起始列，start[1]为y向量的起始列
* @param[in] end     二维结束坐标数组，end[0]为x向量的结束列，end[1]为y向量的结束列
* @param[in] coef    系数矩阵指针，与x矩阵相乘的系数
* @param[in] ldc     coef矩阵的leading dimension
* @param[in] beta    y的缩放系数指针，NULL表示不进行缩放
* @param[in] incb    beta系数递增步长，0表示使用统一系数
* @param[in] ops     操作接口
* 
* @note 实际运算通过dgemm/dscal等BLAS函数完成
*/
static void MultiVecLinearComb(BV x, BV y, int is_vec,
                               int *start, int *end,
                               double *coef, int ldc,
                               double *beta, int incb, struct OPS_ *ops) {
    assert(is_vec == 0);
    PetscScalar *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;
    PetscInt x_ld, y_ld; // BV对象转换为array后的主维度
    BVGetArray(y, &y_array);
    BVGetSizes(y, &y_nrows, NULL, &y_ncols);
    BVGetLeadingDimension(y, &y_ld);
    LAPACKVEC y_vec;
    y_vec.nrows = y_nrows;
    y_vec.ncols = y_ncols;
    y_vec.ldd = y_ld;
    y_vec.data = (double *)y_array;
    if (x == NULL) {
        ops->lapack_ops->MultiVecLinearComb(
            NULL, (void **)&y_vec, is_vec,
            start, end, coef, ldc, beta, incb, ops->lapack_ops);
    } else {
        //assert(end[0]<=start[1]||end[1]<=start[0]);
        const PetscScalar *x_array;
        LAPACKVEC x_vec;
        BVGetArrayRead(x, &x_array);
        BVGetSizes(x, &x_nrows, NULL, &x_ncols);
        BVGetLeadingDimension(x, &x_ld);
        x_vec.nrows = x_nrows;
        x_vec.ncols = x_ncols;
        x_vec.ldd = x_ld;
        x_vec.data = (double *)x_array;
        ops->lapack_ops->MultiVecLinearComb(
            (void **)&x_vec, (void **)&y_vec, is_vec,
            start, end, coef, ldc, beta, incb, ops->lapack_ops);
        BVRestoreArrayRead(x, &x_array);
    }
    BVRestoreArray(y, &y_array);
    return;
}
/* Encapsulation */

/**
 * @brief 打印SLEPC矩阵内容到输出流
 *
 * @param[in] mat  指向Mat矩阵结构的指针
 * @param[in] ops  输出接口
 */
static void SLEPC_MatView(void *mat, struct OPS_ *ops) {
    MatView((Mat)mat, PETSC_VIEWER_STDOUT_WORLD);
    return;
}

/**
* @brief 执行矩阵的线性组合操作：y = alpha x + beta y 

* @param[in] alpha 第一个矩阵的标量系数
* @param[in] matX  指向第一个矩阵数据的指针，矩阵的具体格式由实现定义
* @param[in] beta  第二个矩阵的标量系数
* @param[in] matY  指向第二个矩阵数据的指针，矩阵维度应与matX一致
* @param[in] ops   操作接口
*/
static void SLEPC_MatAxpby(double alpha, void *matX,
                           double beta, void *matY, struct OPS_ *ops) {
    /* y = alpha x + beta y */
    if (beta == 1.0) {
        /* SAME_NONZERO_PATTERN: 结构不变，仅修改已有非零值  最快，不会重新分析结构
         * DIFFERENT_NONZERO_PATTERN: 结构发生变化（新增/删除非零）  重新分析结构，适用于变化的矩阵
         * SUBSET_NONZERO_PATTERN: 非零结构不会增加，只会减少  优化存储，但不会重新分析结构
         */
        /* y = alpha x + y */
        MatAXPY((Mat)matY, alpha, (Mat)matX, SUBSET_NONZERO_PATTERN);
    } else if (alpha == 1.0) {
        /* y = x + beta y */
        MatAYPX((Mat)matY, beta, (Mat)matX, SUBSET_NONZERO_PATTERN);
    } else {
        if (beta == 0.0) {
            MatCopy((Mat)matX, (Mat)matY, DIFFERENT_NONZERO_PATTERN);
            MatScale((Mat)matY, alpha);
        } else {
            MatAXPY((Mat)matY, (alpha - 1.0) / beta, (Mat)matX, SUBSET_NONZERO_PATTERN);
            MatAYPX((Mat)matY, beta, (Mat)matX, SUBSET_NONZERO_PATTERN);
        }
    }
    return;
}

/**
 * @brief 基于 Mat 类型的矩阵创建 BV（Block Vector）对象，并初始化其大小和设置随机值
 * 
 * @param des_bv 要创建BV对象的存储地址
 * @param num_vec BV对象的列数
 * @param src_mat BV对象长度与src_mat的行数一致
 * @param ops 未使用
 */
static void SLEPC_MultiVecCreateByMat(void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops) {
    MultiVecCreateByMat((BV *)des_vec, num_vec, (Mat)src_mat, ops);
    return;
}

/**
 * @brief 析构多维向量
 * 
 * @param des_bv 要析构对象的指针
 * @param num_vec 未用到
 * @param ops 未用到
 */
static void SLEPC_MultiVecDestroy(void ***des_vec, int num_vec, struct OPS_ *ops) {
    MultiVecDestroy((BV *)des_vec, num_vec, ops);
    return;
}

/**
 * @brief 查看多维向量BV的内容
 * 
 * @param x 要查看的多维向量BV
 * @param start 要查看的起始列索引
 * @param end 查看的列数
 * @param ops 未用到
 */
static void SLEPC_MultiVecView(void **x, int start, int end, struct OPS_ *ops) {
    MultiVecView((BV)x, start, end, ops);
    return;
}

/**
 * @brief 计算多个向量的局部内积（或块内积），结果存储在指定矩阵中
 * 输入为slepc的BV结构，将其转换为LAPACKVEC结构，调用LAPACK的多向量内积计算函数计算。
 * 由于并行程序中每个进程的BV结构存储部分数据，因此多进程运行时，效率高于单一进程下LAPACKVEC求解所有数据
 * 
 * @remark 此函数并行效率高
 * @param[in] nsdIP     字符参数，指定内积存储方式（例如'S'表示对称存储）
 * @param[in] x         输入矩阵/向量x（对应运算中的Q矩阵）
 * @param[in] y         输入矩阵/向量y（对应运算中的P矩阵）
 * @param[in] is_vec    向量模式标识
 * @param[in] start     二维数组起始索引[start[0], start[1]]
 * @param[in] end       二维数组结束索引[end[0], end[1]]
 * @param[out] inner_prod 结果存储矩阵指针（存放内积计算结果）
 * @param[in] ldIP      inner_prod矩阵的leading dimension
 * @param[in] ops       运算选项结构体
 * 
 * @return 无返回值，结果存储在inner_prod矩阵中
 */
static void SLEPC_MultiVecLocalInnerProd(char nsdIP,
                                         void **x, void **y, int is_vec, int *start, int *end,
                                         double *inner_prod, int ldIP, struct OPS_ *ops) {
    MultiVecLocalInnerProd(nsdIP,
                           (BV)x, (BV)y, is_vec, start, end,
                           inner_prod, ldIP, ops);
    return;
}

/**
 * @brief 给BV向量集合的[start, end)列范围设置随机值
 * 
 * @param x 目标向量集
 * @param start 起始列索引（包含）
 * @param end   结束列索引（不包含）
 * @param ops   操作接口
 */
static void SLEPC_MultiVecSetRandomValue(void **x, int start, int end, struct OPS_ *ops) {
    MultiVecSetRandomValue((BV)x, start, end, ops);
    return;
}

/**
 * @brief 计算多列向量的线性组合：y = alpha * x + beta * y
 *
 * 对指定列范围内的向量进行BLAS风格的axpby操作，支持矩阵列向量批量处理。
 * 当beta=0时执行向量初始化+累加操作，当x=NULL时仅执行向量缩放操作。
 * 
 * @param alpha 标量系数，作用于x向量
 * @param x 输入向量/矩阵。当为NULL时仅对y执行beta缩放操作
 * @param beta 标量系数，作用于y向量
 * @param y 输入输出向量/矩阵。操作结果将直接修改此数据
 * @param start 二维数组，start[0]表示x的起始列，start[1]表示y的起始列
 * @param end 二维数组，end[0]表示x的结束列，end[1]表示y的结束列
 * @param ops 操作接口
 */
static void SLEPC_MultiVecAxpby(double alpha, void **x,
                                double beta, void **y, int *start, int *end, struct OPS_ *ops) {
    MultiVecAxpby(alpha, (BV)x, beta, (BV)y, start, end, ops);
    return;
}

/**
 * @brief 计算矩阵块与向量块的乘积，并将结果累加到目标向量块中 y = mat * x
 *
 * @param mat   输入矩阵指针。若为NULL，则执行向量块复制操作而非矩阵乘法
 * @param x     输入向量指针，要求数据连续存储（nrows == ldd）
 * @param y     输出向量指针，要求数据连续存储（nrows == ldd）
 * @param start 起始索引数组，start[0]为x向量的起始列，start[1]为y向量的起始列
 * @param end   结束索引数组，end[0]为x向量的结束列，end[1]为y向量的结束列
 * @param ops   操作接口
 */
static void SLEPC_MatDotMultiVec(void *mat, void **x,
                                 void **y, int *start, int *end, struct OPS_ *ops) {
    MatDotMultiVec((Mat)mat, (BV)x, (BV)y, start, end, ops);
    return;
}

/**
 * @brief 计算矩阵转置与向量的乘积，并将结果累加到目标向量中 W = A^T ⋅ V
 * 
 * @param[in] mat    输入稠密矩阵
 * @param[in] x      输入BV向量集
 * @param[out] y     输出BV向量集结果会被累加到此向量
 * @param[in] start  二维起始索引数组，start[0]为x向量的起始列，start[1]为y向量的起始列
 * @param[in] end    二维结束索引数组，end[0]为x向量的结束列，end[1]为y向量的结束列
 * @param[in] ops    操作接口
 * 
 * @warning 输入约束：
 * - 操作范围必须满足 end[0]-start[0] == end[1]-start[1]
 * - 输入/输出向量的长度必须与矩阵维度匹配（通过assert验证）
 */
static void SLEPC_MatTransDotMultiVec(void *mat, void **x,
                                      void **y, int *start, int *end, struct OPS_ *ops) {
    MatTransDotMultiVec((Mat)mat, (BV)x, (BV)y, start, end, ops);
    return;
}

static void SLEPC_MultiGridCreate(void ***A_array, void ***B_array, void ***P_array,
                                  int *num_levels, void *A, void *B, struct OPS_ *ops) {
    /* P 是行多列少, Px 是从粗到细 */
    PetscInt m, n, level;
    Mat *petsc_A_array = NULL, *petsc_B_array = NULL, *petsc_P_array = NULL;
    PC pc;
    Mat *Aarr = NULL, *Parr = NULL;

    PCCreate(PETSC_COMM_WORLD, &pc);
    PCSetOperators(pc, (Mat)A, (Mat)A);
    PCSetType(pc, PCGAMG);
    //PCGAMGSetType(pc,PCGAMGAGG);
    PCGAMGSetType(pc, PCGAMGCLASSICAL);
    PetscPrintf(PETSC_COMM_WORLD, "num_levels = %d\n", *num_levels);
    PCGAMGSetNlevels(pc, *num_levels);
    /* not force coarse grid onto one processor */
    //PCGAMGSetUseParallelCoarseGridSolve(pc,PETSC_TRUE);
    /* this will generally improve the loading balancing of the work on each level 
	 * should use parmetis */
    //   PCGAMGSetRepartition(pc, PETSC_TRUE);
    //	type 	- PCGAMGAGG, PCGAMGGEO, or PCGAMGCLASSICAL
    /* Increasing the threshold decreases the rate of coarsening. 
	 * 0.0 means keep all nonzero entries in the graph; 
	 * negative means keep even zero entries in the graph */
    //PCGAMGSetThresholdScale(pc, 0.5);
    PetscReal th[16] = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
    PCGAMGSetThreshold(pc, th, 16);
    //stop coarsening once the coarse grid has less than <100000> unknowns.
    //PCGAMGSetCoarseEqLim(pc, 50000);
    //there are around <1000> equations on each process
    //PCGAMGSetProcEqLim(pc, 1000);
    PetscPrintf(PETSC_COMM_WORLD, "before PCGAMG SetUp\n");
    PCSetUp(pc);
    PetscPrintf(PETSC_COMM_WORLD, "after  PCGAMG SetUp\n");
    /* the size of Aarr is num_levels-1, Aarr[0] is the coarsest matrix */
    PCGetCoarseOperators(pc, num_levels, &Aarr);
    PetscPrintf(PETSC_COMM_WORLD, "num_levels = %d\n", *num_levels);
    /* the size of Parr is num_levels-1 */
    PCGetInterpolations(pc, num_levels, &Parr);
    /* we should make that zero is the refinest level */
    /* when num_levels == 5, 1 2 3 4 of A_array == 3 2 1 0 of Aarr */
    petsc_A_array = malloc(sizeof(Mat) * (*num_levels));
    petsc_P_array = malloc(sizeof(Mat) * ((*num_levels) - 1));
    petsc_A_array[0] = (Mat)A;
    MatGetSize(petsc_A_array[0], &m, &n);
    PetscPrintf(PETSC_COMM_WORLD, "A_array[%d], m = %d, n = %d\n", 0, m, n);
    for (level = 1; level < (*num_levels); ++level) {
        petsc_A_array[level] = Aarr[(*num_levels) - level - 1];
        MatGetSize(petsc_A_array[level], &m, &n);
        PetscPrintf(PETSC_COMM_WORLD, "A_array[%d], m = %d, n = %d\n", level, m, n);

        petsc_P_array[level - 1] = Parr[(*num_levels) - level - 1];
        MatGetSize(petsc_P_array[level - 1], &m, &n);
        PetscPrintf(PETSC_COMM_WORLD, "P_array[%d], m = %d, n = %d\n", level - 1, m, n);
    }
    (*A_array) = (void **)petsc_A_array;
    (*P_array) = (void **)petsc_P_array;

    PetscFree(Aarr);
    PetscFree(Parr);
    PCDestroy(&pc);

    if (B != NULL) {
        petsc_B_array = malloc(sizeof(Mat) * (*num_levels));
        petsc_B_array[0] = (Mat)B;
        MatGetSize(petsc_B_array[0], &m, &n);
        PetscPrintf(PETSC_COMM_WORLD, "B_array[%d], m = %d, n = %d\n", 0, m, n);
        /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
        for (level = 1; level < (*num_levels); ++level) {
            MatPtAP(petsc_B_array[level - 1], petsc_P_array[level - 1],
                    MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_B_array[level]));
            MatGetSize(petsc_B_array[level], &m, &n);
            PetscPrintf(PETSC_COMM_WORLD, "B_array[%d], m = %d, n = %d\n", level, m, n);
        }
        (*B_array) = (void **)petsc_B_array;
    }
    return;
}
static void SLEPC_MultiGridDestroy(void ***A_array, void ***B_array, void ***P_array,
                                   int *num_levels, struct OPS_ *ops) {
    Mat *petsc_A_array, *petsc_B_array, *petsc_P_array;
    petsc_A_array = (Mat *)(*A_array);
    petsc_P_array = (Mat *)(*P_array);
    int level;
    for (level = 1; level < (*num_levels); ++level) {
        MatDestroy(&(petsc_A_array[level]));
        MatDestroy(&(petsc_P_array[level - 1]));
    }
    free(petsc_A_array);
    free(petsc_P_array);
    (*A_array) = NULL;
    (*P_array) = NULL;

    if (B_array != NULL) {
        petsc_B_array = (Mat *)(*B_array);
        for (level = 1; level < (*num_levels); ++level) {
            MatDestroy(&(petsc_B_array[level]));
        }
        free(petsc_B_array);
        (*B_array) = NULL;
    }
    return;
}

/**
* @brief 执行向量/矩阵的线性组合计算 y = beta*y + x*coef
* 输入为slepc的BV结构，将其转换为LAPACKVEC结构，调用LAPACK的多向量内积计算函数计算。
* 由于并行程序中每个进程的BV结构存储部分数据，因此多进程运行时，效率高于单一进程下LAPACKVEC求解所有数据
* 
* @param[in] x       输入矩阵/向量，NULL表示不参与计算
* @param[in,out] y   输入输出矩阵/向量，存储计算结果
* @param[in] is_vec  保留参数
* @param[in] start   二维起始坐标数组，start[0]为x向量的起始列，start[1]为y向量的起始列
* @param[in] end     二维结束坐标数组，end[0]为x向量的结束列，end[1]为y向量的结束列
* @param[in] coef    系数矩阵指针，与x矩阵相乘的系数
* @param[in] ldc     coef矩阵的leading dimension
* @param[in] beta    y的缩放系数指针，NULL表示不进行缩放
* @param[in] incb    beta系数递增步长，0表示使用统一系数
* @param[in] ops     操作接口
* 
* @note 实际运算通过dgemm/dscal等BLAS函数完成
*/
static void SLEPC_MultiVecLinearComb(
    void **x, void **y, int is_vec,
    int *start, int *end,
    double *coef, int ldc,
    double *beta, int incb, struct OPS_ *ops) {
    //assert(x!=y);
    MultiVecLinearComb(
        (BV)x, (BV)y, is_vec,
        start, end,
        coef, ldc,
        beta, incb, ops);
    return;
}

/**
 * @brief 计算密集矩阵运算qAp = Q^T * A * P 或相关变体，结果存入 qAp 数组
 * 
 * @param ntsA         矩阵A的存储标识符，'S'表示对称矩阵（自动转为'L'下三角），其他字符见DenseMatQtAP说明
 * @param nsdQAP       运算模式标识符，'T'表示需要对结果进行特殊转置存储
 * @param mvQ          输入矩阵Q的向量结构体指针
 * @param matA         输入矩阵A的结构体指针（可为NULL）
 * @param mvP          输入矩阵P的向量结构体指针
 * @param is_vec       指示是否为向量
 * @param start        二维数组指针，指定列起始索引[start[0], start[1]]
 * @param end          二维数组指针，指定列结束索引[end[0], end[1]]
 * @param qAp          输出结果存储数组指针
 * @param ldQAP        qAp数组的行主维度（leading dimension）
 * @param mv_ws        工作空间向量指针，用于临时存储
 * @param ops          操作接口
 */
static void SLEPC_MultiVecQtAP(char ntsA, char nsdQAP,
                               void **mvQ, void *matA, void **mvP, int is_vec,
                               int *start, int *end, double *qAp, int ldQAP,
                               void **mv_ws, struct OPS_ *ops) {
    assert(nsdQAP != 'T');
    assert(is_vec == 0); // Q 和 P 必须是矩阵
    if (nsdQAP == 'D' || (mvQ == mvP && (start[0] != start[1] || end[0] != end[1]))) {
        DefaultMultiVecQtAP(ntsA, nsdQAP,
                            mvQ, matA, mvP, is_vec,
                            start, end, qAp, ldQAP,
                            mv_ws, ops);
    } else {
        BVSetActiveColumns((BV)mvQ, start[0], end[0]);
        BVSetActiveColumns((BV)mvP, start[1], end[1]);
        /**
         * @brief 将 BV 对象与一个矩阵关联起来，关联后，BV 对象可以使用该矩阵进行矩阵-向量乘法操作。
         * @param BV 要设置矩阵的 BV 对象
         * @param Mat 要与 BV 关联的矩阵对象
         * @param PetscBool 是否使用矩阵的转置。如果为 PETSC_TRUE，则使用矩阵的转置；如果为 PETSC_FALSE，则使用矩阵本身。
         */
        BVSetMatrix((BV)mvP, (Mat)matA, PETSC_FALSE);
        BVSetMatrix((BV)mvQ, (Mat)matA, PETSC_FALSE);
        Mat dense_mat;
        const double *source;
        int nrows = end[0] - start[0], ncols = end[1] - start[1], col;
        /**
         * @brief 创建一个顺序密集矩阵。顺序密集矩阵是指存储在单个进程中的密集矩阵，适用于小规模问题或不需要并行计算的场景。
         * @param MPI_Comm MPI通信子 PETSC_COMM_SELF：表示单进程计算
         * @param PetscInt 矩阵的行数
         * @param PetscInt 矩阵的列数
         * @param PetscScalar[] 用于存储矩阵数据的数组，如果传入 NULL，PETSc 会为矩阵分配内存。
         * @param Mat * 输出参数，指向创建的矩阵对象。
         */
        MatCreateSeqDense(PETSC_COMM_SELF, end[0], end[1], NULL, &dense_mat);
        /* Qt A P */
        /* M must be a sequential dense Mat with dimensions m,n at least, 
		 * where m is the number of active columns of Q 
		 * and n is the number of active columns of P. 
		 * Only rows (resp. columns) of M starting from ly (resp. lx) are computed, 
		 * where ly (resp. lx) is the number of leading columns of Q (resp. P). */
        /**
         * @brief 用于计算两个 BV 对象的内积 M = X^T ⋅ Y, 若向量关联矩阵后，则计算qAp = Q^T * A * P
         * @param BV X 第一个 BV 对象
         * @param BV Y 第二个 BV 对象
         * @param Mat M 输出参数，存储内积结果的矩阵。矩阵的维度为 X 的列数 × Y 的列数。
         */
        BVDot((BV)mvP, (BV)mvQ, dense_mat);
        /**
         * @brief 用于获取密集矩阵的只读数组。密集矩阵是指以二维数组形式存储的矩阵。通过该函数，可以访问矩阵的元素数据，但不能修改这些数据。
         * @param Mat 输入的密集矩阵对象。
         * @param const PetscScalar *[] 输出参数，指向矩阵的只读数组
         */
        MatDenseGetArrayRead(dense_mat, &source);
        /* 当 qAp 连续存储 */
#if DEBUG
        int row;
        ops->Printf("(%d, %d), (%d, %d)\n", start[0], end[0], start[1], end[1]);
        for (row = 0; row < end[0]; ++row) {
            for (col = 0; col < end[1]; ++col) {
                ops->Printf("%6.4e\t", source[end[0] * col + row]);
            }
            ops->Printf("%\n");
        }
#endif
        if (start[0] == 0 && ldQAP == nrows) {
            memcpy(qAp, source + nrows * start[1], nrows * ncols * sizeof(double));
        } else {
            for (col = 0; col < ncols; ++col) {
                memcpy(qAp + ldQAP * col, source + end[0] * (start[1] + col) + start[0], nrows * sizeof(double));
            }
        }
        MatDenseRestoreArrayRead(dense_mat, &source);
        MatDestroy(&dense_mat);
    }
    return;
}

/**
 * @brief 计算多个向量的内积:dense_mat=x^T ⋅ y
 * 
 * @param[in] nsdIP     字符参数，指定内积存储方式（例如'S'表示对称存储）
 * @param[in] x         输入向量/矩阵
 * @param[in] y         输入向量/矩阵
 * @param[in] is_vec    向量模式标志位（0-矩阵模式，1-向量模式）
 * @param[in] start     计算区间的起始索引数组
 * @param[in] end       计算区间的结束索引数组
 * @param[out] inner_prod 输出内积结果数组（需预先分配内存）
 * @param[in] ldIP      inner_prod数组的leading dimension
 * @param[in] ops       运算控制参数结构体指针
 * 
 */
static void SLEPC_MultiVecInnerProd(char nsdIP, void **x, void **y, int is_vec, int *start, int *end,
                                    double *inner_prod, int ldIP, struct OPS_ *ops) {
    if (nsdIP == 'D' || (x == y && (start[0] != start[1] || end[0] != end[1]))) {
        DefaultMultiVecInnerProd(nsdIP, x, y, is_vec, start, end,
                                 inner_prod, ldIP, ops);
    } else {
        BVSetActiveColumns((BV)x, start[0], end[0]);
        BVSetActiveColumns((BV)y, start[1], end[1]);
        BVSetMatrix((BV)y, NULL, PETSC_FALSE);
        BVSetMatrix((BV)x, NULL, PETSC_FALSE);
        Mat dense_mat;
        const double *source;
        int nrows = end[0] - start[0], ncols = end[1] - start[1], col;
        MatCreateSeqDense(PETSC_COMM_SELF, end[0], end[1], NULL, &dense_mat);
        BVDot((BV)y, (BV)x, dense_mat);
        MatDenseGetArrayRead(dense_mat, &source);
#if DEBUG
        int row;
        for (row = 0; row < end[0]; ++row) {
            for (col = 0; col < end[1]; ++col) {
                ops->Printf("%6.4e\t", source[end[0] * col + row]);
            }
            ops->Printf("%\n");
        }
#endif
        /* 当 inner_prod 连续存储 */
        if (start[0] == 0 && ldIP == nrows) {
            memcpy(inner_prod, source + nrows * start[1], nrows * ncols * sizeof(double));
        } else {
            for (col = 0; col < ncols; ++col) {
                memcpy(inner_prod + ldIP * col, source + end[0] * (start[1] + col) + start[0], nrows * sizeof(double));
            }
        }
        MatDenseRestoreArrayRead(dense_mat, &source);
        MatDestroy(&dense_mat);
    }
    return;
}

// 获取命令行参数
static int SLEPC_GetOptionFromCommandLine(
    const char *name, char type, void *value,
    int argc, char *argv[], struct OPS_ *ops) {
    PetscBool set;
    int *int_value;
    double *dbl_value;
    char *str_value;
    switch (type) {
    case 'i':
        int_value = (int *)value;
        PetscOptionsGetInt(NULL, NULL, name, int_value, &set);
        break;
    case 'f':
        dbl_value = (double *)value;
        PetscOptionsGetReal(NULL, NULL, name, dbl_value, &set);
        break;
    case 's':
        str_value = (char *)value;
        PetscOptionsGetString(NULL, NULL, name, str_value, 8, &set);
        //set = DefaultGetOptionFromCommandLine(name, type, value, argc, argv, ops);
        break;
    default:
        break;
    }
    return set;
}

void OPS_SLEPC_Set(struct OPS_ *ops) {
    ops->GetOptionFromCommandLine = SLEPC_GetOptionFromCommandLine;
    /* mat */
    ops->MatAxpby = SLEPC_MatAxpby;
    ops->MatView = SLEPC_MatView;
    /* multi-vec */
    ops->MultiVecCreateByMat = SLEPC_MultiVecCreateByMat;
    ops->MultiVecDestroy = SLEPC_MultiVecDestroy;
    ops->MultiVecView = SLEPC_MultiVecView;
    ops->MultiVecLocalInnerProd = SLEPC_MultiVecLocalInnerProd;
    ops->MultiVecSetRandomValue = SLEPC_MultiVecSetRandomValue;
    ops->MultiVecAxpby = SLEPC_MultiVecAxpby;
    ops->MatDotMultiVec = SLEPC_MatDotMultiVec;
    ops->MatTransDotMultiVec = SLEPC_MatTransDotMultiVec;
    ops->MultiVecLinearComb = SLEPC_MultiVecLinearComb;
    if (0) { // no efficiency
        ops->MultiVecQtAP = SLEPC_MultiVecQtAP;
        ops->MultiVecInnerProd = SLEPC_MultiVecInnerProd;
    }
    /* multi grid */
    ops->MultiGridCreate = SLEPC_MultiGridCreate;
    ops->MultiGridDestroy = SLEPC_MultiGridDestroy;
    return;
}

/**
 * @brief 
 *    nbigranks = ((PetscInt)((((PetscReal)size)*proc_rate[level])/((PetscReal)unit))) * (unit);
 *    if (nbigranks < unit) nbigranks = unit<size?unit:size;
 *
 * @param petsc_A_array
 * @param petsc_B_array
 * @param petsc_P_array
 * @param num_levels
 * @param proc_rate
 * @param unit           保证每层nbigranks是unit的倍数
 */
void PETSC_RedistributeDataOfMultiGridMatrixOnEachProcess(
    Mat *petsc_A_array, Mat *petsc_B_array, Mat *petsc_P_array,
    PetscInt num_levels, PetscReal *proc_rate, PetscInt unit) {
    PetscMPIInt rank, size;
    //PetscViewer   viewer;

    PetscInt level, row;
    Mat new_P_H;
    PetscMPIInt nbigranks;
    PetscInt global_nrows, global_ncols;
    PetscInt local_nrows, local_ncols;
    PetscInt new_local_ncols;
    /* 保证每层nbigranks是unit的倍数 */
    PetscInt rstart, rend, ncols;
    const PetscInt *cols;
    const PetscScalar *vals;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    if (proc_rate[0] <= 1.0 && proc_rate[0] > 0.0) {
        PetscPrintf(PETSC_COMM_WORLD, "Warning the refinest matrix cannot be redistributed\n");
    }

    /* 不改变最细层的进程分布 */
    MPI_Comm_dup(PETSC_COMM_WORLD, &MG_COMM[0][0]);
    MG_COMM[0][1] = MPI_COMM_NULL;
    MG_INTERCOMM[0] = MPI_COMM_NULL;
    MG_COMM_COLOR[0] = 0;
    for (level = 1; level < num_levels; ++level) {
        MatGetSize(petsc_P_array[level - 1], &global_nrows, &global_ncols);
        /* 在设定new_P_H的局部行时已经不能用以前P的局部行，因为当前层的A可能已经改变 */
        MatGetLocalSize(petsc_A_array[level - 1], &local_nrows, &local_ncols);
        /* 应该通过ncols_P，即最粗层矩阵大小和进程总数size确定nbigranks */
        nbigranks = ((PetscInt)((((PetscReal)size) * proc_rate[level]) / ((PetscReal)unit))) * (unit);
        if (nbigranks < unit) nbigranks = unit < size ? unit : size;
        /* 若proc_rate设为(0,1)之外，则不进行数据重分配/ */
        if (proc_rate[level] > 1.0 || proc_rate[level] <= 0.0 || nbigranks >= size || nbigranks <= 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Retain data distribution of %D level\n", level);
            /* 创建分层矩阵的通信域 */
            MG_COMM_COLOR[level] = 0;
            /* TODO: 是否可以直接赋值
             * MG_COMM[level][0] = PETSC_COMM_WORLD */
            MPI_Comm_dup(PETSC_COMM_WORLD, &MG_COMM[level][0]);
            MG_COMM[level][1] = MPI_COMM_NULL;
            MG_INTERCOMM[level] = MPI_COMM_NULL;
            continue; /* 直接到下一次循环 */
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "Redistribute data of %D level\n", level);
            PetscPrintf(PETSC_COMM_WORLD, "nbigranks[%D] = %D\n", level, nbigranks);
        }
        /* 上面的判断已经保证 0 < nbigranks < size */
        /* 创建分层矩阵的通信域 */
        int comm_color, local_leader, remote_leader;
        /* 对0到nbigranks-1进程平均分配global_ncols */
        new_local_ncols = 0;
        if (rank < nbigranks) {
            new_local_ncols = global_ncols / nbigranks;
            if (rank < global_ncols % nbigranks) {
                ++new_local_ncols;
            }
            comm_color = 0;
            local_leader = 0;
            remote_leader = nbigranks;
        } else {
            comm_color = 1;
            local_leader = 0; /* 它的全局进程号是nbigranks */
            remote_leader = 0;
        }
        /* 在不同进程中MG_COMM_COLOR[level]是不一样的值，它表征该进程属于哪个通讯域 */
        MG_COMM_COLOR[level] = comm_color;
        /* 分成两个子通讯域, MG_COMM[level][0]从0~(nbigranks-1)
         * MG_COMM[level][0]从nbigranks~(size-1) */
        MPI_Comm_split(PETSC_COMM_WORLD, comm_color, rank, &MG_COMM[level][comm_color]);
        MPI_Intercomm_create(MG_COMM[level][comm_color], local_leader,
                             PETSC_COMM_WORLD, remote_leader, level, &MG_INTERCOMM[level]);

        int aux_size = -1, aux_rank = -1;
        MPI_Comm_rank(MG_COMM[level][comm_color], &aux_rank);
        MPI_Comm_size(MG_COMM[level][comm_color], &aux_size);
        PetscPrintf(PETSC_COMM_SELF, "aux %D/%D, global %D/%D\n",
                    aux_rank, aux_size, rank, size);

        /* 创建新的延拓矩阵, 并用原始的P为之赋值
         * 新的P与原来的P只有 局部列数new_local_ncols 不同 */
        MatCreate(PETSC_COMM_WORLD, &new_P_H);
        MatSetSizes(new_P_H, local_nrows, new_local_ncols, global_nrows, global_ncols);
        //MatSetFromOptions(new_P_H);
        /* can be improved */
        //MatSeqAIJSetPreallocation(new_P_H, 5, NULL);
        //MatMPIAIJSetPreallocation(new_P_H, 3, NULL, 2, NULL);
        MatSetUp(new_P_H);
        MatGetOwnershipRange(petsc_P_array[level - 1], &rstart, &rend);
        for (row = rstart; row < rend; ++row) {
            MatGetRow(petsc_P_array[level - 1], row, &ncols, &cols, &vals);
            MatSetValues(new_P_H, 1, &row, ncols, cols, vals, INSERT_VALUES);
            MatRestoreRow(petsc_P_array[level - 1], row, &ncols, &cols, &vals);
        }
        MatAssemblyBegin(new_P_H, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(new_P_H, MAT_FINAL_ASSEMBLY);

        MatGetLocalSize(petsc_P_array[level - 1], &local_nrows, &local_ncols);
        PetscPrintf(PETSC_COMM_SELF, "[%D] original P_H[%D] local size %D * %D\n",
                    rank, level, local_nrows, local_ncols);
        MatGetLocalSize(new_P_H, &local_nrows, &local_ncols);
        PetscPrintf(PETSC_COMM_SELF, "[%D] new P_H[%D] local size %D * %D\n",
                    rank, level, local_nrows, local_ncols);
        //MatView(petsc_P_array[level-1], viewer);
        //MatView(new_P_H, viewer);
        /* 销毁之前的P_H A_H B_H */
        MatDestroy(&(petsc_P_array[level - 1]));
        MatDestroy(&(petsc_A_array[level]));
        if (petsc_B_array != NULL) {
            MatDestroy(&(petsc_B_array[level]));
        }

        petsc_P_array[level - 1] = new_P_H;
        MatPtAP(petsc_A_array[level - 1], petsc_P_array[level - 1],
                MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_A_array[level]));
        if (petsc_B_array != NULL) {
            MatPtAP(petsc_B_array[level - 1], petsc_P_array[level - 1],
                    MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_B_array[level]));
        }
        //MatView(petsc_A_array[num_levels-1], viewer);
        //MatView(petsc_B_array[num_levels-1], viewer);
        /* 这里需要修改petsc_P_array[level], 原因是
       	 * petsc_A_array[level]修改后，
      	 * 它利用原来的petsc_P_array[level]插值上来的向量已经与petsc_A_array[level]不匹配
      	 * 所以在不修改level+1层的分布结构的情况下，需要对petsc_P_array[level]进行修改 */
        /* 如果当前层不是最粗层，并且，下一层也不进行数据重分配 */
        if (level + 1 < num_levels && (proc_rate[level + 1] > 1.0 || proc_rate[level + 1] <= 0.0)) {
            MatGetSize(petsc_P_array[level], &global_nrows, &global_ncols);
            /*需要当前层A的列 作为P的行 */
            MatGetLocalSize(petsc_A_array[level], &new_local_ncols, &local_ncols);
            /*需要下一层A的行 作为P的列 */
            MatGetLocalSize(petsc_A_array[level + 1], &local_nrows, &new_local_ncols);
            /* 创建新的延拓矩阵, 并用原始的P为之赋值 */
            MatCreate(PETSC_COMM_WORLD, &new_P_H);
            MatSetSizes(new_P_H, local_ncols, local_nrows, global_nrows, global_ncols);
            //MatSetFromOptions(new_P_H);
            /* can be improved */
            //MatSeqAIJSetPreallocation(new_P_H, 5, NULL);
            //MatMPIAIJSetPreallocation(new_P_H, 3, NULL, 2, NULL);
            MatSetUp(new_P_H);
            MatGetOwnershipRange(petsc_P_array[level], &rstart, &rend);
            for (row = rstart; row < rend; ++row) {
                MatGetRow(petsc_P_array[level], row, &ncols, &cols, &vals);
                MatSetValues(new_P_H, 1, &row, ncols, cols, vals, INSERT_VALUES);
                MatRestoreRow(petsc_P_array[level], row, &ncols, &cols, &vals);
            }
            MatAssemblyBegin(new_P_H, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(new_P_H, MAT_FINAL_ASSEMBLY);
            /* 销毁原始的 P_H */
            MatDestroy(&(petsc_P_array[level]));
            petsc_P_array[level] = new_P_H;
        }
    }
    return;
}

#endif
