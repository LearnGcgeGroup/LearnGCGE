/**
 * @brief 基于CCS（压缩列存储）格式的矩阵和向量操作
 */

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include "app_ccs.h"

/**
 * @brief 查看稀疏矩阵数据内容并进行可视化输出
 * 
 * 该函数通过创建临时多维向量结构，将稀疏矩阵数据按列填充到多维向量中，
 * 最后调用底层线性代数库的可视化接口进行矩阵内容查看
 * 
 * @param[in] mat  指向CCSMAT稀疏矩阵结构的指针，包含矩阵维度、列指针、行索引及数据数组
 * @param[in] ops  操作函数集合指针，提供多维向量创建/销毁及LAPACK相关操作接口
 * 
 * @return 无返回值
 */
static void MatView(CCSMAT *mat, struct OPS_ *ops) {
    LAPACKVEC *multi_vec;
    /* 创建与矩阵列数相同的多维向量用于临时存储矩阵数据 */
    ops->MultiVecCreateByMat((void ***)(&multi_vec), mat->ncols, mat, ops);
    int col, i;
    double *destin;
    /* 按列遍历稀疏矩阵数据 */
    for (col = 0; col < mat->ncols; ++col) {
        /* 将当前列的非零元素填充到多维向量的对应列中 */
        for (i = mat->j_col[col]; i < mat->j_col[col + 1]; ++i) {
            /* 计算目标地址：基地址 + 列偏移 + 行偏移 */
            destin = multi_vec->data + (multi_vec->ldd) * col + mat->i_row[i];
            *destin = mat->data[i];
        }
    }
    /* 调用LAPACK层的矩阵可视化接口 */
    ops->lapack_ops->MatView((void *)multi_vec, ops->lapack_ops);
    /* 销毁临时创建的多维向量 */
    ops->lapack_ops->MultiVecDestroy((void ***)(&multi_vec), mat->ncols, ops->lapack_ops);
    return;
}
/* multi-vec */

/**
 * @brief 根据稀疏矩阵创建LAPACK多向量结构
 *
 * @details
 * 该函数为稀疏矩阵的列向量创建对应的LAPACK向量存储结构，包含以下操作：
 * 1. 分配主结构体内存
 * 2. 根据源矩阵列数和参数设置向量维度
 * 3. 分配连续存储空间并初始化为零值
 * 
 * @param[out] des_vec 二级指针，用于接收新创建的LAPACK向量结构
 * @param[in]  num_vec 需要创建的向量数量（对应矩阵列数）
 * @param[in]  src_mat 源稀疏矩阵(CCS格式)，用于获取列维度信息
 * @param[in]  ops     操作控制结构（当前函数暂未使用）
 * 
 * @note 数据存储采用行主序(ROW-MAJOR)布局，ldd参数等于行数
 * @warning 调用者需负责后续内存释放，避免内存泄漏
 */
static void MultiVecCreateByMat(LAPACKVEC **des_vec, int num_vec, CCSMAT *src_mat, struct OPS_ *ops) {
    /* 主结构体内存分配 */
    (*des_vec) = malloc(sizeof(LAPACKVEC));
    /* 维度参数初始化 */
    (*des_vec)->nrows = src_mat->ncols;
    (*des_vec)->ncols = num_vec;
    (*des_vec)->ldd = (*des_vec)->nrows;
    /* 数据空间初始化 */
    (*des_vec)->data = malloc(((*des_vec)->ldd) * ((*des_vec)->ncols) * sizeof(double));
    memset((*des_vec)->data, 0, ((*des_vec)->ldd) * ((*des_vec)->ncols) * sizeof(double));
    return;
}

/**
 * @brief 执行稀疏矩阵与多向量的乘法运算
 * 
 * @param[in] mat   输入的CCS格式稀疏矩阵指针，若为NULL则调用向量运算
 * @param[in] x     输入向量组指针，存储右乘向量
 * @param[out] y    输出向量组指针，存储计算结果
 * @param[in] start 起始索引数组：[x起始索引, y起始索引]（包含）
 * @param[in] end   结束索引数组：[x结束索引, y结束索引]（不包含）
 * @param[in] ops   操作接口结构体指针，提供外部函数调用
 * 
 * @note 实现特点：
 * - 支持Intel MKL加速和非MKL原生实现两种路径
 * - 支持OpenMP多线程并行加速
 * - 当mat=NULL时调用ops->lapack_ops->MultiVecAxpby
 * 
 * @warning 输入校验：
 * - start/end索引差必须相等（assert(end[0]-start[0] == end[1]-start[1])）
 * - 向量必须连续存储（assert(y->nrows == y->ldd 且 x->nrows == x->ldd)）
 */
static void MatDotMultiVec(CCSMAT *mat, LAPACKVEC *x,
                           LAPACKVEC *y, int *start, int *end, struct OPS_ *ops) {
    /* 核心参数校验 */
    assert(end[0] - start[0] == end[1] - start[1]);
    assert(y->nrows == y->ldd); // 要求必须为连续存储
    assert(x->nrows == x->ldd);
    /* 实际处理的向量数量 */
    int num_vec = end[0] - start[0];
    int col;
    if (mat != NULL) {
#if OPS_USE_INTEL_MKL
        /* MKL加速路径 */
        sparse_matrix_t csrA;
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        /*
	 * sparse_status_t mkl_sparse_d_create_csr (
	 *       sparse_matrix_t *A,  
	 *       const sparse_index_base_t indexing,  
	 *       const MKL_INT rows,  const MKL_INT cols,  
	 *       MKL_INT *rows_start,  MKL_INT *rows_end,  MKL_INT *col_indx,  double *values);
	 * sparse_status_t mkl_sparse_destroy (sparse_matrix_t A);
	 * sparse_status_t mkl_sparse_d_mm (
	 *       const sparse_operation_t operation,  
	 *       const double alpha,  
	 *       const sparse_matrix_t A,  const struct matrix_descr descr,  const sparse_layout_t layout,  
	 *       const double *B,  const MKL_INT columns,  const MKL_INT ldb,  
	 *       const double beta,  double *C,  const MKL_INT ldc);
	 */

        /* in process */
        /* CCS转CSR格式说明：
         * - j_col数组同时作为行起始和结束指针（j_col+1为行结束偏移）
         * - 矩阵维度转换：ncols/nrows交换 */
        mkl_sparse_d_create_csr(
            &csrA,
            SPARSE_INDEX_BASE_ZERO,
            mat->ncols, mat->nrows,
            mat->j_col, mat->j_col + 1, mat->i_row, mat->data);
#if OPS_USE_OMP
        /* OpenMP动态分块策略：处理余数分配逻辑 */
#pragma omp parallel num_threads(OMP_NUM_THREADS)
        {
            /* 线程负载计算 */
            int id, length, offset;
            id = omp_get_thread_num();
            length = num_vec / OMP_NUM_THREADS;
            offset = length * id;
            /* 余数分配处理 */
            if (id < num_vec % OMP_NUM_THREADS) {
                ++length;
                offset += id;
            } else {
                offset += num_vec % OMP_NUM_THREADS;
            }
            /* ���� mat �ǶԳƾ���, ���� SPARSE_OPERATION_NON_TRANSPOSE ��Ϊ SPARSE_OPERATION_TRANSPOSE */
            /* MKL稀疏矩阵多向量乘法：
             * - SPARSE_LAYOUT_COLUMN_MAJOR：向量按列存储
             * - 数据指针偏移计算：start[0]+offset 为x向量起始位置 */
            mkl_sparse_d_mm(
                SPARSE_OPERATION_NON_TRANSPOSE,
                1.0,
                csrA, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
                x->data + (start[0] + offset) * x->ldd, length, x->ldd,
                0.0, y->data + (start[1] + offset) * y->ldd, y->ldd);
        }
#else
        /* ���� mat �ǶԳƾ���, ���� SPARSE_OPERATION_NON_TRANSPOSE ��Ϊ SPARSE_OPERATION_TRANSPOSE */
        /* 单线程执行完整向量组计算 */
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            1.0,
            csrA, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
            x->data + start[0] * x->ldd, num_vec, x->ldd,
            0.0, y->data + start[1] * y->ldd, y->ldd);
#endif
        mkl_sparse_destroy(csrA);

#else
        /* 非MKL实现路径 */
        memset(y->data + (y->ldd) * start[1], 0, (y->ldd) * num_vec * sizeof(double));
#if OPS_USE_OMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
        /* 手动实现矩阵向量乘：
         * - 遍历矩阵每列的非零元素
         * - 对每个向量执行乘累加操作 */
        for (col = 0; col < num_vec; ++col) {
            int i, j;
            /* 数据指针初始化 */
            double *dm, *dx, *dy;
            int *i_row;
            dm = mat->data;
            i_row = mat->i_row;
            /* 向量地址计算 */
            dx = x->data + (x->ldd) * (start[0] + col);
            dy = y->data + (y->ldd) * (start[1] + col);
            /* 列遍历 */
            for (j = 0; j < mat->ncols; ++j, ++dx) {
                /* 非零元素遍历 */
                for (i = mat->j_col[j]; i < mat->j_col[j + 1]; ++i) {
                    dy[*i_row++] += (*dm++) * (*dx);
                }
            }
        }
#endif
    } else {
        /* 矩阵为空时的处理路径 */
        ops->lapack_ops->MultiVecAxpby(1.0, (void **)x, 0.0, (void **)y,
                                       start, end, ops->lapack_ops);
    }
    return;
}

/**
 * @brief 执行稀疏矩阵转置与多向量的乘法运算
 * 
 * @param[in] mat   输入的CCS格式稀疏矩阵指针（必须为方阵且对称）
 * @param[in] x     输入向量组指针(LAPACKVEC格式)，存储右乘向量
 * @param[out] y    输出向量组指针(LAPACKVEC格式)，存储计算结果
 * @param[in] start 起始索引数组：[x起始索引, y起始索引]（包含）
 * @param[in] end   结束索引数组：[x结束索引, y结束索引]（不包含）
 * @param[in] ops   操作接口结构体指针，提供外部函数调用
 *
 *
 * @warning 使用限制：
 * - 仅适用于对称矩阵的伪转置乘法
 * - start/end索引差必须相等（assert验证）
 * - 参数有效性由调用者保证（非空指针、合法索引范围等）
 */
static void MatTransDotMultiVec(CCSMAT *mat, LAPACKVEC *x,
                                LAPACKVEC *y, int *start, int *end, struct OPS_ *ops) {
    /* 参数有效性验证 */
    assert(end[0] - start[0] == end[1] - start[1]);
    assert(y->nrows == y->ldd);
    assert(x->nrows == x->ldd);
    assert(mat->nrows == mat->ncols); // 只适用于对称矩阵
    /* 调用核心计算函数（保留原注释语义） */
    /* Only for 对称矩阵 */
    MatDotMultiVec(mat, x, y, start, end, ops);
    return;
}

/**
 * @brief 通过矩阵创建向量对象
 *
 * 本函数封装了MultiVecCreateByMat接口，专门用于根据稀疏矩阵创建单个向量。
 * 适用于LAPACK向量操作场景，通过操作集抽象层实现资源分配和初始化。
 *
 * @param[out] des_vec 双重指针，用于接收新创建的LAPACK向量对象地址
 * @param[in] src_mat  输入参数，指向CCS格式稀疏矩阵的指针
 * @param[in] ops      操作集接口指针，包含内存分配等底层操作函数
 *
 * @note 本函数通过调用MultiVecCreateByMat接口实现，第二个固定参数1表示创建单个向量。
 *       调用者需确保ops参数包含有效的内存分配函数指针。
 */
static void VecCreateByMat(LAPACKVEC **des_vec, CCSMAT *src_mat, struct OPS_ *ops) {
    /* 调用多维向量创建接口，创建维度为1的向量 */
    MultiVecCreateByMat(des_vec, 1, src_mat, ops);
    return;
}

/**
 * @brief 计算矩阵与向量的点乘，并将结果存储到另一个向量中
 * 
 * 该函数通过调用MatDotMultiVec实现矩阵与向量的点乘操作，使用预定义的起始和结束索引范围。
 * 
 * @param mat   CCS格式的稀疏矩阵指针，输入矩阵数据
 * @param x     输入向量指针，参与矩阵乘法计算的向量
 * @param y     输出向量指针，存储矩阵与向量相乘的结果
 * @param ops   运算控制参数结构体指针，可能包含并行计算或数学运算的相关配置
 * 
 * @note start和end数组分别初始化为{0,0}和{1,1}，表示默认的索引范围参数配置，
 *       具体维度含义需结合MatDotMultiVec的实现确定
 */
static void MatDotVec(CCSMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops) {
    // 定义默认的索引范围参数：起始索引和结束索引
    int start[2] = {0, 0}, end[2] = {1, 1};
    /* 调用多向量乘法核心函数，将运算范围限定在预设的索引区间内
     * 该函数可能支持分块计算或并行化计算，通过start/end参数控制处理范围 */
    MatDotMultiVec(mat, x, y, start, end, ops);
    return;
}
/**
 * 执行矩阵转置与向量的点乘操作(包装函数)
 * 
 * @param mat  指向CCSMAT稀疏矩阵的指针，表示输入矩阵
 * @param x    输入向量(LAPACKVEC格式)，参与矩阵转置后的乘法运算
 * @param y    输出向量(LAPACKVEC格式)，存储运算结果
 * @param ops  操作控制结构指针，包含底层运算所需的控制参数或函数指针
 * 
 * @note 本函数通过设置默认的起止范围参数(start=[0,0], end=[1,1])，
 *       将具体计算委托给MatTransDotMultiVec函数实现。这种设计常用于
 *       处理单元素或特定子矩阵的转置乘操作。
 */
static void MatTransDotVec(CCSMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops) {
    // 设置默认索引范围参数
    int start[2] = {0, 0}, end[2] = {1, 1};
    /* 调用多向量计算函数执行核心运算
     * start/end参数在此场景下限定运算范围
     * 当前设置对应单个元素的运算范围 */
    MatTransDotMultiVec(mat, x, y, start, end, ops);
    return;
}

/* Encapsulation */
/**
 * @brief 查看CCS格式矩阵的详细信息
 * 
 * 该函数封装了MatView操作，用于将CCS格式矩阵的内容通过指定操作结构体输出。
 * 主要完成类型转换并调用底层矩阵查看接口。
 *
 * @param[in] mat  待查看的矩阵指针，需强制转换为CCSMAT*类型使用
 * @param[in] ops  操作控制结构体指针，包含输出控制参数和回调函数
 * 
 * @note 该函数为静态封装函数，主要实现类型安全转换功能
 */
static void CCS_MatView(void *mat, struct OPS_ *ops) {
    MatView((CCSMAT *)mat, ops);
    return;
}
/* vec */
/**
 * @brief 通过源矩阵创建CCS向量对象
 * 
 * @param[out] des_vec 指向新创建向量指针的指针（二级指针），输出参数。
 *                   接收通过源矩阵生成的LAPACKVEC向量对象地址
 * @param[in]  src_mat 输入矩阵指针，应指向有效的CCSMAT压缩稀疏列矩阵结构体
 * @param[in]  ops     操作接口结构体指针，包含底层数学运算的实现和上下文配置
 * 
 * @note 该函数为静态封装函数，实际调用LAPACK库的VecCreateByMat实现。
 *       参数强制转换说明：
 *       - des_vec 被转换为 LAPACKVEC** 类型
 *       - src_mat 被转换为 CCSMAT* 类型
 */
static void CCS_VecCreateByMat(void **des_vec, void *src_mat, struct OPS_ *ops) {
    VecCreateByMat((LAPACKVEC **)des_vec, (CCSMAT *)src_mat, ops);
    return;
}
static void CCS_MatDotVec(void *mat, void *x, void *y, struct OPS_ *ops) {
    MatDotVec((CCSMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
    return;
}
static void CCS_MatTransDotVec(void *mat, void *x, void *y, struct OPS_ *ops) {
    MatTransDotVec((CCSMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
    return;
}
/* multi-vec */
/**
 * @brief 执行CCS格式矩阵转置与向量的点积运算
 * 
 * @param[in] mat    CCS格式稀疏矩阵指针，输入矩阵需为CCSMAT类型
 * @param[in] x      输入向量指针，实际类型应为LAPACKVEC列向量
 * @param[out] y     输出向量指针，实际类型应为LAPACKVEC列向量
 * @param[in] ops    运算控制参数结构体，包含算法选项和并行设置
 * 
 * @note 本函数是MatTransDotVec的封装实现，完成以下转换：
 *       1. 将void*参数转换为具体矩阵/向量类型指针
 *       2. 执行数学运算 y = mat^T * x
 *       3. ops参数控制运算选项（如并行模式/算法选择等）
 */
static void CCS_MultiVecCreateByMat(void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops) {
    MultiVecCreateByMat((LAPACKVEC **)des_vec, num_vec, (CCSMAT *)src_mat, ops);
    return;
}

/**
 * @brief CCS矩阵与向量点乘运算的封装函数
 * 
 * 该函数将类型擦除的矩阵和向量参数转换为具体类型后，调用底层的MatDotMultiVec实现
 * 矩阵-向量点乘运算。适用于需要接口泛化的场景。
 *
 * @param mat 类型擦除的矩阵对象指针，实际应为CCSMAT*类型
 * @param x 输入向量指针数组，元素应为LAPACKVEC*类型
 * @param y 输出向量指针数组，元素应为LAPACKVEC*类型
 * @param start 指向运算起始位置索引的指针
 * @param end 指向运算结束位置索引的指针（包含该位置）
 * @param ops 运算上下文对象，包含算法需要的环境参数
 *
 * @note 该函数通过强制类型转换保证类型安全，调用者需确保参数类型正确
 * @see MatDotMultiVec 实际执行运算的底层函数
 */
static void CCS_MatDotMultiVec(void *mat, void **x,
                               void **y, int *start, int *end, struct OPS_ *ops) {
    MatDotMultiVec((CCSMAT *)mat, (LAPACKVEC *)x,
                   (LAPACKVEC *)y, start, end, ops);
    return;
}
static void CCS_MatTransDotMultiVec(void *mat, void **x,
                                    void **y, int *start, int *end, struct OPS_ *ops) {
    MatTransDotMultiVec((CCSMAT *)mat, (LAPACKVEC *)x,
                        (LAPACKVEC *)y, start, end, ops);
    return;
}

void OPS_CCS_Set(struct OPS_ *ops) {
    assert(ops->lapack_ops == NULL);
    OPS_Create(&(ops->lapack_ops));
    OPS_LAPACK_Set(ops->lapack_ops);
    ops->Printf = DefaultPrintf;
    ops->GetOptionFromCommandLine = DefaultGetOptionFromCommandLine;
    ops->GetWtime = DefaultGetWtime;
    ops->MatView = CCS_MatView;
    /* vec */
    ops->VecCreateByMat = CCS_VecCreateByMat;
    ops->VecCreateByVec = ops->lapack_ops->VecCreateByVec;
    ops->VecDestroy = ops->lapack_ops->VecDestroy;
    ops->VecView = ops->lapack_ops->VecView;
    ops->VecInnerProd = ops->lapack_ops->VecInnerProd;
    ops->VecLocalInnerProd = ops->lapack_ops->VecLocalInnerProd;
    ops->VecSetRandomValue = ops->lapack_ops->VecSetRandomValue;
    ops->VecAxpby = ops->lapack_ops->VecAxpby;
    ops->MatDotVec = CCS_MatDotVec;
    ops->MatTransDotVec = CCS_MatTransDotVec;
    /* multi-vec */
    ops->MultiVecCreateByMat = CCS_MultiVecCreateByMat;
    ops->MultiVecCreateByVec = ops->lapack_ops->MultiVecCreateByVec;
    ops->MultiVecCreateByMultiVec = ops->lapack_ops->MultiVecCreateByMultiVec;
    ops->MultiVecDestroy = ops->lapack_ops->MultiVecDestroy;
    ops->GetVecFromMultiVec = ops->lapack_ops->GetVecFromMultiVec;
    ops->RestoreVecForMultiVec = ops->lapack_ops->RestoreVecForMultiVec;
    ops->MultiVecView = ops->lapack_ops->MultiVecView;
    ops->MultiVecLocalInnerProd = ops->lapack_ops->MultiVecLocalInnerProd;
    ops->MultiVecInnerProd = ops->lapack_ops->MultiVecInnerProd;
    ops->MultiVecSetRandomValue = ops->lapack_ops->MultiVecSetRandomValue;
    ops->MultiVecAxpby = ops->lapack_ops->MultiVecAxpby;
    ops->MultiVecLinearComb = ops->lapack_ops->MultiVecLinearComb;
    ops->MatDotMultiVec = CCS_MatDotMultiVec;
    ops->MatTransDotMultiVec = CCS_MatTransDotMultiVec;
    return;
}
