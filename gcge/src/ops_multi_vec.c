/**
 * @file ops_multi_vec.c
 * @brief 实现了多向量操作，包括创建、销毁、线性组合等
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "app_lapack.h"
#include "ops.h"

#define DEBUG 0

void DefaultPrintf(const char *fmt, ...) {
#if OPS_USE_MPI
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (PRINT_RANK == rank) {
        va_list vp;
        va_start(vp, fmt);
        vprintf(fmt, vp);
        va_end(vp);
    }
#else
    va_list vp;
    va_start(vp, fmt);
    vprintf(fmt, vp);
    va_end(vp);
#endif
    return;
}
double DefaultGetWtime(void) {
    double time;
#if OPS_USE_MPI
    time = MPI_Wtime();
#elif OPS_USE_OMP || OPS_USE_INTEL_MKL
    time = omp_get_wtime();
#else
    time = (double)clock() / CLOCKS_PER_SEC;
#endif
    return time;
}

int DefaultGetOptionFromCommandLine(
    const char *name, char type, void *value,
    int argc, char *argv[], struct OPS_ *ops) {
    int arg_idx = 0, set = 0;
    int *int_value;
    double *dbl_value;
    char *str_value;
    while (arg_idx < argc) {
        if (0 == strcmp(argv[arg_idx], name)) {
            ops->Printf("argv[%d] = \"%s\", name = \"%s\"\n", arg_idx, argv[arg_idx], name);
            if (arg_idx + 1 < argc) {
                set = 1;
            } else {
                break;
            }
            switch (type) {
            case 'i':
                int_value = (int *)value;
                *int_value = atoi(argv[++arg_idx]);
                break;
            case 'f':
                dbl_value = (double *)value;
                *dbl_value = atof(argv[++arg_idx]);
                break;
            case 's':
                str_value = (char *)value;
                strcpy(str_value, argv[++arg_idx]);
                break;
            default:
                break;
            }
            break;
        }
        ++arg_idx;
    }
    return set;
}

/**
 * @brief 创建多个向量副本并初始化指针数组
 *
 * 根据源向量模板创建指定数量的向量副本，并将这些副本的指针存储在动态分配的数组中。
 * 该函数主要用于需要批量创建相似向量结构的场景。
 *
 * @param[out] multi_vec 三级指针参数(输出参数)
 *                       1. 第一级指针指向指针数组的存储地址
 *                       2. 最终存储格式为：multi_vec[0] 指向第一个向量指针，multi_vec[1] 指向第二个向量指针，...
 * @param[in]  num_vec   需要创建的向量数量(输入参数)
 * @param[in]  src_vec   源向量模板指针(输入参数)，所有新向量都将基于此模板创建
 * @param[in]  ops       操作接口
 *
 */
void DefaultMultiVecCreateByVec(void ***multi_vec, int num_vec, void *src_vec, struct OPS_ *ops) {
    int col;
    // 分配指针数组内存：包含num_vec个void*元素的空间
    (*multi_vec) = malloc(num_vec * sizeof(void *));
    // 批量创建向量副本
    for (col = 0; col < num_vec; ++col) {
        // 逐列创建向量
        ops->VecCreateByVec((*multi_vec) + col, src_vec, ops);
    }
    return;
}

/**
 * @brief 创建多个向量（multi-vector）并将其存储在数组中，使用指定的矩阵src_mat作为源
 * 
 * 该函数通过提供的操作结构体中的方法，为每个向量调用创建函数，生成多个向量并存储在动态分配的数组中。
 * 
 * @param multi_vec 三级指针，用于输出生成的向量数组。函数将分配内存并修改该指针以指向新数组。
 * @param num_vec 要创建的向量数量，决定数组的长度
 * @param src_mat 源矩阵指针，作为创建向量时的数据来源
 * @param ops 操作接口结构体指针，包含具体的向量创建方法 VecCreateByMat
 * @return 无返回值
 */
void DefaultMultiVecCreateByMat(void ***multi_vec, int num_vec, void *src_mat, struct OPS_ *ops) {
    int col;
    (*multi_vec) = malloc(num_vec * sizeof(void *));
    for (col = 0; col < num_vec; ++col) {
        ops->VecCreateByMat((*multi_vec) + col, src_mat, ops);
    }
    return;
}

/**
 * @brief 基于现有多向量结构创建新的多向量副本
 * 
 * @param[out] multi_vec 三级指针参数(输出参数)
 *                       1. 第一级指针指向指针数组的存储地址
 *                       2. 最终存储格式为：multi_vec[0] 指向第一个向量指针，multi_vec[1] 指向第二个向量指针，...
 * @param[in]  num_vec   需要创建的向量数量
 * @param[in]  src_mv    源多向量指针数组(输入参数)，新向量基于其第一个元素创建模板
 * @param[in]  ops       操作接口
 */
void DefaultMultiVecCreateByMultiVec(void ***multi_vec, int num_vec, void **src_mv, struct OPS_ *ops) {
    int col;
    (*multi_vec) = malloc(num_vec * sizeof(void *));
    for (col = 0; col < num_vec; ++col) {
        ops->VecCreateByVec((*multi_vec) + col, *src_mv, ops);
    }
    return;
}
void DefaultMultiVecDestroy(void ***multi_vec, int num_vec, struct OPS_ *ops) {
    int col;
    for (col = 0; col < num_vec; ++col) {
        ops->VecDestroy((*multi_vec) + col, ops);
    }
    free((*multi_vec));
    *multi_vec = NULL;
    return;
}

/**
 * @brief 从多维向量数组中获取指定列的一维向量指针
 * 
 * 该函数通过数组索引直接从多维向量数组中提取对应列的向量指针，
 * 不涉及内存复制操作，仅进行指针地址传递
 * 
 * @param multi_vec 输入参数，void二级指针类型，表示多维向量数组头指针
 *                   数组每个元素对应一个列向量的起始地址
 * @param col       输入参数，int类型，指定需要获取的列索引
 * @param vec       输出参数，void二级指针类型，用于接收目标列向量的指针地址
 * @param ops       操作接口
 * 
 */
void DefaultGetVecFromMultiVec(void **multi_vec, int col, void **vec, struct OPS_ *ops) {
    // 通过数组索引直接获取目标列的指针地址，完成向量提取
    *vec = multi_vec[col];
    return;
}

/**
 * @brief 默认的多向量恢复向量函数（存根实现）
 *
 * 该函数作为多向量系统中恢复单个向量的默认实现，当前版本未执行实际恢复操作。
 * 函数始终将目标向量指针置空，可能用于接口兼容或后续扩展预留。
 *
 * @param multi_vec 指向多向量结构指针的指针（输入参数，未使用）
 * @param col 需要恢复的向量列索引（输入参数，未使用）
 * @param vec 用于返回恢复后的单个向量指针的二级指针（输出参数）
 * @param ops 操作接口
 */
void DefaultRestoreVecForMultiVec(void **multi_vec, int col, void **vec, struct OPS_ *ops) {
    /* 始终将目标向量置为空指针 */
    *vec = NULL;
    return;
}

/**
 * @brief 遍历并显示指定范围内多个向量的视图信息
 * 
 * @param[in] x     二维指针数组，存储待显示的多向量集合
 *                  数组元素按列组织，每个元素指向一个向量对象
 * @param[in] start 起始列索引(包含)，有效范围应满足 0 <= start < end <= 数组总列数
 * @param[in] end   结束列索引(不包含)，指定操作范围的终止边界
 * @param[in] ops   操作接口
 * 
 */
void DefaultMultiVecView(void **x, int start, int end, struct OPS_ *ops) {
    int col;
    /* 遍历指定列范围，依次调用VecView操作 */
    for (col = start; col < end; ++col) {
        ops->VecView(x[col], ops);
    }
    return;
}

/**
 * @brief 计算多向量之间的局部内积，根据模式处理对称/对角/一般情况
 * 
 * @param[in] nsdIP 计算模式标识符：
 *                  'S' - 对称模式（填充上下三角），
 *                  'D' - 对角模式（仅计算主对角），
 *                  其他 - 常规模式（全矩阵计算）
 * @param[in] x 源多向量指针数组（输入向量集合）
 * @param[in] y 目标多向量指针数组（输入向量集合）
 * @param[in] is_vec 向量类型标识（未在代码中实际使用）
 * @param[in] start 起始索引数组[2]，定义处理范围的起始行列
 * @param[in] end 结束索引数组[2]，定义处理范围的结束行列
 * @param[out] inner_prod 内积结果存储数组（二维矩阵形式）
 * @param[in] ldIP 结果数组的leading dimension（行维度）
 * @param[in] ops 操作函数集指针，包含向量操作接口
 */
void DefaultMultiVecLocalInnerProd(char nsdIP, void **x, void **y, int is_vec, int *start, int *end,
                                   double *inner_prod, int ldIP, struct OPS_ *ops) {
    int row, col, nrows, ncols, length, incx, incy;
    double *source, *destin;
    void *vec_x, *vec_y;
    /* 计算实际处理的行列数 */
    nrows = end[0] - start[0];
    ncols = end[1] - start[1];
    if (nsdIP == 'S') { // 对称矩阵处理模式
        assert(nrows == ncols);
        /* 计算上三角部分的内积 */
        for (col = 0; col < ncols; ++col) {
            ops->GetVecFromMultiVec(y, start[1] + col, &vec_y, ops);
            destin = inner_prod + ldIP * col + col;
            /* 遍历当前列到末尾的行 */
            for (row = col; row < nrows; ++row) {
                ops->GetVecFromMultiVec(x, start[0] + row, &vec_x, ops);
                ops->VecInnerProd(vec_x, vec_y, destin, ops);
                ++destin; // 移动到下一行同列位置
                // 解除`vec_x`指针与`multi_vec`结构的关联（将vec_x置为NULL），避免后续误操作导致的内存问题
                ops->RestoreVecForMultiVec(x, start[0] + row, &vec_x, ops);
            }
            ops->RestoreVecForMultiVec(y, start[1] + col, &vec_y, ops);
        }
        /* 将上三角数据镜像到下三角完成对称填充 */
        for (col = 0; col < ncols; ++col) {
            length = ncols - col - 1;
            source = inner_prod + ldIP * col + (col + 1);
            incx = 1; // 列步长
            destin = inner_prod + ldIP * (col + 1) + col;
            incy = ldIP;
            dcopy(&length, source, &incx, destin, &incy);
        }
    } else if (nsdIP == 'D') { // // 对角模式处理
        assert(nrows == ncols);
        /* 仅计算主对角线上的内积,
         * 即对于[a1,a2],[b1,b2],只计算a1*b1,a2*b2,而不计算a1*b2
          */
        for (col = 0; col < ncols; ++col) {
            ops->GetVecFromMultiVec(y, start[1] + col, &vec_y, ops);
            ops->GetVecFromMultiVec(x, start[0] + col, &vec_x, ops);
            ops->VecInnerProd(vec_x, vec_y, inner_prod + ldIP * col, ops);
            ops->RestoreVecForMultiVec(x, start[0] + col, &vec_x, ops);
            ops->RestoreVecForMultiVec(y, start[1] + col, &vec_y, ops);
        }
    } else { // 常规全矩阵处理模式
        /* 完整计算所有行列组合的内积 */
        for (col = 0; col < ncols; ++col) {
            ops->GetVecFromMultiVec(y, start[1] + col, &vec_y, ops);
            destin = inner_prod + ldIP * col; // 列起始位置
            for (row = 0; row < nrows; ++row) {
                ops->GetVecFromMultiVec(x, start[0] + row, &vec_x, ops);
                ops->VecInnerProd(vec_x, vec_y, destin, ops);
                ++destin; // 行方向递增
                ops->RestoreVecForMultiVec(x, start[0] + row, &vec_x, ops);
            }
            ops->RestoreVecForMultiVec(y, start[1] + col, &vec_y, ops);
        }
    }
    return;
}
/**
 * @brief 计算多个向量的内积（支持MPI并行环境）
 *
 * @param nsdIP    内积类型标识符：
 *                 'D'表示对角矩阵处理模式
 *                 在使用MPI并行环境时，只把'D'（对角矩阵）拿出来单独处理
 * @param x        输入向量数组指针（维度由start/end决定）
 * @param y        输入向量数组指针（维度由start/end决定）
 * @param is_vec   向量模式标识（具体含义取决于MultiVecLocalInnerProd实现）
 * @param start    二维起始索引数组[start0, start1]
 * @param end      二维结束索引数组[end0, end1]
 * @param inner_prod 输出内积结果矩阵（行优先存储）
 * @param ldIP     内积矩阵的导引维度（leading dimension）
 * @param ops      操作集结构体指针，包含本地计算内核
 */
void DefaultMultiVecInnerProd(char nsdIP, void **x, void **y, int is_vec, int *start, int *end,
                              double *inner_prod, int ldIP, struct OPS_ *ops) {
    // 在当前进程的本地数据上计算内积。这一步骤得到的是每个进程上的局部结果
    ops->MultiVecLocalInnerProd(nsdIP, x, y, is_vec, start, end, inner_prod, ldIP, ops);
#if OPS_USE_MPI
    /* 计算实际处理区域的维度 */
    int nrows = end[0] - start[0], ncols = end[1] - start[1];
    /* 特殊处理对角矩阵模式 */
    if (nsdIP == 'D') {
        assert(nrows == ncols);
        nrows = 1; // 压缩为单行存储
    }
    /* 根据内存布局选择不同的MPI归约策略 */
    if (nrows == ldIP) {
        /* 连续内存布局：直接将局部内积结果合并成全局结果 */
        MPI_Allreduce(MPI_IN_PLACE, inner_prod,
                      nrows * ncols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } else {
        /* 非连续内存布局：创建自定义MPI类型处理子矩阵 */
        MPI_Datatype data_type;
        MPI_Op op;
        // 创建描述子矩阵内存布局的数据类型
        CreateMPIDataTypeSubMat(&data_type, nrows, ncols, ldIP);
        // 创建支持子矩阵求和的自定义归约操作，用于对子矩阵的求和
        CreateMPIOpSubMatSum(&op); /* �Ե�һ��������submat */
        /* 执行自定义类型的全局归约 */
        MPI_Allreduce(MPI_IN_PLACE, inner_prod,
                      1, data_type, op, MPI_COMM_WORLD);
        // 清理自定义MPI资源
        DestroyMPIOpSubMatSum(&op);
        DestroyMPIDataTypeSubMat(&data_type);
    }

#endif
    return;
}

/**
 * @brief 为多个向量设置随机值
 *
 * 该函数遍历指定范围内的向量数组元素，为每个向量调用随机值设置函数。
 * 主要用于并行计算环境中对多个向量进行批量随机初始化。
 *
 * @param x 二维指针数组，每个元素指向一个待操作的向量对象
 * @param start 起始索引值（包含）
 * @param end 结束索引值（不包含）
 * @param ops 操作接口
 */
void DefaultMultiVecSetRandomValue(void **x, int start, int end, struct OPS_ *ops) {
    /* 遍历指定列范围，为每个向量设置随机值 */
    int col;
    for (col = start; col < end; ++col) {
        /* 调用操作集中定义的向量随机值设置方法 */
        ops->VecSetRandomValue(x[col], ops);
    }
    return;
}
/**
 * @brief 执行多向量线性组合操作 alpha*x + beta*y
 *
 * 该函数根据参数选择执行不同的多向量线性组合策略。当alpha为0或x为空时，
 * 仅对y向量进行beta缩放操作；否则执行完整的线性组合计算。
 *
 * @param alpha    标量系数，作用于x向量
 * @param x        源多向量数组指针，包含多个列向量
 * @param beta     标量系数，作用于y向量
 * @param y        目标多向量数组指针，包含多个列向量
 * @param start    起始索引数组，start[0]对应x起始列，start[1]对应y起始列
 * @param end      结束索引数组，end[0]对应x结束列，end[1]对应y结束列
 * @param ops      操作接口
 */
void DefaultMultiVecAxpby(
    double alpha, void **x, double beta, void **y,
    int *start, int *end, struct OPS_ *ops) {
    int col, ncols = end[1] - start[1];
    void *vec_x, *vec_y;
    /* 处理alpha为0或x为空的情况：仅执行y向量的beta缩放 */
    if (alpha == 0.0 || x == NULL) {
        for (col = 0; col < ncols; ++col) {
            /* 获取当前列y向量并执行缩放操作 */
            ops->GetVecFromMultiVec(y, start[1] + col, &vec_y, ops);
            ops->VecAxpby(alpha, NULL, beta, vec_y, ops);
            ops->RestoreVecForMultiVec(y, start[1] + col, &vec_y, ops);
        }

    } else {
        /* 执行完整线性组合：alpha*x + beta*y */
        for (col = 0; col < ncols; ++col) {
            /* 获取当前列的x和y向量 */
            ops->GetVecFromMultiVec(y, start[1] + col, &vec_y, ops);
            ops->GetVecFromMultiVec(x, start[0] + col, &vec_x, ops);
            /* 执行向量线性组合运算 */
            ops->VecAxpby(alpha, vec_x, beta, vec_y, ops);
            /* 释放向量资源 */
            ops->RestoreVecForMultiVec(x, start[0] + col, &vec_x, ops);
            ops->RestoreVecForMultiVec(y, start[1] + col, &vec_y, ops);
        }
    }
    return;
}

/**
 * @brief 多向量线性组合计算函数 y<-X*C + Y *diag(beta)
 *
 * 该函数根据系数矩阵和beta参数计算多向量集合的线性组合，支持矩阵模式和向量模式两种操作方式
 *
 * @param x 输入多向量集合指针数组(二维结构)，若为NULL表示纯缩放操作
 * @param y 输出多向量集合指针数组(二维结构)
 * @param is_vec 操作模式标志：0-矩阵模式，1-向量模式
 * @param start_xy 二维起始索引数组，定义操作区域[x_start, y_start]
 * @param end_xy 二维结束索引数组，定义操作区域[x_end, y_end]
 * @param coef 系数矩阵C，以列优先的一维向量形式传入，C(i,k)=coef[i+k*ldc].
 * @param ldc 系数矩阵的列维度(leading dimension)
 * @param beta 缩放系数数组指针，用于最终结果缩放
 * @param incb beta数组的存储步长
 * @param ops 操作接口
 */
void DefaultMultiVecLinearComb(
    void **x, void **y, int is_vec,
    int *start_xy, int *end_xy,
    double *coef, int ldc,
    double *beta, int incb, struct OPS_ *ops) {
    int i, k, nrows, ncols, start[2], end[2];
    double gamma;
    void *vec_x, *vec_y;
    // 计算实际操作的行列范围
    nrows = end_xy[0] - start_xy[0];
    ncols = end_xy[1] - start_xy[1];

    /* 处理纯缩放情况(x或coef为空时) */
    if (x == NULL || coef == NULL) {
        // 遍历所有列进行纯缩放操作
        for (k = 0; k < ncols; ++k) {
            if (beta == NULL)
                gamma = 0.0;
            else
                gamma = *(beta + k * incb);
            /* 矩阵模式: 对y的多向量进行缩放 */
            if (is_vec == 0) {
                // 设置单列操作范围
                start[0] = start_xy[1] + k;
                end[0] = start[0] + 1;
                start[1] = start_xy[1] + k;
                end[1] = start[1] + 1;
                ops->MultiVecAxpby(0, NULL, gamma, y, start, end, ops);
            } else { /* 向量模式: 直接操作单个向量 */
                ops->VecAxpby(0, NULL, gamma, *y, ops);
            }
        }
        /* 处理完整线性组合情况 */
    } else {
        if (is_vec == 0) { /* 矩阵模式下的线性组合 */
            for (k = 0; k < ncols; ++k) {
                if (beta == NULL)
                    gamma = 0.0;
                else
                    gamma = *(beta + k * incb);
                // 对当前列的每个行元素进行组合计算
                for (i = 0; i < nrows; ++i) {
                    // 设置当前元素的操作范围
                    start[0] = start_xy[0] + i;
                    end[0] = start[0] + 1;
                    start[1] = start_xy[1] + k;
                    end[1] = start[1] + 1;
                    /* 首元素计算使用gamma(包含beta项)，后续元素累加 */
                    if (i == 0) {
                        ops->MultiVecAxpby(*(coef + i + k * ldc), x, gamma, y, start, end, ops);
                    } else {
                        ops->MultiVecAxpby(*(coef + i + k * ldc), x, 1.0, y, start, end, ops);
                    }
                }
            }
        } else { /* 向量模式下的线性组合 */
            for (k = 0; k < ncols; ++k) {
                if (beta == NULL)
                    gamma = 0.0;
                else
                    gamma = *(beta + k * incb);
                vec_y = *y; // 获取当前列的输出向量
                for (i = 0; i < nrows; ++i) {
                    // 从多向量集合中提取当前输入向量
                    ops->GetVecFromMultiVec(x, start_xy[0] + i, &vec_x, ops);
                    /* 首元素组合包含beta项，后续进行累加 */
                    if (i == 0) {
                        ops->VecAxpby(*(coef + i + k * ldc), vec_x, gamma, vec_y, ops);
                    } else {
                        ops->VecAxpby(*(coef + i + k * ldc), vec_x, 1.0, vec_y, ops);
                    }
                    // 将修改后的向量写回多向量集合
                    ops->RestoreVecForMultiVec(x, start_xy[0] + i, &vec_x, ops);
                }
                vec_y = NULL; // 清除当前列向量引用
            }
        }
    }
    return;
}

/**
 * @brief 执行矩阵块与多向量块的点乘运算
 *
 * 该函数遍历指定列范围内的矩阵块，对每个列向量执行矩阵-向量点乘操作，
 * 并将结果存储到对应的输出向量块中。
 *
 * @param mat      - 输入矩阵指针，具体类型由ops接口实现决定
 * @param x        - 输入多向量块指针数组，包含待计算的输入向量集合
 * @param y        - 输出多向量块指针数组，用于存储计算结果
 * @param start    - 二维起始索引数组，
 * @param end      - 二维结束索引数组
 * @param ops      - 操作接口
 **/
void DefaultMatDotMultiVec(void *mat, void **x, void **y,
                           int *start, int *end, struct OPS_ *ops) {
    int col, ncols = end[1] - start[1];
    void *vec_x, *vec_y;
    /* 列遍历核心逻辑：
     * 1. 从输出向量块提取当前列向量
     * 2. 从输入向量块提取对应列向量
     * 3. 执行矩阵-向量点乘
     * 4. 释放向量资源保持数据一致性 */
    for (col = 0; col < ncols; ++col) {
        // 获取当前列的输入输出向量
        ops->GetVecFromMultiVec(y, start[1] + col, &vec_y, ops);
        ops->GetVecFromMultiVec(x, start[0] + col, &vec_x, ops);
        // 执行矩阵向量点乘运算
        ops->MatDotVec(mat, vec_x, vec_y, ops);
        // 释放向量资资源
        ops->RestoreVecForMultiVec(x, start[0] + col, &vec_x, ops);
        ops->RestoreVecForMultiVec(y, start[1] + col, &vec_y, ops);
    }
    return;
}

/**
 * @brief 执行矩阵转置与多向量组的点乘操作
 * 
 * 该函数遍历指定的列范围，对每个列向量执行矩阵转置后与对应向量的点乘操作。
 * 具体流程为：
 * 1. 从多向量数组y和x中提取当前列的向量
 * 2. 调用矩阵转置点乘操作接口
 * 3. 将处理后的向量返还到多向量数组
 * 
 * @param[in] mat   矩阵对象指针，需支持MatTransDotVec操作
 * @param[in] x     输入多向量数组指针（二维结构），包含多列源向量
 * @param[in,out] y 输入/输出多向量数组指针（二维结构），包含多列目标向量
 * @param[in] start 起始索引数组：
 *                 start[0] - x向量的起始列索引
 *                 start[1] - y向量的起始列索引
 * @param[in] end   结束索引数组：
 *                 end[0] - x向量的结束列索引
 *                 end[1] - y向量的结束列索引
 * @param[in] ops   操作接口
 **/
void DefaultMatTransDotMultiVec(void *mat, void **x, void **y,
                                int *start, int *end, struct OPS_ *ops) {
    int col, ncols = end[1] - start[1];
    void *vec_x, *vec_y;
    /* 遍历处理指定列范围 */
    for (col = 0; col < ncols; ++col) {
        /* 从多向量数组中提取当前列向量 */
        ops->GetVecFromMultiVec(y, start[1] + col, &vec_y, ops);
        ops->GetVecFromMultiVec(x, start[0] + col, &vec_x, ops);
        /* 执行矩阵转置点乘：mat^T * vec_x -> vec_y */
        ops->MatTransDotVec(mat, vec_x, vec_y, ops);
        /* 返还向量指针到数组 */
        ops->RestoreVecForMultiVec(x, start[0] + col, &vec_x, ops);
        ops->RestoreVecForMultiVec(y, start[1] + col, &vec_y, ops);
    }
    return;
}

/**
 * @brief 计算矩阵或向量的乘积并累加到结果数组中（Q^T*A*P或类似操作）
 * 
 * @param ntsA 矩阵A的转置标识，'N'或'S'表示不转置，'T'表示转置
 * @param ntsdQAP 是否输出Q^TAP前进行转置。.T表示需要转置，其他表示不需要转置
 * @param mvQ 多维向量组Q的指针数组
 * @param matA 矩阵A的指针
 * @param mvP 多维向量组P的指针数组
 * @param is_vec 标识操作对象是否为单个向量(1)还是向量组(0)
 * @param startQP 二维数组操作起始索引[start_dim0, start_dim1]
 * @param endQP 二维数组操作结束索引[end_dim0, end_dim1]
 * @param qAp 结果存储数组指针
 * @param ldQAP 结果数组的leading dimension
 * @param mv_ws 工作空间向量组的指针数组
 * @param ops 操作函数集结构体指针
 */
void DefaultMultiVecQtAP(char ntsA, char ntsdQAP,
                         void **mvQ, void *matA, void **mvP, int is_vec,
                         int *startQP, int *endQP, double *qAp, int ldQAP,
                         void **mv_ws, struct OPS_ *ops) {
    int start[2], end[2], nrows, ncols;
    /* 边界检查：计算有效行列数 */
    nrows = endQP[0] - startQP[0];
    ncols = endQP[1] - startQP[1];
    if (nrows <= 0 || ncols <= 0) return; // 若无有效数据范围，则直接返回
    /* 如果矩阵 A 为空，则直接利用多向量内积计算 Q 和 P 的内积 */
    if (matA == NULL) {
        if (ntsdQAP == 'T') {
            // 当要求转置内积计算时，交换 start 和 end 的顺序
            start[0] = startQP[1];
            end[0] = endQP[1];
            start[1] = startQP[0];
            end[1] = endQP[0];
            // 直接计算内积：P 和 Q（参数已交换，不需要再转置）
            ops->MultiVecInnerProd('N', mvP, mvQ, is_vec, start, end, qAp, ldQAP, ops);
        } else {
            // 按原顺序计算内积
            ops->MultiVecInnerProd(ntsdQAP, mvQ, mvP, is_vec, startQP, endQP, qAp, ldQAP, ops);
        }
        return;
    }

    /* 当矩阵 A 非空时，先计算 A*P 或 A^T*P */
    // 设置工作区间参数，用于后续的矩阵乘法操作
    start[0] = startQP[1];
    end[0] = endQP[1];
    start[1] = 0;
    end[1] = endQP[1] - startQP[1];
    if (is_vec == 0) {
        // 多向量操作
        if (ntsA == 'N' || ntsA == 'S') {
            // 直接计算 A*P，并将结果存放到工作空间 mv_ws 中
            ops->MatDotMultiVec(matA, mvP, mv_ws, start, end, ops);
        } else if (ntsA == 'T') {
            // 计算 A 的转置与 P 的乘积，即 A^T*P
            ops->MatTransDotMultiVec(matA, mvP, mv_ws, start, end, ops);
        }
        // 可选调试代码（目前被注释掉）用于查看 mv_ws 的内容
        //ops->MultiVecView(mv_ws,start[1],end[1],ops);
    } else {
        // 单向量操作
        if (ntsA == 'N' || ntsA == 'S') {
            // 计算 A*P 针对单个向量
            ops->MatDotVec(matA, mvP, mv_ws, ops);
        } else if (ntsA == 'T') {
            // 对于转置操作，同样调用对应的多向量乘法
            ops->MatTransDotMultiVec(matA, mvP, mv_ws, start, end, ops);
        }
    }
#if DEBUG
    ops->Printf("A*P\n");
    ops->MultiVecView(mv_ws, start[1], end[1], ops);
#endif
    /* 计算 Q^T * (A*P) 内积 */
    if (ntsdQAP == 'T') {
        // 当内积要求转置计算时，重新设置区间参数
        start[0] = 0;
        end[0] = endQP[1] - startQP[1];
        start[1] = startQP[0];
        end[1] = endQP[0];
        // 计算内积：使用 mv_ws 与 mvQ，参数为 'N' 表示不再转置
        ops->MultiVecInnerProd('N', mv_ws, mvQ, is_vec, start, end, qAp, ldQAP, ops);
    } else {
        // 常规内积计算：设置区间参数为 Q 的行范围和 mv_ws 的列范围
        start[0] = startQP[0];
        end[0] = endQP[0];
        start[1] = 0;
        end[1] = endQP[1] - startQP[1];
        ops->MultiVecInnerProd(ntsdQAP, mvQ, mv_ws, is_vec, start, end, qAp, ldQAP, ops);
    }
#if DEBUG
    ops->Printf("qAp = %e\n", *qAp);
    ops->Printf("Q\n");
    ops->MultiVecView(mvQ, start[0], end[0], ops);
    ops->Printf("Q*A*P\n");
#endif
    return;
}
