/**
 * @brief 基于LAPACK的矩阵和向量操作
 */

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include "app_lapack.h"

/* matC = alpha*matQ^{\top}*matA*matP + beta*matC 
 * dbl_ws: nrowsA*ncolsC */
/**
 * @brief 计算稠密矩阵运算 C = alpha * Q^T * A * P + beta * C
 *          支持分块处理、对称矩阵优化及并行计算
 *
 * @param ntluA   A矩阵的三角类型('L'下三角/'U'上三角)或普通矩阵('S'或其他)
 * @param nsdC    C矩阵的存储类型('D'对角/'S'对称/其他 普通)
 * @param nrowsA  矩阵A的行数
 * @param ncolsA  矩阵A的列数
 * @param nrowsC  结果矩阵C的行数
 * @param ncolsC  结果矩阵C的列数
 * @param alpha   前乘标量系数
 * @param matQ    矩阵Q的数据指针
 * @param ldQ     Q矩阵的leading dimension
 * @param matA    矩阵A的数据指针(可为NULL触发特殊处理)
 * @param ldA     A矩阵的leading dimension
 * @param matP    矩阵P的数据指针
 * @param ldP     P矩阵的leading dimension
 * @param beta    后乘标量系数
 * @param matC    结果矩阵C的数据指针
 * @param ldC     C矩阵的leading dimension
 * @param dbl_ws  双精度工作空间指针
 */
static void DenseMatQtAP(char ntluA, char nsdC,
                         int nrowsA, int ncolsA, /* matA �������� */
                         int nrowsC, int ncolsC, /* matC �������� */
                         double alpha, double *matQ, int ldQ,
                         double *matA, int ldA,
                         double *matP, int ldP,
                         double beta, double *matC, int ldC,
                         double *dbl_ws) {
    /* 处理空矩阵直接返回 */
    if (nrowsC == 0 || ncolsC == 0) {
        return;
    }
    /* 特殊情况：A矩阵为空矩阵时的处理 */
    if (nrowsA == 0 && ncolsA == 0) {
        int col;
        /* 如果C的存储类型为对角，特殊初始化 */
        if (nsdC == 'D') {
            assert(nrowsC == ncolsC);
            /* 连续存储（主维度为1）时直接清零 */
            if (ldC == 1) {
                memset(matC, 0, ncolsC * sizeof(double));
                /* 非连续存储时逐列清零 */
            } else {
                for (col = 0; col < ncolsC; ++col) {
                    matC[ldC * col] = 0.0;
                }
            }
            /* 非对角矩阵初始化 */
        } else {
            /* 连续存储时（主维度=当前矩阵行数，即非子矩阵）批量清零 */
            if (ldC == nrowsC) {
                memset(matC, 0, nrowsC * ncolsC * sizeof(double));
                /* 非连续存储时（主维度=当前矩阵行数，即当前矩阵为子矩阵）逐列清零 */
            } else {
                for (col = 0; col < ncolsC; ++col) {
                    memset(matC, 0, nrowsC * sizeof(double));
                    matC += ldC;
                }
            }
        }
        return;
    }

    /* 定义BLAS运算字符常量 */
    char charN = 'N', charT = 'T';
    /* 处理A矩阵为空指针的情况 */
    if (matA == NULL) {
        int idx, inc;
        assert(nrowsA == ncolsA); // 要求A为方阵
        /* 对角矩阵计算模式 */
        if (nsdC == 'D') {
            assert(nrowsC == ncolsC); // 要求C也为方阵
            inc = 1;
            /* beta=1时执行累加计算 */
            if (beta == 1.0) {
                if (alpha != 0.0) {
// 并行计算每列的点积并累加
#if OPS_USE_OMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
                    for (idx = 0; idx < ncolsC; ++idx) {
                        matC[ldC * idx] += alpha * ddot(&nrowsA,
                                                        matQ + ldQ * idx, &inc, matP + ldP * idx, &inc); // A为空指针，只用计算Q^T * P
                    }
                }
            } else if (beta == 0.0) {
                if (alpha != 0.0) {
#if OPS_USE_OMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
                    for (idx = 0; idx < ncolsC; ++idx) {
                        matC[ldC * idx] = alpha * ddot(&nrowsA,
                                                       matQ + ldQ * idx, &inc, matP + ldP * idx, &inc);
                    }
                    // alpha, beta 都为0，则将C矩阵清零。和上面清零的逻辑相同
                } else {
                    if (ldC == 1) {
                        memset(matC, 0, ncolsC * sizeof(double));
                    } else {
#if OPS_USE_OMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
                        for (idx = 0; idx < ncolsC; ++idx) {
                            matC[ldC * idx] = 0.0;
                        }
                    }
                }
            } else { // beta!=0/1
#if OPS_USE_OMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
                for (idx = 0; idx < ncolsC; ++idx) {
                    matC[ldC * idx] = alpha * ddot(&nrowsA,
                                                   matQ + ldQ * idx, &inc, matP + ldP * idx, &inc) +
                                      beta * matC[ldC * idx];
                }
            }
            /* 对称矩阵计算模式 */
        } else if (nsdC == 'S') {
            assert(nrowsC == ncolsC); // 要求矩阵C为方针
            inc = 1;
            /* 分块矩阵向量乘法 */
#if OPS_USE_OMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
            for (idx = 0; idx < ncolsC; ++idx) {
                int nrowsC;
                nrowsC = ncolsC - idx;
                // 这里设置incx=1，即提取P矩阵的列向量；如果incx=ldP,则提取行向量
                // 计算alpha * Q^T * P(:,idx)+beta*C(:,idx)
                dgemv(&charT, &nrowsA, &nrowsC,
                      &alpha, matQ + ldQ * idx, &ldQ,
                      matP + ldP * idx, &inc,
                      &beta, matC + ldC * idx + idx, &inc);
                --nrowsC;
                // 因为是对称矩阵，用下三角矩阵的值赋给上三角矩阵
                dcopy(&nrowsC,
                      matC + ldC * idx + (idx + 1), &inc,  /* copy x */
                      matC + ldC * (idx + 1) + idx, &ldC); /* to   y */
            }
        } else {
#if OPS_USE_OMP
#if 0
#pragma omp parallel num_threads(OMP_NUM_THREADS)
			{
				int id, length, offset;
				id     = omp_get_thread_num();
				length = nrowsC/OMP_NUM_THREADS;
				offset = length*id;
				if (id < nrowsC%OMP_NUM_THREADS) {
					++length; offset += id;
				}
				else {
					offset += nrowsC%OMP_NUM_THREADS;
				} 
				dgemm(&charT,&charN,&length,&ncolsC,&nrowsA,
						&alpha,matQ+offset*ldQ,&ldQ,  /* A */
						       matP           ,&ldP,  /* B */
						&beta ,matC+offset    ,&ldC); /* C */
			}
#else
#pragma omp parallel num_threads(OMP_NUM_THREADS)
            { /* C为普通矩阵时的计算模式 */
                // 并行任务分配
                int id, length, offset;
                id = omp_get_thread_num();
                length = ncolsC / OMP_NUM_THREADS;
                offset = length * id;
                if (id < ncolsC % OMP_NUM_THREADS) {
                    ++length;
                    offset += id;
                } else {
                    offset += ncolsC % OMP_NUM_THREADS;
                }
                // 计算 alpha * Q^T * P + beta + C
                dgemm(&charT, &charN, &nrowsC, &length, &nrowsA,
                      &alpha, matQ, &ldQ,                /* A */
                      matP + offset * ldP, &ldP,         /* B */
                      &beta, matC + offset * ldC, &ldC); /* C */
            }

#endif

#else
            dgemm(&charT, &charN, &nrowsC, &ncolsC, &nrowsA,
                  &alpha, matQ, &ldQ, /* A */
                  matP, &ldP,         /* B */
                  &beta, matC, &ldC); /* C */
#endif
        }

    } /* 处理存在A矩阵的情况 */
    else {
        char side;
        int nrowsW, ncolsW, ldW;
        double *matW, zero, one;
        /* 初始化工作矩阵W */
        nrowsW = nrowsA;
        ncolsW = ncolsC;
        ldW = nrowsW;
        matW = dbl_ws;
        zero = 0.0;
        one = 1.0;
        // 如果A是对称矩阵，用上三角、下三角存储，则进行对称矩阵乘法优化
        if (ntluA == 'L' || ntluA == 'U') {
            side = 'L'; /* left or right */
            /* matW = matA*matP */
            dsymm(&side, &ntluA, &nrowsW, &ncolsW,
                  &one, matA, &ldA, matP, &ldP, &zero, matW, &ldW);
        } else { //A为普通矩阵时
                 /* matW = matA*matP */
#if OPS_USE_OMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
            {
                int id, length, offset;
                id = omp_get_thread_num();
                length = ncolsW / OMP_NUM_THREADS;
                offset = length * id;
                if (id < ncolsW % OMP_NUM_THREADS) {
                    ++length;
                    offset += id;
                } else {
                    offset += ncolsW % OMP_NUM_THREADS;
                }
                dgemm(&ntluA, &charN, &nrowsW, &length, &ncolsA,
                      &one, matA, &ldA,
                      matP + offset * ldP, &ldP,
                      &zero, matW + offset * ldW, &ldW);
            }
#else
            dgemm(&ntluA, &charN, &nrowsW, &ncolsW, &ncolsA,
                  &one, matA, &ldA,
                  matP, &ldP,
                  &zero, matW, &ldW);
#endif
        }
        /* matC = alpha*matQ^{\top}*matW + beta*matC */
        // 由于上面已经计算出了A * P的结果，所以直接使用matW即可
        DenseMatQtAP(ntluA, nsdC, nrowsA, nrowsA, nrowsC, ncolsC,
                     alpha, matQ, ldQ, NULL, ldA, matW, ldW,
                     beta, matC, ldC, dbl_ws);
    }
    return;
}

/* multi-vec */

/**
 * 根据给定的向量创建一个具有多个列的向量
 * 此函数用于在 LAPACK 向量结构的基础上，创建一个新的多列向量
 * 它的行数与源向量相同，列数由参数指定
 * 假如原向量是一个5维向量，设定num_vec=4,则会得到5行4列矩阵，主维度为源向量的主维度，且初始化矩阵数据为0
 * 
 * @param des_vec 输出参数，指向创建的多列向量的指针
 * @param num_vec 新创建的矩阵的列数
 * @param src_vec 源向量，新矩阵的行数将与源向量的行数相同
 * @param ops 操作接口
 */
static void MultiVecCreateByVec(LAPACKVEC **des_vec, int num_vec, LAPACKVEC *src_vec, struct OPS_ *ops) {
    // 分配 LAPACKVEC 结构体的空间
    (*des_vec) = malloc(sizeof(LAPACKVEC));
    // 设置新矩阵的行数、列数和数据的主维度
    (*des_vec)->nrows = src_vec->nrows;
    (*des_vec)->ncols = num_vec;
    (*des_vec)->ldd = src_vec->ldd;
    // 为新矩阵的数据分配空间
    (*des_vec)->data = malloc(((*des_vec)->ldd) * ((*des_vec)->ncols) * sizeof(double));
    // 初始化矩阵数据为 0
    memset((*des_vec)->data, 0, ((*des_vec)->ldd) * ((*des_vec)->ncols) * sizeof(double));
    return;
}

/**
 * @brief 根据给定的源矩阵创建多个向量（多向量结构）
 * 
 * 通过复制源矩阵的列数据，创建一个多向量结构。该多向量结构具有与源矩阵列数相同的行数，
 * 并指定数量的列来存储多个向量。所有数据元素初始化为0。
 * 例如原来有一个2行3列的矩阵，指定num_vec=4,则会得到3行4列的矩阵，主维度为3，且数据全部初始化为0
 * 
 * @param des_vec 输出参数，指向新创建的多向量结构指针的指针
 * @param num_vec 要创建的目标向量数量（多向量的列数）
 * @param src_mat 输入参数，源矩阵指针，用于提供列数和数据布局信息
 * @param ops 操作接口
 * @return 无
 */
static void MultiVecCreateByMat(LAPACKVEC **des_vec, int num_vec, LAPACKMAT *src_mat, struct OPS_ *ops) {
    // 分配多向量主结构内存
    (*des_vec) = malloc(sizeof(LAPACKVEC));
    /* 初始化多向量维度参数 */
    (*des_vec)->nrows = src_mat->ncols;
    (*des_vec)->ncols = num_vec;
    (*des_vec)->ldd = (*des_vec)->nrows;
    /* 分配并初始化数据存储空间 */
    // 计算总元素数量：主维度 × 向量数量
    (*des_vec)->data = malloc(((*des_vec)->ldd) * ((*des_vec)->ncols) * sizeof(double));
    // 分配双精度浮点数组内存，并初始化为全零
    memset((*des_vec)->data, 0, ((*des_vec)->ldd) * ((*des_vec)->ncols) * sizeof(double));
    return;
}

/**
 * @brief 基于单个源向量创建多个向量的集合
 * 
 * 该函数根据提供的源向量属性，分配并初始化一个包含多个向量的集合。新向量集合的
 * 行数和主维度与源向量保持一致，列数由参数指定，数据内存初始化为全零。
 * 
 * @param des_vec 输出参数，指向新创建向量集合指针的指针(二级指针)
 * @param num_vec 需要创建的向量数量(新向量的列数)
 * @param src_vec 源向量指针，提供行数和主维度等基础属性
 * @param ops 操作接口
 * 
 * @return 无
 */
static void MultiVecCreateByMultiVec(LAPACKVEC **des_vec, int num_vec, LAPACKVEC *src_vec, struct OPS_ *ops) {
    // 分配多向量主结构内存
    (*des_vec) = malloc(sizeof(LAPACKVEC));
    // 从源向量继承维度属性
    (*des_vec)->nrows = src_vec->nrows;
    (*des_vec)->ncols = num_vec;
    (*des_vec)->ldd = src_vec->ldd;
    // 分配并初始化数据内存空间
    /* 计算总内存需求：主维度 × 列数 × 双精度浮点数大小
       使用malloc分配未初始化的堆内存 */
    (*des_vec)->data = malloc(((*des_vec)->ldd) * ((*des_vec)->ncols) * sizeof(double));
    // 将分配的内存区域清零初始化
    /* 使用memset将所有字节设置为0，确保数值初始状态为0.0 */
    memset((*des_vec)->data, 0, ((*des_vec)->ldd) * ((*des_vec)->ncols) * sizeof(double));
    return;
}
/**
 * @brief 销毁多个LAPACK向量结构体并释放相关内存资源
 * 
 * 用于清理LAPACK向量结构体分配的内存，重置结构体参数，并将指针置为NULL。
 * 注意：当前实现仅处理单个向量结构体，num_vec参数未被实际使用
 * 
 * @param des_vec 指向LAPACKVEC指针的二级指针（输入/输出参数）
 *                函数执行后会销毁目标结构体并将其指针置为NULL
 * @param num_vec 理论上表示向量数量（当前实现中未实际使用）
 * @param ops 操作接口
 * @return 无返回值
 */
static void MultiVecDestroy(LAPACKVEC **des_vec, int num_vec, struct OPS_ *ops) {
    (*des_vec)->nrows = 0;
    (*des_vec)->ncols = 0;
    (*des_vec)->ldd = 0;
    free((*des_vec)->data);
    free(*des_vec);
    *des_vec = NULL;
    return;
}
/**
 * @brief 从多列向量（LAPACKVEC 结构）中提取指定列，生成单列向量
 * 
 * @param multi_vec 输入的多列向量结构指针，包含原始数据矩阵信息
 * @param col 需要提取的列索引
 * @param vec 输出参数，指向新创建的单列向量结构指针的指针
 * @param ops 操作接口
 * @return 无返回值，结果通过vec参数返回
 **/
static void GetVecFromMultiVec(LAPACKVEC *multi_vec, int col, LAPACKVEC **vec, struct OPS_ *ops) {
    // 分配新向量结构体内存
    (*vec) = malloc(sizeof(LAPACKVEC));
    /* 初始化新向量维度信息 */
    (*vec)->nrows = multi_vec->nrows;
    (*vec)->ncols = 1;
    (*vec)->ldd = multi_vec->ldd;
    /* 计算目标列起始地址偏移
     * 使用列优先存储的计算方式：基地址 + 列索引 * leading dimension
     */
    (*vec)->data = multi_vec->data + multi_vec->ldd * col;
    return;
}

/**
 * @brief 销毁并释放指定的向量结构，将其从多维向量中移除
 * 
 * 该函数用于清理并释放由多维向量中指定列对应的向量结构。执行后，
 * 原向量指针将被置空，防止悬空指针。
 * 
 * @param multi_vec 指向多维向量结构的指针
 * @param col 需要操作的列索引
 * @param vec 指向需要销毁的向量指针的指针（双重指针用于修改原指针）
 * @param ops 操作接口
 * @return 无
 */
static void RestoreVecForMultiVec(LAPACKVEC *multi_vec, int col, LAPACKVEC **vec, struct OPS_ *ops) {
    /* 重置向量结构元数据 */
    (*vec)->nrows = 0;
    (*vec)->ncols = 0;
    (*vec)->ldd = 0;
    (*vec)->data = NULL;
    /* 完全释放向量内存并将指针置空 */
    free(*vec);
    *vec = NULL;
    return;
}

/**
 * @brief 以特定格式打印LAPACK向量/矩阵的指定列范围数据
 * 
 * 该函数按行遍历矩阵的每一行，在指定列范围内以科学计数法打印每个元素，
 * 元素间用制表符分隔，每行数据后换行。
 * 
 * @param x     指向LAPACKVEC结构的指针，包含矩阵数据和维度信息
 * @param start 需要打印的起始列索引（包含）
 * @param end   需要打印的结束列索引（不包含）
 * @param ops   包含Printf方法的操作结构体，用于实际输出操作
 * @return 无
 */
static void MultiVecView(LAPACKVEC *x, int start, int end, struct OPS_ *ops) {
    int row, col;
    double *destin;
    // 遍历矩阵的每一行
    for (row = 0; row < x->nrows; ++row) {
        // 遍历指定列范围
        for (col = start; col < end; ++col) {
            // 计算元素地址（列主序存储）：基地址 + 主维度*列号 + 行号
            destin = x->data + (x->ldd) * col + row;
            // 使用科学计数法打印元素值，保留4位小数
            ops->Printf("%6.4e\t", *destin);
        }
        // 完成一行数据打印后换行
        ops->Printf("\n");
    }
    return;
}
/**
 * @brief 计算多个向量的局部内积（或块内积），结果存储在指定矩阵中
 * 
 * 该函数通过调用DenseMatQtAP函数，对输入向量/矩阵的指定子块进行运算，
 * 计算结果矩阵的局部内积。主要用于处理分布式矩阵的局部块运算。
 * 
 * @param[in] nsdIP     字符参数，指定内积存储方式,主要用于传入至DenseMatQtAP('D'对角/'S'对称/'N'普通)
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
                                   LAPACKVEC *x, LAPACKVEC *y, int is_vec, int *start, int *end,
                                   double *inner_prod, int ldIP, struct OPS_ *ops) {
    int nrows = end[0] - start[0], ncols = end[1] - start[1];
    if (nrows > 0 && ncols > 0) {
        DenseMatQtAP('S', nsdIP, x->nrows, y->nrows, nrows, ncols,
                     1.0, x->data + x->ldd * start[0], (x->ldd), /* Q */
                     NULL, 0,                                    /* A */
                     y->data + y->ldd * start[1], (y->ldd),      /* P */
                     0.0, inner_prod, ldIP,
                     NULL);
    }
    return;
}

/**
 * @brief 计算多个向量的内积
 * 
 * 该函数是对MultiVecLocalInnerProd的封装，用于计算多个向量的内积。
 * 适用于需要分块计算或并行计算的情景，通过start/end参数指定计算范围。
 * 
 * @param[in] nsdIP     字符参数，指定内积存储方式（'D'对角/'S'对称/'N'普通），主要用于传入MultiVecLocalInnerProd中
 * @param[in] x         输入向量/矩阵
 * @param[in] y         输入向量/矩阵
 * @param[in] is_vec    向量模式标志位（0-矩阵模式，1-向量模式）
 * @param[in] start     计算区间的起始索引数组
 * @param[in] end       计算区间的结束索引数组
 * @param[out] inner_prod 输出内积结果数组（需预先分配内存）
 * @param[in] ldIP      inner_prod数组的leading dimension
 * @param[in] ops       运算控制参数结构体指针
 * 
 * @note 实际计算工作由MultiVecLocalInnerProd函数完成
 * @warning start/end数组长度应与计算分块数量匹配
 */
static void MultiVecInnerProd(char nsdIP,
                              LAPACKVEC *x, LAPACKVEC *y, int is_vec, int *start, int *end,
                              double *inner_prod, int ldIP, struct OPS_ *ops) {
    MultiVecLocalInnerProd(nsdIP, x, y, is_vec,
                           start, end, inner_prod, ldIP, ops);
    return;
}

/**
 * @brief 为多向量中的指定列范围设置随机值
 *
 * 该函数为LAPACKVEC结构中的指定列范围[start, end)填充随机生成的双精度浮点数。
 * 生成的数值范围为[0.0, 1.0)，使用标准库rand()函数生成。
 *
 * @param x     目标向量/矩阵结构指针，数据按列存储
 * @param start 起始列索引（包含）
 * @param end   结束列索引（不包含）
 * @param ops   操作接口
 */
static void MultiVecSetRandomValue(LAPACKVEC *x, int start, int end, struct OPS_ *ops) {
    int row, col;
    double *destin;
    // 遍历指定列范围
    for (col = start; col < end; ++col) {
        destin = x->data + x->ldd * col;
        for (row = 0; row < x->nrows; ++row) {
            // 生成[0.0, 1.0)范围内的随机浮点数
            *destin = ((double)rand()) / ((double)RAND_MAX + 1);
            ++destin;
        }
    }
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
static void MultiVecAxpby(double alpha, LAPACKVEC *x,
                          double beta, LAPACKVEC *y, int *start, int *end, struct OPS_ *ops) {
    // 校验列范围一致性，确保处理的列数相同
    assert(end[0] - start[0] == end[1] - start[1]);
    int length, ncols = end[1] - start[1];
    double *source, *destin;
    int inc = 1, col;
    // 处理边界条件：无有效列或空矩阵时直接返回
    if (ncols == 0) {
        return;
    }
    if (y->nrows == 0) {
        return;
    }
    // 当y矩阵是连续存储时（行数等于实际存储间距）
    if (y->nrows == y->ldd) {
        /* 批量处理所有列的数据 */
        length = y->nrows * ncols;              // 矩阵的元素个数
        destin = y->data + y->ldd * (start[1]); // destin指向矩阵存储所对应的起始位置
        // 处理beta系数：清零或缩放现有y值，结果储存回原地址
        if (beta == 0.0) {
            memset(destin, 0, length * sizeof(double)); // 不需要scaling，直接清空矩阵
        } else {
            if (beta != 1.0) {
                // 计算beta * y
                dscal(&length, &beta, destin, &inc);
            }
        }
        // 处理x向量的累加操作
        if (x != NULL) {
            assert(x->nrows == y->nrows);
            /* 当x矩阵也是连续存储时，批量处理所有列 */
            if (x->nrows == x->ldd) {
                length = y->nrows * ncols;
                source = x->data + x->ldd * (start[0]);
                destin = y->data + y->ldd * (start[1]);
                daxpy(&length, &alpha, source, &inc, destin, &inc);
            } else {
                /* 逐列处理非连续存储的x矩阵 */
                length = y->nrows;
                for (col = 0; col < ncols; ++col) {
                    source = x->data + x->ldd * (start[0] + col);
                    destin = y->data + y->ldd * (start[1] + col);
                    daxpy(&length, &alpha, source, &inc, destin, &inc);
                }
            }
        }
    } else {
        /* 逐列处理非连续存储的y矩阵 */
        length = y->nrows;
        for (col = 0; col < ncols; ++col) {
            destin = y->data + y->ldd * (start[1] + col);
            // 处理当前列的beta系数
            if (beta == 0.0) {
                memset(destin, 0, length * sizeof(double));
            } else {
                if (beta != 1.0) {
                    dscal(&length, &beta, destin, &inc);
                }
            }
            // 处理当前列的x向量累加
            if (x != NULL) {
                source = x->data + x->ldd * (start[0] + col);
                daxpy(&length, &alpha, source, &inc, destin, &inc);
            }
        }
    }
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
static void MatDotMultiVec(LAPACKMAT *mat, LAPACKVEC *x,
                           LAPACKVEC *y, int *start, int *end, struct OPS_ *ops) {
    // 校验输入参数的维度一致性
    assert(end[0] - start[0] == end[1] - start[1]);
    // 验证输入/输出向量是否为连续存储
    assert(y->nrows == y->ldd);
    assert(x->nrows == x->ldd);
    char charN = 'N'; // BLAS操作符，表示不转置
    double alpha = 1.0, beta = 0.0;
    int ncols = end[1] - start[1]; // 计算要处理的列块大小
    if (ncols == 0) return;        // 处理空操作情况
    // 当矩阵为空时执行向量块复制操作
    if (mat == NULL) {
        int incx = 1, incy = 1;
        ncols = y->nrows * (end[1] - start[1]);
        dcopy(&ncols, x->data + (x->ldd) * start[0], &incx,
              y->data + (y->ldd) * start[1], &incy);
    } else {
#if OPS_USE_OMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
        {
            // 并行化矩阵乘法：将列块分割给多个线程
            int id, length, offset;
            id = omp_get_thread_num();
            length = ncols / OMP_NUM_THREADS;
            offset = length * id;
            // 处理余数分配，前ncols%OMP_NUM_THREADS个线程多处理1列
            if (id < ncols % OMP_NUM_THREADS) {
                ++length;
                offset += id;
            } else {
                offset += ncols % OMP_NUM_THREADS;
            }
            // 执行分块矩阵乘法：y = mat * x
            dgemm(&charN, &charN, &y->nrows, &length, &x->nrows,
                  &alpha, mat->data, &mat->ldd,                            /* A */
                  x->data + x->ldd * (start[0] + offset), &x->ldd,         /* B */
                  &beta, y->data + y->ldd * (start[1] + offset), &y->ldd); /* C */
        }
#else
        dgemm(&charN, &charN, &y->nrows, &ncols, &x->nrows,
              &alpha, mat->data, &mat->ldd,                 /* A */
              x->data + x->ldd * start[0], &x->ldd,         /* B */
              &beta, y->data + y->ldd * start[1], &y->ldd); /* C */
#endif
    }
    return;
}

/**
 * @brief 计算矩阵转置与向量的乘积，并将结果累加到目标向量中
 * 
 * @param[in] mat    输入的稠密矩阵（LAPACKMAT类型指针）
 * @param[in] x      输入向量（LAPACKVEC类型指针）
 * @param[out] y     输出结果向量（LAPACKVEC类型指针），结果会被累加到此向量
 * @param[in] start  二维起始索引数组，start[0]为x向量的起始列，start[1]为y向量的起始列
 * @param[in] end    二维结束索引数组，end[0]为x向量的结束列，end[1]为y向量的结束列
 * @param[in] ops    操作接口
 * 
 * @warning 输入约束：
 * - 操作范围必须满足 end[0]-start[0] == end[1]-start[1]
 * - 输入/输出向量的长度必须与矩阵维度匹配（通过assert验证）
 */
static void MatTransDotMultiVec(LAPACKMAT *mat, LAPACKVEC *x,
                                LAPACKVEC *y, int *start, int *end, struct OPS_ *ops) {
    /* 参数校验：行列操作范围必须相等 */
    assert(end[0] - start[0] == end[1] - start[1]);
    /* 验证向量内存连续性 */
    assert(y->nrows == y->ldd);
    assert(x->nrows == x->ldd);
    char charN = 'N'; // 矩阵不转置
    double alpha = 1.0, beta = 0.0;
    int ncols = end[1] - start[1]; // 实际处理列数

/* 实现选择：默认使用扩展矩阵乘法 */
#if 1
    // 计算
    DenseMatQtAP(charN, charN, x->nrows, x->nrows, y->nrows, ncols,
                 alpha, mat->data, mat->ldd,                  /* Q */
                 NULL, 0,                                     /* A */
                 x->data + (x->ldd) * start[0], x->ldd,       /* P */
                 beta, y->data + (y->ldd) * start[1], y->ldd, /* C */
                 NULL);
#else
    dgemm(&charT, &charN, &(y->nrows), &ncols, &x->nrows,
          &alpha, mat->data, &mat->ldd,                   /* A */
          x->data + (x->ldd) * start[0], &x->ldd,         /* B */
          &beta, y->data + (y->ldd) * start[1], &y->ldd); /* C */
#endif
    return;
}

/**
* @brief 执行向量/矩阵的线性组合计算 y = beta*y + x*coef
* 
* 
* @param[in] x       输入矩阵/向量，LAPACKVEC结构指针，NULL表示不参与计算
* @param[in,out] y   输入输出矩阵/向量，LAPACKVEC结构指针，存储计算结果
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
static void MultiVecLinearComb(
    LAPACKVEC *x, LAPACKVEC *y, int is_vec,
    int *start, int *end,
    double *coef, int ldc,
    double *beta, int incb, struct OPS_ *ops) {
    int nrows, ncols, col, inc, length;
    char charN = 'N';
    double one = 1.0, gamma, *destin;
    /* coef ������������ */
    /* 计算子矩阵的维度 */
    nrows = end[0] - start[0];
    ncols = end[1] - start[1];
    /* 空矩阵或无效y矩阵时直接返回 */
    if (nrows == 0 || ncols == 0) {
        return;
    }
    if (y->nrows == 0) {
        return;
    }

    /* 处理beta系数：决定是否对y进行缩放 */
    if (beta == NULL) {
        gamma = 0.0;
    } else {
        gamma = 1.0;
        inc = 1;
        /* 根据步长模式处理y的缩放 */
        if (incb == 0) {
            if ((*beta) != 1.0) {
                /* 整块连续内存时单次缩放 */
                if (y->ldd == y->nrows) {
                    destin = y->data + y->ldd * (start[1]);
                    length = (y->nrows) * ncols;
                    dscal(&length, beta, destin, &inc);
                    /* 非连续内存时逐列缩放 */
                } else {
                    for (col = 0; col < ncols; ++col) {
                        destin = y->data + y->ldd * (start[1] + col);
                        dscal(&(y->nrows), beta, destin, &inc);
                    }
                }
            }
        }
        /* 带步长的逐列缩放 */
        else {
            for (col = 0; col < ncols; ++col) {
                destin = y->data + y->ldd * (start[1] + col);
                dscal(&(y->nrows), beta + incb * col, destin, &inc);
            }
        }
    }
    /* 执行x与系数的矩阵乘法并累加到y */
    if (x != NULL && coef != NULL) {
#if OPS_USE_OMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
        {
            /* 并行任务分配 */
            int id, length, offset;
            id = omp_get_thread_num();
            length = ncols / OMP_NUM_THREADS;
            offset = length * id;
            /* 处理余数分配 */
            if (id < ncols % OMP_NUM_THREADS) {
                ++length;
                offset += id;
            } else {
                offset += ncols % OMP_NUM_THREADS;
            }
            // 计算y = beta*y + x*coef
            dgemm(&charN, &charN, &y->nrows, &length, &nrows,
                  &one, x->data + x->ldd * start[0], &(x->ldd),               /* A */
                  coef + offset * ldc, &ldc,                                  /* B */
                  &gamma, y->data + y->ldd * (start[1] + offset), &(y->ldd)); /* C */
        }
#else
        dgemm(&charN, &charN, &y->nrows, &ncols, &nrows,
              &one, x->data + x->ldd * start[0], &(x->ldd),    /* A */
              coef, &ldc,                                      /* B */
              &gamma, y->data + y->ldd * start[1], &(y->ldd)); /* C */
#endif
    }
    return;
}

/**
 * @brief 计算密集矩阵运算 Q^T * A * P 或相关变体，结果存入 qAp 数组
 * 
 * @param ntsA         矩阵A的存储标识符，'S'表示对称矩阵（自动转为'L'下三角），其他字符见DenseMatQtAP说明
 * @param ntsdQAP      运算模式标识符，'T'表示需要对结果进行特殊转置存储
 * @param mvQ          输入矩阵Q的向量结构体指针
 * @param matA         输入矩阵A的结构体指针（可为NULL）
 * @param mvP          输入矩阵P的向量结构体指针
 * @param is_vec       指示是否为向量
 * @param start        二维数组指针，指定列起始索引[start[0], start[1]]
 * @param end          二维数组指针，指定列结束索引[end[0], end[1]]
 * @param qAp          输出结果存储数组指针
 * @param ldQAP        qAp数组的行主维度（leading dimension）
 * @param vec_ws       工作空间向量指针，用于临时存储
 * @param ops          操作接口
 */
static void MultiVecQtAP(char ntsA, char ntsdQAP,
                         LAPACKVEC *mvQ, LAPACKMAT *matA, LAPACKVEC *mvP, int is_vec,
                         int *start, int *end, double *qAp, int ldQAP,
                         LAPACKVEC *vec_ws, struct OPS_ *ops) {
    double alpha = 1.0, beta = 0.0, *matA_data;
    int matA_ldd;
    int nrows = end[0] - start[0], ncols = end[1] - start[1];
    // 处理空矩阵的边界情况
    if (nrows == 0 || ncols == 0) return;
    // 转换对称矩阵标识符（'S' -> 'L'）
    if (ntsA == 'S') ntsA = 'L';
    // 初始化矩阵A参数（处理NULL情况）
    if (matA == NULL) {
        matA_data = NULL;
        matA_ldd = 0;
    } else {
        matA_data = matA->data;
        matA_ldd = matA->ldd;
    }
    // 转置存储模式处理分支
    if (ntsdQAP == 'T') {
        // 分配临时存储空间用于转置操作
        double *dbl_ws = malloc(nrows * ncols * sizeof(double));
        /* 调用底层矩阵运算：结果存入临时空间dbl_ws
           - 使用'N'表示不转置原始计算结果
           - 计算结果按列优先存储 */
        DenseMatQtAP(ntsA, 'N',
                     mvQ->nrows, mvP->nrows, /* matA �������� */
                     nrows, ncols,           /* matC �������� */
                     alpha, mvQ->data + mvQ->ldd * start[0], mvQ->ldd,
                     matA_data, matA_ldd,
                     mvP->data + mvP->ldd * start[1], mvP->ldd,
                     beta, dbl_ws, nrows,
                     vec_ws->data);
        // 将列优先的临时结果转置存储到行优先的qAp数组
        double *source, *destin;
        int incx, incy, row;
        source = dbl_ws;
        incx = nrows; // 源数据列步长
        destin = qAp;
        incy = 1; // 目标数据行步长
        for (row = 0; row < nrows; ++row) {
            /*复制数据*/
            dcopy(&ncols, source, &incx, destin, &incy);
            source += 1;     // 移动到下一列起始位置
            destin += ldQAP; // 移动到下一行起始位置
        }
        free(dbl_ws);
    } else {
        // 直接模式：结果直接存入qAp数组
        DenseMatQtAP(ntsA, ntsdQAP,
                     mvQ->nrows, mvP->nrows, /* matA �������� */
                     nrows, ncols,           /* matC �������� */
                     alpha, mvQ->data + mvQ->ldd * start[0], mvQ->ldd,
                     matA_data, matA_ldd,
                     mvP->data + mvP->ldd * start[1], mvP->ldd,
                     beta, qAp, ldQAP,
                     vec_ws->data);
    }
    return;
}

/* vec */
/**
 * @brief 基于现有向量创建新向量
 *
 * 该函数是对MultiVecCreateByVec函数的封装，专门用于创建单个向量的场景。
 * 通过设置nvec参数为1，简化单向量的创建过程。
 * 
 * @param[out] des_vec 输出参数，指向新创建向量指针的指针。调用成功后*des_vec将指向新分配的内存
 * @param[in] src_vec 输入参数，作为创建模板的源向量指针。新向量将继承源向量的维度等属性
 * @param[in] ops 操作指针
 */
static void VecCreateByVec(LAPACKVEC **des_vec, LAPACKVEC *src_vec, struct OPS_ *ops) {
    MultiVecCreateByVec(des_vec, 1, src_vec, ops);
    return;
}

/**
 * @brief 通过矩阵创建向量对象
 * 
 * @param[out] des_vec 指向LAPACKVEC指针的指针，用于接收新创建的向量对象
 * @param[in] src_mat 输入矩阵对象指针，作为向量的数据源
 * @param[in] ops 操作接口
 * 
 * @note 本函数通过调用MultiVecCreateByMat实现单向量创建：
 * - 固定第一个参数为1，表示创建单个向量
 * - 将矩阵结构转换为向量结构
 * 
 */
static void VecCreateByMat(LAPACKVEC **des_vec, LAPACKMAT *src_mat, struct OPS_ *ops) {
    MultiVecCreateByMat(des_vec, 1, src_mat, ops);
    return;
}

/**
 * @brief 销毁LAPACK向量对象并释放相关资源
 *
 * 该函数通过调用MultiVecDestroy接口来释放由des_vec指向的LAPACK向量对象。
 * 调用完成后，会将被释放的向量指针置为NULL，避免野指针问题。
 *
 * @param[in,out] des_vec 指向LAPACK向量指针的二级指针。函数执行后，
 *                        *des_vec将被置为NULL
 * @param[in] ops 操作接口
 *
 * @note 本函数实际调用MultiVecDestroy时传入的num参数固定为1，
 *       表示处理单个向量对象的销毁
 */
static void VecDestroy(LAPACKVEC **des_vec, struct OPS_ *ops) {
    MultiVecDestroy(des_vec, 1, ops);
    return;
}

/**
 * @brief 查看/显示LAPACK向量的内容
 * 
 * 该函数通过调用MultiVecView实现向量查看功能，默认从第0个元素开始，
 * 以步长1显示整个向量的内容。
 * 
 * @param[in] x 待查看的LAPACK向量对象指针，不可为空
 * @param[in] ops 操作控制结构体指针，包含底层实现所需的上下文信息
 * 
 * @return 无
 */
static void VecView(LAPACKVEC *x, struct OPS_ *ops) {
    MultiVecView(x, 0, 1, ops);
    return;
}
/**
 * @brief 计算两个向量的内积（静态函数）
 *
 * @param[in] x 输入向量1指针，指向LAPACK向量结构体
 * @param[in] y 输入向量2指针，指向LAPACK向量结构体
 * @param[out] inner_prod 计算结果输出指针，存储标量内积值
 * @param[in] ops 运算控制结构体指针，包含底层运算接口
 *
 */
static void VecInnerProd(LAPACKVEC *x, LAPACKVEC *y, double *inner_prod, struct OPS_ *ops) {
    /* 定义计算范围的起始和结束索引 */
    int start[2] = {0, 0}, end[2] = {1, 1};
    /* 调用多维向量内积计算函数
     * 参数说明：
     * 'S' - 对称计算模式
     * 0 - 保留参数（未使用）
     * start/end - 子向量范围
     * 1 - 步长参数
     */
    MultiVecInnerProd('S', x, y, 0, start, end, inner_prod, 1, ops);
    return;
}

/**
 * @brief 计算两个向量的局部内积
 * 
 * 通过调用MultiVecLocalInnerProd函数实现，使用'S'模式在指定索引范围内进行计算，
 * 结果存储在inner_prod中。
 * 
 * @param[in] x 第一个输入向量指针
 * @param[in] y 第二个输入向量指针
 * @param[out] inner_prod 存储计算结果的双精度浮点数指针
 * @param[in] ops 操作上下文结构体指针，包含执行环境或配置信息
 * 
 */
static void VecLocalInnerProd(LAPACKVEC *x, LAPACKVEC *y, double *inner_prod, struct OPS_ *ops) {
    int start[2] = {0, 0}, end[2] = {1, 1};
    MultiVecLocalInnerProd('S', x, y, 0, start, end, inner_prod, 1, ops);
    return;
}

/**
 * @brief 为向量设置随机数值
 * 
 * 使用标准随机数生成器填充目标向量的全部元素。本函数通过调用多向量工具函数
 * 实现核心功能，采用默认范围[0,1)的随机数分布。
 * 
 * @param[in,out] x   指向LAPACKVEC结构的指针，接收随机数值的目标向量。
 *                    - 输入时应为已分配内存的向量结构
 *                    - 输出时将包含新生成的随机数值
 * @param[in] ops    操作接口
 * 
 */
static void VecSetRandomValue(LAPACKVEC *x, struct OPS_ *ops) {
    MultiVecSetRandomValue(x, 0, 1, ops);
    return;
}

/**
 * @brief 执行向量线性组合运算 alpha * x + beta * y
 *
 * @param[in] alpha 标量系数，作用于x向量的乘数
 * @param[in] x 输入向量指针，使用LAPACKVEC结构描述的向量
 * @param[in] beta 标量系数，作用于y向量的乘数
 * @param[in,out] y 输入输出向量指针，结果将存储在此向量中
 * @param[in] ops 操作控制结构体指针，包含底层运算所需的控制参数
 *
 */
static void VecAxpby(double alpha, LAPACKVEC *x, double beta, LAPACKVEC *y, struct OPS_ *ops) {
    int start[2] = {0, 0}, end[2] = {1, 1};
    MultiVecAxpby(alpha, x, beta, y, start, end, ops);
    return;
}

/**
 * @brief 计算矩阵与向量的点乘运算
 * 
 * 本函数通过调用底层MatDotMultiVec接口，使用预定义的范围参数完成矩阵-向量乘法。
 * 
 * @param[in] mat 输入矩阵指针，要求矩阵已正确初始化
 * @param[in] x 输入向量指针，参与计算的向量数据
 * @param[out] y 输出向量指针，存储计算结果的内存区域
 * @param[in] ops 操作接口
 * 
 */
static void MatDotVec(LAPACKMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops) {
    /*start/end参数固定为{{0,0},{1,1}}，即提取的是单个向量运算*/
    int start[2] = {0, 0}, end[2] = {1, 1};
    MatDotMultiVec(mat, x, y, start, end, ops);
    return;
}
/**
 * @brief 计算矩阵转置与向量的点积（封装为单向量版本）
 * 
 * 该函数通过设置固定的起始/结束索引，调用多向量版本的矩阵转置点积函数，
 * 实现单向量运算的简化封装。适用于处理单个向量的矩阵转置乘法场景。
 * 
 * @param[in] mat  输入矩阵对象指针，须为有效的LAPACK格式矩阵
 * @param[in] x    输入向量指针，维度应与矩阵列数匹配
 * @param[out] y   输出向量指针，用于存储计算结果，须预分配内存
 * @param[in] ops  运算控制参数指针，包含底层BLAS运算所需配置
 * 
 * @note 通过硬编码的start=[0,0]和end=[1,1]参数，将多向量运算限制为单向量处理。
 *       实际运算委托给MatTransDotMultiVec函数实现
 */
static void MatTransDotVec(LAPACKMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops) {
    int start[2] = {0, 0}, end[2] = {1, 1};
    MatTransDotMultiVec(mat, x, y, start, end, ops);
    return;
}

/**
 * @brief 打印LAPACK矩阵内容到输出流
 *
 * @param[in] mat  指向LAPACKMAT矩阵结构的指针，包含矩阵维度、数据指针和行间距
 * @param[in] ops  输出接口
 *
 * @note 矩阵元素按行主序打印，每列元素用制表符分隔，行末换行
 * @warning 矩阵数据指针必须已正确初始化，ldd值应大于等于矩阵行数
 */
static void MatView(LAPACKMAT *mat, struct OPS_ *ops) {
    int row, col;
    double *destin;
    // 遍历矩阵所有行
    for (row = 0; row < mat->nrows; ++row) {
        // 遍历当前行的所有列
        for (col = 0; col < mat->ncols; ++col) {
            /* 计算元素地址：基地址 + 列号*行间距 + 行偏移
               对应LAPACK的列主序存储模式 */
            destin = mat->data + (mat->ldd) * col + row;
            // 完成一行输出后换行
            ops->Printf("%6.4e\t", *destin);
        }
        ops->Printf("\n");
    }
    return;
}
/* length: length of dbl_ws >= 2*N+(N+1)*NB + min(M,N),
 * where NB is the optimal blocksize and N = end[0]-start
 * length of int_ws is N */
static void DenseMatOrth(double *mat, int nrows, int ldm,
                         int start, int *end, double orth_zero_tol,
                         double *dbl_ws, int length, int *int_ws) {
    /* ȥ��x1�е�x0���� */
    if (start > 0) {
        double *beta, *coef;
        int start_x01[2], end_x01[2], idx;
        int length, inc;
        LAPACKVEC x0, x1;
        x0.nrows = nrows;
        x0.ncols = start;
        x0.ldd = ldm;
        x1.nrows = nrows;
        x1.ncols = *end - start;
        x1.ldd = ldm;
        x0.data = mat;
        x1.data = mat + ldm * start;
        start_x01[0] = 0;
        end_x01[0] = x0.ncols;
        start_x01[1] = 0;
        end_x01[1] = x1.ncols;
        beta = dbl_ws;
        coef = dbl_ws + 1;
        for (idx = 0; idx < 2; ++idx) {
            MultiVecInnerProd('N', &x0, &x1, 0, start_x01, end_x01,
                              coef, x0.ncols, NULL);
            *beta = -1.0;
            inc = 1;
            length = x0.ncols * x1.ncols;
            dscal(&length, beta, coef, &inc);
            *beta = 1.0;
            MultiVecLinearComb(&x0, &x1, 0, start_x01, end_x01,
                               coef, end_x01[0] - start_x01[0], beta, 0, NULL);
        }
    }
    /* ��ѡ��Ԫ��QR�ֽ� */
    int m, n, k, lda, *jpvt, lwork, info, col;
    double *a, *tau, *work;
    m = nrows;
    n = *end - start;
    a = mat + ldm * start;
    jpvt = int_ws;
    tau = dbl_ws;
    lda = ldm;
    work = tau + n;
    lwork = length - n;
    for (col = 0; col < n; ++col) {
        jpvt[col] = 0;
    }
    dgeqp3(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
    /* �õ�a����, ������n, a�ĶԽ��ߴ洢��r_ii */
    for (col = n - 1; col >= 0; --col) {
        if (fabs(a[lda * col + col]) > orth_zero_tol) break;
    }
    n = col + 1;
    k = m < n ? m : n;
    /* ����Q, ����ɶ�x1�������� */
    dorgqr(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    /* ��ǵõ��������������ĩβ */
    *end = start + n;
    return;
}

/* Encapsulation */
static void LAPACK_MatView(void *mat, struct OPS_ *ops) {
    MatView((LAPACKMAT *)mat, ops);
    return;
}
/* vec */
static void LAPACK_VecCreateByVec(void **des_vec, void *src_vec, struct OPS_ *ops) {
    VecCreateByVec((LAPACKVEC **)des_vec, (LAPACKVEC *)src_vec, ops);
    return;
}
static void LAPACK_VecCreateByMat(void **des_vec, void *src_mat, struct OPS_ *ops) {
    VecCreateByMat((LAPACKVEC **)des_vec, (LAPACKMAT *)src_mat, ops);
    return;
}
static void LAPACK_VecDestroy(void **des_vec, struct OPS_ *ops) {
    VecDestroy((LAPACKVEC **)des_vec, ops);
    return;
}
static void LAPACK_VecView(void *x, struct OPS_ *ops) {
    VecView((LAPACKVEC *)x, ops);
    return;
}
static void LAPACK_VecInnerProd(void *x, void *y, double *inner_prod, struct OPS_ *ops) {
    VecInnerProd((LAPACKVEC *)x, (LAPACKVEC *)y, inner_prod, ops);
    return;
}
static void LAPACK_VecLocalInnerProd(void *x, void *y, double *inner_prod, struct OPS_ *ops) {
    VecLocalInnerProd((LAPACKVEC *)x, (LAPACKVEC *)y, inner_prod, ops);
    return;
}
static void LAPACK_VecSetRandomValue(void *x, struct OPS_ *ops) {
    VecSetRandomValue((LAPACKVEC *)x, ops);
    return;
}
static void LAPACK_VecAxpby(double alpha, void *x, double beta, void *y, struct OPS_ *ops) {
    VecAxpby(alpha, (LAPACKVEC *)x, beta, (LAPACKVEC *)y, ops);
    return;
}
static void LAPACK_MatDotVec(void *mat, void *x, void *y, struct OPS_ *ops) {
    MatDotVec((LAPACKMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
    return;
}
static void LAPACK_MatTransDotVec(void *mat, void *x, void *y, struct OPS_ *ops) {
    MatTransDotVec((LAPACKMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
    return;
}
/* multi-vec */
static void LAPACK_MultiVecCreateByVec(void ***des_vec, int num_vec, void *src_vec, struct OPS_ *ops) {
    MultiVecCreateByVec((LAPACKVEC **)des_vec, num_vec, (LAPACKVEC *)src_vec, ops);
    return;
}
static void LAPACK_MultiVecCreateByMat(void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops) {
    MultiVecCreateByMat((LAPACKVEC **)des_vec, num_vec, (LAPACKMAT *)src_mat, ops);
    return;
}
static void LAPACK_MultiVecCreateByMultiVec(void ***des_vec, int num_vec, void **src_vec, struct OPS_ *ops) {
    MultiVecCreateByMultiVec((LAPACKVEC **)des_vec, num_vec, (LAPACKVEC *)src_vec, ops);
    return;
}
static void LAPACK_MultiVecDestroy(void ***des_vec, int num_vec, struct OPS_ *ops) {
    MultiVecDestroy((LAPACKVEC **)des_vec, num_vec, ops);
    return;
}
static void LAPACK_GetVecFromMultiVec(void **multi_vec, int col, void **vec, struct OPS_ *ops) {
    GetVecFromMultiVec((LAPACKVEC *)multi_vec, col, (LAPACKVEC **)vec, ops);
    return;
}
static void LAPACK_RestoreVecForMultiVec(void **multi_vec, int col, void **vec, struct OPS_ *ops) {
    RestoreVecForMultiVec((LAPACKVEC *)multi_vec, col, (LAPACKVEC **)vec, ops);
    return;
}
static void LAPACK_MultiVecView(void **x, int start, int end, struct OPS_ *ops) {
    MultiVecView((LAPACKVEC *)x, start, end, ops);
    return;
}
static void LAPACK_MultiVecLocalInnerProd(char nsdIP,
                                          void **x, void **y, int is_vec, int *start, int *end,
                                          double *inner_prod, int ldIP, struct OPS_ *ops) {
    MultiVecLocalInnerProd(nsdIP,
                           (LAPACKVEC *)x, (LAPACKVEC *)y, is_vec, start, end,
                           inner_prod, ldIP, ops);
    return;
}
static void LAPACK_MultiVecInnerProd(char nsdIP,
                                     void **x, void **y, int is_vec, int *start, int *end,
                                     double *inner_prod, int ldIP, struct OPS_ *ops) {
    MultiVecInnerProd(nsdIP,
                      (LAPACKVEC *)x, (LAPACKVEC *)y, is_vec, start, end,
                      inner_prod, ldIP, ops);
    return;
}
static void LAPACK_MultiVecSetRandomValue(void **x, int start, int end, struct OPS_ *ops) {
    MultiVecSetRandomValue((LAPACKVEC *)x, start, end, ops);
    return;
}
static void LAPACK_MultiVecAxpby(double alpha, void **x,
                                 double beta, void **y, int *start, int *end, struct OPS_ *ops) {
    MultiVecAxpby(alpha, (LAPACKVEC *)x,
                  beta, (LAPACKVEC *)y, start, end, ops);
    return;
}
static void LAPACK_MatDotMultiVec(void *mat, void **x,
                                  void **y, int *start, int *end, struct OPS_ *ops) {
    MatDotMultiVec((LAPACKMAT *)mat, (LAPACKVEC *)x,
                   (LAPACKVEC *)y, start, end, ops);
    return;
}
static void LAPACK_MatTransDotMultiVec(void *mat, void **x,
                                       void **y, int *start, int *end, struct OPS_ *ops) {
    MatTransDotMultiVec((LAPACKMAT *)mat, (LAPACKVEC *)x,
                        (LAPACKVEC *)y, start, end, ops);
    return;
}
static void LAPACK_MultiVecLinearComb(
    void **x, void **y, int is_vec,
    int *start, int *end,
    double *coef, int ldc,
    double *beta, int incb, struct OPS_ *ops) {
    MultiVecLinearComb(
        (LAPACKVEC *)x, (LAPACKVEC *)y, is_vec,
        start, end,
        coef, ldc,
        beta, incb, ops);
    return;
}
static void LAPACK_MultiVecQtAP(char ntsA, char nsdQAP,
                                void **mvQ, void *matA, void **mvP, int is_vec,
                                int *start, int *end, double *qAp, int ldQAP,
                                void **mv_ws, struct OPS_ *ops) {
    MultiVecQtAP(ntsA, nsdQAP,
                 (LAPACKVEC *)mvQ, (LAPACKMAT *)matA, (LAPACKVEC *)mvP, is_vec,
                 start, end, qAp, ldQAP,
                 (LAPACKVEC *)mv_ws, ops);
    return;
}

void MultiGridCreate(LAPACKMAT ***A_array, LAPACKMAT ***B_array, LAPACKMAT ***P_array,
                     int *num_levels, LAPACKMAT *A, LAPACKMAT *B, struct OPS_ *ops) {
    ops->Printf("Just a test, P is fixed\n");
    int nrows, ncols, row, col, level;
    double *dbl_ws;

    (*A_array) = malloc((*num_levels) * sizeof(LAPACKMAT *));
    (*P_array) = malloc((*num_levels - 1) * sizeof(LAPACKMAT *));

    (*A_array)[0] = A;
    for (level = 1; level < *num_levels; ++level) {
        (*A_array)[level] = malloc(sizeof(LAPACKMAT));
        (*P_array)[level - 1] = malloc(sizeof(LAPACKMAT));
    }
    nrows = A->nrows;
    ncols = (nrows - 1) / 2;

    dbl_ws = malloc(nrows * ncols * sizeof(double));
    memset(dbl_ws, 0, nrows * ncols * sizeof(double));
    for (level = 1; level < *num_levels; ++level) {
        ops->Printf("nrows = %d, ncols = %d\n", nrows, ncols);
        (*P_array)[level - 1]->nrows = nrows;
        (*P_array)[level - 1]->ncols = ncols;
        (*P_array)[level - 1]->ldd = nrows;
        (*P_array)[level - 1]->data = malloc(nrows * ncols * sizeof(double));
        memset((*P_array)[level - 1]->data, 0, nrows * ncols * sizeof(double));
        for (col = 0; col < ncols; ++col) {
            row = col * 2 + 1;
            (*P_array)[level - 1]->data[nrows * col + row] = 1.0;
            (*P_array)[level - 1]->data[nrows * col + row - 1] = 0.5;
            (*P_array)[level - 1]->data[nrows * col + row + 1] = 0.5;
        }
        (*A_array)[level]->nrows = ncols;
        (*A_array)[level]->ncols = ncols;
        (*A_array)[level]->ldd = ncols;
        (*A_array)[level]->data = malloc(ncols * ncols * sizeof(double));
        memset((*A_array)[level]->data, 0, ncols * ncols * sizeof(double));
        DenseMatQtAP('L', 'S', nrows, nrows, ncols, ncols,
                     1.0, (*P_array)[level - 1]->data, nrows,
                     (*A_array)[level - 1]->data, nrows,
                     (*P_array)[level - 1]->data, nrows,
                     0.0, (*A_array)[level]->data, ncols, dbl_ws);

        nrows = ncols;
        ncols = (nrows - 1) / 2;
    }
    if (B != NULL) {
        (*B_array) = malloc((*num_levels) * sizeof(LAPACKMAT *));
        (*B_array)[0] = B;
        for (level = 1; level < *num_levels; ++level) {
            (*B_array)[level] = malloc(sizeof(LAPACKMAT));
        }
        nrows = B->nrows;
        ncols = (nrows - 1) / 2;
        for (level = 1; level < *num_levels; ++level) {
            (*B_array)[level]->nrows = ncols;
            (*B_array)[level]->ncols = ncols;
            (*B_array)[level]->ldd = ncols;
            (*B_array)[level]->data = malloc(ncols * ncols * sizeof(double));
            memset((*B_array)[level]->data, 0, ncols * ncols * sizeof(double));
            DenseMatQtAP('L', 'S', nrows, nrows, ncols, ncols,
                         1.0, (*P_array)[level - 1]->data, nrows,
                         (*B_array)[level - 1]->data, nrows,
                         (*P_array)[level - 1]->data, nrows,
                         0.0, (*B_array)[level]->data, ncols, dbl_ws);
            nrows = ncols;
            ncols = (nrows - 1) / 2;
        }
    }
    free(dbl_ws);
    return;
}
void MultiGridDestroy(LAPACKMAT ***A_array, LAPACKMAT ***B_array, LAPACKMAT ***P_array,
                      int *num_levels, struct OPS_ *ops) {
    int level;
    (*A_array)[0] = NULL;
    for (level = 1; level < *num_levels; ++level) {
        free((*A_array)[level]->data);
        (*A_array)[level]->data = NULL;
        free((*A_array)[level]);
        (*A_array)[level] = NULL;
        free((*P_array)[level - 1]->data);
        (*P_array)[level - 1]->data = NULL;
        free((*P_array)[level - 1]);
        (*P_array)[level - 1] = NULL;
    }
    free(*A_array);
    *A_array = NULL;
    free(*P_array);
    *P_array = NULL;

    if (B_array != NULL) {
        (*B_array)[0] = NULL;
        for (level = 1; level < *num_levels; ++level) {
            free((*B_array)[level]->data);
            (*B_array)[level]->data = NULL;
            free((*B_array)[level]);
            (*B_array)[level] = NULL;
        }
        free(*B_array);
        *B_array = NULL;
    }
    return;
}

static void LAPACK_MultiGridCreate(void ***A_array, void ***B_array, void ***P_array,
                                   int *num_levels, void *A, void *B, struct OPS_ *ops) {
    MultiGridCreate((LAPACKMAT ***)A_array, (LAPACKMAT ***)B_array, (LAPACKMAT ***)P_array,
                    num_levels, (LAPACKMAT *)A, (LAPACKMAT *)B, ops);
    return;
}
static void LAPACK_MultiGridDestroy(void ***A_array, void ***B_array, void ***P_array,
                                    int *num_levels, struct OPS_ *ops) {
    MultiGridDestroy((LAPACKMAT ***)A_array, (LAPACKMAT ***)B_array, (LAPACKMAT ***)P_array, num_levels, ops);
    return;
}

void OPS_LAPACK_Set(struct OPS_ *ops) {
    ops->Printf = DefaultPrintf;
    ops->GetWtime = DefaultGetWtime;
    ops->GetOptionFromCommandLine = DefaultGetOptionFromCommandLine;
    ops->MatView = LAPACK_MatView;
    /* vec */
    ops->VecCreateByMat = LAPACK_VecCreateByMat;
    ops->VecCreateByVec = LAPACK_VecCreateByVec;
    ops->VecDestroy = LAPACK_VecDestroy;
    ops->VecView = LAPACK_VecView;
    ops->VecInnerProd = LAPACK_VecInnerProd;
    ops->VecLocalInnerProd = LAPACK_VecLocalInnerProd;
    ops->VecSetRandomValue = LAPACK_VecSetRandomValue;
    ops->VecAxpby = LAPACK_VecAxpby;
    ops->MatDotVec = LAPACK_MatDotVec;
    ops->MatTransDotVec = LAPACK_MatTransDotVec;
    /* multi-vec */
    ops->MultiVecCreateByMat = LAPACK_MultiVecCreateByMat;
    ops->MultiVecCreateByVec = LAPACK_MultiVecCreateByVec;
    ops->MultiVecCreateByMultiVec = LAPACK_MultiVecCreateByMultiVec;
    ops->MultiVecDestroy = LAPACK_MultiVecDestroy;
    ops->GetVecFromMultiVec = LAPACK_GetVecFromMultiVec;
    ops->RestoreVecForMultiVec = LAPACK_RestoreVecForMultiVec;
    ops->MultiVecView = LAPACK_MultiVecView;
    ops->MultiVecLocalInnerProd = LAPACK_MultiVecLocalInnerProd;
    ops->MultiVecInnerProd = LAPACK_MultiVecInnerProd;
    ops->MultiVecSetRandomValue = LAPACK_MultiVecSetRandomValue;
    ops->MultiVecAxpby = LAPACK_MultiVecAxpby;
    ops->MultiVecLinearComb = LAPACK_MultiVecLinearComb;
    ops->MatDotMultiVec = LAPACK_MatDotMultiVec;
    ops->MatTransDotMultiVec = LAPACK_MatTransDotMultiVec;
    if (0)
        ops->MultiVecQtAP = LAPACK_MultiVecQtAP;
    else
        ops->MultiVecQtAP = DefaultMultiVecQtAP;
    /* dense mat */
    ops->lapack_ops = NULL;
    ops->DenseMatQtAP = DenseMatQtAP;
    ops->DenseMatOrth = DenseMatOrth;
    /* multi grid */
    ops->MultiGridCreate = LAPACK_MultiGridCreate;
    ops->MultiGridDestroy = LAPACK_MultiGridDestroy;

    printf("lapack_set\n");
    return;
}
