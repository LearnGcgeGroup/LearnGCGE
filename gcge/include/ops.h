/**
 * @file ops.h
 * @brief 核心操作接口和基本数据结构
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef _OPS_H_
#define _OPS_H_

#include "ops_config.h"

#if OPS_USE_OMP
#include <omp.h>
#endif

#if OPS_USE_INTEL_MKL
#include <mkl.h>
#include <mkl_spblas.h>
#include <omp.h>
#endif

#if OPS_USE_MPI
#include <mpi.h>

extern double *debug_ptr;
int CreateMPIDataTypeSubMat(MPI_Datatype *submat_type,
                            int nrows, int ncols, int ldA);
int DestroyMPIDataTypeSubMat(MPI_Datatype *submat_type);
int CreateMPIOpSubMatSum(MPI_Op *op);
int DestroyMPIOpSubMatSum(MPI_Op *op);
#endif
int SplitDoubleArray(double *destin, int length,
                     int num_group, double min_gap, int min_num, int *displs,
                     double *dbl_ws, int *int_ws);

typedef struct OPS_ {
    void (*Printf)(const char *fmt, ...);
    double (*GetWtime)(void);
    int (*GetOptionFromCommandLine)(
        const char *name, char type, void *data,
        int argc, char *argv[], struct OPS_ *ops);
    /* mat */
    /**
    * @brief 打印擦除类型的矩阵内容到输出流
    *
    * @param[in] mat  指向擦除类型矩阵结构的指针
    * @param[in] ops  操作接口
    *
    */
    void (*MatView)(void *mat, struct OPS_ *ops);
    /* y = alpha x + beta y */
    /**
    * @brief 执行矩阵的线性组合操作：alpha * X + beta * Y

    * @param[in] alpha 第一个矩阵的标量系数
    * @param[in] matX  指向第一个矩阵数据的指针，矩阵的具体格式由实现定义
    * @param[in] beta  第二个矩阵的标量系数
    * @param[in] matY  指向第二个矩阵数据的指针，矩阵维度应与matX一致
    * @param[in] ops   操作接口
    */
    void (*MatAxpby)(double alpha, void *matX, double beta, void *matY, struct OPS_ *ops);
    /* vec */
    /**
    * @brief 根据源矩阵创建新向量
    *
    * @param[out] des_vec 指向新创建向量指针的二级指针（输出参数）
    * @param[in]  src_mat 输入源矩阵指针，作为向量创建的数据来源
    * @param[in]  ops     操作接口
    *
    */
    void (*VecCreateByMat)(void **des_vec, void *src_mat, struct OPS_ *ops);
    /**
    * @brief 根据源向量创建新向量，新向量将继承源向量的维度特性
    * 
    * @param[out] des_vec 指向新创建向量指针的二级指针（输出参数）
    * @param[in] src_vec 输入源向量指针，作为新向量的创建模板
    * @param[in] ops 操作接口
    * 
    */
    void (*VecCreateByVec)(void **des_vec, void *src_vec, struct OPS_ *ops);
    /**
    * @brief 销毁向量des_vec
    *
    * @param[in,out] des_vec 指向待销毁向量指针的二级指针（输入输出参数）
    * @param[in]     ops     操作接口
    *
    */
    void (*VecDestroy)(void **des_vec, struct OPS_ *ops);
    /**
    * @brief 输出向量内容到标准输出流
    *
     * @param[in] x   待展示的向量指针，向量数据格式由具体实现定义
    * @param[in] ops 操作接口
    *
    */
    void (*VecView)(void *x, struct OPS_ *ops);
    /* inner_prod = x'y */
    /**
    * @brief 计算两个向量的全局内积
    * 
    * @param[in] x           输入向量指针，参与内积计算的第一个向量
    * @param[in] y           输入向量指针，参与内积计算的第二个向量
    * @param[out] inner_prod 输出标量结果指针，存储计算结果
    * @param[in] ops         操作接口
    *
    */
    void (*VecInnerProd)(void *x, void *y, double *inner_prod, struct OPS_ *ops);
    /* inner_prod = x'y for each proc */
    /**
    * @brief 计算两个向量的全局内积
    * 
    * @param[in] x           输入向量指针，参与计算的第一个向量
    * @param[in] y           输入向量指针，参与计算的第二个向量
    * @param[out] inner_prod 输出标量指针，存储全局内积计算结果
    * @param[in] ops         操作接口
    *
    */
    void (*VecLocalInnerProd)(void *x, void *y, double *inner_prod, struct OPS_ *ops);
    /**
    * @brief 为向量设置随机数值
    *
    * @param[in,out] x   目标向量指针，将被填充随机数值
    * @param[in]     ops 操作接口
    *
    */
    void (*VecSetRandomValue)(void *x, struct OPS_ *ops);
    /* y = alpha x + beta y */
    /**
    * @brief 执行向量线性组合操作 y = alpha * x + beta * y
    *
    * @param[in] alpha  作用于输入向量x的标量系数
    * @param[in] x      输入向量指针，参与计算的第一个操作数
    * @param[in] beta   作用于输入输出向量y的标量系数
    * @param[in,out] y 输入输出向量指针，存储计算结果的容器
    * @param[in] ops    操作接口
    *
    */
    void (*VecAxpby)(double alpha, void *x, double beta, void *y, struct OPS_ *ops);
    /* y = mat  * x */
    /**
    * @brief 执行矩阵向量乘积运算 y = mat * x
    *
    * @param[in] mat  输入矩阵指针，矩阵存储格式由具体实现定义
    * @param[in] x    输入向量指针，维度需与矩阵列数匹配
    * @param[out] y   输出向量指针，接收运算结果，维度需与矩阵行数匹配
    * @param[in] ops  操作接口
    *
    */
    void (*MatDotVec)(void *mat, void *x, void *y, struct OPS_ *ops);
    /* y = mat' * x */
    /**
 * @brief 执行矩阵转置向量乘积运算 y = mat^T * x
 *
 * @param[in] mat  输入矩阵指针，矩阵存储格式需支持转置操作
 * @param[in] x    输入向量指针，维度需与矩阵转置后的列数匹配
 * @param[out] y   输出向量指针，接收运算结果，维度需与矩阵转置后的行数匹配
 * @param[in] ops  操作接口
 */
    void (*MatTransDotVec)(void *mat, void *x, void *y, struct OPS_ *ops);
    /* multi-vec */
    /**
     * @brief 通过num_vec与src_mat创建multi_vec，并给其分配内存空间
     * @param multi_vec 目标多维向量
     * @param num_vec (入参) 用于设置multi_vec的列数(向量的个数)
     * @param src_mat (入参) 用于设置multi_vec的行数(每一维向量的长度)
     */
    void (*MultiVecCreateByMat)(void ***multi_vec, int num_vec, void *src_mat, struct OPS_ *ops);

    /**
    * @brief 基于源向量创建多维向量结构
    * 
    * @param[out] multi_vec 指针输出参数，接收新创建的多维向量数组
    * @param[in] num_vec    指定要创建的向量个数（多维向量的列数）
    * @param[in] src_vec    源向量指针，作为新向量维度和存储特性的模板
    * @param[in] ops        操作接口
    *
    */
    void (*MultiVecCreateByVec)(void ***multi_vec, int num_vec, void *src_vec, struct OPS_ *ops);
    /**
    * @brief 基于现有多维向量创建新的多维向量结构
    * 
    * @param[out] multi_vec 三维指针输出参数，接收新创建的多维向量数组
    * @param[in] num_vec    指定要创建的向量个数（新多维向量的列数）
    * @param[in] src_mv     源多维向量指针数组，作为新向量结构和存储特性的模板
    * @param[in] ops        操作接口
    *
    */
    void (*MultiVecCreateByMultiVec)(void ***multi_vec, int num_vec, void **src_mv, struct OPS_ *ops);
    /**
    * @brief 销毁多维向量结构并释放相关内存
    * @param[in,out] multi_vec 三维指针参数，指向待销毁的多维向量数组指针
    * @param[in] num_vec       需要销毁的向量个数（多维向量的列数）
    * @param[in] ops           操作接口（用于执行实际内存释放操作）
    */
    void (*MultiVecDestroy)(void ***multi_vec, int num_vec, struct OPS_ *ops);
    /* *vec = multi_vec[col] */
    /**
    * @brief 从多维向量结构中提取指定列的向量
    * 
    * @param[in]  multi_vec  二维指针参数，输入的多维向量数组
    * @param[in]  col        需要提取的列索引(从0开始计数)
    * @param[out] vec        二级指针输出参数，接收提取的向量指针
    * @param[in]  ops        操作接口
    */
    void (*GetVecFromMultiVec)(void **multi_vec, int col, void **vec, struct OPS_ *ops);
    /**
    * @brief 将临时使用的向量归还到多维向量结构
    * 
    * @param[in,out] multi_vec  二维指针参数，操作后恢复完整性的多维向量数组
    * @param[in]  col        需要归还的目标列索引(从0开始计数)
    * @param[in,out] vec     二级指针参数，输入时指向待归还的向量指针，操作后置为NULL
    * @param[in]  ops        操作接口
    */
    void (*RestoreVecForMultiVec)(void **multi_vec, int col, void **vec, struct OPS_ *ops);
    /**
    * @brief 输出多维向量指定列范围内的内容
    * 
    * @param[in] x      二维指针参数，输入的多维向量数组
    * @param[in] start  起始列索引(包含)
    * @param[in] end    结束列索引(不包含)
    * @param[in] ops    操作接口
    */
    void (*MultiVecView)(void **x, int start, int end, struct OPS_ *ops);
    /**
    * @brief 计算多维向量在指定区间的局部内积
    * 
    * @param[in] nsdIP      存储方式标识符
    *                  'S' - 对称模式（填充上下三角），
    *                  'D' - 对角模式（仅计算主对角），
    *                  其他 - 常规模式（全矩阵计算）
    * @param[in] x          输入向量/矩阵数组指针
    * @param[in] y          输入向量/矩阵数组指针
    * @param[in] is_vec     模式标志(0-矩阵模式/1-向量模式)
    * @param[in] start      区间起始索引数组指针
    * @param[in] end        区间结束索引数组指针
    * @param[out] inner_prod 输出内积结果数组指针
    * @param[in] ldIP       结果数组的行主维度
    * @param[in] ops        操作接口
    */
    void (*MultiVecLocalInnerProd)(char nsdIP,
                                   void **x, void **y, int is_vec, int *start, int *end,
                                   double *inner_prod, int ldIP, struct OPS_ *ops);
    /**
     * @brief 计算多个向量的内积
     * 
     * 该函数是对MultiVecLocalInnerProd的封装，用于计算多个向量的内积。
     * 适用于需要分块计算或并行计算的情景，通过start/end参数指定计算范围。
     * 
     * @param[in] nsdIP     字符参数，指定内积存储方式
     *                      'D'表示对角矩阵处理模式/其他字母则为普通矩阵处理模式
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
    void (*MultiVecInnerProd)(char nsdIP,
                              void **x, void **y, int is_vec, int *start, int *end,
                              double *inner_prod, int ldIP, struct OPS_ *ops);
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
    void (*MultiVecSetRandomValue)(void **multi_vec,
                                   int start, int end, struct OPS_ *ops);
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
    void (*MultiVecAxpby)(
        double alpha, void **x, double beta, void **y,
        int *start, int *end, struct OPS_ *ops);
    /* y = x coef + y diag(beta) */
    /**
    * @brief 执行向量/矩阵的线性组合计算 y = beta*y + x*coef
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
    void (*MultiVecLinearComb)(
        void **x, void **y, int is_vec,
        int *start, int *end,
        double *coef, int ldc,
        double *beta, int incb, struct OPS_ *ops);

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
    void (*MatDotMultiVec)(void *mat, void **x, void **y,
                           int *start, int *end, struct OPS_ *ops);
    void (*MatTransDotMultiVec)(void *mat, void **x, void **y,
                                int *start, int *end, struct OPS_ *ops);
    /**
     * @brief 计算密集矩阵运算 qAp = Q^T * A * P 或相关变体，结果存入 qAp 数组
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
    void (*MultiVecQtAP)(char ntsA, char ntsdQAP,
                         void **mvQ, void *matA, void **mvP, int is_vec,
                         int *start, int *end, double *qAp, int ldQAP,
                         void **mv_ws, struct OPS_ *ops);
    /* Dense matrix vector ops */
    struct OPS_ *lapack_ops; /* ���ܾ��������Ĳ��� */
                             /* matC = alpha*matQ^{\top}*matA*matP + beta*matC 
    * dbl_ws: nrowsA*ncolsC */
                             /**
     * @brief 计算稠密矩阵运算 C = alpha * Q^T * A * P + beta * C
     *          支持分块处理、对称矩阵优化及并行计算
     *
     * @param ntluA   A矩阵的三角类型('L'下三角/'U'上三角)或普通矩阵('N')
     * @param nsdC    C矩阵的存储类型('D'对角/'S'对称/'N'普通)
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
    void (*DenseMatQtAP)(char ntluA, char nsdC,
                         int nrowsA, int ncolsA, /* matA �������� */
                         int nrowsC, int ncolsC, /* matC �������� */
                         double alpha, double *matQ, int ldQ,
                         double *matA, int ldA,
                         double *matP, int ldP,
                         double beta, double *matC, int ldC,
                         double *dbl_ws);
    void (*DenseMatOrth)(double *mat, int nrows, int ldm,
                         int start, int *end, double orth_zero_tol,
                         double *dbl_ws, int length, int *int_ws);
    /* linear solver */
    void (*LinearSolver)(void *mat, void *b, void *x,
                         struct OPS_ *ops);
    void *linear_solver_workspace;
    void (*MultiLinearSolver)(void *mat, void **b, void **x,
                              int *start, int *end, struct OPS_ *ops);
    void *multi_linear_solver_workspace;
    /* orthonormal */
    // 将x中(start_x-end_x)间的向量对B进行正交化，返回end_x表示正交化操作的结束位置
    // x：向量数组，start_x：起始向量，end_x：结束向量，B：对B进行正交化，ops：操作接口
    void (*MultiVecOrth)(void **x, int start_x, int *end_x,
                         void *B, struct OPS_ *ops);
    void *orth_workspace;
    /* multi grid */
    /* get multigrid operator for num_levels = 4
 	 * P0     P1       P2
 	 * A0     A1       A2        A3
	 * B0  P0'B0P0  P1'B1P1   P2'B2P2 
	 * A0 is the original matrix */
    void (*MultiGridCreate)(void ***A_array, void ***B_array, void ***P_array,
                            int *num_levels, void *A, void *B, struct OPS_ *ops);
    /* free A1 A2 A3 B1 B2 B3 P0 P1 P2 
	 * A0 and B0 are just pointers */
    void (*MultiGridDestroy)(void ***A_array, void ***B_array, void ***P_array,
                             int *num_levels, struct OPS_ *ops);
    void (*VecFromItoJ)(void **P_array, int level_i, int level_j,
                        void *vec_i, void *vec_j, void **vec_ws, struct OPS_ *ops);
    void (*MultiVecFromItoJ)(void **P_array, int level_i, int level_j,
                             void **multi_vec_i, void **multi_vec_j, int *startIJ, int *endIJ,
                             void ***multi_vec_ws, struct OPS_ *ops);
    /* eigen solver */
    void (*EigenSolver)(void *A, void *B, double *eval, void **evec,
                        int nevGiven, int *nevConv, struct OPS_ *ops);
    void *eigen_solver_workspace;

    /* for pas */
    struct OPS_ *app_ops;
} OPS;

void OPS_Create(OPS **ops);
void OPS_Setup(OPS *ops);
void OPS_Destroy(OPS **ops);

/* multi-vec */
void DefaultPrintf(const char *fmt, ...);
double DefaultGetWtime(void);
int DefaultGetOptionFromCommandLine(
    const char *name, char type, void *value,
    int argc, char *argv[], struct OPS_ *ops);
void DefaultMultiVecCreateByVec(void ***multi_vec, int num_vec, void *src_vec, struct OPS_ *ops);
void DefaultMultiVecCreateByMat(void ***multi_vec, int num_vec, void *src_mat, struct OPS_ *ops);
void DefaultMultiVecCreateByMultiVec(void ***multi_vec, int num_vec, void **src_mv, struct OPS_ *ops);
void DefaultMultiVecDestroy(void ***multi_vec, int num_vec, struct OPS_ *ops);
void DefaultGetVecFromMultiVec(void **multi_vec, int col, void **vec, struct OPS_ *ops);
void DefaultRestoreVecForMultiVec(void **multi_vec, int col, void **vec, struct OPS_ *ops);
void DefaultMultiVecView(void **x, int start, int end, struct OPS_ *ops);
void DefaultMultiVecLocalInnerProd(char nsdIP, void **x, void **y, int is_vec, int *start, int *end,
                                   double *inner_prod, int ldIP, struct OPS_ *ops);
void DefaultMultiVecInnerProd(char nsdIP, void **x, void **y, int is_vec, int *start, int *end,
                              double *inner_prod, int ldIP, struct OPS_ *ops);
void DefaultMultiVecSetRandomValue(void **x, int start, int end, struct OPS_ *ops);
void DefaultMultiVecAxpby(
    double alpha, void **x, double beta, void **y,
    int *start, int *end, struct OPS_ *ops);
void DefaultMultiVecLinearComb(
    void **x, void **y, int is_vec,
    int *start, int *end,
    double *coef, int ldc,
    double *beta, int incb, struct OPS_ *ops);
void DefaultMatDotMultiVec(void *mat, void **x, void **y,
                           int *start, int *end, struct OPS_ *ops);
void DefaultMatTransDotMultiVec(void *mat, void **x, void **y,
                                int *start, int *end, struct OPS_ *ops);
void DefaultMultiVecQtAP(char ntsA, char ntsdQAP,
                         void **mvQ, void *matA, void **mvP, int is_vec,
                         int *startQP, int *endQP, double *qAp, int ldQAP,
                         void **mv_ws, struct OPS_ *ops);
/* multi-grid */
void DefaultVecFromItoJ(void **P_array, int level_i, int level_j,
                        void *vec_i, void *vec_j, void **vec_ws, struct OPS_ *ops);
void DefaultMultiVecFromItoJ(void **P_array, int level_i, int level_j,
                             void **multi_vec_i, void **multi_vec_j, int *startIJ, int *endIJ,
                             void ***multi_vec_ws, struct OPS_ *ops);

#endif /* -- #ifndef _OPS_H_ -- */
