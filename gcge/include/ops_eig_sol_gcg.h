/**
 * @file ops_eig_sol_gcg.h
 * @brief 广义特征值问题的GCG求解器
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef  _OPS_EIG_SOL_GCG_H_
#define  _OPS_EIG_SOL_GCG_H_

#include	"ops.h"
#include    "ops_orth.h"
#include    "ops_lin_sol.h"
#include    "app_lapack.h"

// GCG求解器的结构体
typedef struct GCGSolver_ {
	void   *A;		// 刚度矩阵 
	void   *B;		// 质量矩阵
	double sigma;	// shift
	double *eval;	// 特征值数组 
	void   **evec;	// 特征向量二维数组，分配向量个数 nevMax + block_size
	int    nevMax;  // 整个任务所要求的特征对个数
	int   multiMax; 
	double gapMin; // 两个特征值间认为不是重根的最小间隔
	int    nevInit; // 初始选取X矩阵的列数
	int   nevGiven; // 当前批次求解前，收敛特征对的总个数
	int    nevConv;	// 做输入参数: 期望收敛的特征值个数; 做输出参数: 当前批次求解后，收敛特征对的总个数
	int    block_size; // 分块矩阵W或P的列数，预估大于所要求解的特征值的最大代数重数
	double tol[2] ; // 0: abs_tol, 1: rel_tol
	int numIterMax; // 最大迭代次数
	int    numIter; // 当前迭代次数
	int    sizeV;	// V矩阵的理论列数 
	void   **mv_ws[4]; // 多向量内存空间 0: V,
	double *dbl_ws;    // 双精度内存空间，2*sizeV*sizeV + 2*sizeV, 用于存储子空间投影问题的矩阵和对角元，求得的特征向量和特征值，顺序：[特征值 对角元 矩阵 特征向量] 
	int *int_ws;	   // 整型内存空间
	int  length_dbl_ws;// 双精度内存空间数组长度
	int  check_conv_max_num; // 单次检查收敛性的最大特征对个数
	
	char initX_orth_method[8] ; 
	int    initX_orth_block_size; 
	int  initX_orth_max_reorth; 
	double initX_orth_zero_tol;
	char compP_orth_method[8] ; 
	int    compP_orth_block_size; 
	int  compP_orth_max_reorth; 
	double compP_orth_zero_tol;
	char compW_orth_method[8] ; 
	int    compW_orth_block_size; 
	int  compW_orth_max_reorth; 
	double compW_orth_zero_tol;

	int  user_defined_multi_linear_solver;	// 指定用户自定义的线性方程组求解器，目前仅支持 0： PCG
	int  compW_cg_max_iter    ; 
	double compW_cg_rate; 
	double compW_cg_tol       ; 
	char   compW_cg_tol_type[8]; // 计算W是用的CG法中收敛容差的判断方式(abs or rel)
	int  compW_cg_auto_shift  ; // 是否自动按照内置公式计算shift
	double compW_cg_shift;      // 王博士认为这个成员变量是多余的，没有实际作用
	int  compW_cg_order       ; // 用于是否调用ComputeW12（冗余的函数）的判断，可删去

	int    compRR_min_num     ; 
	double compRR_min_gap; 
	double compRR_tol; 			/*tol for dsyevx_ */

    enum EigenValueExtractType {
        GCGE_BY_ORDER = 0,               // 通过设置阶数提取
        GCGE_BY_FREQUENCY = 1,           // 通过设置频率提取
        GCGE_BY_ORDER_AND_FREQUENCY = 2  // 通过设置阶数和频率提取
    } extract_type;  // 特征值提取方式
    double min_eigenvalue;  // 最小特征值
    double max_eigenvalue;  // 最大特征值

} GCGSolver;

void EigenSolverSetup_GCG(
	int    multiMax, double gapMin, 
	int    nevInit , int    nevMax, int block_size,
	double tol[2]  , int    numIterMax,
	int    user_defined_multi_linear_solver,
	void **mv_ws[4], double *dbl_ws   , int *int_ws, 
	struct OPS_ *ops);
	
void EigenSolverCreateWorkspace_GCG(
	int nevInit, int nevMax, int block_size, void *mat,
	void ***mv_ws, double **dbl_ws, int **int_ws, 
	struct OPS_ *ops);

void EigenSolverDestroyWorkspace_GCG(
	int nevInit, int nevMax, int block_size, void *mat,
	void ***mv_ws, double **dbl_ws, int **int_ws, 
	struct OPS_ *ops);
	
void EigenSolverSetParameters_GCG(
	int    check_conv_max_num,
	const char *initX_orth_method, int initX_orth_block_size, int initX_orth_max_reorth, double initX_orth_zero_tol,
	const char *compP_orth_method, int compP_orth_block_size, int compP_orth_max_reorth, double compP_orth_zero_tol,
	const char *compW_orth_method, int compW_orth_block_size, int compW_orth_max_reorth, double compW_orth_zero_tol,
	int    compW_cg_max_iter , double compW_cg_rate, 
	double compW_cg_tol      , const char *compW_cg_tol_type, int compW_cg_auto_shift  ,
	int    compRR_min_num, double compRR_min_gap, double compRR_tol, 
	struct OPS_ *ops);

void EigenSolverSetParametersFromCommandLine_GCG(
	int argc, char* argv[], struct OPS_ *ops);



// 共享数据临时方案
typedef struct {
    int* sizeN_ptr;  // 存储指向sizeN的指针
    int* startN_ptr;  // 存储指向startN的指针
    int* endN_ptr;  // 存储指向endN的指针
    int* sizeP_ptr;  // 存储指向sizeP的指针
    int* startP_ptr;  // 存储指向startP的指针
    int* endP_ptr;  // 存储指向endP的指针
    int* sizeW_ptr;  // 存储指向sizeW的指针
    int* startW_ptr;  // 存储指向startW的指针
    int* endW_ptr;  // 存储指向endW的指针
    int* sizeC_ptr;  // 存储指向sizeC的指针
    int* sizeX_ptr;  // 存储指向sizeX的指针
    int* sizeV_ptr;  // 存储指向sizeV的指针
    int* endX_ptr;  // 存储指向endX的指针
    // 其他需要共享的指针...
} RangeSharedData;

#endif  /* -- #ifndef _OPS_EIG_SOL_GCG_H_ -- */

