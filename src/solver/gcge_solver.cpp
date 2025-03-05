/**
 * @brief GCGE特征值求解器外层接口
 */

#include "gcge_solver.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>

extern "C" {
#include "app_ccs.h"
#include "app_lapack.h"
#include "ops.h"
#include "ops_config.h"
#include "ops_eig_sol_gcg.h"
#include "ops_lin_sol.h"
#include "ops_orth.h"
}

int destroyMatrixCCS(CCSMAT* ccs_matA, CCSMAT* ccs_matB) {
    delete ccs_matA->j_col;
    ccs_matA->j_col = nullptr;

    delete ccs_matB->j_col;
    ccs_matB->j_col = nullptr;

    return 0;
}

/**
 * @brief eigenSolverGCG：该函数会根据参数设定工作空间，并调用EigenSolver函数求解，最后输出特征值和特征向量
 * 
 * @param in 
 * 		A ：即Ax = λBx的A矩阵，CCSMAT类型（CSC格式），只能为对称矩阵
 * 		B ：即Ax = λBx的B矩阵，置为NULL即为标准特征值问题,当前只能为正定矩阵
 * 		gcgeparam ：算法的参数结构体，包括预期特征值数量、块大小、迭代次数和精度等
 * 		ops ：算法函数指针的结构体
 * 		
 * @param out
 * 		eigenvalue ：用来存放特征值的一维数组
 * 		eigenvector ： 存放特征向量的二维数组（采用push_back方式插入元素）
 * 
 * @return 返回值
 */
int eigenSolverGCG(void* A, void* B, std::vector<double>& eigenvalue, std::vector<std::vector<double>>& eigenvector,
                   struct GcgeParam* gcgeparam, struct OPS_* ops) {
    /* 展示算法调用参数 */
    /* 用户希望收敛的特征对个数 nevConv, 最多返回 nevMax 
	 * 要求 block_size >= multiMax */

    int nevConv = gcgeparam->nevConv;
    int multiMax = gcgeparam->multiMax;
    int nevGiven = gcgeparam->nevGiven;
    int block_size = gcgeparam->block_size;
    int nevMax = gcgeparam->nevMax;
    /* 当特征值收敛 2*block_size 时, 将 P W 部分归入 X 部分, 
	 * 工作空间中的 X 不超过 nevInit (>=3*block_size) */
    int nevInit = gcgeparam->nevInit;
    int max_iter_gcg = gcgeparam->max_iter_gcg;
    double tol_gcg[2] = {gcgeparam->tol_gcg[0], gcgeparam->tol_gcg[1]}; //精度，0是绝对 1 相对
    double gapMin = gcgeparam->gapMin;
    /* 工作空间由 nevMax blockSize nevInit 决定 */
    /* 特征值 特征向量 长度为 nevMax */
    double* eval = eigenvalue.data();
    void** evec;

    ops->MultiVecCreateByMat(&evec, nevMax, A, ops);
    ops->MultiVecSetRandomValue(evec, 0, nevMax, ops);

    void** gcg_mv_ws[4];
    double* dbl_ws; // double类型数据的工作空间
    int* int_ws; // int类型数据的工作空间
    /* 设定 GCG 的工作空间 nevMax+2*block_size, 
	 * block_size, block_size, block_size */
    EigenSolverCreateWorkspace_GCG(nevInit, nevMax, block_size, A, gcg_mv_ws, &dbl_ws, &int_ws, ops);

    ops->Printf("===============================================\n");
    ops->Printf("GCG Eigen Solver\n");
    EigenSolverSetup_GCG(multiMax, gapMin, nevInit, nevMax, block_size, tol_gcg, max_iter_gcg, gcgeparam->flag,
                         gcg_mv_ws, dbl_ws, int_ws, ops);

    int check_conv_max_num = 50;

    char initX_orth_method[8] = "mgs";
    int initX_orth_block_size = 80;
    int initX_orth_max_reorth = 2;
    double initX_orth_zero_tol = 2 * DBL_EPSILON; //1e-12

    char compP_orth_method[8] = "mgs";
    int compP_orth_block_size = -1;
    int compP_orth_max_reorth = 2;
    double compP_orth_zero_tol = 2 * DBL_EPSILON; //1e-12

    char compW_orth_method[8] = "mgs";
    int compW_orth_block_size = 80;
    int compW_orth_max_reorth = 2;
    double compW_orth_zero_tol = 2 * DBL_EPSILON; //1e-12
    int compW_bpcg_max_iter = 100;
    double compW_bpcg_rate = 1e-2;
    double compW_bpcg_tol = 1e-14;
    char compW_bpcg_tol_type[8] = "abs";

    int compRR_min_num = -1;
    double compRR_min_gap = gapMin;
    double compRR_tol = 2 * DBL_EPSILON;

    EigenSolverSetParameters_GCG(check_conv_max_num, initX_orth_method, initX_orth_block_size, initX_orth_max_reorth,
                                 initX_orth_zero_tol, compP_orth_method, compP_orth_block_size, compP_orth_max_reorth,
                                 compP_orth_zero_tol, compW_orth_method, compW_orth_block_size, compW_orth_max_reorth,
                                 compW_orth_zero_tol, compW_bpcg_max_iter, compW_bpcg_rate, compW_bpcg_tol,
                                 compW_bpcg_tol_type, 0, // without shift
                                 compRR_min_num, compRR_min_gap, compRR_tol, ops);

    /* 命令行获取 GCG 的算法参数 勿用 有 BUG, 
	 * 不应该改变 nevMax nevInit block_size, 这些与工作空间有关 */
    //EigenSolverSetParametersFromCommandLine_GCG(argc,argv,ops);
    ops->Printf("nevGiven = %d, nevConv = %d, nevMax = %d, block_size = %d, nevInit = %d\n", nevGiven, nevConv, nevMax,
                block_size, nevInit);
    fflush(stdout);
#if 0
    struct GCGSolver_* gcgsolver;
    gcgsolver = (GCGSolver*)ops->eigen_solver_workspace;
    gcgsolver->tol[0] = 1e-3;
    gcgsolver->tol[1] = 1e-3;
    int initnev = nevConv;
    ops->EigenSolver(A, B, eval, evec, nevGiven, &initnev, ops);
    gcgsolver->tol[0] = tol_gcg[0];
    gcgsolver->tol[1] = tol_gcg[1];
    ops->EigenSolver(A, B, eval, evec, nevInit, &nevConv, ops); //Eigen solver
#else
    ops->EigenSolver(A, B, eval, evec, nevGiven, &nevConv, ops); //Eigen solver
#endif
    ops->Printf("numIter = %d, nevConv = %d\n", ((GCGSolver*)ops->eigen_solver_workspace)->numIter, nevConv);
    ops->Printf("++++++++++++++++++++++++++++++++++++++++++++++\n");

    EigenSolverDestroyWorkspace_GCG(nevInit, nevMax, block_size, A, gcg_mv_ws, &dbl_ws, &int_ws, ops);

    // 开始导出数据
    eigenvalue.resize(nevConv);
    eigenvector.resize(nevConv);
    ops->Printf("eigenvectors\n");
    for (auto i = 0; i < nevConv; ++i) {
        std::cout << "index: " << i + 1 << " eigenvalue: " << eigenvalue[i] << std::endl;
    }
    //ops->MultiVecView(evec,0,nevConv,ops);

    multiVecReturn((LAPACKVEC*)(evec), eigenvalue.size(), eigenvector); // 需先进行类型转化

    ops->MultiVecDestroy(&(evec), nevMax, ops);
    return 0;
}

/**
 * @brief multiVecReturn: 将算法中存放特征向量的类型转化为vec2d
 * 
 * @param in 
 * 		LAPACKVEC *x：算法中存放特征向量的结构体类型
 * 		int end：拿出end个特征向量
 * 
 * @param out
 * 		std::vector<std::vector<double>>& eigenvector：将特征向量放到该二维数组中
 * 
 * @return 返回值
 */
void multiVecReturn(LAPACKVEC* x, int end, std::vector<std::vector<double>>& eigenvector) {
    int row, col;
    double* destin;
    for (col = 0; col < end; ++col) {
        for (row = 0; row < x->nrows; ++row) {
            destin = x->data + (x->ldd) * col + row;
            eigenvector[col].push_back(*destin);
        }
    }
    return;
}
