/**
 * @brief GCGE特征值求解器主入口
 * @author zhangzy(zhiyu.zhang@cqbdri.pku.edu.cn)
 * @date 2025-02-18
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "solver/gcge_solver.h"
#include "io/io_eigen_result.h"

extern "C" {
#include "app_ccs.h"
#include "app_lapack.h"
#include "ops.h"
#include "ops_config.h"
#include "ops_eig_sol_gcg.h"
#include "ops_lin_sol.h"
#include "ops_orth.h"
#include "io/mmio_reader.h"
}

int main(int argc, char *argv[])
{
#if OPS_USE_MPI
    MPI_Init(&argc, &argv);
#endif
    // 1、读取文件
    CCSMAT ccs_matA;
    char *fileA = argv[1];
    CreateCCSFromMTX(&ccs_matA, fileA);

    CCSMAT ccs_matB;
    char *fileB = argv[2];
    CreateCCSFromMTX(&ccs_matB, fileB);

    // 2、设置工作空间
    OPS* ccs_ops = NULL;
    OPS_Create(&ccs_ops);
    OPS_CCS_Set(ccs_ops);
    OPS_Setup(ccs_ops);

    // 3、设置输入参数
    void *matA, *matB;
    OPS* ops;
    ops = ccs_ops;
    matA = static_cast<void*>(&ccs_matA);
    matB = static_cast<void*>(&ccs_matB);
    GcgeParam gcgeparam{20};

    // 4、设置输出对象
    // 当前设置返回收敛的特征值和特征向量
    // 即求解函数中会resize这两个对象
    std::vector<double> eigenvalue(gcgeparam.nevMax, 0);
    std::vector<std::vector<double>> eigenvector(gcgeparam.nevMax);

    // 5、调用求解函数
    eigenSolverGCG(matA, matB, eigenvalue, eigenvector, &gcgeparam, ops);
    
    // 6、特征值和特征向量结果保存和读取
    // IoEigenResult ioer;
    // ioer.saveEigenResult(eigenvalue, eigenvector);
    // std::vector<double> eigenvalue1;
    // std::vector<std::vector<double>> eigenvector1;
    // ioer.readEigenFile(eigenvalue1, eigenvector1);

    // 7、销毁工作空间
    OPS_Destroy(&ccs_ops);
    destroyMatrixCCS(&ccs_matA, &ccs_matB);

#if OPS_USE_MPI
    MPI_Finalize();
#endif
    return 0;
}