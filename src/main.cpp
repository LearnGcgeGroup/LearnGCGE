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
#include "io/read_user_param.h"

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

    // 3.1、读取用户参数文件
    ExtractMethod extractMethod;    // 结构体对象，保存用户设置的特征值抽取方式
    ReadUserParam readUP;           // 读取用户参数对象，读取txt求解参数文件
    char *usrParaFile = argv[3];
    readUP.readUserParam(gcgeparam, extractMethod, usrParaFile);
    gcgeparam.shift = 0;
    if (gcgeparam.nevConv <= 50) {
        gcgeparam.block_size = gcgeparam.nevConv;
        gcgeparam.nevInit = 2 * gcgeparam.nevConv;
        gcgeparam.nevMax = 2 * gcgeparam.nevConv;
    } else if (gcgeparam.nevConv <= 200) {
        gcgeparam.block_size = gcgeparam.nevConv / 5;
        gcgeparam.nevInit = gcgeparam.nevConv;
        gcgeparam.nevMax = gcgeparam.nevInit + gcgeparam.nevConv;
    } else if (gcgeparam.nevConv < 800) {
        gcgeparam.block_size = gcgeparam.nevConv / 8;
        gcgeparam.nevInit = 6 * gcgeparam.block_size;
        gcgeparam.nevMax = gcgeparam.nevInit + gcgeparam.nevConv;
    } else if (gcgeparam.nevConv == 800) {
        gcgeparam.block_size = 80;
        gcgeparam.nevInit = 350;
        gcgeparam.nevMax = 1350;
    } else {
        gcgeparam.block_size = 200;
        gcgeparam.nevInit = 3 * gcgeparam.block_size;
        gcgeparam.nevMax = gcgeparam.nevConv + gcgeparam.nevInit;
    }

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