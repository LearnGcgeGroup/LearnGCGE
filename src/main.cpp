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
#include "io/input_read_tool.h"
#include "error_code.h"

#include <petscmat.h>

#if OPS_USE_SLEPC
#include <slepcbv.h>
#endif

extern "C" {
#include "app_ccs.h"
#include "app_slepc.h"
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
    if (argc < 4) { // 检查参数数量
        std::cerr << "Usage: mpiexec -n <b> program <sourceMatA.mtx> <sourceMatB.mtx> <paramFile.txt>" << std::endl;
        return GCGE_ERR_INPUT;
    }
    
#if OPS_USE_SLEPC
    SlepcInitialize(&argc, &argv, NULL, NULL);
#elif OPS_USE_PETSC
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
#elif OPS_USE_MPI
    MPI_Init(&argc, &argv);
#endif

    // 1、读取文件
    // 1.1、读取矩阵文件
    char *fileA = argv[1];
#if OPS_USE_SLEPC
    Mat sourceMatA;
    auto err = InputReadTool::ReadPetscMatFromMtx(&sourceMatA, fileA);
#elif OPS_USE_PETSC
    Mat tmpA;
    auto err = InputReadTool::ReadPetscMatFromMtx(&tmpA, fileA);
    CCSMAT sourceMatA;
    InputReadTool::ConvertPetscMatToCCSMat(tmpA, sourceMatA);
#else
    CCSMAT sourceMatA;
    auto err = InputReadTool::ReadCcsFromMtx(&sourceMatA, fileA);
#endif
    if (err != GCGE_SUCCESS) {
        return err;
    }

    char *fileB = argv[2];
#if OPS_USE_SLEPC
    Mat sourceMatB;
    err = InputReadTool::ReadPetscMatFromMtx(&sourceMatB, fileB);
#elif OPS_USE_PETSC
    Mat tmpB;
    err = InputReadTool::ReadPetscMatFromMtx(&tmpB, fileB);
    CCSMAT sourceMatB;
    InputReadTool::ConvertPetscMatToCCSMat(tmpB, sourceMatB);
#else
    CCSMAT sourceMatB;
    err = InputReadTool::ReadCcsFromMtx(&sourceMatB, fileB);
#endif
    if (err != GCGE_SUCCESS) {
        return err;
    }

    // 1.2、读取用户参数文件
    GcgeParam gcgeparam{20};
    ExtractMethod extractMethod;    // 结构体对象，保存用户设置的特征值抽取方式
    std::string usrParaFile = argv[3];
    err = InputReadTool::readUserParam(gcgeparam, extractMethod, usrParaFile);
    if (err != GCGE_SUCCESS) {
        return err;
    }

    // 2、设置工作空间
    OPS* ccs_ops = NULL;
    OPS_Create(&ccs_ops);
#if OPS_USE_SLEPC
    OPS_SLEPC_Set(ccs_ops);
#else
    OPS_CCS_Set(ccs_ops);
#endif
    OPS_Setup(ccs_ops);

    // 3、设置输入参数
    void *matA, *matB;
    OPS* ops;
    ops = ccs_ops;
#if OPS_USE_SLEPC
    matA = static_cast<void*>(sourceMatA);
    matB = static_cast<void*>(sourceMatB);
#else
    matA = static_cast<void*>(&sourceMatA);
    matB = static_cast<void*>(&sourceMatB);
#endif

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
#if OPS_USE_SLEPC
    MatDestroy(&sourceMatA);
    MatDestroy(&sourceMatB);
#elif OPS_USE_PETSC
    MatDestroy(&tmpA);
    MatDestroy(&tmpB);
    destroyMatrixCCS(&sourceMatA, &sourceMatB);
#else
    destroyMatrixCCS(&sourceMatA, &sourceMatB);
#endif

#if OPS_USE_SLEPC
    SlepcFinalize();
#elif OPS_USE_PETSC
    PetscCall(PetscFinalize());
#elif OPS_USE_MPI
    MPI_Finalize();
#endif
    std::cout << "GCGE solver finished!" << std::endl;
    return 0;
}
