/**
 * @brief 计算矩阵的负特征值数目
 * @author mading
 * @date 2025-03-24
 */

#include "count_eigen_num.h"

GcgeErrCode CountEigenNum::countEigenNum(void *A, void *B, double a, double b, int &numEigen) {
    PetscMPIInt rank, size; // 进程信息 
    PetscInt rstart, rend;  // 当前进程所拥有的第一行/最后一行的全局索引
    MatGetOwnershipRange((Mat)A, &rstart, &rend);

    Mat A_aB;       // 保存 A-a*B
    Mat A_aB_AIJ;   // 转数据格式
    Mat chol_AaB;   // cholesky分解后的矩阵

    MatDuplicate((Mat)B, MAT_COPY_VALUES, &A_aB);               // 拷贝B，作B = A - a*B
    MatAYPX(A_aB, -a,  (Mat)A, DIFFERENT_NONZERO_PATTERN);      // MatAYPX(Y, a, X, ..)功能为 Y = a * Y + X 
    MatConvert(A_aB, MATAIJ, MAT_INITIAL_MATRIX, &A_aB_AIJ);    // 只支持AIJ格式矩阵分解
    // -----------------------------------------------------------------
    PetscInt nozeroN;           // 某一行的非零数
    const PetscInt *cols;       // 某一行的非零数的列索引数组
    const PetscScalar *values;  // 对应数值
    PetscInt sumN = 0;          // 全部非零数据个数

    for (PetscInt i = rstart; i < rend; i++) {  // 统计个数
        MatGetRow(A_aB_AIJ, i, &nozeroN, &cols, &values);
        sumN += nozeroN;
        MatRestoreRow(A_aB_AIJ, i, &nozeroN, &cols, &values);
    }

    // ia: 行索引数组, ja: 列索引数组
    PetscInt *ia = NULL, *ja = NULL;
    // val: 非零元素数组
    PetscScalar *val = NULL;
    PetscMalloc3(sumN, &ia, sumN, &ja, sumN, &val);

    PetscInt id = 0;
    for (PetscInt i = rstart; i < rend; i++) {  // 收集数值
        MatGetRow(A_aB_AIJ, i, &nozeroN, &cols, &values);
        for (PetscInt j = 0; j < nozeroN; j++) {
            ia[id] = i;
            ja[id] = cols[j];
            val[id++] = values[j];
        }
        MatRestoreRow(A_aB_AIJ, i, &nozeroN, &cols, &values);
    }
    PetscPrintf(PETSC_COMM_SELF, "sumN: %d %lg\n",  sumN, val[id-1]);

    // todo 子进程同步数据到主进程
    // PetscCallMPI(MPI_Bcast(&M, 1, MPI_INT, 0, PETSC_COMM_WORLD));
    // PetscCallMPI(MPI_Bcast(&N, 1, MPI_INT, 0, PETSC_COMM_WORLD));
    // PetscCallMPI(MPI_Bcast(&nz, 1, MPI_INT, 0, PETSC_COMM_WORLD));

    // -----------------------------------------------------------------
    IS row, col;
    MatFactorInfo info;
    PetscInt nneg, nzero, npos;

    MatGetOrdering(A_aB_AIJ, MATORDERINGRCM, &row, &col);       // 矩阵排序
    MatFactorInfoInitialize(&info);                             // MatCholeskyFactor(A_aB, row, &info); // 自带排序
    MatGetFactor(A_aB_AIJ, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &chol_AaB);
    MatCholeskyFactorSymbolic(chol_AaB, A_aB_AIJ, row, &info);  // 符号分析
    MatCholeskyFactorNumeric(chol_AaB, A_aB_AIJ, &info);        // 数值分解
    
    // 计算A - a * B 惯性指数
    MatGetInertia(chol_AaB, &numEigen, &nzero, &npos); 
    printf("nneg: %d, nzero:%d, npos: %d\n", numEigen, nzero, npos);

    /*-----------------------------------------------------------------------------------*/

    Mat A_bB;       // 保存 A-b*B
    Mat A_bB_AIJ;   // 转数据格式
    Mat chol_AbB;   // cholesky分解后的矩阵

    MatDuplicate((Mat)B, MAT_COPY_VALUES, &A_bB); 
    MatAYPX(A_bB, -b,  (Mat)A, DIFFERENT_NONZERO_PATTERN);      
    MatConvert(A_bB, MATAIJ, MAT_INITIAL_MATRIX, &A_bB_AIJ);
    MatGetOrdering(A_bB_AIJ, MATORDERINGRCM, &row, &col);    

    MatGetFactor(A_bB_AIJ, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &chol_AbB);
    MatCholeskyFactorSymbolic(chol_AbB, A_bB_AIJ, row, &info);
    MatCholeskyFactorNumeric(chol_AbB, A_bB_AIJ, &info);

    // 计算A - b * B 惯性指数
    MatGetInertia(chol_AbB, &nneg, &nzero, &npos); 
    printf("nneg: %d, nzero:%d, npos: %d\n", nneg, nzero, npos);

    // 区间内特征值个数
    numEigen = nneg - numEigen;

    PetscFree3(ia, ja, val);

    ISDestroy(&row);
    ISDestroy(&col);
    MatDestroy(&A_aB);
    MatDestroy(&A_bB);
    MatDestroy(&chol_AaB);
    MatDestroy(&chol_AbB);

    return GCGE_SUCCESS;
}