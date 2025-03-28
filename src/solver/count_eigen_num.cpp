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

    ISDestroy(&row);
    ISDestroy(&col);
    MatDestroy(&A_aB);
    MatDestroy(&A_bB);
    MatDestroy(&A_aB_AIJ);
    MatDestroy(&A_bB_AIJ);
    MatDestroy(&chol_AaB);
    MatDestroy(&chol_AbB);

    return GCGE_SUCCESS;
}