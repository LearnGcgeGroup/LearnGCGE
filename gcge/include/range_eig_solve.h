/**
 * @brief 按区间求解特征值的功能实现
 * @note 依赖petsc与slepc实现
 * @date 2025-03-27
 */

#ifndef _OPS_INCLUDE_RANGE_EIG_SOLVELIN_SOL_H_
#define _OPS_INCLUDE_RANGE_EIG_SOLVELIN_SOL_H_

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include	"ops.h"

/**
 * @brief 调用 Rayleigh-Ritz过程 求解子空间投影问题： V^T B (A - bB)^-1 B V C = V^T B V C \Lambda
 * 公式中C：特征向量矩阵, \Lambda: 由特征值形成的对角线矩阵
 * Rayleigh-Ritz过程将大规模特征值问题转化为小规模特征值问题并求解其特征值和特征向量
 * @note 从右往左算
 * @param ss_matA (输出变量)用于存放子空间投影问题的矩阵V^HAV，一个二维数组，大小为 (sizeV−sizeC)×(sizeV−sizeC)
 * @param ss_eval (输出变量)存储计算得到的小规模特征值问题的特征值，一个一维数组，大小为 sizeV−sizeC
 * @param ss_evec (输出变量)存储计算得到的小规模特征值问题的特征向量，一个二维数组，大小为 (sizeV−sizeC)×(sizeV−sizeC)
 * @param tol (输入变量)求解小规模特征值问题的阈值参数，用于控制特征值求解的精度。
 * @param nevConv (输入变量)当前收敛的特征值个数
 * @param ss_diag (输出变量)存储子空间投影问题的矩阵ss_matA的对角部分
 * @param A (输入变量)刚度矩阵
 * @param B (输入变量)质量矩阵
 * @param V (输入变量)子空间基向量矩阵 V
 * @param ops (输入变量)ops上下文
 */
void ComputeRangeRayleighRitz(double *ss_matA, double *ss_eval, double *ss_evec, double tol,
                                int nevConv, double *ss_diag, void *A, void *B,void **V, struct OPS_ *ops);



#endif  /* -- #ifndef _OPS_INCLUDE_RANGE_EIG_SOLVELIN_SOL_H_ -- */
