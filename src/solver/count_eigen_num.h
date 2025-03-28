/**
 * @brief 计算区间范围内的特征值数目
 * @author mading
 * @date 2025-03-24
 */

#ifndef __GCG_INTERVAL_COUNT_EIGEN_NUM_H__
#define __GCG_INTERVAL_COUNT_EIGEN_NUM_H__

#include <petscmat.h>
#include <cmath>
#include "error_code.h"

class CountEigenNum {
public:
    /**
     * @brief 计算区间范围内的特征值数目
     * 
     * @param A 矩阵引用   A*x = lamda*B*x
     * @param B 矩阵引用   A*x = lamda*B*x
     * @param a 区间范围   区间范围[a, b]
     * @param b 区间范围   区间范围[a, b]
     * @param numEigen 区间范围内的特征值数目
     * @return GcgeErrCode 错误码
     */
    static GcgeErrCode countEigenNum(void *A, void *B, double a, double b, int &numEigen);
};
#endif // __GCG_INTERVAL_COUNT_EIGEN_NUM_H__