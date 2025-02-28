/**
 * @brief GCGE特征值求解器外层接口
 */

#ifndef __BEF_GCGE_SRC_GCGE_PARA_H__
#define __BEF_GCGE_SRC_GCGE_PARA_H__

#include <cfloat>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include "io/param_struct.h" // 将求解参数结构体定义单独放置在一个文件中

extern "C" {
#include "app_ccs.h"
}

/**
 * @brief GCG求解算法外层封装接口
 * 
 * @param A 矩阵A
 * @param B 矩阵B
 * @param eigenvalue 返回值(特征值)
 * @param eigenvector 返回值(特征向量)
 * @param gcgeparam GCGE算法参数
 * @param ops GCGE工作空间
 * @return int 错误码
 */
int eigenSolverGCG(void* A, void* B, std::vector<double>& eigenvalue, std::vector<std::vector<double>>& eigenvector,
                          struct GcgeParam* gcgeparam, struct OPS_* ops);

/**
 * @brief 将特征向量集存储在二维数组中
 * 
 * @param x 特征向量集
 * @param end 特征向量个数
 * @param eigenvector 二维数组
 */
void multiVecReturn(LAPACKVEC* x, int end, std::vector<std::vector<double>>& eigenvector);

/**
 * @brief 析构CCS格式的矩阵
 * 
 * @param ccs_matA A矩阵
 * @param ccs_matB B矩阵
 * @return int 错误码
 */
int destroyMatrixCCS(CCSMAT* ccs_matA, CCSMAT* ccs_matB);

#endif // __BEF_GCGE_SRC_GCGE_PARA_H__