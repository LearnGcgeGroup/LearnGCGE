/**
 * @brief 基于CCS（压缩列存储）格式的矩阵和向量操作
 */
#ifndef _APP_CCS_H_
#define _APP_CCS_H_

#include "app_lapack.h"
#include "ops.h"

/**
 * @brief CCS（Compressed Column Storage）稀疏矩阵存储结构体
 * 
 * @details 采用压缩列存储格式表示稀疏矩阵，包含以下核心成员：
 * 
 * @var CCSMAT::data   存储非零元素的数组，按列主序排列
 * @var CCSMAT::i_row  存储非零元素对应的行索引数组，与data数组一一对应
 * @var CCSMAT::j_col  列偏移数组，每个元素表示对应列的非零元素起始位置
 *                     (j_col[n]到j_col[n+1]-1为第n列的非零元素索引范围)
 * @var CCSMAT::nrows  矩阵总行数（最大行号+1）
 * @var CCSMAT::ncols  矩阵总列数（最大列号+1）
 * 
 * @note 典型内存布局示例（3x3矩阵）：
 * [0,0,1]        data  = [3,1,4,2,5]  
 * [2,3,4]  CCS格式 →   i_row = [1,2,0,2,1]
 * [0,5,0]        j_col = [0,2,3,5] (ncols=3)
 */
typedef struct CCSMAT_ {
    double *data;
    int *i_row;
    int *j_col;
    int nrows;
    int ncols;
} CCSMAT;

void OPS_CCS_Set(struct OPS_ *ops);

#endif /* -- #ifndef _APP_CCS_H_ -- */
