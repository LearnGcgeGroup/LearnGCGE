#ifndef __SRC_IO_MMLOADER_H__
#define __SRC_IO_MMLOADER_H__
#include <petscmat.h>
#include <petscsys.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "mmio.h"

/**
 * @brief 将MTX文件中的数据读取到PETSC格式的矩阵中
 * 
 * @param A 矩阵地址
 * @param filein 文件路径
 * @param aijonly 存储的矩阵格式是否为MATAIJ: MATAIJ: 通用稀疏矩阵存储。MATSBAIJ: 针对对称块矩阵优化，存储和计算效率更高。
 * @return PetscErrorCode 错误码
 */
PetscErrorCode MatCreateFromMTX(Mat *A, const char *filein, PetscBool aijonly);
#ifdef __cplusplus
}
#endif
#endif // __SRC_IO_MMLOADER_H__