/**
 * @brief 特征值和特征向量写入txt文件
 * @author mading
 * @date 2025-02-21
 */

#ifndef _MMIO_EIGEN_RESULT_SAVE_H_
#define _MMIO_EIGEN_RESULT_SAVE_H_

#include <stdio.h>
#include <vector>
#include <fstream>
#include <string>
#include <array>
#include <charconv>     // c++ 17

#define MBATCH_SIZE 10000
#define MBUFFER_SIZE 1000000

/**
 * @brief 将特征值和特征向量结果写入txt文件
 * 
 * @param eigenvalue GCGE求解的特征值结果
 * @param eigenvector GCGE求解的特征向量结果
 * @return int 错误码 0：正常， -1：错误
 */
int eigenResultSave(const std::vector<double> &eigenvalue, const std::vector<std::vector<double>> &eigenvector);


/**
 * @brief 将一个std::vector<double>写入txt文件
 * 
 * @param outFile 输出流ofstream
 * @param eigenvalue 一个vector<double>数据
 * @return int 错误码 0：正常， -1：错误
 */
int eigenVectorSave(std::ofstream &outFile, const std::vector<double> &eigenvalue);

#endif
