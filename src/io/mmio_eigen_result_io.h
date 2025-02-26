/**
 * @brief 特征值和特征向量写入txt文件
 * @author mading
 * @date 2025-02-21
 */

#ifndef _MMIO_EIGEN_RESULT_IO_H_
#define _MMIO_EIGEN_RESULT_IO_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <array>
#include <charconv> // c++ 17

#define MBATCH_SIZE 10000
#define MBUFFER_SIZE 1000000
#define PROTOCOL_LINE 9 // 特征值和特征向量文件协议描述行数

class MmioEigenResultIO
{
public:
    std::vector<double> eigenvalue_;               // 读取时，保存特征值
    std::vector<std::vector<double>> eigenvector_; // 读取时，保存特征向量

private:
    int size_ = 0;      // 特征值格式
    int dimension_ = 0; // 特征向量维度

public:
    /**
     * @brief 将特征值和特征向量结果写入txt文件
     *
     * @param eigenvalue GCGE求解的特征值结果
     * @param eigenvector GCGE求解的特征向量结果
     * @return int 错误码 0：正常， -1：错误
     */
    int eigenResultSave(const std::vector<double> &eigenvalue, const std::vector<std::vector<double>> &eigenvector);

    /**
     * @brief 将特征值和特征向量结果文件读入
     *
     * @param fileName 默认的特征值和特征向量结果文件,并支持指定文件读取
     * @return int 错误码 0：正常， -1：错误
     */
    int eigenFileRead(std::string fileName = "eigenValueResult.txt");

private:
    /**
     * @brief 将一个std::vector<double>写入txt文件
     *
     * @param outFile 输出流ofstream
     * @param eigenvalue 一个vector<double>数据
     * @return int 错误码 0：正常， -1：错误
     */
    int eigenVectorSave(std::ofstream &outFile, const std::vector<double> &eigenvalue);

    /**
     * @brief 将一个特征值数据写入eigenvalue_
     *
     * @param line 特征值所指行的字符串
     * @return int 错误码 0：正常， -1：错误
     */
    int eigenvalueWrite(std::string &line);

    /**
     * @brief 将一个特征向量数据写入eigenvector_
     *
     * @param line 特征值所指行的字符串
     * @return int 错误码 0：正常， -1：错误
     */
    int eigenvectorWrite(std::string &line);
};

#endif
