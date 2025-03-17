/**
 * @brief 读取输入数据的工具
 * @author zhangzy(zhiyu.zhang@cqbdri.pku.edu.cn)
 * @date 2025-03-11
 */

#include "input_read_tool.h"
#include "io/mmio_reader.h"
#include "io/mmloader.h"
#include <iostream>
#include <vector>

GcgeErrCode InputReadTool::ReadCcsFromMtx(CCSMAT* ccs_mat, char* file_matrix) {
    if (ccs_mat == NULL) {
        std::cerr << "Error: ccs_mat is NULL." << std::endl;
        return GCGE_ERR_NULLPTR;
    }
    if (file_matrix == NULL) {
        std::cerr << "Error: file_matrix is NULL." << std::endl;
        return GCGE_ERR_NULLPTR;
    }
    CreateCCSFromMTX(ccs_mat, file_matrix);
    return GCGE_SUCCESS;
}

GcgeErrCode InputReadTool::ReadPetscMatFromMtx(Mat* petsc_matA, char* file_matrix) {
    if (petsc_matA == NULL) {
        std::cerr << "Error: petsc_matA is NULL." << std::endl;
        return GCGE_ERR_NULLPTR;
    }
    if (file_matrix == NULL) {
        std::cerr << "Error: file_matrix is NULL." << std::endl;
        return GCGE_ERR_NULLPTR;
    }
    PetscBool aijonly = PETSC_FALSE;
    MatCreateFromMTX(petsc_matA, file_matrix, aijonly);

    return GCGE_SUCCESS;
}

GcgeErrCode InputReadTool::readUserParam(GcgeParam& param, ExtractMethod& method, std::string paramFileName) {
    if (paramFileName.empty()) {
        std::cerr << "Error: paramFileName is empty." << std::endl;
        return GCGE_ERR_FILE;
    }
    std::ifstream inFile(paramFileName); // 打开一个文件
    if (!inFile.is_open()) {
        std::cerr << "Error: Can't read the user param file: " << paramFileName << std::endl;
        return GCGE_ERR_FILE;
    }

    std::string line; // 临时保存getline()的字符串数据

    // 1.读取协议描述
    for (int i = 0; i < PROTOCOL_LINE; i++) {
        std::getline(inFile, line);
    }

    // 2.处理extractionMethod: <enum>
    std::getline(inFile, line);
    int type = obtainIntNumber(line);
    if (type == 0) {
        method.extractType = BY_ORDER;
    } else if (type == 1) {
        method.extractType = BY_FREQUENCY;
    } else if (type == 2) {
        method.extractType = BY_ORDER_AND_FREQUENCY;
    } else {
        std::cerr << "Error: extractMethod only support [0, 1, 2]" << std::endl;
        return GCGE_ERR_INPUT;
    }

    // 3.处理extractionOrder: <uint32_t>
    std::getline(inFile, line);
    int order = obtainIntNumber(line);
    method.extractOrder = order;

    // 4.处理minFreq: <double>
    std::getline(inFile, line);
    double minFreq = obtainDoubleNumber(line);
    method.minFreq = minFreq;

    // 5.处理maxFreq: <double>
    std::getline(inFile, line);
    double maxFreq = obtainDoubleNumber(line);
    method.maxFreq = maxFreq;

    // 6.处理maxIteration: <uint32_t>
    std::getline(inFile, line);
    int maxIteration = obtainIntNumber(line);
    param.max_iter_gcg = maxIteration;

    // 7.处理absoluteError: <double>
    std::getline(inFile, line);
    double absoluteError = obtainDoubleNumber(line);
    param.tol_gcg[0] = absoluteError;

    // 8.处理relativeError: <double>
    std::getline(inFile, line);
    double relativeError = obtainDoubleNumber(line);
    param.tol_gcg[1] = relativeError;

    // 9.处理initBlock: <uint32_t>
    std::getline(inFile, line);
    int initBlock = obtainIntNumber(line);

    // 10.处理pWBlock: <uint32_t>
    std::getline(inFile, line);
    int pWBlock = obtainIntNumber(line);

    inFile.close(); // 关闭文件
    return GCGE_SUCCESS;
}

int InputReadTool::obtainIntNumber(std::string& lineStr) {
    int startPos = lineStr.find(":") + 1;
    int endPos = lineStr.size();
    int num = 0;
    try {
        num = std::stoi(lineStr.substr(startPos, endPos));
        return num;
    } catch (const std::exception& e) {
        std::cout << "Error: Can't read user parameter correctly" << std::endl;
        return 0;
    }
}

double InputReadTool::obtainDoubleNumber(std::string& lineStr) {
    int startPos = lineStr.find(":") + 1;
    int endPos = lineStr.size();
    double num = 0;
    try {
        num = std::stod(lineStr.substr(startPos, endPos));
        return num;
    } catch (const std::exception& e) {
        std::cout << "Error: Can't read user parameter correctly" << std::endl;
        return 0;
    }
}

void InputReadTool::ConvertPetscMatToCCSMat(Mat src, CCSMAT &des) {
    // 0️⃣ 在我们的问题中，petsc的矩阵均是MATAIJ格式(对称)，因此需要将其转换为AIJ格式，以使用MatGetRow接口
    Mat srcMatAij;
    MatConvert(src, MATAIJ, MAT_INITIAL_MATRIX, &srcMatAij);

    PetscInt m, n;
    MatGetSize(srcMatAij, &m, &n);  // 获取矩阵的行数和列数

    // 1️⃣ 统计每列的非零元素数量，预分配 j_col
    std::vector<int> col_counts(n, 0);  // 统计每列的非零元素数目

    for (PetscInt i = 0; i < m; ++i) {
        PetscInt num_nonzeros;
        const PetscInt *col_indices;
        const PetscScalar *values_local;

        MatGetRow(srcMatAij, i, &num_nonzeros, &col_indices, &values_local);

        for (PetscInt k = 0; k < num_nonzeros; ++k) {
            col_counts[col_indices[k]]++;  // 统计该列的非零元素个数
        }

        MatRestoreRow(srcMatAij, i, &num_nonzeros, &col_indices, &values_local);
    }

    // 2️⃣ 计算 j_col（列偏移数组）
    std::vector<int> col_offsets(n + 1, 0);
    for (PetscInt j = 0; j < n; ++j) {
        col_offsets[j + 1] = col_offsets[j] + col_counts[j];
    }

    // 3️⃣ 预分配 CCS 结构的 data 和 i_row
    PetscInt nnz = col_offsets[n];  // 总非零元素数
    std::vector<double> values(nnz);
    std::vector<int> row_indices(nnz);

    // 4️⃣ 填充 data 和 i_row
    std::vector<int> col_pos(n, 0);  // 记录每列当前填充位置
    for (PetscInt j = 0; j < n; ++j) {
        col_pos[j] = col_offsets[j];  // 初始化每列的存储起点
    }

    for (PetscInt i = 0; i < m; ++i) {
        PetscInt num_nonzeros;
        const PetscInt *col_indices;
        const PetscScalar *values_local;

        MatGetRow(srcMatAij, i, &num_nonzeros, &col_indices, &values_local);

        for (PetscInt k = 0; k < num_nonzeros; ++k) {
            PetscInt j = col_indices[k];  // 该非零元素所在列
            PetscInt pos = col_pos[j];    // 该列当前填充位置

            row_indices[pos] = i;
            values[pos] = values_local[k];

            col_pos[j]++;  // 更新该列的填充位置
        }

        MatRestoreRow(srcMatAij, i, &num_nonzeros, &col_indices, &values_local);
    }

    // 5️⃣ 赋值给 des 结构
    des.nrows = m;
    des.ncols = n;
    des.data = new double[values.size()];
    des.i_row = new int[row_indices.size()];
    des.j_col = new int[col_offsets.size()];

    std::copy(values.begin(), values.end(), des.data);
    std::copy(row_indices.begin(), row_indices.end(), des.i_row);
    std::copy(col_offsets.begin(), col_offsets.end(), des.j_col);

    // 6️⃣ 释放资源
    MatDestroy(&srcMatAij);
}
