/**
 * @brief 读取输入数据的工具
 * @author zhangzy(zhiyu.zhang@cqbdri.pku.edu.cn)
 * @date 2025-03-11
 */

#include "input_read_tool.h"
#include "io/mmio_reader.h"
#include <iostream>

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

GcgeErrCode InputReadTool::ReadPetscMatFromMtx(CCSMAT* ccs_mat, char* file_matrix) {
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
