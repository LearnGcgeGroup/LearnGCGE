/**
 * @brief 特征值和特征向量写入txt文件，从txt文件读入已有结果
 * @author mading
 * @date 2025-02-21
 */
 
#include "io_eigen_result.h"

int IoEigenResult::saveEigenResult(const std::vector<double>& eigenvalue, const std::vector<std::vector<double>>& eigenvector, std::string fileName) {
    std::ofstream outFile(fileName, std::ios::binary);  // 创建并打开一个文件
    if (!outFile.is_open()) {
        std::cerr << "Error: Can't create a txt file to save eigen result !!!" << std::endl;
        return -1;            // todo 明确错误码
    }

    // 1.写入特征值与特征向量文件协议内容
    std::string protocolHeader = R"(%% Eigenvalue and Eigenvector File Protocol
%-----------------------------------------------------------------------------------------------------------------------------------------------------
% size: number of eigenvalues
% After <size>, a line containing eigenvalues as double values (corresponding to std::vector<double> in the software)
% rows: number of eigenvectors
% vector: dimension of each eigenvector
% After <rows> <vector>, there are 'rows' lines, each representing an eigenvector (corresponding to std::vector<std::vector<double>> in the software)
%----------------------------------------------------------------------------------------------------------------------------------------------------)";
    outFile << protocolHeader << std::endl;

    // 2.写入特征值与特征向量结果
    int size = eigenvalue.size();
    if (size == 0) { // 结果为空时
        outFile << "eigenValue:" << std::endl;
        outFile << "0" << std::endl;
        outFile << std::endl;
        outFile << "eigenVector:" << std::endl;
        outFile << "0 0" << std::endl;
    } else {        // 结果存在时
        outFile << "eigenValue:" << std::endl;
        outFile << size << std::endl;

        // 存特征值
        saveOneVector(outFile, eigenvalue); 

        outFile << std::endl;
        outFile << "eigenVector:" << std::endl;
        
        int sizeVector = eigenvector[0].size();
        outFile << size << " " << sizeVector << std::endl;

        // 存特征向量
        for (int i = 0; i < size; i++) {
            saveOneVector(outFile, eigenvector[i]); 
        }
    }

    outFile.close();
    return 0;
}

int IoEigenResult::saveOneVector(std::ofstream& outFile, const std::vector<double>& oneVector) {
    bool isFirst = true;// 是否为第一个数据
    for (const auto& value : oneVector) {
        if (isFirst) {  // 首个数据前无空格
            outFile << std::fixed << std::setprecision(PRECISION) << value;
            isFirst = false;
        } else {
            outFile << " " << std::fixed << std::setprecision(PRECISION) << value;
        }
    }

    outFile << std::endl;   // 换行
    return 0;
}

int IoEigenResult::readEigenFile(std::vector<double>& eigenvalue, std::vector<std::vector<double>>& eigenvector, std::string fileName) {
    std::ifstream inFile(fileName); // 打开一个文件
    if (!inFile.is_open()) {
        std::cerr << "Error: Can't read the eigen result file: " << fileName << std::endl;
        return -1; // todo 明确错误码
    }

    std::string line; // 临时保存getline()的字符串数据

    // 1.读取协议描述
    for (int i = 0; i < PROTOCOL_LINE; i++) {
        std::getline(inFile, line);
    }

    // 2.处理特征值数目信息
    std::getline(inFile, line); // 特征值数目信息
    int sizeLine = line.length();
    numEigenvalue_ = std::stoi(line.substr(0, sizeLine));

    // 3.读入特征值
    std::getline(inFile, line);
    readEigenvalue(eigenvalue, line);

    // 4.处理 空的一行 和 "eigenVector:" 信息
    std::getline(inFile, line);
    std::getline(inFile, line);

    // 5.处理特征向量数目信息
    std::getline(inFile, line);
    int sizeLine_vec = line.length();
    dimension_ = std::stoi(line.substr(sizeLine + 1, sizeLine_vec));

    // 6.读入特征向量
    for (int i = 0; i < numEigenvalue_; i++) {
        std::getline(inFile, line); // 读入一行特征向量数据
        readEigenvector(eigenvector, line);
    }

    inFile.close(); // 关闭文件
    return 0;
}

int IoEigenResult::readEigenvalue(std::vector<double>& eigenvalue, std::string& eigenvalueStr) {
    // 指定所需空间
    eigenvalue.reserve(numEigenvalue_); 

    std::istringstream iss(eigenvalueStr);
    std::string token;
    while (iss >> token) {
        eigenvalue.push_back(std::stod(token));
    }
    return 0;
}

int IoEigenResult::readEigenvector(std::vector<std::vector<double>>& eigenvector, std::string& eigenvectorStr) {
    std::vector<double> vectorTemp; // 临时存放特征向量
    vectorTemp.reserve(dimension_); // 指定所需空间

    std::istringstream iss(eigenvectorStr);
    std::string token;
    while (iss >> token) {
        vectorTemp.push_back(std::stod(token));
    }
    eigenvector.push_back(vectorTemp);
    return 0;
}