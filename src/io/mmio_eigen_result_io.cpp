/**
 * @brief 将特征值和特征向量结果写入txt文件
 *
 * @author mading
 * @date 2025-02-21
 */

#include "mmio_eigen_result_io.h"

int MmioEigenResultIO::eigenResultSave(const std::vector<double> &eigenvalue, const std::vector<std::vector<double>> &eigenvector)
{
    std::ofstream outFile("eigenValueResult.txt", std::ios::binary); // 创建并打开一个文件
    if (!outFile.is_open())
    {
        std::cerr << "Error: Can't create a txt file to save eigenValue !!!" << std::endl;
        return -1; // todo 明确错误码
    }

    // 1.写入特征值与特征向量文件协议内容
    std::string valueHeader = R"(%% Eigenvalue and Eigenvector File Protocol
%-----------------------------------------------------------------------------------------------------------------------------------------------------
% size: number of eigenvalues
% After <size>, a line containing eigenvalues as double values (corresponding to std::vector<double> in the software)
% rows: number of eigenvectors
% vector: dimension of each eigenvector
% After <rows> <vector>, there are 'rows' lines, each representing an eigenvector (corresponding to std::vector<std::vector<double>> in the software)
%----------------------------------------------------------------------------------------------------------------------------------------------------)";
    outFile << valueHeader << std::endl;

    // 2.写入特征值与特征向量结果
    int size = eigenvalue.size();
    if (size == 0)
    { // 结果为空时
        outFile << "eigenValue:\n<0>\n\neigenVector:\n<0><0>" << std::endl;
    }
    else
    { // 结果存在时
        int size_vec = eigenvector[0].size();
        outFile << "eigenValue:\n<" << size << ">\n";

        eigenVectorSave(outFile, eigenvalue); // 存特征值

        outFile << "\n";
        outFile << "eigenVector:\n<" << size << "><" << size_vec << ">" << std::endl;

        for (int i = 0; i < size; i++)
        { // 存特征向量
            eigenVectorSave(outFile, eigenvector[i]);
        }
    }

    outFile.close(); // 关闭文件
    return 0;
}

int MmioEigenResultIO::eigenVectorSave(std::ofstream &outFile, const std::vector<double> &eigenvalue)
{
    outFile << "\t";
    for (const auto &value : eigenvalue)
    {
        outFile << " " << value;
    }

    // todo 大量结果数据的高效写入
    // // 初始化缓冲区
    // std::vector<char> buffer;
    // buffer.reserve(MBUFFER_SIZE);

    // // 辅助函数：刷新缓冲区到文件
    // auto flush_buffer = [&] {
    //    outFile.write(buffer.data(), buffer.size());
    //    buffer.clear();
    // };

    // // 数值转换临时缓存（每个double最多24字符）
    // std::array<char, 32> num_str;

    // // 分批次写入缓存buffer
    // for (size_t batch_start = 0; batch_start < eigenvalue.size(); batch_start += MBATCH_SIZE){
    //    size_t batch_end = std::min(batch_start + MBATCH_SIZE, eigenvalue.size());

    //    for (size_t i = batch_start; i < batch_end; ++i){
    //       // 快速将double转为字符串
    //       auto [ptr, ec] = std::to_chars(
    //          num_str.data(), num_str.data() + num_str.size(),
    //          eigenvalue[i], std::chars_format::fixed, 6
    //      );

    //      // 将转换结果写入缓冲区
    //      const size_t num_len = ptr - num_str.data();
    //      if (buffer.size() + num_len > buffer.capacity()) flush_buffer();
    //      buffer.insert(buffer.end(), num_str.begin(), num_str.begin() + num_len);
    //    }
    // }

    // flush_buffer();

    outFile << "\n";
    return 0;
}

int MmioEigenResultIO::eigenFileRead(std::string fileName)
{
    std::ifstream inFile(fileName); // 打开一个文件
    if (!inFile.is_open())
    {
        std::cerr << "Error: Can't read the eigen result file \"" << fileName << "\"" << std::endl;
        return -1; // todo 明确错误码
    }

    std::string line; // 临时保存getline()的字符串数据

    // 1.读取协议描述
    for (int i = 0; i < PROTOCOL_LINE; i++) // PROTOCOL_LINE=9,表示特征值和特征向量文件协议描述行数
    {
        std::getline(inFile, line);
    }

    // 2.处理特征值数目信息
    std::getline(inFile, line); // 特征值数目信息
    int sizeLength = line.length();
    size_ = std::stoi(line.substr(1, sizeLength - 2));

    // 3.读入特征值
    std::getline(inFile, line);
    eigenvalueWrite(line);

    // 4.处理 空的一行 和 "eigenVector:" 信息
    std::getline(inFile, line);
    std::getline(inFile, line);

    // 5.处理特征向量数目信息
    std::getline(inFile, line);
    int sizeLength_vec = line.length();
    dimension_ = std::stoi(line.substr(sizeLength + 1, sizeLength_vec - 2));

    // 6.读入特征向量
    eigenvector_.reserve(size_); // 提前指定所需空间
    for (int i = 0; i < size_; i++)
    {
        std::getline(inFile, line); // 读入一行特征向量数据
        eigenvectorWrite(line);
    }

    // std::cout << "eigenvector_.size(): " << eigenvector_.size() << std::endl;
    // std::cout << "eigenvector_[0].size(): " << eigenvector_[0].size() << std::endl;
    inFile.close(); // 关闭文件
    return 0;
}

int MmioEigenResultIO::eigenvalueWrite(std::string &line)
{
    line.erase(0, line.find_first_not_of(" ")); // 去除前导空格
    line.erase(line.find_last_not_of(" ") + 1); // 去除尾随空格

    eigenvalue_.reserve(size_); // 提前指定所需空间

    std::istringstream iss(line);
    std::string token;
    while (iss >> token)
    {
        eigenvalue_.push_back(std::stod(token));
    }
    return 0;
}

int MmioEigenResultIO::eigenvectorWrite(std::string &line)
{
    line.erase(0, line.find_first_not_of(" ")); // 去除前导空格
    line.erase(line.find_last_not_of(" ") + 1); // 去除尾随空格

    std::vector<double> vectorTemp;
    vectorTemp.reserve(dimension_); // 提前指定所需空间

    std::istringstream iss(line);
    std::string token;
    while (iss >> token)
    {
        vectorTemp.push_back(std::stod(token));
    }
    eigenvector_.push_back(vectorTemp);
    return 0;
}