/**
 * @brief 将特征值和特征向量结果写入txt文件
 * @author mading
 * @date 2025-02-21
 */
 
 #include "mmio_eigen_result_save.h"

 int eigenResultSave(const std::vector<double> & eigenvalue, const std::vector<std::vector<double>> & eigenvector)
 {
    std::ofstream outFile("eigenValueResult.txt", std::ios::binary);  // 创建并打开一个文件
    if (!outFile.is_open()) {
        printf("Can't create a txt file to save eigenValue !!!");
        return -1;            // todo 明确错误码
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
    if(size == 0)    // 结果为空时
    {    
      outFile << "eigenValue:\n<0>\n\neigenVector:\n<0><0>" << std::endl;
    }
    else            // 结果存在时
    {
      int size_vec = eigenvector[0].size();
      outFile << "eigenValue:\n<" << size << ">\n";

      eigenVectorSave(outFile, eigenvalue);     // 存特征值

      outFile << "\n";
      outFile << "eigenVector:\n" << "<"<< size << ">" << "<"<< size_vec << ">" <<std::endl;

      for(int i=0; i <size; i++){               // 存特征向量
         eigenVectorSave(outFile, eigenvector[i]);
      }
    }

    outFile.close(); // 关闭文件

    return 0;
 }


 int eigenVectorSave(std::ofstream &outFile, const std::vector<double> &eigenvalue)
 {
   outFile << "\t";
   for(const auto& value : eigenvalue){
      outFile << " " << value;
   }

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