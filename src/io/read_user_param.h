/**
 * @brief 读取用户设置的求解参数txt文档
 * @author mading
 * @date 2025-02-27
 */

#ifndef _READ_USER_PARAM_H_
#define _READ_USER_PARAM_H_

#include <iostream>
#include <fstream>
#include <string>
#include "param_struct.h"   // 求解参数的结构体定义
#include <stdexcept>        // 处理异常

class ReadUserParam {
private:
    const int PROTOCOL_LINE = 12; // GCGE用户输入参数协议描述行数

public:
    /**
     * @brief 读取用户设置的求解参数txt文档
     *
     * @param param GcgeParam的结构体对象
     * @param method 抽取方法参数保存的结构体
     * @param paramFileName 求解参数文件名称
     * @return int类型的数字
     */
    int readUserParam(GcgeParam& param, ExtractMethod& method, std::string paramFileName = "usrParam.txt");

private:
    /**
     * @brief 从字符串中获得一个int类型数字
     *
     * @param lineStr 字符串
     * @return int类型的数字
     */
    int obtainIntNumber(std::string& lineStr);

    /**
     * @brief 从字符串中获得一个double类型数字
     *
     * @param lineStr 字符串
     * @return double类型的数字
     */
    double obtainDoubleNumber(std::string& lineStr);
};

#endif