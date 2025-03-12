/**
 * @brief 读取输入数据的工具
 * @author zhangzy(zhiyu.zhang@cqbdri.pku.edu.cn)
 * @date 2025-03-11
 */

#ifndef __GCGE_SRC_IO_READ_USER_PARAM_H__
#define __GCGE_SRC_IO_READ_USER_PARAM_H__

#include <iostream>
#include <fstream>
#include <string>
#include "param_struct.h"   // 求解参数的结构体定义
#include <stdexcept>        // 处理异常
#include "error_code.h"

extern "C" {
    #include "app_ccs.h"
    #include "app_lapack.h"
    #include "ops.h"
    #include "ops_config.h"
    #include "ops_eig_sol_gcg.h"
    #include "ops_lin_sol.h"
    #include "ops_orth.h"
    // #include "mmloader.h"
}

class InputReadTool {
public:
    static const int PROTOCOL_LINE = 12; // GCGE用户输入参数协议描述行数

public:
    /**
     * @brief 读取MTX文件生成CCS格式的矩阵
     * 
     * @param ccs_matA 矩阵地址
     * @param file_matrix 文件路径
     * @return GcgeErrCode 错误码
     */
    static GcgeErrCode ReadCcsFromMtx(CCSMAT *ccs_matA, char* file_matrix);

    /**
     * @brief 读取MTX文件生成CCS格式的矩阵
     * 
     * @param ccs_matA 矩阵地址
     * @param file_matrix 文件路径
     * @return GcgeErrCode 错误码
     */
    static GcgeErrCode ReadPetscMatFromMtx(CCSMAT *ccs_matA, char* file_matrix);

    /**
     * @brief 读取用户设置的求解参数txt文档
     *
     * @param param GcgeParam的结构体对象
     * @param method 抽取方法参数保存的结构体
     * @param paramFileName 求解参数文件名称
     * @return GcgeErrCode 错误码
     */
    static GcgeErrCode readUserParam(GcgeParam& param, ExtractMethod& method, std::string paramFileName = "usrParam.txt");

private:
    /**
     * @brief 从字符串中获得一个int类型数字
     *
     * @param lineStr 字符串
     * @return int类型的数字
     */
    static int obtainIntNumber(std::string& lineStr);

    /**
     * @brief 从字符串中获得一个double类型数字
     *
     * @param lineStr 字符串
     * @return double类型的数字
     */
    static double obtainDoubleNumber(std::string& lineStr);
};

#endif // __GCGE_SRC_IO_READ_USER_PARAM_H__