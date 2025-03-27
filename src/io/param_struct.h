/**
 * @brief 求解参数的结构体定义
 * @author mading
 * @date 2025-02-27
 */

#ifndef _PARAM_STRUCT_H_
#define _PARAM_STRUCT_H_

enum ExtractType {
    BY_ORDER = 0,               // 通过设置阶数提取
    BY_FREQUENCY = 1,           // 通过设置频率提取
    BY_ORDER_AND_FREQUENCY = 2  // 通过设置阶数和频率提取
};

struct ExtractMethod {
    ExtractType extractType{BY_ORDER};     // 提取方式，默认设置阶数提取
    int extractOrder{10};   // 提取阶数，默认提前前10阶
    double minFreq{0};      // 最小频率，默认0
    double maxFreq{100};    // 最小频率，默认100
};

struct GcgeParam {
    int nevConv{5};                // 希望收敛到的特征值个数
    int block_size{nevConv};       // 分块矩阵W或P的列数，预估大于所要求解的特征值的最大代数重数
    int nevInit{2 * nevConv};      // 初始选取X矩阵的列数
    int max_iter_gcg{1000};        // 最大迭代次数
    int nevMax{2 * nevConv};       // 整个任务所要求的特征对个数
    double tol_gcg[2]{1e-1, 1e-5}; //精度，0是绝对， 1 相对
    double shift = 1;
    int nevGiven{0}; // 当前批次求解前，收敛特征对的总个数
    int multiMax{1};
    int flag{0}; // 是否使用外部线性方程组求解器
    double gapMin{1e-5};
    ExtractMethod extMethod; // 提取特征值的方式
};

#endif