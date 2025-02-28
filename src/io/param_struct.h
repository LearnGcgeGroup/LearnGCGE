/**
 * @brief 求解参数的结构体定义
 * @author mading
 * @date 2025-02-27
 */

#ifndef _PARAM_STRUCT_H_
#define _PARAM_STRUCT_H_

struct GcgeParam {
    int nevConv{5};                // 希望收敛到的特征值个数
    int block_size{nevConv};       // 块大小
    int nevInit{2 * nevConv};      // 初始X块的大小
    int max_iter_gcg{1000};        // 最大迭代次数
    int nevMax{2 * nevConv};       // 最大特征值个数
    double tol_gcg[2]{1e-1, 1e-5}; //精度，0是绝对， 1 相对
    double shift = 1;
    int nevGiven{0};
    int multiMax{1};
    int flag{0}; // 是否使用外部线性方程组求解器
    double gapMin{1e-5};
};

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

#endif