/**
 * @brief 按范围求解特征值需要的一些计算工具函数
 * @date 2025-03-29
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief 二分查找,找最接近 target 的索引
 * 
 * @param arr 要查找的升序数组
 * @param left 查找范围的左边界
 * @param right 查找范围的右边界
 * @param target 目标值
 * @return int 查找到的索引
 */
static int befemBinarySearch(double *arr, int left, int right, double target) {
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (fabs(arr[mid] - target) < 1e-9) { // 允许浮点数误差
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    // 此时 left 指向最接近 target 的位置
    if (left > 0 && fabs(arr[left - 1] - target) < fabs(arr[left] - target)) {
        return left - 1; // 取更接近的元素
    }
    return left; // 返回最接近 target 的位置
}

/**
 * @brief 查找最接近 target 的 targetCount 个值的索引下界(起始位置)
 * @note 最终范围为[start, start + targetCount)
 * @param[in] ss_eval 要查找的升序数组
 * @param[in] startN 查找范围的左边界
 * @param[in] endW 查找范围的右边界
 * @param[in] target 目标值
 * @param[in] targetCount 要查找的个数
 * @param[out] start 找到的索引下界(起始位置)
 */
static void findClosestIndices(double *ss_eval, int startN, int endW, double target, double targetCount, int *start) {
    int index = befemBinarySearch(ss_eval, startN, endW, target); // 找到最接近 target 的索引

    // 双指针法寻找最接近的 targetCount 个元素
    int left = index - 1;
    int right = index;
    int count = 0;
    
    while (count < targetCount && left >= startN && right < endW) {
        if (fabs(ss_eval[left] - target) <= fabs(ss_eval[right] - target)) {
            left--;
        } else {
            right++;
        }
        count++;
    }

    // 补全剩余的元素
    while (count < targetCount && left >= startN) {
        left--;
        count++;
    }
    while (count < targetCount && right < endW) {
        right++;
        count++;
    }

    // 由于 left 多减了一次，所以需要 +1; 同时我们统计的是离startN的偏移量因此要减去startN
    *start = left + 1 - startN;
}
