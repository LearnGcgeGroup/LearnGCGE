/**
 * @brief 基于LAPACK的矩阵和向量操作
 */

#ifndef _APP_LAPACK_H_
#define _APP_LAPACK_H_

#include "ops.h"

/* LAPACK矩阵结构体
 * 用于表示LAPACK库使用的列优先存储矩阵
 * 
 * 参数说明:
 *   data  - 指向双精度浮点数数组的指针，按列优先顺序存储矩阵元素
 *   nrows - 矩阵的行数
 *   ncols - 矩阵的列数
 *   ldd   - LAPACK中的Leading Dimension（主维度），
 *           对列优先存储的矩阵，表示分配的内存中每列的行数
 *           通常 ldd >= nrows，用于处理子矩阵或非连续存储
 */
typedef struct LAPACKMAT_ {
    double *data;
    int nrows;
    int ncols;
    int ldd;
} LAPACKMAT;
/* LAPACK向量类型别名
 * 与LAPACKMAT共享相同的内存结构，向量视为ncols=1的列矩阵
 * 
 *   data  - 指向连续存储的向量元素数组
 *   nrows - 向量长度
 *   ncols - 固定为1
 *   ldd   - 通常等于nrows
 */
typedef LAPACKMAT LAPACKVEC;

void OPS_LAPACK_Set(struct OPS_ *ops);

/**
 * @brief 计算双精度浮点数向量中所有元素的绝对值之和
 *
 * @param n 整数，输入向量中的元素数量。
 * @param dx 双精度浮点数数组，包含输入向量的元素。dim = ( 1 + ( n - 1 )*abs( incx) )
 * @param incx 整数，DX数组中元素之间的存储间隔。
 * @return 双精度浮点类型，返回所有元素的绝对值之和
 */
#define dasum FORTRAN_WRAPPER(dasum)
/**
 * @brief 计算标量 da 与向量 dx 的乘积，加到向量 dy 上，即 dy = da * dx + dy。
 *
 * @param n 整数，向量的元素数量。
 * @param da 双精度浮点数，乘法因子。
 * @param dx 双精度浮点数数组，包含输入向量 dx 的元素。dim = ( 1 + ( n - 1 )*abs( incx ) )
 * @param incx 整数，dx 数组中元素之间的存储间隔。
 * @param dy 双精度浮点数数组，包含输入向量 dy 的元素。dim = ( 1 + ( n - 1 )*abs( incy ) )
 * @param incy 整数，dy 数组中元素之间的存储间隔。
 */
#define daxpy FORTRAN_WRAPPER(daxpy)
/**
 * @brief 将向量 dx 的元素复制到向量 dy 中，即 dy = dx。
 *
 * @param n 整数，向量的元素数量。
 * @param dx 双精度浮点数数组，包含输入向量 dx 的元素。dim = ( 1 + ( n - 1 )*abs( incx ) )
 * @param incx 整数，dx 数组中元素之间的存储间隔。
 * @param dy 双精度浮点数数组，包含输入向量 dy 的元素。dim = ( 1 + ( n - 1 )*abs( incy ) )
 * @param incy 整数，dy 数组中元素之间的存储间隔。
 */
#define dcopy FORTRAN_WRAPPER(dcopy)
/**
 * @brief 计算向量 dx 和 dy 的点积，即 dx 与 dy 的点积。
 *
 * @param n 整数，向量的元素数量。
 * @param dx 双精度浮点数数组，包含输入向量 dx 的元素。dim = ( 1 + ( n - 1 )*abs( incx ) )
 * @param incx 整数，dx 数组中元素之间的存储间隔。
 * @param dy 双精度浮点数数组，包含输入向量 dy 的元素。dim = ( 1 + ( n - 1 )*abs( incy ) )
 * @param incy 整数，dy 数组中元素之间的存储间隔。
 * @return 双精度浮点数，向量 dx 和 dy 的点积
 */
#define ddot FORTRAN_WRAPPER(ddot)
/**
 * @brief 计算矩阵乘法 C := alpha * op(A) * op(B) + beta * C
 *        其中 A、B 和 C 是矩阵，alpha 和 beta 是标量，transa 和 transb 指示是否对矩阵 A 和 B 进行转置。
 *        op(X) 表示根据transa/transb的指示对 X 进行操作之后的矩阵，即 op(X) = X 或 op(X) = X^T。
 *
 * @param transa 字符型，指示是否对矩阵 A 进行转置，取值为 'N'/'n'（不转置）或 'T'/'t'/'C'/'c'（转置）。
 * @param transb 字符型，指示是否对矩阵 B 进行转置，取值为 'N'/'n'（不转置）或 'T'/'t'/'C'/'c'（转置）。
 * @param m 整数，矩阵 op(A) 和 C 的行数。要求 m >= 0。
 * @param n 整数，矩阵 op(B) 的列数和 C 的列数。要求 n >= 0。
 * @param k 整数，矩阵 op(A) 的列数和矩阵 op(B) 的行数。要求 k >= 0。
 * @param alpha 双精度浮点数，乘法因子。
 * @param a 双精度浮点数数组，包含矩阵 A 的元素。维度为 lda*ka,其中,当 transa = 'N'/'n'时 , ka = k;否则，ka = m.
 * @param lda 整数，a 数组的第一维度。当 transa = 'N'/'n'时，lda >= max(1,m);否则，lda >= max(1,k)。
 * @param b 双精度浮点数数组，包含矩阵 B 的元素。维度为 ldb*kb, 其中,当 transa = 'N'/'n'时 , kb = n;否则，kb = k。
 * @param ldb 整数，b 数组的第一维度。当 transa = 'N'/'n'时，ldb >= max(1,k);否则，ldb >= max(1,n)。
 * @param beta 双精度浮点数，乘法因子。当 beta = 0 时，C 可以不用在输入中设置。
 * @param c 双精度浮点数数组，包含矩阵 C 的元素。输入时，数组 C 的前 m 行 n 列部分必须包含矩阵 C，除非 beta 为零，在这种情况下，输入时不需要设置 C。输出时，数组 C 被替换为 m 行 n 列的矩阵 ( alpha*op( A )op( B ) + betaC )。
 * @param ldc 整数，c 数组的第一维度。 ldc >= max(1, m)。
 * @return 结果存储在 c 数组中。
 */
#define dgemm FORTRAN_WRAPPER(dgemm)
/**
 * @brief 计算矩阵和向量的乘积 y := alpha * A * x + beta * y 或 y := alpha * A^T * x + beta * y
 *        其中 A 是矩阵，x 和 y 是向量，alpha 和 beta 是标量，trans 指示是否对矩阵 A 进行转置。
 *
 * @param trans 字符型，指示是否对矩阵 A 进行转置，取值为 'N'/'n'（不转置）或 'T'/'t'/'C'/'c'（转置）。
 * @param m 整数，矩阵 A 的行数。m >= 0。
 * @param n 整数，矩阵 A 的列数。n >= 0
 * @param alpha 双精度浮点数，乘法因子。
 * @param a 双精度浮点数数组，包含矩阵 A 的元素。维度为 lda*n。
 * @param lda 整数，a 数组的第一维度。
 * @param x 双精度浮点数数组，包含向量 x 的元素。dim = ( 1 + ( n - 1 )*abs( incx ) )。
 * @param incx 向量 x 中元素之间的存储间隔,不能为0。
 * @param beta 双精度浮点数，乘法因子。当 beta=0 时，y 不需要在输入中设置。
 * @param y 双精度浮点数数组，包含向量 y 的元素。当 trans = 'N'/'n'时 dim >= ( 1 + ( m - 1 )* abs( incy ) ); 否则dim >= ( 1 + ( n - 1 )* abs( incy ) )。
 * 
 * @return 返回计算得到的 y 向量,储存在 y 数组中。
 */
#define dgemv FORTRAN_WRAPPER(dgemv)
/**
 * @brief 获取双精度浮点计算环境中机器相关常数。
 *
 * @param cmach 字符型，指定要返回的机器常数类型。
 *        - 'E'/'e': DLAMCH 返回 eps（相对机器精度）。
 *        - 'S'/'s': DLAMCH 返回 sfmin（安全最小值，使得 1/sfmin不会溢出）。
 *        - 'B'/'b': DLAMCH 返回 base（机器的数值基数）。
 *        - 'P'/'p': DLAMCH 返回 prec（精度，计算公式为 eps * base）。
 *        - 'N'/'n': DLAMCH 返回 t（尾数部分以 base 为底的位数）。
 *        - 'R'/'r': DLAMCH 返回 rnd（如果加法中发生四舍五入，返回 1.0，否则返回 0.0）。
 *        - 'M'/'m': DLAMCH 返回 emin（发生渐进下溢之前的最小指数）。
 *        - 'U'/'u': DLAMCH 返回 rmin（下溢阈值，计算公式为 base^(emin-1)）。
 *        - 'L'/'l': DLAMCH 返回 emax（发生溢出之前的最大指数）。
 *        - 'O'/'o': DLAMCH 返回 rmax（溢出阈值，计算公式为 (base^emax) * (1 - eps)）。
 * @return 一个双精度浮点数，表示对应的机器常数。
 */
#define dlamch FORTRAN_WRAPPER(dlamch)
/**
 * @brief 返回向量 dx 中元素绝对值最大的元素的索引。
 *
 * @param n 整数，向量的元素数量。
 * @param dx 双精度浮点数数组，包含输入向量 dx 的元素。dim = ( 1 + ( n - 1 )*abs( incx ) )
 * @param incx 整数，dx 数组中元素之间的存储间隔。
 * 
 * @return 向量 dx 中元素绝对值最大的元素的索引。
 */
#define idamax FORTRAN_WRAPPER(idamax)
/**
 * @brief 将标量 da 乘以向量 dx，结果存储回 dx 中，即 dx = da * dx。
 *
 * @param n 整数，向量的元素数量。
 * @param da 双精度浮点数，乘法因子。
 * @param dx 双精度浮点数数组，包含输入向量 dx 的元素。dim = ( 1 + ( n - 1 )*abs( incx ) )
 * @param incx 整数，dx 数组中元素之间的存储间隔。
 */
#define dscal FORTRAN_WRAPPER(dscal)
/**
 * @brief 计算矩阵乘法 ( C = alpha * A * B + beta * C ) 或 ( C = alpha * B * A + beta * C )，
 *        其中 A 是对称矩阵，B 和 C 是 ( m * n ) 维矩阵，alpha 和 beta是标量，
 *        side 和 uplo 指示矩阵的位置和存储方式。
 *
 * @param side 字符型，指示矩阵 A 在乘法中的位置，取值为 'L'（左乘，C := alpha * A * B + beta * C)
 *             或 'R'（右乘， C := alpha * B * A + beta * C)。
 * @param uplo 字符型，指示矩阵 A 的存储方式，取值为 'U'（上三角）或 'L'（下三角）。
 * @param m 整数，矩阵 C 的行数，m >=0 。
 * @param n 整数，矩阵 C 的列数，n >= 0 。
 * @param alpha 标量，乘法因子。
 * @param a 双精度浮点数数组，包含矩阵 A 的元素，维度为(lda, ka)。
 * @param lda 整数，a 数组的第一维度。
 * @param b 双精度浮点数数组，包含矩阵 B 的元素。
 * @param ldb 整数，b 数组的第一维度。
 * @param beta 标量，乘法因子。当 beta 被设置为 0 时，C 不需要在输入中设置。
 * @param c 双精度浮点数数组，包含矩阵 C 的元素。
 * @param ldc 整数，c 数组的第一维度。
 *
 *  @return 计算结果储存在 c 数组中。
 */
#define dsymm FORTRAN_WRAPPER(dsymm)
/**
 * @brief 计算矩阵和向量的乘积 ( y = alpha * A * x + beta \* y \)，
 *        其中 A 是 ( n * n ) 维对称矩阵，x 和 y 是向量，alpha 和 beta 是标量，
 *        uplo 指示矩阵 A 的存储方式。
 *
 * @param uplo 字符型，指示矩阵 A 的存储方式，取值为 'U'（上三角）或 'L'（下三角）。
 * @param n 整数，矩阵 A 的维度，\( n \geq 0 \)。
 * @param alpha 标量，乘法因子。
 * @param a 双精度浮点数数组，包含矩阵 A 的元素，维度为(lda, n)。
 * @param lda 整数，a 数组的第一维度。
 * @param x 双精度浮点数数组，包含向量 x 的元素。
 * @param incx 整数，x 数组中元素之间的存储间隔。
 * @param beta 标量，乘法因子。当 beta 被设置为 0 时，y 不需要在输入中设置。
 * @param y 双精度浮点数数组，包含向量 y 的元素。
 * @param incy 整数，y 数组中元素之间的存储间隔。
 */
#define dsymv FORTRAN_WRAPPER(dsymv)
/**
 * @brief 计算矩阵 A 的列主元置换的 QR 分解，满足以下形式：\( A \* P = Q \* R \)。
 *
 * @param m 整数，矩阵 A 的行数，\( m \geq 0 \)。
 * @param n 整数，矩阵 A 的列数，\( n \geq 0 \)。
 * @param a 双精度浮点数数组，包含矩阵 A 的元素，维度为(lda, n)。
 * @param lda 整数，a 数组的第一维度。
 * @param jpvt 整数数组，包含列交换信息，维度为 n。
 * @param tau 双精度浮点数数组，包含反射系数，维度为 min(m, n)。
 * @param work 双精度浮点数数组，维度为 max(1, lwork)。
 * @param lwork 整数，work 数组的大小，lwork ≥ 3n+1。
 * @param info 整数，返回状态信息。
 */
#define dgeqp3 FORTRAN_WRAPPER(dgeqp3)
/**
 * @brief 生成矩阵 A 的正交矩阵 Q，使得 \( A = Q \* R \)，其中 R 是上三角矩阵。
 *
 * @param m 整数，矩阵 A 的行数，\( m \geq 0 \)。
 * @param n 整数，矩阵 A 的列数，\( m \geq n \geq 0 \)。
 * @param k 整数，用于定义矩阵 Q 的初等反射子（Householder 反射子）的数量。
 * @param a 双精度浮点数数组，包含矩阵 A 的元素，维度为(lda, n)。
 * @param lda 整数，a 数组的第一维度。
 * @param tau 双精度浮点数数组，包含反射系数。
 * @param work 双精度浮点数数组，维度为 max(1, lwork)。
 * @param lwork 整数，work 数组的大小，LWORK ≥ max(1, N)。
 * @param info 整数，返回状态信息。
 */
#define dorgqr FORTRAN_WRAPPER(dorgqr)
/**
 * @brief 计算实矩阵 A 的 RQ 分解，即 \( A = R \* Q \)。
 *
 * @param m 整数，矩阵 A 的行数，\( m \geq 0 \)。
 * @param n 整数，矩阵 A 的列数，\( n \geq 0 \)。
 * @param a 双精度浮点数数组，包含矩阵 A 的元素，维度为(lda, n)。
 * @param lda 整数，a 数组的第一维度。
 * @param tau 双精度浮点数数组，存储初等反射子的标量因子。
 * @param work 双精度浮点数数组，维度为 max(1, lwork)。
 * @param lwork 整数，work 数组的大小，LWORK ≥ 1。
 * @param info 整数，返回状态信息。
 */
#define dgerqf FORTRAN_WRAPPER(dgerqf)
/**
 * @brief 生成 m×n 维实矩阵 Q，其行进行了正交归一化。
 *
 * @param m 整数，矩阵 Q 的行数，\( m \geq 0 \)。
 * @param n 整数，矩阵 Q 的列数，\( n \geq m \)。
 * @param k 整数，初等反射子的数量。
 * @param a 双精度浮点数数组，维度为(lda, n)。
 * @param lda 整数，a 数组的第一维度。
 * @param tau 双精度浮点数数组，维度为 k。
 * @param work 双精度浮点数数组，维度为 max(1, lwork)。
 * @param lwork 整数，work 数组的大小，lwork ≥ 1。
 * @param info 整数，返回状态信息。
 */
#define dorgrq FORTRAN_WRAPPER(dorgrq)
/**
 * @brief 计算实对称矩阵 A 的特征值和特征向量（可选择）。
 *
 * @param jobz 字符型，指示是否计算特征向量：
 *             - 'V'：计算特征向量和特征值。
 *             - 'N'：不计算特征向量，只计算特征值。
 * @param uplo 字符型，指示矩阵 A 的存储方式：
 *             - 'U'：矩阵的上三角部分包含数据。
 *             - 'L'：矩阵的下三角部分包含数据。
 * @param n 整数，矩阵 A 的维度（即行数和列数）。n >= 0
 * @param a A 是双精度浮点数数组，维度为 (lda, n)。
 *          输入时，A 存储对称矩阵：
 *          - 若 UPLO = 'U'，则 A 的前 N×N 个上三角部分包含矩阵 A 的上三角部分。
 *          - 若 UPLO = 'L'，则 A 的前 N×N 个下三角部分包含矩阵 A 的下三角部分。
 *          计算完成后：
 *          - 若 JOBZ = 'V'，且 INFO = 0，则 A 包含矩阵 A 的正交归一化特征向量。
 *          - 若 JOBZ = 'N'，则在退出时，A 的下三角（若 UPLO = 'L'）或上三角（若 UPLO = 'U'），包括对角线元素，都会被破坏。
 * @param lda 整数，数组 a 的第一维度。lda >= max(1, n)
 * @param w 双精度浮点数数组，dim=n。如果 info=0，特征值按升序排列。
 * @param work 双精度浮点数数组。dim = max(1, lwork)。如果 info=0，lwork(1) 返回最优的 lwork
 * @param lwork 整数，work 数组的大小。必须满足 lwork ≥ max(1, 3 * n - 1)。
 *              为了获得最佳计算效率，建议 lwork ≥ (nb + 2) * n，其中 nb 是 ilaenv 返回的 dsytrd 例程的块大小。
 *              如果 lwork = -1，则进行工作空间查询，例程只计算 work 数组的最优大小，并将该值作为 work 数组的第一个元素返回，
 *              同时不会由 xerbla 触发与 lwork 相关的错误消息。
 * @param info 整数，返回状态信息：
 *             - info = 0: 成功退出；
 *             - info < 0: 如果 info = -i，那么第 i 个参数有非法值；
 *             - info > 0: 如果 INFO = i，则算法未能收敛；在中间三对角形式中，有 i 个非对角元素未能收敛到零。
 */
#define dsyev FORTRAN_WRAPPER(dsyev)
/**
 * @brief 计算实对称矩阵 A 的特定特征值，并可选计算对应的特征向量。
 *        特征值和特征向量的选择可以通过指定特征值的取值范围或索引范围来实现。
 *
 * @param jobz 字符型，指示是否计算特征向量：
 *             - 'V'：计算特征向量。
 *             - 'N'：不计算特征向量，只计算特征值。
 * @param range 字符型，指定特征值的选择范围：
 *              - 'A'：计算所有特征值。
 *              - 'V'：计算位于指定区间 [vl, vu] 中的特征值。
 *              - 'I'：计算指定索引区间 [il, iu] 中的特征值。
 * @param uplo 字符型，指示矩阵 A 的存储方式：
 *             - 'U'：矩阵的上三角部分包含数据。
 *             - 'L'：矩阵的下三角部分包含数据。
 * @param n 整数，矩阵 A 的维度。n >= 0
 * @param a 双精度浮点数数组，维度为(lda, n)。
 *          输入时，A 存储对称矩阵：
 *          - 若 UPLO = 'U'，则 A 的前 N×N 个上三角部分包含矩阵 A 的上三角部分。
 *          - 若 UPLO = 'L'，则 A 的前 N×N 个下三角部分包含矩阵 A 的下三角部分。
 *          计算完成后，A 的下三角（若 UPLO = 'L'）或上三角（若 UPLO = 'U'），包括对角线元素，都将被销毁。
 * @param lda 整数，数组 a 的第一维度。lda >= max(1, n)
 * @param vl 双精度浮点数，如果 RANGE = 'V'，则 VL 表示搜索特征值的区间下界，且必须满足 VL < VU。
 *           如果 RANGE = 'A' 或 'I'，则不会引用 VL。
 * @param vu 双精度浮点数，如果 RANGE = 'V'，则 VU 表示搜索特征值的区间上界，且必须满足 VL < VU。
 *           如果 RANGE = 'A' 或 'I'，则不会引用 VU。
 * @param il 整数，如果 RANGE = 'I'，则 il 表示要返回的最小特征值的索引。
 *           必须满足 1 <= il <= iu <= n（当 n > 0 时）；如果 n = 0，则 il = 1 且 iu = 0。
 *           如果 range = 'A' 或 'V'，则不会引用 IL。
 * @param iu 整数，如果 range = 'I'，则 iu 表示要返回的最大特征值的索引。
 *           必须满足 1 <= il <= iu <= n（当 n > 0 时）；如果 n = 0，则 il = 1 且 iu = 0。
 *           如果 range = 'A' 或 'V'，则不会引用 IU。
 * @param abstol 双精度浮点数，表示特征值计算的绝对误差容限。
 *               当确定一个近似特征值位于宽度小于或等于以下值的区间 [a, b] 内时，该特征值被认为已收敛：
 *               abstol + EPS * max( |a|, |b| )。其中，EPS 是机器精度。
 *               如果 abstol 小于或等于零，则会使用 EPS * |T| 代替，其中 |T| 是将 A 化为三对角形式后得到的三对角矩阵的 1 - 范数。
 *               当 abstol 设为欠流阈值的两倍（即 2 * DLAMCH('S')）时，特征值的计算精度最高，而不是设为零。
 *               如果该例程返回 info > 0，表明某些特征向量未收敛，可尝试将 abstol 设为 2 * DLAMCH('S')。
 * @param m 整数，找到的特征值总数，0 <= m <= n。如果 range = 'A'，则 m = n；如果 range = 'I'，则 m = iu - il + 1。
 * @param w 双精度浮点数数组，维度为 (N)。在正常退出时，前 M 个元素包含按升序排列的选定特征值。
 * @param z 双精度浮点数数组，维度为 (LDZ, max(1, M))。
 *          如果 jobz = 'V'，且 info = 0，则 z 的前 m 列包含矩阵 A 的正交特征向量，
 *          这些特征向量对应于选定的特征值，z 的第 i 列包含与 w(i) 相关联的特征向量。
 *          如果特征向量未收敛，则该列包含特征向量的最新近似值，并且该特征向量的索引会返回在 ifail 中。
 *          如果 jobz = 'N'，则 z 不被引用。
 *          注意：用户必须确保在数组 z 中提供至少 max(1, m) 列。如果 range = 'V'，则 m 的确切值无法预先确定，必须使用上界。
 * @param ldz 整数，表示数组 z 的主维度。ldz >= 1，如果 jobz = 'V'，则 ldz >= max(1, n)。
 * @param work 双精度浮点数数组，dim = max(1, lwork)。如果 info = 0，lwork(1) 返回最优的 lwork
 * @param lwork 整数，表示数组 work 的长度。lwork >= 1，当 n <= 1 时；否则，lwork >= 8 * N。
 *              为了最优效率，lwork >= (nb + 3) * n，其中 nb 是 DSYTRD 和 DORMTR 的块大小的最大值，由 ilaenv 返回。
 *              如果 lwork = -1，则假定为工作区查询；该例程仅计算 work 数组的最优大小，
 *              将此值作为 work 数组的第一个元素返回，并且不会因 lwork 引发与 XERBLA 相关的错误消息。
 * @param iwork 整数数组，dim = 5 * n。
 * @param ifail 整数数组，维度为 (N)。
 *              - 如果 jobz = 'V'，则当 info = 0 时，ifail 的前 M 个元素为零。
 *              - 如果 info > 0，则 ifail 包含未能收敛的特征向量的索引。
 *              - 如果 jobz = 'N'，则 ifail 不被引用。
 * @param info 整数，返回状态信息：
 *             - info = 0: 成功退出；
 *             - info < 0: 如果 info = -i，那么第 i 个参数有非法值；
 *             - info > 0: 如果 INFO = i，如果有 i 个特征向量未能收敛，它们的索引将存储在数组 ifail 中。
 */
#define dsyevx FORTRAN_WRAPPER(dsyevx)

#if !OPS_USE_INTEL_MKL
/* BLAS */
double dasum(int *n, double *dx, int *incx);
int daxpy(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
int dcopy(int *n, double *dx, int *incx, double *dy, int *incy);
double ddot(int *n, double *dx, int *incx, double *dy, int *incy);
int dgemm(char *transa, char *transb, int *m, int *n, int *k,
          double *alpha, double *a, int *lda,
          double *b, int *ldb,
          double *beta, double *c, int *ldc);
int dgemv(char *trans, int *m, int *n,
          double *alpha, double *a, int *lda,
          double *x, int *incx,
          double *beta, double *y, int *incy);
double dlamch(char *cmach);
int idamax(int *n, double *dx, int *incx);
int dscal(int *n, double *da, double *dx, int *incx);
int dsymm(char *side, char *uplo, int *m, int *n,
          double *alpha, double *a, int *lda,
          double *b, int *ldb,
          double *beta, double *c, int *ldc);
int dsymv(char *uplo, int *n,
          double *alpha, double *a, int *lda,
          double *x, int *incx,
          double *beta, double *y, int *incy);
/* LAPACK */
/* DGEQP3 computes a QR factorization with column pivoting of 
 * a matrix A:  A*P = Q*R  using Level 3 BLAS 
 * LWORK >= 2*N+( N+1 )*NB, where NB is the optimal blocksize */
int dgeqp3(int *m, int *n, double *a, int *lda, int *jpvt,
           double *tau, double *work, int *lwork, int *info);
/* DORGQR generates an M-by-N real matrix Q with 
 * orthonormal columns 
 * K is the number of elementary reflectors whose product 
 * defines the matrix Q. N >= K >= 0.
 * LWORK >= N*NB, where NB is the optimal blocksize */
int dorgqr(int *m, int *n, int *k, double *a, int *lda,
           double *tau, double *work, int *lwork, int *info);
/* The length of the array WORK.  LWORK >= 1, when N <= 1;
 * otherwise 8*N.
 * For optimal efficiency, LWORK >= (NB+3)*N,
 * where NB is the max of the blocksize for DSYTRD and DORMTR
 * returned by ILAENV. */
/* RQ factorization */
int dgerqf(int *m, int *n, double *a, int *lda,
           double *tau, double *work, int *lwork, int *info);
int dorgrq(int *m, int *n, int *k, double *a, int *lda,
           double *tau, double *work, int *lwork, int *info);
int dsyev(char *jobz, char *uplo, int *n,
          double *a, int *lda, double *w,
          double *work, int *lwork, int *info);
int dsyevx(char *jobz, char *range, char *uplo, int *n,
           double *a, int *lda, double *vl, double *vu, int *il, int *iu,
           double *abstol, int *m, double *w, double *z, int *ldz,
           double *work, int *lwork, int *iwork, int *ifail, int *info);
#endif

#endif /* -- #ifndef _APP_LAPACK_H_ -- */
