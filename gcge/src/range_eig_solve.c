/**
 * @brief 按区间求解特征值的功能实现
 * @note 依赖petsc与slepc实现
 * @date 2025-03-27
 */

#include "range_eig_solve.h"
#include "ops_eig_sol_gcg.h"
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <slepcbv.h>
#include <slepceps.h>

#define DEBUG 0
#define TIME_GCG 0
#define PRINT_FIRST_UNCONV 0

extern struct GCGSolver_ *gcg_solver;
extern RangeSharedData rangeSharedData;
extern int sizeN, startN, endN;
extern int sizeP, startP, endP;
extern int sizeW, startW, endW;
extern int sizeC, sizeX, sizeV, endX;

extern void **mv_ws[3];
extern double *dbl_ws;
extern int *int_ws;
extern struct OPS_ *ops_gcg;

/**
 * @brief 求解A * X = B 的多维线性方程组问题
 * 
 * @param ksp 线性方程组求解器
 * @param B 多维右端项
 * @param X 解向量组
 */
static void SolveLinearSystemsWithBV(KSP ksp, BV *B, BV *X) {
    PetscInt i, n, k;
    Vec b_vec; // 右端向量与解向量
    Vec b_data;

    // 获取 BV 的尺寸：n = 向量长度，k = BV 的列数
    BVGetSizes(*B, NULL, &n, &k);
    PetscErrorCode err;
    // 创建 BV 对象 X，用于存储解向量（和 B 同尺寸）
    err = BVDuplicate(*B, X);
    if (err) {
        printf("BVDuplicate error: %d\n", err);
        exit(1);
    }
    // 创建临时 PETSc Vec 对象用于 KSPSolve
    VecCreate(PETSC_COMM_WORLD, &b_vec);
    VecSetSizes(b_vec, n, n);
    if (err) {
        printf("VecSetSizes error: %d\n", err);
        exit(1);
    }

    // 遍历 BV 的每一列，求解 A x_i = b_i
    for (i = 0; i < k; i++) {
        // 从 BV B 中提取第 i 列到 b_vec
        err = BVGetColumn(*B, i, &b_data);
        // 创建 Vec b，复制 a 的大小和并行信息
        err = VecDuplicate(b_data, &b_vec);
        // 将 Vec a 的内容复制到 Vec b
        err = VecCopy(b_data, b_vec);
#if DEBUG
        printf("zzy before\n");
        VecView(b_vec, PETSC_VIEWER_STDOUT_WORLD);
#endif
        if (err) {
            printf("VecCopy error: %d\n", err);
            exit(1);
        }
        // 调用 KSPSolve 求解 A x_i = b_i
        err = KSPSolve(ksp, b_vec, b_vec);

#if DEBUG
        printf("zzy after\n");
        VecView(b_vec, PETSC_VIEWER_STDOUT_WORLD);
        exit(1);
#endif
        if (err) {
            printf("KSPSolve error: %d\n", err);
            exit(1);
        }
        // 恢复 BV 的数组（避免内存问题）
        BVRestoreColumn(*B, i, &b_data);
        // 将解向量 b_vec 存入 BV X 的第 i 列
        BVInsertVec(*X, i, b_vec);
    }
#if DEBUG
        printf("11111\n");
#endif

    // 清理临时向量
    VecDestroy(&b_vec);
    return;
}

void ComputeRangeRayleighRitz(double *ss_matA, double *ss_eval, double *ss_evec, double tol,
                              int nevConv, double *ss_diag, void *A, void *B, void **V, struct OPS_ *ops) {
    // 0、更新参数
    sizeV = sizeX + sizeP + sizeW;
    // 通过nevConv更新N与sizeC：nevConv - sizeC为新增的收敛特征值个数
    startN = startN + (nevConv - sizeC); // startN从未收敛的第一个特征值开始
    endN = endN + (nevConv - sizeC);
    endN = (endN < endX) ? endN : endX;

    sizeN = endN - startN;
    sizeC = nevConv;

    // 1、先计算Y = B V部分，并存储之后复用：暂时每次调用都创建，之后将此部分内存移出复用内存空间
    BV Y; // SLEPc 多向量
    int num_vec = *rangeSharedData.sizeV_ptr - *rangeSharedData.sizeC_ptr;
    printf("----ComputeRangeRayleighRitz ss_matA size: = %d\n", num_vec);
    ops->MultiVecCreateByMat(&Y, num_vec, B, ops);
    // 设置 BV 的活动列数 sizeX -sizeC
    BVSetActiveColumns((BV)V, *rangeSharedData.sizeC_ptr, *rangeSharedData.sizeV_ptr);
    BVSetActiveColumns(Y, *rangeSharedData.sizeC_ptr, *rangeSharedData.sizeV_ptr);
    // 执行矩阵与 BV 对象的乘法操作 Y = B ⋅ V
    BVMatMult((BV)V, (Mat)B, Y);

#if DEBUG
    ops->MultiVecView(Y, 0, 2, ops);
        {
            // 将BV转换成matlab矩阵的代码
            Mat         A;
            PetscViewer viewer;
            // 假设 bv 已经初始化并填充数据
            BVCreateMat((BV)V, &A);
            // 以 MATLAB 格式写入文件
            PetscViewerASCIIOpen(PETSC_COMM_WORLD, "V_output.m", &viewer);
            PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
            MatView(A, viewer);
            PetscViewerPopFormat(viewer);
            PetscViewerDestroy(&viewer);
            // 释放资源
            MatDestroy(&A);
        }
#endif
    // 2、计算(A - bB)
    Mat A_minus_bB; // 拷贝一份数据 用于作 A - a * B 和 A -b * B
    // 对计算 A - a * B 的负特征值数
    // 拷贝一份A数据到tempA
    MatDuplicate((Mat)A, MAT_COPY_VALUES, &A_minus_bB);
    PetscScalar val;
    PetscInt row = 0, col = 0;
    MatGetValue((Mat)A_minus_bB, row, col, &val);
#if DEBUG
    PetscPrintf(PETSC_COMM_WORLD, "A[%d][%d] = %g\n", row, col, (double)PetscRealPart(val));
#endif
// 计算 A_minus_bB = A - a * B
    MatAXPY(A_minus_bB, -gcg_solver->max_eigenvalue, (Mat)B, SUBSET_NONZERO_PATTERN);
#if DEBUG
    printf("gcg_solver->max_eigenvalue = %f\n", gcg_solver->max_eigenvalue);
#endif
    row = 0, col = 1;
    MatGetValue(A_minus_bB, row, col, &val);
#if DEBUG
    PetscPrintf(PETSC_COMM_WORLD, "A_minus_bB[%d][%d] = %g\n", row, col, (double)PetscRealPart(val));
#endif

    // 3、计算A_minus_bB^-1 * Y : (A_minus_bB = A - bB； Y = B V；)
    // 3.1、创建存储A_minus_bB^-1 * Y结果的数据结构
    BV X; // 存储 A_minus_bB^-1 * Y 的结果
    // 3.2、对A_minus_bB执行cholesky分解，得出M = L * L^T，在之后对Y的每一个向量进行求解时，可复用cholesky结果
    KSP ksp;
    PC pc;

    /* 创建线性求解器上下文 */
    PetscErrorCode ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    CHKERRQ(ierr);

    /* 设置运算符(矩阵) */
    ierr = KSPSetOperators(ksp, A_minus_bB, A_minus_bB);
    CHKERRQ(ierr);

    /* 设置求解器类型 - 对于对称不定系统，MINRES是一个好选择 */
    ierr = KSPSetType(ksp, KSPMINRES);
    CHKERRQ(ierr);

    /* 设置预条件器 - 对于对称不定系统，可以使用Jacobi或块Jacobi */
    ierr = KSPGetPC(ksp, &pc);
    CHKERRQ(ierr);
    ierr = PCSetType(pc, PCJACOBI);
    CHKERRQ(ierr);

    /* 设置求解器参数 */
    ierr = KSPSetTolerances(ksp, 1.e-14, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    CHKERRQ(ierr);

    // 3.3、计算A_minus_bB^-1 * Y
    SolveLinearSystemsWithBV(ksp, &Y, &X);
    // 清理
    KSPDestroy(&ksp);
#if DEBUG
    // 用于查看线性方程组求解是否正确
    ops->MultiVecView(X, 0, 2, ops);
#endif
    // 4、计算 V^T * B * X: (X = A_minus_bB^-1 * Y)
    // 因为B对称，所以V^T * B = V^T * B^T = (B * V)^T = Y^T, 因此计算 V^T * B * X = Y^T * X
    // 4.1、创建存储 Y^T * X 结果的数据结构，实际就是ss_matA
    Mat Y_TX;
    PetscInt k;
    ierr = BVGetActiveColumns(X, NULL, &k); // 获取 BV 维度
    if (ierr) {
        printf("BVGetActiveColumns error: %d\n", ierr);
        exit(1);
    }
    MatCreate(PETSC_COMM_WORLD, &Y_TX);
    MatSetSizes(Y_TX, k, k, k, k); // ？
    MatSetType(Y_TX, MATDENSE);    // MATDENSE=“稠密”-用于稠密矩阵的矩阵类型。当使用单个过程通信器构建时，此矩阵类型与MATSEQDENSE相同，否则与MATMPIDENSE相同。
    MatSetUp(Y_TX);

    // 计算 Y^T * X
    BVMatProject(Y, NULL, X, Y_TX);
    // MatView(Y_TX, PETSC_VIEWER_STDOUT_WORLD);
    // 5、将所有计算结果放置到原位置
    // 将数据拷贝到 mat (PETSc 默认列主序)
    for (PetscInt j = 0; j < k; j++) {     // 遍历列
        for (PetscInt i = 0; i < k; i++) { // 遍历行
            MatGetValue(Y_TX, i, j, &val);
            ss_matA[j * k + i] = val;
        }
    }
#if DEBUG
    printf("ss_matA:\n");
    for (PetscInt j = 0; j < k; j++) {
        for (PetscInt i = 0; i < k; i++) {
            printf("%e ", ss_matA[j * k + i]);
        }
        printf("\n");
    }
    exit(1);
#endif

    // 释放资源
    MatDestroy(&Y_TX);
    BVDestroy(&X);
    BVDestroy(&Y);
    MatDestroy(&A_minus_bB);

    // #################################################以上内容为修改的内容，即组装小规模矩阵，以下未修改，即求解特征值 ############################
    // 6、计算特征值
    int nrows, ncols, nrowsA, ncolsA, length, incx, incy, idx, start[2], end[2];
    double *source, *destin, alpha;

    /* 已收敛部分C不再考虑，更新 ss_mat ss_evec 起始地址*/
    // 由于sizeC大小变更，ss_matA与ss_evec均向后平移相应位置
    ss_matA = ss_diag + (sizeV - sizeC);
    ss_evec = ss_matA + (sizeV - sizeC) * (sizeV - sizeC);

    /* 记录对角线部分 */
    length = sizeV - sizeC;
    source = ss_matA;
    incx = (sizeV - sizeC) + 1;
    destin = ss_diag;
    incy = 1;
    dcopy(&length, source, &incx, destin, &incy);

    /* 对 ss_matA 进行 shift */
    // 实现逻辑只对对角线元素进行了shift
    // 该部分代码会进入执行(double类型值永远不会相等)，若想在compW_cg_shift不为0才进入，需要修改if判断逻辑
    if (gcg_solver->compW_cg_shift != 0.0) {
        alpha = 1.0;
        length = sizeV - sizeC;
        source = &(gcg_solver->compW_cg_shift);
        incx = 0;
        destin = ss_matA;
        incy = (sizeV - sizeC) + 1;
        daxpy(&length, &alpha, source, &incx, destin, &incy);
    }

#if DEBUG
    int row, col;
    ops_gcg->Printf("ss_diag:\n");
    for (idx = 0; idx < length; ++idx)
        ops_gcg->Printf("%f\n", destin[idx]);
#endif
    /* 基于LAPACK计算小规模特征值问题的参数设置 */
    char JOBZ, RANGE, UPLO;
    int LDA;
    int M; // 输出变量：找到的特征值总数
    int LDZ, INFO;
    int N; // 矩阵的阶数(行数/列数)
    int LWORK, *IWORK, *IFAIL;
    double ABSTOL;
    double *AA; // 输入矩阵A
    double *W;  // 输出变量：前 M 个元素包含按升序排列的选中特征值
    double *Z;  // 输出变量：前 M 列包含对应于选中特征值的正交特征向量
    double *WORK;
    JOBZ = 'V';          // 表示计算特征值和特征向量
    RANGE = 'A';         // 表示计算所有特征值
    UPLO = 'U';          // 表示存储上三角部分
    LDA = sizeV - sizeC; // 数组A的首维长度
    ABSTOL = tol;        // 特征值的绝对误差容限
    LDZ = sizeV - sizeC;
    IWORK = int_ws;
    INFO = 0;
    /* 不再计算 C 部分 */
    N = sizeV - sizeC;
    M = N;
    IFAIL = int_ws + 5 * N;
    AA = ss_matA;
    W = ss_eval + sizeC;
    Z = ss_evec;
    WORK = Z + LDZ * N;
    /* ss_diag ss_matA ss_evec 剩下的空间 */
    LWORK = gcg_solver->length_dbl_ws - (WORK - gcg_solver->dbl_ws);

#if DEBUG
    ops_gcg->Printf("LWORK = %d\n", LWORK);
    ops_gcg->Printf("dsyevx: AA\n");
    for (row = 0; row < N; ++row) {
        for (col = 0; col < N; ++col) {
            ops_gcg->Printf("%6.4e\t", AA[row + col * LDA]);
        }
        ops_gcg->Printf("\n");
    }
#endif

#if OPS_USE_MPI
    /* 当 PAS 调用 GCG 时, 且使用并行怎么办?
     * 没关系, PAS 需要保证每个进程都有特征向量
     * 同时, 这样的分批计算, 不仅仅是效率的提升
     * 更重要的是, 保证, 每个进程的特征向量完全一致 */
    int *displs;
    int sendcount, *recvcounts;
    double *recvbuf;
    int IL, IU;
    int rank, nproc;

    /* 每列多一行, 将特征值拷贝至此, 进行通讯 */
    LDZ = LDZ + 1;
    /* 特征向量不包含 C 的部分 */
    Z = ss_evec;
    /* 重置工作空间 */
    WORK = Z + LDZ * N;
    LWORK = LWORK - N;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    /* 分类特征值 */
    destin = ss_eval + sizeC;
    length = N;
    /* 分类特征值 */
    if (gcg_solver->compRR_min_num <= 0) {
        gcg_solver->compRR_min_num = N / (nproc + 2) > 10 ? N / (nproc + 2) : 10;
    }
    displs = malloc((2 * nproc + 1) * sizeof(int)); /* ������Ҫ 2*nproc+1 */
    if (rank == 0) {
        SplitDoubleArray(destin, length, nproc,
                         gcg_solver->compRR_min_gap,
                         gcg_solver->compRR_min_num,
                         displs, dbl_ws, int_ws);
    }
    MPI_Bcast(displs, nproc + 1, MPI_INT, 0, MPI_COMM_WORLD);
    sendcount = displs[rank + 1] - displs[rank];
    recvcounts = displs + nproc + 1;
    for (idx = 0; idx < nproc; ++idx) {
        recvcounts[idx] = displs[idx + 1] - displs[idx];
    }
    RANGE = 'I';
    /* 1 <= IL <= IU <= N */
    IL = displs[rank] + 1;
    IU = displs[rank + 1];
    M = IU - IL + 1;
    /* 不同进程 W Z 不同 */
    W += displs[rank];
    Z += LDZ * displs[rank];

#if TIME_GCG
    time_gcg.dsyevx_time -= ops_gcg->GetWtime();
#endif
    // printf("%d\n",sendcount);
    if (sendcount > 0) {
#if DEBUG
        ops_gcg->Printf("dsyevx: N   = %d, M  = %d, LDA = %d, IL = %d, IU  = %d, LDZ = %d\n",
                        N, M, LDA, IL, IU, LDZ);
#endif
        dsyevx(&JOBZ, &RANGE, &UPLO, &N, AA, &LDA,
               NULL, NULL, &IL, &IU, &ABSTOL, &M,
               W, Z, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO);
        assert(M == IU - IL + 1);
        if (INFO != 0) {
            ops_gcg->Printf("dsyevx: INFO = %d\n", INFO);
        }
    }
#if TIME_GCG
    time_gcg.dsyevx_time += ops_gcg->GetWtime();
    // ops_gcg->Printf("dsyevx = %.2f\n",time_gcg.dsyevx_time);
#endif
    /* 将计算得到的特征值复制到 Z 的最后一行 */
    length = sendcount;
    source = W;
    incx = 1;
    destin = Z + LDZ - 1;
    incy = LDZ;
    dcopy(&length, source, &incx, destin, &incy);
    recvbuf = ss_evec;
    sendcount *= LDZ;
    for (idx = 0; idx < nproc; ++idx) {
        recvcounts[idx] *= LDZ;
        displs[idx + 1] *= LDZ;
    }
    /* 全聚集特征对, 发送和接收都是连续数据 */

#if DEBUG
    ops_gcg->Printf("before allgaterv sendcount = %d\n", sendcount);
#endif
    MPI_Allgatherv(MPI_IN_PLACE, sendcount, MPI_DOUBLE,
                   recvbuf, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
#if DEBUG
    ops_gcg->Printf("after  allgaterv sendcount = %d\n", sendcount);
#endif
    free(displs);
    /* 将 Z 的最后一行复制给特征值 */
    length = N;
    source = ss_evec + LDZ - 1;
    incx = LDZ;
    destin = ss_eval + sizeC;
    incy = 1;
    dcopy(&length, source, &incx, destin, &incy);
    /* 移动特征向量 */
#if DEBUG
    ops_gcg->Printf("before memmove length = %d\n", length);
#endif
    length = N;
    destin = ss_evec;
    source = ss_evec;
    for (idx = 0; idx < N; ++idx) {
        /* 保证 source 在被覆盖之前
         * 将重叠区域的字节拷贝到 destin 中 */
        memmove(destin, source, length * sizeof(double));
        destin += N;
        source += LDZ;
    }
#if DEBUG
    ops_gcg->Printf("after  memmove length = %d\n", length);
#endif

#else

#if DEBUG
    ops_gcg->Printf("dsyevx: N = %d, M = %d\n", N, M);
#endif

#if TIME_GCG
    time_gcg.dsyevx_time -= ops_gcg->GetWtime();
#endif
    /* 保证 ss_evec 是正交归一的 */
    dsyevx(&JOBZ, &RANGE, &UPLO, &N, AA, &LDA,
           NULL, NULL, NULL, NULL, &ABSTOL, &M,
           W, Z, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO);
    assert(INFO == 0);
#if TIME_GCG
    time_gcg.dsyevx_time += ops_gcg->GetWtime();
#endif
#if DEBUG
    ops_gcg->Printf("dsyevx: N = %d, M = %d\n", N, M);
#endif
    assert(M == N);

#endif
#if 1
    // 打印求解出来的小规模特征值
    printf("    convergenced eigen count: %d, value: \n", M);
    for (idx = 0; idx < M; ++idx) {
        ops_gcg->Printf("        count: %d,  %e\n", idx, 1.0 / W[idx] + gcg_solver->max_eigenvalue);
    }
    // exit(1);
#endif
    /* 恢复ss_matA对角线部分 */
    length = sizeV - sizeC;
    source = ss_diag;
    incx = 1;
    destin = ss_matA;
    incy = (sizeV - sizeC) + 1;
    dcopy(&length, source, &incx, destin, &incy);

    /* 恢复特征值 W */
    // 对本次求解的sizeV - sizeC个特征值进行shift，求解的特征值存储在(W = ss_eval + sizeC) 内存位置，从代码看shift的值为-compW_cg_shift
    // 意义是什么？
    if (gcg_solver->compW_cg_shift != 0.0) {
        alpha = -1.0;
        length = sizeV - sizeC;
        source = &(gcg_solver->compW_cg_shift);
        incx = 0;
        destin = ss_eval + sizeC;
        incy = 1;
        daxpy(&length, &alpha, source, &incx, destin, &incy);
    }

#if DEBUG
    ops_gcg->Printf("dsyevx: ss_evec\n");
    for (row = 0; row < N; ++row) {
        for (col = 0; col < M; ++col) {
            ops_gcg->Printf("%6.4e\t", Z[row + col * LDZ]);
        }
        ops_gcg->Printf("\n");
    }
    ops_gcg->Printf("dsyevx: ss_eval\n");
    for (row = 0; row < M; ++row)
        ops_gcg->Printf("%6.4e\n", W[row]);
    ops_gcg->Printf("dsyevx: AA\n");
    for (row = 0; row < N; ++row) {
        for (col = 0; col < N; ++col) {
            ops_gcg->Printf("%6.4e\t", AA[row + col * LDA]);
        }
        ops_gcg->Printf("\n");
    }
#endif
#if TIME_GCG
    time_gcg.compRR_time += ops_gcg->GetWtime();
#endif
    return;
}
