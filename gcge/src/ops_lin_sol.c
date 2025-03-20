

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ops_lin_sol.h"
#define DEBUG 0

#define TIME_BPCG 0
#define TIME_BAMG 0

typedef struct TimeBlockPCG_ {
    double allreduce_time;
    double axpby_time;
    double innerprod_time;
    double matvec_time;
    double time_total;
} TimeBlockPCG;

typedef struct TimeBlockAMG_ {
    double axpby_time;
    double bpcg_time;
    double fromitoj_time;
    double matvec_time;
    double time_total;
} TimeBlockAMG;

struct TimeBlockPCG_ time_bpcg = {0.0, 0.0, 0.0, 0.0, 0.0};
struct TimeBlockAMG_ time_bamg = {0.0, 0.0, 0.0, 0.0, 0.0};

void PCG(void *mat, void *b, void *x, struct OPS_ *ops) {
    PCGSolver *pcg = (PCGSolver *)ops->linear_solver_workspace;
    int niter, max_iter = pcg->max_iter;
    double rate = pcg->rate, tol = pcg->tol;
    double alpha, beta, rho1, rho2, init_error, last_error, pTw;
    void *r, *p, *w;
    r = pcg->vec_ws[0];
    p = pcg->vec_ws[1];
    w = pcg->vec_ws[2];

    // tol = tol*norm2(b)
    if (0 == strcmp("rel", pcg->tol_type)) {
        ops->VecInnerProd(b, b, &pTw, ops);
        tol = tol * sqrt(pTw);
    }
    ops->MatDotVec(mat, x, r, ops);
    ops->VecAxpby(1.0, b, -1.0, r, ops); //r = b-A*x
    ops->VecInnerProd(r, r, &rho2, ops);
    init_error = sqrt(rho2);
    last_error = init_error;
    niter = 0;

    while ((last_error > rate * init_error) && (last_error > tol) && (niter < max_iter)) {
        //compute the value of beta
        if (niter == 0)
            beta = 0.0;
        else
            beta = rho2 / rho1;
        //set rho1 as rho2
        rho1 = rho2;
        //compute the new direction: p = r + beta * p
        ops->VecAxpby(1.0, r, beta, p, ops);
        //compute the vector w = A*p
        ops->MatDotVec(mat, p, w, ops);
        //compute the value pTw = p^T * w
        ops->VecInnerProd(p, w, &pTw, ops);
        //compute the value of alpha
        alpha = rho2 / pTw;
        //compute the new solution x = alpha * p + x
        ops->VecAxpby(alpha, p, 1.0, x, ops);
        //compute the new residual: r = - alpha*w + r
        ops->VecAxpby(-alpha, w, 1.0, r, ops);
        //compute the new rho2
        ops->VecInnerProd(r, r, &rho2, ops);
        last_error = sqrt(rho2);
        //update the iteration time
        ++niter;
    }
    pcg->niter = niter;
    pcg->residual = last_error;
    return;
}

/**
 * @brief �ڵ���LinearSolver֮ǰ��Ҫ����LinearSolver
 *        �ٴε���LinearSolverʱ�������������ʱ�ռ䲻�䣬�����ٴε���
 */
void LinearSolverSetup_PCG(int max_iter, double rate, double tol,
                           const char *tol_type, void *vec_ws[3], void *pc, struct OPS_ *ops) {
    /* ֻ��ʼ��һ�Σ���ȫ�ֿɼ� */
    static PCGSolver pcg_static = {
        .max_iter = 50,
        .rate = 1e-2,
        .tol = 1e-12,
        .tol_type = "abs",
        .vec_ws = {},
        .pc = NULL};
    pcg_static.max_iter = max_iter;
    pcg_static.rate = rate;
    pcg_static.tol = tol;
    strcpy(pcg_static.tol_type, tol_type);
    pcg_static.vec_ws[0] = vec_ws[0];
    pcg_static.vec_ws[1] = vec_ws[1];
    pcg_static.vec_ws[2] = vec_ws[2];
    pcg_static.niter = 0;
    pcg_static.residual = -1.0;

    ops->linear_solver_workspace = (void *)(&pcg_static);
    ops->LinearSolver = PCG;
    return;
}

/**
 * @brief 实现块预条件共轭梯度 (BlockPCG) 求解器。
 *
 * 该函数使用 BlockPCG 算法求解块线性系统Ax=b,其中A是稀疏对称正定矩阵。它对解向量进行迭代更新，直到满足收敛准则。
 * TODO:相较于标准 PCG , Block PCG 允许 同时求解多个未知量 (多个列向量)
 * 
 * @param mat 指向线性系统中的矩阵的指针。
 * @param mv_b 指向右端项向量的指针。
 * @param mv_x 指向解向量的指针。
 * @param start_bx 指向块向量起始索引的指针。
 * @param end_bx 指向块向量结束索引的指针。
 * @param ops 指向 OPS_ 结构的指针，该结构包含操作函数指针。
 *
 * @note 此实现支持基于 MPI 的并行计算和时间分析。
 *
 * 该函数遵循以下主要步骤：
 * 1. 计算右端项 (RHS) 向量的范数。
 * 2. 初始化残差并确定未收敛的块。
 * 3. 使用 PCG 方法迭代更新解：
 *    - 计算搜索方向。
 *    - 执行矩阵-向量乘法。
 *    - 计算内积并更新系数。
 *    - 更新解向量和残差向量。
 * 4. 检查收敛准则，满足条件时停止。
 *
 * @note 可以启用时间宏 (TIME_BPCG) 进行性能分析。
 *
 * @warning 在调用此函数之前，请确保输入指针已正确分配。
 */
void BlockPCG(void *mat, void **mv_b, void **mv_x,
              int *start_bx, int *end_bx, struct OPS_ *ops) {
    // 接收了 ops->multi_linear_solver_workspace 指向的已有工作空间。
    BlockPCGSolver *bpcg = (BlockPCGSolver *)ops->multi_linear_solver_workspace; // 使用前要保证ops->multi_linear_solver_workspace已经初始化，不能指向NULL
    /*niter：当前迭代次数		max_iter：最大迭代次数
	num_block：未收敛向量的块数		num_unconv：未收敛向量个数
	unconv：存储未收敛向量的索引	block：存储块分区信息*/
    int niter, max_iter = bpcg->max_iter, idx, col, length, start[2], end[2],
               num_block, *block, pre_num_unconv, num_unconv, *unconv;
    double rate = bpcg->rate, tol = bpcg->tol;
    double alpha, beta, *rho1, *rho2, *pTw, *norm_b, *init_res, *last_res, *destin;
    void **mv_r, **mv_p, **mv_w;
    mv_r = bpcg->mv_ws[0]; // 余量向量 r
    mv_p = bpcg->mv_ws[1]; // 搜索方向向量 p
    mv_w = bpcg->mv_ws[2]; // 预处理后的向量 A * p
    assert(end_bx[0] - start_bx[0] == end_bx[1] - start_bx[1]);
    /*dbl_ws中的位置分配：
    *   norm_b---(num_unconv个double)---rho1---(num_unconv个double)---rho2---(num_unconv个double)---pTw---(num_unconv个double)---init_res---(num_unconv个double)---last_res
    */
    num_unconv = end_bx[0] - start_bx[0]; // 最初预设所有向量都未受敛
    // 分配各个double变量在dbl_ws中的存储地址
    norm_b = bpcg->dbl_ws;
    rho1 = norm_b + num_unconv;
    rho2 = rho1 + num_unconv;
    pTw = rho2 + num_unconv;
    init_res = pTw + num_unconv;
    last_res = init_res + num_unconv;
    unconv = bpcg->int_ws;       // int工作空间的地址赋给unconv,用来存储未收敛向量的索引
    block = unconv + num_unconv; // 指向int_ws工作空间的起始位置偏移num_unconv 个int的位置

// 如果需要记录用时
#if TIME_BPCG
    time_bpcg.allreduce_time = 0.0;
    time_bpcg.axpby_time = 0.0;
    time_bpcg.innerprod_time = 0.0;
    time_bpcg.matvec_time = 0.0;
#endif

    /*计算右端项b的范数*/
    if (0 == strcmp("rel", bpcg->tol_type)) { //如果误差形式为相对误差
#if TIME_BPCG
        time_bpcg.innerprod_time -= ops->GetWtime();
#endif
        start[0] = start_bx[0];
        end[0] = end_bx[0];
        start[1] = start_bx[0];
        end[1] = end_bx[0];
        // 计算向量内积，将mv_b的内积存储在norm_b中
        // nsdIP设置为'D'是因为对于[a1,a2],[a1,a2],只计算a1*a1,a2*a2,而不计算a1*a2
        ops->MultiVecInnerProd('D', mv_b, mv_b, 0, start, end, norm_b, 1, ops);
#if TIME_BPCG
        time_bpcg.innerprod_time += ops->GetWtime();
#endif
        for (idx = 0; idx < num_unconv; ++idx) {
            norm_b[idx] = sqrt(norm_b[idx]); // 逐个计算sqrt(b^T*b)
        }
    } else if (0 == strcmp("user", bpcg->tol_type)) { //如果误差形式为用户自定义
        /* user defined norm_b */
        for (idx = 0; idx < num_unconv; ++idx) {
            norm_b[idx] = fabs(norm_b[idx]); // 计算绝对值
            //ops->Printf("%e\n",norm_b[idx]);
        }
    } else {
        for (idx = 0; idx < num_unconv; ++idx) { // 如果误差形式为绝对误差
            norm_b[idx] = 1.0;                   // 初始化为1
        }
    }

#if TIME_BPCG
    time_bpcg.matvec_time -= ops->GetWtime();
#endif
    /* 计算初始残差 r = b - Ax */
    start[0] = start_bx[1];
    end[0] = end_bx[1];
    start[1] = 0;
    end[1] = num_unconv;
    // 计算 A*x
    if (bpcg->MatDotMultiVec != NULL) {                             // 如果求解器带了自定义的MatDotMultiVec函数
        bpcg->MatDotMultiVec(mv_x, mv_r, start, end, mv_p, 0, ops); // TODO bpcg->MatDotMultiVec怎么定义的?可能和预条件有关
    } else {                                                        // 否则调用默认的MatDotMultiVec函数，y = Ax, 结果存储在mv_r中
        ops->MatDotMultiVec(mat, mv_x, mv_r, start, end, ops);
    }

#if TIME_BPCG
    time_bpcg.matvec_time += ops->GetWtime();
#endif

#if TIME_BPCG
    time_bpcg.axpby_time -= ops->GetWtime();
#endif
    start[0] = start_bx[0];
    end[0] = end_bx[0];
    start[1] = 0;
    end[1] = num_unconv;
    // 对于未受敛的向量，计算r = b - A*x.这里mv_r存的是上面调用默认的MatDotMultiVec函数记算出来的A*x
    ops->MultiVecAxpby(1.0, mv_b, -1.0, mv_r, start, end, ops);
#if TIME_BPCG
    time_bpcg.axpby_time += ops->GetWtime();
#endif

#if TIME_BPCG
    time_bpcg.innerprod_time -= ops->GetWtime();
#endif
    /*计算初始 ρ = r^T r 并检查是否满足收敛条件*/
    start[0] = 0;
    end[0] = num_unconv;
    start[1] = 0;
    end[1] = num_unconv;
    // 计算残差r的范数，存到rho2中
    ops->MultiVecInnerProd('D', mv_r, mv_r, 0, start, end, rho2, 1, ops);
#if TIME_BPCG
    time_bpcg.innerprod_time += ops->GetWtime();
#endif
    for (idx = 0; idx < num_unconv; ++idx) {
        init_res[idx] = sqrt(rho2[idx]); // 逐个计算sqrt(r^T*r)
    }
    /* �ж������� */
    pre_num_unconv = num_unconv;
    num_unconv = 0;
    for (idx = 0; idx < pre_num_unconv; ++idx) {
        if (init_res[idx] > tol * norm_b[idx]) { // 逐个判断是否有sqrt(r^T*r) > tol * ||b||,tol为容差限度
            unconv[num_unconv] = idx;            // 如果满足则说明未受敛，将未受敛向量索引存入unconv.
            rho2[num_unconv] = rho2[idx];
            ++num_unconv;
        } // 最终num_unconv更新为此次检查后未受敛的向量个数
    }
#if DEBUG
    if (num_unconv > 0) {
        ops->Printf("BlockPCG: initial residual[%d] = %6.4e\n", unconv[0], init_res[unconv[0]] / norm_b[unconv[0]]);
    } else {
        ops->Printf("BlockPCG: initial residual[%d] = %6.4e\n", 0, init_res[0] / norm_b[0]);
    }
#endif
    niter = 0;
    /*主循环*/
    while (niter < max_iter && num_unconv > 0) { // 循环之迭代次数达到上限/所有向量都收敛
                                                 /*把未知相邻的未受敛向量分到同一个块*/
        num_block = 0;
        block[num_block] = 0;
        ++num_block;
        for (idx = 1; idx < num_unconv; ++idx) {
            if (unconv[idx] - unconv[idx - 1] > 1) { // 如果未受敛的两个向量不是相邻的
                block[num_block] = idx;              // 就进行分块，将这个idx作为下一个块的起始位置存入block数组中
                ++num_block;
            }
        } // 最终num_block表示块的数量
        block[num_block] = num_unconv; // 将最后一个块的结束位置存入block数组中
        /* for each block */
        destin = pTw;
        for (idx = 0; idx < num_block; ++idx) {   // 对每一个块
            length = block[idx + 1] - block[idx]; // 当前处理的块有多少个向量
            for (col = block[idx]; col < block[idx + 1]; ++col) {
                /*更新搜索方向 p*/
                if (niter == 0)
                    beta = 0.0;
                else
                    beta = rho2[col] / rho1[col];

#if TIME_BPCG
                time_bpcg.axpby_time -= ops->GetWtime();
#endif
                start[0] = unconv[col];
                end[0] = unconv[col] + 1;
                start[1] = unconv[col];
                end[1] = unconv[col] + 1;
                // 如果是第一次迭代，则将搜索方向初始化为r;否则，设置为r+rho2[col] / rho1[col]*p
                ops->MultiVecAxpby(1.0, mv_r, beta, mv_p, start, end, ops);
#if TIME_BPCG
                time_bpcg.axpby_time += ops->GetWtime();
#endif
            }
#if TIME_BPCG
            time_bpcg.matvec_time -= ops->GetWtime();
#endif
            //compute the vector w = A*p
            /*计算w = A*p*/
            start[0] = unconv[block[idx]];
            end[0] = unconv[block[idx + 1] - 1] + 1;
            start[1] = unconv[block[idx]];
            end[1] = unconv[block[idx + 1] - 1] + 1;
            if (bpcg->MatDotMultiVec != NULL) {
                bpcg->MatDotMultiVec(mv_p, mv_w, start, end, mv_b, start_bx[0], ops);
            } else {
                ops->MatDotMultiVec(mat, mv_p, mv_w, start, end, ops);
            }

#if TIME_BPCG
            time_bpcg.matvec_time += ops->GetWtime();
#endif

#if TIME_BPCG
            time_bpcg.innerprod_time -= ops->GetWtime();
#endif
            //compute the value pTw = p^T * w
            // 计算pTw = p^T * w=p^T *A*p，存在destin指向的地址
            ops->MultiVecLocalInnerProd('D', mv_p, mv_w, 0, start, end, destin, 1, ops);
#if TIME_BPCG
            time_bpcg.innerprod_time += ops->GetWtime();
#endif
            destin += length; // destin指向下一个块对应的地址
        }

#if OPS_USE_MPI
#if TIME_BPCG
        time_bpcg.allreduce_time -= ops->GetWtime();
#endif
        // 确保每个进程中 pTw 数组的各元素都包含了所有进程对应元素的总和
        MPI_Allreduce(MPI_IN_PLACE, pTw, num_unconv, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#if TIME_BPCG
        time_bpcg.allreduce_time += ops->GetWtime();
#endif
#endif
        /* 计算alpha = rho / (p^T * A * p)*/
        //set rho1 as rho2
        int inc = 1;
        // 把rho2赋值给rho1
        dcopy(&num_unconv, rho2, &inc, rho1, &inc);
        /* for each block */
        destin = rho2;
        for (idx = 0; idx < num_block; ++idx) {
            length = block[idx + 1] - block[idx];
            for (col = block[idx]; col < block[idx + 1]; ++col) {
                //compute the value of alpha
                alpha = rho2[col] / pTw[col];
#if TIME_BPCG
                time_bpcg.axpby_time -= ops->GetWtime();
#endif
                //compute the new solution x = alpha * p + x
                /*迭代新的解 x = x + alpha * p */
                start[0] = unconv[col];
                end[0] = unconv[col] + 1;
                start[1] = start_bx[1] + unconv[col];
                end[1] = start_bx[1] + unconv[col] + 1;
                ops->MultiVecAxpby(alpha, mv_p, 1.0, mv_x, start, end, ops);
                //compute the new residual: r = - alpha*w + r
                start[0] = unconv[col];
                end[0] = unconv[col] + 1;
                start[1] = unconv[col];
                end[1] = unconv[col] + 1;
                // 计算新的残差: r = - alpha*A*p + r
                // 原来的r = b - A*x,更新之后只用继续减去alpha*A*p
                ops->MultiVecAxpby(-alpha, mv_w, 1.0, mv_r, start, end, ops);
#if TIME_BPCG
                time_bpcg.axpby_time += ops->GetWtime();
#endif
            }
#if TIME_BPCG
            time_bpcg.innerprod_time -= ops->GetWtime();
#endif
            //compute the new rho2
            // 对每个块计算新的rho2
            start[0] = unconv[block[idx]];
            end[0] = unconv[block[idx + 1] - 1] + 1;
            start[1] = unconv[block[idx]];
            end[1] = unconv[block[idx + 1] - 1] + 1;
            // rho2 = r^T * r。此时destin指向的是rho2数组的存储地址
            ops->MultiVecLocalInnerProd('D', mv_r, mv_r, 0, start, end, destin, 1, ops);
#if TIME_BPCG
            time_bpcg.innerprod_time += ops->GetWtime();
#endif
            destin += length;
        }
#if OPS_USE_MPI
#if TIME_BPCG
        time_bpcg.allreduce_time -= ops->GetWtime();
#endif
        MPI_Allreduce(MPI_IN_PLACE, rho2, num_unconv, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#if TIME_BPCG
        time_bpcg.allreduce_time += ops->GetWtime();
#endif
#endif
        for (idx = 0; idx < num_unconv; ++idx) {
            //if (bpcg->tol_type=='U' && niter > 10) {
            //last_res[unconv[idx]] = (1.1*last_res[unconv[idx]])<sqrt(rho2[idx])?1e-16:sqrt(rho2[idx]);
            //}
            //else {
            // 计算新的残差范数 ||r|| = sqrt(r^T * r)
            last_res[unconv[idx]] = sqrt(rho2[idx]);
            //}
        }
#if DEBUG
        ops->Printf("niter = %d, num_unconv = %d, residual[%d] = %6.4e\n",
                    niter + 1, num_unconv, unconv[0], last_res[unconv[0]] / norm_b[unconv[0]]);
#endif
        /* 判断收敛性 */
        pre_num_unconv = num_unconv;
        num_unconv = 0;
        for (idx = 0; idx < pre_num_unconv; ++idx) {
            col = unconv[idx];
            // 判断是否满足终止条件
            // 需要满足两个条件：1.残差小于rate*初始残差，2.残差小于tol*||b||
            // TODO:为什么需要满足第一个条件？
            if ((last_res[col] > rate * init_res[col]) && (last_res[col] > tol * norm_b[col])) {
                unconv[num_unconv] = col;
                /* 需将 rho1 rho2 未收敛部分顺序前移 */
                rho1[num_unconv] = rho1[idx];
                rho2[num_unconv] = rho2[idx];
                ++num_unconv;
            } // 更新未受敛的向量个数
        }
        //update the iteration time

        ++niter;

#if DEBUG
        if (niter % 5 == 0) {
            ops->Printf("BlockPCG: niter = %d, num_unconv = %d, residual[%d] = %6.4e\n",
                        niter, num_unconv, unconv[0], last_res[unconv[0]] / norm_b[unconv[0]]);
        }
#endif
    }
    // 存储最终迭代次数和残差
    if (niter > 0) {
        bpcg->niter = niter;
        bpcg->residual = last_res[unconv[0]];
    } else {
        bpcg->niter = niter;
        bpcg->residual = init_res[0];
    }

#if TIME_BPCG
    ops->Printf("|--BPCG----------------------------\n");
    time_bpcg.time_total = time_bpcg.allreduce_time + time_bpcg.axpby_time + time_bpcg.innerprod_time + time_bpcg.matvec_time;
    ops->Printf("|allreduce  axpby  inner_prod  matvec\n");
    ops->Printf("|%.2f\t%.2f\t%.2f\t%.2f\n",
                time_bpcg.allreduce_time,
                time_bpcg.axpby_time,
                time_bpcg.innerprod_time,
                time_bpcg.matvec_time);
    ops->Printf("|%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n",
                time_bpcg.allreduce_time / time_bpcg.time_total * 100,
                time_bpcg.axpby_time / time_bpcg.time_total * 100,
                time_bpcg.innerprod_time / time_bpcg.time_total * 100,
                time_bpcg.matvec_time / time_bpcg.time_total * 100);
    ops->Printf("|--BPCG----------------------------\n");
    time_bpcg.allreduce_time = 0.0;
    time_bpcg.axpby_time = 0.0;
    time_bpcg.innerprod_time = 0.0;
    time_bpcg.matvec_time = 0.0;
#endif

    return;
}

/**
 * @brief 设置块预条件共轭梯度法求解器的相关参数
 * 
 * @param max_iter 求解器的最大迭代次数
 * @param rate 求解器的收敛速率
 * @param tol 收敛容差
 * @param tol_type 收敛容差的判断方式(abs or rel)
 * @param mv_ws 提供矩阵向量乘法的工作空间(临时空间)
 * @param dbl_ws 提供双精度浮点数的工作空间(临时空间)
 * @param int_ws 提供整数类型的工作空间(临时空间)
 * @param pc 预条件子（Preconditioner）的指针
 * @param MatDotMultiVec 矩阵向量乘法函数的指针
 * @param ops 操作集合
 */
void MultiLinearSolverSetup_BlockPCG(int max_iter, double rate, double tol,
                                     const char *tol_type, void **mv_ws[3], double *dbl_ws, int *int_ws,
                                     void *pc, void (*MatDotMultiVec)(void **x, void **y, int *, int *, void **z, int s, struct OPS_ *),
                                     struct OPS_ *ops) {
    /* ֻ��ʼ��һ�Σ���ȫ�ֿɼ� */
    static BlockPCGSolver bpcg_static = {
        .max_iter = 50,
        .rate = 1e-2,
        .tol = 1e-12,
        .tol_type = "abs",
        .mv_ws = {},
        .pc = NULL,
        .MatDotMultiVec = NULL};
    bpcg_static.max_iter = max_iter;
    bpcg_static.rate = rate;
    bpcg_static.tol = tol;
    strcpy(bpcg_static.tol_type, tol_type);
    bpcg_static.mv_ws[0] = mv_ws[0];
    bpcg_static.mv_ws[1] = mv_ws[1];
    bpcg_static.mv_ws[2] = mv_ws[2];
    bpcg_static.dbl_ws = dbl_ws;
    bpcg_static.int_ws = int_ws;
    bpcg_static.MatDotMultiVec = MatDotMultiVec;

    bpcg_static.niter = 0;
    bpcg_static.residual = -1.0;

    ops->multi_linear_solver_workspace = (void *)(&bpcg_static);
    ops->MultiLinearSolver = BlockPCG;
    return;
}
static void BlockAlgebraicMultiGrid(int current_level,
                                    void **mv_b, void **mv_x, int *start_bx, int *end_bx, struct OPS_ *ops) {
#if DEBUG
    ops->Printf("current level = %d\n", current_level);
#endif
    BlockAMGSolver *bamg = (BlockAMGSolver *)ops->multi_linear_solver_workspace;
    void (*multi_linear_sol)(void *, void **, void **, int *, int *, struct OPS_ *);
    multi_linear_sol = ops->MultiLinearSolver;

    assert(end_bx[0] - start_bx[0] == end_bx[1] - start_bx[1]);
    int coarsest_level = bamg->num_levels - 1, coarse_level;
    int start[2], end[2], block_size = end_bx[1] - start_bx[1];
    void *A = bamg->A_array[current_level];
    void **mv_ws[3], **mv_r, **coarse_b, **coarse_x;
    mv_ws[0] = bamg->mv_array_ws[2][current_level];
    mv_ws[1] = bamg->mv_array_ws[3][current_level];
    mv_ws[2] = bamg->mv_array_ws[4][current_level];
    /* --------------------------------------------------------------- */
    MultiLinearSolverSetup_BlockPCG(
        bamg->max_iter[current_level * 2 + 1], bamg->rate[current_level],
        bamg->tol[current_level], bamg->tol_type,
        mv_ws, bamg->dbl_ws, bamg->int_ws, NULL, NULL, ops);
#if DEBUG
    int idx;
    for (idx = 0; idx <= current_level; ++idx) ops->Printf("--");
    ops->Printf("level = %d, pre-smooth\n", current_level);
#endif
#if DEBUG
    ops->Printf("--initi-solve------------------\n");
    ops->MultiVecView(mv_x, start_bx[1], end_bx[1], ops);
#endif

#if TIME_BAMG
    time_bamg.bpcg_time -= ops->GetWtime();
#endif
    ops->MultiLinearSolver(A, mv_b, mv_x, start_bx, end_bx, ops);
#if TIME_BAMG
    time_bamg.bpcg_time += ops->GetWtime();
#endif

#if DEBUG
    ops->Printf("--after-solve------------------\n");
    ops->MultiVecView(mv_x, start_bx[1], end_bx[1], ops);
#endif
    if (current_level < coarsest_level) {
#if TIME_BAMG
        time_bamg.matvec_time -= ops->GetWtime();
#endif
        //����residual = b - A*x
        start[0] = start_bx[1];
        end[0] = end_bx[1];
        start[1] = 0;
        end[1] = block_size;
        mv_r = bamg->mv_array_ws[2][current_level];
        ops->MatDotMultiVec(A, mv_x, mv_r, start, end, ops);
#if TIME_BAMG
        time_bamg.matvec_time += ops->GetWtime();
#endif

#if TIME_BAMG
        time_bamg.axpby_time -= ops->GetWtime();
#endif
        start[0] = start_bx[0];
        end[0] = end_bx[0];
        start[1] = 0;
        end[1] = block_size;
        ops->MultiVecAxpby(1.0, mv_b, -1.0, mv_r, start, end, ops);
#if TIME_BAMG
        time_bamg.axpby_time += ops->GetWtime();
#endif

#if TIME_BAMG
        time_bamg.fromitoj_time -= ops->GetWtime();
#endif
        // ��residualͶӰ��������
        coarse_level = current_level + 1;
        coarse_b = bamg->mv_array_ws[0][coarse_level];
        coarse_x = bamg->mv_array_ws[1][coarse_level];
        start[0] = 0;
        end[0] = block_size;
        start[1] = 0;
        end[1] = block_size;
        ops->MultiVecFromItoJ(bamg->P_array, current_level, coarse_level,
                              mv_r, coarse_b, start, end, bamg->mv_array_ws[4], ops);
#if TIME_BAMG
        time_bamg.fromitoj_time += ops->GetWtime();
#endif

#if DEBUG
        ops->Printf("---mv r-----\n");
        ops->MultiVecView(mv_r, 0, block_size, ops);
#endif
#if DEBUG
        ops->Printf("---coarse b-----\n");
        ops->MultiVecView(coarse_b, 0, block_size, ops);
#endif

#if TIME_BAMG
        time_bamg.axpby_time -= ops->GetWtime();
#endif
        // �ȸ�coarse_x����ֵ0
        ops->MultiVecAxpby(0.0, NULL, 0.0, coarse_x, start, end, ops);
#if TIME_BAMG
        time_bamg.axpby_time += ops->GetWtime();
#endif

        // �����������⣬���õݹ�
        ops->multi_linear_solver_workspace = (void *)bamg;
        BlockAlgebraicMultiGrid(coarse_level, coarse_b, coarse_x, start, end, ops);

#if TIME_BAMG
        time_bamg.fromitoj_time -= ops->GetWtime();
#endif
        // �Ѵ������ϵĽ��ֵ��ϸ�����ټӵ�ǰ�⻬�õ��Ľ��ƽ���
        ops->MultiVecFromItoJ(bamg->P_array, coarse_level, current_level,
                              coarse_x, mv_r, start, end, bamg->mv_array_ws[4], ops);
#if TIME_BAMG
        time_bamg.fromitoj_time += ops->GetWtime();
#endif

#if DEBUG
        ops->Printf("---after FromItoJ-----\n");
        ops->MultiVecView(mv_r, start_bx[1], end_bx[1], ops);
#endif

        // У�� x = x+residual
        start[0] = 0;
        end[0] = block_size;
        start[1] = start_bx[1];
        end[1] = end_bx[1];
#if DEBUG
        ops->Printf("---before x = x+residual-----\n");
        ops->MultiVecView(mv_x, start_bx[1], end_bx[1], ops);
#endif

#if TIME_BAMG
        time_bamg.axpby_time -= ops->GetWtime();
#endif
        ops->MultiVecAxpby(1.0, mv_r, 1.0, mv_x, start, end, ops);
#if TIME_BAMG
        time_bamg.axpby_time += ops->GetWtime();
#endif

#if DEBUG
        ops->Printf("---after x = x+residual------\n");
        ops->MultiVecView(mv_x, start_bx[1], end_bx[1], ops);
#endif
        // ��⻬
        MultiLinearSolverSetup_BlockPCG(
            bamg->max_iter[current_level * 2 + 2], bamg->rate[current_level],
            bamg->tol[current_level], bamg->tol_type,
            mv_ws, bamg->dbl_ws, bamg->int_ws, NULL, NULL, ops);
#if DEBUG
        ops->Printf("---initi solver ------------\n");
        ops->MultiVecView(mv_x, start_bx[1], end_bx[1], ops);
#endif
#if DEBUG
        for (idx = 0; idx <= current_level; ++idx) ops->Printf("--");
        ops->Printf("level = %d, post-smooth\n", current_level);
#endif

#if TIME_BAMG
        time_bamg.bpcg_time -= ops->GetWtime();
#endif
        ops->MultiLinearSolver(A, mv_b, mv_x, start_bx, end_bx, ops);
#if TIME_BAMG
        time_bamg.bpcg_time += ops->GetWtime();
#endif

#if DEBUG
        ops->Printf("---after solver ------------\n");
        ops->MultiVecView(mv_x, start_bx[1], end_bx[1], ops);
#endif
    }
    bamg->residual = ((BlockPCGSolver *)ops->multi_linear_solver_workspace)->residual;
    /* �����Խⷨ������Ϊ BlockAMG */
    ops->multi_linear_solver_workspace = (void *)bamg;
    ops->MultiLinearSolver = multi_linear_sol;
    return;
}
static void BlockAMG(void *mat, void **mv_b, void **mv_x,
                     int *start_bx, int *end_bx, struct OPS_ *ops) {
#if TIME_BAMG
    time_bamg.axpby_time = 0.0;
    time_bamg.bpcg_time = 0.0;
    time_bamg.fromitoj_time = 0.0;
    time_bamg.matvec_time = 0.0;
#endif
    int idx;
    BlockAMGSolver *bamg = (BlockAMGSolver *)ops->multi_linear_solver_workspace;
    for (idx = 0; idx < bamg->max_iter[0]; ++idx) {
        BlockAlgebraicMultiGrid(0, mv_b, mv_x, start_bx, end_bx, ops);
#if DEBUG
        ops->Printf("BlockAMG: niter = %d, residual = %6.4e\n", idx + 1, bamg->residual);
#endif
        if (bamg->residual < bamg->tol[0]) break;
    }
#if TIME_BAMG
    ops->Printf("|--BAMG----------------------------\n");
    time_bamg.time_total = time_bamg.axpby_time + time_bamg.bpcg_time + time_bamg.fromitoj_time + time_bamg.matvec_time;
    ops->Printf("|axpby  bpcg  fromitoj  matvec\n");
    ops->Printf("|%.2f\t%.2f\t%.2f\t%.2f\n",
                time_bamg.axpby_time,
                time_bamg.bpcg_time,
                time_bamg.fromitoj_time,
                time_bamg.matvec_time);
    ops->Printf("|%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n",
                time_bamg.axpby_time / time_bamg.time_total * 100,
                time_bamg.bpcg_time / time_bamg.time_total * 100,
                time_bamg.fromitoj_time / time_bamg.time_total * 100,
                time_bamg.matvec_time / time_bamg.time_total * 100);
    ops->Printf("|--BAMG----------------------------\n");
    time_bamg.axpby_time = 0.0;
    time_bamg.bpcg_time = 0.0;
    time_bamg.fromitoj_time = 0.0;
    time_bamg.matvec_time = 0.0;
#endif
    return;
}
void MultiLinearSolverSetup_BlockAMG(int *max_iter, double *rate, double *tol,
                                     const char *tol_type, void **A_array, void **P_array, int num_levels,
                                     void ***mv_array_ws[5], double *dbl_ws, int *int_ws,
                                     void *pc, struct OPS_ *ops) {
    /* ֻ��ʼ��һ�Σ���ȫ�ֿɼ� */
    static BlockAMGSolver bamg_static = {
        .max_iter = NULL,
        .rate = NULL,
        .tol = NULL,
        .tol_type = "abs",
        .mv_array_ws = {},
        .dbl_ws = NULL,
        .int_ws = NULL,
        .pc = NULL};
    bamg_static.max_iter = max_iter;
    bamg_static.rate = rate;
    bamg_static.tol = tol;
    strcpy(bamg_static.tol_type, tol_type);
    bamg_static.A_array = A_array;
    bamg_static.P_array = P_array;
    bamg_static.num_levels = num_levels;
    bamg_static.mv_array_ws[0] = mv_array_ws[0];
    bamg_static.mv_array_ws[1] = mv_array_ws[1];
    bamg_static.mv_array_ws[2] = mv_array_ws[2];
    bamg_static.mv_array_ws[3] = mv_array_ws[3];
    bamg_static.mv_array_ws[4] = mv_array_ws[4];
    bamg_static.dbl_ws = dbl_ws;
    bamg_static.int_ws = int_ws;
    bamg_static.niter = 0;
    bamg_static.residual = -1.0;

    ops->multi_linear_solver_workspace = (void *)(&bamg_static);
    ops->MultiLinearSolver = BlockAMG;
    return;
}
