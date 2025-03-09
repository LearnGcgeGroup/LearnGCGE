/**
 * Chinese Encoding Format: UTF-8
 * 
 * Updated on 2025-03-07 by 吴卓轩
 * 
 * 本代码实现了一个用于矩阵正交化的库，主要包含了两种正交化方法：
 * 1. 分块的改进GS正交化：Modified Gram-Schmidt (MGS)
 * 2. EVD-二分递归正交化：Binary Gram-Schmidt (BGS)
*/

/**
 * 本代码直接用到的多向量函数：
 * @param MultiVecQtAP
 * @param MultiVecAxpby
 * @param MultiVecLinearComb
 */

#include	<stdio.h>
#include	<stdlib.h>
#include    <assert.h> 
#include	<float.h>	// 调用 DBL_EPSILON
#include    <math.h>
#include    <time.h>

#include    "ops_orth.h"

#define  DEBUG 0
#define  TIME_MGS 0
#define  TIME_BGS 0

typedef struct TimeMGS_ {
	double axpby_time;
	double line_comb_time;
	double orth_self_time;
    double qAp_time;
	double time_total;
} TimeMGS;

typedef struct TimeBGS_ {
	double axpby_time;
	double line_comb_time;
	double orth_self_time;
    double qAp_time;
	double time_total;
} TimeBGS;

struct TimeMGS_ time_mgs = {0.0,0.0,0.0,0.0,0.0};
struct TimeBGS_ time_bgs = {0.0,0.0,0.0,0.0,0.0};

/**
 * @brief 这是一个核心子函数，使用 Modified Gram-Schmidt 正交化小块矩阵
 * 
 * 对一组向量进行规范正交化处理，确保它们彼此正交且均为单位向量。
 * 正交化方法是 MGS. 
 * 使用重正交化来提高数值稳定性。
 * 
 * @param x 			待正交化的向量族。
 * @param start_x 		从x中的第start_x个向量开始进行正交化处理（含）。
 * @param end_x 		待正交化的最后一个向量的索引（不含）。
 * @param B 			用于定义B-内积。B = NULL时使用标准内积。
 * 
 * @param max_reorth 	表示最大重正交化次数（实际上，重正交化次数为(max_reorth - 2)）。
 * @param orth_zero_tol 零向量容差。如果r_k小于该容差，则认为r_k = 0。用于判别内积是否达到0. 
 * @param reorth_tol 	重正交化容差。如果重正交化时的内积矩阵的绝对最大值小于该容差，则停止重正交化。
 * 
 * @param mv_ws 		多向量工作空间。
 * @param dbl_ws 		双精度工作空间。
 * @param ops 			包含各种操作函数的实现。
 */
static void OrthSelf(void **x, int start_x, int *end_x, void *B, int max_reorth, double orth_zero_tol, double reorth_tol, void **mv_ws, double *dbl_ws, struct OPS_ *ops)
{
	/** 
	 * Q: 为什么 end_x 是 int* 类型？
	 * A: 可能某些函数需要将 end_x 作为传出参数，并考虑到封装函数的一致性，仅是目前未见具体应用。
	 */
	
	if (*end_x<=start_x) return;
	
	// 准备一些仅函数内使用的临时变量
	int    k, start[2], end[2], length, inc, idx, idx_abs_max; 
	double *r_k, *beta, *coef;
	r_k  = dbl_ws;   // r_k 用于计算第k列与k~end列向量的B-内积，这里r_k[0]为第k列的自内积，r_k[k+1:end]为k列与k+1:end列的互内积。
	beta = dbl_ws;   // beta 与 r_k 共享同一个缓存地址，事实上，当真正启用beta时，r_k[0]不再需要。此举仅为节省空间。
	coef = dbl_ws+1; // coef 取为 r_k+1 的地址，可以存取 r_k[k+1:end] 的量。此举也为节省空间。

	// 进入对待处理矩阵的按列循环
	for (k = start_x; k < (*end_x); ++k) {
		// 准备 计算 x 的 第k~end列[k,end] 和 第k列[k,k+1) 的 B-内积矩阵 所需指标
		start[0] = k; end[0] = *end_x;
		start[1] = k; end[1] = k+1;

		// 计算部分的x'Bx，并储存到r_k. r_k应为一个 (*end_x-k) x 1 大小的 B-内积矩阵。
		ops->MultiVecQtAP('S','N',x,B,x,0,start,end,r_k,end[0]-start[0],mv_ws,ops); //'S'表示B是对称的，'N'表示不做额外的转置。
		/** 
		 * Q: 为什么r_k可能不是向量，却是 double* 类型？
		 * A: 用列优先一维数组储存矩阵，行数为ldQAP = end[0]-start[0]，调用(i,j)元使用r_k[i + j * ldQAP]. 
		 */

		// 为r_k的第一个元素*r_k（x的第k列与自身的B-内积）开根号，使得r_k的第一个元素成为 x的第k列 的 B-范数。
		*r_k = sqrt(*r_k);

		// 如果x的第k列的范数等于0，即这个向量经过投影到前面向量的正交补后为0，说明它可以被前面的向量线性表出，该向量不能张出一维空间，需要进行剔除。
		if (*r_k < orth_zero_tol) {
			ops->Printf("r_[%d] = %6.4e\n",k,*r_k);
			// 如果第k列不是最后一列，需要将后面的向量补位到k处。
			if (k < *end_x-1) {
				start[0] = *end_x-1; end[0] = *end_x; // *end_x 实际上无用
				start[1] = k       ; end[1] = k+1   ;
				// 将 x 的最后一列赋值给 x 的第 k 列
				ops->MultiVecAxpby(1.0,x,0.0,x,start,end,ops); // 这里很巧妙地运用axpby函数实现了赋值的效果
			}
			--k; --(*end_x); // 回退列指标 k ，并缩减需要计算的列的范围。
			// 如果第k列已经是最后一列，则在--(*end_x)后返回循环自然会直接退出。
			continue;
		}
		// 如果x的第k列的范数非零，说明该向量可以正交于前面的向量并张出一维空间。
		else {
			// 将 x 的第k列归一化（除以自身范数*r_k）
			start[0] = k; end[0] = k+1;
			start[1] = k; end[1] = k+1;
			*r_k = 1.0/(*r_k);
			ops->MultiVecAxpby(0.0,NULL,*r_k,x,start,end,ops);
		}

		// 如果第k个向量不是最后一个向量，则将后续向量投影到该第k个向量的正交补空间中。
		// 公式：X[k+1:end] ←-- X[k+1:end] - X[k]X[k]'BX[k+1:end]）
		// 简记：X ←-- X - xx'BX
		// 这里还涉及到向量归一化的问题，虽然X[k]已经被归一化，但之前计算的r_k[k+1:end]仍然是为归一化时得到的内积矩阵。
		// 在下面的注释中，用X_0[k]表示归一化前的第k列向量。
		if (k<*end_x-1) {
			length = *end_x-(k+1); // length：需要被投影的向量数，即X[k+1:end]包含的向量的数量。
			// 用dscal求 coef = X[k]'BX[k+1:end] = -X_0[k]'BX[k+1:end] / norm(X_0[k])
			*r_k *= -1.0; inc = 1; // 现在的*r_k是"X_0[k]的范数的负倒数"
			dscal(&length,r_k,coef,&inc); // 为 X_0[k]'BX[k+1:end] 数乘"X_0[k]的范数的负倒数"，实则是求
			/**
			 * Q: 为什么似乎没有初始化coef？
			 * A: coef事实上被初始化为r_k+1，即r_k[k+1:end]的值。
			 * Q: 为什么r_k[>=1]会发生改变？虽然后续不再用到这些量。
			 * A: dscal函数本身只读取r_k[0]的值，不改变任何r_k的值，但会改变coef，其地址恰为r_k+1. 
			 * Q: 为什么用不到r_k[>=1]，之前不只计算r_k[0]？
			 * A: 其实用到了，只是以coef变量的形态用的。
			 */
			// 计算：X[k+1:end] <-- X[k+1:end] - X[k] * X_0[k]'BX[k+1:end] / norm(X_0[k])
			// 即 X[k+1:end] <-- X[k+1:end] - X[k] * X[k]'BX[k+1:end]
			*beta = 1.0;
			start[0] = k  ; end[0] = k+1   ;
			start[1] = k+1; end[1] = *end_x;
			ops->MultiVecLinearComb(x,x,0,start,end,coef,end[0]-start[0],beta,0,ops);
			/**
			 * Q: 为什么不直接用未归一化的X计算，非要先对[k]归一化，再对[k+1:end]归一化？
			 * A: 首先要保证结果是规范化的，在此基础上为了减少时间成本，故选择如此流程。
			 */

			// 理论上已经完成投影，但为了数值稳定性，这里进行重正交化
			for (idx = 1; idx < -1+max_reorth; ++idx) { // 重正交化的次数不超过 (max_reorth - 2) 次。
				// 计算 X[k+1:esnd] 与 X[k] 的内积矩阵，储存至coef中，coef是 length x 1 的矩阵。
				start[0] = k+1; end[0] = *end_x;
				start[1] = k  ; end[1] = k+1   ;
				ops->MultiVecQtAP('S','N',x,B,x,0,start,end,coef,end[0]-start[0],mv_ws,ops);
				// 计算 coef = -coef
				length = (*end_x-k-1); // 理论上和之前计算出来的 length 应该一样，本行代码似乎冗余。
				*beta = -1.0; inc = 1;
				dscal(&length,beta,coef,&inc);
				// 这里coef的行数end[0]-start[0]被更改为1了，虽然coef仍然是矩阵的一维表示法，但此时相当于取其转置。
				// 计算 X[k+1:end] <-- X[k+1:end] + X[k] * coef'
				// 即 X[k+1:end] <-- X[k+1:end] - X[k] * X[k]' B X[k+1:esnd]
				*beta = 1.0;
				start[0] = k  ; end[0] = k+1   ;
				start[1] = k+1; end[1] = *end_x;
				ops->MultiVecLinearComb(x,x,0,start,end,coef,end[0]-start[0],beta,0,ops);			
				// 计算coef的绝对最大值是否小于重正交容差reorth_tol，如果足够小，则退出重正交化。
				idx_abs_max = idamax(&length,coef,&inc);
				if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
				   break;
				}
			}			
		}
	}
	return;
}

/**
 * @brief 这是一个核心子函数，使用基于 EigenValue Decomposition 的矩阵开方法正交化小块矩阵
 * 
 * 对一组向量进行规范正交化处理，确保它们彼此正交且均为单位向量。
 * 正交化方法是 EVD-矩阵开方法。
 * 使用重正交化来提高数值稳定性。
 * 
 * @param x 			待正交化的向量族。
 * @param start_x 		从x中的第start_x个向量开始进行正交化处理（含）。
 * @param end_x 		待正交化的最后一个向量的索引（不含）。
 * @param B 			用于定义B-内积。B = NULL时使用标准内积。
 * 
 * @param max_reorth 	表示最大重正交化次数（实际上，重正交化次数为(max_reorth + 1)）。
 * @param orth_zero_tol 零向量容差。如果r_k小于该容差，则认为r_k = 0。用于判别内积是否达到0. 
 * @param reorth_tol 	重正交化容差。如果重正交化时的内积矩阵的绝对最大值小于该容差，则停止重正交化。
 * 
 * @param mv_ws 		多向量工作空间。
 * @param dbl_ws 		双精度工作空间。
 * @param ops 			包含各种操作函数的实现。
 */
static void OrthSelfEVP(void **x, int start_x, int *end_x, void *B, int max_reorth, double orth_zero_tol, double reorth_tol, void **mv_ws, double *dbl_ws, struct OPS_ *ops)
{
	if (*end_x<=start_x) return;

	// 准备一些仅函数内使用的临时变量
	int    k, start[2], end[2], idx, inc = 1, lin_dep; 
	char   JOBZ = 'V', UPLO = 'L';
	int    N, LDA, LWORK, INFO;
	double *A, *W, *WORK;

	// 为了数值稳定性，进行重正交化（此处较原代码做了顺序上的微调，以便理解）
	for (idx = 0; idx < 1+max_reorth; ++idx) { // 重正交化的次数不超过 (max_reorth + 1) 次。
		if (*end_x==start_x) return;
		assert(*end_x-start_x>=1); // 确保要处理的矩阵不空
		// 计算内积矩阵 A = x'Bx
		A = dbl_ws;
		start[0] = start_x; end[0] = *end_x;
		start[1] = start_x; end[1] = *end_x;
		ops->MultiVecQtAP('S','S',x,B,x,0,start,end,A,end[0]-start[0],mv_ws,ops);
		// 确保成功计算A的特征值分解 A diag{W} A' <-- A，其中W是升序排列的特征值构成的向量。
		N = *end_x-start_x; LDA = N  ; LWORK = 3*N*N-N; W = A+N*N; WORK  = W+N;
		dsyev(&JOBZ,&UPLO,&N,A,&LDA,W,WORK,&LWORK,&INFO);
		assert(INFO==0);

		lin_dep = 0; // 初始化线性相关向量数
		// 遍历每一个特征值
		for (k = 0; k < N; ++k) {
			assert(W[k] > -orth_zero_tol); // 确保特征值非负
			// 如果特征值为正，则计算其开方倒数（^-0.5）
			if (fabs(W[k]) > orth_zero_tol) {
				W[k] = 1.0/sqrt(W[k]);
			}
			// 如果特征值为0，则在lin_dep上记录一次
			else {
				++lin_dep;
			}
		}
		// 若存在线性相关向量，输出线性相关向量的数量信息
		if (lin_dep > 0) {
			ops->Printf("There has %d linear dependent vec\n",lin_dep);
		}

		// 由于特征值是升序排列的，故零特征值及其对应的特征向量应在最前方。
		// 剔除零特征值的特征向量：计算 mv_ws = x * A(:, lin_dep:end)
		start[0] = start_x  ; end[0] = *end_x;
		start[1] = 0        ; end[1] = N-lin_dep;
		ops->MultiVecLinearComb(x,mv_ws,0,start,end,A+LDA*lin_dep,end[0]-start[0],NULL,0,ops);
		// 剔除零特征值：mv_ws = mv_ws * diag{W(lin_dep:end)}
		// 即 mv_ws = x * A(:, lin_dep:end) * diag{lambda^(-0.5)(lin_dep:end)}
		ops->MultiVecLinearComb(NULL,mv_ws,0,start,end,NULL,0,W+lin_dep,1,ops);
		*end_x = *end_x - lin_dep; // 缩减需要计算的列的范围
		// 利用axpby函数进行赋值：x <-- mv_ws
		start[0] = 0        ; end[0] = N-lin_dep;
		start[1] = start_x  ; end[1] = *end_x;
		ops->MultiVecAxpby(1.0,mv_ws,0.0,x,start,end,ops);
		// 如果不再有线性相关项，且特征值之和足够接近N（理论上W应为全一向量），则停止重正交化。
		if (lin_dep==0&&fabs(dasum(&N,W,&inc)-N)<reorth_tol) {
			break;
		}	
	}
	return;
}


/**
 * @brief 这是一个主函数，使用分块 Modified Gram-Schmidt 方法正交化大矩阵
 * 
 * 对一组向量进行规范正交化处理，确保它们彼此正交且均为单位向量。
 * 正交化方法是 分块MGS方法。
 * 使用重正交化来提高数值稳定性。
 * 
 * @param x 			待正交化的向量族。
 * @param start_x 		从x中的第start_x个向量开始进行正交化处理（含）。
 * @param end_x 		待正交化的最后一个向量的索引（不含）。
 * @param B 			用于定义B-内积。B = NULL时使用标准内积。
 * 
 * @param ops 			包含各种操作函数的实现。
 */
static void ModifiedGramSchmidt(void **x, int start_x, int *end_x, void *B, struct OPS_ *ops)
{
	if (*end_x <= start_x) return;

	// 从 ops 中调取参数到 mgs_orth 中
	ModifiedGramSchmidtOrth *mgs_orth = (ModifiedGramSchmidtOrth*)ops->orth_workspace;
	int    start[2], end[2], length, block_size, idx, idx_abs_max;
	int    incx; 
	double *coef   , *beta , orth_zero_tol, reorth_tol;
	void   **mv_ws;
	orth_zero_tol = mgs_orth->orth_zero_tol;
	reorth_tol    = mgs_orth->reorth_tol;
	block_size    = mgs_orth->block_size;
	mv_ws         = mgs_orth->mv_ws ;
	beta          = mgs_orth->dbl_ws;		// beta 指向地址 dbl_ws
	start[0] = 0     ; end[0] = start_x;
	start[1] = end[0]; end[1] = *end_x ;
	coef     = beta+1;						// coef 指向地址 dbl_ws+1
	
	// 如果[0,start)非空，则认为[0,start)部分已经自正交，并把[start,end]部分投影到[0,start)的正交补中
	if (start_x > 0) {
		// 重正交化，提高数值稳定性
		for (idx = 0; idx < 1+mgs_orth->max_reorth; ++idx) { // 最大重正交化次数为 (mgs_orth->max_reorth + 1)	
			// 计算内积矩阵 coef = X[0:start)' B X[start:end]，得到 start x end+1-start 大小的矩阵
			ops->MultiVecQtAP('S','N',x,B,x,0,start,end,coef,end[0]-start[0],mv_ws,ops);
			// 计算 coef  <--  -coef
			length = (end[1] - start[1])*(end[0] - start[0]);
			*beta = -1.0; incx = 1;
			dscal(&length,beta,coef,&incx);
			// 计算 X[start:end]  <--  X[0:start) * coef + X[start:end]
			// 即 X[start:end]  <--  X[start:end] - X[0:start) X[0:start)' B X[start:end]
			*beta = 1.0;
			ops->MultiVecLinearComb(x,x,0,start,end,coef,end[0]-start[0],beta,0,ops);
			// 计算coef的绝对最大值是否小于重正交容差reorth_tol，如果足够小，则退出重正交化。
			idx_abs_max = idamax(&length,coef,&incx);
			if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
			   break;
			}
		}
	}
	
	
	int init_start = start_x, init_end; // 定义init_值为当前处理的块的start,end指标
	// 动态调整 block_size
	// 如果 block_size 非正，取默认值 max{2, 总向量数的一半}
	if (block_size <= 0) {
	   block_size = (*end_x-init_start)/2 > 2 ? (*end_x-init_start)/2 : 2;
	}
	// 确保 block_size 不超过总向量数。
	block_size = (block_size<*end_x-init_start)?block_size:(*end_x-init_start);
	// ops->Printf("start_x = %d, block_size = %d, end_x = %d\n",start_x,block_size,*end_x);
	// 当待处理向量组非空，进入循环
	while (block_size > 0) {
		// 规范正交化 X[init_start : init_start + block_size]，函数内含重正交化措施
		start[1] = init_start; end[1] = start[1]+block_size;
		OrthSelf(x,start[1],&(end[1]),B,mgs_orth->max_reorth,orth_zero_tol,reorth_tol,mv_ws,mgs_orth->dbl_ws,ops);	      
		// 为 init_end 赋值为 init_start + block_size
		init_end = end[1];
		
		// 处理剩余长度
		length = block_size - (end[1]-start[1]); // length 一定为 0 ？
		// ops->Printf("length = %d\n",length);
		length = (length<*end_x-end[1]-length)?length:(*end_x-end[1]-length);
		if (length > 0) {
			// 如果 length > 0，
			// 利用 axpby 赋值：X[init_end : init_end + length)  <--  X[*end - length : *end)
			end[0]   = *end_x; start[0] = end[0]-length;
			start[1] = init_end; end[1] = start[1]+length;
			ops->MultiVecAxpby(1.0,x,0.0,x,start,end,ops);
		}
		*end_x = *end_x - (block_size-(init_end-init_start));

		if ( init_end < (*end_x) && init_start < init_end ) {	
			for (idx = 0; idx < 1+mgs_orth->max_reorth; ++idx) {
				/* 锟斤拷锟斤拷锟斤拷锟斤拷时, 锟斤拷锟斤拷锟斤拷锟斤拷 Bx, 锟斤拷 mv_ws 锟斤拷锟斤拷, 锟斤拷锟斤拷, QtAP 锟斤拷 coef 锟斤拷为 'T', 锟斤拷锟斤拷锟斤拷要转锟斤拷 */
				if (B!=NULL && idx > 0) {
					start[0] = init_end  ; end[0] = *end_x;
					start[1] = 0         ; end[1] = init_end - init_start;		
					ops->MultiVecQtAP('S','T',x,NULL,mv_ws,0,start,end,
							coef,end[1]-start[1],mv_ws,ops);
				}
				else {
					start[0] = init_end  ; end[0] = *end_x  ;
					start[1] = init_start; end[1] = init_end;		
					ops->MultiVecQtAP('S','T',x,B,x,0,start,end,
							coef,end[1]-start[1],mv_ws,ops);
				}

				length = (end[1] - start[1])*(end[0] - start[0]); 
				*beta  = -1.0; incx = 1;
				dscal(&length,beta,coef,&incx);
				*beta  = 1.0;

				start[0] = init_start; end[0] = init_end;
				start[1] = init_end  ; end[1] = *end_x  ;
				ops->MultiVecLinearComb(x,x,0,start,end,
						coef,end[0]-start[0],beta,0,ops);
				incx = 1;
				idx_abs_max = idamax(&length,coef,&incx);
				if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
				   break;
				}
			}
		}
		init_start = init_end;
		block_size = (block_size<*end_x-init_start)?block_size:(*end_x-init_start);		
	}
	return;
}

void MultiVecOrthSetup_ModifiedGramSchmidt(
		int block_size, int max_reorth, double orth_zero_tol, 
		void **mv_ws, double *dbl_ws, struct OPS_ *ops)
{
	static ModifiedGramSchmidtOrth mgs_orth_static = {
		.block_size = -1  , .orth_zero_tol = 20*DBL_EPSILON,
		.max_reorth = 3   , .reorth_tol    = 50*DBL_EPSILON,
		.mv_ws      = NULL, .dbl_ws        = NULL};
	mgs_orth_static.block_size    = block_size   ;
	mgs_orth_static.orth_zero_tol = orth_zero_tol;
	mgs_orth_static.max_reorth    = max_reorth;
	mgs_orth_static.mv_ws         = mv_ws ;
	mgs_orth_static.dbl_ws        = dbl_ws;
	ops->orth_workspace = (void *)&mgs_orth_static;
	ops->MultiVecOrth = ModifiedGramSchmidt;
	return;
}


static void OrthBinary(void **x,int start_x, int *end_x, void *B, char orth_self_method,
	int block_size, int max_reorth, double orth_zero_tol, double reorth_tol,
	void **mv_ws, double *dbl_ws, struct OPS_ *ops)
{
	if (*end_x<=start_x) return;
#if DEBUG
	ops->Printf("%d,%d,%d\n",start_x,*end_x,block_size);
#endif
		
	int ncols = *end_x-start_x, length, start[2], end[2], idx, inc, idx_abs_max;
	double *beta = dbl_ws, *coef = beta+1;
	if (ncols<=block_size) {
#if TIME_BGS
		time_bgs.orth_self_time -= ops->GetWtime();
#endif
		if (orth_self_method=='E') {
			OrthSelfEVP(x,start_x,end_x,B,
					max_reorth,orth_zero_tol,reorth_tol,mv_ws,coef,ops);
		}
		else {
			OrthSelf(x,start_x,end_x,B,
					max_reorth,orth_zero_tol,reorth_tol,mv_ws,coef,ops);
		}
#if TIME_BGS
		time_bgs.orth_self_time += ops->GetWtime();
#endif
	}
	else {
		start[0] = start_x; end[0] = start_x+ncols/2;
		start[1] = end[0] ; end[1] = *end_x;
		/* 锟斤拷锟斤拷锟斤拷 X0: end[0] 锟斤拷锟杰伙拷谋锟?? */
		OrthBinary(x,start[0],&end[0],B,orth_self_method,
		      block_size,max_reorth,orth_zero_tol,reorth_tol,
		      mv_ws,dbl_ws,ops);		
		//ops->MultiVecView(x, 80, 81, ops);
		//ops->Printf("start = %d-%d, end = %d-%d\n", 
		//      start[0], start[1], end[0], end[1]);
		/* 去锟斤拷 X1 锟斤拷 X0 锟侥诧拷锟斤拷 */
		for (idx = 0; idx < 1+max_reorth; ++idx) {
#if TIME_BGS
			time_bgs.qAp_time -= ops->GetWtime();
#endif
			int start_QtAP[2], end_QtAP[2];
			if (B!=NULL && idx > 0) {
				start_QtAP[0] = start[1]; end_QtAP[0] = end[1];
				start_QtAP[1] = 0       ; end_QtAP[1] = end[0]-start[0];
				ops->MultiVecQtAP('S','T',x,NULL,mv_ws,0,start_QtAP,end_QtAP,
						coef,end_QtAP[1]-start_QtAP[1],mv_ws,ops);
			}
			else {
				start_QtAP[0] = start[1]; end_QtAP[0] = end[1];
				start_QtAP[1] = start[0]; end_QtAP[1] = end[0];
				ops->MultiVecQtAP('S','T',x,B,x,0,start_QtAP,end_QtAP,
						coef,end_QtAP[1]-start_QtAP[1],mv_ws,ops);
			}
#if TIME_BGS
			time_bgs.qAp_time += ops->GetWtime();
#endif

			length = (end[1] - start[1])*(end[0] - start[0]);
			*beta  = -1.0; inc = 1;
			dscal(&length,beta,coef,&inc);		
			*beta  = 1.0;
#if TIME_BGS
			time_bgs.line_comb_time -= ops->GetWtime();
#endif			
			ops->MultiVecLinearComb(x,x,0,start,end,
					coef,end[0]-start[0],beta,0,ops);
#if TIME_BGS
			time_bgs.line_comb_time += ops->GetWtime();
#endif			

			idx_abs_max = idamax(&length,coef,&inc);
			if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
#if DEBUG 
			   ops->Printf("X1 - X0: The number of reorth = %d\n", idx);
#endif
			   break;
			}
		}			
		/* 锟斤拷锟斤拷锟斤拷 X1 */ 
		OrthBinary(x,start[1],&end[1],B,orth_self_method,
		      block_size,max_reorth,orth_zero_tol,reorth_tol,
		      mv_ws,dbl_ws,ops);
		/* 锟斤拷 X0 锟斤拷锟斤拷锟斤拷锟斤拷夭锟斤拷锟斤拷锟?? X1 锟叫碉拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟?? */
		length = start[0]+ncols/2-end[0]; /* 锟斤拷锟斤拷锟斤拷夭锟斤拷值某锟斤拷锟?? */
		*end_x = end[1]-length;
		/* X1 锟斤拷锟斤拷锟斤拷锟街匡拷锟斤拷为 X0 锟斤拷锟皆诧拷锟斤拷锟结供锟侥筹拷锟斤拷*/
		length = (length<end[1]-start[1])?length:(end[1]-start[1]);
		start[1] = end[0]; /* X0 锟斤拷锟斤拷锟斤拷氐锟斤拷锟绞嘉伙拷锟?? */
		end[0]	 = end[1]; /* X1 锟斤拷锟斤拷锟斤拷锟街的斤拷锟斤拷位锟斤拷 */
		start[0] = end[0]   - length;
		end[1]   = start[1] + length;
#if TIME_BGS
		time_bgs.axpby_time -= ops->GetWtime();
#endif 
		ops->MultiVecAxpby(1.0,x,0.0,x,start,end,ops);
#if TIME_BGS
		time_bgs.axpby_time += ops->GetWtime();
#endif				
	}
	return;
}


// 二分递归GramSchmidt正交化
// Input: 
// 		x 矩阵	B 矩阵
//		start_x 起始位置
//		ops 上下文
// Output: 
// 		x 矩阵X部分随机生成
//		end_x 

static void BinaryGramSchmidt(void **x, int start_x, int *end_x, 
		void *B, struct OPS_ *ops)
{
	if (*end_x<=start_x) return;
#if TIME_BGS
	time_bgs.axpby_time     = 0.0;
	time_bgs.line_comb_time = 0.0;
	time_bgs.orth_self_time = 0.0;
	time_bgs.qAp_time       = 0.0;
#endif
	
	BinaryGramSchmidtOrth *bgs_orth = 
		(BinaryGramSchmidtOrth*)ops->orth_workspace;
	int    start[2], end[2], block_size, idx, length, inc, idx_abs_max;
	double *coef   , *beta , orth_zero_tol, reorth_tol;
	void   **mv_ws;
	orth_zero_tol = bgs_orth->orth_zero_tol;
	reorth_tol    = bgs_orth->reorth_tol;
	block_size    = bgs_orth->block_size;
	mv_ws         = bgs_orth->mv_ws;
	beta          = bgs_orth->dbl_ws;
	/* 去掉 X1 中 X0 的部分 */
	if (start_x > 0) {
		start[0] = 0     ; end[0] = start_x;
		start[1] = end[0]; end[1] = *end_x ;
		coef     = beta+1; 
		for (idx = 0; idx < 1+bgs_orth->max_reorth; ++idx) {
#if TIME_BGS
			time_bgs.qAp_time -= ops->GetWtime();
#endif
			ops->MultiVecQtAP('S','N',x,B,x,0,start,end,
					coef,end[0]-start[0],mv_ws,ops);
#if TIME_BGS
			time_bgs.qAp_time += ops->GetWtime();
#endif
			length = (end[1] - start[1])*(end[0] - start[0]); 
			*beta  = -1.0; inc = 1;
			dscal(&length,beta,coef,&inc);		
			*beta  = 1.0;
#if TIME_BGS
			time_bgs.line_comb_time -= ops->GetWtime();
#endif
			ops->MultiVecLinearComb(x,x,0,start,end,
					coef,end[0]-start[0],beta,0,ops);
#if TIME_BGS
			time_bgs.line_comb_time += ops->GetWtime();
#endif		
			idx_abs_max = idamax(&length,coef,&inc);
			if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
#if DEBUG
			   ops->Printf("X1 - X0: The number of reorth = %d\n", idx);
#endif
			   break;
			}
		}
	}
	/* 二分块正交化 */
	/* 4<= block_size <= (*end_x-start_x)/4 */
	char orth_self_method;
	if ((*end_x-start_x)<16) {
		if (block_size<=0) {
			block_size = 4;
		}
		/* 使用 OrthSelf */
		orth_self_method = 'M';		
	}
	else {		
		if (block_size<=0 || block_size>(*end_x-start_x)/4) {
			block_size = (*end_x-start_x)/4;
		}
		/* 使用 OrthSelfEVP */
		orth_self_method = 'E';
	}
	//ops->Printf("start_x = %d, end_x = %d, block_size = %d\n", start_x, *end_x, block_size);

	OrthBinary(x,start_x,end_x,B,orth_self_method,
	      block_size,bgs_orth->max_reorth,orth_zero_tol,reorth_tol,
	      mv_ws,bgs_orth->dbl_ws,ops);
	      
	      
#if TIME_BGS
	ops->Printf("|--BGS----------------------------\n");
	time_bgs.time_total = time_bgs.axpby_time
		+time_bgs.line_comb_time
		+time_bgs.orth_self_time
		+time_bgs.qAp_time;
	ops->Printf("|axpby  line_comb  orth_self  qAp\n");
	ops->Printf("|%.2f\t%.2f\t%.2f\t%.2f\n",
		time_bgs.axpby_time,		
		time_bgs.line_comb_time,		
		time_bgs.orth_self_time,		
		time_bgs.qAp_time);
	ops->Printf("|%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n",
		time_bgs.axpby_time    /time_bgs.time_total*100,
		time_bgs.line_comb_time/time_bgs.time_total*100,
		time_bgs.orth_self_time/time_bgs.time_total*100,
		time_bgs.qAp_time      /time_bgs.time_total*100);
	ops->Printf("|--BGS----------------------------\n");
	time_bgs.axpby_time     = 0.0;
	time_bgs.line_comb_time = 0.0;
	time_bgs.orth_self_time = 0.0;
	time_bgs.qAp_time       = 0.0;	
#endif
	return;
}


void MultiVecOrthSetup_BinaryGramSchmidt(
		int block_size, int max_reorth, double orth_zero_tol, 
		void **mv_ws, double *dbl_ws, struct OPS_ *ops)
{
	static BinaryGramSchmidtOrth bgs_orth_static = {
		.block_size = 16  , .orth_zero_tol = 20*DBL_EPSILON, 
		.max_reorth = 4   , .reorth_tol    = 50*DBL_EPSILON,
		.mv_ws      = NULL, .dbl_ws        = NULL};
	bgs_orth_static.block_size    = block_size   ;
	bgs_orth_static.orth_zero_tol = orth_zero_tol;
	bgs_orth_static.max_reorth    = max_reorth;
	bgs_orth_static.mv_ws         = mv_ws ;
	bgs_orth_static.dbl_ws        = dbl_ws;
	ops->orth_workspace = (void *)&bgs_orth_static;
	ops->MultiVecOrth = BinaryGramSchmidt;
	return;
}
