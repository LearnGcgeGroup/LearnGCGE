/**
 * @brief 针对开源库mmio的高层封装
 * 提供通过一行函数调用读取MTX文件的功能
 */
#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_

#ifndef VALUE_TYPE
#define VALUE_TYPE  double
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif
#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE VALUE_TYPE
#endif
#ifdef __cplusplus
extern "C"{
#endif
#include "mmio.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
void exclusive_scan(MAT_PTR_TYPE* input, int length)
{
    if (length == 0 || length == 1) return;

    MAT_PTR_TYPE old_val, new_val;

    old_val  = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++) {
        new_val  = input[i];
        input[i] = old_val + input[i - 1];
        // printf("the new value:%d\n",input[i]);
        old_val = new_val;
    }
}

int read_mtx_header(char* filename, int* isComplex)
{
    MM_typecode matcode;
    FILE*       f;
    int         isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0;
    *isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL) return -1;

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode)) {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode)) {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode)) {
        *isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode)) {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    fclose(f);

    return 0;
}

/**
 * @brief 通过一个函数读取MTX文件的信息
 * 
 * @param m 行数
 * @param n 列数
 * @param nnz 非零元个数
 * @param isSymmetric 是否对称
 * @param base 元素索引起始值：0或1
 * @param csrRowPtr 行指针
 * @param csrColIdx 列索引
 * @param csrVal 非零元值数组
 * @param filename MTX文件路径
 * @return int 错误码
 */
int mmio_allinone(int* m, int* n, int* nnz, int* isSymmetric, int* base, int** csrRowPtr, int** csrColIdx, MAT_VAL_TYPE** csrVal, char* filename)
{
    int          m_tmp, n_tmp;
    MAT_PTR_TYPE nnz_tmp;

    int         ret_code;
    MM_typecode matcode;
    FILE*       f;

    MAT_PTR_TYPE nnz_mtx_report;
    int          isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL) return -1;

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode)) {
        isPattern = 1; /*printf("type = Pattern\n");*/
        printf("type = Pattern\n");
    }
    if (mm_is_real(matcode)) {
        isReal = 1; /*printf("type = real\n");*/
        printf("type = real\n");
    }
    if (mm_is_complex(matcode)) {
        isComplex = 1; /*printf("type = complex\n");*/
        printf("type = complex\n");
    }
    if (mm_is_integer(matcode)) {
        isInteger = 1; /*printf("type = integer\n");*/
        printf("type = integer\n");
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    // ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0) return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
        isSymmetric_tmp = 1;
        printf("input matrix is symmetric = true\n");
    } else {
        printf("input matrix is symmetric = false\n");
    }

    MAT_PTR_TYPE* csrRowPtr_counter = (MAT_PTR_TYPE*)malloc((m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));

    int*          csrRowIdx_tmp = (int*)malloc(nnz_mtx_report * sizeof(int));
    int*          csrColIdx_tmp = (int*)malloc(nnz_mtx_report * sizeof(int));
    MAT_VAL_TYPE* csrVal_tmp    = (MAT_VAL_TYPE*)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
        int    idxi, idxj;
        double fval, fval_im;
        int    ival;
        int    returnvalue;

        if (isReal) {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        } else if (isComplex) {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        } else if (isInteger) {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval        = ival;
        } else if (isPattern) {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval        = 1.0;
        }

        if ((*base) == 1) { // adjust from 1-based to 0-based
            idxi--;
            idxj--;
        }
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i]    = fval;
    }

    if (f != stdin) fclose(f);

    if (isSymmetric_tmp) {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }
    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp + 1);

    MAT_PTR_TYPE* csrRowPtr_alias = (MAT_PTR_TYPE*)malloc((m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    nnz_tmp                       = csrRowPtr_counter[m_tmp];
    int*          csrColIdx_alias = (int*)malloc(nnz_tmp * sizeof(int));
    MAT_VAL_TYPE* csrVal_alias    = (MAT_VAL_TYPE*)malloc(nnz_tmp * sizeof(MAT_VAL_TYPE));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));

    if (isSymmetric_tmp) {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) {
                MAT_PTR_TYPE offset     = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset]    = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset                  = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset]    = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            } else {
                MAT_PTR_TYPE offset     = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset]    = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    } else {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
            MAT_PTR_TYPE offset     = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset]    = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    *m           = m_tmp;
    *n           = n_tmp;
    *nnz         = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrVal    = csrVal_alias;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

int mmio_allinone_complex(int* m, int* n, int* nnz, int* isSymmetric, int* base, int** csrRowPtr, int** csrColIdx, MAT_VAL_TYPE** csrVal_re,
                          MAT_VAL_TYPE** csrVal_im, char* filename)
{
    int          m_tmp, n_tmp;
    MAT_PTR_TYPE nnz_tmp;

    int         ret_code;
    MM_typecode matcode;
    FILE*       f;

    MAT_PTR_TYPE nnz_mtx_report;
    int          isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL) return -1;

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode)) {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode)) {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode)) {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode)) {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0) return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    } else {
        // printf("input matrix is symmetric = false\n");
    }

    MAT_PTR_TYPE* csrRowPtr_counter = (MAT_PTR_TYPE*)malloc((m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));

    int*          csrRowIdx_tmp = (int*)malloc(nnz_mtx_report * sizeof(int));
    int*          csrColIdx_tmp = (int*)malloc(nnz_mtx_report * sizeof(int));
    MAT_VAL_TYPE* csrVal_tmp    = (MAT_VAL_TYPE*)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE* csrVal_tmp_re = (MAT_VAL_TYPE*)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE* csrVal_tmp_im = (MAT_VAL_TYPE*)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
        int    idxi, idxj;
        double fval, fval_im;
        int    ival = 0;

        if (isReal) {
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        } else if (isComplex) {
            fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
            csrVal_tmp_re[i] = fval;
            csrVal_tmp_im[i] = fval_im;
        } else if (isInteger) {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        } else if (isPattern) {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        if ((*base) == 1) { // adjust from 1-based to 0-based
            idxi--;
            idxj--;
        }
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i]    = fval;
    }

    if (f != stdin) fclose(f);

    if (isSymmetric_tmp) {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp + 1);

    MAT_PTR_TYPE* csrRowPtr_alias = (MAT_PTR_TYPE*)malloc((m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    nnz_tmp                       = csrRowPtr_counter[m_tmp];
    int*          csrColIdx_alias = (int*)malloc(nnz_tmp * sizeof(int));
    MAT_VAL_TYPE* csrVal_alias    = (MAT_VAL_TYPE*)malloc(nnz_tmp * sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE* csrVal_alias_re = (MAT_VAL_TYPE*)malloc(nnz_tmp * sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE* csrVal_alias_im = (MAT_VAL_TYPE*)malloc(nnz_tmp * sizeof(MAT_VAL_TYPE));
    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));

    if (isSymmetric_tmp) {
        if (isReal) {
            for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {

                if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) {
                    MAT_PTR_TYPE offset     = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                    csrColIdx_alias[offset] = csrColIdx_tmp[i];
                    csrVal_alias[offset]    = csrVal_tmp[i];
                    csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                    offset                  = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                    csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                    csrVal_alias[offset]    = csrVal_tmp[i];
                    csrRowPtr_counter[csrColIdx_tmp[i]]++;
                } else {
                    MAT_PTR_TYPE offset     = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                    csrColIdx_alias[offset] = csrColIdx_tmp[i];
                    csrVal_alias[offset]    = csrVal_tmp[i];
                    csrRowPtr_counter[csrRowIdx_tmp[i]]++;
                }
            }
        }
        if (isComplex) {

            for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {

                if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) {
                    MAT_PTR_TYPE offset     = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                    csrColIdx_alias[offset] = csrColIdx_tmp[i];
                    csrVal_alias_re[offset] = csrVal_tmp_re[i];
                    csrVal_alias_im[offset] = csrVal_tmp_im[i];
                    csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                    offset                  = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                    csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                    csrVal_alias_re[offset] = csrVal_tmp_re[i];
                    csrVal_alias_im[offset] = csrVal_tmp_im[i];
                    csrRowPtr_counter[csrColIdx_tmp[i]]++;
                } else {
                    MAT_PTR_TYPE offset     = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                    csrColIdx_alias[offset] = csrColIdx_tmp[i];
                    csrVal_alias_re[offset] = csrVal_tmp_re[i];
                    csrVal_alias_im[offset] = csrVal_tmp_im[i];
                    csrRowPtr_counter[csrRowIdx_tmp[i]]++;
                }
            }
        }
    } else {
        if (isReal) {

            for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
                MAT_PTR_TYPE offset     = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset]    = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }

        if (isComplex) {

            for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
                MAT_PTR_TYPE offset     = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias_re[offset] = csrVal_tmp_re[i];
                csrVal_alias_im[offset] = csrVal_tmp_im[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }

    *m           = m_tmp;
    *n           = n_tmp;
    *nnz         = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    if (isComplex) {
        *csrVal_re = csrVal_alias_re;
        *csrVal_im = csrVal_alias_im;
    } else {
        *csrVal_re = csrVal_alias;
        *csrVal_im = 0;
    }

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrVal_tmp_re);
    free(csrVal_tmp_im);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}
#ifdef __cplusplus
}
#endif
#endif
