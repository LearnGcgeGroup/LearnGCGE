#include "mmloader.h"

PetscErrorCode MatCreateFromMTX(Mat *A, const char *filein, PetscBool aijonly) {
    MM_typecode matcode;
    FILE *file;
    // M: 行数, N: 列数, nz: 非零元素数
    PetscInt M, N, nz;
    // ia: 行索引数组, ja: 列索引数组
    PetscInt *ia = NULL, *ja = NULL;
    // val: 非零元素数组
    PetscScalar *val = NULL;
    PetscBool symmetric = PETSC_FALSE, skew = PETSC_FALSE;
    PetscMPIInt size, rank;

    PetscFunctionBeginUser;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    printf("Rank %d: size = %d\n", rank, size);

    if (rank == 0) {
        PetscCall(PetscFOpen(PETSC_COMM_SELF, filein, "r", &file));
        PetscCheck(mm_read_banner(file, &matcode) == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not process Matrix Market banner.");
        /*  This is how one can screen matrix types if their application */
        /*  only supports a subset of the Matrix Market data types.      */
        PetscCheck(mm_is_matrix(matcode) && mm_is_sparse(matcode) && (mm_is_real(matcode) || mm_is_integer(matcode)), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input must be a sparse real or integer matrix.");

        if (mm_is_symmetric(matcode)) symmetric = PETSC_TRUE;
        if (mm_is_skew(matcode)) skew = PETSC_TRUE;

        /* Find out size of sparse matrix .... */
        PetscCheck(mm_read_mtx_crd_size(file, &M, &N, &nz) == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Size of sparse matrix is wrong.");

        /* Reserve memory for matrices */
        PetscCall(PetscMalloc3(nz, &ia, nz, &ja, nz, &val));
        /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
        /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
        /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)  */
        PetscInt row, col;
        PetscScalar value;
        PetscInt ninput; // 返回值
        for (PetscInt i = 0; i < nz; i++) {
            ninput = fscanf(file, "%d %d %lg\n", &ia[i], &ja[i], &val[i]);
            PetscCheck(ninput >= 3, PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED, "Badly formatted input file");
            ia[i]--;
            ja[i]--;                              /* adjust from 1-based to 0-based */
            

            // 原始读取方法
            // fscanf(file, "%d %d %lg\n", &row, &col, &value);
            // ia[i] = row - 1;
            // ja[i] = col - 1;
            // val[i] = value;
        }
        PetscCall(PetscFClose(PETSC_COMM_SELF, file));
    }

    /* 广播 M, N, nz 到所有进程 */
    PetscCallMPI(MPI_Bcast(&M, 1, MPI_INT, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Bcast(&N, 1, MPI_INT, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Bcast(&nz, 1, MPI_INT, 0, PETSC_COMM_WORLD));
    if (rank != 0) {
        PetscCall(PetscMalloc3(nz, &ia, nz, &ja, nz, &val));
    }
    PetscCallMPI(MPI_Bcast(ia, nz, MPI_INT, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Bcast(ja, nz, MPI_INT, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Bcast(val, nz, MPIU_SCALAR, 0, PETSC_COMM_WORLD));

    PetscInt mstart = (M * rank) / size;
    PetscInt mend = (M * (rank + 1)) / size;
    PetscInt mlocal = mend - mstart;
    printf("Rank %d: mstart = %d, mend = %d, mlocal = %d\n", rank, mstart, mend, mlocal);
    /* Create, preallocate, and then assemble the matrix */
    PetscCall(MatCreate(PETSC_COMM_WORLD, A));
    PetscCall(MatSetSizes(*A, mlocal, PETSC_DECIDE, M, N));
    // MATSEQSBAIJ或MATMPISBAIJ 设置后，只需要填充将上三角矩阵，petsc自动补齐下三角矩阵
    PetscCall(MatSetType(*A, MATMPISBAIJ));
    // PetscCall(MatSetFromOptions(*A));
    PetscCall(MatSetUp(*A));

    /* Add values to upper triangular part for some cases */
    for (PetscInt i = 0; i < nz; i++) {
        if (ja[i] >= mstart && ja[i] < mend) {
            PetscCall(MatSetValues(*A, 1, &ja[i], 1, &ia[i], &val[i], INSERT_VALUES));
            // if (symmetric && ia[i] != ja[i]) {
            // PetscCall(MatSetValues(*A, 1, &ja[i], 1, &ia[i], &val[i], INSERT_VALUES));
            // }
        }
    }

    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));

    PetscCall(PetscFree3(ia, ja, val));

    PetscInt local_rows, local_cols; // 本地行数和列数
    // 获取全局大小和本地大小
    PetscCall(MatGetLocalSize(*A, &local_rows, &local_cols)); // 本地大小
    printf("Rank %d: Global rows = %d, Local rows = %d\n", rank, M, local_rows);
    // 打印结果
    // PetscPrintf(PETSC_COMM_WORLD, "Rank %d: Global rows = %d, Local rows = %d\n", rank, M, local_rows);
    //  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));  // 等待所有进程到达此处
    PetscFunctionReturn(PETSC_SUCCESS);
}
