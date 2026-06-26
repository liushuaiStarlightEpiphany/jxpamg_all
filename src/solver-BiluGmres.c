#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include "jx_mv.h"
#include "jx_bsr_mv.h"
#include "jx_parbsr_mv.h"
#include "jx_krylov.h"
#include "jx_parbilu.h"
#include "bsr_block_ops.h"

typedef double Real;
typedef int Int;

/* bilu_gmres_setup is a no-op; bilu_gmres_solve calls jx_BILUSolve */
static JX_Int bilu_gmres_setup(JX_Solver solver, JX_Matrix A, JX_Vector b, JX_Vector x)
{
    (void)A; (void)b; (void)x;
    return 0;
}

static JX_Int bilu_gmres_solve(JX_Solver solver, JX_Matrix A, JX_Vector b, JX_Vector x)
{
    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    (void)A;
    jx_ParBILUData *ilu_data = (jx_ParBILUData *)solver;
    JX_Int ret = jx_BILUSolve((void*)ilu_data, ilu_data->matA,
                        (jx_ParVector*)b, (jx_ParVector*)x);
    return ret;
}

jx_Vector* read_rhs_file(const char* filename, int expected_size);
void store_x(int n, Real *x, char *filename);

/* --- LU factorization verification helpers --- */
static double frob_norm(jx_BSRMatrix* A, int bnnz) {
    JX_Real* d = jx_BSRMatrixData(A);
    JX_Int* ip = jx_BSRMatrixI(A);
    int nr = jx_BSRMatrixNumRows(A);
    double s = 0.0;
    for (int k = 0; k < ip[nr]; k++)
        for (int i = 0; i < bnnz; i++)
            s += d[k * bnnz + i] * d[k * bnnz + i];
    return sqrt(s);
}

static int verify_ilu_factorization(void* ilu_vdata, jx_ParBSRMatrix* A_par, int myid) {
    jx_ParBILUData* bilu = (jx_ParBILUData*)ilu_vdata;
    jx_ParBSRMatrix* L_par = jx_ParILUDataMatL(bilu);
    jx_ParBSRMatrix* U_par = jx_ParILUDataMatU(bilu);
    jx_ParVector* D_array = jx_ParILUDataD(bilu);
    if (!L_par || !U_par) {
        if (myid == 0) printf("[ILU Verify] L_par/U_par unavailable\n");
        return -1;
    }

    int blk = jx_ParBSRMatrixBlockSize(A_par);
    int bnnz = blk * blk;
    jx_BSRMatrix* A_diag = jx_ParBSRMatrixDiag(A_par);
    jx_BSRMatrix* L_diag = jx_ParBSRMatrixDiag(L_par);
    jx_BSRMatrix* U_diag = jx_ParBSRMatrixDiag(U_par);
    int n_loc = jx_BSRMatrixNumRows(L_diag);

    double normA_sq = frob_norm(A_diag, bnnz);
    normA_sq *= normA_sq;
    MPI_Allreduce(MPI_IN_PLACE, &normA_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double normA = sqrt(normA_sq);

    double diag_err_sq = 0.0, diff_lu_sq = 0.0, full_err_sq = 0.0;
    double d_min = 1e300, d_max = 0.0;

    if (D_array) {
        JX_Real* Dd = jx_VectorData(jx_ParVectorLocalVector(D_array));
        for (int i = 0; i < n_loc * bnnz; i++) {
            double a = fabs(Dd[i]);
            if (a > 0 && a < d_min) d_min = a;
            if (a > d_max) d_max = a;
        }

        /* ||A_diag - I||_F */
        {
            JX_Real* Ad = jx_BSRMatrixData(A_diag);
            JX_Int* Ai = jx_BSRMatrixI(A_diag);
            JX_Int* Aj = jx_BSRMatrixJ(A_diag);
            for (int r = 0; r < n_loc; r++)
                for (int p = Ai[r]; p < Ai[r+1]; p++)
                    if (Aj[p] == r)
                        for (int i = 0; i < blk; i++)
                            for (int j = 0; j < blk; j++) {
                                double d = Ad[p*bnnz + i*blk + j] - (i==j ? 1.0 : 0.0);
                                diag_err_sq += d * d;
                            }
        }

        /* Compute D = inv(D_inv) */
        JX_Real* D = malloc(n_loc * bnnz * sizeof(JX_Real));
        for (int r = 0; r < n_loc; r++) {
            memcpy(D + r*bnnz, Dd + r*bnnz, bnnz * sizeof(JX_Real));
            jx_bsr_block_inv(D + r*bnnz, blk);
        }

        JX_Int *Li = jx_BSRMatrixI(L_diag), *Lj = jx_BSRMatrixJ(L_diag);
        JX_Real *Ld = jx_BSRMatrixData(L_diag);
        JX_Int *Ui = jx_BSRMatrixI(U_diag), *Uj = jx_BSRMatrixJ(U_diag);
        JX_Real *Ud = jx_BSRMatrixData(U_diag);
        JX_Int *Ai2 = jx_BSRMatrixI(A_diag), *Aj2 = jx_BSRMatrixJ(A_diag);
        JX_Real *Ad2 = jx_BSRMatrixData(A_diag);

        JX_Real *contrib = malloc(bnnz * sizeof(JX_Real));
        JX_Real *temp = malloc(bnnz * sizeof(JX_Real));
        JX_Real *temp2 = malloc(bnnz * sizeof(JX_Real));

        /* ||A - (I+L)*D*(I+U)||_F (A pattern only) — LDU form: U stored as D⁻¹·C */
        for (int r = 0; r < n_loc; r++) {
            const JX_Real* Dr = D + r*bnnz;
            for (int p = Ai2[r]; p < Ai2[r+1]; p++) {
                int c = Aj2[p];
                memset(contrib, 0, bnnz * sizeof(JX_Real));
                if (r == c)
                    for (int i = 0; i < bnnz; i++) contrib[i] += Dr[i];
                if (c > r)
                    for (int q = Ui[r]; q < Ui[r+1]; q++)
                        if (Uj[q] == c) {
                            jx_bsr_block_matmul(Dr, Ud + q*bnnz, temp, blk);
                            for (int i = 0; i < bnnz; i++) contrib[i] += temp[i];
                            break;
                        }
                for (int q = Li[r]; q < Li[r+1]; q++) {
                    int k = Lj[q];
                    if (k == c) {
                        jx_bsr_block_matmul(Ld + q*bnnz, D + k*bnnz, temp, blk);
                        for (int i = 0; i < bnnz; i++) contrib[i] += temp[i];
                    }
                    for (int s = Ui[k]; s < Ui[k+1]; s++)
                        if (Uj[s] == c) {
                            jx_bsr_block_matmul(D + k*bnnz, Ud + s*bnnz, temp, blk);
                            jx_bsr_block_matmul(Ld + q*bnnz, temp, temp2, blk);
                            for (int i = 0; i < bnnz; i++) contrib[i] += temp2[i];
                            break;
                        }
                }
                const JX_Real* Ablk = Ad2 + p*bnnz;
                for (int i = 0; i < bnnz; i++) { double d = Ablk[i] - contrib[i]; diff_lu_sq += d*d; }
            }
        }

        /* Full reconstruction error: ||A - (I+L)*D*(I+U)||_F (LDU form) */
        for (int r = 0; r < n_loc; r++) {
            const JX_Real* Dr = D + r*bnnz;
            for (int c = 0; c < n_loc; c++) {
                memset(contrib, 0, bnnz * sizeof(JX_Real));
                if (c == r)
                    for (int i = 0; i < bnnz; i++) contrib[i] += Dr[i];
                else if (c > r)
                    for (int q = Ui[r]; q < Ui[r+1]; q++)
                        if (Uj[q] == c) {
                            jx_bsr_block_matmul(Dr, Ud + q*bnnz, temp, blk);
                            for (int i = 0; i < bnnz; i++) contrib[i] += temp[i];
                            break;
                        }
                for (int q = Li[r]; q < Li[r+1]; q++) {
                    int k = Lj[q];
                    if (k == c) {
                        jx_bsr_block_matmul(Ld + q*bnnz, D + k*bnnz, temp, blk);
                        for (int i = 0; i < bnnz; i++) contrib[i] += temp[i];
                    }
                    for (int s = Ui[k]; s < Ui[k+1]; s++)
                        if (Uj[s] == c) {
                            jx_bsr_block_matmul(D + k*bnnz, Ud + s*bnnz, temp, blk);
                            jx_bsr_block_matmul(Ld + q*bnnz, temp, temp2, blk);
                            for (int i = 0; i < bnnz; i++) contrib[i] += temp2[i];
                            break;
                        }
                }
                int found = 0;
                for (int p = Ai2[r]; p < Ai2[r+1]; p++)
                    if (Aj2[p] == c) {
                        const JX_Real* Ablk = Ad2 + p*bnnz;
                        for (int i = 0; i < bnnz; i++) { double d = Ablk[i] - contrib[i]; full_err_sq += d*d; }
                        found = 1; break;
                    }
                if (!found)
                    for (int i = 0; i < bnnz; i++) full_err_sq += contrib[i]*contrib[i];
            }
        }

        free(contrib); free(temp); free(temp2); free(D);
    }

    MPI_Allreduce(MPI_IN_PLACE, &diag_err_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &diff_lu_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &full_err_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &d_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &d_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("\n--- LU Factorization Verification ---\n");
        printf("  ||A||_F = %.6e\n", normA);
        if (D_array) {
            printf("  D_inv: |min|=%6e, |max|=%6e\n", d_min, d_max);
            printf("  ||A_diag - I||_F / ||A||_F: %.6e\n", sqrt(diag_err_sq)/normA);
            printf("  ||A - (I+L)*D*(I+U)||_F / ||A||_F: %.10e  (A pattern)\n", sqrt(diff_lu_sq)/normA);
            printf("  ||A - (I+L)*D*(I+U)||_F / ||A||_F: %.10e  (full, incl. fill-in)\n", sqrt(full_err_sq)/normA);
        }
        printf("--- End LU Verification ---\n\n");
    }
    return 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

#ifdef _OPENMP
    omp_set_num_threads(1);
#endif

    int myid, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (myid == 0) {
            printf("Usage: %s <matrix_file> <rhs_file> <output_file> [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -sid <id>     solver id (default: 27 for BILU-GMRES)\n");
            printf("  -kdim <k>     Krylov subspace dimension (default: 30)\n");
            printf("  -maxit <n>    max iterations (default: 100)\n");
            printf("  -tol <t>      tolerance (default: 1e-7)\n");
            printf("  -binary       read BSR matrix in binary format\n");
        }
        MPI_Finalize();
        return 1;
    }

    char *filename_matrix = argv[1];
    char *filename_b      = argv[2];
    char *filename_x      = argv[3];
    int binary = 0;

    Int solver_id  = 27;
    Int k_dim      = 30;
    Int max_iter   = 100;
    Real tol       = 1e-7;
    Int print_level = 1;

    int arg_index = 4;
    while (arg_index < argc) {
        if (strcmp(argv[arg_index], "-sid") == 0)
            solver_id = atoi(argv[++arg_index]);
        else if (strcmp(argv[arg_index], "-kdim") == 0)
            k_dim = atoi(argv[++arg_index]);
        else if (strcmp(argv[arg_index], "-maxit") == 0)
            max_iter = atoi(argv[++arg_index]);
        else if (strcmp(argv[arg_index], "-tol") == 0)
            tol = atof(argv[++arg_index]);
        else if (strcmp(argv[arg_index], "-binary") == 0)
            binary = 1;
        arg_index++;
    }

    if (myid == 0) {
        printf("===================================================================\n");
        printf("BSR Solver\n");
        printf("Matrix: %s\n", filename_matrix);
        printf("RHS:    %s\n", filename_b);
        printf("Output: %s\n", filename_x);
        printf("sid:    %d  (27=BILU-GMRES)\n", solver_id);
        printf("k_dim:  %d\n", k_dim);
        printf("max_it: %d\n", max_iter);
        printf("tol:    %.1e\n", tol);
        printf("binary: %s\n", binary ? "yes" : "no");
        printf("MPI:    %d procs\n", size);
        printf("===================================================================\n\n");
    }

    /* ------------------------------------------------------------
       1. 读取BSR矩阵 (rank 0)
       ------------------------------------------------------------ */
    jx_BSRMatrix *A_bsr = NULL;
    jx_Vector *b_Ser = NULL;
    jx_Vector *x_Ser = NULL;
    int Scalar_nrows = 0;

    if (myid == 0) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        if (binary)
            A_bsr = jx_BSRMatrixRead_Binary(filename_matrix);
        else
            A_bsr = jx_BSRMatrixRead(filename_matrix);

        if (!A_bsr) {
            printf("Error reading BSR matrix from %s\n", filename_matrix);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int blk_size = jx_BSRMatrixBlockSize(A_bsr);
        int nblk_row = jx_BSRMatrixNumRows(A_bsr);
        int nblk_col = jx_BSRMatrixNumCols(A_bsr);
        Scalar_nrows = nblk_row * blk_size;

        gettimeofday(&t1, NULL);
        double read_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1e6;
        printf("Matrix read: %.4f s\n", read_time);
        printf("  Block size: %d\n", blk_size);
        printf("  Block rows: %d\n", nblk_row);
        printf("  Block cols: %d\n", nblk_col);
        printf("  Block nnz:  %d\n", jx_BSRMatrixNumNonzeros(A_bsr));
        printf("  Scalar dim: %d x %d\n\n", Scalar_nrows, Scalar_nrows);

        /* 读取右端项 */
        gettimeofday(&t0, NULL);
        b_Ser = read_rhs_file(filename_b, Scalar_nrows);
        if (!b_Ser) {
            printf("Error reading RHS, using default (all ones)\n");
            b_Ser = jx_SeqVectorCreate(Scalar_nrows);
            jx_SeqVectorInitialize(b_Ser);
            JX_Real *bd = jx_VectorData(b_Ser);
            for (int i = 0; i < Scalar_nrows; i++) bd[i] = 1.0;
        }

        x_Ser = jx_SeqVectorCreate(Scalar_nrows);
        jx_SeqVectorInitialize(x_Ser);
        JX_Real *xd = jx_VectorData(x_Ser);
        for (int i = 0; i < Scalar_nrows; i++) xd[i] = 0.0;
        gettimeofday(&t1, NULL);
        printf("Vector setup: %.4f s\n\n", (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1e6);
    }

    /* ------------------------------------------------------------
       2. 生成分区并分发矩阵
       ------------------------------------------------------------ */
    JX_BigInt *global_row_start = NULL;
    JX_BigInt *global_col_start = NULL;
    if (myid == 0) {
        jx_GeneratePartitioning(jx_BSRMatrixNumRows(A_bsr), size, &global_row_start);
        jx_GeneratePartitioning(jx_BSRMatrixNumCols(A_bsr), size, &global_col_start);
    }

    jx_ParBSRMatrix *A_parbsr = jx_BSRMatrixToParBSRMatrix(
        MPI_COMM_WORLD, A_bsr, global_row_start, global_col_start);

    if (!A_parbsr) {
        if (myid == 0) printf("Error creating parallel BSR matrix\n");
        MPI_Finalize();
        return 1;
    }

    int blk_size = jx_ParBSRMatrixBlockSize(A_parbsr);
    int local_nrows = jx_ParBSRMatrixNumRows(A_parbsr);
    int global_nrows = jx_ParBSRMatrixGlobalNumRows(A_parbsr);
    int local_scalar  = local_nrows * blk_size;
    int global_scalar = global_nrows * blk_size;

    if (myid == 0) {
        printf("Parallel matrix distribution done.\n");
        printf("  Global block rows: %d\n", global_nrows);
        printf("  Global scalar rows: %d\n\n", global_scalar);
    }

    /* ------------------------------------------------------------
       3. 创建并行向量
       ------------------------------------------------------------ */
    JX_Int *partitioning;
    jx_ParBSRMatrixGetRowPartitioning(A_parbsr, &partitioning);

    jx_ParVector *par_rhs = jx_VectorToParVector(
        jx_ParBSRMatrixComm(A_parbsr), b_Ser, partitioning);
    jx_ParVector *par_sol = jx_VectorToParVector(
        jx_ParBSRMatrixComm(A_parbsr), x_Ser, partitioning);
    /* partitioning is shared; only one vector should own it */
    jx_ParVectorOwnsPartitioning(par_sol) = 0;

    if (!par_rhs || !par_sol) {
        if (myid == 0) printf("Failed to create parallel vectors\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ------------------------------------------------------------
       4. 求解
       ------------------------------------------------------------ */
    struct timeval t_start, t_stop;
    gettimeofday(&t_start, NULL);

    JX_Solver ilu_solver = NULL;
    JX_Solver gmres_solver = NULL;

    /* 创建 BILU 预条件器 */

    double starttime = 0.0, endtime = 0.0;
    JX_Int TTest = 1;

    if (TTest) starttime = jx_MPI_Wtime();

    JX_BILUCreate(&ilu_solver);
    JX_ILUSetType(ilu_solver, 3);
    JX_ILUSetLevelOfFill(ilu_solver, 0);
    JX_ILUSetMaxIter(ilu_solver, 1);
    // JX_ILUSetIRIters(ilu_solver, 1);
    // JX_ILUSetLowerJacobiIters(ilu_solver, 3);
    // JX_ILUSetUpperJacobiIters(ilu_solver, 3);
    JX_ILUSetTol(ilu_solver, tol);
    JX_ILUSetTriSolve(ilu_solver, 14);
    JX_ILUSetLogging(ilu_solver, 1);
    JX_ILUSetPrintLevel(ilu_solver, print_level);
    JX_ILUSetsweep(ilu_solver, 10);

    ((jx_ParBILUData*)ilu_solver)->matA = A_parbsr;

    jx_BILUSetup((void*)ilu_solver, A_parbsr, par_rhs, par_sol);


    // verify_ilu_factorization(ilu_solver, A_parbsr, myid);

    /* 创建 GMRES 求解器 (BSR版本) */
    JX_ParBSRGMRESCreate(MPI_COMM_WORLD, &gmres_solver);
    JX_GMRESSetKDim(gmres_solver, k_dim);
    JX_GMRESSetIsCheckRestarted(gmres_solver, 1);
    JX_GMRESSetMaxIter(gmres_solver, max_iter);
    JX_GMRESSetTol(gmres_solver, tol);
    JX_GMRESSetLogging(gmres_solver, 1);
    JX_GMRESSetPrintLevel(gmres_solver, print_level);

    JX_GMRESSetPrecond(gmres_solver,
        (JX_PtrToSolverFcn)bilu_gmres_solve,
        (JX_PtrToSolverFcn)bilu_gmres_setup, ilu_solver);

    JX_ParBSRGMRESSetup(gmres_solver, (JX_Matrix)A_parbsr,
                  (JX_Vector)par_rhs, (JX_Vector)par_sol);

    if (TTest)
    {
        endtime = jx_MPI_Wtime();
        jx_GetWallTime(MPI_COMM_WORLD, "BILU(0)-GMRES Setup", starttime, endtime, 0, 2);
    }

    if (TTest) starttime = jx_MPI_Wtime();
    JX_ParBSRGMRESSolve(gmres_solver, (JX_Matrix)A_parbsr,
                                (JX_Matrix)A_parbsr,
                                (JX_Vector)par_rhs, (JX_Vector)par_sol);
    if (TTest)
    {
        endtime = jx_MPI_Wtime();
        jx_GetWallTime(MPI_COMM_WORLD, "BILU(0)-GMRES Solve", starttime, endtime, 0, 2);
    }

    JX_Int num_iterations = 0;
    JX_Real final_res_norm = 0.0;
    JX_GMRESGetNumIterations(gmres_solver, &num_iterations);
    JX_GMRESGetFinalRelativeResidualNorm(gmres_solver, &final_res_norm);

    gettimeofday(&t_stop, NULL);
    Real total_time = (t_stop.tv_sec - t_start.tv_sec) +
                      (t_stop.tv_usec - t_start.tv_usec) / 1e6;

    if (myid == 0) {
        printf("\n=================================================\n");
        printf("Solution Results:\n");
        printf("  Iterations:    %d\n", num_iterations);
        printf("  Final rel res: %.4le\n", final_res_norm);
        printf("  Total time:    %.4f s\n", total_time);
        printf("=================================================\n\n");
    }

    /* ------------------------------------------------------------
       5. 清理
       ------------------------------------------------------------ */
    /* JX_BILUDestroy 会将 BSR 矩阵强转为 CSR 去释放（类型不匹配导致崩溃）。
       先清空 matA/matL/matU，然后手动用正确的 BSR 函数释放。 */
    {
        jx_ParBILUData *ilu = (jx_ParBILUData *)ilu_solver;
        ilu->matA = NULL;
        ilu->matL = NULL;
        ilu->matU = NULL;
    }
    JX_BILUDestroy(ilu_solver);
    JX_ParBSRGMRESDestroy(gmres_solver);

    jx_ParVectorDestroy(par_rhs);
    jx_ParVectorDestroy(par_sol);

    /* A_parbsr 的 row_starts/col_starts 就是 global_row_start/global_col_start，
       jx_ParBSRMatrixDestroy 不会自动释放它们，需先手动释放 */
    if (global_row_start) jx_TFree(global_row_start);
    if (global_col_start) jx_TFree(global_col_start);
    jx_ParBSRMatrixDestroy(A_parbsr);

    if (myid == 0) {
        jx_BSRMatrixDestroy(A_bsr);
        jx_SeqVectorDestroy(b_Ser);
        jx_SeqVectorDestroy(x_Ser);
    }

    MPI_Finalize();
    return 0;
}

/* 读取右端项文件 */
jx_Vector* read_rhs_file(const char* filename, int expected_size)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening RHS file: %s\n", filename);
        return NULL;
    }

    int nentries;
    if (fscanf(file, "%d", &nentries) != 1) {
        printf("Error reading number of entries from RHS file\n");
        fclose(file);
        return NULL;
    }

    if (expected_size != -1 && nentries > expected_size) {
        printf("Warning: RHS entries %d > expected size %d\n", nentries, expected_size);
    }

    int vec_size = (expected_size > 0) ? expected_size : nentries;
    jx_Vector *rhs = jx_SeqVectorCreate(vec_size);
    jx_SeqVectorInitialize(rhs);
    JX_Real *rhs_data = jx_VectorData(rhs);
    for (int i = 0; i < vec_size; i++) rhs_data[i] = 0.0;

    int index;
    double value;
    for (int i = 0; i < nentries; i++) {
        if (fscanf(file, "%d %lf", &index, &value) == 2) {
            if (index >= 0 && index < vec_size)
                rhs_data[index] = value;
        }
    }

    fclose(file);
    return rhs;
}

void store_x(Int n, Real *x, char *filename)
{
    FILE *p = fopen(filename, "w");
    fprintf(p, "%d\n", n);
    for (Int i = 0; i < n; i++)
        fprintf(p, "%.15e\n", x[i]);
    fclose(p);
}

