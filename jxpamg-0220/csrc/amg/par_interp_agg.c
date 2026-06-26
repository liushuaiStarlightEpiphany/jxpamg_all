//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_interp_agg.c --  Form interpolation matrix for aggregation-type AMG
 *  Date: 2025/10/08
 */ 


#include "jx_pamg.h"

/**
 * \fn JX_Int jx_form_tentative_p(jx_ParCSRMatrix* A, JX_Int* Vertices,
 *                                        JX_Int NumAggregates, JX_Int num_coarse_global,
 *                                        jx_ParCSRMatrix** P)
 *
 * \brief Form tentative interpolation matrix for aggregation-type AMG
 *
 * \param A                 The parallel coefficient matrices (Input)
 * \param Vertices          The aggregation of vertices on the current processor (Input)
 * \param NumAggregates     The number of aggregations on the current processor (Input)
 * \param num_coarse_global The number of global aggregations, i.e., num_coarse_global = sum(NumAggregates[i]),i=1,...,np (Input)
 * \param P                 The interpolation matrix (Output)
 *
 * \author Li Zhao
 * \date   10/06/2024
 *
 * \note The interpolation matrix only considers the diagonal part, and currently does not consider the non diagonal part
 */
JX_Int jx_form_tentative_p(jx_ParCSRMatrix* A, JX_Int* Vertices, JX_Int NumAggregates, JX_Int num_coarse_global,
                                   jx_ParCSRMatrix** P)
{
    JX_Int           status = JX_SUCCESS;
    JX_Int           i, j;
    jx_ParCSRMatrix* tentp = NULL;

    MPI_Comm comm = jx_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

    // jx_MemoryLocation location = jx_ParCSRMatrixMemoryLocation(A);
    jx_CSRMatrix*     A_diag   = jx_ParCSRMatrixDiag(A);

    JX_BigInt* row_starts        = jx_ParCSRMatrixRowStarts(A);
    JX_Int     row               = jx_CSRMatrixNumRows(A_diag);
    JX_BigInt  global_num_vars   = jx_ParCSRMatrixGlobalNumRows(A);
    JX_BigInt  col_starts[2]     = {0, 0};
    JX_Int     num_cols_offd     = 0;
    JX_Int     num_nonzeros_diag = 0;
    JX_Int     num_nonzeros_offd = 0;

#if 0
    JX_Int     num_coarse_global = -1;
    jx_MPI_Allreduce(&NumAggregates, &num_coarse_global, 1, JX_MPI_INT, jx_MPI_SUM, comm);
#endif

    // Form col_starts
    JX_Int* recv_buffer = jx_MAlloc(np * sizeof(JX_Int));
    MPI_Allgather(&NumAggregates, 1, JX_MPI_INT, recv_buffer, 1, JX_MPI_INT, comm);
    col_starts[0] = 0;
    col_starts[1] = recv_buffer[0];
    for (i = 0; i < iam; i++) {
        col_starts[0] += recv_buffer[i];
        col_starts[1] += recv_buffer[i + 1];
    }
    jx_Free(recv_buffer);

#if 0
    DEBUG_PRINTF("rank: %d, NumAggregates = %d\n", iam, NumAggregates);
    // DEBUG_PRINTF("rank: %d, row_starts = {%d, %d}\n", iam, row_starts[0], row_starts[1]);
    // DEBUG_PRINTF("rank: %d, col_starts = {%d, %d}\n", iam, col_starts[0], col_starts[1]);
    if (iam == 0) DEBUG_PRINTF("rank: %d, global_num_vars = %d, num_coarse_global = %d\n", iam, global_num_vars, num_coarse_global);
#endif

    /* Form tentative prolongation */
    JX_Int * P_diag_i = NULL, *P_diag_j = NULL;
    JX_Real* P_diag_data = NULL;
    JX_Int * P_offd_i = NULL, *P_offd_j = NULL;
    JX_Real* P_offd_data = NULL;

    P_diag_i = jx_MAlloc((row + 1) * sizeof(JX_Int));
    P_offd_i = jx_CAlloc(row + 1, sizeof(JX_Int)); // P_offd_i = 0

    // first run
    for (i = 0, j = 0; i < row; i++) {
        P_diag_i[i] = j;
        if (Vertices[i] > JX_UNPT) j++;
    }
    P_diag_i[row]     = j;
    num_nonzeros_diag = j;

    // allocate memory for P_diag
    P_diag_j    = jx_MAlloc(num_nonzeros_diag * sizeof(JX_Int));
    P_diag_data = jx_MAlloc(num_nonzeros_diag * sizeof(JX_Real));

    // second run
    for (i = 0, j = 0; i < row; i++) {
        // P_diag_i[i] = j;
        if (Vertices[i] > JX_UNPT) {
            P_diag_j[j]    = Vertices[i];
            P_diag_data[j] = 1.0;
            j++;
        }
    }

    // output
    tentp = jx_ParCSRMatrixCreate(comm, global_num_vars, num_coarse_global, row_starts, col_starts, num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);

    jx_CSRMatrix* tentp_diag     = jx_ParCSRMatrixDiag(tentp);
    jx_CSRMatrixI(tentp_diag)    = P_diag_i;
    jx_CSRMatrixJ(tentp_diag)    = P_diag_j;
    jx_CSRMatrixData(tentp_diag) = P_diag_data;

    jx_CSRMatrix* tentp_offd  = jx_ParCSRMatrixOffd(tentp);
    jx_CSRMatrixI(tentp_offd) = P_offd_i;
    if (num_cols_offd) {
        jx_CSRMatrixData(tentp_offd) = P_offd_data;
        jx_CSRMatrixJ(tentp_offd)    = P_offd_j;
    }

    if (np > 1) {
        jx_MatvecCommPkgCreate(tentp);
    }

    *P = tentp;

    // DEBUG_PRINTF("rank: %d, End\n", iam);

    // return
    return status;
}

/**
 * \fn JX_Int jx_form_tentative_p_par (jx_ParCSRMatrix* A, JX_Int* Vertices, JX_Int* Vert_owner
 *                                             JX_Int NumAggregates, JX_Int num_coarse_global,
 *                                             jx_ParCSRMatrix** P)
 *
 * \brief Form tentative interpolation matrix for parallel aggregation-type AMG
 *
 * \param A                 The parallel coefficient matrices (Input)
 * \param Vertices          The aggregation of vertices on the current processor (Input)
 * \param Vert_owner        The vertex aggregation number on the current processor belongs to the process (Input)
 * \param NumAggregates     The number of aggregations on the current processor (Input)
 * \param num_coarse_global The number of global aggregations, i.e., num_coarse_global = sum(NumAggregates[i]),i=1,...,np (Input)
 * \param P                 The interpolation matrix (Output)
 *
 * \author Li Zhao
 * \date   12/06/2024
 *
 */
JX_Int jx_form_tentative_p_par(jx_ParCSRMatrix* A, JX_Int* Vertices, JX_Int* Vert_owner, JX_Int NumAggregates,
                                       JX_Int num_coarse_global, jx_ParCSRMatrix** P)
{
    JX_Int           status = JX_SUCCESS;
    JX_Int           i, j;
    jx_ParCSRMatrix* tentp = NULL;

    MPI_Comm comm = jx_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

    // DEBUG_PRINTF("rank: %d, Begin\n", iam);

    // jx_MemoryLocation location = jx_ParCSRMatrixMemoryLocation(A);
    jx_CSRMatrix*     A_diag   = jx_ParCSRMatrixDiag(A);

    // JX_BigInt* row_starts      = jx_ParCSRMatrixRowStarts(A);
    JX_BigInt* row_starts      = jx_ParCSRMatrixColStarts(A);
    JX_Int     row             = jx_CSRMatrixNumRows(A_diag);
    JX_BigInt  global_num_vars = jx_ParCSRMatrixGlobalNumRows(A);
    JX_BigInt  col_starts[2]   = {0, 0};

    // Form col_starts
    JX_Int* recv_buffer     = jx_MAlloc(np * sizeof(JX_Int));
    JX_Int* recv_buffer_sum = jx_MAlloc(np * sizeof(JX_Int));
    MPI_Allgather(&NumAggregates, 1, JX_MPI_INT, recv_buffer, 1, JX_MPI_INT, comm);
    col_starts[0] = 0;
    col_starts[1] = recv_buffer[0];
    for (i = 0; i < iam; i++) {
        col_starts[0] += recv_buffer[i];
        col_starts[1] += recv_buffer[i + 1];
    }
    recv_buffer_sum[0] = 0;
    for (i = 1; i < np; i++) recv_buffer_sum[i] = recv_buffer_sum[i - 1] + recv_buffer[i - 1]; // paritial sum

#if 0
    DEBUG_PRINTF("rank: %d, NumAggregates = %d\n", iam, NumAggregates);
    // DEBUG_PRINTF("rank: %d, row_starts = {%d, %d}\n", iam, row_starts[0], row_starts[1]);
    // DEBUG_PRINTF("rank: %d, col_starts = {%d, %d}\n", iam, col_starts[0], col_starts[1]);
    if (iam == 0) DEBUG_PRINTF("rank: %d, global_num_vars = %d, num_coarse_global = %d\n", iam, global_num_vars, num_coarse_global);
#endif

    /* Form tentative prolongation */
    JX_Int * P_diag_i = NULL, *P_diag_j = NULL, *P_offd_global_j = NULL;
    JX_Real* P_diag_data = NULL;
    JX_Int * P_offd_i = NULL, *P_offd_j = NULL;
    JX_Real* P_offd_data   = NULL;
    JX_Int*  col_map_offd  = NULL;
    JX_Int   num_cols_offd = 0;
    JX_Int   num_nnz_diag  = 0;
    JX_Int   num_nnz_offd  = 0;

    P_diag_i = jx_CAlloc(row + 1, sizeof(JX_Int));
    P_offd_i = jx_CAlloc(row + 1, sizeof(JX_Int)); // P_offd_i = 0

    // first run
    P_diag_i[0] = 0;
    P_offd_i[0] = 0;
#ifdef JX_USING_OPENMP
#pragma omp parallel for 
#endif
    for (i = 0; i < row; i++) {
        if (Vertices[i] > JX_UNPT) {
            if (Vert_owner[i] == iam) {
                num_nnz_diag++;
            } else {
                num_nnz_offd++;
            }
        }
        P_diag_i[i + 1] = num_nnz_diag;
        P_offd_i[i + 1] = num_nnz_offd;
    }

    // allocate memory for P_diag
    P_diag_j    = jx_MAlloc(num_nnz_diag * sizeof(JX_Int));
    P_diag_data = jx_MAlloc(num_nnz_diag * sizeof(JX_Real));
    // allocate memory for P_offd
    P_offd_j    = jx_MAlloc(num_nnz_offd * sizeof(JX_Int));
    P_offd_data = jx_MAlloc(num_nnz_offd * sizeof(JX_Real));

    col_map_offd    = jx_MAlloc(num_nnz_offd * sizeof(JX_Int));
    JX_Int* map = jx_MAlloc(num_coarse_global * sizeof(JX_Int));
    for (i = 0; i < num_coarse_global; i++) map[i] = -1;

    // second run
    num_nnz_diag  = 0;
    num_nnz_offd  = 0;
    num_cols_offd = 0;
#ifdef JX_USING_OPENMP
#pragma omp parallel for 
#endif
    for (i = 0; i < row; i++) {
        if (Vertices[i] > JX_UNPT) {
            if (Vert_owner[i] == iam) {
                P_diag_j[num_nnz_diag]    = Vertices[i];
                P_diag_data[num_nnz_diag] = 1.0;
                num_nnz_diag++;
            } else {
                JX_Int global_col_index = Vertices[i] + recv_buffer_sum[Vert_owner[i]];
                // JX_Int local_col_index  = Vertices[i];
                // DEBUG_PRINTF("rank: %d, local_col_index %d, num_nnz_offd %d\n", iam, local_col_index, num_nnz_offd);
                if (map[global_col_index] < 0) {
                    map[global_col_index] = num_cols_offd;
                    // P_offd_j[num_nnz_offd]      = num_cols_offd;
                    P_offd_j[num_nnz_offd]      = global_col_index;
                    col_map_offd[num_cols_offd] = global_col_index;
                    num_cols_offd++;
                } else {
                    // P_offd_j[num_nnz_offd] = map[global_col_index];
                    P_offd_j[num_nnz_offd] = global_col_index;
                }
                P_offd_data[num_nnz_offd] = 1.0;
                num_nnz_offd++;
            }
        }
    }

    //! Note that col_map_offd must be in increasing order, zhaoli, 2024.12.07
    {
        jx_BigQsort0(col_map_offd, 0, num_cols_offd - 1);
        // jx_BinarySearch for JX_BigInt
        for (j = 0; j < num_nnz_offd; ++j) P_offd_j[j] = jx_BinarySearch(col_map_offd, P_offd_j[j], num_cols_offd);
    }

    // free
    if (num_cols_offd == 0) {
        jx_TFree(P_offd_j);
        jx_TFree(P_offd_data);
        jx_TFree(col_map_offd);
        col_map_offd = NULL;
        num_nnz_offd = 0;
    } else if (num_cols_offd < num_nnz_offd) {
        col_map_offd = jx_TReAlloc(col_map_offd, JX_Int, num_cols_offd);
    }
    jx_Free(recv_buffer);
    jx_Free(recv_buffer_sum);
    jx_Free(map);

    //* output
    tentp = jx_ParCSRMatrixCreate(comm, global_num_vars, num_coarse_global, row_starts, col_starts, num_cols_offd, num_nnz_diag, num_nnz_offd);

    jx_CSRMatrix* tentp_diag               = jx_ParCSRMatrixDiag(tentp);
    jx_CSRMatrixI(tentp_diag)              = P_diag_i;
    jx_CSRMatrixJ(tentp_diag)              = P_diag_j;
    jx_CSRMatrixData(tentp_diag)           = P_diag_data;
    // jx_CSRMatrixMemoryLocation(tentp_diag) = location;

    jx_CSRMatrix* tentp_offd  = jx_ParCSRMatrixOffd(tentp);
    jx_CSRMatrixI(tentp_offd) = P_offd_i;
    if (num_cols_offd) {
        jx_CSRMatrixJ(tentp_offd)              = P_offd_j;
        jx_CSRMatrixData(tentp_offd)           = P_offd_data;
        // jx_CSRMatrixMemoryLocation(tentp_offd) = location;
        jx_ParCSRMatrixColMapOffd(tentp)       = col_map_offd;
    }

    if (np > 1) {
        jx_MatvecCommPkgCreate(tentp);
    }

    *P = tentp;

    // DEBUG_PRINTF("rank: %d, End\n", iam);

    // return
    return status;
}

/**
 * \fn JX_Int jx_smooth_agg(jx_ParCSRMatrix* A, jx_ParCSRMatrix* tentp, jx_ParCSRMatrix* S,
 *                                  JX_Int filter, JX_Real smooth_factor, jx_ParCSRMatrix** P)
 *
 * \brief Smooth the tentative prolongation
 *
 * \param A                 Pointer to the coefficient matrices
 * \param tentp             Pointer to the tentative prolongation operators
 * \param S                 Pointer to strongly coupled neighbors
 * \param filter            filter matrix A if true; otherwise don't filter
 * \param smooth_factor     smooth factor of damped Jacobian smoother
 * \param P                 Pointer to the prolongation operators
 *
 * \author Li Zhao
 * \date   12/13/2024
 *
 */
JX_Int jx_smooth_agg(jx_ParCSRMatrix* A, jx_ParCSRMatrix* tentp, jx_ParCSRMatrix* S, JX_Int filter, JX_Real smooth_factor,
                             jx_ParCSRMatrix** P)
{
    jx_CSRMatrix* A_diag      = jx_ParCSRMatrixDiag(A);
    JX_Int*       A_diag_i    = jx_CSRMatrixI(A_diag);
    JX_Int*       A_diag_j    = jx_CSRMatrixJ(A_diag);
    JX_Complex*   A_diag_data = jx_CSRMatrixData(A_diag);
    JX_Int        row         = jx_CSRMatrixNumRows(A_diag);

    jx_CSRMatrix* A_offd      = jx_ParCSRMatrixOffd(A);
    JX_Int*       A_offd_i    = jx_CSRMatrixI(A_offd);
    JX_Int*       A_offd_j    = jx_CSRMatrixJ(A_offd);
    JX_Complex*   A_offd_data = jx_CSRMatrixData(A_offd);

    JX_Int           status = JX_SUCCESS; // return status
    JX_Int           i, k, j;
    jx_ParCSRMatrix* J = NULL;

    MPI_Comm comm = jx_ParCSRMatrixComm(A);
    // MPI_CLAIM(comm);

    // JX_MemoryLocation location = jx_ParCSRMatrixMemoryLocation(A);
    JX_Complex*       diag     = jx_MAlloc(row * sizeof(JX_Complex));

    //* Step 1. Form smoother */

    /* Without filter: Using A for damped Jacobian smoother */
    if (filter == jx_false || S == NULL) {

        // copy structure from A
        J = jx_ParCSRMatrixClone(A, 0);

        JX_Complex* J_diag_data = jx_CSRMatrixData(jx_ParCSRMatrixDiag(J));
        JX_Complex* J_offd_data = jx_CSRMatrixData(jx_ParCSRMatrixOffd(J));

        // get the diagonal entries of A and check it.
        // if it is too small, use Richardson smoother for the corresponding row
#ifdef JX_USING_OPENMP
#pragma omp parallel for private(i) 
#endif
        for (i = 0; i < row; ++i) {
            JX_Complex d = A_diag_data[A_diag_i[i]];
            if (jx_abs(d) < 1e-6)
                diag[i] = 1.0;
            else
                diag[i] = d;
        }

#ifdef JX_USING_OPENMP
#pragma omp parallel for private(i, k) 
#endif
        for (i = 0; i < row; i++) {
            // diagonal parts
            JX_Int ibegin   = A_diag_i[i];
            JX_Int iend     = A_diag_i[i + 1];
            J_diag_data[ibegin] = 1.0 - smooth_factor * A_diag_data[ibegin] / diag[i];
            for (k = ibegin + 1; k < iend; k++) {
                J_diag_data[k] = -smooth_factor * A_diag_data[k] / diag[i];
            }
            // non-diagonal parts
            ibegin = A_offd_i[i];
            iend   = A_offd_i[i + 1];
            for (k = ibegin; k < iend; k++) {
                J_offd_data[k] = -smooth_factor * A_offd_data[k] / diag[i];
            }
        }
    }

    /* Using filtered A for damped Jacobian smoother */
    else {
        jx_ParCSRMatrixReorder(S);

        jx_CSRMatrix* S_diag      = jx_ParCSRMatrixDiag(S);
        JX_Int*       S_diag_i    = jx_CSRMatrixI(S_diag);
        JX_Int*       S_diag_j    = jx_CSRMatrixJ(S_diag);
        JX_Complex*   S_diag_data = jx_CSRMatrixData(S_diag);

        jx_CSRMatrix* S_offd      = jx_ParCSRMatrixOffd(S);
        JX_Int*       S_offd_i    = jx_CSRMatrixI(S_offd);
        JX_Int*       S_offd_j    = jx_CSRMatrixJ(S_offd);
        JX_Complex*   S_offd_data = jx_CSRMatrixData(S_offd);
        JX_Int        S_offd_nnz  = jx_CSRMatrixNumNonzeros(S_offd);

        JX_Complex row_sum_A, row_sum_S;

#ifdef JX_USING_OPENMP
#pragma omp parallel for private(i, k, row_sum_A, row_sum_S) 
#endif
        for (i = 0; i < row; ++i) {
            row_sum_A = 0.0;
            row_sum_S = 0.0;

            //! A
            // diagonal parts
            JX_Int ibegin = A_diag_i[i];
            JX_Int iend   = A_diag_i[i + 1];
            for (k = ibegin + 1; k < iend; ++k) {
                row_sum_A += A_diag_data[k];
            }
            // non-diagonal parts
            ibegin = A_offd_i[i];
            iend   = A_offd_i[i + 1];
            for (k = ibegin; k < iend; k++) {
                row_sum_A += A_offd_data[k];
            }

            //! S
            // non-diagonal parts
            ibegin = S_offd_i[i];
            iend   = S_offd_i[i + 1];
            for (k = ibegin; k < iend; k++) {
                row_sum_S += S_offd_data[k];
            }
            // diagonal parts
            ibegin = S_diag_i[i];
            iend   = S_diag_i[i + 1];
            for (k = ibegin + 1; k < iend; ++k) {
                row_sum_S += S_diag_data[k];
            }

            // modified diagonal entry of S
            S_diag_data[ibegin] += row_sum_A - row_sum_S;
        }

        // copy structure from S (filtered A)
        J = jx_ParCSRMatrixClone(S, 0);

        JX_Complex* J_diag_data = jx_CSRMatrixData(jx_ParCSRMatrixDiag(J));
        JX_Complex* J_offd_data = jx_CSRMatrixData(jx_ParCSRMatrixOffd(J));

        // get the diagonal entries of S and check it.
        // if it is too small, use Richardson smoother for the corresponding row
#ifdef JX_USING_OPENMP
#pragma omp parallel for private(i) 
#endif
        for (i = 0; i < row; ++i) {
            JX_Complex d = S_diag_data[S_diag_i[i]];
            if (jx_abs(d) < 1e-6)
                diag[i] = 1.0;
            else
                diag[i] = d;
        }

#ifdef JX_USING_OPENMP
#pragma omp parallel for private(i, k) 
#endif
        for (i = 0; i < row; i++) {
            // diagonal parts
            JX_Int ibegin   = S_diag_i[i];
            JX_Int iend     = S_diag_i[i + 1];
            J_diag_data[ibegin] = 1.0 - smooth_factor * S_diag_data[ibegin] / diag[i];
            for (k = ibegin + 1; k < iend; k++) {
                J_diag_data[k] = -smooth_factor * S_diag_data[k] / diag[i];
            }
            // non-diagonal parts
            ibegin = S_offd_i[i];
            iend   = S_offd_i[i + 1];
            for (k = ibegin; k < iend; k++) {
                J_offd_data[k] = -smooth_factor * S_offd_data[k] / diag[i];
            }
        }
    }

    //* Step 2. Smooth the tentative prolongation P = J*tenp */
    *P = jx_ParMatmul(J, tentp); // P = J*tenp

    //* free
    jx_TFree(diag);
    jx_ParCSRMatrixDestroy(J);

    // return
    return status;
}
