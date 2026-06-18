//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_interp_agg.c --  Form interpolation matrix for aggregation-type AMG
 *  Date: 2025/10/08
 */ 


#include "jxf_pamg.h"

/**
 * \fn JXF_Int jxf_form_tentative_p(jxf_ParCSRMatrix* A, JXF_Int* Vertices,
 *                                        JXF_Int NumAggregates, JXF_Int num_coarse_global,
 *                                        jxf_ParCSRMatrix** P)
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
JXF_Int jxf_form_tentative_p(jxf_ParCSRMatrix* A, JXF_Int* Vertices, JXF_Int NumAggregates, JXF_Int num_coarse_global,
                                   jxf_ParCSRMatrix** P)
{
    JXF_Int           status = JXF_SUCCESS;
    JXF_Int           i, j;
    jxf_ParCSRMatrix* tentp = NULL;

    MPI_Comm comm = jxf_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

    // jxf_MemoryLocation location = jxf_ParCSRMatrixMemoryLocation(A);
    jxf_CSRMatrix*     A_diag   = jxf_ParCSRMatrixDiag(A);

    JXF_BigInt* row_starts        = jxf_ParCSRMatrixRowStarts(A);
    JXF_Int     row               = jxf_CSRMatrixNumRows(A_diag);
    JXF_BigInt  global_num_vars   = jxf_ParCSRMatrixGlobalNumRows(A);
    JXF_BigInt  col_starts[2]     = {0, 0};
    JXF_Int     num_cols_offd     = 0;
    JXF_Int     num_nonzeros_diag = 0;
    JXF_Int     num_nonzeros_offd = 0;

#if 0
    JXF_Int     num_coarse_global = -1;
    jxf_MPI_Allreduce(&NumAggregates, &num_coarse_global, 1, JXF_MPI_INT, jxf_MPI_SUM, comm);
#endif

    // Form col_starts
    JXF_Int* recv_buffer = jxf_MAlloc(np * sizeof(JXF_Int));
    MPI_Allgather(&NumAggregates, 1, JXF_MPI_INT, recv_buffer, 1, JXF_MPI_INT, comm);
    col_starts[0] = 0;
    col_starts[1] = recv_buffer[0];
    for (i = 0; i < iam; i++) {
        col_starts[0] += recv_buffer[i];
        col_starts[1] += recv_buffer[i + 1];
    }
    jxf_Free(recv_buffer);

#if 0
    DEBUG_PRINTF("rank: %d, NumAggregates = %d\n", iam, NumAggregates);
    // DEBUG_PRINTF("rank: %d, row_starts = {%d, %d}\n", iam, row_starts[0], row_starts[1]);
    // DEBUG_PRINTF("rank: %d, col_starts = {%d, %d}\n", iam, col_starts[0], col_starts[1]);
    if (iam == 0) DEBUG_PRINTF("rank: %d, global_num_vars = %d, num_coarse_global = %d\n", iam, global_num_vars, num_coarse_global);
#endif

    /* Form tentative prolongation */
    JXF_Int * P_diag_i = NULL, *P_diag_j = NULL;
    JXF_Real* P_diag_data = NULL;
    JXF_Int * P_offd_i = NULL, *P_offd_j = NULL;
    JXF_Real* P_offd_data = NULL;

    P_diag_i = jxf_MAlloc((row + 1) * sizeof(JXF_Int));
    P_offd_i = jxf_CAlloc(row + 1, sizeof(JXF_Int)); // P_offd_i = 0

    // first run
    for (i = 0, j = 0; i < row; i++) {
        P_diag_i[i] = j;
        if (Vertices[i] > JXF_UNPT) j++;
    }
    P_diag_i[row]     = j;
    num_nonzeros_diag = j;

    // allocate memory for P_diag
    P_diag_j    = jxf_MAlloc(num_nonzeros_diag * sizeof(JXF_Int));
    P_diag_data = jxf_MAlloc(num_nonzeros_diag * sizeof(JXF_Real));

    // second run
    for (i = 0, j = 0; i < row; i++) {
        // P_diag_i[i] = j;
        if (Vertices[i] > JXF_UNPT) {
            P_diag_j[j]    = Vertices[i];
            P_diag_data[j] = 1.0;
            j++;
        }
    }

    // output
    tentp = jxf_ParCSRMatrixCreate(comm, global_num_vars, num_coarse_global, row_starts, col_starts, num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);

    jxf_CSRMatrix* tentp_diag     = jxf_ParCSRMatrixDiag(tentp);
    jxf_CSRMatrixI(tentp_diag)    = P_diag_i;
    jxf_CSRMatrixJ(tentp_diag)    = P_diag_j;
    jxf_CSRMatrixData(tentp_diag) = P_diag_data;

    jxf_CSRMatrix* tentp_offd  = jxf_ParCSRMatrixOffd(tentp);
    jxf_CSRMatrixI(tentp_offd) = P_offd_i;
    if (num_cols_offd) {
        jxf_CSRMatrixData(tentp_offd) = P_offd_data;
        jxf_CSRMatrixJ(tentp_offd)    = P_offd_j;
    }

    if (np > 1) {
        jxf_MatvecCommPkgCreate(tentp);
    }

    *P = tentp;

    // DEBUG_PRINTF("rank: %d, End\n", iam);

    // return
    return status;
}

/**
 * \fn JXF_Int jxf_form_tentative_p_par (jxf_ParCSRMatrix* A, JXF_Int* Vertices, JXF_Int* Vert_owner
 *                                             JXF_Int NumAggregates, JXF_Int num_coarse_global,
 *                                             jxf_ParCSRMatrix** P)
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
JXF_Int jxf_form_tentative_p_par(jxf_ParCSRMatrix* A, JXF_Int* Vertices, JXF_Int* Vert_owner, JXF_Int NumAggregates,
                                       JXF_Int num_coarse_global, jxf_ParCSRMatrix** P)
{
    JXF_Int           status = JXF_SUCCESS;
    JXF_Int           i, j;
    jxf_ParCSRMatrix* tentp = NULL;

    MPI_Comm comm = jxf_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

    // DEBUG_PRINTF("rank: %d, Begin\n", iam);

    // jxf_MemoryLocation location = jxf_ParCSRMatrixMemoryLocation(A);
    jxf_CSRMatrix*     A_diag   = jxf_ParCSRMatrixDiag(A);

    // JXF_BigInt* row_starts      = jxf_ParCSRMatrixRowStarts(A);
    JXF_BigInt* row_starts      = jxf_ParCSRMatrixColStarts(A);
    JXF_Int     row             = jxf_CSRMatrixNumRows(A_diag);
    JXF_BigInt  global_num_vars = jxf_ParCSRMatrixGlobalNumRows(A);
    JXF_BigInt  col_starts[2]   = {0, 0};

    // Form col_starts
    JXF_Int* recv_buffer     = jxf_MAlloc(np * sizeof(JXF_Int));
    JXF_Int* recv_buffer_sum = jxf_MAlloc(np * sizeof(JXF_Int));
    MPI_Allgather(&NumAggregates, 1, JXF_MPI_INT, recv_buffer, 1, JXF_MPI_INT, comm);
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
    JXF_Int * P_diag_i = NULL, *P_diag_j = NULL, *P_offd_global_j = NULL;
    JXF_Real* P_diag_data = NULL;
    JXF_Int * P_offd_i = NULL, *P_offd_j = NULL;
    JXF_Real* P_offd_data   = NULL;
    JXF_Int*  col_map_offd  = NULL;
    JXF_Int   num_cols_offd = 0;
    JXF_Int   num_nnz_diag  = 0;
    JXF_Int   num_nnz_offd  = 0;

    P_diag_i = jxf_CAlloc(row + 1, sizeof(JXF_Int));
    P_offd_i = jxf_CAlloc(row + 1, sizeof(JXF_Int)); // P_offd_i = 0

    // first run
    P_diag_i[0] = 0;
    P_offd_i[0] = 0;
#ifdef JXF_USING_OPENMP
#pragma omp parallel for 
#endif
    for (i = 0; i < row; i++) {
        if (Vertices[i] > JXF_UNPT) {
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
    P_diag_j    = jxf_MAlloc(num_nnz_diag * sizeof(JXF_Int));
    P_diag_data = jxf_MAlloc(num_nnz_diag * sizeof(JXF_Real));
    // allocate memory for P_offd
    P_offd_j    = jxf_MAlloc(num_nnz_offd * sizeof(JXF_Int));
    P_offd_data = jxf_MAlloc(num_nnz_offd * sizeof(JXF_Real));

    col_map_offd    = jxf_MAlloc(num_nnz_offd * sizeof(JXF_Int));
    JXF_Int* map = jxf_MAlloc(num_coarse_global * sizeof(JXF_Int));
    for (i = 0; i < num_coarse_global; i++) map[i] = -1;

    // second run
    num_nnz_diag  = 0;
    num_nnz_offd  = 0;
    num_cols_offd = 0;
#ifdef JXF_USING_OPENMP
#pragma omp parallel for 
#endif
    for (i = 0; i < row; i++) {
        if (Vertices[i] > JXF_UNPT) {
            if (Vert_owner[i] == iam) {
                P_diag_j[num_nnz_diag]    = Vertices[i];
                P_diag_data[num_nnz_diag] = 1.0;
                num_nnz_diag++;
            } else {
                JXF_Int global_col_index = Vertices[i] + recv_buffer_sum[Vert_owner[i]];
                // JXF_Int local_col_index  = Vertices[i];
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
        jxf_BigQsort0(col_map_offd, 0, num_cols_offd - 1);
        // jxf_BinarySearch for JXF_BigInt
        for (j = 0; j < num_nnz_offd; ++j) P_offd_j[j] = jxf_BinarySearch(col_map_offd, P_offd_j[j], num_cols_offd);
    }

    // free
    if (num_cols_offd == 0) {
        jxf_TFree(P_offd_j);
        jxf_TFree(P_offd_data);
        jxf_TFree(col_map_offd);
        col_map_offd = NULL;
        num_nnz_offd = 0;
    } else if (num_cols_offd < num_nnz_offd) {
        col_map_offd = jxf_TReAlloc(col_map_offd, JXF_Int, num_cols_offd);
    }
    jxf_Free(recv_buffer);
    jxf_Free(recv_buffer_sum);
    jxf_Free(map);

    //* output
    tentp = jxf_ParCSRMatrixCreate(comm, global_num_vars, num_coarse_global, row_starts, col_starts, num_cols_offd, num_nnz_diag, num_nnz_offd);

    jxf_CSRMatrix* tentp_diag               = jxf_ParCSRMatrixDiag(tentp);
    jxf_CSRMatrixI(tentp_diag)              = P_diag_i;
    jxf_CSRMatrixJ(tentp_diag)              = P_diag_j;
    jxf_CSRMatrixData(tentp_diag)           = P_diag_data;
    // jxf_CSRMatrixMemoryLocation(tentp_diag) = location;

    jxf_CSRMatrix* tentp_offd  = jxf_ParCSRMatrixOffd(tentp);
    jxf_CSRMatrixI(tentp_offd) = P_offd_i;
    if (num_cols_offd) {
        jxf_CSRMatrixJ(tentp_offd)              = P_offd_j;
        jxf_CSRMatrixData(tentp_offd)           = P_offd_data;
        // jxf_CSRMatrixMemoryLocation(tentp_offd) = location;
        jxf_ParCSRMatrixColMapOffd(tentp)       = col_map_offd;
    }

    if (np > 1) {
        jxf_MatvecCommPkgCreate(tentp);
    }

    *P = tentp;

    // DEBUG_PRINTF("rank: %d, End\n", iam);

    // return
    return status;
}

/**
 * \fn JXF_Int jxf_smooth_agg(jxf_ParCSRMatrix* A, jxf_ParCSRMatrix* tentp, jxf_ParCSRMatrix* S,
 *                                  JXF_Int filter, JXF_Real smooth_factor, jxf_ParCSRMatrix** P)
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
JXF_Int jxf_smooth_agg(jxf_ParCSRMatrix* A, jxf_ParCSRMatrix* tentp, jxf_ParCSRMatrix* S, JXF_Int filter, JXF_Real smooth_factor,
                             jxf_ParCSRMatrix** P)
{
    jxf_CSRMatrix* A_diag      = jxf_ParCSRMatrixDiag(A);
    JXF_Int*       A_diag_i    = jxf_CSRMatrixI(A_diag);
    JXF_Int*       A_diag_j    = jxf_CSRMatrixJ(A_diag);
    JXF_Complex*   A_diag_data = jxf_CSRMatrixData(A_diag);
    JXF_Int        row         = jxf_CSRMatrixNumRows(A_diag);

    jxf_CSRMatrix* A_offd      = jxf_ParCSRMatrixOffd(A);
    JXF_Int*       A_offd_i    = jxf_CSRMatrixI(A_offd);
    JXF_Int*       A_offd_j    = jxf_CSRMatrixJ(A_offd);
    JXF_Complex*   A_offd_data = jxf_CSRMatrixData(A_offd);

    JXF_Int           status = JXF_SUCCESS; // return status
    JXF_Int           i, k, j;
    jxf_ParCSRMatrix* J = NULL;

    MPI_Comm comm = jxf_ParCSRMatrixComm(A);
    // MPI_CLAIM(comm);

    // JXF_MemoryLocation location = jxf_ParCSRMatrixMemoryLocation(A);
    JXF_Complex*       diag     = jxf_MAlloc(row * sizeof(JXF_Complex));

    //* Step 1. Form smoother */

    /* Without filter: Using A for damped Jacobian smoother */
    if (filter == jxf_false || S == NULL) {

        // copy structure from A
        J = jxf_ParCSRMatrixClone(A, 0);

        JXF_Complex* J_diag_data = jxf_CSRMatrixData(jxf_ParCSRMatrixDiag(J));
        JXF_Complex* J_offd_data = jxf_CSRMatrixData(jxf_ParCSRMatrixOffd(J));

        // get the diagonal entries of A and check it.
        // if it is too small, use Richardson smoother for the corresponding row
#ifdef JXF_USING_OPENMP
#pragma omp parallel for private(i) 
#endif
        for (i = 0; i < row; ++i) {
            JXF_Complex d = A_diag_data[A_diag_i[i]];
            if (jxf_abs(d) < 1e-6)
                diag[i] = 1.0;
            else
                diag[i] = d;
        }

#ifdef JXF_USING_OPENMP
#pragma omp parallel for private(i, k) 
#endif
        for (i = 0; i < row; i++) {
            // diagonal parts
            JXF_Int ibegin   = A_diag_i[i];
            JXF_Int iend     = A_diag_i[i + 1];
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
        jxf_ParCSRMatrixReorder(S);

        jxf_CSRMatrix* S_diag      = jxf_ParCSRMatrixDiag(S);
        JXF_Int*       S_diag_i    = jxf_CSRMatrixI(S_diag);
        JXF_Int*       S_diag_j    = jxf_CSRMatrixJ(S_diag);
        JXF_Complex*   S_diag_data = jxf_CSRMatrixData(S_diag);

        jxf_CSRMatrix* S_offd      = jxf_ParCSRMatrixOffd(S);
        JXF_Int*       S_offd_i    = jxf_CSRMatrixI(S_offd);
        JXF_Int*       S_offd_j    = jxf_CSRMatrixJ(S_offd);
        JXF_Complex*   S_offd_data = jxf_CSRMatrixData(S_offd);
        JXF_Int        S_offd_nnz  = jxf_CSRMatrixNumNonzeros(S_offd);

        JXF_Complex row_sum_A, row_sum_S;

#ifdef JXF_USING_OPENMP
#pragma omp parallel for private(i, k, row_sum_A, row_sum_S) 
#endif
        for (i = 0; i < row; ++i) {
            row_sum_A = 0.0;
            row_sum_S = 0.0;

            //! A
            // diagonal parts
            JXF_Int ibegin = A_diag_i[i];
            JXF_Int iend   = A_diag_i[i + 1];
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
        J = jxf_ParCSRMatrixClone(S, 0);

        JXF_Complex* J_diag_data = jxf_CSRMatrixData(jxf_ParCSRMatrixDiag(J));
        JXF_Complex* J_offd_data = jxf_CSRMatrixData(jxf_ParCSRMatrixOffd(J));

        // get the diagonal entries of S and check it.
        // if it is too small, use Richardson smoother for the corresponding row
#ifdef JXF_USING_OPENMP
#pragma omp parallel for private(i) 
#endif
        for (i = 0; i < row; ++i) {
            JXF_Complex d = S_diag_data[S_diag_i[i]];
            if (jxf_abs(d) < 1e-6)
                diag[i] = 1.0;
            else
                diag[i] = d;
        }

#ifdef JXF_USING_OPENMP
#pragma omp parallel for private(i, k) 
#endif
        for (i = 0; i < row; i++) {
            // diagonal parts
            JXF_Int ibegin   = S_diag_i[i];
            JXF_Int iend     = S_diag_i[i + 1];
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
    *P = jxf_ParMatmul(J, tentp); // P = J*tenp

    //* free
    jxf_TFree(diag);
    jxf_ParCSRMatrixDestroy(J);

    // return
    return status;
}
