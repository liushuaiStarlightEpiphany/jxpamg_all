//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_aggregation.c --  aggregation algorithms for ParCSR
 *  Date: 2025/10/08
 */ 


#include "jx_pamg.h"

// internal debug parameter
#define AGG_DEBUG 0

/* Declaration of private functions */
static JX_Int jx_nodes_renumbering(jx_ParCSRMatrix* A, JX_Int* riperm, JX_Int* iperm);
static JX_Int jx_form_pairwise(jx_ParCSRMatrix* A, JX_Int num_pair, JX_Real k_tg, JX_Int* NumAggregates, JX_Int** Vertices);
static JX_Int jx_form_pairwise_permutation(jx_ParCSRMatrix* A, JX_Int num_pair, JX_Real k_tg, JX_Int* riperm, JX_Int* iperm,
                                                   JX_Int* NumAggregates, JX_Int** Vertices);

/**
 * \fn JX_Int jx_aggregation_vmb(jx_ParCSRMatrix* A, JX_Int NumLevels,
 *                                       JX_Real strong_coupled, JX_Real tentative_smooth,
 *                                       JX_Int max_aggregation, JX_Int* NumAggregates,
 *                                       JX_Int** Vertices)
 *
 * \brief Form aggregation based on strong coupled neighbors
 *
 * \param A                 The parallel coefficient matrices (Input)
 * \param NumLevels         Current level number (Input)
 * \param strong_coupled    Strongly coupled threshold (Input)
 * \param tentative_smooth  Smoothing factor for tentative prolongation (Input)
 * \param max_aggregation   Max size of aggregations (Input)
 * \param NumAggregates     The number of aggregations on the current processor (Output)
 * \param Vertices          The aggregation of vertices on the current processor (Output)
 * \param S_ptr             Pointer to strongly coupled neighbors (Output)
 *
 * \author Li Zhao
 * \date   10/05/2024
 *
 * \note Setup A, P, PT and levels using the unsmoothed aggregation algorithm;
 *       Refer to P. Vanek, J. Madel and M. Brezina
 *       "Algebraic Multigrid on Unstructured Meshes", 1994
 *
 * \note This algorithm only considers the aggregation between local degrees of freedom,
 *       and does not take into account cross processor scenarios.
 *
 */
JX_Int jx_aggregation_vmb(jx_ParCSRMatrix* A, JX_Int NumLevels, JX_Real strong_coupled, JX_Real tentative_smooth,
                                  JX_Int max_aggregation, JX_Int* NumAggregates, JX_Int** Vertices, jx_ParCSRMatrix** S_ptr)
{
    jx_CSRMatrix* A_diag      = jx_ParCSRMatrixDiag(A);
    JX_Int*       A_diag_i    = jx_CSRMatrixI(A_diag);
    JX_Int*       A_diag_j    = jx_CSRMatrixJ(A_diag);
    JX_Complex*   A_diag_data = jx_CSRMatrixData(A_diag);

    // jx_CSRMatrix* A_offd   = jx_ParCSRMatrixOffd(A);
    // JX_Int*       A_offd_i = jx_CSRMatrixI(A_offd);
    // JX_Int*       A_offd_j = jx_CSRMatrixJ(A_offd);

    JX_Int row        = jx_CSRMatrixNumRows(A_diag);
    JX_Int A_diag_nnz = jx_CSRMatrixNumNonzeros(A_diag);
    // JX_BigInt* row_starts = jx_ParCSRMatrixRowStarts(A);

    // return status
    JX_Int status = JX_SUCCESS;

    // local variables
    JX_Int  num_left = row;
    JX_Int  subset, count;
    JX_Int* vertices;

    JX_Real  strongly_coupled, strongly_coupled2;
    JX_Int   i, j, index, row_start, row_end;
    JX_Int * NIA, *NJA; // strongly coupled neighborhood
    JX_Real* Nval;
    MPI_Comm     comm = jx_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

#if AGG_DEBUG
    DEBUG_PRINTF("rank: %d, Begin\n", iam);
#endif

#if 0
    char matfile[128];
    sprintf(matfile, "out/A_diag_%d.dat", iam);
    jx_CSRMatrixPrint(A_diag, matfile);
    jx_CSRMatrix* A_seq = jx_ParCSRMatrixToCSRMatrixAll(A);
    if (iam == 0) jx_CSRMatrixPrint(A_seq, "out/A_seq.dat");
#endif

    // get diag
    // JX_MemoryLocation location = jx_ParCSRMatrixMemoryLocation(A);
    JX_Complex*       diag_abs = jx_MAlloc(row * sizeof(JX_Complex));
    jx_CSRMatrixExtractDiagonal(A_diag, diag_abs, 1);

    if (tentative_smooth > JX_REAL_EPSILON) {
        strongly_coupled = strong_coupled * jx_pow(0.5, NumLevels); // 0.5^L, L=0,1,2...
    } else {
        strongly_coupled = strong_coupled;
    }
    strongly_coupled2 = jx_pow(strongly_coupled, 2);

#if 0
    DEBUG_PRINTF("rank: %d, NumLevels: %d, strongly_coupled: %f, tentative_smooth: %f\n", iam, NumLevels, strongly_coupled, tentative_smooth);
#endif

    /*------------------------------------------*/
    /*    Form strongly coupled neighborhood    */
    /*------------------------------------------*/
    NIA  = jx_MAlloc((row + 1) * sizeof(JX_Int));
    NJA  = jx_MAlloc(A_diag_nnz * sizeof(JX_Int));
    Nval = jx_MAlloc(A_diag_nnz * sizeof(JX_Real));

    // Note that strongly coupled neighborhood only need to be found in the diagonal matrix part, and non local parts do not need to be considered.
    for (index = i = 0; i < row; ++i) {
        NIA[i]    = index;
        row_start = A_diag_i[i];
        row_end   = A_diag_i[i + 1];
        for (j = row_start; j < row_end; ++j) {
            if ((A_diag_j[j] == i) || (jx_pow(A_diag_data[j], 2) >= strongly_coupled2 * diag_abs[i] * diag_abs[A_diag_j[j]])) {
                NJA[index]  = A_diag_j[j];
                Nval[index] = A_diag_data[j];
                index++;
            }
        }
    }
    NIA[row] = index;
    jx_Free(diag_abs);

    NJA  = jx_TReAlloc(NJA, JX_Int, index);
    Nval = jx_TReAlloc(Nval, JX_Real, index);

    /*------------------------------------------*/
    /*             Initialization               */
    /*------------------------------------------*/
    vertices = jx_MAlloc(row * sizeof(JX_Int));
    for (i = 0; i < row; ++i) vertices[i] = JX_INPT;

    *NumAggregates = 0;

    /*----------------------------------------------------------------------------------------*/
    /* Step 1. Choose non-intersecting strongly coupled neighborhoods as initial aggregation  */
    /* Note that isolation points do not participate in aggregation                           */
    /*----------------------------------------------------------------------------------------*/
    for (i = 0; i < row; ++i) {
        if ((A_diag_i[i + 1] - A_diag_i[i]) == 1) {
            vertices[i] = JX_UNPT; // isolation points
            num_left--;
        } else {
            subset    = jx_true;
            row_start = NIA[i];
            row_end   = NIA[i + 1];
            for (j = row_start; j < row_end; ++j) {
                if (vertices[NJA[j]] >= JX_UNPT) {
                    subset = jx_false;
                    break;
                }
            }
            if (subset) {
                count       = 0;
                vertices[i] = *NumAggregates;
                num_left--;
                count++;
                row_start = NIA[i];
                row_end   = NIA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    if ((NJA[j] != i) && (count < max_aggregation)) {
                        vertices[NJA[j]] = *NumAggregates;
                        num_left--;
                        count++;
                    }
                }
                (*NumAggregates)++;
            }
        }
    }

    /*----------------------------------------------------------------------------------------*/
    /* Step 2. Add point sets to the initial aggregation                                      */
    /*----------------------------------------------------------------------------------------*/
    JX_Int* temp_C       = jx_MAlloc(row * sizeof(JX_Int));
    JX_Int* num_each_agg = jx_CAlloc(*NumAggregates, sizeof(JX_Int)); // initialize num_each_agg[i] = 0

    for (i = 0; i < row; ++i) {
        temp_C[i] = vertices[i];
        if (vertices[i] >= 0) num_each_agg[vertices[i]]++;
    }

    for (i = 0; i < row; ++i) {
        if (vertices[i] < JX_UNPT) {
            row_start = NIA[i];
            row_end   = NIA[i + 1];

            for (j = row_start; j < row_end; ++j) {
                if (temp_C[NJA[j]] > JX_UNPT && num_each_agg[temp_C[NJA[j]]] < max_aggregation) {
                    vertices[i] = temp_C[NJA[j]];
                    num_left--;
                    num_each_agg[temp_C[NJA[j]]]++;
                    break;
                }
            }
        }
    }

    /*----------------------------------------------------------------------------------------*/
    /* Step 3. tackle the remaining points                                                    */
    /*----------------------------------------------------------------------------------------*/
    while (num_left > 0) {
        for (i = 0; i < row; ++i) {
            if (vertices[i] < JX_UNPT) {
                count       = 0;
                vertices[i] = *NumAggregates;
                num_left--;
                count++;
                row_start = NIA[i];
                row_end   = NIA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    if ((NJA[j] != i) && (vertices[NJA[j]] < JX_UNPT) && (count < max_aggregation)) {
                        vertices[NJA[j]] = *NumAggregates;
                        num_left--;
                        count++;
                    }
                }
                (*NumAggregates)++;
            }
        }
    }

#if 0
    DEBUG_PRINTF("rank: %d, NumVertices: %d, NumAggregates: %d\n", iam, row, *NumAggregates);
    if (*NumAggregates == 0) {
        for (i = 0; i < row; ++i) jx_printf("vertices[%d] = %d\n", i, vertices[i]);
    }
#endif

    jx_ParCSRMatrix* S           = jx_ParCSRMatrixCreate(comm, jx_ParCSRMatrixGlobalNumRows(A), jx_ParCSRMatrixGlobalNumCols(A),
                                                                 jx_ParCSRMatrixRowStarts(A), jx_ParCSRMatrixColStarts(A), 0, index, 0);
    jx_CSRMatrix*    S_diag      = jx_ParCSRMatrixDiag(S);
    JX_Int*          S_diag_i    = jx_CSRMatrixI(S_diag);
    JX_Int*          S_diag_j    = jx_CSRMatrixJ(S_diag);
    JX_Complex*      S_diag_data = jx_CSRMatrixData(S_diag);

    jx_Free(S_diag_i);
    jx_Free(S_diag_j);
    jx_Free(S_diag_data);

    jx_CSRMatrixI(S_diag)    = NIA;
    jx_CSRMatrixJ(S_diag)    = NJA;
    jx_CSRMatrixData(S_diag) = Nval;

    jx_CSRMatrix* S_offd  = jx_ParCSRMatrixOffd(S);
    jx_CSRMatrixI(S_offd) = jx_CAlloc(row + 1, sizeof(JX_Int));

    // END:
    *Vertices = vertices;
    *S_ptr    = S;

    // jx_Free(NIA);
    // NIA = NULL;
    // jx_Free(NJA);
    // NJA = NULL;
    // jx_Free(Nval);
    // Nval = NULL;
    jx_Free(temp_C);
    temp_C = NULL;
    jx_Free(num_each_agg);
    num_each_agg = NULL;

#if AGG_DEBUG
    DEBUG_PRINTF("rank: %d, End\n", iam);
#endif

    return status;
}

/**
 * \fn JX_Int jx_aggregation_vmb_par(jx_ParCSRMatrix* A, JX_Int NumLevels,
 *                                       JX_Real strong_coupled, JX_Real tentative_smooth,
 *                                       JX_Int max_aggregation, JX_Int* NumAggregates,
 *                                       JX_Int** Vertices, JX_Int** Vertices_owner, jx_ParCSRMatrix** S_ptr)
 *
 * \brief Form aggregation based on strong coupled neighbors
 *
 * \param A                 The parallel coefficient matrices (Input)
 * \param NumLevels         Current level number (Input)
 * \param strong_coupled    Strongly coupled threshold (Input)
 * \param tentative_smooth  Smoothing factor for tentative prolongation (Input)
 * \param max_aggregation   Max size of aggregations (Input)
 * \param NumAggregates     The number of aggregations on the current processor (Output)
 * \param Vertices          The aggregation number of vertices on the current processor (Output)
 * \param Vertices_owner    The aggregation number of vertices on the current processor belongs to the process (Output)
 * \param S_ptr             Pointer to strongly coupled neighbors (Output)
 *
 * \author Li Zhao
 * \date   12/08/2024
 *
 * \ref 1、Setup A, P, PT and levels using the unsmoothed aggregation algorithm;
 *       Refer to P. Vanek, J. Madel and M. Brezina
 *       "Algebraic Multigrid on Unstructured Meshes", 1994
 *
 *       2、R. S. Tuminaro and C. Tong, "Parallel Smoothed Aggregation Multigrid : Aggregation Strategies on Massively Parallel Machines,"
 *       SC '00: Proceedings of the 2000 ACM/IEEE Conference on Supercomputing, Dallas, TX, USA, 2000, pp. 5-5, doi: 10.1109/SC.2000.10008.
 *
 *       3、D. Demidov. AMGCL software: amgcl\amgcl\mpi\coarsening\pmis.hpp, The source code is available at https://github.com/ddemidov/amgcl.
 *
 * \note This algorithm accounts for the aggregation of local and non-local degrees of freedom
 *       based on the parallel maximum independent subset (PMIS) approach.
 *
 */
JX_Int jx_aggregation_vmb_par(jx_ParCSRMatrix* A, JX_Int NumLevels, JX_Real strong_coupled, JX_Real tentative_smooth,
                                      JX_Int max_aggregation, JX_Int* NumAggregates, JX_Int** Vertices, JX_Int** Vertices_owner,
                                      jx_ParCSRMatrix** S_ptr)
{
    jx_CSRMatrix* A_diag      = jx_ParCSRMatrixDiag(A);
    JX_Int*       A_diag_i    = jx_CSRMatrixI(A_diag);
    JX_Int*       A_diag_j    = jx_CSRMatrixJ(A_diag);
    JX_Complex*   A_diag_data = jx_CSRMatrixData(A_diag);

    jx_CSRMatrix* A_offd          = jx_ParCSRMatrixOffd(A);
    JX_Int*       A_offd_i        = jx_CSRMatrixI(A_offd);
    JX_Int*       A_offd_j        = jx_CSRMatrixJ(A_offd);
    JX_Complex*   A_offd_data     = jx_CSRMatrixData(A_offd);
    JX_Int        A_num_cols_offd = jx_CSRMatrixNumCols(A_offd);
    JX_BigInt*    A_col_map_offd  = jx_ParCSRMatrixColMapOffd(A);

    JX_Int            row        = jx_CSRMatrixNumRows(A_diag);
    JX_Int            A_diag_nnz = jx_CSRMatrixNumNonzeros(A_diag);
    JX_Int            A_offd_nnz = jx_CSRMatrixNumNonzeros(A_offd);
    JX_BigInt*        row_starts = jx_ParCSRMatrixRowStarts(A);
    JX_Int            global_row = jx_ParCSRMatrixGlobalNumRows(A);
    JX_BigInt         row_first  = jx_ParCSRMatrixFirstRowIndex(A);
    // JX_MemoryLocation location   = jx_ParCSRMatrixMemoryLocation(A);
    jx_ParCSRCommPkg* A_comm_pkg = jx_ParCSRMatrixCommPkg(A);
    if (!A_comm_pkg) {
        jx_MatvecCommPkgCreate(A);
        A_comm_pkg = jx_ParCSRMatrixCommPkg(A);
    }

    // return status
    JX_Int status   = JX_SUCCESS;
    JX_Int agg_iter = 0, agg_maxit = 100;

    JX_Real strongly_coupled, strongly_coupled2;
    JX_Int  i, j, k, index, index_offd, row_start, row_end;
    MPI_Comm    comm = jx_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

#if AGG_DEBUG
    IAM_0 { DEBUG_PRINTF("rank: %d, Begin\n", iam); }
#endif

    if (tentative_smooth > JX_REAL_EPSILON) {
        strongly_coupled = strong_coupled * jx_pow(0.5, NumLevels); // 0.5^L, L=0,1,2...
    } else {
        strongly_coupled = strong_coupled;
    }
    strongly_coupled2 = jx_pow(strongly_coupled, 2);

    /*------------------------------------------*/
    /*  1. Form strongly coupled neighborhood   */
    /*------------------------------------------*/
    JX_Int*     S_diag_i    = jx_MAlloc((row + 1) * sizeof(JX_Int));
    JX_Int*     S_diag_j    = jx_MAlloc(A_diag_nnz * sizeof(JX_Int));
    JX_Complex* S_diag_data = jx_MAlloc(A_diag_nnz * sizeof(JX_Complex));
    JX_Int*     S_offd_i    = jx_MAlloc((row + 1) * sizeof(JX_Int));
    JX_Int*     S_offd_j    = jx_MAlloc(A_offd_nnz * sizeof(JX_Int));
    // JX_Int*     S_offd_global_j = jx_MAlloc(A_offd_nnz * sizeof(JX_Int));
    JX_Complex* S_offd_data = jx_MAlloc(A_offd_nnz * sizeof(JX_Complex));

    /*
     * S1.1 Obtain diag
     */
    // get diag entries of diag matrix
    JX_Complex* diag_abs = jx_MAlloc(row * sizeof(JX_Complex));
    jx_CSRMatrixExtractDiagonal(A_diag, diag_abs, 1);

    // get diag entries of offd matrix
    JX_Complex* remote_diag_abs = jx_MAlloc(A_num_cols_offd * sizeof(JX_Complex));
    jx_ParCSRCommPkgGetRemoteData_Complex(A_comm_pkg, diag_abs, remote_diag_abs);

    /*
     * S1.2 Compute strongly coupled neighborhood matrix
     */
    index                        = 0;
    index_offd                   = 0;
    JX_Int  global_col_index = 0;
    JX_Int  S_num_cols_offd  = 0;
    JX_Int* map              = jx_CTAlloc(JX_Int, A_num_cols_offd);
    JX_Int* S_col_map_offd   = jx_CTAlloc(JX_Int, A_num_cols_offd);
    for (i = 0; i < A_num_cols_offd; i++) map[i] = -1;

    for (i = 0; i < row; ++i) {
        // diagonal matrix part
        S_diag_i[i] = index;
        row_start   = A_diag_i[i];
        row_end     = A_diag_i[i + 1];
        for (j = row_start; j < row_end; ++j) {
            if ((A_diag_j[j] == i) || (jx_pow(A_diag_data[j], 2) > strongly_coupled2 * diag_abs[i] * diag_abs[A_diag_j[j]])) {
                S_diag_j[index]    = A_diag_j[j];
                S_diag_data[index] = A_diag_data[j];
                index++;
            }
        }
        // non-diagonal matrix part
        S_offd_i[i] = index_offd;
        row_start   = A_offd_i[i];
        row_end     = A_offd_i[i + 1];
        for (j = row_start; j < row_end; ++j) {
            global_col_index = A_col_map_offd[A_offd_j[j]];
            // aij^2 >= strongly_coupled2 * |aii| * |ajj|, where ajj is remote dof
            if (jx_pow(A_offd_data[j], 2) > strongly_coupled2 * diag_abs[i] * remote_diag_abs[A_offd_j[j]]) {
                if (map[A_offd_j[j]] < 0) {
                    map[A_offd_j[j]] = S_num_cols_offd;
                    // S_offd_j[index_offd]            = S_num_cols_offd;
                    S_offd_j[index_offd]            = global_col_index;
                    S_col_map_offd[S_num_cols_offd] = global_col_index;
                    S_num_cols_offd++;
                } else {
                    // S_offd_j[index_offd] = map[A_offd_j[j]];
                    S_offd_j[index_offd] = global_col_index;
                }
                S_offd_data[index_offd] = A_offd_data[j];
                index_offd++;
            }
        }
    }
    S_diag_i[row] = index;
    S_offd_i[row] = index_offd;
    // DEBUG_PRINTF("rank: %d, index %d, index_offd %d\n", iam, index, index_offd);

    //! Note that S_col_map_offd must be in increasing order
    {
        jx_BigQsort0(S_col_map_offd, 0, S_num_cols_offd - 1);
        for (j = 0; j < index_offd; ++j)
            S_offd_j[j] = jx_BinarySearch(S_col_map_offd, S_offd_j[j], S_num_cols_offd); // jx_BinarySearch for JX_BigInt
        // jx_TFree(S_offd_global_j);                                                    // free S_offd_global_j
    }

    // if (S_num_cols_offd == 0) {
    //     jx_TFree(S_col_map_offd);
    //     S_col_map_offd = NULL;
    // } else if (S_num_cols_offd < A_num_cols_offd) {
    //     S_col_map_offd = jx_TReAlloc(S_col_map_offd, JX_Int, S_num_cols_offd);
    // }
    S_col_map_offd = jx_TReAlloc(S_col_map_offd, JX_Int, S_num_cols_offd);
    S_diag_j       = jx_TReAlloc(S_diag_j, JX_Int, index);
    S_diag_data    = jx_TReAlloc(S_diag_data, JX_Complex, index);
    S_offd_j       = jx_TReAlloc(S_offd_j, JX_Int, index_offd);
    S_offd_data    = jx_TReAlloc(S_offd_data, JX_Complex, index_offd);

    jx_Free(diag_abs);
    jx_Free(remote_diag_abs);
    jx_Free(map);

    // Create S
    jx_CSRMatrix* S_diag               = jx_CSRMatrixCreate(row, row, index);
    jx_CSRMatrixI(S_diag)              = S_diag_i;
    jx_CSRMatrixJ(S_diag)              = S_diag_j;
    jx_CSRMatrixData(S_diag)           = S_diag_data;
    // jx_CSRMatrixMemoryLocation(S_diag) = location;

    jx_CSRMatrix* S_offd  = jx_CSRMatrixCreate(row, S_num_cols_offd, index_offd);
    jx_CSRMatrixI(S_offd) = S_offd_i;
    if (S_num_cols_offd) {
        jx_CSRMatrixJ(S_offd)              = S_offd_j;
        jx_CSRMatrixData(S_offd)           = S_offd_data;
        // jx_CSRMatrixMemoryLocation(S_offd) = location;
    }

    jx_ParCSRMatrix* S = jx_ParCSRMatrixCreate(comm, global_row, global_row, row_starts, row_starts, S_num_cols_offd, index, index_offd);
    jx_CSRMatrixDestroy(jx_ParCSRMatrixDiag(S));
    jx_CSRMatrixDestroy(jx_ParCSRMatrixOffd(S));
    jx_ParCSRMatrixDiag(S)       = S_diag;
    jx_ParCSRMatrixOffd(S)       = S_offd;
    jx_ParCSRMatrixColMapOffd(S) = S_col_map_offd;

    /*-------------------------------------------------------*/
    /*  2. Get symbolic square of the connectivity matrix.   */
    /*-------------------------------------------------------*/
#if AGG_DEBUG > 1
    JX_Real time_start, time_end, time;
    time_start = jx_MPI_Wtime();
#endif

    // jx_ParCSRMatrix* S2 = jx_ParTMatmul(S, S); // S2 = S' * S
    // jx_ParCSRMatrix* S2 = jx_ParMatmul(S, S); // S2 = S * S
    jx_ParCSRMatrix* S2 = jx_ParMatmul_NozeroStruct(S, S);

#if AGG_DEBUG > 1
    time_end = jx_MPI_Wtime();
    time     = time_end - time_start;
    if (iam == 0) {
        DEBUG_PRINTF("rank: %d, S2 = S * S time:  %.4f s\n", iam, time);
    }
#endif

#if AGG_DEBUG && 0
    // jx_ParCSRMatrixPrint(A, "A.dat");
    // jx_ParCSRMatrixPrint(S, "S.dat");
    // jx_ParCSRMatrixPrint(S2, "S2.dat");
#endif

    /*-------------------------------------------------------*/
    /*  3. Apply PMIS algorithm to the symbolic square.      */
    /*-------------------------------------------------------*/
    JX_Int        n_undone  = 0;
    jx_CSRMatrix* S2_diag   = jx_ParCSRMatrixDiag(S2);
    JX_Int*       S2_diag_i = jx_CSRMatrixI(S2_diag);
    JX_Int*       S2_diag_j = jx_CSRMatrixJ(S2_diag);

    jx_CSRMatrix* S2_offd   = jx_ParCSRMatrixOffd(S2);
    JX_Int*       S2_offd_i = jx_CSRMatrixI(S2_offd);
    JX_Int*       S2_offd_j = jx_CSRMatrixJ(S2_offd);

    JX_Int  S2_num_cols_offd = jx_CSRMatrixNumCols(S2_offd);
    JX_Int* S2_col_map_offd  = jx_ParCSRMatrixColMapOffd(S2);

    jx_ParCSRCommPkg* S2_comm_pkg = jx_ParCSRMatrixCommPkg(S2);
    if (!S2_comm_pkg) {
        jx_MatvecCommPkgCreate(S2);
        S2_comm_pkg = jx_ParCSRMatrixCommPkg(S2);
    }

#if AGG_DEBUG > 1
    DEBUG_PRINTF("rank: %d, row %d, row_starts = {%d, %d}, S2_num_cols_offd: %d\n", iam, row, row_starts[0], row_starts[1], S2_num_cols_offd);
#endif

    JX_Int* loc_state = jx_MAlloc(row * sizeof(JX_Int));
    JX_Int* loc_owner = jx_MAlloc(row * sizeof(JX_Int));

    JX_Int num_sends     = jx_ParCSRCommPkgNumSends(S2_comm_pkg); // Number of target processes sent
    JX_Int num_recvs     = jx_ParCSRCommPkgNumRecvs(S2_comm_pkg); // Number of target processes received
    JX_Int sends_vec_len = jx_ParCSRCommPkgSendMapStart(S2_comm_pkg, num_sends) - jx_ParCSRCommPkgSendMapStart(S2_comm_pkg, 0);
    JX_Int recvs_vec_len = jx_ParCSRCommPkgRecvVecStart(S2_comm_pkg, num_recvs) - jx_ParCSRCommPkgRecvVecStart(S2_comm_pkg, 0);
    // DEBUG_PRINTF("rank %d, num_recvs %d, recvs_vec_len %d, num_sends %d, sends_vec_len %d\n", iam, num_recvs, recvs_vec_len, num_sends, sends_vec_len);

    JX_Int* rem_state = jx_MAlloc(recvs_vec_len * sizeof(JX_Int));
    JX_Int* rem_owner = jx_MAlloc(recvs_vec_len * sizeof(JX_Int));

    for (i = 0; i < recvs_vec_len; ++i) {
        rem_state[i] = JX_UNPT;
        rem_owner[i] = -1;
    }

    /*
     * S3.1 Remove lonely/isolated nodes in the sense of S2.
     * Note that isolation points do not participate in aggregation
     */
    JX_Int max_nnz_per_row = 0; // Maximum non-zero element number in per row

#ifdef JX_USING_OPENMP
#pragma omp parallel for  reduction(+ : n_undone)
#endif
    for (i = 0; i < row; ++i) {
        JX_Int wl = S_diag_i[i + 1] - S_diag_i[i];
        JX_Int wr = S2_offd_i[i + 1] - S2_offd_i[i];

        if (wl + wr == 1) {
            loc_state[i] = JX_G0PT; // -5
            ++n_undone;
        } else {
            loc_state[i] = JX_UNPT; // -1
        }
        max_nnz_per_row = jx_max(max_nnz_per_row, wl + wr); // Maximum non-zero element number in per row

        loc_owner[i] = -1;
    }

    n_undone = row - n_undone;

    /*
     * S3.2 Exchange state
     */
    jx_ParCSRCommPkgGetRemoteData_Int(S2_comm_pkg, loc_state, rem_state);

#if AGG_DEBUG > 2
    for (i = 0; i < recvs_vec_len; ++i) printf("after rank %d, rem_state[%d] = %d\n", iam, i, rem_state[i]);
#endif

    /*
     * S3.3 Start aggregation
     */
    JX_Int  send_pts_offset = 10, send_pts_max_len = send_pts_offset * recvs_vec_len;
    JX_Int* send_pts_num   = jx_CTAlloc(JX_Int, num_recvs);
    JX_Int* send_pts_start = jx_CTAlloc(JX_Int, num_recvs + 1);
    JX_Int* send_pts = jx_CTAlloc(JX_Int, send_pts_max_len); // max length,  at least twice, but when it equals twice, an error occurs
    JX_Int* map_remote_rank2id = jx_CTAlloc(JX_Int, np);     // waste some memory here
    JX_Int* recv_pts           = NULL;

    MPI_Request* send_cnt_req = jx_CTAlloc(MPI_Request, num_recvs);
    MPI_Request* send_pts_req = jx_CTAlloc(MPI_Request, num_recvs);

    JX_Int  naggr = 0, n_undone_global = -1;
    JX_Int  ibegin, iend, id = 0, c, c_remote_rank, c_loc_indx;
    JX_Int  nbr_num         = 0;
    JX_Int* nbr_arr         = jx_MAlloc(max_nnz_per_row * sizeof(JX_Int));
    JX_Int* S2_remote_ranks = jx_MAlloc(recvs_vec_len * sizeof(JX_Int));

    // Auxiliary variables
    JX_Int* S_offd_col_to_S2_offd_col = jx_MAlloc(S_num_cols_offd * sizeof(JX_Int));

    // max length (corresponding to send_pts),  at least twice, but when it equals twice, an error occurs, zhaoli, 2024.12.08
    for (i = 0; i < num_recvs + 1; ++i) send_pts_start[i] = jx_ParCSRCommPkgRecvVecStart(S2_comm_pkg, i) * send_pts_offset;
    // for (i = 0; i < np; ++i) map_remote_rank2id[i] = -1; // debug
    for (i = 0; i < num_recvs; ++i) map_remote_rank2id[jx_ParCSRCommPkgRecvProc(S2_comm_pkg, i)] = i;

    {
        JX_Int* map_tmp = jx_MAlloc(global_row * sizeof(JX_Int));
        // for (i = 0; i < global_row; ++i) map_tmp[i] = -1; // debug
        for (i = 0; i < S2_num_cols_offd; ++i) map_tmp[S2_col_map_offd[i]] = i;
        for (i = 0; i < S_num_cols_offd; ++i) S_offd_col_to_S2_offd_col[i] = map_tmp[S_col_map_offd[i]];
        jx_Free(map_tmp);
    }

    jx_ParCSRCommPkgGetRemoteRankArray(S2_comm_pkg, S2_remote_ranks);
    // jx_ParCSRCommPkgGetRemoteRankArray(S_comm_pkg, S_remote_ranks);

    while (jx_true) {
        for (i = 0; i < num_recvs; ++i) send_pts_num[i] = 0;

        if (n_undone) {
            for (i = 0; i < row; ++i) {
                if (loc_state[i] != JX_UNPT) continue;

                // Tackle boundary points and inner points
                if (S2_offd_i[i + 1] > S2_offd_i[i]) { // Boundary points

                    jx_bool selectable = jx_true;
                    ibegin                 = S2_offd_i[i];
                    iend                   = S2_offd_i[i + 1];
                    for (j = ibegin; j < iend; ++j) {
                        c             = S2_offd_j[j];
                        c_remote_rank = S2_remote_ranks[c];
#if AGG_DEBUG > 1
                        if (c >= recvs_vec_len || c < 0) {
                            WARN_PRINTF("rem_state: rank[%d], c = %d out of range [0, %d], line: %d\n", iam, c, recvs_vec_len, __LINE__); // DEBUG
                        }
#endif
                        if (rem_state[c] == JX_UNPT && c_remote_rank > iam) {
                            selectable = jx_false;
                            break;
                        }
                    }

                    if (!selectable) continue;

                    id           = naggr++;
                    loc_owner[i] = iam;
                    loc_state[i] = id;
                    --n_undone;

                    // S gives immediate neighbors
                    ibegin = S_diag_i[i];
                    iend   = S_diag_i[i + 1];
                    for (j = ibegin; j < iend; ++j) {
                        c = S_diag_j[j];
                        if (c != i) {
                            if (loc_state[c] == JX_UNPT) --n_undone;
                            // TODO: It is possible to change the aggregation number of point C here, as point C may already be in another aggregation
#if AGG_DEBUG > 1
                            if (c >= row || c < 0) {
                                WARN_PRINTF("loc_state: rank[%d], c = %d out of range [0, %d], line: %d\n", iam, c, row, __LINE__); // DEBUG
                            }
#endif
                            loc_owner[c] = iam;
                            loc_state[c] = id;
                        }
                    }

                    ibegin = S_offd_i[i];
                    iend   = S_offd_i[i + 1];
                    for (j = ibegin; j < iend; ++j) {
                        // printf("S: rank %d, ibegin %d, iend %d, j %d\n", iam, ibegin, iend, j);
                        // Note that the 'off' entries in S cannot be replaced with the 'remote_rank' entries in S2!!!
                        // Local column numbers may not be the same, global column numbers are the same
                        c             = S_offd_col_to_S2_offd_col[S_offd_j[j]]; // S_offd_j[j] -> S2_offd_j[j]
                        c_remote_rank = S2_remote_ranks[c];
#if AGG_DEBUG > 1
                        if (c >= recvs_vec_len || c < 0) {
                            WARN_PRINTF("rem_state: rank[%d], c = %d out of range [0, %d], line: %d\n", iam, c, recvs_vec_len, __LINE__); // DEBUG
                        }
#endif
                        rem_state[c] = id;
                        c_loc_indx   = map_remote_rank2id[c_remote_rank];

#if AGG_DEBUG && 0
                        if (c_remote_rank == 2) {
                            printf("line %d: iam %d, c_remote_rank %d, c %d, id %d, c_loc_indx %d [send pro %d], S2_col_map_offd %d S_col_map_offd %d\n",
                                   __LINE__, iam, c_remote_rank, c, id, c_loc_indx, jx_ParCSRCommPkgRecvProc(S2_comm_pkg, c_loc_indx), S2_col_map_offd[c],
                                   S_col_map_offd[S_offd_j[j]]);
                        }
#endif
                        // cord c and id
                        JX_Int ID = send_pts_start[c_loc_indx] + send_pts_num[c_loc_indx];
                        // check, zhaoli, 2024.12.14
                        if (ID >= send_pts_max_len - 1) {
                            WARN_PRINTF("send_pts: rank[%d], ID = %d out of range [0, %d), line: %d\n", iam, ID + 1, send_pts_max_len, __LINE__); // DEBUG
                        }

                        send_pts[ID]     = S2_col_map_offd[c]; // here is global column index
                        send_pts[ID + 1] = id;
                        send_pts_num[c_loc_indx] += 2;
#if AGG_DEBUG > 2
                        printf("S: rank %d, c_remote_rank %d, c_loc_indx %d, S2_col_map_offd[%d] %d, id %d\n", iam, c_remote_rank, c_loc_indx, c,
                               S2_col_map_offd[c], id);
#endif
                    }

                    // S2 gives removed neighbors
                    ibegin = S2_diag_i[i];
                    iend   = S2_diag_i[i + 1];
                    for (j = ibegin; j < iend; ++j) {
                        c = S2_diag_j[j];
                        if (c != i && loc_state[c] == JX_UNPT) {
#if AGG_DEBUG > 1
                            if (c >= row || c < 0) {
                                WARN_PRINTF("loc_state: rank[%d], c = %d out of range [0, %d], line: %d\n", iam, c, row, __LINE__); // DEBUG
                            }
#endif
                            loc_owner[c] = iam;
                            loc_state[c] = id;
                            --n_undone;
                        }
                    }

                    ibegin = S2_offd_i[i];
                    iend   = S2_offd_i[i + 1];
                    for (j = ibegin; j < iend; ++j) {
                        c             = S2_offd_j[j];
                        c_remote_rank = S2_remote_ranks[c];
#if AGG_DEBUG > 1
                        if (c >= recvs_vec_len || c < 0) {
                            WARN_PRINTF("rem_state: rank[%d], c = %d out of range [0, %d], line: %d\n", iam, c, recvs_vec_len, __LINE__); // DEBUG
                        }
#endif
                        if (rem_state[c] == JX_UNPT) {
                            rem_state[c] = id;

                            c_loc_indx = map_remote_rank2id[c_remote_rank];
#if AGG_DEBUG && 0
                            if (c_remote_rank == 2) {
                                printf("line %d: iam %d, c_remote_rank %d, c %d, id %d, c_loc_indx %d [send pro %d], S2_col_map_offd %d\n", __LINE__, iam,
                                       c_remote_rank, c, id, c_loc_indx, jx_ParCSRCommPkgRecvProc(S2_comm_pkg, c_loc_indx), S2_col_map_offd[c]);
                            }
#endif
                            // cord c and id
                            JX_Int ID = send_pts_start[c_loc_indx] + send_pts_num[c_loc_indx];
                            // check, zhaoli, 2024.12.14
                            if (ID >= send_pts_max_len - 1) {
                                WARN_PRINTF("send_pts: rank[%d], ID = %d out of range [0, %d), line: %d\n", iam, ID + 1, send_pts_max_len, __LINE__); // DEBUG
                            }

                            send_pts[ID]     = S2_col_map_offd[c]; // here is global column index
                            send_pts[ID + 1] = id;
                            send_pts_num[c_loc_indx] += 2;
#if AGG_DEBUG > 2
                            printf("S2: rank %d, c_remote_rank %d, c_loc_indx %d, S2_col_map_offd[%d] %d, id %d\n", iam, c_remote_rank, c_loc_indx, c,
                                   S2_col_map_offd[c], id);
#endif
                        }
                    }

                } else {
                    // Inner points
                    id           = naggr++;
                    loc_owner[i] = iam;
                    loc_state[i] = id;
                    n_undone--;

                    nbr_num = 0;
                    ibegin  = S_diag_i[i];
                    iend    = S_diag_i[i + 1];
                    for (j = ibegin; j < iend; ++j) {
                        c = S_diag_j[j];

                        if (c != i && loc_state[c] != JX_G0PT) {
                            if (loc_state[c] == JX_UNPT) --n_undone;
#if AGG_DEBUG > 1
                            if (c >= row || c < 0) {
                                WARN_PRINTF("loc_state: rank[%d], c = %d out of range [0, %d], line: %d\n", iam, c, row, __LINE__); // DEBUG
                            }
#endif
                            // TODO: It is possible to change the aggregation number of point C here, as point C may already be in another aggregation
                            loc_owner[c]       = iam;
                            loc_state[c]       = id;
                            nbr_arr[nbr_num++] = c;
                        }
                    }

                    // Neighbors' neighbors will not be involved in ghost points
                    for (k = 0; k < nbr_num; k++) // 第 i 号点的邻居 k
                    {
                        ibegin = S_diag_i[k];
                        iend   = S_diag_i[k + 1];
                        for (j = ibegin; j < iend; ++j) {
                            c = S_diag_j[j];
                            if (c != k && loc_state[c] == JX_UNPT) {
#if AGG_DEBUG > 1
                                if (c >= row || c < 0) {
                                    WARN_PRINTF("loc_state: rank[%d], c = %d out of range [0, %d], line: %d\n", iam, c, row, __LINE__); // DEBUG
                                }
#endif
                                loc_owner[c] = iam;
                                loc_state[c] = id;
                                --n_undone;
                            }
                        }
                    }
                }
            } // end for i
        } // end if n_undone

        // Send
        for (i = 0; i < num_recvs; ++i) {
            JX_Int npts = send_pts_num[i];
            JX_Int ip   = jx_ParCSRCommPkgRecvProc(S2_comm_pkg, i);
            jx_MPI_Isend(&npts, 1, JX_MPI_INT, ip, 0, comm, &send_cnt_req[i]);

            if (!npts) continue;

            jx_MPI_Isend(&send_pts[send_pts_start[i]], npts, JX_MPI_INT, ip, 1, comm, &send_pts_req[i]);
        }

        // Recv
        for (i = 0; i < num_sends; ++i) {
            JX_Int npts;
            JX_Int ip = jx_ParCSRCommPkgSendProc(S2_comm_pkg, i);
            jx_MPI_Recv(&npts, 1, JX_MPI_INT, ip, 0, comm, MPI_STATUS_IGNORE);

            if (!npts) continue;

            recv_pts = jx_CTAlloc(JX_Int, npts);
            MPI_Recv(&recv_pts[0], npts, JX_MPI_INT, ip, 1, comm, MPI_STATUS_IGNORE);

            // Update the loc_state of the current process
            for (k = 0; k < npts; k += 2) {
                c  = recv_pts[k] - jx_ParCSRMatrixFirstRowIndex(S2);
                id = recv_pts[k + 1];
#if AGG_DEBUG > 1
                // DEBUG_PRINTF("rank: %d, c %d, ip %d, id %d\n", iam, c, ip, id);
                if (c >= row || c < 0) {
                    WARN_PRINTF("loc_state: rank[%d], c = %d out of range [0, %d], line: %d, %d - %d - %d\n", iam, c, row, __LINE__, recv_pts[k],
                                jx_ParCSRMatrixFirstRowIndex(S2), id); // DEBUG
                }
#endif
                if (loc_state[c] == JX_UNPT) --n_undone;

                loc_owner[c] = ip;
                loc_state[c] = id;
            }
            jx_Free(recv_pts);
        }

#if AGG_DEBUG > 1
        for (i = 0; i < row; i++) {
            printf("rank[%d], loc_state[%d] = %d, loc_owner[%d] = %d\n", iam, i, loc_state[i], i, loc_owner[i]);
        }
#endif

        for (i = 0; i < num_recvs; ++i) {
            JX_Int npts = send_pts_num[i];
            MPI_Wait(&send_cnt_req[i], MPI_STATUS_IGNORE);

            if (!npts) continue;
            MPI_Wait(&send_pts_req[i], MPI_STATUS_IGNORE);
        }

        // Exchange state
        jx_ParCSRCommPkgGetRemoteData_Int(S2_comm_pkg, loc_state, rem_state);

        // check
        jx_MPI_Allreduce(&n_undone, &n_undone_global, 1, JX_MPI_INT, MPI_SUM, comm);
#if AGG_DEBUG > 1
        DEBUG_PRINTF("rank: %d, n_undone_global %d\n", iam, n_undone_global);
#endif
        agg_iter++;
        if (agg_iter > agg_maxit && iam == 0) {
            WARN_PRINTF("The maximum number (%d) of aggregating has been reached, and the program has been forcibly stopped!\n", agg_maxit);
            exit(0);
        }

        if (0 == n_undone_global) break;
    } // end while

    /*
     * S3.4 Some of the aggregates could potentially vanish during expansion step above.
     * We need to exclude those and renumber the rest.
     */
    jx_ParCSRCommPkgGetRemoteData_Int(S2_comm_pkg, loc_owner, rem_owner);

    JX_Int* new_id = jx_CTAlloc(JX_Int, naggr + 1);
    for (i = 0; i < row; ++i) {
        if (loc_owner[i] == iam && loc_state[i] >= 0) new_id[loc_state[i] + 1] = 1;
    }

    for (i = 0; i < recvs_vec_len; ++i) {
        if (rem_owner[i] == iam && rem_state[i] >= 0) new_id[rem_state[i] + 1] = 1;
    }

    // A part sum of a sequence
    for (i = 1; i < naggr + 1; ++i) new_id[i] += new_id[i - 1];

    JX_Int diff = naggr - new_id[naggr], diff_global = 0;
    jx_MPI_Allreduce(&diff, &diff_global, 1, JX_MPI_INT, MPI_SUM, comm);

    if (diff_global > 0) {
        naggr = new_id[naggr];

        for (i = 0; i < row; ++i) {
            if (loc_owner[i] == iam && loc_state[i] >= 0) {
                loc_state[i] = new_id[loc_state[i]]; // Mapping from old number to new number
            }
        }

        for (i = 0; i < num_recvs; ++i) send_pts_num[i] = 0;

        for (c = 0; c < recvs_vec_len; ++c) {
            c_remote_rank = S2_remote_ranks[c];
            c_loc_indx    = map_remote_rank2id[c_remote_rank];

            if (rem_owner[c] == iam && rem_state[c] >= 0) {
                // record c and id
                send_pts[send_pts_start[c_loc_indx] + send_pts_num[c_loc_indx]++] = S2_col_map_offd[c]; // here is global column index
                send_pts[send_pts_start[c_loc_indx] + send_pts_num[c_loc_indx]++] = new_id[rem_state[c]];
            }
        }

        // Send
        for (i = 0; i < num_recvs; ++i) {
            JX_Int npts = send_pts_num[i];
            JX_Int ip   = jx_ParCSRCommPkgRecvProc(S2_comm_pkg, i);
            jx_MPI_Isend(&npts, 1, JX_MPI_INT, ip, 0, comm, &send_cnt_req[i]);

            if (!npts) continue;

            jx_MPI_Isend(&send_pts[send_pts_start[i]], npts, JX_MPI_INT, ip, 1, comm, &send_pts_req[i]);
        }

        // Recv
        for (i = 0; i < num_sends; ++i) {
            JX_Int npts;
            JX_Int ip = jx_ParCSRCommPkgSendProc(S2_comm_pkg, i);
            jx_MPI_Recv(&npts, 1, JX_MPI_INT, ip, 0, comm, MPI_STATUS_IGNORE);

            if (!npts) continue;

            recv_pts = jx_CTAlloc(JX_Int, npts);
            MPI_Recv(&recv_pts[0], npts, JX_MPI_INT, ip, 1, comm, MPI_STATUS_IGNORE);

            // Update the loc_state of the current process
            for (k = 0; k < npts; k += 2) {
                c  = recv_pts[k] - jx_ParCSRMatrixFirstRowIndex(A);
                id = recv_pts[k + 1];

                loc_state[c] = id;
            }
            jx_Free(recv_pts);
        }

        for (i = 0; i < num_recvs; ++i) {
            JX_Int npts = send_pts_num[i];
            MPI_Wait(&send_cnt_req[i], MPI_STATUS_IGNORE);

            if (!npts) continue;
            MPI_Wait(&send_pts_req[i], MPI_STATUS_IGNORE);
        }
    }

    //* cleanup
    // jx_ParCSRMatrixDestroy(S);
    jx_ParCSRMatrixDestroy(S2);

    jx_Free(rem_state);
    jx_Free(rem_owner);
    jx_Free(nbr_arr);

    jx_Free(S2_remote_ranks);
    jx_Free(map_remote_rank2id);
    jx_Free(S_offd_col_to_S2_offd_col);
    jx_Free(send_pts_req);
    jx_Free(send_cnt_req);

    jx_Free(send_pts_num);
    jx_Free(send_pts_start);
    jx_Free(send_pts);

    jx_Free(new_id);

    //* Output
    *NumAggregates  = naggr;
    *Vertices       = loc_state;
    *Vertices_owner = loc_owner;
    *S_ptr          = S;

#if AGG_DEBUG > 2
    for (i = 0; i < row; i++) {
        printf("rank[%d], loc_state[%d] = %d, loc_owner[%d] = %d\n", iam, i, loc_state[i], i, loc_owner[i]);
    }
#endif

#if AGG_DEBUG
    IAM_0 { DEBUG_PRINTF("rank: %d, End\n", iam); }
#endif

    return status;
}

/**
 * \fn jx_aggregation_symmpair(jx_ParCSRMatrix* A, JX_Int level_now, JX_Int pair_number,
 *                                 JX_Real quality_bound, JX_Int min_num_coarse,
 *                                 JX_Int* NumAggregates, JX_Int** Vertices)
 *
 * \brief AMG coarsening based on symmetric pairwise matching aggregation
 *
 * \param A                 The parallel coefficient matrices (Input)
 * \param level_now         Current level number (Input)
 * \param pair_number       Number of pairs in matching (Input)
 * \param quality_bound     Convergence factor of two grids (Input)
 * \param min_num_coarse    The coarsest grid degree of freedom size (Input)
 * \param NumAggregates     The number of aggregations on the current processor (Output)
 * \param Vertices          The aggregation of vertices on the current processor (Output)
 *
 * \author Li Zhao
 * \date   10/08/2024
 *
 * \note Setup A, P, PT and levels using the pairwise aggregation;
 *       Refer to A. Napov and Y. Notay
 *       "An algebraic multigrid method with guaranteed convergence rate", 2012
 *
 * \note This algorithm only considers the aggregation between local degrees of freedom,
 *       and does not take into account cross processor scenarios.
 */
JX_Int jx_aggregation_symmpair(jx_ParCSRMatrix* A, JX_Int level_now, JX_Int pair_number, JX_Real quality_bound,
                                       JX_Int min_num_coarse, JX_Int* NumAggregates, JX_Int** Vertices)

{
    jx_CSRMatrix* A_diag      = jx_ParCSRMatrixDiag(A);
    JX_Int*       A_diag_i    = jx_CSRMatrixI(A_diag);
    JX_Int*       A_diag_j    = jx_CSRMatrixJ(A_diag);
    JX_Complex*   A_diag_data = jx_CSRMatrixData(A_diag);

    JX_Int            row      = jx_CSRMatrixNumRows(A_diag);
    // JX_MemoryLocation location = jx_ParCSRMatrixMemoryLocation(A);

    JX_Int  i, j, k, num_agg = 0, aggindex, num_coarse_global;
    JX_Int  lvl     = 0;
    JX_Real isorate = 0.0;

    JX_Int            dopass = 0, domin = 0, domin_global = 0;
    JX_Int            status   = JX_SUCCESS;
    JX_Int**          vertices = jx_MAlloc(pair_number * sizeof(JX_Int*));
    jx_ParCSRMatrix** A_arr    = jx_MAlloc(pair_number * sizeof(jx_ParCSRMatrix*));
    jx_ParCSRMatrix*  P        = NULL;
    //
    JX_Int *riperm = NULL, *iperm = NULL;

    MPI_Comm comm = jx_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);
    // JX_Int bandwidth = fasp_dcsr_bandwidth(&mgl[level].A);
    // if (bandwidth > 5.0) param->quality_bound = quality_bound = 1.0 * bandwidth;

    /*-- setup permutation for renumbering --*/
    if (level_now == 0) {
        // JX_Real t1 = jx_MPI_Wtime();
        /*-- initialize permutation --*/
        riperm = jx_MAlloc(row * sizeof(JX_Int));
        iperm  = jx_MAlloc(row * sizeof(JX_Int));

        /*-- setup permutation for renumbering --*/
        jx_nodes_renumbering(A, riperm, iperm);
        // JX_Real t2 = jx_MPI_Wtime();
        // IAM_0 { DEBUG_PRINTF("renumbering time: %e\n", t2 - t1); }
    }

    A_arr[lvl] = A;
    for (i = 1; i <= pair_number; ++i) {
        // DEBUG_PRINTF("row: %d, PW aggregation number: %d, pair_number: %d\n", row, i, pair_number);

        /*-- generate aggregations by pairwise matching --*/
        // if (i == 1)
        if (level_now == 0 && i == 1)
            status = jx_form_pairwise_permutation(A_arr[lvl], i, quality_bound, riperm, iperm, &num_agg, &vertices[lvl]);
        else
            status = jx_form_pairwise(A_arr[lvl], i, quality_bound, &num_agg, &vertices[lvl]);

        // compute the number of coarsen vertices
        jx_MPI_Allreduce(&num_agg, &num_coarse_global, 1, JX_MPI_INT, MPI_SUM, comm);

        if (num_coarse_global == 0) {
            if (iam == 0) {
                ERROR_PRINTF("%s %d: num_coarse_global = %d\n", __FUNCTION__, __LINE__, num_coarse_global);
            }
            exit(0);
        }

        /*-- check number of aggregates in the first pass --*/
        if (i == 1 && num_coarse_global < min_num_coarse) {
            for (domin = k = 0; k < row; k++) {
                if (vertices[lvl][k] == JX_G0PT) domin++;
            }
            jx_MPI_Allreduce(&domin, &domin_global, 1, JX_MPI_INT, MPI_SUM, comm);
            isorate = (JX_Real)num_coarse_global / domin_global;
            if (isorate < 0.1) {
                status = JX_ERROR_AMG_COARSEING;
                // if (iam == 0) DEBUG_PRINTF("JX_ERROR_AMG_COARSEING\n");
                goto END;
            }
        }

        if (i < pair_number) {

            /*-- Perform aggressive coarsening only up to the specified level --*/
            if (num_coarse_global < min_num_coarse) {
                // DEBUG_PRINTF("num_coarse_global: %d < min_num_coarse: %d\n", num_coarse_global, min_num_coarse);
                // for (JX_Int jj = 0; jj < row; jj++) {
                //     printf("Vertices[%d] = %d\n", jj, vertices[lvl][jj]);
                // }
                lvl++;
                break;
            }

            /*-- Form Prolongation --*/
            jx_form_tentative_p(A_arr[lvl], vertices[lvl], num_agg, num_coarse_global, &P);

            /*-- Form coarse level stiffness matrix --*/
            jx_ParCSRMatrix* RT = P;
            jx_BoomerAMGBuildCoarseOperator(RT, A_arr[lvl], P, &A_arr[lvl + 1]);

            jx_ParCSRMatrixDestroy(P);
        }
        lvl++;    // recording level number of vertices[lvl]
        dopass++; // recording level number of A_arr
    }
    // DEBUG_PRINTF("lvl: %d, dopass: %d\n", lvl, dopass);

    // Form global aggregation indices
    if (lvl > 1) {
        for (i = 0; i < row; ++i) {
            aggindex = vertices[0][i];
            if (aggindex < 0) continue;
            for (j = 1; j < lvl; ++j) aggindex = vertices[j][aggindex];
            vertices[0][i] = aggindex;
        }
    }

END:
    *NumAggregates = num_agg;
    *Vertices      = vertices[0];

    /*-- clean memory --*/
    A_arr[0]    = NULL;
    vertices[0] = NULL;
    for (i = 1; i < dopass; ++i) {
        jx_ParCSRMatrixDestroy(A_arr[i]);
    }
    for (i = 1; i < lvl; ++i) {
        jx_Free(vertices[i]);
    }
    jx_Free(A_arr);
    jx_Free(vertices);
    A_arr    = NULL;
    vertices = NULL;

    if (level_now == 0) {
        jx_Free(riperm);
        jx_Free(iperm);
        riperm = NULL;
        iperm  = NULL;
    }

    // return
    return status;
}

/*
 *---------------------------------------------------------------------------------
 *  private functions
 *---------------------------------------------------------------------------------
 */
/**
 * \fn static JX_Int jx_nodes_renumbering(jx_ParCSRMatrix* A, JX_Int* riperm, JX_Int* iperm)
 *
 *
 * \brief Renumber the nodes according to their degree, in ascending order.
 *
 * \param A                 Pointer to the coefficient matrices
 * \param riperm            Mapping from new node to old node numbers (Output)
 * \param iperm             Mapping from old node to new node numbers, i.e., inverse of 'riperm'  (Output)
 *
 * \author Li Zhao
 * \date   10/19/2024
 *
 */
static JX_Int jx_nodes_renumbering(jx_ParCSRMatrix* A, JX_Int* riperm, JX_Int* iperm)
{
    jx_CSRMatrix* A_diag   = jx_ParCSRMatrixDiag(A);
    JX_Int*       A_diag_i = jx_CSRMatrixI(A_diag);
    JX_Int*       A_diag_j = jx_CSRMatrixJ(A_diag);

    jx_CSRMatrix* A_offd          = jx_ParCSRMatrixOffd(A);
    JX_Int*       A_offd_i        = jx_CSRMatrixI(A_offd);
    JX_Int        num_cols_A_offd = jx_CSRMatrixNumCols(A_offd);

    JX_Int n = jx_CSRMatrixNumRows(A_diag);

    // return status
    JX_Int status = JX_SUCCESS;

    JX_Int  ifirst = 1, ilast = n;
    JX_Int  mindg   = n + 1; // initialize minimum degree
    JX_Int  dg_diag = 0, dg_offd = 0, dg;
    JX_Int  i, i1, i2, jj, j, kk, j1, j2, ijs, ijs1, ijs2;
    JX_Bool exc;

    MPI_Comm comm = jx_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

    i2 = 0;
    jj = -1;
    for (i = 0; i < n; i++) { // Loop through the nodes
        dg_diag = A_diag_i[i + 1] - A_diag_i[i];
        if (num_cols_A_offd)
            dg_offd = A_offd_i[i + 1] - A_offd_i[i];
        else
            dg_offd = 0;
        dg = dg_diag + dg_offd; // dg = degree+1

        if (dg > 1) {         // dg > 1 (valid degree)
            iperm[i] = -dg;   // The degree of node i (marked with a negative value)
            if (dg < mindg) { // Find the minimum degree `mindg`, and record the corresponding node's index `jj`.
                mindg = dg;
                jj    = i;
            }
        } else { // dg = 1, Renumber the nodes (invalid degree, degree = 0)
            riperm[i2] = i;
            iperm[i]   = i2;
            i2++;
        }
    } // end for

    ijs = 0;
    i1  = i2;

label15:
    if (i2 < n) { // If it's not an identity matrix, append the node with the smallest degree to the end of `riperm`.
        riperm[i2] = jj;
        iperm[jj]  = i2;
    }

    while (i1 <= i2 && i2 < n) {
        i    = riperm[i1];
        ijs1 = i2 + 1;
        j1   = A_diag_i[i];
        j2   = A_diag_i[i + 1];

        // Loop through the neighboring nodes of node `i` (excluding node `i` itself and cross-process nodes).
        for (kk = j1; kk < j2; kk++) {
            j = A_diag_j[kk];
            if (j == i) continue;
            if (iperm[j] < 0) { // `iperm(j) < 0` means that node `j` has a degree greater than 1.
                i2++;
                riperm[i2] = j;
            }
        }

        ijs2 = i2;
        exc  = (ijs2 > ijs1);
        // Renumber the neighboring nodes in ascending order based on their degree.
        while (exc) {
            exc = jx_false;
            for (kk = ijs1 + 1; kk <= ijs2; kk++) {
                if (iperm[riperm[kk]] > iperm[riperm[kk - 1]]) { // iperm(riperm(kk)) > iperm(riperm(kk-1))
                    j              = riperm[kk];
                    riperm[kk]     = riperm[kk - 1];
                    riperm[kk - 1] = j;
                    exc            = jx_true;
                }
            }
        }

        // Update: Mapping from the old node `riperm(kk)` to the new node `kk`.
        for (kk = ijs1; kk <= ijs2; kk++) {
            iperm[riperm[kk]] = kk;
        }

        i1++;
    } // while

    if (i2 < n - 1) {
        jj = 0;
        while (jj == 0) {
            ijs++;
            if (ijs > n) {
                mindg++;
                ijs = 0;
            }
            ijs1 = ijs;

            dg_offd = 0;
            dg_diag = A_diag_i[ijs1 + 1] - A_diag_i[ijs1];
            if (num_cols_A_offd) dg_offd = A_offd_i[ijs1 + 1] - A_offd_i[ijs1];
            dg = dg_diag + dg_offd; // dg = degree+1

            if (iperm[ijs1] < 0 && dg == mindg) {
                jj = ijs1;
            }
        }
        i2++;
        goto label15;
    }

#if 0 // Debug
    IAM_0
    {
        for (i = 0; i < n; i++) {
            dg_diag = A_diag_i[i + 1] - A_diag_i[i];
            if (num_cols_A_offd)
                dg_offd = A_offd_i[i + 1] - A_offd_i[i];
            else
                dg_offd = 0;
            dg = dg_diag + dg_offd; // dg = degree+1

            printf("iperm[%d] = %d, riperm[%d] = %d, degree = %d\n", i + 1, iperm[i] + 1, i + 1, riperm[i] + 1, dg);
        }
    }
#endif

    return status;
}

/**
 * \fn static JX_Int jx_form_pairwise_permutation(jx_ParCSRMatrix* A, JX_Int num_pair,
 *                                            JX_Real k_tg, JX_Int* riperm, JX_Int* iperm,
 *                                            JX_Int* NumAggregates, JX_Int** Vertices)
 *
 * \brief Form aggregation based on pairwise matching with permutation
 *
 * \param A                 Pointer to the coefficient matrices
 * \param num_pair          Number of pairs in matching
 * \param k_tg              Convergence factor of two grids
 * \param riperm            Mapping from new node to old node numbers
 * \param iperm             Mapping from old node to new node numbers, i.e., inverse of 'riperm'
 * \param NumAggregates     The number of aggregations on the current processor (Output)
 * \param Vertices          The aggregation of vertices on the current processor (Output)
 *
 * \author Li Zhao
 * \date   10/19/2024
 *
 * \note Refer to Artem Napov and Yvan Notay "An algebraic multigrid
 *       method with guaranteed convergence rate" 2011.
 *
 * \note This algorithm only considers the aggregation between local degrees of freedom,
 *       and does not take into account cross processor scenarios.
 */
static JX_Int jx_form_pairwise_permutation(jx_ParCSRMatrix* A, JX_Int num_pair, JX_Real k_tg, JX_Int* riperm, JX_Int* iperm,
                                                   JX_Int* NumAggregates, JX_Int** Vertices)
{
    jx_CSRMatrix* A_diag      = jx_ParCSRMatrixDiag(A);
    JX_Int*       A_diag_i    = jx_CSRMatrixI(A_diag);
    JX_Int*       A_diag_j    = jx_CSRMatrixJ(A_diag);
    JX_Complex*   A_diag_data = jx_CSRMatrixData(A_diag);

    jx_CSRMatrix* A_offd          = jx_ParCSRMatrixOffd(A);
    JX_Int*       A_offd_i        = jx_CSRMatrixI(A_offd);
    JX_Int*       A_offd_j        = jx_CSRMatrixJ(A_offd);
    JX_Complex*   A_offd_data     = jx_CSRMatrixData(A_offd);
    JX_Int        num_cols_A_offd = jx_CSRMatrixNumCols(A_offd);
    JX_BigInt*    col_map_offd    = jx_ParCSRMatrixColMapOffd(A);
    // num_nonzeros_offd = jx_CSRMatrixNumNonzeros(offd);

    JX_Int            row        = jx_CSRMatrixNumRows(A_diag);
    JX_Int            A_diag_nnz = jx_CSRMatrixNumNonzeros(A_diag);
    // JX_MemoryLocation location   = jx_ParCSRMatrixMemoryLocation(A);

    // return status
    JX_Int  status = JX_SUCCESS;
    JX_Int  i, j, old_i, row_start, row_end;
    JX_Real sum;

    JX_Int  col, ipair = -1;
    JX_Real mu, min_mu, aii, ajj, aij;
    JX_Real temp1, temp2, temp3, temp4;

    MPI_Comm comm = jx_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

    JX_Real checkdd = 0.5;
    checkdd             = jx_max(jx_abs(checkdd), (k_tg + 1) / (k_tg - 1));
    /*---------------------------------------------------------*/
    /* Step 1. select extremely strong diagonal dominate rows  */
    /*         and store in G0.                                */
    /*         G0        : vertices->val[i] = JX_G0PT      */
    /*         Remaining : vertices->val[i] = JX_UNPT      */
    /*---------------------------------------------------------*/
    JX_Int* vertices = jx_MAlloc(row * sizeof(JX_Int));

    if (num_pair == 1) {
        for (i = 0; i < row; i++) {
            sum = 0.0; // row sum (excluding diagonal entry)

            // diagonal blocks
            row_start = A_diag_i[i];
            row_end   = A_diag_i[i + 1];
#if 1
            aii = A_diag_data[row_start]; // diagonal entry
            for (j = row_start + 1; j < row_end; j++) sum += jx_abs(A_diag_data[j]);
#else
            for (j = row_start; j < row_end; j++) {
                if (A_diag_j[j] != i) {
                    sum += jx_abs(A_diag_data[j]);
                } else
                    aii = A_diag_data[j]; // diagonal entry
            }
#endif

            // non-diagonal blocks
            if (num_cols_A_offd) {
                row_start = A_offd_i[i];
                row_end   = A_offd_i[i + 1];
                for (j = row_start; j < row_end; j++) {
                    // J = col_map_offd[offd_j[j]] + (JX_BigInt)base_j;
                    sum += jx_abs(A_offd_data[j]);
                }
            }

            if (aii >= checkdd * sum) {
                vertices[i] = JX_G0PT;
            } else {
                vertices[i] = JX_UNPT;
            }
        }
    } else {
        for (i = 0; i < row; ++i) vertices[i] = JX_UNPT;
    }

    /*---------------------------------------------------------*/
    /* Step 2. compute row sum (off-diagonal) for each vertex  */
    /*---------------------------------------------------------*/
    JX_Real* s    = jx_MAlloc(row * sizeof(JX_Real));
    JX_Real* diag = jx_MAlloc(row * sizeof(JX_Real));

    for (i = 0; i < row; i++) {
        s[i]    = 0.0;
        diag[i] = 0.0;

        if (vertices[i] == JX_G0PT) continue;

        // diagonal blocks
        row_start = A_diag_i[i];
        row_end   = A_diag_i[i + 1];
#if 1
        diag[i] = A_diag_data[row_start]; // diagonal entry
        for (j = row_start + 1; j < row_end; j++) s[i] -= A_diag_data[j];
#else
        for (j = row_start; j < row_end; j++) {
            if (A_diag_j[j] != i) {
                s[i] -= A_diag_data[j];
            } else
                diag[i] = A_diag_data[j];
        }
#endif
        // non-diagonal blocks
        if (num_cols_A_offd) {
            row_start = A_offd_i[i];
            row_end   = A_offd_i[i + 1];
            for (j = row_start; j < row_end; j++) {
                s[i] -= A_offd_data[j];
            }
        }
    }

    /*---------------------------------------------------------*/
    /* Step 3. start the pairwise aggregation                  */
    /*---------------------------------------------------------*/

    *NumAggregates = 0;
#if 0
    JX_Real rsi, rsj, sig1, sig2, epsr;
    JX_Real repsmach = JX_REAL_EPSILON;
    JX_Real del1, eta1, del2, eta2, del12, vals, valp, tent, val = 0.0;
#endif

    for (i = 0; i < row; i++) { // cycle the new node number

        old_i = riperm[i]; // take old node number

        if (vertices[old_i] != JX_UNPT) continue;

        min_mu = JX_REAL_MAX;

        row_start = A_diag_i[old_i];
        row_end   = A_diag_i[old_i + 1];

        aii   = diag[old_i];
        ipair = -1;

#if 0
        for (j = row_start; j < row_end; j++) {
            col = A_diag_j[j];
            if (col == old_i) continue;
            if (vertices[col] != JX_UNPT) continue;

            aij  = A_diag_data[j];
            ajj  = diag[col];
            vals = -aij;

            if (0) { // zerors = 0
                rsi = 0.0;
                rsj = 0.0;
            } else {
                rsi = -s[i] + aii;
                rsj = -s[col] + ajj;
            }

            sig1 = s[old_i] - vals;
            sig2 = s[col] - vals;
            epsr = repsmach * jx_abs(vals);

            if (sig1 > 0.0) {
                del1 = rsi;
                eta1 = rsi + 2 * sig1;
            } else {
                del1 = rsi + 2 * sig1;
                eta1 = rsi;
            }
            if (eta1 < -epsr) continue;

            if (sig2 > 0.0) {
                del2 = rsj;
                eta2 = rsj + 2 * sig2;
            } else {
                del2 = rsj + 2 * sig2;
                eta2 = rsj;
            }
            if (eta2 < -epsr) continue;

            if (vals > 0.0) {
                if (jx_abs(del1) < epsr && jx_abs(del2) < epsr) {
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else if (jx_abs(del1) < epsr) {
                    if (del2 < -epsr) continue;
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else if (jx_abs(del2) < epsr) {
                    if (del1 < -epsr) continue;
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else {
                    del12 = del1 + del2;
                    if (del12 < -epsr) continue;
                    valp = vals + del1 * del2 / del12;
                    if (valp < 0.0) continue;
                    valp = (vals + (eta1 * eta2) / (eta1 + eta2)) / valp;
                }
            } else {
                if (del1 <= 0.0 || del2 <= 0.0) continue;
                valp = vals + del1 * del2 / (del1 + del2);
                if (valp < 0.0) continue;
                vals = vals + (eta1 * eta2) / (eta1 + eta2);
                if (vals < 0.0) continue;
                valp = vals / valp;
            }
            if (valp > k_tg) continue;
            tent = valp;

            if (ipair == -1 || 16 * (tent - val) < -1.0 || (16 * (tent - val) < 1.0 && iperm[col] < iperm[ipair])) {
                ipair = col;
                val   = tent;
            } else {
                continue;
            }
        }

        vertices[old_i] = *NumAggregates;
        if (ipair > -1) vertices[ipair] = *NumAggregates;

        *NumAggregates += 1;
#else
        for (j = row_start; j < row_end; j++) {
            col = A_diag_j[j];
            if (col == old_i) continue;
            if (vertices[col] != JX_UNPT) continue;

            aij = A_diag_data[j];
            ajj = diag[col];

            temp1 = aii + s[old_i] + 2 * aij;
            temp2 = ajj + s[col] + 2 * aij;
            temp2 = 1.0 / temp1 + 1.0 / temp2;

            temp3 = jx_max(jx_abs(aii - s[old_i]), JX_REAL_EPSILON); // avoid temp3 to be zero
            temp4 = jx_max(jx_abs(ajj - s[col]), JX_REAL_EPSILON);   // avoid temp4 to be zero
            temp4 = -aij + 1. / (1.0 / temp3 + 1.0 / temp4);
            // avoid temp4 to be zero
            if (jx_abs(temp4) < JX_REAL_EPSILON) temp4 = (temp4 > 0) ? JX_REAL_EPSILON : -JX_REAL_EPSILON;

            mu = (-aij + 1.0 / temp2) / temp4;

            if (min_mu > mu) {
                min_mu = mu;
                ipair  = col;
            }
        }

        vertices[old_i] = *NumAggregates;

        if (min_mu <= k_tg) vertices[ipair] = *NumAggregates;

        *NumAggregates += 1;
#endif
    }

    jx_Free(diag);
    jx_Free(s);

    // END
    *Vertices = vertices;

    return status;
}

static JX_Int jx_form_pairwise(jx_ParCSRMatrix* A, JX_Int num_pair, JX_Real k_tg, JX_Int* NumAggregates, JX_Int** Vertices)
{
    jx_CSRMatrix* A_diag      = jx_ParCSRMatrixDiag(A);
    JX_Int*       A_diag_i    = jx_CSRMatrixI(A_diag);
    JX_Int*       A_diag_j    = jx_CSRMatrixJ(A_diag);
    JX_Complex*   A_diag_data = jx_CSRMatrixData(A_diag);

    jx_CSRMatrix* A_offd          = jx_ParCSRMatrixOffd(A);
    JX_Int*       A_offd_i        = jx_CSRMatrixI(A_offd);
    JX_Int*       A_offd_j        = jx_CSRMatrixJ(A_offd);
    JX_Complex*   A_offd_data     = jx_CSRMatrixData(A_offd);
    JX_Int        num_cols_A_offd = jx_CSRMatrixNumCols(A_offd);
    JX_BigInt*    col_map_offd    = jx_ParCSRMatrixColMapOffd(A);
    // num_nonzeros_offd = jx_CSRMatrixNumNonzeros(offd);

    JX_Int            row        = jx_CSRMatrixNumRows(A_diag);
    JX_Int            A_diag_nnz = jx_CSRMatrixNumNonzeros(A_diag);
    // JX_MemoryLocation location   = jx_ParCSRMatrixMemoryLocation(A);

    // return status
    JX_Int  status = JX_SUCCESS;
    JX_Int  i, j, row_start, row_end;
    JX_Real sum;

    JX_Int  col, ipair = -1;
    JX_Real mu, min_mu, aii, ajj, aij;
    JX_Real temp1, temp2, temp3, temp4;

    MPI_Comm comm = jx_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

    JX_Real checkdd = 0.5;
    checkdd             = jx_max(jx_abs(checkdd), (k_tg + 1) / (k_tg - 1));
    /*---------------------------------------------------------*/
    /* Step 1. select extremely strong diagonal dominate rows  */
    /*         and store in G0.                                */
    /*         G0        : vertices->val[i] = JX_G0PT      */
    /*         Remaining : vertices->val[i] = JX_UNPT      */
    /*---------------------------------------------------------*/
    JX_Int* vertices = jx_MAlloc(row * sizeof(JX_Int));

    if (num_pair == 1) {
        for (i = 0; i < row; i++) {
            sum = 0.0; // row sum (excluding diagonal entry)

            // diagonal blocks
            row_start = A_diag_i[i];
            row_end   = A_diag_i[i + 1];
#if 1
            aii = A_diag_data[row_start]; // diagonal entry
            for (j = row_start + 1; j < row_end; j++) sum += jx_abs(A_diag_data[j]);
#else
            for (j = row_start; j < row_end; j++) {
                if (A_diag_j[j] != i) {
                    sum += jx_abs(A_diag_data[j]);
                } else
                    aii = A_diag_data[j]; // diagonal entry
            }
#endif
            // non-diagonal blocks
            if (num_cols_A_offd) {
                row_start = A_offd_i[i];
                row_end   = A_offd_i[i + 1];
                for (j = row_start; j < row_end; j++) {
                    // J = col_map_offd[offd_j[j]] + (JX_BigInt)base_j;
                    sum += jx_abs(A_offd_data[j]);
                }
            }

            if (aii >= checkdd * sum) {
                vertices[i] = JX_G0PT;
            } else {
                vertices[i] = JX_UNPT;
            }
        }
    } else {
        for (i = 0; i < row; ++i) vertices[i] = JX_UNPT;
    }

    /*---------------------------------------------------------*/
    /* Step 2. compute row sum (off-diagonal) for each vertex  */
    /*---------------------------------------------------------*/
    JX_Real* s    = jx_MAlloc(row * sizeof(JX_Real));
    JX_Real* diag = jx_MAlloc(row * sizeof(JX_Real));

    for (i = 0; i < row; i++) {
        s[i]    = 0.0;
        diag[i] = 0.0;

        if (vertices[i] == JX_G0PT) continue;

        // diagonal blocks
        row_start = A_diag_i[i];
        row_end   = A_diag_i[i + 1];
#if 1
        diag[i] = A_diag_data[row_start]; // diagonal entry
        for (j = row_start + 1; j < row_end; j++) s[i] -= A_diag_data[j];
#else
        for (j = row_start; j < row_end; j++) {
            if (A_diag_j[j] != i) {
                s[i] -= A_diag_data[j];
            } else
                diag[i] = A_diag_data[j];
        }
#endif
        // non-diagonal blocks
        if (num_cols_A_offd) {
            row_start = A_offd_i[i];
            row_end   = A_offd_i[i + 1];
            for (j = row_start; j < row_end; j++) {
                s[i] -= A_offd_data[j];
            }
        }
    }

    /*---------------------------------------------------------*/
    /* Step 3. start the pairwise aggregation                  */
    /*---------------------------------------------------------*/

    *NumAggregates = 0;
#if 0
    JX_Real rsi, rsj, sig1, sig2, epsr;
    JX_Real repsmach = JX_REAL_EPSILON;
    JX_Real del1, eta1, del2, eta2, del12, vals, valp, tent, val = 0.0;
#endif

    for (i = 0; i < row; i++) {

        if (vertices[i] != JX_UNPT) continue;

        min_mu = JX_REAL_MAX;

        row_start = A_diag_i[i];
        row_end   = A_diag_i[i + 1];

        aii   = diag[i];
        ipair = -1;

#if 0
        for (j = row_start; j < row_end; j++) {
            col = A_diag_j[j];
            if (col == i) continue;
            if (vertices[col] != JX_UNPT) continue;

            aij  = A_diag_data[j];
            ajj  = diag[col];
            vals = -aij;

            if (0) { // zerors = 0
                rsi = 0.0;
                rsj = 0.0;
            } else {
                rsi = -s[i] + aii;
                rsj = -s[col] + ajj;
            }

            sig1 = s[i] - vals;
            sig2 = s[col] - vals;
            epsr = repsmach * jx_abs(vals);

            if (sig1 > 0.0) {
                del1 = rsi;
                eta1 = rsi + 2 * sig1;
            } else {
                del1 = rsi + 2 * sig1;
                eta1 = rsi;
            }
            if (eta1 < -epsr) continue;

            if (sig2 > 0.0) {
                del2 = rsj;
                eta2 = rsj + 2 * sig2;
            } else {
                del2 = rsj + 2 * sig2;
                eta2 = rsj;
            }
            if (eta2 < -epsr) continue;

            if (vals > 0.0) {
                if (jx_abs(del1) < epsr && jx_abs(del2) < epsr) {
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else if (jx_abs(del1) < epsr) {
                    if (del2 < -epsr) continue;
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else if (jx_abs(del2) < epsr) {
                    if (del1 < -epsr) continue;
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else {
                    del12 = del1 + del2;
                    if (del12 < -epsr) continue;
                    valp = vals + del1 * del2 / del12;
                    if (valp < 0.0) continue;
                    valp = (vals + (eta1 * eta2) / (eta1 + eta2)) / valp;
                }
            } else {
                if (del1 <= 0.0 || del2 <= 0.0) continue;
                valp = vals + del1 * del2 / (del1 + del2);
                if (valp < 0.0) continue;
                vals = vals + (eta1 * eta2) / (eta1 + eta2);
                if (vals < 0.0) continue;
                valp = vals / valp;
            }
            if (valp > k_tg) continue;
            tent = valp;

            if (ipair == -1 || 16 * (tent - val) < -1.0 || (16 * (tent - val) < 1.0 && col < ipair)) {
                ipair = col;
                val   = tent;
            } else {
                continue;
            }
        }

        vertices[i] = *NumAggregates;
        if (ipair > -1) vertices[ipair] = *NumAggregates;

        *NumAggregates += 1;
#else
        for (j = row_start; j < row_end; j++) {
            col = A_diag_j[j];
            if (col == i) continue;
            if (vertices[col] != JX_UNPT) continue;

            aij = A_diag_data[j];
            ajj = diag[col];

            temp1 = aii + s[i] + 2 * aij;
            temp2 = ajj + s[col] + 2 * aij;
            temp2 = 1.0 / temp1 + 1.0 / temp2;

            temp3 = jx_max(jx_abs(aii - s[i]), JX_REAL_EPSILON);   // avoid temp3 to be zero
            temp4 = jx_max(jx_abs(ajj - s[col]), JX_REAL_EPSILON); // avoid temp4 to be zero
            temp4 = -aij + 1. / (1.0 / temp3 + 1.0 / temp4);
            // avoid temp4 to be zero
            if (jx_abs(temp4) < JX_REAL_EPSILON) temp4 = (temp4 > 0) ? JX_REAL_EPSILON : -JX_REAL_EPSILON;

            mu = (-aij + 1.0 / temp2) / temp4;

            if (min_mu > mu) {
                min_mu = mu;
                ipair  = col;
            }
        }

        vertices[i] = *NumAggregates;

        if (min_mu <= k_tg) vertices[ipair] = *NumAggregates;

        *NumAggregates += 1;
#endif
    }

    jx_Free(diag);
    jx_Free(s);

    // END
    *Vertices = vertices;

    return status;
}