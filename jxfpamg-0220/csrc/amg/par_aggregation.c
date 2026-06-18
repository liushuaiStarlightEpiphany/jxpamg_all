//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_aggregation.c --  aggregation algorithms for ParCSR
 *  Date: 2025/10/08
 */ 


#include "jxf_pamg.h"

// internal debug parameter
#define AGG_DEBUG 0

/* Declaration of private functions */
static JXF_Int jxf_nodes_renumbering(jxf_ParCSRMatrix* A, JXF_Int* riperm, JXF_Int* iperm);
static JXF_Int jxf_form_pairwise(jxf_ParCSRMatrix* A, JXF_Int num_pair, JXF_Real k_tg, JXF_Int* NumAggregates, JXF_Int** Vertices);
static JXF_Int jxf_form_pairwise_permutation(jxf_ParCSRMatrix* A, JXF_Int num_pair, JXF_Real k_tg, JXF_Int* riperm, JXF_Int* iperm,
                                                   JXF_Int* NumAggregates, JXF_Int** Vertices);

/**
 * \fn JXF_Int jxf_aggregation_vmb(jxf_ParCSRMatrix* A, JXF_Int NumLevels,
 *                                       JXF_Real strong_coupled, JXF_Real tentative_smooth,
 *                                       JXF_Int max_aggregation, JXF_Int* NumAggregates,
 *                                       JXF_Int** Vertices)
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
JXF_Int jxf_aggregation_vmb(jxf_ParCSRMatrix* A, JXF_Int NumLevels, JXF_Real strong_coupled, JXF_Real tentative_smooth,
                                  JXF_Int max_aggregation, JXF_Int* NumAggregates, JXF_Int** Vertices, jxf_ParCSRMatrix** S_ptr)
{
    jxf_CSRMatrix* A_diag      = jxf_ParCSRMatrixDiag(A);
    JXF_Int*       A_diag_i    = jxf_CSRMatrixI(A_diag);
    JXF_Int*       A_diag_j    = jxf_CSRMatrixJ(A_diag);
    JXF_Complex*   A_diag_data = jxf_CSRMatrixData(A_diag);

    // jxf_CSRMatrix* A_offd   = jxf_ParCSRMatrixOffd(A);
    // JXF_Int*       A_offd_i = jxf_CSRMatrixI(A_offd);
    // JXF_Int*       A_offd_j = jxf_CSRMatrixJ(A_offd);

    JXF_Int row        = jxf_CSRMatrixNumRows(A_diag);
    JXF_Int A_diag_nnz = jxf_CSRMatrixNumNonzeros(A_diag);
    // JXF_BigInt* row_starts = jxf_ParCSRMatrixRowStarts(A);

    // return status
    JXF_Int status = JXF_SUCCESS;

    // local variables
    JXF_Int  num_left = row;
    JXF_Int  subset, count;
    JXF_Int* vertices;

    JXF_Real  strongly_coupled, strongly_coupled2;
    JXF_Int   i, j, index, row_start, row_end;
    JXF_Int * NIA, *NJA; // strongly coupled neighborhood
    JXF_Real* Nval;
    MPI_Comm     comm = jxf_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

#if AGG_DEBUG
    DEBUG_PRINTF("rank: %d, Begin\n", iam);
#endif

#if 0
    char matfile[128];
    sprintf(matfile, "out/A_diag_%d.dat", iam);
    jxf_CSRMatrixPrint(A_diag, matfile);
    jxf_CSRMatrix* A_seq = jxf_ParCSRMatrixToCSRMatrixAll(A);
    if (iam == 0) jxf_CSRMatrixPrint(A_seq, "out/A_seq.dat");
#endif

    // get diag
    // JXF_MemoryLocation location = jxf_ParCSRMatrixMemoryLocation(A);
    JXF_Complex*       diag_abs = jxf_MAlloc(row * sizeof(JXF_Complex));
    jxf_CSRMatrixExtractDiagonal(A_diag, diag_abs, 1);

    if (tentative_smooth > JXF_REAL_EPSILON) {
        strongly_coupled = strong_coupled * jxf_pow(0.5, NumLevels); // 0.5^L, L=0,1,2...
    } else {
        strongly_coupled = strong_coupled;
    }
    strongly_coupled2 = jxf_pow(strongly_coupled, 2);

#if 0
    DEBUG_PRINTF("rank: %d, NumLevels: %d, strongly_coupled: %f, tentative_smooth: %f\n", iam, NumLevels, strongly_coupled, tentative_smooth);
#endif

    /*------------------------------------------*/
    /*    Form strongly coupled neighborhood    */
    /*------------------------------------------*/
    NIA  = jxf_MAlloc((row + 1) * sizeof(JXF_Int));
    NJA  = jxf_MAlloc(A_diag_nnz * sizeof(JXF_Int));
    Nval = jxf_MAlloc(A_diag_nnz * sizeof(JXF_Real));

    // Note that strongly coupled neighborhood only need to be found in the diagonal matrix part, and non local parts do not need to be considered.
    for (index = i = 0; i < row; ++i) {
        NIA[i]    = index;
        row_start = A_diag_i[i];
        row_end   = A_diag_i[i + 1];
        for (j = row_start; j < row_end; ++j) {
            if ((A_diag_j[j] == i) || (jxf_pow(A_diag_data[j], 2) >= strongly_coupled2 * diag_abs[i] * diag_abs[A_diag_j[j]])) {
                NJA[index]  = A_diag_j[j];
                Nval[index] = A_diag_data[j];
                index++;
            }
        }
    }
    NIA[row] = index;
    jxf_Free(diag_abs);

    NJA  = jxf_TReAlloc(NJA, JXF_Int, index);
    Nval = jxf_TReAlloc(Nval, JXF_Real, index);

    /*------------------------------------------*/
    /*             Initialization               */
    /*------------------------------------------*/
    vertices = jxf_MAlloc(row * sizeof(JXF_Int));
    for (i = 0; i < row; ++i) vertices[i] = JXF_INPT;

    *NumAggregates = 0;

    /*----------------------------------------------------------------------------------------*/
    /* Step 1. Choose non-intersecting strongly coupled neighborhoods as initial aggregation  */
    /* Note that isolation points do not participate in aggregation                           */
    /*----------------------------------------------------------------------------------------*/
    for (i = 0; i < row; ++i) {
        if ((A_diag_i[i + 1] - A_diag_i[i]) == 1) {
            vertices[i] = JXF_UNPT; // isolation points
            num_left--;
        } else {
            subset    = jxf_true;
            row_start = NIA[i];
            row_end   = NIA[i + 1];
            for (j = row_start; j < row_end; ++j) {
                if (vertices[NJA[j]] >= JXF_UNPT) {
                    subset = jxf_false;
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
    JXF_Int* temp_C       = jxf_MAlloc(row * sizeof(JXF_Int));
    JXF_Int* num_each_agg = jxf_CAlloc(*NumAggregates, sizeof(JXF_Int)); // initialize num_each_agg[i] = 0

    for (i = 0; i < row; ++i) {
        temp_C[i] = vertices[i];
        if (vertices[i] >= 0) num_each_agg[vertices[i]]++;
    }

    for (i = 0; i < row; ++i) {
        if (vertices[i] < JXF_UNPT) {
            row_start = NIA[i];
            row_end   = NIA[i + 1];

            for (j = row_start; j < row_end; ++j) {
                if (temp_C[NJA[j]] > JXF_UNPT && num_each_agg[temp_C[NJA[j]]] < max_aggregation) {
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
            if (vertices[i] < JXF_UNPT) {
                count       = 0;
                vertices[i] = *NumAggregates;
                num_left--;
                count++;
                row_start = NIA[i];
                row_end   = NIA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    if ((NJA[j] != i) && (vertices[NJA[j]] < JXF_UNPT) && (count < max_aggregation)) {
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
        for (i = 0; i < row; ++i) jxf_printf("vertices[%d] = %d\n", i, vertices[i]);
    }
#endif

    jxf_ParCSRMatrix* S           = jxf_ParCSRMatrixCreate(comm, jxf_ParCSRMatrixGlobalNumRows(A), jxf_ParCSRMatrixGlobalNumCols(A),
                                                                 jxf_ParCSRMatrixRowStarts(A), jxf_ParCSRMatrixColStarts(A), 0, index, 0);
    jxf_CSRMatrix*    S_diag      = jxf_ParCSRMatrixDiag(S);
    JXF_Int*          S_diag_i    = jxf_CSRMatrixI(S_diag);
    JXF_Int*          S_diag_j    = jxf_CSRMatrixJ(S_diag);
    JXF_Complex*      S_diag_data = jxf_CSRMatrixData(S_diag);

    jxf_Free(S_diag_i);
    jxf_Free(S_diag_j);
    jxf_Free(S_diag_data);

    jxf_CSRMatrixI(S_diag)    = NIA;
    jxf_CSRMatrixJ(S_diag)    = NJA;
    jxf_CSRMatrixData(S_diag) = Nval;

    jxf_CSRMatrix* S_offd  = jxf_ParCSRMatrixOffd(S);
    jxf_CSRMatrixI(S_offd) = jxf_CAlloc(row + 1, sizeof(JXF_Int));

    // END:
    *Vertices = vertices;
    *S_ptr    = S;

    // jxf_Free(NIA);
    // NIA = NULL;
    // jxf_Free(NJA);
    // NJA = NULL;
    // jxf_Free(Nval);
    // Nval = NULL;
    jxf_Free(temp_C);
    temp_C = NULL;
    jxf_Free(num_each_agg);
    num_each_agg = NULL;

#if AGG_DEBUG
    DEBUG_PRINTF("rank: %d, End\n", iam);
#endif

    return status;
}

/**
 * \fn JXF_Int jxf_aggregation_vmb_par(jxf_ParCSRMatrix* A, JXF_Int NumLevels,
 *                                       JXF_Real strong_coupled, JXF_Real tentative_smooth,
 *                                       JXF_Int max_aggregation, JXF_Int* NumAggregates,
 *                                       JXF_Int** Vertices, JXF_Int** Vertices_owner, jxf_ParCSRMatrix** S_ptr)
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
JXF_Int jxf_aggregation_vmb_par(jxf_ParCSRMatrix* A, JXF_Int NumLevels, JXF_Real strong_coupled, JXF_Real tentative_smooth,
                                      JXF_Int max_aggregation, JXF_Int* NumAggregates, JXF_Int** Vertices, JXF_Int** Vertices_owner,
                                      jxf_ParCSRMatrix** S_ptr)
{
    jxf_CSRMatrix* A_diag      = jxf_ParCSRMatrixDiag(A);
    JXF_Int*       A_diag_i    = jxf_CSRMatrixI(A_diag);
    JXF_Int*       A_diag_j    = jxf_CSRMatrixJ(A_diag);
    JXF_Complex*   A_diag_data = jxf_CSRMatrixData(A_diag);

    jxf_CSRMatrix* A_offd          = jxf_ParCSRMatrixOffd(A);
    JXF_Int*       A_offd_i        = jxf_CSRMatrixI(A_offd);
    JXF_Int*       A_offd_j        = jxf_CSRMatrixJ(A_offd);
    JXF_Complex*   A_offd_data     = jxf_CSRMatrixData(A_offd);
    JXF_Int        A_num_cols_offd = jxf_CSRMatrixNumCols(A_offd);
    JXF_BigInt*    A_col_map_offd  = jxf_ParCSRMatrixColMapOffd(A);

    JXF_Int            row        = jxf_CSRMatrixNumRows(A_diag);
    JXF_Int            A_diag_nnz = jxf_CSRMatrixNumNonzeros(A_diag);
    JXF_Int            A_offd_nnz = jxf_CSRMatrixNumNonzeros(A_offd);
    JXF_BigInt*        row_starts = jxf_ParCSRMatrixRowStarts(A);
    JXF_Int            global_row = jxf_ParCSRMatrixGlobalNumRows(A);
    JXF_BigInt         row_first  = jxf_ParCSRMatrixFirstRowIndex(A);
    // JXF_MemoryLocation location   = jxf_ParCSRMatrixMemoryLocation(A);
    jxf_ParCSRCommPkg* A_comm_pkg = jxf_ParCSRMatrixCommPkg(A);
    if (!A_comm_pkg) {
        jxf_MatvecCommPkgCreate(A);
        A_comm_pkg = jxf_ParCSRMatrixCommPkg(A);
    }

    // return status
    JXF_Int status   = JXF_SUCCESS;
    JXF_Int agg_iter = 0, agg_maxit = 100;

    JXF_Real strongly_coupled, strongly_coupled2;
    JXF_Int  i, j, k, index, index_offd, row_start, row_end;
    MPI_Comm    comm = jxf_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

#if AGG_DEBUG
    IAM_0 { DEBUG_PRINTF("rank: %d, Begin\n", iam); }
#endif

    if (tentative_smooth > JXF_REAL_EPSILON) {
        strongly_coupled = strong_coupled * jxf_pow(0.5, NumLevels); // 0.5^L, L=0,1,2...
    } else {
        strongly_coupled = strong_coupled;
    }
    strongly_coupled2 = jxf_pow(strongly_coupled, 2);

    /*------------------------------------------*/
    /*  1. Form strongly coupled neighborhood   */
    /*------------------------------------------*/
    JXF_Int*     S_diag_i    = jxf_MAlloc((row + 1) * sizeof(JXF_Int));
    JXF_Int*     S_diag_j    = jxf_MAlloc(A_diag_nnz * sizeof(JXF_Int));
    JXF_Complex* S_diag_data = jxf_MAlloc(A_diag_nnz * sizeof(JXF_Complex));
    JXF_Int*     S_offd_i    = jxf_MAlloc((row + 1) * sizeof(JXF_Int));
    JXF_Int*     S_offd_j    = jxf_MAlloc(A_offd_nnz * sizeof(JXF_Int));
    // JXF_Int*     S_offd_global_j = jxf_MAlloc(A_offd_nnz * sizeof(JXF_Int));
    JXF_Complex* S_offd_data = jxf_MAlloc(A_offd_nnz * sizeof(JXF_Complex));

    /*
     * S1.1 Obtain diag
     */
    // get diag entries of diag matrix
    JXF_Complex* diag_abs = jxf_MAlloc(row * sizeof(JXF_Complex));
    jxf_CSRMatrixExtractDiagonal(A_diag, diag_abs, 1);

    // get diag entries of offd matrix
    JXF_Complex* remote_diag_abs = jxf_MAlloc(A_num_cols_offd * sizeof(JXF_Complex));
    jxf_ParCSRCommPkgGetRemoteData_Complex(A_comm_pkg, diag_abs, remote_diag_abs);

    /*
     * S1.2 Compute strongly coupled neighborhood matrix
     */
    index                        = 0;
    index_offd                   = 0;
    JXF_Int  global_col_index = 0;
    JXF_Int  S_num_cols_offd  = 0;
    JXF_Int* map              = jxf_CTAlloc(JXF_Int, A_num_cols_offd);
    JXF_Int* S_col_map_offd   = jxf_CTAlloc(JXF_Int, A_num_cols_offd);
    for (i = 0; i < A_num_cols_offd; i++) map[i] = -1;

    for (i = 0; i < row; ++i) {
        // diagonal matrix part
        S_diag_i[i] = index;
        row_start   = A_diag_i[i];
        row_end     = A_diag_i[i + 1];
        for (j = row_start; j < row_end; ++j) {
            if ((A_diag_j[j] == i) || (jxf_pow(A_diag_data[j], 2) > strongly_coupled2 * diag_abs[i] * diag_abs[A_diag_j[j]])) {
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
            if (jxf_pow(A_offd_data[j], 2) > strongly_coupled2 * diag_abs[i] * remote_diag_abs[A_offd_j[j]]) {
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
        jxf_BigQsort0(S_col_map_offd, 0, S_num_cols_offd - 1);
        for (j = 0; j < index_offd; ++j)
            S_offd_j[j] = jxf_BinarySearch(S_col_map_offd, S_offd_j[j], S_num_cols_offd); // jxf_BinarySearch for JXF_BigInt
        // jxf_TFree(S_offd_global_j);                                                    // free S_offd_global_j
    }

    // if (S_num_cols_offd == 0) {
    //     jxf_TFree(S_col_map_offd);
    //     S_col_map_offd = NULL;
    // } else if (S_num_cols_offd < A_num_cols_offd) {
    //     S_col_map_offd = jxf_TReAlloc(S_col_map_offd, JXF_Int, S_num_cols_offd);
    // }
    S_col_map_offd = jxf_TReAlloc(S_col_map_offd, JXF_Int, S_num_cols_offd);
    S_diag_j       = jxf_TReAlloc(S_diag_j, JXF_Int, index);
    S_diag_data    = jxf_TReAlloc(S_diag_data, JXF_Complex, index);
    S_offd_j       = jxf_TReAlloc(S_offd_j, JXF_Int, index_offd);
    S_offd_data    = jxf_TReAlloc(S_offd_data, JXF_Complex, index_offd);

    jxf_Free(diag_abs);
    jxf_Free(remote_diag_abs);
    jxf_Free(map);

    // Create S
    jxf_CSRMatrix* S_diag               = jxf_CSRMatrixCreate(row, row, index);
    jxf_CSRMatrixI(S_diag)              = S_diag_i;
    jxf_CSRMatrixJ(S_diag)              = S_diag_j;
    jxf_CSRMatrixData(S_diag)           = S_diag_data;
    // jxf_CSRMatrixMemoryLocation(S_diag) = location;

    jxf_CSRMatrix* S_offd  = jxf_CSRMatrixCreate(row, S_num_cols_offd, index_offd);
    jxf_CSRMatrixI(S_offd) = S_offd_i;
    if (S_num_cols_offd) {
        jxf_CSRMatrixJ(S_offd)              = S_offd_j;
        jxf_CSRMatrixData(S_offd)           = S_offd_data;
        // jxf_CSRMatrixMemoryLocation(S_offd) = location;
    }

    jxf_ParCSRMatrix* S = jxf_ParCSRMatrixCreate(comm, global_row, global_row, row_starts, row_starts, S_num_cols_offd, index, index_offd);
    jxf_CSRMatrixDestroy(jxf_ParCSRMatrixDiag(S));
    jxf_CSRMatrixDestroy(jxf_ParCSRMatrixOffd(S));
    jxf_ParCSRMatrixDiag(S)       = S_diag;
    jxf_ParCSRMatrixOffd(S)       = S_offd;
    jxf_ParCSRMatrixColMapOffd(S) = S_col_map_offd;

    /*-------------------------------------------------------*/
    /*  2. Get symbolic square of the connectivity matrix.   */
    /*-------------------------------------------------------*/
#if AGG_DEBUG > 1
    JXF_Real time_start, time_end, time;
    time_start = jxf_MPI_Wtime();
#endif

    // jxf_ParCSRMatrix* S2 = jxf_ParTMatmul(S, S); // S2 = S' * S
    // jxf_ParCSRMatrix* S2 = jxf_ParMatmul(S, S); // S2 = S * S
    jxf_ParCSRMatrix* S2 = jxf_ParMatmul_NozeroStruct(S, S);

#if AGG_DEBUG > 1
    time_end = jxf_MPI_Wtime();
    time     = time_end - time_start;
    if (iam == 0) {
        DEBUG_PRINTF("rank: %d, S2 = S * S time:  %.4f s\n", iam, time);
    }
#endif

#if AGG_DEBUG && 0
    // jxf_ParCSRMatrixPrint(A, "A.dat");
    // jxf_ParCSRMatrixPrint(S, "S.dat");
    // jxf_ParCSRMatrixPrint(S2, "S2.dat");
#endif

    /*-------------------------------------------------------*/
    /*  3. Apply PMIS algorithm to the symbolic square.      */
    /*-------------------------------------------------------*/
    JXF_Int        n_undone  = 0;
    jxf_CSRMatrix* S2_diag   = jxf_ParCSRMatrixDiag(S2);
    JXF_Int*       S2_diag_i = jxf_CSRMatrixI(S2_diag);
    JXF_Int*       S2_diag_j = jxf_CSRMatrixJ(S2_diag);

    jxf_CSRMatrix* S2_offd   = jxf_ParCSRMatrixOffd(S2);
    JXF_Int*       S2_offd_i = jxf_CSRMatrixI(S2_offd);
    JXF_Int*       S2_offd_j = jxf_CSRMatrixJ(S2_offd);

    JXF_Int  S2_num_cols_offd = jxf_CSRMatrixNumCols(S2_offd);
    JXF_Int* S2_col_map_offd  = jxf_ParCSRMatrixColMapOffd(S2);

    jxf_ParCSRCommPkg* S2_comm_pkg = jxf_ParCSRMatrixCommPkg(S2);
    if (!S2_comm_pkg) {
        jxf_MatvecCommPkgCreate(S2);
        S2_comm_pkg = jxf_ParCSRMatrixCommPkg(S2);
    }

#if AGG_DEBUG > 1
    DEBUG_PRINTF("rank: %d, row %d, row_starts = {%d, %d}, S2_num_cols_offd: %d\n", iam, row, row_starts[0], row_starts[1], S2_num_cols_offd);
#endif

    JXF_Int* loc_state = jxf_MAlloc(row * sizeof(JXF_Int));
    JXF_Int* loc_owner = jxf_MAlloc(row * sizeof(JXF_Int));

    JXF_Int num_sends     = jxf_ParCSRCommPkgNumSends(S2_comm_pkg); // Number of target processes sent
    JXF_Int num_recvs     = jxf_ParCSRCommPkgNumRecvs(S2_comm_pkg); // Number of target processes received
    JXF_Int sends_vec_len = jxf_ParCSRCommPkgSendMapStart(S2_comm_pkg, num_sends) - jxf_ParCSRCommPkgSendMapStart(S2_comm_pkg, 0);
    JXF_Int recvs_vec_len = jxf_ParCSRCommPkgRecvVecStart(S2_comm_pkg, num_recvs) - jxf_ParCSRCommPkgRecvVecStart(S2_comm_pkg, 0);
    // DEBUG_PRINTF("rank %d, num_recvs %d, recvs_vec_len %d, num_sends %d, sends_vec_len %d\n", iam, num_recvs, recvs_vec_len, num_sends, sends_vec_len);

    JXF_Int* rem_state = jxf_MAlloc(recvs_vec_len * sizeof(JXF_Int));
    JXF_Int* rem_owner = jxf_MAlloc(recvs_vec_len * sizeof(JXF_Int));

    for (i = 0; i < recvs_vec_len; ++i) {
        rem_state[i] = JXF_UNPT;
        rem_owner[i] = -1;
    }

    /*
     * S3.1 Remove lonely/isolated nodes in the sense of S2.
     * Note that isolation points do not participate in aggregation
     */
    JXF_Int max_nnz_per_row = 0; // Maximum non-zero element number in per row

#ifdef JXF_USING_OPENMP
#pragma omp parallel for  reduction(+ : n_undone)
#endif
    for (i = 0; i < row; ++i) {
        JXF_Int wl = S_diag_i[i + 1] - S_diag_i[i];
        JXF_Int wr = S2_offd_i[i + 1] - S2_offd_i[i];

        if (wl + wr == 1) {
            loc_state[i] = JXF_G0PT; // -5
            ++n_undone;
        } else {
            loc_state[i] = JXF_UNPT; // -1
        }
        max_nnz_per_row = jxf_max(max_nnz_per_row, wl + wr); // Maximum non-zero element number in per row

        loc_owner[i] = -1;
    }

    n_undone = row - n_undone;

    /*
     * S3.2 Exchange state
     */
    jxf_ParCSRCommPkgGetRemoteData_Int(S2_comm_pkg, loc_state, rem_state);

#if AGG_DEBUG > 2
    for (i = 0; i < recvs_vec_len; ++i) printf("after rank %d, rem_state[%d] = %d\n", iam, i, rem_state[i]);
#endif

    /*
     * S3.3 Start aggregation
     */
    JXF_Int  send_pts_offset = 10, send_pts_max_len = send_pts_offset * recvs_vec_len;
    JXF_Int* send_pts_num   = jxf_CTAlloc(JXF_Int, num_recvs);
    JXF_Int* send_pts_start = jxf_CTAlloc(JXF_Int, num_recvs + 1);
    JXF_Int* send_pts = jxf_CTAlloc(JXF_Int, send_pts_max_len); // max length,  at least twice, but when it equals twice, an error occurs
    JXF_Int* map_remote_rank2id = jxf_CTAlloc(JXF_Int, np);     // waste some memory here
    JXF_Int* recv_pts           = NULL;

    MPI_Request* send_cnt_req = jxf_CTAlloc(MPI_Request, num_recvs);
    MPI_Request* send_pts_req = jxf_CTAlloc(MPI_Request, num_recvs);

    JXF_Int  naggr = 0, n_undone_global = -1;
    JXF_Int  ibegin, iend, id = 0, c, c_remote_rank, c_loc_indx;
    JXF_Int  nbr_num         = 0;
    JXF_Int* nbr_arr         = jxf_MAlloc(max_nnz_per_row * sizeof(JXF_Int));
    JXF_Int* S2_remote_ranks = jxf_MAlloc(recvs_vec_len * sizeof(JXF_Int));

    // Auxiliary variables
    JXF_Int* S_offd_col_to_S2_offd_col = jxf_MAlloc(S_num_cols_offd * sizeof(JXF_Int));

    // max length (corresponding to send_pts),  at least twice, but when it equals twice, an error occurs, zhaoli, 2024.12.08
    for (i = 0; i < num_recvs + 1; ++i) send_pts_start[i] = jxf_ParCSRCommPkgRecvVecStart(S2_comm_pkg, i) * send_pts_offset;
    // for (i = 0; i < np; ++i) map_remote_rank2id[i] = -1; // debug
    for (i = 0; i < num_recvs; ++i) map_remote_rank2id[jxf_ParCSRCommPkgRecvProc(S2_comm_pkg, i)] = i;

    {
        JXF_Int* map_tmp = jxf_MAlloc(global_row * sizeof(JXF_Int));
        // for (i = 0; i < global_row; ++i) map_tmp[i] = -1; // debug
        for (i = 0; i < S2_num_cols_offd; ++i) map_tmp[S2_col_map_offd[i]] = i;
        for (i = 0; i < S_num_cols_offd; ++i) S_offd_col_to_S2_offd_col[i] = map_tmp[S_col_map_offd[i]];
        jxf_Free(map_tmp);
    }

    jxf_ParCSRCommPkgGetRemoteRankArray(S2_comm_pkg, S2_remote_ranks);
    // jxf_ParCSRCommPkgGetRemoteRankArray(S_comm_pkg, S_remote_ranks);

    while (jxf_true) {
        for (i = 0; i < num_recvs; ++i) send_pts_num[i] = 0;

        if (n_undone) {
            for (i = 0; i < row; ++i) {
                if (loc_state[i] != JXF_UNPT) continue;

                // Tackle boundary points and inner points
                if (S2_offd_i[i + 1] > S2_offd_i[i]) { // Boundary points

                    jxf_bool selectable = jxf_true;
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
                        if (rem_state[c] == JXF_UNPT && c_remote_rank > iam) {
                            selectable = jxf_false;
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
                            if (loc_state[c] == JXF_UNPT) --n_undone;
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
                                   __LINE__, iam, c_remote_rank, c, id, c_loc_indx, jxf_ParCSRCommPkgRecvProc(S2_comm_pkg, c_loc_indx), S2_col_map_offd[c],
                                   S_col_map_offd[S_offd_j[j]]);
                        }
#endif
                        // cord c and id
                        JXF_Int ID = send_pts_start[c_loc_indx] + send_pts_num[c_loc_indx];
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
                        if (c != i && loc_state[c] == JXF_UNPT) {
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
                        if (rem_state[c] == JXF_UNPT) {
                            rem_state[c] = id;

                            c_loc_indx = map_remote_rank2id[c_remote_rank];
#if AGG_DEBUG && 0
                            if (c_remote_rank == 2) {
                                printf("line %d: iam %d, c_remote_rank %d, c %d, id %d, c_loc_indx %d [send pro %d], S2_col_map_offd %d\n", __LINE__, iam,
                                       c_remote_rank, c, id, c_loc_indx, jxf_ParCSRCommPkgRecvProc(S2_comm_pkg, c_loc_indx), S2_col_map_offd[c]);
                            }
#endif
                            // cord c and id
                            JXF_Int ID = send_pts_start[c_loc_indx] + send_pts_num[c_loc_indx];
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

                        if (c != i && loc_state[c] != JXF_G0PT) {
                            if (loc_state[c] == JXF_UNPT) --n_undone;
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
                            if (c != k && loc_state[c] == JXF_UNPT) {
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
            JXF_Int npts = send_pts_num[i];
            JXF_Int ip   = jxf_ParCSRCommPkgRecvProc(S2_comm_pkg, i);
            jxf_MPI_Isend(&npts, 1, JXF_MPI_INT, ip, 0, comm, &send_cnt_req[i]);

            if (!npts) continue;

            jxf_MPI_Isend(&send_pts[send_pts_start[i]], npts, JXF_MPI_INT, ip, 1, comm, &send_pts_req[i]);
        }

        // Recv
        for (i = 0; i < num_sends; ++i) {
            JXF_Int npts;
            JXF_Int ip = jxf_ParCSRCommPkgSendProc(S2_comm_pkg, i);
            jxf_MPI_Recv(&npts, 1, JXF_MPI_INT, ip, 0, comm, MPI_STATUS_IGNORE);

            if (!npts) continue;

            recv_pts = jxf_CTAlloc(JXF_Int, npts);
            MPI_Recv(&recv_pts[0], npts, JXF_MPI_INT, ip, 1, comm, MPI_STATUS_IGNORE);

            // Update the loc_state of the current process
            for (k = 0; k < npts; k += 2) {
                c  = recv_pts[k] - jxf_ParCSRMatrixFirstRowIndex(S2);
                id = recv_pts[k + 1];
#if AGG_DEBUG > 1
                // DEBUG_PRINTF("rank: %d, c %d, ip %d, id %d\n", iam, c, ip, id);
                if (c >= row || c < 0) {
                    WARN_PRINTF("loc_state: rank[%d], c = %d out of range [0, %d], line: %d, %d - %d - %d\n", iam, c, row, __LINE__, recv_pts[k],
                                jxf_ParCSRMatrixFirstRowIndex(S2), id); // DEBUG
                }
#endif
                if (loc_state[c] == JXF_UNPT) --n_undone;

                loc_owner[c] = ip;
                loc_state[c] = id;
            }
            jxf_Free(recv_pts);
        }

#if AGG_DEBUG > 1
        for (i = 0; i < row; i++) {
            printf("rank[%d], loc_state[%d] = %d, loc_owner[%d] = %d\n", iam, i, loc_state[i], i, loc_owner[i]);
        }
#endif

        for (i = 0; i < num_recvs; ++i) {
            JXF_Int npts = send_pts_num[i];
            MPI_Wait(&send_cnt_req[i], MPI_STATUS_IGNORE);

            if (!npts) continue;
            MPI_Wait(&send_pts_req[i], MPI_STATUS_IGNORE);
        }

        // Exchange state
        jxf_ParCSRCommPkgGetRemoteData_Int(S2_comm_pkg, loc_state, rem_state);

        // check
        jxf_MPI_Allreduce(&n_undone, &n_undone_global, 1, JXF_MPI_INT, MPI_SUM, comm);
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
    jxf_ParCSRCommPkgGetRemoteData_Int(S2_comm_pkg, loc_owner, rem_owner);

    JXF_Int* new_id = jxf_CTAlloc(JXF_Int, naggr + 1);
    for (i = 0; i < row; ++i) {
        if (loc_owner[i] == iam && loc_state[i] >= 0) new_id[loc_state[i] + 1] = 1;
    }

    for (i = 0; i < recvs_vec_len; ++i) {
        if (rem_owner[i] == iam && rem_state[i] >= 0) new_id[rem_state[i] + 1] = 1;
    }

    // A part sum of a sequence
    for (i = 1; i < naggr + 1; ++i) new_id[i] += new_id[i - 1];

    JXF_Int diff = naggr - new_id[naggr], diff_global = 0;
    jxf_MPI_Allreduce(&diff, &diff_global, 1, JXF_MPI_INT, MPI_SUM, comm);

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
            JXF_Int npts = send_pts_num[i];
            JXF_Int ip   = jxf_ParCSRCommPkgRecvProc(S2_comm_pkg, i);
            jxf_MPI_Isend(&npts, 1, JXF_MPI_INT, ip, 0, comm, &send_cnt_req[i]);

            if (!npts) continue;

            jxf_MPI_Isend(&send_pts[send_pts_start[i]], npts, JXF_MPI_INT, ip, 1, comm, &send_pts_req[i]);
        }

        // Recv
        for (i = 0; i < num_sends; ++i) {
            JXF_Int npts;
            JXF_Int ip = jxf_ParCSRCommPkgSendProc(S2_comm_pkg, i);
            jxf_MPI_Recv(&npts, 1, JXF_MPI_INT, ip, 0, comm, MPI_STATUS_IGNORE);

            if (!npts) continue;

            recv_pts = jxf_CTAlloc(JXF_Int, npts);
            MPI_Recv(&recv_pts[0], npts, JXF_MPI_INT, ip, 1, comm, MPI_STATUS_IGNORE);

            // Update the loc_state of the current process
            for (k = 0; k < npts; k += 2) {
                c  = recv_pts[k] - jxf_ParCSRMatrixFirstRowIndex(A);
                id = recv_pts[k + 1];

                loc_state[c] = id;
            }
            jxf_Free(recv_pts);
        }

        for (i = 0; i < num_recvs; ++i) {
            JXF_Int npts = send_pts_num[i];
            MPI_Wait(&send_cnt_req[i], MPI_STATUS_IGNORE);

            if (!npts) continue;
            MPI_Wait(&send_pts_req[i], MPI_STATUS_IGNORE);
        }
    }

    //* cleanup
    // jxf_ParCSRMatrixDestroy(S);
    jxf_ParCSRMatrixDestroy(S2);

    jxf_Free(rem_state);
    jxf_Free(rem_owner);
    jxf_Free(nbr_arr);

    jxf_Free(S2_remote_ranks);
    jxf_Free(map_remote_rank2id);
    jxf_Free(S_offd_col_to_S2_offd_col);
    jxf_Free(send_pts_req);
    jxf_Free(send_cnt_req);

    jxf_Free(send_pts_num);
    jxf_Free(send_pts_start);
    jxf_Free(send_pts);

    jxf_Free(new_id);

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
 * \fn jxf_aggregation_symmpair(jxf_ParCSRMatrix* A, JXF_Int level_now, JXF_Int pair_number,
 *                                 JXF_Real quality_bound, JXF_Int min_num_coarse,
 *                                 JXF_Int* NumAggregates, JXF_Int** Vertices)
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
JXF_Int jxf_aggregation_symmpair(jxf_ParCSRMatrix* A, JXF_Int level_now, JXF_Int pair_number, JXF_Real quality_bound,
                                       JXF_Int min_num_coarse, JXF_Int* NumAggregates, JXF_Int** Vertices)

{
    jxf_CSRMatrix* A_diag      = jxf_ParCSRMatrixDiag(A);
    JXF_Int*       A_diag_i    = jxf_CSRMatrixI(A_diag);
    JXF_Int*       A_diag_j    = jxf_CSRMatrixJ(A_diag);
    JXF_Complex*   A_diag_data = jxf_CSRMatrixData(A_diag);

    JXF_Int            row      = jxf_CSRMatrixNumRows(A_diag);
    // JXF_MemoryLocation location = jxf_ParCSRMatrixMemoryLocation(A);

    JXF_Int  i, j, k, num_agg = 0, aggindex, num_coarse_global;
    JXF_Int  lvl     = 0;
    JXF_Real isorate = 0.0;

    JXF_Int            dopass = 0, domin = 0, domin_global = 0;
    JXF_Int            status   = JXF_SUCCESS;
    JXF_Int**          vertices = jxf_MAlloc(pair_number * sizeof(JXF_Int*));
    jxf_ParCSRMatrix** A_arr    = jxf_MAlloc(pair_number * sizeof(jxf_ParCSRMatrix*));
    jxf_ParCSRMatrix*  P        = NULL;
    //
    JXF_Int *riperm = NULL, *iperm = NULL;

    MPI_Comm comm = jxf_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);
    // JXF_Int bandwidth = fasp_dcsr_bandwidth(&mgl[level].A);
    // if (bandwidth > 5.0) param->quality_bound = quality_bound = 1.0 * bandwidth;

    /*-- setup permutation for renumbering --*/
    if (level_now == 0) {
        // JXF_Real t1 = jxf_MPI_Wtime();
        /*-- initialize permutation --*/
        riperm = jxf_MAlloc(row * sizeof(JXF_Int));
        iperm  = jxf_MAlloc(row * sizeof(JXF_Int));

        /*-- setup permutation for renumbering --*/
        jxf_nodes_renumbering(A, riperm, iperm);
        // JXF_Real t2 = jxf_MPI_Wtime();
        // IAM_0 { DEBUG_PRINTF("renumbering time: %e\n", t2 - t1); }
    }

    A_arr[lvl] = A;
    for (i = 1; i <= pair_number; ++i) {
        // DEBUG_PRINTF("row: %d, PW aggregation number: %d, pair_number: %d\n", row, i, pair_number);

        /*-- generate aggregations by pairwise matching --*/
        // if (i == 1)
        if (level_now == 0 && i == 1)
            status = jxf_form_pairwise_permutation(A_arr[lvl], i, quality_bound, riperm, iperm, &num_agg, &vertices[lvl]);
        else
            status = jxf_form_pairwise(A_arr[lvl], i, quality_bound, &num_agg, &vertices[lvl]);

        // compute the number of coarsen vertices
        jxf_MPI_Allreduce(&num_agg, &num_coarse_global, 1, JXF_MPI_INT, MPI_SUM, comm);

        if (num_coarse_global == 0) {
            if (iam == 0) {
                ERROR_PRINTF("%s %d: num_coarse_global = %d\n", __FUNCTION__, __LINE__, num_coarse_global);
            }
            exit(0);
        }

        /*-- check number of aggregates in the first pass --*/
        if (i == 1 && num_coarse_global < min_num_coarse) {
            for (domin = k = 0; k < row; k++) {
                if (vertices[lvl][k] == JXF_G0PT) domin++;
            }
            jxf_MPI_Allreduce(&domin, &domin_global, 1, JXF_MPI_INT, MPI_SUM, comm);
            isorate = (JXF_Real)num_coarse_global / domin_global;
            if (isorate < 0.1) {
                status = JXF_ERROR_AMG_COARSEING;
                // if (iam == 0) DEBUG_PRINTF("JXF_ERROR_AMG_COARSEING\n");
                goto END;
            }
        }

        if (i < pair_number) {

            /*-- Perform aggressive coarsening only up to the specified level --*/
            if (num_coarse_global < min_num_coarse) {
                // DEBUG_PRINTF("num_coarse_global: %d < min_num_coarse: %d\n", num_coarse_global, min_num_coarse);
                // for (JXF_Int jj = 0; jj < row; jj++) {
                //     printf("Vertices[%d] = %d\n", jj, vertices[lvl][jj]);
                // }
                lvl++;
                break;
            }

            /*-- Form Prolongation --*/
            jxf_form_tentative_p(A_arr[lvl], vertices[lvl], num_agg, num_coarse_global, &P);

            /*-- Form coarse level stiffness matrix --*/
            jxf_ParCSRMatrix* RT = P;
            jxf_BoomerAMGBuildCoarseOperator(RT, A_arr[lvl], P, &A_arr[lvl + 1]);

            jxf_ParCSRMatrixDestroy(P);
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
        jxf_ParCSRMatrixDestroy(A_arr[i]);
    }
    for (i = 1; i < lvl; ++i) {
        jxf_Free(vertices[i]);
    }
    jxf_Free(A_arr);
    jxf_Free(vertices);
    A_arr    = NULL;
    vertices = NULL;

    if (level_now == 0) {
        jxf_Free(riperm);
        jxf_Free(iperm);
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
 * \fn static JXF_Int jxf_nodes_renumbering(jxf_ParCSRMatrix* A, JXF_Int* riperm, JXF_Int* iperm)
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
static JXF_Int jxf_nodes_renumbering(jxf_ParCSRMatrix* A, JXF_Int* riperm, JXF_Int* iperm)
{
    jxf_CSRMatrix* A_diag   = jxf_ParCSRMatrixDiag(A);
    JXF_Int*       A_diag_i = jxf_CSRMatrixI(A_diag);
    JXF_Int*       A_diag_j = jxf_CSRMatrixJ(A_diag);

    jxf_CSRMatrix* A_offd          = jxf_ParCSRMatrixOffd(A);
    JXF_Int*       A_offd_i        = jxf_CSRMatrixI(A_offd);
    JXF_Int        num_cols_A_offd = jxf_CSRMatrixNumCols(A_offd);

    JXF_Int n = jxf_CSRMatrixNumRows(A_diag);

    // return status
    JXF_Int status = JXF_SUCCESS;

    JXF_Int  ifirst = 1, ilast = n;
    JXF_Int  mindg   = n + 1; // initialize minimum degree
    JXF_Int  dg_diag = 0, dg_offd = 0, dg;
    JXF_Int  i, i1, i2, jj, j, kk, j1, j2, ijs, ijs1, ijs2;
    JXF_Bool exc;

    MPI_Comm comm = jxf_ParCSRMatrixComm(A);
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
            exc = jxf_false;
            for (kk = ijs1 + 1; kk <= ijs2; kk++) {
                if (iperm[riperm[kk]] > iperm[riperm[kk - 1]]) { // iperm(riperm(kk)) > iperm(riperm(kk-1))
                    j              = riperm[kk];
                    riperm[kk]     = riperm[kk - 1];
                    riperm[kk - 1] = j;
                    exc            = jxf_true;
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
 * \fn static JXF_Int jxf_form_pairwise_permutation(jxf_ParCSRMatrix* A, JXF_Int num_pair,
 *                                            JXF_Real k_tg, JXF_Int* riperm, JXF_Int* iperm,
 *                                            JXF_Int* NumAggregates, JXF_Int** Vertices)
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
static JXF_Int jxf_form_pairwise_permutation(jxf_ParCSRMatrix* A, JXF_Int num_pair, JXF_Real k_tg, JXF_Int* riperm, JXF_Int* iperm,
                                                   JXF_Int* NumAggregates, JXF_Int** Vertices)
{
    jxf_CSRMatrix* A_diag      = jxf_ParCSRMatrixDiag(A);
    JXF_Int*       A_diag_i    = jxf_CSRMatrixI(A_diag);
    JXF_Int*       A_diag_j    = jxf_CSRMatrixJ(A_diag);
    JXF_Complex*   A_diag_data = jxf_CSRMatrixData(A_diag);

    jxf_CSRMatrix* A_offd          = jxf_ParCSRMatrixOffd(A);
    JXF_Int*       A_offd_i        = jxf_CSRMatrixI(A_offd);
    JXF_Int*       A_offd_j        = jxf_CSRMatrixJ(A_offd);
    JXF_Complex*   A_offd_data     = jxf_CSRMatrixData(A_offd);
    JXF_Int        num_cols_A_offd = jxf_CSRMatrixNumCols(A_offd);
    JXF_BigInt*    col_map_offd    = jxf_ParCSRMatrixColMapOffd(A);
    // num_nonzeros_offd = jxf_CSRMatrixNumNonzeros(offd);

    JXF_Int            row        = jxf_CSRMatrixNumRows(A_diag);
    JXF_Int            A_diag_nnz = jxf_CSRMatrixNumNonzeros(A_diag);
    // JXF_MemoryLocation location   = jxf_ParCSRMatrixMemoryLocation(A);

    // return status
    JXF_Int  status = JXF_SUCCESS;
    JXF_Int  i, j, old_i, row_start, row_end;
    JXF_Real sum;

    JXF_Int  col, ipair = -1;
    JXF_Real mu, min_mu, aii, ajj, aij;
    JXF_Real temp1, temp2, temp3, temp4;

    MPI_Comm comm = jxf_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

    JXF_Real checkdd = 0.5;
    checkdd             = jxf_max(jxf_abs(checkdd), (k_tg + 1) / (k_tg - 1));
    /*---------------------------------------------------------*/
    /* Step 1. select extremely strong diagonal dominate rows  */
    /*         and store in G0.                                */
    /*         G0        : vertices->val[i] = JXF_G0PT      */
    /*         Remaining : vertices->val[i] = JXF_UNPT      */
    /*---------------------------------------------------------*/
    JXF_Int* vertices = jxf_MAlloc(row * sizeof(JXF_Int));

    if (num_pair == 1) {
        for (i = 0; i < row; i++) {
            sum = 0.0; // row sum (excluding diagonal entry)

            // diagonal blocks
            row_start = A_diag_i[i];
            row_end   = A_diag_i[i + 1];
#if 1
            aii = A_diag_data[row_start]; // diagonal entry
            for (j = row_start + 1; j < row_end; j++) sum += jxf_abs(A_diag_data[j]);
#else
            for (j = row_start; j < row_end; j++) {
                if (A_diag_j[j] != i) {
                    sum += jxf_abs(A_diag_data[j]);
                } else
                    aii = A_diag_data[j]; // diagonal entry
            }
#endif

            // non-diagonal blocks
            if (num_cols_A_offd) {
                row_start = A_offd_i[i];
                row_end   = A_offd_i[i + 1];
                for (j = row_start; j < row_end; j++) {
                    // J = col_map_offd[offd_j[j]] + (JXF_BigInt)base_j;
                    sum += jxf_abs(A_offd_data[j]);
                }
            }

            if (aii >= checkdd * sum) {
                vertices[i] = JXF_G0PT;
            } else {
                vertices[i] = JXF_UNPT;
            }
        }
    } else {
        for (i = 0; i < row; ++i) vertices[i] = JXF_UNPT;
    }

    /*---------------------------------------------------------*/
    /* Step 2. compute row sum (off-diagonal) for each vertex  */
    /*---------------------------------------------------------*/
    JXF_Real* s    = jxf_MAlloc(row * sizeof(JXF_Real));
    JXF_Real* diag = jxf_MAlloc(row * sizeof(JXF_Real));

    for (i = 0; i < row; i++) {
        s[i]    = 0.0;
        diag[i] = 0.0;

        if (vertices[i] == JXF_G0PT) continue;

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
    JXF_Real rsi, rsj, sig1, sig2, epsr;
    JXF_Real repsmach = JXF_REAL_EPSILON;
    JXF_Real del1, eta1, del2, eta2, del12, vals, valp, tent, val = 0.0;
#endif

    for (i = 0; i < row; i++) { // cycle the new node number

        old_i = riperm[i]; // take old node number

        if (vertices[old_i] != JXF_UNPT) continue;

        min_mu = JXF_REAL_MAX;

        row_start = A_diag_i[old_i];
        row_end   = A_diag_i[old_i + 1];

        aii   = diag[old_i];
        ipair = -1;

#if 0
        for (j = row_start; j < row_end; j++) {
            col = A_diag_j[j];
            if (col == old_i) continue;
            if (vertices[col] != JXF_UNPT) continue;

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
            epsr = repsmach * jxf_abs(vals);

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
                if (jxf_abs(del1) < epsr && jxf_abs(del2) < epsr) {
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else if (jxf_abs(del1) < epsr) {
                    if (del2 < -epsr) continue;
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else if (jxf_abs(del2) < epsr) {
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
            if (vertices[col] != JXF_UNPT) continue;

            aij = A_diag_data[j];
            ajj = diag[col];

            temp1 = aii + s[old_i] + 2 * aij;
            temp2 = ajj + s[col] + 2 * aij;
            temp2 = 1.0 / temp1 + 1.0 / temp2;

            temp3 = jxf_max(jxf_abs(aii - s[old_i]), JXF_REAL_EPSILON); // avoid temp3 to be zero
            temp4 = jxf_max(jxf_abs(ajj - s[col]), JXF_REAL_EPSILON);   // avoid temp4 to be zero
            temp4 = -aij + 1. / (1.0 / temp3 + 1.0 / temp4);
            // avoid temp4 to be zero
            if (jxf_abs(temp4) < JXF_REAL_EPSILON) temp4 = (temp4 > 0) ? JXF_REAL_EPSILON : -JXF_REAL_EPSILON;

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

    jxf_Free(diag);
    jxf_Free(s);

    // END
    *Vertices = vertices;

    return status;
}

static JXF_Int jxf_form_pairwise(jxf_ParCSRMatrix* A, JXF_Int num_pair, JXF_Real k_tg, JXF_Int* NumAggregates, JXF_Int** Vertices)
{
    jxf_CSRMatrix* A_diag      = jxf_ParCSRMatrixDiag(A);
    JXF_Int*       A_diag_i    = jxf_CSRMatrixI(A_diag);
    JXF_Int*       A_diag_j    = jxf_CSRMatrixJ(A_diag);
    JXF_Complex*   A_diag_data = jxf_CSRMatrixData(A_diag);

    jxf_CSRMatrix* A_offd          = jxf_ParCSRMatrixOffd(A);
    JXF_Int*       A_offd_i        = jxf_CSRMatrixI(A_offd);
    JXF_Int*       A_offd_j        = jxf_CSRMatrixJ(A_offd);
    JXF_Complex*   A_offd_data     = jxf_CSRMatrixData(A_offd);
    JXF_Int        num_cols_A_offd = jxf_CSRMatrixNumCols(A_offd);
    JXF_BigInt*    col_map_offd    = jxf_ParCSRMatrixColMapOffd(A);
    // num_nonzeros_offd = jxf_CSRMatrixNumNonzeros(offd);

    JXF_Int            row        = jxf_CSRMatrixNumRows(A_diag);
    JXF_Int            A_diag_nnz = jxf_CSRMatrixNumNonzeros(A_diag);
    // JXF_MemoryLocation location   = jxf_ParCSRMatrixMemoryLocation(A);

    // return status
    JXF_Int  status = JXF_SUCCESS;
    JXF_Int  i, j, row_start, row_end;
    JXF_Real sum;

    JXF_Int  col, ipair = -1;
    JXF_Real mu, min_mu, aii, ajj, aij;
    JXF_Real temp1, temp2, temp3, temp4;

    MPI_Comm comm = jxf_ParCSRMatrixComm(A);
    MPI_CLAIM(comm);

    JXF_Real checkdd = 0.5;
    checkdd             = jxf_max(jxf_abs(checkdd), (k_tg + 1) / (k_tg - 1));
    /*---------------------------------------------------------*/
    /* Step 1. select extremely strong diagonal dominate rows  */
    /*         and store in G0.                                */
    /*         G0        : vertices->val[i] = JXF_G0PT      */
    /*         Remaining : vertices->val[i] = JXF_UNPT      */
    /*---------------------------------------------------------*/
    JXF_Int* vertices = jxf_MAlloc(row * sizeof(JXF_Int));

    if (num_pair == 1) {
        for (i = 0; i < row; i++) {
            sum = 0.0; // row sum (excluding diagonal entry)

            // diagonal blocks
            row_start = A_diag_i[i];
            row_end   = A_diag_i[i + 1];
#if 1
            aii = A_diag_data[row_start]; // diagonal entry
            for (j = row_start + 1; j < row_end; j++) sum += jxf_abs(A_diag_data[j]);
#else
            for (j = row_start; j < row_end; j++) {
                if (A_diag_j[j] != i) {
                    sum += jxf_abs(A_diag_data[j]);
                } else
                    aii = A_diag_data[j]; // diagonal entry
            }
#endif
            // non-diagonal blocks
            if (num_cols_A_offd) {
                row_start = A_offd_i[i];
                row_end   = A_offd_i[i + 1];
                for (j = row_start; j < row_end; j++) {
                    // J = col_map_offd[offd_j[j]] + (JXF_BigInt)base_j;
                    sum += jxf_abs(A_offd_data[j]);
                }
            }

            if (aii >= checkdd * sum) {
                vertices[i] = JXF_G0PT;
            } else {
                vertices[i] = JXF_UNPT;
            }
        }
    } else {
        for (i = 0; i < row; ++i) vertices[i] = JXF_UNPT;
    }

    /*---------------------------------------------------------*/
    /* Step 2. compute row sum (off-diagonal) for each vertex  */
    /*---------------------------------------------------------*/
    JXF_Real* s    = jxf_MAlloc(row * sizeof(JXF_Real));
    JXF_Real* diag = jxf_MAlloc(row * sizeof(JXF_Real));

    for (i = 0; i < row; i++) {
        s[i]    = 0.0;
        diag[i] = 0.0;

        if (vertices[i] == JXF_G0PT) continue;

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
    JXF_Real rsi, rsj, sig1, sig2, epsr;
    JXF_Real repsmach = JXF_REAL_EPSILON;
    JXF_Real del1, eta1, del2, eta2, del12, vals, valp, tent, val = 0.0;
#endif

    for (i = 0; i < row; i++) {

        if (vertices[i] != JXF_UNPT) continue;

        min_mu = JXF_REAL_MAX;

        row_start = A_diag_i[i];
        row_end   = A_diag_i[i + 1];

        aii   = diag[i];
        ipair = -1;

#if 0
        for (j = row_start; j < row_end; j++) {
            col = A_diag_j[j];
            if (col == i) continue;
            if (vertices[col] != JXF_UNPT) continue;

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
            epsr = repsmach * jxf_abs(vals);

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
                if (jxf_abs(del1) < epsr && jxf_abs(del2) < epsr) {
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else if (jxf_abs(del1) < epsr) {
                    if (del2 < -epsr) continue;
                    valp = 1.0 + (eta1 * eta2) / (vals * (eta1 + eta2));
                } else if (jxf_abs(del2) < epsr) {
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
            if (vertices[col] != JXF_UNPT) continue;

            aij = A_diag_data[j];
            ajj = diag[col];

            temp1 = aii + s[i] + 2 * aij;
            temp2 = ajj + s[col] + 2 * aij;
            temp2 = 1.0 / temp1 + 1.0 / temp2;

            temp3 = jxf_max(jxf_abs(aii - s[i]), JXF_REAL_EPSILON);   // avoid temp3 to be zero
            temp4 = jxf_max(jxf_abs(ajj - s[col]), JXF_REAL_EPSILON); // avoid temp4 to be zero
            temp4 = -aij + 1. / (1.0 / temp3 + 1.0 / temp4);
            // avoid temp4 to be zero
            if (jxf_abs(temp4) < JXF_REAL_EPSILON) temp4 = (temp4 > 0) ? JXF_REAL_EPSILON : -JXF_REAL_EPSILON;

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

    jxf_Free(diag);
    jxf_Free(s);

    // END
    *Vertices = vertices;

    return status;
}