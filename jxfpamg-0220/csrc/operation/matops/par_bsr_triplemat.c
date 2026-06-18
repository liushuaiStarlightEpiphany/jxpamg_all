//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_bsr_triplemat.c -- RAP for jxf_CSRMatrix class.
 *  Date: 2025/10/08
 */ 
/*--------------------------------------------------------------------------
 * used in RAP function - block size must be an argument because RAP_int may
 * by NULL
 *--------------------------------------------------------------------------*/

#include "jxf_parbsr_mv.h"


jxf_BSRMatrix* jxf_ExchangeRAPBlockData(jxf_BSRMatrix* RAP_int, jxf_ParCSRCommPkg* comm_pkg_RT, JXF_Int block_size)
{
    JXF_Int*     RAP_int_i;
    JXF_BigInt*  RAP_int_j    = NULL;
    JXF_Complex* RAP_int_data = NULL;
    JXF_Int      num_cols     = 0;

    MPI_Comm    comm            = jxf_ParCSRCommPkgComm(comm_pkg_RT);
    JXF_Int  num_recvs       = jxf_ParCSRCommPkgNumRecvs(comm_pkg_RT);
    JXF_Int* recv_procs      = jxf_ParCSRCommPkgRecvProcs(comm_pkg_RT);
    JXF_Int* recv_vec_starts = jxf_ParCSRCommPkgRecvVecStarts(comm_pkg_RT);
    JXF_Int  num_sends       = jxf_ParCSRCommPkgNumSends(comm_pkg_RT);
    JXF_Int* send_procs      = jxf_ParCSRCommPkgSendProcs(comm_pkg_RT);
    JXF_Int* send_map_starts = jxf_ParCSRCommPkgSendMapStarts(comm_pkg_RT);

    /*   JXF_Int block_size = jxf_BSRMatrixBlockSize(RAP_int); */

    jxf_BSRMatrix* RAP_ext;

    JXF_Int*     RAP_ext_i;
    JXF_BigInt*  RAP_ext_j    = NULL;
    JXF_Complex* RAP_ext_data = NULL;

    jxf_ParCSRCommHandle* comm_handle  = NULL;
    jxf_ParCSRCommPkg*    tmp_comm_pkg = NULL;

    JXF_Int* jdata_recv_vec_starts;
    JXF_Int* jdata_send_map_starts;

    JXF_Int num_rows;
    JXF_Int num_nonzeros;
    JXF_Int i, j, bnnz;
    JXF_Int num_procs, my_id;

    jxf_MPI_Comm_size(comm, &num_procs);
    jxf_MPI_Comm_rank(comm, &my_id);

    bnnz = block_size * block_size;

    RAP_ext_i             = jxf_CTAlloc(JXF_Int, send_map_starts[num_sends] + 1);
    jdata_recv_vec_starts = jxf_CTAlloc(JXF_Int, num_recvs + 1);
    jdata_send_map_starts = jxf_CTAlloc(JXF_Int, num_sends + 1);

    /*--------------------------------------------------------------------------
     * recompute RAP_int_i so that RAP_int_i[j+1] contains the number of
     * elements of row j (to be determined through send_map_elmnts on the
     * receiving end)
     *--------------------------------------------------------------------------*/

    if (num_recvs) {
        RAP_int_i    = jxf_BSRMatrixI(RAP_int);
        RAP_int_j    = jxf_BSRMatrixBigJ(RAP_int);
        RAP_int_data = jxf_BSRMatrixData(RAP_int);
        num_cols     = jxf_BSRMatrixNumCols(RAP_int);
    }
    jdata_recv_vec_starts[0] = 0;
    for (i = 0; i < num_recvs; i++) {
        jdata_recv_vec_starts[i + 1] = RAP_int_i[recv_vec_starts[i + 1]];
    }

    for (i = num_recvs; i > 0; i--)
        for (j = recv_vec_starts[i]; j > recv_vec_starts[i - 1]; j--) {
            RAP_int_i[j] -= RAP_int_i[j - 1];
        }

    /*--------------------------------------------------------------------------
     * initialize communication
     *--------------------------------------------------------------------------*/

    if (num_recvs && num_sends) {
        comm_handle = jxf_ParCSRCommHandleCreate(12, comm_pkg_RT, &RAP_int_i[1], &RAP_ext_i[1]);
    } else if (num_recvs) {
        comm_handle = jxf_ParCSRCommHandleCreate(12, comm_pkg_RT, &RAP_int_i[1], NULL);
    } else if (num_sends) {
        comm_handle = jxf_ParCSRCommHandleCreate(12, comm_pkg_RT, NULL, &RAP_ext_i[1]);
    }

    /* Create temporary communication package - note: send and recv are reversed */
    jxf_ParCSRCommPkgCreateAndFill(comm, num_sends, send_procs, jdata_send_map_starts, num_recvs, recv_procs, jdata_recv_vec_starts, NULL, &tmp_comm_pkg);

    jxf_ParCSRCommHandleDestroy(comm_handle);
    comm_handle = NULL;

    /*--------------------------------------------------------------------------
     * compute num_nonzeros for RAP_ext
     *--------------------------------------------------------------------------*/

    for (i = 0; i < num_sends; i++) {
        for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++) {
            RAP_ext_i[j + 1] += RAP_ext_i[j];
        }
    }

    num_rows     = send_map_starts[num_sends];
    num_nonzeros = RAP_ext_i[num_rows];
    if (num_nonzeros) {
        RAP_ext_j    = jxf_CTAlloc(JXF_BigInt, num_nonzeros);
        RAP_ext_data = jxf_CTAlloc(JXF_Complex, num_nonzeros * bnnz);
    }

    for (i = 0; i < num_sends + 1; i++) {
        jdata_send_map_starts[i] = RAP_ext_i[send_map_starts[i]];
    }

    comm_handle = jxf_ParBSRCommHandleCreate(1, bnnz, tmp_comm_pkg, (void*)RAP_int_data, (void*)RAP_ext_data);
    jxf_ParBSRCommHandleDestroy(comm_handle);
    comm_handle = NULL;

    comm_handle = jxf_ParCSRCommHandleCreate(21, tmp_comm_pkg, RAP_int_j, RAP_ext_j);
    RAP_ext     = jxf_BSRMatrixCreate(block_size, num_rows, num_cols, num_nonzeros);

    jxf_BSRMatrixI(RAP_ext) = RAP_ext_i;
    if (num_nonzeros) {
        jxf_BSRMatrixBigJ(RAP_ext) = RAP_ext_j;
        jxf_BSRMatrixData(RAP_ext) = RAP_ext_data;
    }

    /* Free memory */
    jxf_TFree(jdata_recv_vec_starts);
    jxf_TFree(jdata_send_map_starts);
    jxf_TFree(tmp_comm_pkg);
    jxf_ParCSRCommHandleDestroy(comm_handle);
    comm_handle = NULL;

    return RAP_ext;
}

/*--------------------------------------------------------------------------
 * jxf_ParBSRMatrixRAP
 *--------------------------------------------------------------------------*/

JXF_Int jxf_ParBSRMatrixRAP(jxf_ParBSRMatrix* RT, jxf_ParBSRMatrix* A, jxf_ParBSRMatrix* P, jxf_ParBSRMatrix** RAP_ptr)

{
    MPI_Comm comm = jxf_ParBSRMatrixComm(A);

    jxf_BSRMatrix*     RT_diag          = jxf_ParBSRMatrixDiag(RT);
    jxf_BSRMatrix*     RT_offd          = jxf_ParBSRMatrixOffd(RT);
    JXF_Int            num_cols_offd_RT = jxf_BSRMatrixNumCols(RT_offd);
    JXF_Int            num_rows_offd_RT = jxf_BSRMatrixNumRows(RT_offd);
    jxf_ParCSRCommPkg* comm_pkg_RT      = jxf_ParBSRMatrixCommPkg(RT);
    JXF_Int            num_recvs_RT     = 0;
    JXF_Int            num_sends_RT     = 0;
    JXF_Int*           send_map_starts_RT;
    JXF_Int*           send_map_elmts_RT;

    jxf_BSRMatrix* A_diag = jxf_ParBSRMatrixDiag(A);

    JXF_Complex* A_diag_data = jxf_BSRMatrixData(A_diag);
    JXF_Int*     A_diag_i    = jxf_BSRMatrixI(A_diag);
    JXF_Int*     A_diag_j    = jxf_BSRMatrixJ(A_diag);
    JXF_Int      block_size  = jxf_BSRMatrixBlockSize(A_diag);

    jxf_BSRMatrix* A_offd = jxf_ParBSRMatrixOffd(A);

    JXF_Complex* A_offd_data = jxf_BSRMatrixData(A_offd);
    JXF_Int*     A_offd_i    = jxf_BSRMatrixI(A_offd);
    JXF_Int*     A_offd_j    = jxf_BSRMatrixJ(A_offd);

    JXF_Int num_cols_diag_A = jxf_BSRMatrixNumCols(A_diag);
    JXF_Int num_cols_offd_A = jxf_BSRMatrixNumCols(A_offd);

    jxf_BSRMatrix* P_diag = jxf_ParBSRMatrixDiag(P);

    JXF_Complex* P_diag_data = jxf_BSRMatrixData(P_diag);
    JXF_Int*     P_diag_i    = jxf_BSRMatrixI(P_diag);
    JXF_Int*     P_diag_j    = jxf_BSRMatrixJ(P_diag);

    jxf_BSRMatrix* P_offd         = jxf_ParBSRMatrixOffd(P);
    JXF_BigInt*    col_map_offd_P = jxf_ParBSRMatrixColMapOffd(P);

    JXF_Complex* P_offd_data = jxf_BSRMatrixData(P_offd);
    JXF_Int*     P_offd_i    = jxf_BSRMatrixI(P_offd);
    JXF_Int*     P_offd_j    = jxf_BSRMatrixJ(P_offd);

    JXF_BigInt  first_col_diag_P = jxf_ParBSRMatrixFirstColDiag(P);
    JXF_BigInt  last_col_diag_P;
    JXF_Int     num_cols_diag_P     = jxf_BSRMatrixNumCols(P_diag);
    JXF_Int     num_cols_offd_P     = jxf_BSRMatrixNumCols(P_offd);
    JXF_BigInt* coarse_partitioning = jxf_ParBSRMatrixColStarts(P);
    JXF_BigInt  row_starts[2], col_starts[2];

    jxf_ParBSRMatrix* RAP;
    JXF_BigInt*       col_map_offd_RAP;

    jxf_BSRMatrix* RAP_int = NULL;

    JXF_Complex* RAP_int_data;
    JXF_Int*     RAP_int_i;
    JXF_BigInt*  RAP_int_j;

    jxf_BSRMatrix* RAP_ext;

    JXF_Complex* RAP_ext_data;
    JXF_Int*     RAP_ext_i;
    JXF_BigInt*  RAP_ext_j;

    jxf_BSRMatrix* RAP_diag;

    JXF_Complex* RAP_diag_data;
    JXF_Int*     RAP_diag_i;
    JXF_Int*     RAP_diag_j;

    jxf_BSRMatrix* RAP_offd;

    JXF_Complex* RAP_offd_data;
    JXF_Int*     RAP_offd_i;
    JXF_Int*     RAP_offd_j;

    JXF_Int RAP_size;
    JXF_Int RAP_ext_size;
    JXF_Int RAP_diag_size;
    JXF_Int RAP_offd_size;
    JXF_Int P_ext_diag_size;
    JXF_Int P_ext_offd_size;
    JXF_Int first_col_diag_RAP;
    JXF_Int last_col_diag_RAP;
    JXF_Int num_cols_offd_RAP = 0;

    jxf_BSRMatrix* R_diag;

    JXF_Complex* R_diag_data;
    JXF_Int*     R_diag_i;
    JXF_Int*     R_diag_j;

    jxf_BSRMatrix* R_offd;

    JXF_Complex* R_offd_data;
    JXF_Int*     R_offd_i;
    JXF_Int*     R_offd_j;

    jxf_BSRMatrix* Ps_ext;

    JXF_Complex* Ps_ext_data;
    JXF_Int*     Ps_ext_i;
    JXF_BigInt*  Ps_ext_j;

    JXF_Complex* P_ext_diag_data;
    JXF_Int*     P_ext_diag_i;
    JXF_Int*     P_ext_diag_j;

    JXF_Complex* P_ext_offd_data;
    JXF_Int*     P_ext_offd_i;
    JXF_Int*     P_ext_offd_j;

    JXF_BigInt* col_map_offd_Pext;
    JXF_Int*    map_P_to_Pext;
    JXF_Int*    map_P_to_RAP;
    JXF_Int*    map_Pext_to_RAP;

    JXF_Int*    P_marker;
    JXF_Int**   P_mark_array;
    JXF_Int**   A_mark_array;
    JXF_Int*    A_marker;
    JXF_BigInt* temp = NULL;

    JXF_Int n_coarse;
    JXF_Int num_cols_offd_Pext = 0;

    JXF_Int    ic, i, j, k, bnnz, kk;
    JXF_Int    i1, i2, i3, ii, ns, ne, size, rest;
    JXF_Int    cnt, cnt_offd, cnt_diag;
    JXF_Int    jj1, jj2, jj3, jcol;
    JXF_BigInt value;

    JXF_Int *jj_count, *jj_cnt_diag, *jj_cnt_offd;
    JXF_Int  jj_counter, jj_count_diag, jj_count_offd;
    JXF_Int  jj_row_begining, jj_row_begin_diag, jj_row_begin_offd;
    JXF_Int  start_indexing = 0; /* start indexing for RAP_data at 0 */
    JXF_Int  num_nz_cols_A;
    JXF_Int  num_procs;
    JXF_Int  num_threads, ind;

    JXF_Complex* r_entries;
    JXF_Complex* r_a_products;
    JXF_Complex* r_a_p_products;

    JXF_Complex zero = 0.0;

    /*-----------------------------------------------------------------------
     *  Copy ParCSRBlockMatrix RT into CSRBlockMatrix R so that we have
     *  row-wise access to restriction .
     *-----------------------------------------------------------------------*/

    jxf_MPI_Comm_size(comm, &num_procs);
    /* num_threads = jxf_NumThreads(); */
    num_threads = 1;

    bnnz           = block_size * block_size;
    r_a_products   = jxf_TAlloc(JXF_Complex, bnnz);
    r_a_p_products = jxf_TAlloc(JXF_Complex, bnnz);

    if (comm_pkg_RT) {
        num_recvs_RT       = jxf_ParCSRCommPkgNumRecvs(comm_pkg_RT);
        num_sends_RT       = jxf_ParCSRCommPkgNumSends(comm_pkg_RT);
        send_map_starts_RT = jxf_ParCSRCommPkgSendMapStarts(comm_pkg_RT);
        send_map_elmts_RT  = jxf_ParCSRCommPkgSendMapElmts(comm_pkg_RT);
    }

    jxf_BSRMatrixTranspose(RT_diag, &R_diag, 1);
    if (num_cols_offd_RT) {
        jxf_BSRMatrixTranspose(RT_offd, &R_offd, 1);
        R_offd_data = jxf_BSRMatrixData(R_offd);
        R_offd_i    = jxf_BSRMatrixI(R_offd);
        R_offd_j    = jxf_BSRMatrixJ(R_offd);
    }

    /*-----------------------------------------------------------------------
     *  Access the CSR vectors for R. Also get sizes of fine and
     *  coarse grids.
     *-----------------------------------------------------------------------*/

    R_diag_data = jxf_BSRMatrixData(R_diag);
    R_diag_i    = jxf_BSRMatrixI(R_diag);
    R_diag_j    = jxf_BSRMatrixJ(R_diag);

    n_coarse      = jxf_ParBSRMatrixGlobalNumCols(P);
    num_nz_cols_A = num_cols_diag_A + num_cols_offd_A;

    /*-----------------------------------------------------------------------
     *  Generate Ps_ext, i.e. portion of P that is stored on neighbor procs
     *  and needed locally for triple matrix product
     *-----------------------------------------------------------------------*/

    if (num_procs > 1) {
        Ps_ext      = jxf_ParBSRMatrixExtractBExt(P, A, 1);
        Ps_ext_data = jxf_BSRMatrixData(Ps_ext);
        Ps_ext_i    = jxf_BSRMatrixI(Ps_ext);
        Ps_ext_j    = jxf_BSRMatrixBigJ(Ps_ext);
    }

    P_ext_diag_i    = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
    P_ext_offd_i    = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
    P_ext_diag_size = 0;
    P_ext_offd_size = 0;
    last_col_diag_P = first_col_diag_P + (JXF_BigInt)num_cols_diag_P - 1;

    for (i = 0; i < num_cols_offd_A; i++) {
        for (j = Ps_ext_i[i]; j < Ps_ext_i[i + 1]; j++)
            if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P) {
                P_ext_offd_size++;
            } else {
                P_ext_diag_size++;
            }
        P_ext_diag_i[i + 1] = P_ext_diag_size;
        P_ext_offd_i[i + 1] = P_ext_offd_size;
    }

    if (P_ext_diag_size) {
        P_ext_diag_j    = jxf_CTAlloc(JXF_Int, P_ext_diag_size);
        P_ext_diag_data = jxf_CTAlloc(JXF_Complex, P_ext_diag_size * bnnz);
    }
    if (P_ext_offd_size) {
        P_ext_offd_j    = jxf_CTAlloc(JXF_Int, P_ext_offd_size);
        P_ext_offd_data = jxf_CTAlloc(JXF_Complex, P_ext_offd_size * bnnz);
    }

    cnt_offd = 0;
    cnt_diag = 0;
    cnt      = 0;
    for (i = 0; i < num_cols_offd_A; i++) {
        for (j = Ps_ext_i[i]; j < Ps_ext_i[i + 1]; j++)
            if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P) {
                Ps_ext_j[cnt_offd] = Ps_ext_j[j];
                for (kk = 0; kk < bnnz; kk++) {
                    P_ext_offd_data[cnt_offd * bnnz + kk] = Ps_ext_data[j * bnnz + kk];
                }
                cnt_offd++;
            } else {
                P_ext_diag_j[cnt_diag] = (JXF_Int)(Ps_ext_j[j] - first_col_diag_P);
                for (kk = 0; kk < bnnz; kk++) {
                    P_ext_diag_data[cnt_diag * bnnz + kk] = Ps_ext_data[j * bnnz + kk];
                }
                cnt_diag++;
            }
    }
    if (P_ext_offd_size || num_cols_offd_P) {
        temp = jxf_CTAlloc(JXF_BigInt, P_ext_offd_size + num_cols_offd_P);
        for (i = 0; i < P_ext_offd_size; i++) {
            temp[i] = Ps_ext_j[i];
        }
        cnt = P_ext_offd_size;
        for (i = 0; i < num_cols_offd_P; i++) {
            temp[cnt++] = col_map_offd_P[i];
        }
    }
    if (cnt) {
        jxf_BigQsort0(temp, 0, cnt - 1);

        num_cols_offd_Pext = 1;
        value              = temp[0];
        for (i = 1; i < cnt; i++) {
            if (temp[i] > value) {
                value                      = temp[i];
                temp[num_cols_offd_Pext++] = value;
            }
        }
    }

    if (num_cols_offd_Pext) {
        col_map_offd_Pext = jxf_CTAlloc(JXF_BigInt, num_cols_offd_Pext);
    }

    for (i = 0; i < num_cols_offd_Pext; i++) {
        col_map_offd_Pext[i] = temp[i];
    }

    if (P_ext_offd_size || num_cols_offd_P) {
        jxf_TFree(temp);
    }

    for (i = 0; i < P_ext_offd_size; i++) P_ext_offd_j[i] = jxf_BigBinarySearch(col_map_offd_Pext, Ps_ext_j[i], num_cols_offd_Pext);
    if (num_cols_offd_P) {
        map_P_to_Pext = jxf_CTAlloc(JXF_Int, num_cols_offd_P);

        cnt = 0;
        for (i = 0; i < num_cols_offd_Pext; i++)
            if (col_map_offd_Pext[i] == col_map_offd_P[cnt]) {
                map_P_to_Pext[cnt++] = i;
                if (cnt == num_cols_offd_P) {
                    break;
                }
            }
    }

    if (num_procs > 1) {
        jxf_BSRMatrixDestroy(Ps_ext);
        Ps_ext = NULL;
    }

    /*-----------------------------------------------------------------------
     *  First Pass: Determine size of RAP_int and set up RAP_int_i if there
     *  are more than one processor and nonzero elements in R_offd
     *-----------------------------------------------------------------------*/

    P_mark_array = jxf_CTAlloc(JXF_Int*, num_threads);
    A_mark_array = jxf_CTAlloc(JXF_Int*, num_threads);

    if (num_cols_offd_RT) {
        jj_count = jxf_CTAlloc(JXF_Int, num_threads);

        for (ii = 0; ii < num_threads; ii++) {
            size = num_cols_offd_RT / num_threads;
            rest = num_cols_offd_RT - size * num_threads;
            if (ii < rest) {
                ns = ii * size + ii;
                ne = (ii + 1) * size + ii + 1;
            } else {
                ns = ii * size + rest;
                ne = (ii + 1) * size + rest;
            }

            /*--------------------------------------------------------------------
             *  Allocate marker arrays.
             *--------------------------------------------------------------------*/

            if (num_cols_offd_Pext || num_cols_diag_P) {
                P_mark_array[ii] = jxf_CTAlloc(JXF_Int, num_cols_diag_P + num_cols_offd_Pext);
                P_marker         = P_mark_array[ii];
            }
            A_mark_array[ii] = jxf_CTAlloc(JXF_Int, num_nz_cols_A);
            A_marker         = A_mark_array[ii];

            /*--------------------------------------------------------------------
             *  Initialize some stuff.
             *--------------------------------------------------------------------*/

            jj_counter = start_indexing;
            for (ic = 0; ic < num_cols_diag_P + num_cols_offd_Pext; ic++) {
                P_marker[ic] = -1;
            }
            for (i = 0; i < num_nz_cols_A; i++) {
                A_marker[i] = -1;
            }

            /*--------------------------------------------------------------------
             *  Loop over exterior c-points
             *--------------------------------------------------------------------*/

            for (ic = ns; ic < ne; ic++) {

                jj_row_begining = jj_counter;

                /*-----------------------------------------------------------------
                 *  Loop over entries in row ic of R_offd.
                 *-----------------------------------------------------------------*/

                for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic + 1]; jj1++) {
                    i1 = R_offd_j[jj1];

                    /*--------------------------------------------------------------
                     *  Loop over entries in row i1 of A_offd.
                     *--------------------------------------------------------------*/

                    for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++) {
                        i2 = A_offd_j[jj2];

                        /*-----------------------------------------------------------
                         *  Check A_marker to see if point i2 has been previously
                         *  visited. New entries in RAP only occur from unmarked points.
                         *-----------------------------------------------------------*/

                        if (A_marker[i2] != ic) {

                            /*--------------------------------------------------------
                             *  Mark i2 as visited.
                             *--------------------------------------------------------*/

                            A_marker[i2] = ic;

                            /*--------------------------------------------------------
                             *  Loop over entries in row i2 of P_ext.
                             *--------------------------------------------------------*/

                            for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++) {
                                i3 = P_ext_diag_j[jj3];

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, mark it and
                                 *  increment counter.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = jj_counter;
                                    jj_counter++;
                                }
                            }
                            for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++) {
                                i3 = P_ext_offd_j[jj3] + num_cols_diag_P;

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, mark it and
                                 *  increment counter.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = jj_counter;
                                    jj_counter++;
                                }
                            }
                        }
                    }
                    /*--------------------------------------------------------------
                     *  Loop over entries in row i1 of A_diag.
                     *--------------------------------------------------------------*/

                    for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++) {
                        i2 = A_diag_j[jj2];

                        /*-----------------------------------------------------------
                         *  Check A_marker to see if point i2 has been previously
                         *  visited. New entries in RAP only occur from unmarked
                         * points.
                         *-----------------------------------------------------------*/

                        if (A_marker[i2 + num_cols_offd_A] != ic) {

                            /*--------------------------------------------------------
                             *  Mark i2 as visited.
                             *--------------------------------------------------------*/

                            A_marker[i2 + num_cols_offd_A] = ic;

                            /*--------------------------------------------------------
                             *  Loop over entries in row i2 of P_diag.
                             *--------------------------------------------------------*/

                            for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++) {
                                i3 = P_diag_j[jj3];

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, mark it and
                                 *  increment counter.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = jj_counter;
                                    jj_counter++;
                                }
                            }

                            /*--------------------------------------------------------
                             *  Loop over entries in row i2 of P_offd.
                             *--------------------------------------------------------*/

                            for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++) {
                                i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, mark it and
                                 *  increment counter.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = jj_counter;
                                    jj_counter++;
                                }
                            }
                        }
                    }
                }
            }
            jj_count[ii] = jj_counter;
        }

        /*-----------------------------------------------------------------------
         *  Allocate RAP_int_data and RAP_int_j arrays.
         *-----------------------------------------------------------------------*/

        for (i = 0; i < num_threads - 1; i++) {
            jj_count[i + 1] += jj_count[i];
        }

        RAP_size                    = jj_count[num_threads - 1];
        RAP_int_i                   = jxf_CTAlloc(JXF_Int, num_cols_offd_RT + 1);
        RAP_int_data                = jxf_CTAlloc(JXF_Complex, RAP_size * bnnz);
        RAP_int_j                   = jxf_CTAlloc(JXF_BigInt, RAP_size);
        RAP_int_i[num_cols_offd_RT] = RAP_size;

        /*-----------------------------------------------------------------------
         *  Second Pass: Fill in RAP_int_data and RAP_int_j.
         *-----------------------------------------------------------------------*/

        for (ii = 0; ii < num_threads; ii++) {
            size = num_cols_offd_RT / num_threads;
            rest = num_cols_offd_RT - size * num_threads;
            if (ii < rest) {
                ns = ii * size + ii;
                ne = (ii + 1) * size + ii + 1;
            } else {
                ns = ii * size + rest;
                ne = (ii + 1) * size + rest;
            }

            /*--------------------------------------------------------------------
             *  Initialize some stuff.
             *--------------------------------------------------------------------*/

            if (num_cols_offd_Pext || num_cols_diag_P) {
                P_marker = P_mark_array[ii];
            }
            A_marker = A_mark_array[ii];

            jj_counter = start_indexing;
            if (ii > 0) {
                jj_counter = jj_count[ii - 1];
            }

            for (ic = 0; ic < num_cols_diag_P + num_cols_offd_Pext; ic++) {
                P_marker[ic] = -1;
            }
            for (i = 0; i < num_nz_cols_A; i++) {
                A_marker[i] = -1;
            }

            /*--------------------------------------------------------------------
             *  Loop over exterior c-points.
             *--------------------------------------------------------------------*/

            for (ic = ns; ic < ne; ic++) {
                jj_row_begining = jj_counter;
                RAP_int_i[ic]   = jj_counter;

                /*-----------------------------------------------------------------
                 *  Loop over entries in row ic of R_offd.
                 *-----------------------------------------------------------------*/

                for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic + 1]; jj1++) {
                    i1        = R_offd_j[jj1];
                    r_entries = &(R_offd_data[jj1 * bnnz]);

                    /*--------------------------------------------------------------
                     *  Loop over entries in row i1 of A_offd.
                     *--------------------------------------------------------------*/

                    for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++) {
                        i2 = A_offd_j[jj2];
                        jxf_BSRMatrixBlockMultAdd(r_entries, &(A_offd_data[jj2 * bnnz]), zero, r_a_products, block_size);

                        /*-----------------------------------------------------------
                         *  Check A_marker to see if point i2 has been previously
                         *  visited.New entries in RAP only occur from unmarked points.
                         *-----------------------------------------------------------*/

                        if (A_marker[i2] != ic) {
                            /*--------------------------------------------------------
                             *  Mark i2 as visited.
                             *--------------------------------------------------------*/

                            A_marker[i2] = ic;

                            /*--------------------------------------------------------
                             *  Loop over entries in row i2 of P_ext.
                             *--------------------------------------------------------*/

                            for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++) {
                                i3 = P_ext_diag_j[jj3];
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_ext_diag_data[jj3 * bnnz]), zero, r_a_p_products, block_size);

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, create a new
                                 *  entry. If it has, add new contribution.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = jj_counter;
                                    for (kk = 0; kk < bnnz; kk++) RAP_int_data[jj_counter * bnnz + kk] = r_a_p_products[kk];
                                    RAP_int_j[jj_counter] = i3 + first_col_diag_P;
                                    jj_counter++;
                                } else {
                                    for (kk = 0; kk < bnnz; kk++) RAP_int_data[P_marker[i3] * bnnz + kk] += r_a_p_products[kk];
                                }
                            }
                            for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++) {
                                i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_ext_offd_data[jj3 * bnnz]), zero, r_a_p_products, block_size);

                                /*--------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, create a new
                                 *  entry. If it has, add new contribution.
                                 *--------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = jj_counter;
                                    for (kk = 0; kk < bnnz; kk++) RAP_int_data[jj_counter * bnnz + kk] = r_a_p_products[kk];
                                    RAP_int_j[jj_counter] = col_map_offd_Pext[i3 - num_cols_diag_P];
                                    jj_counter++;
                                } else {
                                    for (kk = 0; kk < bnnz; kk++) RAP_int_data[P_marker[i3] * bnnz + kk] += r_a_p_products[kk];
                                }
                            }
                        }

                        /*-----------------------------------------------------------
                         *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                         *  no new entries in RAP and can just add new contributions.
                         *-----------------------------------------------------------*/

                        else {
                            for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++) {
                                i3 = P_ext_diag_j[jj3];
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_ext_diag_data[jj3 * bnnz]), zero, r_a_p_products, block_size);
                                for (kk = 0; kk < bnnz; kk++) RAP_int_data[P_marker[i3] * bnnz + kk] += r_a_p_products[kk];
                            }
                            for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++) {
                                i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_ext_offd_data[jj3 * bnnz]), zero, r_a_p_products, block_size);
                                ind = P_marker[i3] * bnnz;
                                for (kk = 0; kk < bnnz; kk++) {
                                    RAP_int_data[ind++] += r_a_p_products[kk];
                                }
                            }
                        }
                    }

                    /*--------------------------------------------------------------
                     *  Loop over entries in row i1 of A_diag.
                     *--------------------------------------------------------------*/

                    for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++) {
                        i2 = A_diag_j[jj2];
                        jxf_BSRMatrixBlockMultAdd(r_entries, &(A_diag_data[jj2 * bnnz]), zero, r_a_products, block_size);

                        /*-----------------------------------------------------------
                         *  Check A_marker to see if point i2 has been previously
                         *  visited. New entries in RAP only occur from unmarked points.
                         *-----------------------------------------------------------*/

                        if (A_marker[i2 + num_cols_offd_A] != ic) {

                            /*--------------------------------------------------------
                             *  Mark i2 as visited.
                             *--------------------------------------------------------*/

                            A_marker[i2 + num_cols_offd_A] = ic;

                            /*--------------------------------------------------------
                             *  Loop over entries in row i2 of P_diag.
                             *--------------------------------------------------------*/

                            for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++) {
                                i3 = P_diag_j[jj3];
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_diag_data[jj3 * bnnz]), zero, r_a_p_products, block_size);

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, create a new
                                 *  entry. If it has, add new contribution.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = jj_counter;
                                    ind          = jj_counter * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_int_data[ind++] = r_a_p_products[kk];
                                    }
                                    RAP_int_j[jj_counter] = (JXF_BigInt)i3 + first_col_diag_P;
                                    jj_counter++;
                                } else {
                                    ind = P_marker[i3] * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_int_data[ind++] += r_a_p_products[kk];
                                    }
                                }
                            }
                            for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++) {
                                i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_offd_data[jj3 * bnnz]), zero, r_a_p_products, block_size);

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, create a new
                                 *  entry. If it has, add new contribution.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begining) {
                                    P_marker[i3] = jj_counter;
                                    ind          = jj_counter * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_int_data[ind++] = r_a_p_products[kk];
                                    }
                                    RAP_int_j[jj_counter] = col_map_offd_Pext[i3 - num_cols_diag_P];
                                    jj_counter++;
                                } else {
                                    ind = P_marker[i3] * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_int_data[ind++] += r_a_p_products[kk];
                                    }
                                }
                            }
                        }

                        /*-----------------------------------------------------------
                         *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                         *  no new entries in RAP and can just add new contributions.
                         *-----------------------------------------------------------*/

                        else {
                            for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++) {
                                i3 = P_diag_j[jj3];
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_diag_data[jj3 * bnnz]), zero, r_a_p_products, block_size);
                                ind = P_marker[i3] * bnnz;
                                for (kk = 0; kk < bnnz; kk++) {
                                    RAP_int_data[ind++] += r_a_p_products[kk];
                                }
                            }
                            for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++) {
                                i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_offd_data[jj3 * bnnz]), zero, r_a_p_products, block_size);
                                ind = P_marker[i3] * bnnz;
                                for (kk = 0; kk < bnnz; kk++) {
                                    RAP_int_data[ind++] += r_a_p_products[kk];
                                }
                            }
                        }
                    }
                }
            }
            if (num_cols_offd_Pext || num_cols_diag_P) {
                jxf_TFree(P_mark_array[ii]);
            }
            jxf_TFree(A_mark_array[ii]);
        }

        RAP_int                       = jxf_BSRMatrixCreate(block_size, num_cols_offd_RT, num_rows_offd_RT, RAP_size);
        jxf_BSRMatrixI(RAP_int)    = RAP_int_i;
        jxf_BSRMatrixBigJ(RAP_int) = RAP_int_j;
        jxf_BSRMatrixData(RAP_int) = RAP_int_data;
        jxf_TFree(jj_count);
    }

    RAP_ext_size = 0;
    if (num_sends_RT || num_recvs_RT) {
        RAP_ext      = jxf_ExchangeRAPBlockData(RAP_int, comm_pkg_RT, block_size);
        RAP_ext_i    = jxf_BSRMatrixI(RAP_ext);
        RAP_ext_j    = jxf_BSRMatrixBigJ(RAP_ext);
        RAP_ext_data = jxf_BSRMatrixData(RAP_ext);
        RAP_ext_size = RAP_ext_i[jxf_BSRMatrixNumRows(RAP_ext)];
    }
    if (num_cols_offd_RT) {
        jxf_BSRMatrixDestroy(RAP_int);
        RAP_int = NULL;
    }

    RAP_diag_i = jxf_CTAlloc(JXF_Int, num_cols_diag_P + 1);
    RAP_offd_i = jxf_CTAlloc(JXF_Int, num_cols_diag_P + 1);

    first_col_diag_RAP = first_col_diag_P;
    last_col_diag_RAP  = first_col_diag_P + (JXF_BigInt)num_cols_diag_P - 1;

    /*-----------------------------------------------------------------------
     *  check for new nonzero columns in RAP_offd generated through RAP_ext
     *-----------------------------------------------------------------------*/

    if (RAP_ext_size || num_cols_offd_Pext) {
        temp = jxf_CTAlloc(JXF_BigInt, RAP_ext_size + num_cols_offd_Pext);
        cnt  = 0;
        for (i = 0; i < RAP_ext_size; i++)
            if (RAP_ext_j[i] < first_col_diag_RAP || RAP_ext_j[i] > last_col_diag_RAP) {
                temp[cnt++] = RAP_ext_j[i];
            }
        for (i = 0; i < num_cols_offd_Pext; i++) {
            temp[cnt++] = col_map_offd_Pext[i];
        }

        if (cnt) {
            jxf_BigQsort0(temp, 0, cnt - 1);
            value             = temp[0];
            num_cols_offd_RAP = 1;
            for (i = 1; i < cnt; i++) {
                if (temp[i] > value) {
                    value                     = temp[i];
                    temp[num_cols_offd_RAP++] = value;
                }
            }
        }

        /* now evaluate col_map_offd_RAP */
        if (num_cols_offd_RAP) {
            col_map_offd_RAP = jxf_CTAlloc(JXF_BigInt, num_cols_offd_RAP);
        }

        for (i = 0; i < num_cols_offd_RAP; i++) {
            col_map_offd_RAP[i] = temp[i];
        }

        jxf_TFree(temp);
    }

    if (num_cols_offd_P) {
        map_P_to_RAP = jxf_CTAlloc(JXF_Int, num_cols_offd_P);

        cnt = 0;
        for (i = 0; i < num_cols_offd_RAP; i++)
            if (col_map_offd_RAP[i] == col_map_offd_P[cnt]) {
                map_P_to_RAP[cnt++] = i;
                if (cnt == num_cols_offd_P) {
                    break;
                }
            }
    }

    if (num_cols_offd_Pext) {
        map_Pext_to_RAP = jxf_CTAlloc(JXF_Int, num_cols_offd_Pext);

        cnt = 0;
        for (i = 0; i < num_cols_offd_RAP; i++)
            if (col_map_offd_RAP[i] == col_map_offd_Pext[cnt]) {
                map_Pext_to_RAP[cnt++] = i;
                if (cnt == num_cols_offd_Pext) {
                    break;
                }
            }
    }

    /*-----------------------------------------------------------------------
     *  Convert RAP_ext column indices
     *-----------------------------------------------------------------------*/

    for (i = 0; i < RAP_ext_size; i++)
        if (RAP_ext_j[i] < first_col_diag_RAP || RAP_ext_j[i] > last_col_diag_RAP)
            RAP_ext_j[i] = (JXF_BigInt)(num_cols_diag_P) + jxf_BigBinarySearch(col_map_offd_RAP, RAP_ext_j[i], num_cols_offd_RAP);
        else {
            RAP_ext_j[i] -= first_col_diag_RAP;
        }

    /*-----------------------------------------------------------------------
     *  Initialize some stuff.
     *-----------------------------------------------------------------------*/

    jj_cnt_diag = jxf_CTAlloc(JXF_Int, num_threads);
    jj_cnt_offd = jxf_CTAlloc(JXF_Int, num_threads);

    for (ii = 0; ii < num_threads; ii++) {
        size = num_cols_diag_P / num_threads;
        rest = num_cols_diag_P - size * num_threads;
        if (ii < rest) {
            ns = ii * size + ii;
            ne = (ii + 1) * size + ii + 1;
        } else {
            ns = ii * size + rest;
            ne = (ii + 1) * size + rest;
        }

        P_mark_array[ii] = jxf_CTAlloc(JXF_Int, num_cols_diag_P + num_cols_offd_RAP);
        A_mark_array[ii] = jxf_CTAlloc(JXF_Int, num_nz_cols_A);
        P_marker         = P_mark_array[ii];
        A_marker         = A_mark_array[ii];
        jj_count_diag    = start_indexing;
        jj_count_offd    = start_indexing;

        for (ic = 0; ic < num_cols_diag_P + num_cols_offd_RAP; ic++) {
            P_marker[ic] = -1;
        }
        for (i = 0; i < num_nz_cols_A; i++) {
            A_marker[i] = -1;
        }

        /*-----------------------------------------------------------------------
         *  Loop over interior c-points.
         *-----------------------------------------------------------------------*/

        for (ic = ns; ic < ne; ic++) {

            /*--------------------------------------------------------------------
             *  Set marker for diagonal entry, RAP_{ic,ic}. and for all points
             *  being added to row ic of RAP_diag and RAP_offd through RAP_ext
             *--------------------------------------------------------------------*/

            P_marker[ic]      = jj_count_diag;
            jj_row_begin_diag = jj_count_diag;
            jj_row_begin_offd = jj_count_offd;
            jj_count_diag++;

            for (i = 0; i < num_sends_RT; i++)
                for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i + 1]; j++)
                    if (send_map_elmts_RT[j] == ic) {
                        for (k = RAP_ext_i[j]; k < RAP_ext_i[j + 1]; k++) {
                            jcol = (JXF_Int)RAP_ext_j[k];
                            if (jcol < num_cols_diag_P) {
                                if (P_marker[jcol] < jj_row_begin_diag) {
                                    P_marker[jcol] = jj_count_diag;
                                    jj_count_diag++;
                                }
                            } else {
                                if (P_marker[jcol] < jj_row_begin_offd) {
                                    P_marker[jcol] = jj_count_offd;
                                    jj_count_offd++;
                                }
                            }
                        }
                        break;
                    }

            /*-----------------------------------------------------------------
             *  Loop over entries in row ic of R_diag.
             *-----------------------------------------------------------------*/

            for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic + 1]; jj1++) {
                i1 = R_diag_j[jj1];

                /*-----------------------------------------------------------------
                 *  Loop over entries in row i1 of A_offd.
                 *-----------------------------------------------------------------*/

                if (num_cols_offd_A) {
                    for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++) {
                        i2 = A_offd_j[jj2];

                        /*-----------------------------------------------------------
                         *  Check A_marker to see if point i2 has been previously
                         *  visited.New entries in RAP only occur from unmarked points.
                         *-----------------------------------------------------------*/

                        if (A_marker[i2] != ic) {
                            /*--------------------------------------------------------
                             *  Mark i2 as visited.
                             *--------------------------------------------------------*/

                            A_marker[i2] = ic;

                            /*--------------------------------------------------------
                             *  Loop over entries in row i2 of P_ext.
                             *--------------------------------------------------------*/

                            for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++) {
                                i3 = P_ext_diag_j[jj3];

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, mark it and
                                 *  increment counter.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begin_diag) {
                                    P_marker[i3] = jj_count_diag;
                                    jj_count_diag++;
                                }
                            }
                            for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++) {
                                i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, mark it and
                                 *  increment counter.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begin_offd) {
                                    P_marker[i3] = jj_count_offd;
                                    jj_count_offd++;
                                }
                            }
                        }
                    }
                }

                /*-----------------------------------------------------------------
                 *  Loop over entries in row i1 of A_diag.
                 *-----------------------------------------------------------------*/

                for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++) {
                    i2 = A_diag_j[jj2];

                    /*--------------------------------------------------------------
                     *  Check A_marker to see if point i2 has been previously
                     *  visited. New entries in RAP only occur from unmarked points.
                     *--------------------------------------------------------------*/

                    if (A_marker[i2 + num_cols_offd_A] != ic) {

                        /*-----------------------------------------------------------
                         *  Mark i2 as visited.
                         *-----------------------------------------------------------*/

                        A_marker[i2 + num_cols_offd_A] = ic;

                        /*-----------------------------------------------------------
                         *  Loop over entries in row i2 of P_diag.
                         *-----------------------------------------------------------*/

                        for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++) {
                            i3 = P_diag_j[jj3];

                            /*--------------------------------------------------------
                             *  Check P_marker to see that RAP_{ic,i3} has not already
                             *  been accounted for. If it has not, mark it and increment
                             *  counter.
                             *--------------------------------------------------------*/

                            if (P_marker[i3] < jj_row_begin_diag) {
                                P_marker[i3] = jj_count_diag;
                                jj_count_diag++;
                            }
                        }

                        /*-----------------------------------------------------------
                         *  Loop over entries in row i2 of P_offd.
                         *-----------------------------------------------------------*/

                        if (num_cols_offd_P) {
                            for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++) {
                                i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, mark it and
                                 *  increment counter.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begin_offd) {
                                    P_marker[i3] = jj_count_offd;
                                    jj_count_offd++;
                                }
                            }
                        }
                    }
                }
            }

            /*--------------------------------------------------------------------
             * Set RAP_diag_i and RAP_offd_i for this row.
             *--------------------------------------------------------------------*/
        }
        jj_cnt_diag[ii] = jj_count_diag;
        jj_cnt_offd[ii] = jj_count_offd;
    }

    for (i = 0; i < num_threads - 1; i++) {
        jj_cnt_diag[i + 1] += jj_cnt_diag[i];
        jj_cnt_offd[i + 1] += jj_cnt_offd[i];
    }

    jj_count_diag = jj_cnt_diag[num_threads - 1];
    jj_count_offd = jj_cnt_offd[num_threads - 1];

    RAP_diag_i[num_cols_diag_P] = jj_count_diag;
    RAP_offd_i[num_cols_diag_P] = jj_count_offd;

    /*-----------------------------------------------------------------------
     *  Allocate RAP_diag_data and RAP_diag_j arrays.
     *  Allocate RAP_offd_data and RAP_offd_j arrays.
     *-----------------------------------------------------------------------*/

    RAP_diag_size = jj_count_diag;
    if (RAP_diag_size) {
        RAP_diag_data = jxf_CTAlloc(JXF_Complex, RAP_diag_size * bnnz);
        RAP_diag_j    = jxf_CTAlloc(JXF_Int, RAP_diag_size);
    }

    RAP_offd_size = jj_count_offd;
    if (RAP_offd_size) {
        RAP_offd_data = jxf_CTAlloc(JXF_Complex, RAP_offd_size * bnnz);
        RAP_offd_j    = jxf_CTAlloc(JXF_Int, RAP_offd_size);
    }

    if (RAP_offd_size == 0 && num_cols_offd_RAP != 0) {
        num_cols_offd_RAP = 0;
        jxf_TFree(col_map_offd_RAP);
    }

    /*-----------------------------------------------------------------------
     *  Second Pass: Fill in RAP_diag_data and RAP_diag_j.
     *  Second Pass: Fill in RAP_offd_data and RAP_offd_j.
     *-----------------------------------------------------------------------*/

    for (ii = 0; ii < num_threads; ii++) {
        size = num_cols_diag_P / num_threads;
        rest = num_cols_diag_P - size * num_threads;
        if (ii < rest) {
            ns = ii * size + ii;
            ne = (ii + 1) * size + ii + 1;
        } else {
            ns = ii * size + rest;
            ne = (ii + 1) * size + rest;
        }

        /*-----------------------------------------------------------------------
         *  Initialize some stuff.
         *-----------------------------------------------------------------------*/

        P_marker = P_mark_array[ii];
        A_marker = A_mark_array[ii];
        for (ic = 0; ic < num_cols_diag_P + num_cols_offd_RAP; ic++) {
            P_marker[ic] = -1;
        }
        for (i = 0; i < num_nz_cols_A; i++) {
            A_marker[i] = -1;
        }

        jj_count_diag = start_indexing;
        jj_count_offd = start_indexing;
        if (ii > 0) {
            jj_count_diag = jj_cnt_diag[ii - 1];
            jj_count_offd = jj_cnt_offd[ii - 1];
        }

        /*-----------------------------------------------------------------------
         *  Loop over interior c-points.
         *-----------------------------------------------------------------------*/

        for (ic = ns; ic < ne; ic++) {

            /*--------------------------------------------------------------------
             *  Create diagonal entry, RAP_{ic,ic} and add entries of RAP_ext
             *--------------------------------------------------------------------*/

            P_marker[ic]      = jj_count_diag;
            jj_row_begin_diag = jj_count_diag;
            jj_row_begin_offd = jj_count_offd;
            RAP_diag_i[ic]    = jj_row_begin_diag;
            RAP_offd_i[ic]    = jj_row_begin_offd;
            ind               = jj_count_diag * bnnz;
            for (kk = 0; kk < bnnz; kk++) {
                RAP_diag_data[ind++] = zero;
            }
            RAP_diag_j[jj_count_diag] = ic;
            jj_count_diag++;

            for (i = 0; i < num_sends_RT; i++)
                for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i + 1]; j++)
                    if (send_map_elmts_RT[j] == ic) {
                        for (k = RAP_ext_i[j]; k < RAP_ext_i[j + 1]; k++) {
                            jcol = (JXF_Int)RAP_ext_j[k];
                            if (jcol < num_cols_diag_P) {
                                if (P_marker[jcol] < jj_row_begin_diag) {
                                    P_marker[jcol] = jj_count_diag;
                                    ind            = jj_count_diag * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_diag_data[ind++] = RAP_ext_data[k * bnnz + kk];
                                    }
                                    RAP_diag_j[jj_count_diag] = jcol;
                                    jj_count_diag++;
                                } else {
                                    ind = P_marker[jcol] * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_diag_data[ind++] += RAP_ext_data[k * bnnz + kk];
                                    }
                                }
                            } else {
                                if (P_marker[jcol] < jj_row_begin_offd) {
                                    P_marker[jcol] = jj_count_offd;
                                    ind            = jj_count_offd * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_offd_data[ind++] = RAP_ext_data[k * bnnz + kk];
                                    }
                                    RAP_offd_j[jj_count_offd] = jcol - num_cols_diag_P;
                                    jj_count_offd++;
                                } else {
                                    ind = P_marker[jcol] * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_offd_data[ind++] += RAP_ext_data[k * bnnz + kk];
                                    }
                                }
                            }
                        }
                        break;
                    }

            /*--------------------------------------------------------------------
             *  Loop over entries in row ic of R_diag.
             *--------------------------------------------------------------------*/

            for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic + 1]; jj1++) {
                i1        = R_diag_j[jj1];
                r_entries = &(R_diag_data[jj1 * bnnz]);

                /*-----------------------------------------------------------------
                 *  Loop over entries in row i1 of A_offd.
                 *-----------------------------------------------------------------*/

                if (num_cols_offd_A) {
                    for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++) {
                        i2 = A_offd_j[jj2];
                        jxf_BSRMatrixBlockMultAdd(r_entries, &(A_offd_data[jj2 * bnnz]), zero, r_a_products, block_size);

                        /*-----------------------------------------------------------
                         *  Check A_marker to see if point i2 has been previously
                         *  visited.New entries in RAP only occur from unmarked points.
                         *-----------------------------------------------------------*/

                        if (A_marker[i2] != ic) {
                            /*--------------------------------------------------------
                             *  Mark i2 as visited.
                             *--------------------------------------------------------*/

                            A_marker[i2] = ic;

                            /*--------------------------------------------------------
                             *  Loop over entries in row i2 of P_ext.
                             *--------------------------------------------------------*/

                            for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++) {
                                i3 = P_ext_diag_j[jj3];
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_ext_diag_data[jj3 * bnnz]), zero, r_a_p_products, block_size);

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, create a new
                                 *  entry. If it has, add new contribution.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begin_diag) {
                                    P_marker[i3] = jj_count_diag;
                                    ind          = jj_count_diag * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_diag_data[ind++] = r_a_p_products[kk];
                                    }
                                    RAP_diag_j[jj_count_diag] = i3;
                                    jj_count_diag++;
                                } else {
                                    ind = P_marker[i3] * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_diag_data[ind++] += r_a_p_products[kk];
                                    }
                                }
                            }
                            for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++) {
                                i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_ext_offd_data[jj3 * bnnz]), zero, r_a_p_products, block_size);

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not
                                 *  been accounted for. If it has not, create a new
                                 *  entry. If it has, add new contribution.
                                 *-----------------------------------------------------*/
                                if (P_marker[i3] < jj_row_begin_offd) {
                                    P_marker[i3] = jj_count_offd;
                                    ind          = jj_count_offd * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_offd_data[ind++] = r_a_p_products[kk];
                                    }
                                    RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                                    jj_count_offd++;
                                } else {
                                    ind = P_marker[i3] * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_offd_data[ind++] += r_a_p_products[kk];
                                    }
                                }
                            }
                        }

                        /*-----------------------------------------------------------
                         *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                         *  no new entries in RAP and can just add new contributions.
                         *-----------------------------------------------------------*/
                        else {
                            for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++) {
                                i3 = P_ext_diag_j[jj3];
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_ext_diag_data[jj3 * bnnz]), zero, r_a_p_products, block_size);
                                ind = P_marker[i3] * bnnz;
                                for (kk = 0; kk < bnnz; kk++) {
                                    RAP_diag_data[ind++] += r_a_p_products[kk];
                                }
                            }
                            for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++) {
                                i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_ext_offd_data[jj3 * bnnz]), zero, r_a_p_products, block_size);
                                ind = P_marker[i3] * bnnz;
                                for (kk = 0; kk < bnnz; kk++) {
                                    RAP_offd_data[ind++] += r_a_p_products[kk];
                                }
                            }
                        }
                    }
                }

                /*-----------------------------------------------------------------
                 *  Loop over entries in row i1 of A_diag.
                 *-----------------------------------------------------------------*/

                for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++) {
                    i2 = A_diag_j[jj2];
                    jxf_BSRMatrixBlockMultAdd(r_entries, &(A_diag_data[jj2 * bnnz]), zero, r_a_products, block_size);

                    /*--------------------------------------------------------------
                     *  Check A_marker to see if point i2 has been previously
                     *  visited. New entries in RAP only occur from unmarked points.
                     *--------------------------------------------------------------*/

                    if (A_marker[i2 + num_cols_offd_A] != ic) {

                        /*-----------------------------------------------------------
                         *  Mark i2 as visited.
                         *-----------------------------------------------------------*/

                        A_marker[i2 + num_cols_offd_A] = ic;

                        /*-----------------------------------------------------------
                         *  Loop over entries in row i2 of P_diag.
                         *-----------------------------------------------------------*/

                        for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++) {
                            i3 = P_diag_j[jj3];
                            jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_diag_data[jj3 * bnnz]), zero, r_a_p_products, block_size);

                            /*--------------------------------------------------------
                             *  Check P_marker to see that RAP_{ic,i3} has not already
                             *  been accounted for. If it has not, create a new entry.
                             *  If it has, add new contribution.
                             *--------------------------------------------------------*/

                            if (P_marker[i3] < jj_row_begin_diag) {
                                P_marker[i3] = jj_count_diag;
                                ind          = jj_count_diag * bnnz;
                                for (kk = 0; kk < bnnz; kk++) {
                                    RAP_diag_data[ind++] = r_a_p_products[kk];
                                }
                                RAP_diag_j[jj_count_diag] = P_diag_j[jj3];
                                jj_count_diag++;
                            } else {
                                ind = P_marker[i3] * bnnz;
                                for (kk = 0; kk < bnnz; kk++) {
                                    RAP_diag_data[ind++] += r_a_p_products[kk];
                                }
                            }
                        }
                        if (num_cols_offd_P) {
                            for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++) {
                                i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_offd_data[jj3 * bnnz]), zero, r_a_p_products, block_size);

                                /*-----------------------------------------------------
                                 *  Check P_marker to see that RAP_{ic,i3} has not already
                                 *  been accounted for. If it has not, create a new entry.
                                 *  If it has, add new contribution.
                                 *-----------------------------------------------------*/

                                if (P_marker[i3] < jj_row_begin_offd) {
                                    P_marker[i3] = jj_count_offd;
                                    ind          = jj_count_offd * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_offd_data[ind++] = r_a_p_products[kk];
                                    }
                                    RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                                    jj_count_offd++;
                                } else {
                                    ind = P_marker[i3] * bnnz;
                                    for (kk = 0; kk < bnnz; kk++) {
                                        RAP_offd_data[ind++] += r_a_p_products[kk];
                                    }
                                }
                            }
                        }
                    }

                    /*--------------------------------------------------------------
                     *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                     *  no new entries in RAP and can just add new contributions.
                     *--------------------------------------------------------------*/

                    else {
                        for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++) {
                            i3 = P_diag_j[jj3];
                            jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_diag_data[jj3 * bnnz]), zero, r_a_p_products, block_size);
                            ind = P_marker[i3] * bnnz;
                            for (kk = 0; kk < bnnz; kk++) {
                                RAP_diag_data[ind++] += r_a_p_products[kk];
                            }
                        }
                        if (num_cols_offd_P) {
                            for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++) {
                                i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                                jxf_BSRMatrixBlockMultAdd(r_a_products, &(P_offd_data[jj3 * bnnz]), zero, r_a_p_products, block_size);
                                ind = P_marker[i3] * bnnz;
                                for (kk = 0; kk < bnnz; kk++) {
                                    RAP_offd_data[ind++] += r_a_p_products[kk];
                                }
                            }
                        }
                    }
                }
            }
        }
        jxf_TFree(P_mark_array[ii]);
        jxf_TFree(A_mark_array[ii]);
    }

    for (i = 0; i < 2; i++) {
        row_starts[i] = col_starts[i] = coarse_partitioning[i];
    }

    RAP = jxf_ParBSRMatrixCreate(comm, block_size, n_coarse, n_coarse, row_starts, col_starts, num_cols_offd_RAP, RAP_diag_size, RAP_offd_size);

    RAP_diag                    = jxf_ParBSRMatrixDiag(RAP);
    jxf_BSRMatrixI(RAP_diag) = RAP_diag_i;
    if (RAP_diag_size) {
        jxf_BSRMatrixData(RAP_diag) = RAP_diag_data;
        jxf_BSRMatrixJ(RAP_diag)    = RAP_diag_j;
    }

    RAP_offd                    = jxf_ParBSRMatrixOffd(RAP);
    jxf_BSRMatrixI(RAP_offd) = RAP_offd_i;
    if (num_cols_offd_RAP) {
        jxf_BSRMatrixData(RAP_offd)     = RAP_offd_data;
        jxf_BSRMatrixJ(RAP_offd)        = RAP_offd_j;
        jxf_ParBSRMatrixColMapOffd(RAP) = col_map_offd_RAP;
    }
    if (num_procs > 1) {
        jxf_BlockMatvecCommPkgCreate(RAP);
    }

    *RAP_ptr = RAP;

    /*-----------------------------------------------------------------------
     *  Free R, P_ext and marker arrays.
     *-----------------------------------------------------------------------*/

    jxf_BSRMatrixDestroy(R_diag);
    R_diag = NULL;

    if (num_cols_offd_RT) {
        jxf_BSRMatrixDestroy(R_offd);
        R_offd = NULL;
    }

    if (num_sends_RT || num_recvs_RT) {
        jxf_BSRMatrixDestroy(RAP_ext);
        RAP_ext = NULL;
    }
    jxf_TFree(P_mark_array);
    jxf_TFree(A_mark_array);
    jxf_TFree(P_ext_diag_i);
    jxf_TFree(P_ext_offd_i);
    jxf_TFree(jj_cnt_diag);
    jxf_TFree(jj_cnt_offd);
    if (num_cols_offd_P) {
        jxf_TFree(map_P_to_Pext);
        jxf_TFree(map_P_to_RAP);
    }
    if (num_cols_offd_Pext) {
        jxf_TFree(col_map_offd_Pext);
        jxf_TFree(map_Pext_to_RAP);
    }
    if (P_ext_diag_size) {
        jxf_TFree(P_ext_diag_data);
        jxf_TFree(P_ext_diag_j);
    }
    if (P_ext_offd_size) {
        jxf_TFree(P_ext_offd_data);
        jxf_TFree(P_ext_offd_j);
    }

    jxf_TFree(r_a_products);
    jxf_TFree(r_a_p_products);

    return jxf_error_flag;
}
