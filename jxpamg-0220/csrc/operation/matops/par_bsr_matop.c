//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_bsr_matop.c --  Matrix opterators for the par_bsr matrix
 *  Date: 2025/10/08
 */ 

#include "jx_parbsr_mv.h"

/* ----------------------------------------------------------------------
 * jx_ParBSRMatrixExtractBExt: extracts rows from B which are located on
 * other processors and needed for multiplication with A locally. The rows
 * are returned as BSRMatrix.
 * ---------------------------------------------------------------------*/

jx_BSRMatrix* jx_ParBSRMatrixExtractBExt(jx_ParBSRMatrix* B, jx_ParBSRMatrix* A, JX_Int data)
{
    MPI_Comm       comm           = jx_ParBSRMatrixComm(B);
    JX_BigInt  first_col_diag = jx_ParBSRMatrixFirstColDiag(B);
    JX_BigInt* col_map_offd   = jx_ParBSRMatrixColMapOffd(B);
    JX_Int     block_size     = jx_ParBSRMatrixBlockSize(B);

    jx_ParCSRCommPkg* comm_pkg        = jx_ParBSRMatrixCommPkg(A);
    JX_Int            num_recvs       = jx_ParCSRCommPkgNumRecvs(comm_pkg);
    JX_Int*           recv_vec_starts = jx_ParCSRCommPkgRecvVecStarts(comm_pkg);
    JX_Int            num_sends       = jx_ParCSRCommPkgNumSends(comm_pkg);
    JX_Int*           send_map_starts = jx_ParCSRCommPkgSendMapStarts(comm_pkg);
    JX_Int*           send_map_elmts  = jx_ParCSRCommPkgSendMapElmts(comm_pkg);

    jx_ParCSRCommHandle* comm_handle;
    jx_ParCSRCommPkg*    tmp_comm_pkg = NULL;

    jx_BSRMatrix* diag = jx_ParBSRMatrixDiag(B);

    JX_Int*     diag_i    = jx_BSRMatrixI(diag);
    JX_Int*     diag_j    = jx_BSRMatrixJ(diag);
    JX_Complex* diag_data = jx_BSRMatrixData(diag);

    jx_BSRMatrix* offd = jx_ParBSRMatrixOffd(B);

    JX_Int*     offd_i    = jx_BSRMatrixI(offd);
    JX_Int*     offd_j    = jx_BSRMatrixJ(offd);
    JX_Complex* offd_data = jx_BSRMatrixData(offd);

    JX_Int*     B_int_i;
    JX_BigInt*  B_int_j;
    JX_Complex* B_int_data;

    JX_Int num_cols_B, num_nonzeros;
    JX_Int num_rows_B_ext;
    JX_Int num_procs, my_id;

    jx_BSRMatrix* B_ext;

    JX_Int*     B_ext_i;
    JX_BigInt*  B_ext_j;
    JX_Complex* B_ext_data;

    JX_Int* jdata_recv_vec_starts;
    JX_Int* jdata_send_map_starts;

    JX_Int i, j, k, l, counter, bnnz;
    JX_Int start_index;
    JX_Int j_cnt, jrow;

    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);

    bnnz           = block_size * block_size;
    num_cols_B     = jx_ParCSRMatrixGlobalNumCols(B);
    num_rows_B_ext = recv_vec_starts[num_recvs];
    B_int_i        = jx_CTAlloc(JX_Int, send_map_starts[num_sends] + 1);
    B_ext_i        = jx_CTAlloc(JX_Int, num_rows_B_ext + 1);
    /*--------------------------------------------------------------------------
     * generate B_int_i through adding number of row-elements of offd and diag
     * for corresponding rows. B_int_i[j+1] contains the number of elements of
     * a row j (which is determined through send_map_elmts)
     *--------------------------------------------------------------------------*/
    B_int_i[0]   = 0;
    j_cnt        = 0;
    num_nonzeros = 0;
    for (i = 0; i < num_sends; i++) {
        for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++) {
            jrow             = send_map_elmts[j];
            B_int_i[++j_cnt] = offd_i[jrow + 1] - offd_i[jrow] + diag_i[jrow + 1] - diag_i[jrow];
            num_nonzeros += B_int_i[j_cnt];
        }
    }

    /*--------------------------------------------------------------------------
     * initialize communication
     *--------------------------------------------------------------------------*/
    comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, &B_int_i[1], &B_ext_i[1]);

    B_int_j = jx_CTAlloc(JX_BigInt, num_nonzeros);
    if (data) {
        B_int_data = jx_CTAlloc(JX_Complex, num_nonzeros * bnnz);
    }

    jdata_send_map_starts    = jx_CTAlloc(JX_Int, num_sends + 1);
    jdata_recv_vec_starts    = jx_CTAlloc(JX_Int, num_recvs + 1);
    start_index              = B_int_i[0];
    jdata_send_map_starts[0] = start_index;
    counter                  = 0;
    for (i = 0; i < num_sends; i++) {
        num_nonzeros = counter;
        for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++) {
            jrow = send_map_elmts[j];
            for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++) {
                B_int_j[counter] = (JX_BigInt)diag_j[k] + first_col_diag;
                if (data) {
                    for (l = 0; l < bnnz; l++) {
                        B_int_data[counter * bnnz + l] = diag_data[k * bnnz + l];
                    }
                }
                counter++;
            }
            for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++) {
                B_int_j[counter] = col_map_offd[offd_j[k]];
                if (data) {
                    for (l = 0; l < bnnz; l++) B_int_data[counter * bnnz + l] = offd_data[k * bnnz + l];
                }
                counter++;
            }
        }
        num_nonzeros = counter - num_nonzeros;
        start_index += num_nonzeros;
        jdata_send_map_starts[i + 1] = start_index;
    }

    /* Create temporary communication package */
    jx_ParCSRCommPkgCreateAndFill(comm, num_recvs, jx_ParCSRCommPkgRecvProcs(comm_pkg), jdata_recv_vec_starts, num_sends,
                                      jx_ParCSRCommPkgSendProcs(comm_pkg), jdata_send_map_starts, NULL, &tmp_comm_pkg);

    jx_ParCSRCommHandleDestroy(comm_handle);
    comm_handle = NULL;

    /*--------------------------------------------------------------------------
     * after communication exchange B_ext_i[j+1] contains the number of elements
     * of a row j !
     * evaluate B_ext_i and compute num_nonzeros for B_ext
     *--------------------------------------------------------------------------*/

    for (i = 0; i < num_recvs; i++) {
        for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++) {
            B_ext_i[j + 1] += B_ext_i[j];
        }
    }

    num_nonzeros = B_ext_i[num_rows_B_ext];

    B_ext   = jx_BSRMatrixCreate(block_size, num_rows_B_ext, num_cols_B, num_nonzeros);
    B_ext_j = jx_CTAlloc(JX_BigInt, num_nonzeros);
    if (data) {
        B_ext_data = jx_CTAlloc(JX_Complex, num_nonzeros * bnnz);
    }

    for (i = 0; i < num_recvs; i++) {
        start_index                  = B_ext_i[recv_vec_starts[i]];
        num_nonzeros                 = B_ext_i[recv_vec_starts[i + 1]] - start_index;
        jdata_recv_vec_starts[i + 1] = B_ext_i[recv_vec_starts[i + 1]];
    }

    comm_handle = jx_ParCSRCommHandleCreate(21, tmp_comm_pkg, B_int_j, B_ext_j);
    jx_ParCSRCommHandleDestroy(comm_handle);
    comm_handle = NULL;

    if (data) {
        comm_handle = jx_ParBSRCommHandleCreate(1, bnnz, tmp_comm_pkg, B_int_data, B_ext_data);
        jx_ParBSRCommHandleDestroy(comm_handle);
        comm_handle = NULL;
    }

    jx_BSRMatrixI(B_ext)    = B_ext_i;
    jx_BSRMatrixBigJ(B_ext) = B_ext_j;
    if (data) {
        jx_BSRMatrixData(B_ext) = B_ext_data;
    }

    /* Free memory */
    jx_TFree(jdata_send_map_starts);
    jx_TFree(jdata_recv_vec_starts);
    jx_TFree(tmp_comm_pkg);
    jx_TFree(B_int_i);
    jx_TFree(B_int_j);
    if (data) {
        jx_TFree(B_int_data);
    }

    return B_ext;
}

/*--------------------------------------------------------------------------
 * jx_ParBSRMatrixReorder:
 *
 * Reorders the column and data arrays of a the diagonal component of a square
 * ParBSR matrix, such that the first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

JX_Int jx_ParBSRMatrixReorder(jx_ParBSRMatrix* A)
{
    JX_BigInt     nrows_A = jx_ParBSRMatrixGlobalNumRows(A);
    JX_BigInt     ncols_A = jx_ParBSRMatrixGlobalNumCols(A);
    jx_BSRMatrix* A_diag  = jx_ParBSRMatrixDiag(A);

    if (nrows_A != ncols_A) {
        jx_error_w_msg(JX_ERROR_GENERIC, " Error! Matrix should be square!\n");
        return jx_error_flag;
    }

    jx_BSRMatrixReorder(A_diag);

    return jx_error_flag;
}
