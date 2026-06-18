//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_bsr_matop.c --  Matrix opterators for the par_bsr matrix
 *  Date: 2025/10/08
 */ 

#include "jxf_parbsr_mv.h"

/* ----------------------------------------------------------------------
 * jxf_ParBSRMatrixExtractBExt: extracts rows from B which are located on
 * other processors and needed for multiplication with A locally. The rows
 * are returned as BSRMatrix.
 * ---------------------------------------------------------------------*/

jxf_BSRMatrix* jxf_ParBSRMatrixExtractBExt(jxf_ParBSRMatrix* B, jxf_ParBSRMatrix* A, JXF_Int data)
{
    MPI_Comm       comm           = jxf_ParBSRMatrixComm(B);
    JXF_BigInt  first_col_diag = jxf_ParBSRMatrixFirstColDiag(B);
    JXF_BigInt* col_map_offd   = jxf_ParBSRMatrixColMapOffd(B);
    JXF_Int     block_size     = jxf_ParBSRMatrixBlockSize(B);

    jxf_ParCSRCommPkg* comm_pkg        = jxf_ParBSRMatrixCommPkg(A);
    JXF_Int            num_recvs       = jxf_ParCSRCommPkgNumRecvs(comm_pkg);
    JXF_Int*           recv_vec_starts = jxf_ParCSRCommPkgRecvVecStarts(comm_pkg);
    JXF_Int            num_sends       = jxf_ParCSRCommPkgNumSends(comm_pkg);
    JXF_Int*           send_map_starts = jxf_ParCSRCommPkgSendMapStarts(comm_pkg);
    JXF_Int*           send_map_elmts  = jxf_ParCSRCommPkgSendMapElmts(comm_pkg);

    jxf_ParCSRCommHandle* comm_handle;
    jxf_ParCSRCommPkg*    tmp_comm_pkg = NULL;

    jxf_BSRMatrix* diag = jxf_ParBSRMatrixDiag(B);

    JXF_Int*     diag_i    = jxf_BSRMatrixI(diag);
    JXF_Int*     diag_j    = jxf_BSRMatrixJ(diag);
    JXF_Complex* diag_data = jxf_BSRMatrixData(diag);

    jxf_BSRMatrix* offd = jxf_ParBSRMatrixOffd(B);

    JXF_Int*     offd_i    = jxf_BSRMatrixI(offd);
    JXF_Int*     offd_j    = jxf_BSRMatrixJ(offd);
    JXF_Complex* offd_data = jxf_BSRMatrixData(offd);

    JXF_Int*     B_int_i;
    JXF_BigInt*  B_int_j;
    JXF_Complex* B_int_data;

    JXF_Int num_cols_B, num_nonzeros;
    JXF_Int num_rows_B_ext;
    JXF_Int num_procs, my_id;

    jxf_BSRMatrix* B_ext;

    JXF_Int*     B_ext_i;
    JXF_BigInt*  B_ext_j;
    JXF_Complex* B_ext_data;

    JXF_Int* jdata_recv_vec_starts;
    JXF_Int* jdata_send_map_starts;

    JXF_Int i, j, k, l, counter, bnnz;
    JXF_Int start_index;
    JXF_Int j_cnt, jrow;

    jxf_MPI_Comm_size(comm, &num_procs);
    jxf_MPI_Comm_rank(comm, &my_id);

    bnnz           = block_size * block_size;
    num_cols_B     = jxf_ParCSRMatrixGlobalNumCols(B);
    num_rows_B_ext = recv_vec_starts[num_recvs];
    B_int_i        = jxf_CTAlloc(JXF_Int, send_map_starts[num_sends] + 1);
    B_ext_i        = jxf_CTAlloc(JXF_Int, num_rows_B_ext + 1);
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
    comm_handle = jxf_ParCSRCommHandleCreate(11, comm_pkg, &B_int_i[1], &B_ext_i[1]);

    B_int_j = jxf_CTAlloc(JXF_BigInt, num_nonzeros);
    if (data) {
        B_int_data = jxf_CTAlloc(JXF_Complex, num_nonzeros * bnnz);
    }

    jdata_send_map_starts    = jxf_CTAlloc(JXF_Int, num_sends + 1);
    jdata_recv_vec_starts    = jxf_CTAlloc(JXF_Int, num_recvs + 1);
    start_index              = B_int_i[0];
    jdata_send_map_starts[0] = start_index;
    counter                  = 0;
    for (i = 0; i < num_sends; i++) {
        num_nonzeros = counter;
        for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++) {
            jrow = send_map_elmts[j];
            for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++) {
                B_int_j[counter] = (JXF_BigInt)diag_j[k] + first_col_diag;
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
    jxf_ParCSRCommPkgCreateAndFill(comm, num_recvs, jxf_ParCSRCommPkgRecvProcs(comm_pkg), jdata_recv_vec_starts, num_sends,
                                      jxf_ParCSRCommPkgSendProcs(comm_pkg), jdata_send_map_starts, NULL, &tmp_comm_pkg);

    jxf_ParCSRCommHandleDestroy(comm_handle);
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

    B_ext   = jxf_BSRMatrixCreate(block_size, num_rows_B_ext, num_cols_B, num_nonzeros);
    B_ext_j = jxf_CTAlloc(JXF_BigInt, num_nonzeros);
    if (data) {
        B_ext_data = jxf_CTAlloc(JXF_Complex, num_nonzeros * bnnz);
    }

    for (i = 0; i < num_recvs; i++) {
        start_index                  = B_ext_i[recv_vec_starts[i]];
        num_nonzeros                 = B_ext_i[recv_vec_starts[i + 1]] - start_index;
        jdata_recv_vec_starts[i + 1] = B_ext_i[recv_vec_starts[i + 1]];
    }

    comm_handle = jxf_ParCSRCommHandleCreate(21, tmp_comm_pkg, B_int_j, B_ext_j);
    jxf_ParCSRCommHandleDestroy(comm_handle);
    comm_handle = NULL;

    if (data) {
        comm_handle = jxf_ParBSRCommHandleCreate(1, bnnz, tmp_comm_pkg, B_int_data, B_ext_data);
        jxf_ParBSRCommHandleDestroy(comm_handle);
        comm_handle = NULL;
    }

    jxf_BSRMatrixI(B_ext)    = B_ext_i;
    jxf_BSRMatrixBigJ(B_ext) = B_ext_j;
    if (data) {
        jxf_BSRMatrixData(B_ext) = B_ext_data;
    }

    /* Free memory */
    jxf_TFree(jdata_send_map_starts);
    jxf_TFree(jdata_recv_vec_starts);
    jxf_TFree(tmp_comm_pkg);
    jxf_TFree(B_int_i);
    jxf_TFree(B_int_j);
    if (data) {
        jxf_TFree(B_int_data);
    }

    return B_ext;
}

/*--------------------------------------------------------------------------
 * jxf_ParBSRMatrixReorder:
 *
 * Reorders the column and data arrays of a the diagonal component of a square
 * ParBSR matrix, such that the first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

JXF_Int jxf_ParBSRMatrixReorder(jxf_ParBSRMatrix* A)
{
    JXF_BigInt     nrows_A = jxf_ParBSRMatrixGlobalNumRows(A);
    JXF_BigInt     ncols_A = jxf_ParBSRMatrixGlobalNumCols(A);
    jxf_BSRMatrix* A_diag  = jxf_ParBSRMatrixDiag(A);

    if (nrows_A != ncols_A) {
        jxf_error_w_msg(JXF_ERROR_GENERIC, " Error! Matrix should be square!\n");
        return jxf_error_flag;
    }

    jxf_BSRMatrixReorder(A_diag);

    return jxf_error_flag;
}
