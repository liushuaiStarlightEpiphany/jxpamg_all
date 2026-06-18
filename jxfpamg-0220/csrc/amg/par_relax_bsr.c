//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_bsr_relax.c --  Some relaxation functions for the par_bsr matrix
 *  Date: 2025/10/08
 */ 


#include "jxf_pamg.h"

/*---------------------------------------------------------------------------
 * jxf_ParBSRJacobiRelax: Weighted Jacobi relaxation for ParBSR
 *--------------------------------------------------------------------------*/
void jxf_ParBSRJacobiRelax(jxf_ParBSRMatrix* A, jxf_ParVector* x, jxf_ParVector* b, jxf_ParVector* Vtemp, JXF_Real relax_weight,
                              JXF_Real omega, JXF_Int forward_or_backward)
{

    MPI_Comm comm = jxf_ParBSRMatrixComm(A);

    jxf_BSRMatrix* A_diag      = jxf_ParBSRMatrixDiag(A);
    JXF_Real*      A_diag_data = jxf_BSRMatrixData(A_diag);
    JXF_Int*       A_diag_i    = jxf_BSRMatrixI(A_diag);
    JXF_Int*       A_diag_j    = jxf_BSRMatrixJ(A_diag);

    jxf_BSRMatrix* A_offd      = jxf_ParBSRMatrixOffd(A);
    JXF_Int*       A_offd_i    = jxf_BSRMatrixI(A_offd);
    JXF_Real*      A_offd_data = jxf_BSRMatrixData(A_offd);
    JXF_Int*       A_offd_j    = jxf_BSRMatrixJ(A_offd);

    jxf_ParCSRCommPkg*    comm_pkg = jxf_ParBSRMatrixCommPkg(A);
    jxf_ParCSRCommHandle* comm_handle;

    JXF_Int block_size = jxf_BSRMatrixBlockSize(A_diag);
    JXF_Int bnnz       = block_size * block_size;

    JXF_BigInt n_global;
    JXF_Int    n             = jxf_BSRMatrixNumRows(A_diag);
    JXF_Int    num_cols_offd = jxf_BSRMatrixNumCols(A_offd);
    JXF_BigInt first_index   = jxf_ParVectorFirstIndex(x);

    jxf_Vector* x_local = jxf_ParVectorLocalVector(x);
    JXF_Real*   x_data  = jxf_VectorData(x_local);

    jxf_Vector* b_local = jxf_ParVectorLocalVector(b);
    JXF_Real*   b_data  = jxf_VectorData(b_local);

    jxf_Vector* Vtemp_local = jxf_ParVectorLocalVector(Vtemp);
    JXF_Real*   Vtemp_data  = jxf_VectorData(Vtemp_local);
    JXF_Real*   Vext_data;
    JXF_Real*   v_buf_data;

    JXF_Real* tmp_data;

    JXF_Int size, rest, ne, ns;

    JXF_Int i, j, k;
    JXF_Int ii, jj;

    JXF_Int relax_error = 0;
    JXF_Int num_sends;
    JXF_Int index, start;
    JXF_Int num_procs, num_threads, my_id;

    JXF_Real *res_vec, *out_vec, *tmp_vec;
    JXF_Real *res0_vec, *res2_vec;
    JXF_Real  one_minus_weight;
    JXF_Real  one_minus_omega;
    JXF_Real  prod;

    jxf_CSRMatrix* A_CSR;
    JXF_Int*       A_CSR_i;
    JXF_Int*       A_CSR_j;
    JXF_Real*      A_CSR_data;

    jxf_Vector* f_vector;
    JXF_Real*   f_vector_data;

    jxf_ParCSRMatrix* A_ParCSR;

    JXF_Real* A_mat;
    JXF_Real* b_vec;

    JXF_Int column;

    /* initialize some stuff */
    one_minus_weight = 1.0 - relax_weight;
    one_minus_omega  = 1.0 - omega;
    jxf_MPI_Comm_size(comm, &num_procs);
    jxf_MPI_Comm_rank(comm, &my_id);
    /* num_threads = jxf_NumThreads(); */
    num_threads = 1;

    res_vec = jxf_CTAlloc(JXF_Real, block_size);
    out_vec = jxf_CTAlloc(JXF_Real, block_size);
    tmp_vec = jxf_CTAlloc(JXF_Real, block_size);

    if (!comm_pkg) {
        jxf_BlockMatvecCommPkgCreate(A);
        comm_pkg = jxf_ParBSRMatrixCommPkg(A);
    }

    /*---------------------------------------------------------------------------
      Jacobi
      ---------------------------------------------------------------------------*/

    if (num_procs > 1) {
        num_sends  = jxf_ParCSRCommPkgNumSends(comm_pkg);
        v_buf_data = jxf_CTAlloc(JXF_Real, jxf_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * block_size);
        Vext_data  = jxf_CTAlloc(JXF_Real, num_cols_offd * block_size);
        if (num_cols_offd) {
            A_offd_j    = jxf_BSRMatrixJ(A_offd);
            A_offd_data = jxf_BSRMatrixData(A_offd);
        }
        index = 0;
        for (i = 0; i < num_sends; i++) {
            start = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < jxf_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++) {
                for (k = 0; k < block_size; k++) {
                    v_buf_data[index++] = x_data[jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j) * block_size + k];
                }
            }
        }

        /* we need to use the block comm handle here - since comm_pkg is nodal based */
        comm_handle = jxf_ParBSRCommHandleCreate(1, block_size, comm_pkg, v_buf_data, Vext_data);
    }

    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/

    for (i = 0; i < n * block_size; i++) {
        Vtemp_data[i] = x_data[i];
    }
    if (num_procs > 1) {
        jxf_ParBSRCommHandleDestroy(comm_handle); /* now Vext_data is populated */
        comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    for (i = 0; i < n; i++) {

        /*-----------------------------------------------------------
         * If diagonal is nonzero, relax point i; otherwise, skip it.
         *-----------------------------------------------------------*/

        for (k = 0; k < block_size; k++) {
            res_vec[k] = b_data[i * block_size + k];
        }
        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
            ii = A_diag_j[jj];
            /* res -= A_diag_data[jj] * Vtemp_data[ii]; */
            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res_vec, block_size);
        }
        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
            ii = A_offd_j[jj];
            /* res -= A_offd_data[jj] * Vext_data[ii]; */
            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
        }

        /* if diag is singular, then skip this point */
        if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
            for (k = 0; k < block_size; k++) {
                x_data[i * block_size + k] *= one_minus_weight;
                x_data[i * block_size + k] += relax_weight * out_vec[k];
            }
        }
    }

    /* Free temporary arrays */
    if (num_procs > 1) {
        jxf_TFree(Vext_data);
        jxf_TFree(v_buf_data);
    }

    jxf_TFree(res_vec);
    jxf_TFree(out_vec);
    jxf_TFree(tmp_vec);
}

/*---------------------------------------------------------------------------
 * jxf_ParBSRHGSRelax: HGS or HSOR relaxation with forward_or_backward for ParBSR
 * hybrid: SOR-J mix off-processor, SOR on-processor with outer relaxation parameters
 * Let relax_weight = 1, i.e., Hybrid: G-S on proc. and Jacobi off proc.
 *---------------------------------------------------------------------------*/
void jxf_ParBSRHGSRelax(jxf_ParBSRMatrix* A, jxf_ParVector* x, jxf_ParVector* b, jxf_ParVector* Vtemp, JXF_Real relax_weight,
                           JXF_Real omega, JXF_Int forward_or_backward)
{

//     printf(" [%s:%d] Input parameters check:\n", __FUNCTION__, __LINE__);
// printf("  A = %p\n", (void*)A);
// printf("  x = %p\n", (void*)x);
// printf("  b = %p\n", (void*)b);
// printf("  Vtemp = %p\n", (void*)Vtemp);
// fflush(stdout);

// if (!A || !x || !b || !Vtemp) {
//     printf("ERROR: One or more input parameters are NULL!\n");
//     fflush(stdout);
//     return;
// }

// printf(" [%s:%d] jxf_ParBSRMatrix = %p\n", __FUNCTION__, __LINE__, (void*)A);
// fflush(stdout);

    MPI_Comm comm = jxf_ParBSRMatrixComm(A);


    jxf_BSRMatrix* A_diag      = jxf_ParBSRMatrixDiag(A);
    JXF_Real*      A_diag_data = jxf_BSRMatrixData(A_diag);
    JXF_Int*       A_diag_i    = jxf_BSRMatrixI(A_diag);
    JXF_Int*       A_diag_j    = jxf_BSRMatrixJ(A_diag);
    jxf_BSRMatrix* A_offd      = jxf_ParBSRMatrixOffd(A);
    JXF_Int*       A_offd_i    = jxf_BSRMatrixI(A_offd);
    JXF_Real*      A_offd_data = jxf_BSRMatrixData(A_offd);
    JXF_Int*       A_offd_j    = jxf_BSRMatrixJ(A_offd);

    jxf_ParCSRCommPkg*    comm_pkg = jxf_ParBSRMatrixCommPkg(A);
    jxf_ParCSRCommHandle* comm_handle;

    JXF_Int block_size = jxf_BSRMatrixBlockSize(A_diag);
    JXF_Int bnnz       = block_size * block_size;

    JXF_BigInt n_global;
    JXF_Int    n             = jxf_BSRMatrixNumRows(A_diag);
    JXF_Int    num_cols_offd = jxf_BSRMatrixNumCols(A_offd);
    JXF_BigInt first_index   = jxf_ParVectorFirstIndex(x);

    jxf_Vector* x_local = jxf_ParVectorLocalVector(x);
    JXF_Real*   x_data  = jxf_VectorData(x_local);

    jxf_Vector* b_local = jxf_ParVectorLocalVector(b);
    JXF_Real*   b_data  = jxf_VectorData(b_local);

    jxf_Vector* Vtemp_local = jxf_ParVectorLocalVector(Vtemp);
    JXF_Real*   Vtemp_data  = jxf_VectorData(Vtemp_local);
    JXF_Real*   Vext_data;
    JXF_Real*   v_buf_data;

    JXF_Real* tmp_data;

    JXF_Int size, rest, ne, ns;

    JXF_Int i, j, k;
    JXF_Int ii, jj;

    JXF_Int relax_error = 0;
    JXF_Int num_sends;
    JXF_Int index, start;
    JXF_Int num_procs, num_threads, my_id;

    JXF_Real *res_vec, *out_vec, *tmp_vec;
    JXF_Real *res0_vec, *res2_vec;
    JXF_Real  one_minus_weight;
    JXF_Real  one_minus_omega;
    JXF_Real  prod;

    jxf_CSRMatrix* A_CSR;
    JXF_Int*       A_CSR_i;
    JXF_Int*       A_CSR_j;
    JXF_Real*      A_CSR_data;

    jxf_Vector* f_vector;
    JXF_Real*   f_vector_data;

    jxf_ParCSRMatrix* A_ParCSR;

    JXF_Real* A_mat;
    JXF_Real* b_vec;

    JXF_Int column;
    /* initialize some stuff */
    one_minus_weight = 1.0 - relax_weight;
    one_minus_omega  = 1.0 - omega;

    jxf_MPI_Comm_size(comm, &num_procs);
    jxf_MPI_Comm_rank(comm, &my_id);
    /* num_threads = jxf_NumThreads(); */
    num_threads = 1; //! here fix it equal to 1

    res_vec = jxf_CTAlloc(JXF_Real, block_size);
    out_vec = jxf_CTAlloc(JXF_Real, block_size);
    tmp_vec = jxf_CTAlloc(JXF_Real, block_size);

    if (!comm_pkg) {
        jxf_BlockMatvecCommPkgCreate(A);
        comm_pkg = jxf_ParBSRMatrixCommPkg(A);
    }

    /*---------------------------------------------------------------------------
      Hybrid: G-S on proc. and Jacobi off proc.
      ---------------------------------------------------------------------------*/
    if (num_procs > 1) {
        num_sends  = jxf_ParCSRCommPkgNumSends(comm_pkg);
        v_buf_data = jxf_CTAlloc(JXF_Real, jxf_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * block_size);
        Vext_data  = jxf_CTAlloc(JXF_Real, num_cols_offd * block_size);
        if (num_cols_offd) {
            A_offd_j    = jxf_BSRMatrixJ(A_offd);
            A_offd_data = jxf_BSRMatrixData(A_offd);
        }
        index = 0;
        for (i = 0; i < num_sends; i++) {
            start = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < jxf_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++) {
                for (k = 0; k < block_size; k++) {
                    v_buf_data[index++] = x_data[jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j) * block_size + k];
                }
            }
        }

        /* we need to use the block comm handle here - since comm_pkg is nodal based */
        comm_handle = jxf_ParBSRCommHandleCreate(1, block_size, comm_pkg, v_buf_data, Vext_data);
    }


    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/

    for (i = 0; i < n * block_size; i++) {
        Vtemp_data[i] = x_data[i];
    }

    if (num_procs > 1) {
        jxf_ParBSRCommHandleDestroy(comm_handle); /* now Vext_data is populated */
        comm_handle = NULL;
    }

    // forward or backward GS sweep
    JXF_Int begin_idx = 0, end_idx = n, step = 1;
    if (forward_or_backward == -1) {
        begin_idx = n - 1;
        end_idx   = -1;
        step      = -1;
    }

    /*-----------------------------------------------------------------
     * relax weight and omega = 1
     *-----------------------------------------------------------------*/

    if (jxf_abs(relax_weight - 1) < JXF_REAL_EPSILON && jxf_abs(omega - 1) < JXF_REAL_EPSILON) {

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/
        if (num_threads > 1) {
            tmp_data = jxf_CTAlloc(JXF_Real, n);

            for (i = 0; i < n; i++) {
                tmp_data[i] = x_data[i];
            }

            for (j = 0; j < num_threads; j++) {
                size = n / num_threads;
                rest = n - size * num_threads;
                if (j < rest) {
                    ns = j * size + j;
                    ne = (j + 1) * size + j + 1;
                } else {
                    ns = j * size + rest;
                    ne = (j + 1) * size + rest;
                }

                // forward or backward GS sweep
                begin_idx = ns, end_idx = ne, step = 1;
                if (forward_or_backward == -1) {
                    begin_idx = ne - 1;
                    end_idx   = ns - 1;
                    step      = -1;
                }
                // for (i = ns; i < ne; i++) /* interior points first (forward sweep) */
                for (int i = begin_idx; i != end_idx; i += step) {
                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/

                    for (k = 0; k < block_size; k++) {
                        res_vec[k] = b_data[i * block_size + k];
                    }
                    for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                        ii = A_diag_j[jj];
                        if (ii >= ns && ii < ne) {
                            /*  res -= A_diag_data[jj] * x_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                    /* if diag is singular, then skip this point */
                    if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] = out_vec[k];
                        }
                    }
                } /* for loop over points */
            } /* foor loop over threads */
            jxf_TFree(tmp_data);
        } else /* num_threads = 1 */
        {

            // for (i = 0; i < n; i++) /* interior points first (forward sweep) */
            for (int i = begin_idx; i != end_idx; i += step) {

                /*-----------------------------------------------------------
                 * If diagonal is nonzero, relax point i; otherwise, skip it.
                 *-----------------------------------------------------------*/
                for (k = 0; k < block_size; k++) {
                    res_vec[k] = b_data[i * block_size + k];
                }
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                    ii = A_diag_j[jj];
                    /* res -= A_diag_data[jj] * x_data[ii]; */
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii]; */
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
                    for (k = 0; k < block_size; k++) {
                        x_data[i * block_size + k] = out_vec[k];
                    }
                }
            } /* for loop over points */
             

        } /* end of num_threads = 1 */
    } else {
        /*-----------------------------------------------------------------
         * relax weight and omega do not = 1
         *-----------------------------------------------------------------*/
        prod     = (1.0 - relax_weight * omega);
        res0_vec = jxf_CTAlloc(JXF_Real, block_size);
        res2_vec = jxf_CTAlloc(JXF_Real, block_size);

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/

        if (num_threads > 1) {
            tmp_data = jxf_CTAlloc(JXF_Real, n);
            for (i = 0; i < n; i++) {
                tmp_data[i] = x_data[i];
            }
            for (j = 0; j < num_threads; j++) {
                size = n / num_threads;
                rest = n - size * num_threads;
                if (j < rest) {
                    ns = j * size + j;
                    ne = (j + 1) * size + j + 1;
                } else {
                    ns = j * size + rest;
                    ne = (j + 1) * size + rest;
                }
                // forward or backward GS sweep
                begin_idx = ns, end_idx = ne, step = 1;
                if (forward_or_backward == -1) {
                    begin_idx = ne - 1;
                    end_idx   = ns - 1;
                    step      = -1;
                }
                // for (i = ns; i < ne; i++) /* interior points first (forward sweep) */
                for (int i = begin_idx; i != end_idx; i += step) {
                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
                    for (k = 0; k < block_size; k++) {
                        res_vec[k]  = b_data[i * block_size + k];
                        res0_vec[k] = 0.0;
                        res2_vec[k] = 0.0;
                    }
                    for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                        ii = A_diag_j[jj];
                        if (ii >= ns && ii < ne) {
                            /* res0 -= A_diag_data[jj] * x_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                            /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                            jxf_BSRMatrixBlockMatvec_Stable(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] *= prod;
                       x_data[i] += relax_weight*(omega*res + res0 +
                       one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                    for (k = 0; k < block_size; k++) {
                        tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                    }
                    if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] *= prod;
                            x_data[i * block_size + k] += relax_weight * out_vec[k];
                        }
                    }

                } /* end of loop over points */
            } /* end of loop over threads */
            jxf_TFree(tmp_data);
        } else /* num_threads = 1 */
        {
            // for (i = 0; i < n; i++) /* interior points first (forward sweep) */
            for (int i = begin_idx; i != end_idx; i += step) {
                /*-----------------------------------------------------------
                 * If diagonal is nonzero, relax point i; otherwise, skip it.
                 *-----------------------------------------------------------*/
                for (k = 0; k < block_size; k++) {
                    res_vec[k]  = b_data[i * block_size + k];
                    res0_vec[k] = 0.0;
                    res2_vec[k] = 0.0;
                }
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                    ii = A_diag_j[jj];
                    /* res0 -= A_diag_data[jj] * x_data[ii]; */
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                    /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                    jxf_BSRMatrixBlockMatvec_Stable(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii];*/
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] *= prod;
                   x_data[i] += relax_weight*(omega*res + res0 +
                   one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                for (k = 0; k < block_size; k++) {
                    tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                }
                if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                    for (k = 0; k < block_size; k++) {
                        x_data[i * block_size + k] *= prod;
                        x_data[i * block_size + k] += relax_weight * out_vec[k];
                    }
                }
            } /* end of loop over points */
        } /* end num_threads = 1 */

        jxf_TFree(res0_vec);
        jxf_TFree(res2_vec);
    } /* end of check relax weight and omega */

    /* Free temporary arrays */
    if (num_procs > 1) {
        jxf_TFree(Vext_data);
        jxf_TFree(v_buf_data);
    }

    jxf_TFree(res_vec);
    jxf_TFree(out_vec);
    jxf_TFree(tmp_vec);
}

/*-----------------------------------------------------------------
  HSGS:  Hybrid: Jacobi off-processor, Symm. Gauss-Seidel/ SSOR on-processor with outer relaxation parameter
 *-----------------------------------------------------------------*/
// forward_or_backward don't be used here: 1: forward, -1: backward
void jxf_ParBSRHSGSRelax(jxf_ParBSRMatrix* A, jxf_ParVector* x, jxf_ParVector* b, jxf_ParVector* Vtemp, JXF_Real relax_weight,
                            JXF_Real omega, JXF_Int forward_or_backward)
{

    MPI_Comm comm = jxf_ParBSRMatrixComm(A);

    jxf_BSRMatrix* A_diag      = jxf_ParBSRMatrixDiag(A);
    JXF_Real*      A_diag_data = jxf_BSRMatrixData(A_diag);
    JXF_Int*       A_diag_i    = jxf_BSRMatrixI(A_diag);
    JXF_Int*       A_diag_j    = jxf_BSRMatrixJ(A_diag);

    jxf_BSRMatrix* A_offd      = jxf_ParBSRMatrixOffd(A);
    JXF_Int*       A_offd_i    = jxf_BSRMatrixI(A_offd);
    JXF_Real*      A_offd_data = jxf_BSRMatrixData(A_offd);
    JXF_Int*       A_offd_j    = jxf_BSRMatrixJ(A_offd);

    jxf_ParCSRCommPkg*    comm_pkg = jxf_ParBSRMatrixCommPkg(A);
    jxf_ParCSRCommHandle* comm_handle;

    JXF_Int block_size = jxf_BSRMatrixBlockSize(A_diag);
    JXF_Int bnnz       = block_size * block_size;

    JXF_BigInt n_global;
    JXF_Int    n             = jxf_BSRMatrixNumRows(A_diag);
    JXF_Int    num_cols_offd = jxf_BSRMatrixNumCols(A_offd);
    JXF_BigInt first_index   = jxf_ParVectorFirstIndex(x);

    jxf_Vector* x_local = jxf_ParVectorLocalVector(x);
    JXF_Real*   x_data  = jxf_VectorData(x_local);

    jxf_Vector* b_local = jxf_ParVectorLocalVector(b);
    JXF_Real*   b_data  = jxf_VectorData(b_local);

    jxf_Vector* Vtemp_local = jxf_ParVectorLocalVector(Vtemp);
    JXF_Real*   Vtemp_data  = jxf_VectorData(Vtemp_local);
    JXF_Real*   Vext_data;
    JXF_Real*   v_buf_data;

    JXF_Real* tmp_data;

    JXF_Int size, rest, ne, ns;

    JXF_Int i, j, k;
    JXF_Int ii, jj;

    JXF_Int relax_error = 0;
    JXF_Int num_sends;
    JXF_Int index, start;
    JXF_Int num_procs, num_threads, my_id;

    JXF_Real *res_vec, *out_vec, *tmp_vec;
    JXF_Real *res0_vec, *res2_vec;
    JXF_Real  one_minus_weight;
    JXF_Real  one_minus_omega;
    JXF_Real  prod;

    jxf_CSRMatrix* A_CSR;
    JXF_Int*       A_CSR_i;
    JXF_Int*       A_CSR_j;
    JXF_Real*      A_CSR_data;

    jxf_Vector* f_vector;
    JXF_Real*   f_vector_data;

    jxf_ParCSRMatrix* A_ParCSR;

    JXF_Real* A_mat;
    JXF_Real* b_vec;

    JXF_Int column;

    /* initialize some stuff */
    one_minus_weight = 1.0 - relax_weight;
    one_minus_omega  = 1.0 - omega;
    jxf_MPI_Comm_size(comm, &num_procs);
    jxf_MPI_Comm_rank(comm, &my_id);
    /* num_threads = jxf_NumThreads(); */
    num_threads = 1;

    res_vec = jxf_CTAlloc(JXF_Real, block_size);
    out_vec = jxf_CTAlloc(JXF_Real, block_size);
    tmp_vec = jxf_CTAlloc(JXF_Real, block_size);

    if (!comm_pkg) {
        jxf_BlockMatvecCommPkgCreate(A);
        comm_pkg = jxf_ParBSRMatrixCommPkg(A);
    }

    /*-----------------------------------------------------------------
                  Hybrid: Jacobi off-processor,
                  Symm. Gauss-Seidel/ SSOR on-processor
                  with outer relaxation parameter
     *-----------------------------------------------------------------*/

    if (num_procs > 1) {
        num_sends = jxf_ParCSRCommPkgNumSends(comm_pkg);

        v_buf_data = jxf_CTAlloc(JXF_Real, jxf_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * block_size);

        Vext_data = jxf_CTAlloc(JXF_Real, num_cols_offd * block_size);

        if (num_cols_offd) {
            A_offd_j    = jxf_BSRMatrixJ(A_offd);
            A_offd_data = jxf_BSRMatrixData(A_offd);
        }

        index = 0;
        for (i = 0; i < num_sends; i++) {
            start = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < jxf_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++) {
                for (k = 0; k < block_size; k++) {
                    v_buf_data[index++] = x_data[jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j) * block_size + k];
                }
            }
        }

        /* we need to use the block comm handle here - since comm_pkg is nodal based */
        comm_handle = jxf_ParBSRCommHandleCreate(1, block_size, comm_pkg, v_buf_data, Vext_data);

        jxf_ParBSRCommHandleDestroy(comm_handle);
        comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    if (jxf_abs(relax_weight - 1) < JXF_REAL_EPSILON && jxf_abs(omega - 1) < JXF_REAL_EPSILON) {
        if (num_threads > 1) {
            tmp_data = jxf_CTAlloc(JXF_Real, n);
            for (i = 0; i < n; i++) {
                tmp_data[i] = x_data[i];
            }
            for (j = 0; j < num_threads; j++) {
                size = n / num_threads;
                rest = n - size * num_threads;
                if (j < rest) {
                    ns = j * size + j;
                    ne = (j + 1) * size + j + 1;
                } else {
                    ns = j * size + rest;
                    ne = (j + 1) * size + rest;
                }
                for (i = ns; i < ne; i++) /* interior points first */
                {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/

                    for (k = 0; k < block_size; k++) {
                        res_vec[k] = b_data[i * block_size + k];
                    }

                    for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                        ii = A_diag_j[jj];
                        if (ii >= ns && ii < ne) {
                            /* res -= A_diag_data[jj] * x_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];

                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                    /* if diag is singular, then skip this point */
                    if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] = out_vec[k];
                        }
                    }
                } /* end of interior points loop */

                for (i = ne - 1; i > ns - 1; i--) /* interior points first */
                {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/

                    for (k = 0; k < block_size; k++) {
                        res_vec[k] = b_data[i * block_size + k];
                    }

                    for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                        ii = A_diag_j[jj];
                        if (ii >= ns && ii < ne) {

                            /* res -= A_diag_data[jj] * x_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);

                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii]; */
                        jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                    /* if diag is singular, then skip this point */
                    if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] = out_vec[k];
                        }
                    }
                } /* end of loop over points */
            } /* end of loop over threads */
            jxf_TFree(tmp_data);
        } else /* num_thread ==1 */
        {
            for (i = 0; i < n; i++) /* interior points first */
            {

                /*-----------------------------------------------------------
                 * If diagonal is nonzero, relax point i; otherwise, skip it.
                 *-----------------------------------------------------------*/
                for (k = 0; k < block_size; k++) {
                    res_vec[k] = b_data[i * block_size + k];
                }
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                    ii = A_diag_j[jj];
                    /* res -= A_diag_data[jj] * x_data[ii]; */
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];

                    /* res -= A_offd_data[jj] * Vext_data[ii]; */
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
                    for (k = 0; k < block_size; k++) {
                        x_data[i * block_size + k] = out_vec[k];
                    }
                }
            } /* end of loop over points */

            for (i = n - 1; i > -1; i--) /* interior points first */
            {

                /*-----------------------------------------------------------
                 * If diagonal is nonzero, relax point i; otherwise, skip it.
                 *-----------------------------------------------------------*/
                for (k = 0; k < block_size; k++) {
                    res_vec[k] = b_data[i * block_size + k];
                }

                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                    ii = A_diag_j[jj];
                    /* res -= A_diag_data[jj] * x_data[ii]; */
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii]; */
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
                    for (k = 0; k < block_size; k++) {
                        x_data[i * block_size + k] = out_vec[k];
                    }
                }

            } /* end loop over points */
        } /* end of num_threads = 1 */

    } else {
        /*-----------------------------------------------------------------
         * relax weight and omega do not = 1
         *-----------------------------------------------------------------*/

        prod     = (1.0 - relax_weight * omega);
        res0_vec = jxf_CTAlloc(JXF_Real, block_size);
        res2_vec = jxf_CTAlloc(JXF_Real, block_size);
        for (i = 0; i < n; i++) {
            Vtemp_data[i] = x_data[i];
        }
        prod = (1.0 - relax_weight * omega);

        if (num_threads > 1) {
            tmp_data = jxf_CTAlloc(JXF_Real, n);
            for (i = 0; i < n; i++) {
                tmp_data[i] = x_data[i];
            }
            for (j = 0; j < num_threads; j++) {
                size = n / num_threads;
                rest = n - size * num_threads;
                if (j < rest) {
                    ns = j * size + j;
                    ne = (j + 1) * size + j + 1;
                } else {
                    ns = j * size + rest;
                    ne = (j + 1) * size + rest;
                }
                for (i = ns; i < ne; i++) /* interior points first */
                {
                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/

                    for (k = 0; k < block_size; k++) {
                        res_vec[k]  = b_data[i * block_size + k];
                        res0_vec[k] = 0.0;
                        res2_vec[k] = 0.0;
                    }

                    for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                        ii = A_diag_j[jj];

                        if (ii >= ns && ii < ne) {
                            /* res0 -= A_diag_data[jj] * x_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                            /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                            jxf_BSRMatrixBlockMatvec_Stable(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }

                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii]; */
                        jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] *= prod;
                       x_data[i] += relax_weight*(omega*res + res0 +
                       one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                    for (k = 0; k < block_size; k++) {
                        tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                    }
                    if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] *= prod;
                            x_data[i * block_size + k] += relax_weight * out_vec[k];
                        }
                    }
                }

                for (i = ne - 1; i > ns - 1; i--) /* interior points first */
                {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
                    for (k = 0; k < block_size; k++) {
                        res_vec[k]  = b_data[i * block_size + k];
                        res0_vec[k] = 0.0;
                        res2_vec[k] = 0.0;
                    }

                    for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                        ii = A_diag_j[jj];
                        if (ii >= ns && ii < ne) {
                            /* res0 -= A_diag_data[jj] * x_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                            /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                            jxf_BSRMatrixBlockMatvec_Stable(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] *= prod;
                       x_data[i] += relax_weight*(omega*res + res0 +
                       one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                    for (k = 0; k < block_size; k++) {
                        tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                    }
                    if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] *= prod;
                            x_data[i * block_size + k] += relax_weight * out_vec[k];
                        }
                    }
                } /* end of loop over points */
            } /* loop over threads end */
            jxf_TFree(tmp_data);

        } else /* num threads = 1 */
        {
            for (i = 0; i < n; i++) /* interior points first */
            {

                /*-----------------------------------------------------------
                 * If diagonal is nonzero, relax point i; otherwise, skip it.
                 *-----------------------------------------------------------*/
                for (k = 0; k < block_size; k++) {
                    res_vec[k]  = b_data[i * block_size + k];
                    res0_vec[k] = 0.0;
                    res2_vec[k] = 0.0;
                }
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                    ii = A_diag_j[jj];
                    /* res0 -= A_diag_data[jj] * x_data[ii]; */
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                    /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                    jxf_BSRMatrixBlockMatvec_Stable(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii];*/
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] *= prod;
                   x_data[i] += relax_weight*(omega*res + res0 +
                   one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                for (k = 0; k < block_size; k++) {
                    tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                }
                if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                    for (k = 0; k < block_size; k++) {
                        x_data[i * block_size + k] *= prod;
                        x_data[i * block_size + k] += relax_weight * out_vec[k];
                    }
                }
            }
            for (i = n - 1; i > -1; i--) /* interior points first */
            {

                /*-----------------------------------------------------------
                 * If diagonal is nonzero, relax point i; otherwise, skip it.
                 *-----------------------------------------------------------*/
                for (k = 0; k < block_size; k++) {
                    res_vec[k]  = b_data[i * block_size + k];
                    res0_vec[k] = 0.0;
                    res2_vec[k] = 0.0;
                }
                for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++) {
                    ii = A_diag_j[jj];
                    /* res0 -= A_diag_data[jj] * x_data[ii]; */
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                    /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                    jxf_BSRMatrixBlockMatvec_Stable(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii];*/
                    jxf_BSRMatrixBlockMatvec_Stable(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] *= prod;
                   x_data[i] += relax_weight*(omega*res + res0 +
                   one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                for (k = 0; k < block_size; k++) {
                    tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                }
                if (jxf_BSRMatrixBlockInvMatvec_Stable(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                    for (k = 0; k < block_size; k++) {
                        x_data[i * block_size + k] *= prod;
                        x_data[i * block_size + k] += relax_weight * out_vec[k];
                    }
                }
            } /* end of loop over points */
        } /* end num_threads = 1 */

        jxf_TFree(res0_vec);
        jxf_TFree(res2_vec);
    } /* end of check relax weight and omega */

    /* Free temporary arrays */
    if (num_procs > 1) {
        jxf_TFree(Vext_data);
        jxf_TFree(v_buf_data);
    }

    jxf_TFree(res_vec);
    jxf_TFree(out_vec);
    jxf_TFree(tmp_vec);
}