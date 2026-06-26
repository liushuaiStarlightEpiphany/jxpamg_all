//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_bsr_relax.c --  Some relaxation functions for the par_bsr matrix
 *  Date: 2025/10/08
 */ 


#include "jx_pamg.h"

/*---------------------------------------------------------------------------
 * jx_ParBSRJacobiRelax: Weighted Jacobi relaxation for ParBSR
 *--------------------------------------------------------------------------*/
void jx_ParBSRJacobiRelax(jx_ParBSRMatrix* A, jx_ParVector* x, jx_ParVector* b, jx_ParVector* Vtemp, JX_Real relax_weight,
                              JX_Real omega, JX_Int forward_or_backward)
{

    MPI_Comm comm = jx_ParBSRMatrixComm(A);

    jx_BSRMatrix* A_diag      = jx_ParBSRMatrixDiag(A);
    JX_Real*      A_diag_data = jx_BSRMatrixData(A_diag);
    JX_Int*       A_diag_i    = jx_BSRMatrixI(A_diag);
    JX_Int*       A_diag_j    = jx_BSRMatrixJ(A_diag);

    jx_BSRMatrix* A_offd      = jx_ParBSRMatrixOffd(A);
    JX_Int*       A_offd_i    = jx_BSRMatrixI(A_offd);
    JX_Real*      A_offd_data = jx_BSRMatrixData(A_offd);
    JX_Int*       A_offd_j    = jx_BSRMatrixJ(A_offd);

    jx_ParCSRCommPkg*    comm_pkg = jx_ParBSRMatrixCommPkg(A);
    jx_ParCSRCommHandle* comm_handle;

    JX_Int block_size = jx_BSRMatrixBlockSize(A_diag);
    JX_Int bnnz       = block_size * block_size;

    JX_BigInt n_global;
    JX_Int    n             = jx_BSRMatrixNumRows(A_diag);
    JX_Int    num_cols_offd = jx_BSRMatrixNumCols(A_offd);
    JX_BigInt first_index   = jx_ParVectorFirstIndex(x);

    jx_Vector* x_local = jx_ParVectorLocalVector(x);
    JX_Real*   x_data  = jx_VectorData(x_local);

    jx_Vector* b_local = jx_ParVectorLocalVector(b);
    JX_Real*   b_data  = jx_VectorData(b_local);

    jx_Vector* Vtemp_local = jx_ParVectorLocalVector(Vtemp);
    JX_Real*   Vtemp_data  = jx_VectorData(Vtemp_local);
    JX_Real*   Vext_data;
    JX_Real*   v_buf_data;

    JX_Real* tmp_data;

    JX_Int size, rest, ne, ns;

    JX_Int i, j, k;
    JX_Int ii, jj;

    JX_Int relax_error = 0;
    JX_Int num_sends;
    JX_Int index, start;
    JX_Int num_procs, num_threads, my_id;

    JX_Real *res_vec, *out_vec, *tmp_vec;
    JX_Real *res0_vec, *res2_vec;
    JX_Real  one_minus_weight;
    JX_Real  one_minus_omega;
    JX_Real  prod;

    jx_CSRMatrix* A_CSR;
    JX_Int*       A_CSR_i;
    JX_Int*       A_CSR_j;
    JX_Real*      A_CSR_data;

    jx_Vector* f_vector;
    JX_Real*   f_vector_data;

    jx_ParCSRMatrix* A_ParCSR;

    JX_Real* A_mat;
    JX_Real* b_vec;

    JX_Int column;

    /* initialize some stuff */
    one_minus_weight = 1.0 - relax_weight;
    one_minus_omega  = 1.0 - omega;
    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);
    /* num_threads = jx_NumThreads(); */
    num_threads = 1;

    res_vec = jx_CTAlloc(JX_Real, block_size);
    out_vec = jx_CTAlloc(JX_Real, block_size);
    tmp_vec = jx_CTAlloc(JX_Real, block_size);

    if (!comm_pkg) {
        jx_BlockMatvecCommPkgCreate(A);
        comm_pkg = jx_ParBSRMatrixCommPkg(A);
    }

    /*---------------------------------------------------------------------------
      Jacobi
      ---------------------------------------------------------------------------*/

    if (num_procs > 1) {
        num_sends  = jx_ParCSRCommPkgNumSends(comm_pkg);
        v_buf_data = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * block_size);
        Vext_data  = jx_CTAlloc(JX_Real, num_cols_offd * block_size);
        if (num_cols_offd) {
            A_offd_j    = jx_BSRMatrixJ(A_offd);
            A_offd_data = jx_BSRMatrixData(A_offd);
        }
        index = 0;
        for (i = 0; i < num_sends; i++) {
            start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++) {
                for (k = 0; k < block_size; k++) {
                    v_buf_data[index++] = x_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg, j) * block_size + k];
                }
            }
        }

        /* we need to use the block comm handle here - since comm_pkg is nodal based */
        comm_handle = jx_ParBSRCommHandleCreate(1, block_size, comm_pkg, v_buf_data, Vext_data);
    }

    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/

    for (i = 0; i < n * block_size; i++) {
        Vtemp_data[i] = x_data[i];
    }
    if (num_procs > 1) {
        jx_ParBSRCommHandleDestroy(comm_handle); /* now Vext_data is populated */
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
            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res_vec, block_size);
        }
        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
            ii = A_offd_j[jj];
            /* res -= A_offd_data[jj] * Vext_data[ii]; */
            jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
        }

        /* if diag is singular, then skip this point */
        if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
            for (k = 0; k < block_size; k++) {
                x_data[i * block_size + k] *= one_minus_weight;
                x_data[i * block_size + k] += relax_weight * out_vec[k];
            }
        }
    }

    /* Free temporary arrays */
    if (num_procs > 1) {
        jx_TFree(Vext_data);
        jx_TFree(v_buf_data);
    }

    jx_TFree(res_vec);
    jx_TFree(out_vec);
    jx_TFree(tmp_vec);
}

/*---------------------------------------------------------------------------
 * jx_ParBSRHGSRelax: HGS or HSOR relaxation with forward_or_backward for ParBSR
 * hybrid: SOR-J mix off-processor, SOR on-processor with outer relaxation parameters
 * Let relax_weight = 1, i.e., Hybrid: G-S on proc. and Jacobi off proc.
 *---------------------------------------------------------------------------*/
void jx_ParBSRHGSRelax(jx_ParBSRMatrix* A, jx_ParVector* x, jx_ParVector* b, jx_ParVector* Vtemp, JX_Real relax_weight,
                           JX_Real omega, JX_Int forward_or_backward)
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

// printf(" [%s:%d] jx_ParBSRMatrix = %p\n", __FUNCTION__, __LINE__, (void*)A);
// fflush(stdout);

    MPI_Comm comm = jx_ParBSRMatrixComm(A);


    jx_BSRMatrix* A_diag      = jx_ParBSRMatrixDiag(A);
    JX_Real*      A_diag_data = jx_BSRMatrixData(A_diag);
    JX_Int*       A_diag_i    = jx_BSRMatrixI(A_diag);
    JX_Int*       A_diag_j    = jx_BSRMatrixJ(A_diag);
    jx_BSRMatrix* A_offd      = jx_ParBSRMatrixOffd(A);
    JX_Int*       A_offd_i    = jx_BSRMatrixI(A_offd);
    JX_Real*      A_offd_data = jx_BSRMatrixData(A_offd);
    JX_Int*       A_offd_j    = jx_BSRMatrixJ(A_offd);

    jx_ParCSRCommPkg*    comm_pkg = jx_ParBSRMatrixCommPkg(A);
    jx_ParCSRCommHandle* comm_handle;

    JX_Int block_size = jx_BSRMatrixBlockSize(A_diag);
    JX_Int bnnz       = block_size * block_size;

    JX_BigInt n_global;
    JX_Int    n             = jx_BSRMatrixNumRows(A_diag);
    JX_Int    num_cols_offd = jx_BSRMatrixNumCols(A_offd);
    JX_BigInt first_index   = jx_ParVectorFirstIndex(x);

    jx_Vector* x_local = jx_ParVectorLocalVector(x);
    JX_Real*   x_data  = jx_VectorData(x_local);

    jx_Vector* b_local = jx_ParVectorLocalVector(b);
    JX_Real*   b_data  = jx_VectorData(b_local);

    jx_Vector* Vtemp_local = jx_ParVectorLocalVector(Vtemp);
    JX_Real*   Vtemp_data  = jx_VectorData(Vtemp_local);
    JX_Real*   Vext_data;
    JX_Real*   v_buf_data;

    JX_Real* tmp_data;

    JX_Int size, rest, ne, ns;

    JX_Int i, j, k;
    JX_Int ii, jj;

    JX_Int relax_error = 0;
    JX_Int num_sends;
    JX_Int index, start;
    JX_Int num_procs, num_threads, my_id;

    JX_Real *res_vec, *out_vec, *tmp_vec;
    JX_Real *res0_vec, *res2_vec;
    JX_Real  one_minus_weight;
    JX_Real  one_minus_omega;
    JX_Real  prod;

    jx_CSRMatrix* A_CSR;
    JX_Int*       A_CSR_i;
    JX_Int*       A_CSR_j;
    JX_Real*      A_CSR_data;

    jx_Vector* f_vector;
    JX_Real*   f_vector_data;

    jx_ParCSRMatrix* A_ParCSR;

    JX_Real* A_mat;
    JX_Real* b_vec;

    JX_Int column;
    /* initialize some stuff */
    one_minus_weight = 1.0 - relax_weight;
    one_minus_omega  = 1.0 - omega;

    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);
    /* num_threads = jx_NumThreads(); */
    num_threads = 1; //! here fix it equal to 1

    res_vec = jx_CTAlloc(JX_Real, block_size);
    out_vec = jx_CTAlloc(JX_Real, block_size);
    tmp_vec = jx_CTAlloc(JX_Real, block_size);

    if (!comm_pkg) {
        jx_BlockMatvecCommPkgCreate(A);
        comm_pkg = jx_ParBSRMatrixCommPkg(A);
    }

    /*---------------------------------------------------------------------------
      Hybrid: G-S on proc. and Jacobi off proc.
      ---------------------------------------------------------------------------*/
    if (num_procs > 1) {
        num_sends  = jx_ParCSRCommPkgNumSends(comm_pkg);
        v_buf_data = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * block_size);
        Vext_data  = jx_CTAlloc(JX_Real, num_cols_offd * block_size);
        if (num_cols_offd) {
            A_offd_j    = jx_BSRMatrixJ(A_offd);
            A_offd_data = jx_BSRMatrixData(A_offd);
        }
        index = 0;
        for (i = 0; i < num_sends; i++) {
            start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++) {
                for (k = 0; k < block_size; k++) {
                    v_buf_data[index++] = x_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg, j) * block_size + k];
                }
            }
        }

        /* we need to use the block comm handle here - since comm_pkg is nodal based */
        comm_handle = jx_ParBSRCommHandleCreate(1, block_size, comm_pkg, v_buf_data, Vext_data);
    }


    /*-----------------------------------------------------------------
     * Copy current approximation into temporary vector.
     *-----------------------------------------------------------------*/

    for (i = 0; i < n * block_size; i++) {
        Vtemp_data[i] = x_data[i];
    }

    if (num_procs > 1) {
        jx_ParBSRCommHandleDestroy(comm_handle); /* now Vext_data is populated */
        comm_handle = NULL;
    }

    // forward or backward GS sweep
    JX_Int begin_idx = 0, end_idx = n, step = 1;
    if (forward_or_backward == -1) {
        begin_idx = n - 1;
        end_idx   = -1;
        step      = -1;
    }

    /*-----------------------------------------------------------------
     * relax weight and omega = 1
     *-----------------------------------------------------------------*/

    if (jx_abs(relax_weight - 1) < JX_REAL_EPSILON && jx_abs(omega - 1) < JX_REAL_EPSILON) {

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/
        if (num_threads > 1) {
            tmp_data = jx_CTAlloc(JX_Real, n);

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
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                    /* if diag is singular, then skip this point */
                    if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] = out_vec[k];
                        }
                    }
                } /* for loop over points */
            } /* foor loop over threads */
            jx_TFree(tmp_data);
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
                    jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii]; */
                    jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
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
        res0_vec = jx_CTAlloc(JX_Real, block_size);
        res2_vec = jx_CTAlloc(JX_Real, block_size);

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/

        if (num_threads > 1) {
            tmp_data = jx_CTAlloc(JX_Real, n);
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
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                            /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                            jx_BSRMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] *= prod;
                       x_data[i] += relax_weight*(omega*res + res0 +
                       one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                    for (k = 0; k < block_size; k++) {
                        tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                    }
                    if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] *= prod;
                            x_data[i * block_size + k] += relax_weight * out_vec[k];
                        }
                    }

                } /* end of loop over points */
            } /* end of loop over threads */
            jx_TFree(tmp_data);
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
                    jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                    /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                    jx_BSRMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii];*/
                    jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] *= prod;
                   x_data[i] += relax_weight*(omega*res + res0 +
                   one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                for (k = 0; k < block_size; k++) {
                    tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                }
                if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                    for (k = 0; k < block_size; k++) {
                        x_data[i * block_size + k] *= prod;
                        x_data[i * block_size + k] += relax_weight * out_vec[k];
                    }
                }
            } /* end of loop over points */
        } /* end num_threads = 1 */

        jx_TFree(res0_vec);
        jx_TFree(res2_vec);
    } /* end of check relax weight and omega */

    /* Free temporary arrays */
    if (num_procs > 1) {
        jx_TFree(Vext_data);
        jx_TFree(v_buf_data);
    }

    jx_TFree(res_vec);
    jx_TFree(out_vec);
    jx_TFree(tmp_vec);
}

/*-----------------------------------------------------------------
  HSGS:  Hybrid: Jacobi off-processor, Symm. Gauss-Seidel/ SSOR on-processor with outer relaxation parameter
 *-----------------------------------------------------------------*/
// forward_or_backward don't be used here: 1: forward, -1: backward
void jx_ParBSRHSGSRelax(jx_ParBSRMatrix* A, jx_ParVector* x, jx_ParVector* b, jx_ParVector* Vtemp, JX_Real relax_weight,
                            JX_Real omega, JX_Int forward_or_backward)
{

    MPI_Comm comm = jx_ParBSRMatrixComm(A);

    jx_BSRMatrix* A_diag      = jx_ParBSRMatrixDiag(A);
    JX_Real*      A_diag_data = jx_BSRMatrixData(A_diag);
    JX_Int*       A_diag_i    = jx_BSRMatrixI(A_diag);
    JX_Int*       A_diag_j    = jx_BSRMatrixJ(A_diag);

    jx_BSRMatrix* A_offd      = jx_ParBSRMatrixOffd(A);
    JX_Int*       A_offd_i    = jx_BSRMatrixI(A_offd);
    JX_Real*      A_offd_data = jx_BSRMatrixData(A_offd);
    JX_Int*       A_offd_j    = jx_BSRMatrixJ(A_offd);

    jx_ParCSRCommPkg*    comm_pkg = jx_ParBSRMatrixCommPkg(A);
    jx_ParCSRCommHandle* comm_handle;

    JX_Int block_size = jx_BSRMatrixBlockSize(A_diag);
    JX_Int bnnz       = block_size * block_size;

    JX_BigInt n_global;
    JX_Int    n             = jx_BSRMatrixNumRows(A_diag);
    JX_Int    num_cols_offd = jx_BSRMatrixNumCols(A_offd);
    JX_BigInt first_index   = jx_ParVectorFirstIndex(x);

    jx_Vector* x_local = jx_ParVectorLocalVector(x);
    JX_Real*   x_data  = jx_VectorData(x_local);

    jx_Vector* b_local = jx_ParVectorLocalVector(b);
    JX_Real*   b_data  = jx_VectorData(b_local);

    jx_Vector* Vtemp_local = jx_ParVectorLocalVector(Vtemp);
    JX_Real*   Vtemp_data  = jx_VectorData(Vtemp_local);
    JX_Real*   Vext_data;
    JX_Real*   v_buf_data;

    JX_Real* tmp_data;

    JX_Int size, rest, ne, ns;

    JX_Int i, j, k;
    JX_Int ii, jj;

    JX_Int relax_error = 0;
    JX_Int num_sends;
    JX_Int index, start;
    JX_Int num_procs, num_threads, my_id;

    JX_Real *res_vec, *out_vec, *tmp_vec;
    JX_Real *res0_vec, *res2_vec;
    JX_Real  one_minus_weight;
    JX_Real  one_minus_omega;
    JX_Real  prod;

    jx_CSRMatrix* A_CSR;
    JX_Int*       A_CSR_i;
    JX_Int*       A_CSR_j;
    JX_Real*      A_CSR_data;

    jx_Vector* f_vector;
    JX_Real*   f_vector_data;

    jx_ParCSRMatrix* A_ParCSR;

    JX_Real* A_mat;
    JX_Real* b_vec;

    JX_Int column;

    /* initialize some stuff */
    one_minus_weight = 1.0 - relax_weight;
    one_minus_omega  = 1.0 - omega;
    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);
    /* num_threads = jx_NumThreads(); */
    num_threads = 1;

    res_vec = jx_CTAlloc(JX_Real, block_size);
    out_vec = jx_CTAlloc(JX_Real, block_size);
    tmp_vec = jx_CTAlloc(JX_Real, block_size);

    if (!comm_pkg) {
        jx_BlockMatvecCommPkgCreate(A);
        comm_pkg = jx_ParBSRMatrixCommPkg(A);
    }

    /*-----------------------------------------------------------------
                  Hybrid: Jacobi off-processor,
                  Symm. Gauss-Seidel/ SSOR on-processor
                  with outer relaxation parameter
     *-----------------------------------------------------------------*/

    if (num_procs > 1) {
        num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);

        v_buf_data = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * block_size);

        Vext_data = jx_CTAlloc(JX_Real, num_cols_offd * block_size);

        if (num_cols_offd) {
            A_offd_j    = jx_BSRMatrixJ(A_offd);
            A_offd_data = jx_BSRMatrixData(A_offd);
        }

        index = 0;
        for (i = 0; i < num_sends; i++) {
            start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++) {
                for (k = 0; k < block_size; k++) {
                    v_buf_data[index++] = x_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg, j) * block_size + k];
                }
            }
        }

        /* we need to use the block comm handle here - since comm_pkg is nodal based */
        comm_handle = jx_ParBSRCommHandleCreate(1, block_size, comm_pkg, v_buf_data, Vext_data);

        jx_ParBSRCommHandleDestroy(comm_handle);
        comm_handle = NULL;
    }

    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/

    if (jx_abs(relax_weight - 1) < JX_REAL_EPSILON && jx_abs(omega - 1) < JX_REAL_EPSILON) {
        if (num_threads > 1) {
            tmp_data = jx_CTAlloc(JX_Real, n);
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
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];

                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                    /* if diag is singular, then skip this point */
                    if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
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
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);

                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii]; */
                        jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                    /* if diag is singular, then skip this point */
                    if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] = out_vec[k];
                        }
                    }
                } /* end of loop over points */
            } /* end of loop over threads */
            jx_TFree(tmp_data);
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
                    jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];

                    /* res -= A_offd_data[jj] * Vext_data[ii]; */
                    jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
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
                    jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii]; */
                    jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] = res / A_diag_data[A_diag_i[i]]; */
                if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], res_vec, out_vec, block_size) == 0) {
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
        res0_vec = jx_CTAlloc(JX_Real, block_size);
        res2_vec = jx_CTAlloc(JX_Real, block_size);
        for (i = 0; i < n; i++) {
            Vtemp_data[i] = x_data[i];
        }
        prod = (1.0 - relax_weight * omega);

        if (num_threads > 1) {
            tmp_data = jx_CTAlloc(JX_Real, n);
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
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                            /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                            jx_BSRMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }

                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii]; */
                        jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] *= prod;
                       x_data[i] += relax_weight*(omega*res + res0 +
                       one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                    for (k = 0; k < block_size; k++) {
                        tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                    }
                    if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
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
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                            /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                            jx_BSRMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                        } else {
                            /* res -= A_diag_data[jj] * tmp_data[ii]; */
                            jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &tmp_data[ii * block_size], 1.0, res_vec, block_size);
                        }
                    }
                    for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                    }
                    /* x_data[i] *= prod;
                       x_data[i] += relax_weight*(omega*res + res0 +
                       one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                    for (k = 0; k < block_size; k++) {
                        tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                    }
                    if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                        for (k = 0; k < block_size; k++) {
                            x_data[i * block_size + k] *= prod;
                            x_data[i * block_size + k] += relax_weight * out_vec[k];
                        }
                    }
                } /* end of loop over points */
            } /* loop over threads end */
            jx_TFree(tmp_data);

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
                    jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                    /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                    jx_BSRMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii];*/
                    jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] *= prod;
                   x_data[i] += relax_weight*(omega*res + res0 +
                   one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                for (k = 0; k < block_size; k++) {
                    tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                }
                if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
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
                    jx_BSRMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz], &x_data[ii * block_size], 1.0, res0_vec, block_size);
                    /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                    jx_BSRMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz], &Vtemp_data[ii * block_size], 1.0, res2_vec, block_size);
                }
                for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++) {
                    ii = A_offd_j[jj];
                    /* res -= A_offd_data[jj] * Vext_data[ii];*/
                    jx_BSRMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz], &Vext_data[ii * block_size], 1.0, res_vec, block_size);
                }
                /* x_data[i] *= prod;
                   x_data[i] += relax_weight*(omega*res + res0 +
                   one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                for (k = 0; k < block_size; k++) {
                    tmp_vec[k] = omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                }
                if (jx_BSRMatrixBlockInvMatvec(&A_diag_data[A_diag_i[i] * bnnz], tmp_vec, out_vec, block_size) == 0) {
                    for (k = 0; k < block_size; k++) {
                        x_data[i * block_size + k] *= prod;
                        x_data[i * block_size + k] += relax_weight * out_vec[k];
                    }
                }
            } /* end of loop over points */
        } /* end num_threads = 1 */

        jx_TFree(res0_vec);
        jx_TFree(res2_vec);
    } /* end of check relax weight and omega */

    /* Free temporary arrays */
    if (num_procs > 1) {
        jx_TFree(Vext_data);
        jx_TFree(v_buf_data);
    }

    jx_TFree(res_vec);
    jx_TFree(out_vec);
    jx_TFree(tmp_vec);
}