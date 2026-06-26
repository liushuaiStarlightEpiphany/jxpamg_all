//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_interp_UA.c
 *  Date: 2025/06/03
 */ 
//========================================================================//
//  JXPAMG - Optimized UA Interpolation                                 //
//========================================================================//

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGBuildUAInterp
 * \brief Optimized Unsmoothed Aggregation Interpolation
 * 
 * Directly uses aggregate IDs as column indices, similar to FASP approach
 */
JX_Int
jx_PAMGBuildUAInterp(jx_ParCSRMatrix *par_A,
                     JX_Int *aggregates,
                     JX_Int num_aggregates,
                     JX_Int *num_cpts_global,
                     JX_Int num_functions,
                     JX_Int *dof_func,
                     JX_Int debug_flag,
                     jx_ParCSRMatrix **P_ptr)
{
    MPI_Comm comm = jx_ParCSRMatrixComm(par_A);
    jx_CSRMatrix *A_diag = jx_ParCSRMatrixDiag(par_A);
    JX_Int num_rows = jx_CSRMatrixNumRows(A_diag);
    
    JX_Int my_id, num_procs;
    JX_Int i,j;
    jx_MPI_Comm_rank(comm, &my_id);
    jx_MPI_Comm_size(comm, &num_procs);
    
    // printf("jx_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    // Step 1: Validate aggregates (ensure they are within valid range)
    for ( i = 0; i < num_rows; i++) {
        if (aggregates[i] < 0 || aggregates[i] >= num_aggregates) {
            // Handle invalid aggregates - assign to first aggregate
            aggregates[i] = 0;
        }
    }
    // printf("jx_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    
    // Step 2: Build boolean prolongation operator directly using aggregate IDs
    JX_Int *P_i = jx_CTAlloc(JX_Int, num_rows + 1);
    
    // Build CSR structure - each row has exactly one entry
    for ( i = 0,j = 0; i < num_rows; i++) {
        P_i[i] = j;
        if (aggregates[i] > -1) j++;
    }
    P_i[num_rows] = j;
    // jx_printf("nnz = %d\n", j);
    JX_Int nnz = j; // Each fine point maps to exactly one coarse point
    JX_Int *P_j = jx_CTAlloc(JX_Int, nnz);
    JX_Real *P_data = jx_CTAlloc(JX_Real, nnz);
    // Directly use aggregate ID as column index (like FASP)
    // second run
    for (i = 0, j = 0; i < num_rows; i ++) {
        P_i[i] = j;
        if (aggregates[i] > -1) {
            P_j[j] = aggregates[i];
            // jx_printf("P_j[%d] = %d\n", j,P_j[j]);
            P_data[j] = 1.0;
            j++;
        }
    }
    // printf("jx_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    
    // Step 3: Create parallel matrix
#ifdef JX_NO_GLOBAL_PARTITION
    JX_Int total_global_cpts;
    if (my_id == (num_procs - 1)) {
        total_global_cpts = num_cpts_global[1];
    }
    jx_MPI_Bcast(&total_global_cpts, 1, JX_MPI_INT, num_procs - 1, comm);
#else
    JX_Int total_global_cpts = num_cpts_global[num_procs];
//    jx_printf("total_global_cpts = %d\n", total_global_cpts);
#endif
    
    jx_ParCSRMatrix *par_P = jx_ParCSRMatrixCreate(comm,
                                                  jx_ParCSRMatrixGlobalNumRows(par_A),
                                                  total_global_cpts,
                                                  jx_ParCSRMatrixColStarts(par_A),
                                                  num_cpts_global,
                                                  0, nnz, 0);
    
    // Set diagonal part
    jx_CSRMatrix *P_diag = jx_ParCSRMatrixDiag(par_P);
    jx_CSRMatrixI(P_diag) = P_i;
    jx_CSRMatrixJ(P_diag) = P_j;
    jx_CSRMatrixData(P_diag) = P_data;
    jx_CSRMatrixNumNonzeros(P_diag) = nnz;
    // printf("jx_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    
    // Off-diagonal part is empty for UA interpolation
    jx_CSRMatrix *P_offd = jx_ParCSRMatrixOffd(par_P);
    jx_CSRMatrixI(P_offd) = jx_CTAlloc(JX_Int, num_rows + 1);
    for ( i = 0; i <= num_rows; i++) {
        jx_CSRMatrixI(P_offd)[i] = 0;
    }
    // printf("jx_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    
    // Set column map for off-diagonal part (even though it's empty)
    // This is necessary for JXPAMG's internal consistency
    JX_Int num_cols_offd = 0;
    jx_ParCSRMatrixColMapOffd(par_P) = jx_CTAlloc(JX_Int, num_cols_offd);
    jx_CSRMatrixNumCols(P_offd) = num_cols_offd;
    
    // Set communication package
    jx_ParCSRMatrixOwnsRowStarts(par_P) = 0;
    
    *P_ptr = par_P;
    
    return 0;
}