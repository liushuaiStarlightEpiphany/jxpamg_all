//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_interp_UA.c
 *  Date: 2025/06/03
 */ 
//========================================================================//
//  JXFPAMG - Optimized UA Interpolation                                 //
//========================================================================//

#include "jxf_pamg.h"

/*!
 * \fn JXF_Int jxf_PAMGBuildUAInterp
 * \brief Optimized Unsmoothed Aggregation Interpolation
 * 
 * Directly uses aggregate IDs as column indices, similar to FASP approach
 */
JXF_Int
jxf_PAMGBuildUAInterp(jxf_ParCSRMatrix *par_A,
                     JXF_Int *aggregates,
                     JXF_Int num_aggregates,
                     JXF_Int *num_cpts_global,
                     JXF_Int num_functions,
                     JXF_Int *dof_func,
                     JXF_Int debug_flag,
                     jxf_ParCSRMatrix **P_ptr)
{
    MPI_Comm comm = jxf_ParCSRMatrixComm(par_A);
    jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(par_A);
    JXF_Int num_rows = jxf_CSRMatrixNumRows(A_diag);
    
    JXF_Int my_id, num_procs;
    JXF_Int i,j;
    jxf_MPI_Comm_rank(comm, &my_id);
    jxf_MPI_Comm_size(comm, &num_procs);
    
    // printf("jxf_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    // Step 1: Validate aggregates (ensure they are within valid range)
    for ( i = 0; i < num_rows; i++) {
        if (aggregates[i] < 0 || aggregates[i] >= num_aggregates) {
            // Handle invalid aggregates - assign to first aggregate
            aggregates[i] = 0;
        }
    }
    // printf("jxf_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    
    // Step 2: Build boolean prolongation operator directly using aggregate IDs
    JXF_Int *P_i = jxf_CTAlloc(JXF_Int, num_rows + 1);
    
    // Build CSR structure - each row has exactly one entry
    for ( i = 0,j = 0; i < num_rows; i++) {
        P_i[i] = j;
        if (aggregates[i] > -1) j++;
    }
    P_i[num_rows] = j;
    // jxf_printf("nnz = %d\n", j);
    JXF_Int nnz = j; // Each fine point maps to exactly one coarse point
    JXF_Int *P_j = jxf_CTAlloc(JXF_Int, nnz);
    JXF_Real *P_data = jxf_CTAlloc(JXF_Real, nnz);
    // Directly use aggregate ID as column index (like FASP)
    // second run
    for (i = 0, j = 0; i < num_rows; i ++) {
        P_i[i] = j;
        if (aggregates[i] > -1) {
            P_j[j] = aggregates[i];
            // jxf_printf("P_j[%d] = %d\n", j,P_j[j]);
            P_data[j] = 1.0;
            j++;
        }
    }
    // printf("jxf_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    
    // Step 3: Create parallel matrix
#ifdef JXF_NO_GLOBAL_PARTITION
    JXF_Int total_global_cpts;
    if (my_id == (num_procs - 1)) {
        total_global_cpts = num_cpts_global[1];
    }
    jxf_MPI_Bcast(&total_global_cpts, 1, JXF_MPI_INT, num_procs - 1, comm);
#else
    JXF_Int total_global_cpts = num_cpts_global[num_procs];
//    jxf_printf("total_global_cpts = %d\n", total_global_cpts);
#endif
    
    jxf_ParCSRMatrix *par_P = jxf_ParCSRMatrixCreate(comm,
                                                  jxf_ParCSRMatrixGlobalNumRows(par_A),
                                                  total_global_cpts,
                                                  jxf_ParCSRMatrixColStarts(par_A),
                                                  num_cpts_global,
                                                  0, nnz, 0);
    
    // Set diagonal part
    jxf_CSRMatrix *P_diag = jxf_ParCSRMatrixDiag(par_P);
    jxf_CSRMatrixI(P_diag) = P_i;
    jxf_CSRMatrixJ(P_diag) = P_j;
    jxf_CSRMatrixData(P_diag) = P_data;
    jxf_CSRMatrixNumNonzeros(P_diag) = nnz;
    // printf("jxf_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    
    // Off-diagonal part is empty for UA interpolation
    jxf_CSRMatrix *P_offd = jxf_ParCSRMatrixOffd(par_P);
    jxf_CSRMatrixI(P_offd) = jxf_CTAlloc(JXF_Int, num_rows + 1);
    for ( i = 0; i <= num_rows; i++) {
        jxf_CSRMatrixI(P_offd)[i] = 0;
    }
    // printf("jxf_PAMGBuildUAInterp,  %s, %d\n", __FUNCTION__, __LINE__);
    
    // Set column map for off-diagonal part (even though it's empty)
    // This is necessary for JXFPAMG's internal consistency
    JXF_Int num_cols_offd = 0;
    jxf_ParCSRMatrixColMapOffd(par_P) = jxf_CTAlloc(JXF_Int, num_cols_offd);
    jxf_CSRMatrixNumCols(P_offd) = num_cols_offd;
    
    // Set communication package
    jxf_ParCSRMatrixOwnsRowStarts(par_P) = 0;
    
    *P_ptr = par_P;
    
    return 0;
}