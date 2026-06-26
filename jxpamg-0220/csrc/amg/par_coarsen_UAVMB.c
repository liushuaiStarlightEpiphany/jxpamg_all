//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_coarsen_UAVMB.c
 *  Date: 2025/06/03
 */ 

#include "jx_pamg.h"

#define C_PT 1
#define F_PT -1
#define Z_PT -2
#define SF_PT -3  /* special fine points */
#define UNDECIDED 0 

/*!
 * \fn JX_Int jx_PAMGCoarsenVMB
 * \brief VMB's coarsening algorithm for unsmoothed aggregation
 * \date 2025/06/03
 */ 
//========================================================================//
//  JXPAMG - UA-VMB Coarsening (Aggregate-based)                         //
//========================================================================//

#include "jx_pamg.h"
/*!
 * \fn JX_Int jx_PAMGCoarsenUAVMB
 * \brief Unsmoothed Aggregation with VMB coarsening algorithm (Aggregate-based)
 */
JX_Int
jx_PAMGCoarsenUAVMB(jx_ParCSRMatrix *par_S,
                    jx_ParCSRMatrix *par_A,
                    JX_Int num_functions,
                    JX_Int *dof_func,
                    JX_Int debug_flag,
                    JX_Int **aggregates_ptr,
                    JX_Int *num_aggregates_ptr)
{
   MPI_Comm comm = jx_ParCSRMatrixComm(par_A);
   jx_CSRMatrix *A_diag = jx_ParCSRMatrixDiag(par_A);

   JX_Int num_rows = jx_CSRMatrixNumRows(A_diag);
   
   JX_Int *aggregates;
   JX_Int i, j, k, my_id, num_procs;
   JX_Int agg_id = 0;
   
   jx_MPI_Comm_rank(comm, &my_id);
   jx_MPI_Comm_size(comm, &num_procs);
   
   // Allocate memory for aggregates
   aggregates = jx_CTAlloc(JX_Int, num_rows);
   for (i = 0; i < num_rows; i++) {
      aggregates[i] = UNASSIGNED_AGG;
   }
   
   // Step 1: Create strength matrix
//    jx_ParCSRMatrix *par_S;
//    jx_PAMGCreateS(par_A, strong_threshold, 1.0, num_functions, dof_func, &par_S);
   jx_CSRMatrix *S_diag = jx_ParCSRMatrixDiag(par_S);
   JX_Int *S_i = jx_CSRMatrixI(S_diag);
   JX_Int *S_j = jx_CSRMatrixJ(S_diag);
   
   // Step 2: Identify isolated points (no strong connections)
   for (i = 0; i < num_rows; i++) {
      JX_Int num_strong = S_i[i+1] - S_i[i];
      if (num_strong == 0) {
            aggregates[i] = ISOLATED_POINT;
      }
   }
   
   // Step 3: VMB aggregation main algorithm
   for (i = 0; i < num_rows; i++) {
      if (aggregates[i] != UNASSIGNED_AGG) continue;
      
      // Start new aggregate with current point as root
      aggregates[i] = agg_id;
      
      // Add strongly connected neighbors (distance-1)
      for (j = S_i[i]; j < S_i[i+1]; j++) {
            JX_Int neighbor = S_j[j];
            if (aggregates[neighbor] == UNASSIGNED_AGG) {
               aggregates[neighbor] = agg_id;
            }
      }
      
      // Add distance-2 neighbors (VMB specific)
      for (j = S_i[i]; j < S_i[i+1]; j++) {
            JX_Int neighbor = S_j[j];
            for (k = S_i[neighbor]; k < S_i[neighbor+1]; k++) {
               JX_Int nneighbor = S_j[k];
               if (aggregates[nneighbor] == UNASSIGNED_AGG && nneighbor != i) {
                  aggregates[nneighbor] = agg_id;
               }
            }
      }
      
      agg_id++;
   }

   // 处理孤立点（确保所有点都有聚集）
   for (i = 0; i < num_rows; i++) {
    //   jx_printf("aggregates[%d] = %d\n", i, aggregates[i]);
      if (aggregates[i] == UNASSIGNED_AGG || aggregates[i] == ISOLATED_POINT) {
         // 将孤立点分配到第一个聚集
         aggregates[i] = 0;
      }
   }

   // // 添加全局编号处理：

   // JX_Int global_start = 0;
   // JX_Int local_num_agg = agg_id;  // 本地聚集数量

   // // 计算每个进程的起始聚集编号
   // MPI_Exscan(&local_num_agg, &global_start, 1, JX_MPI_INT, MPI_SUM, comm);

   // // 调整本地聚集编号为全局编号
   // for (i = 0; i < num_rows; i++) {
   //    if (aggregates[i] >= 0) {
   //       aggregates[i] += global_start;
   //    }
   //    // 孤立点保持不变（负值）
   // }
   
   // || VMB now is absolutely local, offd of S is useless || ---------------------------------------------------
//    // Step 4: Handle MPI communication for boundary aggregates
//    if (num_procs > 1) {
//         jx_ParCSRCommPkg *comm_pkg = jx_ParCSRMatrixCommPkg(par_A);
//         if (!comm_pkg) {
//             jx_MatvecCommPkgCreate(par_A);
//             comm_pkg = jx_ParCSRMatrixCommPkg(par_A);
//         }
        
//         JX_Int num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
//         JX_Int num_cols_offd = jx_CSRMatrixNumCols(jx_ParCSRMatrixOffd(par_A));
        
//         // Allocate memory for off-processor aggregate information
//         JX_Int *agg_offd = jx_CTAlloc(JX_Int, num_cols_offd);
        
//         // Pack send buffer - careful indexing
//         JX_Int send_buf_size = jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
//         JX_Int *send_buf = jx_CTAlloc(JX_Int, send_buf_size);
        
//         JX_Int index = 0;
//         for (i = 0; i < num_sends; i++) {
//             JX_Int start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
//             JX_Int end = jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
            
//             for (j = start; j < end; j++) {
//                 JX_Int local_idx = jx_ParCSRCommPkgSendMapElmt(comm_pkg, j);
//                 if (local_idx < 0 || local_idx >= num_rows) {
//                     // Error handling for invalid index
//                     jx_printf("ERROR: Invalid send map element %d at position %d\n", local_idx, j);
//                     send_buf[index++] = ISOLATED_POINT;
//                 } else {
//                     send_buf[index++] = aggregates[local_idx];
//                 }
//             }
//         }
        
//         // Communicate aggregate information
//         jx_ParCSRCommHandle *comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, send_buf, agg_offd);
//         jx_ParCSRCommHandleDestroy(comm_handle);
        
//         // Process received aggregate info - ensure consistency
//         // For boundary points, we need to ensure aggregates match across processes
//         // This is a simplified approach - in production code, you'd need a more robust matching algorithm
        
//         JX_Int *col_map_offd = jx_ParCSRMatrixColMapOffd(par_A);
//         for (i = 0; i < num_cols_offd; i++) {
//             JX_Int global_id = col_map_offd[i];
//             // In a real implementation, you'd need to map this back to local indices
//             // and ensure consistent aggregation across processes
//         }
//    }
   *aggregates_ptr = aggregates;
   *num_aggregates_ptr = agg_id;
//    jx_printf("agg_id = %d\n", agg_id);
   
//    jx_ParCSRMatrixDestroy(par_S);
   
   return 0;
}

/*!
 * \fn JX_Int jx_PAMGGetCoarseParamsFromAggregates
 * \brief Extract coarse parameters from aggregates information
 */
JX_Int
jx_PAMGGetCoarseParamsFromAggregates(MPI_Comm comm,
                                     JX_Int local_num_vars,
                                     JX_Int num_functions,
                                     JX_Int *dof_func,
                                     JX_Int *aggregates,
                                     JX_Int num_aggregates,
                                     JX_Int **coarse_dof_func_ptr,
                                     JX_Int **coarse_pnts_global_ptr)
{
    JX_Int my_id, num_procs;
    JX_Int i, agg;
    jx_MPI_Comm_rank(comm, &my_id);
    jx_MPI_Comm_size(comm, &num_procs);
    
    // Count coarse points (one per aggregate)
    JX_Int coarse_size = num_aggregates;
    
    // Allocate coarse dof function (if needed)
    JX_Int *coarse_dof_func = NULL;
    if (num_functions > 1) {
        coarse_dof_func = jx_CTAlloc(JX_Int, coarse_size);
        // For each aggregate, determine the DOF function from the first point
        for ( agg = 0; agg < num_aggregates; agg++) {
            JX_Int found = 0;
            for ( i = 0; i < local_num_vars && !found; i++) {
                if (aggregates[i] == agg) {
                    coarse_dof_func[agg] = dof_func[i];
                    found = 1;
                }
            }
            if (!found) {
                // Should not happen, but provide a default
                coarse_dof_func[agg] = 0;
            }
        }
    }
    
    // Create global coarse points array
    JX_Int *coarse_pnts_global = jx_CTAlloc(JX_Int, num_procs + 1);
    
    // Gather local coarse sizes from all processes
    JX_Int *local_sizes = jx_CTAlloc(JX_Int, num_procs);
    jx_MPI_Allgather(&coarse_size, 1, JX_MPI_INT, 
                    local_sizes, 1, JX_MPI_INT, comm);
//    jx_printf("coarse_size = %d, local_sizes[0] = %d\n", coarse_size,local_sizes[0]);


    
    // Calculate global offsets
    coarse_pnts_global[0] = 0;
    for (i = 0; i < num_procs; i++) {
        coarse_pnts_global[i+1] = coarse_pnts_global[i] + local_sizes[i];
    }
//    jx_printf("coarse_pnts_global[0] = %d, coarse_pnts_global[1] = %d\n", coarse_pnts_global[0],coarse_pnts_global[1]);

    
    *coarse_dof_func_ptr = coarse_dof_func;
    *coarse_pnts_global_ptr = coarse_pnts_global;
    
    jx_TFree(local_sizes);
    
    return coarse_size;
}
