//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_coarsen_VMB.c
 *  Date: 2025/06/03
 */ 

#include "jxf_pamg.h"

#define C_PT 1
#define F_PT -1
#define Z_PT -2
#define SF_PT -3  /* special fine points */
#define UNDECIDED 0 

/*!
 * \fn JXF_Int jxf_PAMGCoarsenVMB
 * \brief VMB's coarsening algorithm for unsmoothed aggregation
 * \date 2025/06/03
 */ 
JXF_Int
jxf_PAMGCoarsenVMB( jxf_ParCSRMatrix    *par_S,
                   jxf_ParCSRMatrix    *par_A,
                   JXF_Int                 measure_type,
                   JXF_Int                 debug_flag,
                   JXF_Int               **CF_marker_ptr )
{
      MPI_Comm             comm     = jxf_ParCSRMatrixComm(par_S);
      jxf_ParCSRCommPkg    *comm_pkg = jxf_ParCSRMatrixCommPkg(par_S);
      
      jxf_CSRMatrix *S_diag        = jxf_ParCSRMatrixDiag(par_S);
      jxf_CSRMatrix *S_offd        = jxf_ParCSRMatrixOffd(par_S);
      JXF_Int          *S_i           = jxf_CSRMatrixI(S_diag);
      JXF_Int          *S_j           = jxf_CSRMatrixJ(S_diag);
      JXF_Int          *S_offd_i      = jxf_CSRMatrixI(S_offd);
      JXF_Int          *S_offd_j      = jxf_CSRMatrixJ(S_offd);
      JXF_Int           num_variables = jxf_CSRMatrixNumRows(S_diag);
      JXF_Int           num_cols_offd = jxf_CSRMatrixNumCols(S_offd);
      JXF_Int          *col_map_offd  = jxf_ParCSRMatrixColMapOffd(par_S);
      
      JXF_Int          *CF_marker;
      JXF_Int          *CF_marker_offd = NULL;
      JXF_Int          *int_buf_data = NULL;
      
      JXF_Int           i, j, k, n;
      JXF_Int           num_procs, my_id;
      JXF_Int           num_sends = 0;
      JXF_Int           ierr = 0;
      JXF_Real        wall_time = 0.0;
      
      JXF_Int          *visited;
      JXF_Int          *aggregate_ids;
      JXF_Int           num_aggregates = 0;
      JXF_Real          strong_threshold = 0.25; // Default strength threshold
      
      jxf_MPI_Comm_size(comm, &num_procs);
      jxf_MPI_Comm_rank(comm, &my_id);
      
      if (debug_flag == 3) wall_time = jxf_time_getWallclockSeconds();
      
      // Allocate CF_marker array
      CF_marker = jxf_CTAlloc(JXF_Int, num_variables);
      
      // Initialize all points as unvisited (UNDECIDED)
      for (i = 0; i < num_variables; i++) {
         CF_marker[i] = UNDECIDED;
      }
      
      // Allocate visited array
      visited = jxf_CTAlloc(JXF_Int, num_variables);
      for (i = 0; i < num_variables; i++) {
         visited[i] = 0;
      }
      
      // Allocate aggregate IDs array
      aggregate_ids = jxf_CTAlloc(JXF_Int, num_variables);
      for (i = 0; i < num_variables; i++) {
         aggregate_ids[i] = -1;
      }
      
      // VMB Aggregation Algorithm - Revised implementation
      for (i = 0; i < num_variables; i++) {
         if (CF_marker[i] != UNDECIDED) continue;
         
         // Skip isolated points (no connections)
         if ((S_i[i+1] - S_i[i]) == 0 && (S_offd_i[i+1] - S_offd_i[i]) == 0) {
               CF_marker[i] = SF_PT; // Special fine point
               continue;
         }
         
         // Start new aggregate
         num_aggregates++;
         CF_marker[i] = num_aggregates;
         aggregate_ids[i] = num_aggregates;
         
         // Add strongly connected neighbors to the same aggregate
         for (j = S_i[i]; j < S_i[i+1]; j++) {
               n = S_j[j];
               if (CF_marker[n] == UNDECIDED) {
                  // Check if this is a strong connection
                  // For Boolean strength matrix, all connections are considered strong
                  CF_marker[n] = num_aggregates;
                  aggregate_ids[n] = num_aggregates;
               }
         }
         
         // Process off-diagonal connections
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++) {
               n = S_offd_j[j];
               // For off-processor points, we need to handle them differently
               // This will be handled in the communication phase
         }
      }
      
      // Exchange boundary data for CF_marker in parallel
      if (num_procs > 1 && comm_pkg) {
         num_sends = jxf_ParCSRCommPkgNumSends(comm_pkg);
         
         // 确保通信包有效
         if (num_sends > 0) {
            JXF_Int send_size = jxf_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
            
            // 分配缓冲区内存
            CF_marker_offd = jxf_CTAlloc(JXF_Int, num_cols_offd);
            int_buf_data = jxf_CTAlloc(JXF_Int, send_size);
            
            // 确保内存分配成功
            if (CF_marker_offd == NULL || int_buf_data == NULL) {
                  jxf_printf("ERROR: Failed to allocate memory for communication buffers\n");
                  return -1;
            }
            
            // 初始化缓冲区
            for (i = 0; i < num_cols_offd; i++) {
                  CF_marker_offd[i] = UNDECIDED;
            }
            
            for (i = 0; i < send_size; i++) {
                  int_buf_data[i] = UNDECIDED;
            }
         
         // 打包发送数据
         JXF_Int index = 0;
         for (i = 0; i < num_sends; i++) {
               JXF_Int start = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
               JXF_Int end = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
               for (j = start; j < end; j++) {
                  JXF_Int elmt = jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j);
                  if (elmt >= 0 && elmt < num_variables) {
                     int_buf_data[index] = CF_marker[elmt];
                  } else {
                     int_buf_data[index] = UNDECIDED;
                  }
                  index++;
               }
         }
         
         // Communicate CF markers
         jxf_ParCSRCommHandle *comm_handle = jxf_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
         if (comm_handle == NULL) {
            jxf_printf("ERROR: Failed to create communication handle\n");
            return -1;
         }
         jxf_ParCSRCommHandleDestroy(comm_handle);
         
         // Process received data - resolve conflicts at boundaries
         // This is a simplified approach - in practice, you may need a more sophisticated algorithm
         for (i = 0; i < num_cols_offd; i++) {
               if (CF_marker_offd[i] > 0) {
                  // Find the local point corresponding to this off-processor point
                  for (j = 0; j < num_variables; j++) {
                     for (k = S_offd_i[j]; k < S_offd_i[j+1]; k++) {
                           if (S_offd_j[k] == i) {
                              // If the local point is not yet assigned, assign it to the same aggregate
                              if (CF_marker[j] == UNDECIDED) {
                                 CF_marker[j] = CF_marker_offd[i];
                              }
                              break;
                           }
                     }
                  }
               }
         }
         }
         jxf_TFree(CF_marker_offd);
         jxf_TFree(int_buf_data);
      }
      
      // Final pass: ensure all points are assigned
      for (i = 0; i < num_variables; i++) {
         if (CF_marker[i] == UNDECIDED) {
               // Assign to nearest aggregate or mark as fine point
               JXF_Int assigned = 0;
               
               // Check diagonal connections
               for (j = S_i[i]; j < S_i[i+1]; j++) {
                  n = S_j[j];
                  if (CF_marker[n] > 0 && CF_marker[n] != SF_PT) {
                     CF_marker[i] = CF_marker[n];
                     assigned = 1;
                     break;
                  }
               }
               
               // Check off-diagonal connections
               if (!assigned) {
                  for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++) {
                     n = S_offd_j[j];
                     // For off-processor points, we can't directly access their CF_marker
                     // This would require additional communication
                  }
               }
               
               // If still not assigned, mark as fine point
               if (!assigned) {
                  CF_marker[i] = F_PT;
               }
         }
      }
      
      if (debug_flag == 3) {
         wall_time = jxf_time_getWallclockSeconds() - wall_time;
         jxf_printf("Proc = %d    VMB Coarsening Time = %f, Num Aggregates = %d\n", 
                     my_id, wall_time, num_aggregates);
      }
      
      jxf_TFree(visited);
      jxf_TFree(aggregate_ids);
      
      *CF_marker_ptr = CF_marker;
      return ierr;
}