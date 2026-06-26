//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_bsr_matvec.c -- basic operations for parallel BSR matrix-vector multiplication.
 *  Date: 2025/10/08
 */ 

#include "jx_mv.h"
#include "jx_parbsr_mv.h"

/*!
 * \fn JX_Int jx_ParBSRMatrixMatvec
 * \brief Perform y := alpha*A*x + beta*y for parallel BSR matrix.
 * \date 2025/10/08
 */
JX_Int
jx_ParBSRMatrixMatvec( JX_Real           alpha,
                        jx_ParBSRMatrix *A,
                        jx_ParVector    *x,
                        JX_Real           beta,
                        jx_ParVector    *y  )
{
   jx_ParCSRCommHandle *comm_handle;
   jx_ParCSRCommPkg    *comm_pkg;
   jx_BSRMatrix        *diag, *offd;
   jx_Vector           *x_local, *y_local, *x_tmp;
   
   JX_BigInt        num_rows = jx_ParBSRMatrixGlobalNumRows(A);
   JX_BigInt        num_cols = jx_ParBSRMatrixGlobalNumCols(A);
   JX_BigInt        x_size   = jx_ParVectorGlobalSize(x);
   JX_BigInt        y_size   = jx_ParVectorGlobalSize(y);
   
   JX_Int           num_cols_offd;
   JX_Int           i, j, k, index;
   JX_Int           blk_size, size;
   JX_Int           start, finish, elem;
   JX_Int           ierr = 0, nprocs, num_sends, mypid;
   
   JX_Real        *x_tmp_data, *x_buf_data, *x_local_data;

   JX_Real         wall_time = 0.0;  /* for debugging instrumentation  */

   if (jx__global_mvcpu_flag) wall_time = jx_time_getWallclockSeconds();

   /*---------------------------------------------------------------------
    *  Get MPI information and matrix components
    *--------------------------------------------------------------------*/
   jx_MPI_Comm_size(jx_ParBSRMatrixComm(A), &nprocs);
   jx_MPI_Comm_rank(jx_ParBSRMatrixComm(A), &mypid);
   
   comm_pkg      = jx_ParBSRMatrixCommPkg(A);
   num_rows      = jx_ParBSRMatrixGlobalNumRows(A);
   num_cols      = jx_ParBSRMatrixGlobalNumCols(A);
   blk_size      = jx_ParBSRMatrixBlockSize(A);
   diag          = jx_ParBSRMatrixDiag(A);
   offd          = jx_ParBSRMatrixOffd(A);
   num_cols_offd = jx_BSRMatrixNumCols(offd);
   x_local       = jx_ParVectorLocalVector(x);
   y_local       = jx_ParVectorLocalVector(y);
   x_size        = jx_ParVectorGlobalSize(x);
   y_size        = jx_ParVectorGlobalSize(y);
   x_local_data  = jx_VectorData(x_local);

    if (!A || !x || !y) {
        printf("Rank %d: ERROR: Null pointer in ParBSRMatrixMatvec!\n", mypid);
        return -1;
    }

   /*---------------------------------------------------------------------
    *  Check for size compatibility.
    *--------------------------------------------------------------------*/

   if (num_cols * blk_size != x_size) {
       ierr = 11;
   }

   if (num_rows * blk_size != y_size) {
       ierr = 12;
   }

   if (num_cols * blk_size != x_size && num_rows * blk_size != y_size) {
       ierr = 13;
   }

   /*---------------------------------------------------------------------
    *  Single process case: just do local computation
    *--------------------------------------------------------------------*/
    if (nprocs == 1) {
        jx_BSRMatrixMatvec(alpha, diag, x_local, beta, y_local);
        
        if (jx__global_mvcpu_flag) jx_total_elapsed_time_matvec += (jx_time_getWallclockSeconds() - wall_time);
        return ierr;
    }

   /*---------------------------------------------------------------------
    *  Multiple processes: need communication
    *--------------------------------------------------------------------*/
   x_tmp = jx_SeqVectorCreate(num_cols_offd * blk_size);
   jx_SeqVectorInitialize(x_tmp);
   x_tmp_data = jx_VectorData(x_tmp);
   /*---------------------------------------------------------------------
     * If there exists no CommPkg for A, a CommPkg is generated
     *--------------------------------------------------------------------*/
   if (!comm_pkg) {
       jx_BlockMatvecCommPkgCreate(A);
       comm_pkg = jx_ParBSRMatrixCommPkg(A);
   }

   num_sends  = jx_ParCSRCommPkgNumSends(comm_pkg);
   size       = jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * blk_size;
   x_buf_data = jx_CTAlloc(JX_Real, size);

   /*---------------------------------------------------------------------
    * Pack send data
    *--------------------------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++) {
       start  = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
       finish = jx_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
       for (j = start; j < finish; j++) {
           elem = jx_ParCSRCommPkgSendMapElmt(comm_pkg, j) * blk_size;
           for (k = 0; k < blk_size; k++) {
               x_buf_data[index++] = x_local_data[elem++];
           }
       }
   }

   /*---------------------------------------------------------------------
     * Create communication handle for BSR
     *--------------------------------------------------------------------*/
   comm_handle = jx_ParBSRCommHandleCreate(1, blk_size, comm_pkg, x_buf_data, x_tmp_data);

    if (comm_handle == NULL) 
    {
        printf("comm_handle is NULL, returning\n");
        return jx_error_flag;
    }

   /*---------------------------------------------------------------------
     * Local computation: y = beta*y + alpha*diag*x
     *--------------------------------------------------------------------*/
   jx_BSRMatrixMatvec(alpha, diag, x_local, beta, y_local);

   /*---------------------------------------------------------------------
     * Wait for communication to complete
     *--------------------------------------------------------------------*/
   jx_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /*---------------------------------------------------------------------
     * Add contribution from off-diagonal part: y += alpha*offd*x_tmp
     *--------------------------------------------------------------------*/
   if (num_cols_offd) {
       jx_BSRMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);
   }

   /*---------------------------------------------------------------------
    * Clean up
    *--------------------------------------------------------------------*/
   jx_SeqVectorDestroy(x_tmp);

   x_tmp = NULL;
   jx_TFree(x_buf_data);

   if (jx__global_mvcpu_flag) jx_total_elapsed_time_matvec += (jx_time_getWallclockSeconds() - wall_time);

   return ierr;
}
