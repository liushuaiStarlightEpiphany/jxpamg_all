//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_bsr_matvec.c -- basic operations for parallel BSR matrix-vector multiplication.
 *  Date: 2025/10/08
 */ 

#include "jxf_mv.h"
#include "jxf_parbsr_mv.h"

/*!
 * \fn JXF_Int jxf_ParBSRMatrixMatvec
 * \brief Perform y := alpha*A*x + beta*y for parallel BSR matrix.
 * \date 2025/10/08
 */
JXF_Int
jxf_ParBSRMatrixMatvec( JXF_Real           alpha,
                       jxf_ParBSRMatrix *A,
                       jxf_ParVector    *x,
                       JXF_Real           beta,
                       jxf_ParVector    *y  )
{
   jxf_ParCSRCommHandle *comm_handle;
   jxf_ParCSRCommPkg    *comm_pkg;
   jxf_BSRMatrix        *diag, *offd;
   jxf_Vector           *x_local, *y_local, *x_tmp;
   
   JXF_BigInt        num_rows = jxf_ParBSRMatrixGlobalNumRows(A);
   JXF_BigInt        num_cols = jxf_ParBSRMatrixGlobalNumCols(A);
   JXF_BigInt        x_size   = jxf_ParVectorGlobalSize(x);
   JXF_BigInt        y_size   = jxf_ParVectorGlobalSize(y);
   
   JXF_Int           num_cols_offd;
   JXF_Int           i, j, k, index;
   JXF_Int           blk_size, size;
   JXF_Int           start, finish, elem;
   JXF_Int           ierr = 0, nprocs, num_sends, mypid;
   
   JXF_Real        *x_tmp_data, *x_buf_data, *x_local_data;

   JXF_Real         wall_time = 0.0;  /* for debugging instrumentation  */

   if (jxf__global_mvcpu_flag) wall_time = jxf_time_getWallclockSeconds();

   /*---------------------------------------------------------------------
    *  Get MPI information and matrix components
    *--------------------------------------------------------------------*/
   jxf_MPI_Comm_size(jxf_ParBSRMatrixComm(A), &nprocs);
   jxf_MPI_Comm_rank(jxf_ParBSRMatrixComm(A), &mypid);
   
   comm_pkg      = jxf_ParBSRMatrixCommPkg(A);
   num_rows      = jxf_ParBSRMatrixGlobalNumRows(A);
   num_cols      = jxf_ParBSRMatrixGlobalNumCols(A);
   blk_size      = jxf_ParBSRMatrixBlockSize(A);
   diag          = jxf_ParBSRMatrixDiag(A);
   offd          = jxf_ParBSRMatrixOffd(A);
   num_cols_offd = jxf_BSRMatrixNumCols(offd);
   x_local       = jxf_ParVectorLocalVector(x);
   y_local       = jxf_ParVectorLocalVector(y);
   x_size        = jxf_ParVectorGlobalSize(x);
   y_size        = jxf_ParVectorGlobalSize(y);
   x_local_data  = jxf_VectorData(x_local);


    // printf("Rank %d: Entering jxf_ParBSRMatrixMatvec\n", mypid);
    // printf("Rank %d: A=%p, x=%p, y=%p\n", mypid, (void*)A, (void*)x, (void*)y);
    // fflush(stdout);
    
    if (!A || !x || !y) {
        printf("Rank %d: ERROR: Null pointer in ParBSRMatrixMatvec!\n", mypid);
        fflush(stdout);
        return -1;
    }
    
    // 检查对角和离对角部分
    // printf("Rank %d: A->diag=%p, A->offd=%p\n", mypid, 
    //        (void*)jxf_ParBSRMatrixDiag(A), (void*)jxf_ParBSRMatrixOffd(A));
    // fflush(stdout);


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
       jxf_BSRMatrixMatvec(alpha, diag, x_local, beta, y_local);
       
       if (jxf__global_mvcpu_flag) jxf_total_elapsed_time_matvec += (jxf_time_getWallclockSeconds() - wall_time);
       return ierr;
   }

   /*---------------------------------------------------------------------
    *  Multiple processes: need communication
    *--------------------------------------------------------------------*/
   x_tmp = jxf_SeqVectorCreate(num_cols_offd * blk_size);
   jxf_SeqVectorInitialize(x_tmp);
   x_tmp_data = jxf_VectorData(x_tmp);
    // printf("Rank %d: Entering jxf_SeqVectorInitialize\n", mypid);
   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated
    *--------------------------------------------------------------------*/
   if (!comm_pkg) {
        // printf("Rank %d: Entering jxf_BlockMatvecCommPkgCreate\n", mypid);
       jxf_BlockMatvecCommPkgCreate(A);
    //    printf("Rank %d: After jxf_BlockMatvecCommPkgCreate\n", mypid);
       comm_pkg = jxf_ParBSRMatrixCommPkg(A);
       
   }

   num_sends  = jxf_ParCSRCommPkgNumSends(comm_pkg);
   size       = jxf_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * blk_size;
   x_buf_data = jxf_CTAlloc(JXF_Real, size);

    // printf("Rank %d: after jxf_BlockMatvecCommPkgCreate\n", mypid);

   /*---------------------------------------------------------------------
    * Pack send data
    *--------------------------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++) {
       start  = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
       finish = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
       for (j = start; j < finish; j++) {
           elem = jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j) * blk_size;
           for (k = 0; k < blk_size; k++) {
               x_buf_data[index++] = x_local_data[elem++];
           }
       }
   }

    // printf("Rank %d: Entering Pack send data\n", mypid);

   /*---------------------------------------------------------------------
    * Create communication handle for BSR
    *--------------------------------------------------------------------*/
   comm_handle = jxf_ParBSRCommHandleCreate(1, blk_size, comm_pkg, x_buf_data, x_tmp_data);

    // jxf_MPI_Barrier(jxf_ParBSRMatrixComm(A));
    // printf("Rank %d: after jxf_ParBSRCommHandleCreate\n", mypid);
    // printf("Rank %d: after jxf_ParBSRCommHandleCreate, comm_handle=%p\n", mypid, (void*)comm_handle);
    // fflush(stdout);
    
    if (comm_handle == NULL) 
    {
        printf("comm_handle is NULL, returning\n");
        fflush(stdout);
        return jxf_error_flag;
    }
    // // 验证设置
    // printf("Rank %d: Set comm_handle fields:\n", mypid);
    // printf("  comm_pkg = %p\n", (void*)comm_handle->comm_pkg);
    // printf("  send_data = %p\n", (void*)comm_handle->send_data);
    // printf("  recv_data = %p\n", (void*)comm_handle->recv_data);
    // printf("  num_requests = %d\n", comm_handle->num_requests);
    // printf("  requests = %p\n", (void*)comm_handle->requests);
    // fflush(stdout);
   /*---------------------------------------------------------------------
    * Local computation: y = beta*y + alpha*diag*x
    *--------------------------------------------------------------------*/
   jxf_BSRMatrixMatvec(alpha, diag, x_local, beta, y_local);

    // printf("Rank %d: Entering jxf_BSRMatrixMatvec\n", mypid);

   /*---------------------------------------------------------------------
    * Wait for communication to complete
    *--------------------------------------------------------------------*/
//        // 验证设置
//     printf("Rank %d: Set comm_handle fields:\n", mypid);
//     printf("  comm_pkg = %p\n", (void*)comm_handle->comm_pkg);
//     printf("  send_data = %p\n", (void*)comm_handle->send_data);
//     printf("  recv_data = %p\n", (void*)comm_handle->recv_data);
//     printf("  num_requests = %d\n", comm_handle->num_requests);
//     printf("  requests = %p\n", (void*)comm_handle->requests);
//     fflush(stdout);

   jxf_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

    // printf("Rank %d: Entering jxf_ParBSRCommHandleDestroy\n", mypid);

   /*---------------------------------------------------------------------
    * Add contribution from off-diagonal part: y += alpha*offd*x_tmp
    *--------------------------------------------------------------------*/
   if (num_cols_offd) {
       jxf_BSRMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);
   }

    // printf("Rank %d: Entering jxf_BSRMatrixMatvec\n", mypid);

   /*---------------------------------------------------------------------
    * Clean up
    *--------------------------------------------------------------------*/
   jxf_SeqVectorDestroy(x_tmp);
    // printf("Rank %d: Entering jxf_SeqVectorDestroy\n", mypid);

   x_tmp = NULL;
   jxf_TFree(x_buf_data);

   if (jxf__global_mvcpu_flag) jxf_total_elapsed_time_matvec += (jxf_time_getWallclockSeconds() - wall_time);

   return ierr;
}

/*!
 * \fn JXF_Int jxf_ParBSRMatrixMatvecT
 * \brief Perform y := alpha*A^T*x + beta*y for parallel BSR matrix.
 * \date 2025/10/08
 */
JXF_Int
jxf_ParBSRMatrixMatvecT( JXF_Real           alpha,
                        jxf_ParBSRMatrix *A,
                        jxf_ParVector    *x,
                        JXF_Real           beta,
                        jxf_ParVector    *y  )
{
   jxf_ParCSRCommHandle *comm_handle;
   jxf_ParCSRCommPkg    *comm_pkg = jxf_ParBSRMatrixCommPkg(A);
   jxf_BSRMatrix        *diag     = jxf_ParBSRMatrixDiag(A);
   jxf_BSRMatrix        *offd     = jxf_ParBSRMatrixOffd(A);
   jxf_Vector           *x_local  = jxf_ParVectorLocalVector(x);
   jxf_Vector           *y_local  = jxf_ParVectorLocalVector(y);
   jxf_Vector           *y_tmp;
   
   JXF_Real        *y_local_data;
   JXF_Int          blk_size = jxf_ParBSRMatrixBlockSize(A);
   JXF_BigInt       x_size   = jxf_ParVectorGlobalSize(x);
   JXF_BigInt       y_size   = jxf_ParVectorGlobalSize(y);
   JXF_Real        *y_tmp_data, *y_buf_data;

   JXF_BigInt       num_rows      = jxf_ParBSRMatrixGlobalNumRows(A);
   JXF_BigInt       num_cols      = jxf_ParBSRMatrixGlobalNumCols(A);
   JXF_Int          num_cols_offd = jxf_BSRMatrixNumCols(offd);

   JXF_Int          i, j, index, start, finish, elem, num_sends;
   JXF_Int          size, k;

   JXF_Int          ierr = 0, nprocs;

   JXF_Real         wall_time = 0.0;  /* for debugging instrumentation  */

   if (jxf__global_mvcpu_flag) wall_time = jxf_time_getWallclockSeconds();

   /*---------------------------------------------------------------------
    *  Check for size compatibility
    *--------------------------------------------------------------------*/
   jxf_MPI_Comm_size(jxf_ParBSRMatrixComm(A), &nprocs);

   if (num_rows * blk_size != x_size) {
       ierr = 1;
   }

   if (num_cols * blk_size != y_size) {
       ierr = 2;
   }

   if (num_rows * blk_size != x_size && num_cols * blk_size != y_size) {
       ierr = 3;
   }

   /*---------------------------------------------------------------------
    *  Single process case
    *--------------------------------------------------------------------*/
   if (nprocs == 1) {
       jxf_BSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);
       
       if (jxf__global_mvcpu_flag) jxf_total_elapsed_time_matvec += (jxf_time_getWallclockSeconds() - wall_time);
       return ierr;
   }

   /*---------------------------------------------------------------------
    *  Multiple processes case
    *--------------------------------------------------------------------*/
   y_tmp = jxf_SeqVectorCreate(num_cols_offd * blk_size);
   jxf_SeqVectorInitialize(y_tmp);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, create one
    *--------------------------------------------------------------------*/
   if (!comm_pkg) {
       jxf_BlockMatvecCommPkgCreate(A);
       comm_pkg = jxf_ParBSRMatrixCommPkg(A);
   }

   num_sends  = jxf_ParCSRCommPkgNumSends(comm_pkg);
   size       = jxf_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * blk_size;
   y_buf_data = jxf_CTAlloc(JXF_Real, size);

   y_tmp_data   = jxf_VectorData(y_tmp);
   y_local_data = jxf_VectorData(y_local);

   /*---------------------------------------------------------------------
    * Compute y_tmp = alpha * offd^T * x_local
    *--------------------------------------------------------------------*/
   if (num_cols_offd) {
       jxf_BSRMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);
   }

   /*---------------------------------------------------------------------
    * Create communication handle for receiving data
    *--------------------------------------------------------------------*/
   comm_handle = jxf_ParBSRCommHandleCreate(2, blk_size, comm_pkg, y_tmp_data, y_buf_data);

   /*---------------------------------------------------------------------
    * Local computation: y = beta*y + alpha*diag^T*x
    *--------------------------------------------------------------------*/
   jxf_BSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);

   /*---------------------------------------------------------------------
    * Wait for communication to complete
    *--------------------------------------------------------------------*/
   jxf_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /*---------------------------------------------------------------------
    * Add received data to local vector
    *--------------------------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++) {
       start  = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
       finish = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);

       for (j = start; j < finish; j++) {
           elem = jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j) * blk_size;
           for (k = 0; k < blk_size; k++) {
               y_local_data[elem++] += y_buf_data[index++];
           }
       }
   }

   /*---------------------------------------------------------------------
    * Clean up
    *--------------------------------------------------------------------*/
   jxf_TFree(y_buf_data);
   jxf_SeqVectorDestroy(y_tmp);
   y_tmp = NULL;

   if (jxf__global_mvcpu_flag) jxf_total_elapsed_time_matvec += (jxf_time_getWallclockSeconds() - wall_time);

   return ierr;
}

