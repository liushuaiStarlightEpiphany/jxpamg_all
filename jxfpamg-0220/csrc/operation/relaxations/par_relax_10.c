//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax_10.c
 *  Date: 2021/06/29
 */ 

#include "jxf_mv.h"

/*!
 * \fn JXF_Int jxf_PAMGRelax10
 * \brief Direct Solved by PARDISO directly
 * \date 2011/09/03
 */
JXF_Int  
jxf_PAMGRelax10( jxf_ParCSRMatrix *par_matrix,
               jxf_ParVector    *par_rhs,
               JXF_Int             *cf_marker,
               JXF_Int              relax_points,
               JXF_Real           relax_weight,
               JXF_Real           omega,
               jxf_ParVector    *par_app,
               jxf_ParVector    *Vtemp )
{
   MPI_Comm         comm         = jxf_ParCSRMatrixComm(par_matrix);
   jxf_CSRMatrix    *A_diag       = jxf_ParCSRMatrixDiag(par_matrix);
   jxf_ParCSRCommPkg    *comm_pkg = jxf_ParCSRMatrixCommPkg(par_matrix);

   JXF_Int             n_global = jxf_ParCSRMatrixGlobalNumRows(par_matrix);
   JXF_Int             n        = jxf_CSRMatrixNumRows(A_diag);
   JXF_Int             first_index   = jxf_ParVectorFirstIndex(par_app);
   
   jxf_Vector      *u_local = jxf_ParVectorLocalVector(par_app);
   JXF_Real         *u_data  = jxf_VectorData(u_local);

   jxf_CSRMatrix    *A_CSR;
   JXF_Int             *A_CSR_i;   
   JXF_Int             *A_CSR_j;
   JXF_Real          *A_CSR_data;
   
   jxf_Vector       *f_vector;
   JXF_Real          *f_vector_data;

   JXF_Int             i;
   JXF_Int             k;
   JXF_Int             column;
   JXF_Int             relax_error = 0;
   JXF_Int             num_procs, my_id;


   jxf_MPI_Comm_size(comm, &num_procs);  
   jxf_MPI_Comm_rank(comm, &my_id);  

  /*-------------------------------------------------------------------------
   * added by peghoty, if the comm_pkg of par_matrix is not created  
   * previously, something will be wrong when the "relax" function  
   * is called on multi-processors occasions.  2009/07/24
   *----------------------------------------------------------------------- */
   
   if (!comm_pkg)
   {
      jxf_MatvecCommPkgCreate(par_matrix);
      comm_pkg = jxf_ParCSRMatrixCommPkg(par_matrix); 
   }

   //------------------------------------------------------------------------------//
   //                   Direct solve: use PARDISO                                  //
   //------------------------------------------------------------------------------//

   /*-----------------------------------------------------------------
   *  Generate CSR matrix from ParCSRMatrix par_matrix
   *-----------------------------------------------------------------*/
#ifdef JXF_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = jxf_ParCSRMatrixToCSRMatrixAll(par_matrix);
         f_vector = jxf_ParVectorToVectorAll(par_rhs);
	if (n)
	{
#else
	if (n)
	{
	   A_CSR = jxf_ParCSRMatrixToCSRMatrixAll(par_matrix);
	   f_vector = jxf_ParVectorToVectorAll(par_rhs);
#endif

      // PARDISO
      // printf("jxf_qsort1, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
      for ( i = 0; i < A_CSR->num_rows; i++)
      {
         k = A_CSR->i[i];
         jxf_qsort1( &A_CSR->j[k], &A_CSR->data[k], 0, A_CSR->i[i+1] - A_CSR->i[i] - 1 );
      }
      // printf("jxf_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
      jxf_pardiso(A_CSR, f_vector, u_local);
      // printf("jxf_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

 	   // free 
      jxf_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      jxf_SeqVectorDestroy(f_vector);
      f_vector = NULL;
      
   }
#ifdef JXF_NO_GLOBAL_PARTITION
   else
   {
      jxf_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      jxf_SeqVectorDestroy(f_vector);
      f_vector = NULL;
   }
#endif

   return(relax_error); 
}
