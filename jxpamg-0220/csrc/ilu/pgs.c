//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  pgs.c
 *  Date: 2024/07/09
 *  zlj
 */ 

#include "jx_pgs.h"


/*!
 * \fn JX_Int JX_PGSSetup
 * \date 2024/07/09
 */
JX_Int 
JX_PGSSetup( JX_Solver        solver, 
                   JX_ParCSRMatrix  par_matrix )
{
   return( jx_PGSSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_PGSSetup
 * \brief Setup phase of PGS preconditioner.
 * \date 2024/07/09
 */
JX_Int
jx_PGSSetup( void             *solver, 
                   jx_ParCSRMatrix  *par_matrix )
{
   return 0;
}

/*!
 * \fn JX_Int JX_PGSPrecond
 * \brief Solve phase of PGS preconditioner.
 * \date 2024/07/09
 */
JX_Int 
JX_PGSPrecond( JX_Solver       solver,
                     JX_ParCSRMatrix par_matrix,
                     JX_ParVector    par_rhs,
                     JX_ParVector    par_app  )
{
   return( jx_PGSPrecond( (void *) solver,
                                (jx_ParCSRMatrix *) par_matrix,
                                (jx_ParVector *) par_rhs,
                                (jx_ParVector *) par_app ) );
}

/*!
 * \fn JX_Int jx_PGSPrecond
 * \brief PGS preconditioner.
 * \param solver pointer to NULL
 * \param par_matrix pointer to the coefficient matrix 
 * \param par_rhs pointer to the right hand side vector
 * \param par_app pointer to the approxamation vector
 * \date 2024/07/09
 */
JX_Int
jx_PGSPrecond( void            *solver,
                     jx_ParCSRMatrix *par_matrix,
                     jx_ParVector    *par_rhs,
                     jx_ParVector    *par_app  )
{
   MPI_Comm         comm         = jx_ParCSRMatrixComm(par_matrix);
   jx_PGS_data  *pgs_data = solver;

   int myid;
   jx_MPI_Comm_rank(comm, &myid);  
   JX_Int times = (pgs_data)->times;       
   JX_Int cycle_count = 0;
//    if (myid == 0) printf("PGS times=%d, %s, %s, %d\n",times, __FILE__, __FUNCTION__, __LINE__);

   while (cycle_count < times)
   {
      jx_PGSTimes(solver, par_matrix, par_rhs, par_app); 
      cycle_count ++;
   }
   
   return 0; 
}


/*!
 * \fn JX_Int jx_PGSTimes
 * \brief PGS preconditioner.
 * \param solver pointer to NULL
 * \param par_matrix pointer to the coefficient matrix 
 * \param par_rhs pointer to the right hand side vector
 * \param par_app pointer to the approxamation vector
 * \date 2024/07/09
 */
JX_Int
jx_PGSTimes( void            *solver,
                     jx_ParCSRMatrix *par_matrix,
                     jx_ParVector    *par_rhs,
                     jx_ParVector    *par_app  )
{
   MPI_Comm         comm         = jx_ParCSRMatrixComm(par_matrix);
   jx_CSRMatrix    *A_diag       = jx_ParCSRMatrixDiag(par_matrix);
   JX_Real          *A_diag_data  = jx_CSRMatrixData(A_diag);
   JX_Int             *A_diag_i     = jx_CSRMatrixI(A_diag);
   JX_Int             *A_diag_j     = jx_CSRMatrixJ(A_diag);
   jx_CSRMatrix    *A_offd       = jx_ParCSRMatrixOffd(par_matrix);
   JX_Int             *A_offd_i     = jx_CSRMatrixI(A_offd);
   JX_Real          *A_offd_data  = jx_CSRMatrixData(A_offd);
   JX_Int             *A_offd_j     = jx_CSRMatrixJ(A_offd);
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);
   jx_ParCSRCommHandle *comm_handle = NULL;

   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   JX_Int             num_cols_offd = jx_CSRMatrixNumCols(A_offd);
   
   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);

   jx_Vector      *f_local = jx_ParVectorLocalVector(par_rhs);
   JX_Real         *f_data  = jx_VectorData(f_local);

   JX_Real 	  *Vext_data  = NULL;
   JX_Real 	  *v_buf_data = NULL;
   JX_Real 	  *tmp_data   = NULL;

   JX_Int             i, j, ierr = 0;
   JX_Int             ii, jj;
   JX_Int             ns, ne, size, rest;
   JX_Int             relax_error = 0;
   JX_Int             num_sends = 0;
   JX_Int             index, start;
   JX_Int             num_procs, num_threads, myid;

   JX_Real          zero = 0.0;
   JX_Real          res, res0, res2;
   JX_Real          prod;



   jx_MPI_Comm_size(comm, &num_procs);  
   jx_MPI_Comm_rank(comm, &myid);  
   num_threads = jx_NumThreads();

   // if (myid == 0) jx_printf("\n >>> jx_Pre_PGS end \n\n");
//    if (myid == 0) printf("jx_Pre_PGS start, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
   
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_matrix);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix); 
   }

      //------------------------------------------------------------------------------//
      //  Hybrid: Jacobi off-processor, Symm. Gauss-Seidel/ SSOR on-processor         //
      //  with outer relaxation parameter                                             //
      //------------------------------------------------------------------------------//
        
        /*-----------------------------------------------------------------
         * Copy current approximation into temporary vector.
         *-----------------------------------------------------------------*/
         if (num_procs > 1)
         {
            num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
            v_buf_data = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
            Vext_data = jx_CTAlloc(JX_Real, num_cols_offd);
        
            if (num_cols_offd)
            {
               A_offd_j = jx_CSRMatrixJ(A_offd);
               A_offd_data = jx_CSRMatrixData(A_offd);
            }
 
            index = 0;
            for (i = 0; i < num_sends; i ++)
            {
               start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j ++)
               {
                  v_buf_data[index ++] = u_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
               }
            }
 
            comm_handle = jx_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);

            jx_ParCSRCommHandleDestroy(comm_handle);
            comm_handle = NULL;
	 }

//    if (myid == 0) printf("jx_Pre_PGS Relax start, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/

	       if (num_threads > 1)
	       {
	          tmp_data = jx_CTAlloc(JX_Real, n);
#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
	          for (i = 0; i < n; i ++)
	          {
	             tmp_data[i] = u_data[i];
	          }
#define JX_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../../include/jx_smp_forloop.h"
	          for (j = 0; j < num_threads; j ++)
	          {
	             size = n/num_threads;
	             rest = n - size*num_threads;
	             if (j < rest)
	             {
	                ns = j*size+j;
	                ne = (j+1)*size+j+1;
	             }
	             else
	             {
	                ns = j*size+rest;
	                ne = (j+1)*size+rest;
	             }
	            
	             for (i = ns; i < ne; i ++)  /* interior points first */
	             {

	               /*-----------------------------------------------------------
	                * If diagonal is nonzero, relax point i; otherwise, skip it.
	                *-----------------------------------------------------------*/
             
	                if ( A_diag_data[A_diag_i[i]] != zero)
	                {
	                   res = f_data[i];
	                   for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
	                   {
	                      ii = A_diag_j[jj];
	                      if (ii >= ns && ii < ne)
	                      {
	                         res -= A_diag_data[jj] * u_data[ii];
	                      }
	                      else
	                      {
	                         res -= A_diag_data[jj] * tmp_data[ii];
	                      }
	                   }
	                  
	                   for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
	                   {
	                      ii = A_offd_j[jj];
	                      res -= A_offd_data[jj] * Vext_data[ii];
	                   }
	                   u_data[i] = res / A_diag_data[A_diag_i[i]];
	                }
	             }
	            
	             for (i = ne-1; i > ns-1; i --)  /* interior points first */
	             {

	               /*-----------------------------------------------------------
	                * If diagonal is nonzero, relax point i; otherwise, skip it.
	                *-----------------------------------------------------------*/
             
	                if ( A_diag_data[A_diag_i[i]] != zero)
	                {
	                   res = f_data[i];
	                   for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
	                   {
	                      ii = A_diag_j[jj];
	                      if (ii >= ns && ii < ne)
	                      {
	                         res -= A_diag_data[jj] * u_data[ii];
	                      }
	                      else
	                      {
	                         res -= A_diag_data[jj] * tmp_data[ii];
	                      }
	                   }
	                  
	                   for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
	                   {
	                      ii = A_offd_j[jj];
	                      res -= A_offd_data[jj] * Vext_data[ii];
	                   }
	                   u_data[i] = res / A_diag_data[A_diag_i[i]];
	                }
	             }
	          }
	          jx_TFree(tmp_data);
	       }
	       else
	       {
            // if (myid == 0) printf("jx_Pre_PGS Relax nt1 start, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

            // if (myid == 0) printf("jx_Pre_PGS Relax nt1 forward start, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

	          for (i = 0; i < n; i ++)  /* interior points first */
	          {

	            /*-----------------------------------------------------------
	             * If diagonal is nonzero, relax point i; otherwise, skip it.
	             *-----------------------------------------------------------*/
             
	             if ( A_diag_data[A_diag_i[i]] != zero)
	             {
	                res = f_data[i];
	                for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
	                {
	                   ii = A_diag_j[jj];
	                   res -= A_diag_data[jj] * u_data[ii];
	                }
	                for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
	                {
	                   ii = A_offd_j[jj];
	                   res -= A_offd_data[jj] * Vext_data[ii];
	                }
	                u_data[i] = res / A_diag_data[A_diag_i[i]];
	             }
	          }

            // if (myid == 0) printf("jx_Pre_PGS Relax nt1 forward end, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
            // if (myid == 0) printf("jx_Pre_PGS Relax nt1 backward start, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
	         
	          for (i = n-1; i > -1; i --)	/* interior points first */
	          {

	            /*-----------------------------------------------------------
	             * If diagonal is nonzero, relax point i; otherwise, skip it.
	             *-----------------------------------------------------------*/
             
	             if ( A_diag_data[A_diag_i[i]] != zero)
	             {
	                res = f_data[i];
	                for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
	                {
	                   ii = A_diag_j[jj];
	                   res -= A_diag_data[jj] * u_data[ii];
	                }
	                for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
	                {
	                   ii = A_offd_j[jj];
	                   res -= A_offd_data[jj] * Vext_data[ii];
	                }
	                u_data[i] = res / A_diag_data[A_diag_i[i]];
	             }
	          }
            // if (myid == 0) printf("jx_Pre_PGS Relax nt1 backward end, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

	       }
//    if (myid == 0) printf("jx_Pre_PGS Relax end, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

         if (num_procs > 1)
         {
            jx_TFree(Vext_data);
            jx_TFree(v_buf_data);
         }

   return ierr;
}


JX_Int
JX_PGSSetTimes( JX_Solver solver, JX_Int  pgs_times )
{
   return( jx_PGSSetTimes( (void *) solver, pgs_times ) );
}


JX_Int
jx_PGSSetTimes( void *data, JX_Int pgs_times )
{
   jx_PGS_data  *pgs_data = data;
 
   if (!pgs_data)
   {
      jx_printf("Warning! PGS object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (pgs_times < 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   (pgs_data)->times = pgs_times;

   return jx_error_flag;
}