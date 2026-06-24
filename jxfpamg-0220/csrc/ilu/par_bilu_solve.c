#include "jxf_parbilu.h"
#include "jxf_krylov.h"
#include "jxf_parbsr_mv.h"
#include <float.h>
#define HYPRE_USING_CUDA 1
#define HYPRE_USING_GPU 1

// extern JXF_Real Tri_Total_time;
JXF_Real Tri_Total_time;
/*--------------------------------------------------------------------
 * jxf_ILUSolve
 *
 * TODO (VPM): Change variable names of F_array and U_array
 *--------------------------------------------------------------------*/

JXF_Int
jxf_BILUSolve( void               *ilu_vdata,
                jxf_ParBSRMatrix *A,
                jxf_ParVector    *f,
                jxf_ParVector    *u )
{
   MPI_Comm              comm               = jxf_ParBSRMatrixComm(A);
   jxf_ParBILUData     *ilu_data           = (jxf_ParBILUData*) ilu_vdata;

   /* Matrices */
   jxf_ParBSRMatrix   *matmL              = jxf_ParILUDataMatLModified(ilu_data);
   jxf_ParBSRMatrix   *matmU              = jxf_ParILUDataMatUModified(ilu_data);
   jxf_ParBSRMatrix   *matA               = jxf_ParILUDataMatA(ilu_data);
   jxf_ParBSRMatrix   *matL               = jxf_ParILUDataMatL(ilu_data);
   jxf_ParBSRMatrix   *matU               = jxf_ParILUDataMatU(ilu_data);
   jxf_ParBSRMatrix   *matS               = jxf_ParILUDataMatS(ilu_data);
   JXF_Real           *matD               = jxf_ParILUDataMatD(ilu_data);
   JXF_Real           *matmD              = jxf_ParILUDataMatDModified(ilu_data);

   /* Vectors */
   JXF_Int             ilu_type           = jxf_ParILUDataIluType(ilu_data);
   JXF_Int            *perm               = jxf_ParILUDataPerm(ilu_data);
   JXF_Int            *qperm              = jxf_ParILUDataQPerm(ilu_data);
   jxf_ParVector      *F_array            = jxf_ParILUDataF(ilu_data);
   jxf_ParVector      *D_array            = jxf_ParILUDataD(ilu_data);
   jxf_ParVector      *U_array            = jxf_ParILUDataU(ilu_data);

   /* Solver settings */
   JXF_Real            tol                = jxf_ParILUDataTol(ilu_data);
   JXF_Int             logging            = jxf_ParILUDataLogging(ilu_data);
   JXF_Int             print_level        = jxf_ParILUDataPrintLevel(ilu_data);
   JXF_Int             max_iter           = jxf_ParILUDataMaxIter(ilu_data);
   JXF_Int             tri_solve          = jxf_ParILUDataTriSolve(ilu_data);
   JXF_Int             lower_jacobi_iters = jxf_ParILUDataLowerJacobiIters(ilu_data);
   JXF_Int             upper_jacobi_iters = jxf_ParILUDataUpperJacobiIters(ilu_data);
   JXF_Int             IR_iters           = jxf_ParILUDataIRIters(ilu_data);
   JXF_Real           *norms              = jxf_ParILUDataRelResNorms(ilu_data);
   jxf_ParVector      *Ftemp              = jxf_ParILUDataFTemp(ilu_data);
   jxf_ParVector      *Utemp              = jxf_ParILUDataUTemp(ilu_data);
   jxf_ParVector      *Xtemp              = jxf_ParILUDataXTemp(ilu_data);
   jxf_ParVector      *Ytemp              = jxf_ParILUDataYTemp(ilu_data);
   JXF_Real           *fext               = jxf_ParILUDataFExt(ilu_data);
   JXF_Real           *uext               = jxf_ParILUDataUExt(ilu_data);
   jxf_ParVector      *residual           = NULL;
   JXF_Real            alpha              = -1.0;
   JXF_Real            beta               = 1.0;
   JXF_Real            conv_factor        = 0.0;
   JXF_Real            resnorm            = 1.0;
   JXF_Real            init_resnorm       = 0.0;
   JXF_Real            rel_resnorm;
   JXF_Real            rhs_norm           = 0.0;
   JXF_Real            old_resnorm;
   JXF_Real            ieee_check         = 0.0;
   JXF_Real            operat_cmplxty     = jxf_ParILUDataOperatorComplexity(ilu_data);
   JXF_Int             Solve_err_flag;
   JXF_Int             iter, num_procs, my_id;
   JXF_Real             starttime = 0.0, endtime= 0.0;

   /* problem size */
   JXF_Int             n                  = jxf_ParBSRMatrixNumRows(A);
   JXF_Int             nLU                = n;              //jxf_ParILUDataNLU(ilu_data);
   JXF_Int            *u_end              = jxf_ParILUDataUEnd(ilu_data);

   /* Schur system solve */
//    JXF_Solver          schur_solver       = jxf_ParILUDataSchurSolver(ilu_data);
//    JXF_Solver          schur_precond      = jxf_ParILUDataSchurPrecond(ilu_data);
   jxf_ParVector      *rhs                = jxf_ParILUDataRhs(ilu_data);
   jxf_ParVector      *x                  = jxf_ParILUDataX(ilu_data);


   if (logging > 1)
   {
      residual = jxf_ParILUDataResidual(ilu_data);
   }

   jxf_ParILUDataNumIterations(ilu_data) = 0;

   jxf_MPI_Comm_size(comm, &num_procs);
   jxf_MPI_Comm_rank(comm, &my_id);

   double t_bilu_start = MPI_Wtime(), t_section;


   Solve_err_flag = 0;

   /*-----------------------------------------------------------------------
     *     write some initial info
     *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1 && tol > 0.)
   {
      jxf_printf("\n\n ILU SOLVER SOLUTION INFO:\n");
   }

   /*-----------------------------------------------------------------------
    *    Compute initial residual and print
    *-----------------------------------------------------------------------*/

   if (print_level > 1 || logging > 1 || tol > 0.)
   {
      if (logging > 1)
      {
         jxf_ParVectorCopy(f, residual);
         if (tol > 0.0)
         {
            jxf_ParBSRMatrixMatvec(alpha, A, u, beta, residual);
         }
         resnorm = sqrt(jxf_ParVectorInnerProd(residual, residual));
      }
      else
      {
         jxf_ParVectorCopy(f, Ftemp);
         if (tol > 0.0)
         {
            jxf_ParBSRMatrixMatvec(alpha, A, u, beta, Ftemp);
         }
         resnorm = sqrt(jxf_ParVectorInnerProd(Ftemp, Ftemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resnorm != 0.)
      {
         ieee_check = resnorm / resnorm; /* INF -> NaN conversion */
      }
      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (print_level > 0)
         {
            jxf_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            jxf_printf("ERROR -- jxf_ILUSolve: INFs and/or NaNs detected in input.\n");
            jxf_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            jxf_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         jxf_error(JXF_ERROR_GENERIC);

         return jxf_error_flag;
      }

      init_resnorm = resnorm;
      rhs_norm = sqrt(jxf_ParVectorInnerProd(f, f));
      if (rhs_norm > DBL_EPSILON) //JXF_REAL_EPSILON
      {
         rel_resnorm = init_resnorm / rhs_norm;
      }
      else
      {
         /* rhs is zero, return a zero solution */
         jxf_ParVectorSetConstantValues(U_array, 0.0);
         if (logging > 0)
         {
            rel_resnorm = 0.0;
            jxf_ParILUDataFinalRelResidualNorm(ilu_data) = rel_resnorm;
         }

         return jxf_error_flag;
      }
   }
   else
   {
      rel_resnorm = 1.;
   }

   if (my_id == 0 && print_level > 1)
   {
      jxf_printf("                                            relative\n");
      jxf_printf("               residual        factor       residual\n");
      jxf_printf("               --------        ------       --------\n");
      jxf_printf("    Initial    %e                 %e\n", init_resnorm,
                   rel_resnorm);
   }

   matA    = A;
   U_array = u;
   F_array = f;

   JXF_Int **L_levels, **U_levels;
   JXF_Int *L_level_sizes, *U_level_sizes;
   JXF_Int L_num_levels, U_num_levels;
   // JXF_Int *U_perm, U_iperm;
   // JXF_Int *L_perm, L_iperm;

   // U_perm = jxf_ParILUDataU_perm(ilu_data);
   // U_iperm = jxf_ParILUDataU_iperm(ilu_data);

   // L_perm = jxf_ParILUDataL_perm(ilu_data);
   // L_iperm = jxf_ParILUDataL_iperm(ilu_data);

   L_levels = jxf_ParILUDataL_levels(ilu_data);
   L_level_sizes = jxf_ParILUDataL_level_sizes(ilu_data);
   L_num_levels = jxf_ParILUDataL_num_levels(ilu_data);

   U_levels = jxf_ParILUDataU_levels(ilu_data);
   U_level_sizes = jxf_ParILUDataU_level_sizes(ilu_data);
   U_num_levels = jxf_ParILUDataL_num_levels(ilu_data);
   /************** Main Solver Loop - always do 1 iteration ************/
   iter = 0;
   while ((rel_resnorm >= tol || iter < 1) &&
          (iter < max_iter))
   {
      switch (tri_solve) 
      {
         // case 1:
         //    // starttime = jxf_MPI_Wtime();
         //    jxf_ILUSolveLU(matA, F_array, U_array, perm, n,
         //                   matL, matD, matU, Utemp, Ftemp);
         //    // jxf_ILUSolveLUIter(matA, F_array, U_array, perm, n,
         //    //       matL, matD, matU, Utemp, Ftemp,
         //    //       lower_jacobi_iters, upper_jacobi_iters);
         //    // endtime = jxf_MPI_Wtime();
         //    // Tri_Total_time +=  endtime - starttime;    
         //    break;
         // case 2:
         //    starttime = jxf_MPI_Wtime();
         //    jxf_TriangularSolveLevel(matA, F_array, U_array, n,
         //                            matL, matD, matU, Utemp, Ftemp,
         //                            ilu_data->nlevL, ilu_data->jlevL, ilu_data->ilevL,
         //                            ilu_data->nlevU, ilu_data->jlevU, ilu_data->ilevU);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;   
         //    break;

         // case 3:  //多色GS-IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_MultiColorTriangularSolve(matA, F_array, U_array, n,
         //                            matL, matD, matU, Ftemp,Utemp, 
         //                           ilu_data->nlevL, ilu_data->jlevL, ilu_data->ilevL,
         //                            ilu_data->nlevU, ilu_data->jlevU, ilu_data->ilevU,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;
         // break;

         // case 4:  //直接法-IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_MultiGS_TriangularSolve(matA, F_array, U_array, n,
         //                            matL, matD, matU, Ftemp,Utemp,IR_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;
         // break;

         // case 6:   //jacobi
         //    starttime = jxf_MPI_Wtime();
         //    jxf_ILUSolveLUIter(matA, F_array, U_array, perm, n,
         //                      matL, matD, matU, Utemp, Ftemp,
         //                      lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                     
         // break;

         // case 7:  //层次调度 + IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_MultiL_TriangularSolve(matA, F_array, U_array, n,
         //                            matL, matD, matU, Ftemp,Utemp, 
         //                           ilu_data->nlevL, ilu_data->jlevL, ilu_data->ilevL,
         //                            ilu_data->nlevU, ilu_data->jlevU, ilu_data->ilevU,IR_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                           
         // break;

         // case 8:  //8GS + IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_MultiColorTriangularSolve(matA, F_array, U_array, n,
         //                            matL, matD, matU, Ftemp,Utemp, 
         //                           ilu_data->nlevL, ilu_data->jlevL, ilu_data->ilevL,
         //                            ilu_data->nlevU, ilu_data->jlevU, ilu_data->ilevU,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                           
         // break;

         // case 9:   //jacobi-IR LU整体做循环
         //    starttime = jxf_MPI_Wtime();
         //    jxf_ILUSolveLUIter_v2(matA, F_array, U_array, perm, n,
         //                      matL, matD, matU, Utemp, Ftemp,
         //                      lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                     
         // break;

         // case 10:  //函数-jacobi-IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_JacobiTriangularSolve_spmv(matA, F_array, U_array, perm, n,matL, matD,
         //                       matU, Utemp, Ftemp,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                       
         // break;

         // case 16:
         //    starttime = jxf_MPI_Wtime();
         //    jxf_JacobiTriangularSolve_spmv1(matA, F_array, U_array, perm, n, matL, matD, D_array,
         //                       matU, Utemp, Ftemp,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                       
         // break;

         // // case 10:  //GS-jacobi-IR
         // //    starttime = jxf_MPI_Wtime();
         // //    jxf_gsJacobiTriangularSolve(matA, F_array, U_array, perm, n,matL, matD,
         // //                       matU, Utemp, Ftemp,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         // //    endtime = jxf_MPI_Wtime();
         // //    Tri_Total_time +=  endtime - starttime;                                       
         // // break;

         // case 11:  //JGS-IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_JGS_TriangularSolve(matA, F_array, U_array, perm, n,matL, matD,
         //                       matU, Utemp, Ftemp,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                       
         // break;

         // case 12:  //多色GS-IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_MultiColorTriangularSolve1(matA, F_array, U_array, n,matL, matD, matU, Ftemp,Utemp, 
         //                               ilu_data->nlevL, ilu_data->jlevL, ilu_data->ilevL,ilu_data->nlevU, 
         //                               ilu_data->jlevU, ilu_data->ilevU,IR_iters,lower_jacobi_iters, upper_jacobi_iters,
         //                               ilu_data->L_perm,ilu_data->L_iperm, ilu_data->U_perm,ilu_data->U_iperm);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;
         // break;

         // case 13:  //层次调度 + IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_MultiL_TriangularSolve1(matA, F_array, U_array, n,
         //                            matL, matD, matU, Ftemp,Utemp, 
         //                           ilu_data->nlevL, ilu_data->jlevL, ilu_data->ilevL,
         //                            ilu_data->nlevU, ilu_data->jlevU, ilu_data->ilevU,IR_iters,
         //                             ilu_data->L_perm,ilu_data->L_iperm, ilu_data->U_perm,ilu_data->U_iperm);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                           
         // break;
         
         // case 14:  //完全分布式 + IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_parJacobi_TriangularSolve(matA, F_array, U_array,n,matL, D_array,
         //                        matU, Utemp, Ftemp,Xtemp,Ytemp,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                           
         // break;

          case 14:  //完全分布式 + IR
             starttime = jxf_MPI_Wtime();
             jxf_par_Block_Jacobi_TriangularSolve_B(matA, F_array, U_array,n,matL, D_array,
                                 matU, Utemp, Ftemp,Xtemp,Ytemp,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
             endtime = jxf_MPI_Wtime();
             Tri_Total_time +=  endtime - starttime;
         break;

         // case 5:  //jacobi-IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_JacobiTriangularSolve(matA, F_array, U_array, perm, n,matL, matD,
         //                       matU, Utemp, Ftemp,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                       
         // break;

         // case 20:  //jacobi-IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_JacobiTriangularSolve22(matA, F_array, U_array, perm, n,matL, matD,
         //                       matU, Utemp, Ftemp,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                       
         // break;

         // case 21:  //jacobi-IR
         //    starttime = jxf_MPI_Wtime();
         //    jxf_JacobiTriangularSolve33(matA, F_array, U_array, perm, n,matL, matD,
         //                       matU, Utemp, Ftemp,IR_iters,lower_jacobi_iters, upper_jacobi_iters);
         //    endtime = jxf_MPI_Wtime();
         //    Tri_Total_time +=  endtime - starttime;                                       
         // break;
      }
      /*---------------------------------------------------------------
       *    Compute residual and residual norm
       *----------------------------------------------------------------*/

      if (print_level > 1 || logging > 1 || tol > 0.)
      {
         old_resnorm = resnorm;

         if (logging > 1)
         {
            jxf_ParVectorCopy(F_array, residual);
            jxf_ParBSRMatrixMatvec(alpha, matA, U_array, beta, residual);
            resnorm = sqrt(jxf_ParVectorInnerProd(residual, residual));
         }
         else
         {
            jxf_ParVectorCopy(F_array, Ftemp);
            jxf_ParBSRMatrixMatvec(alpha, matA, U_array, beta, Ftemp);
            resnorm = sqrt(jxf_ParVectorInnerProd(Ftemp, Ftemp));
         }

         if (old_resnorm)
         {
            conv_factor = resnorm / old_resnorm;
         }
         else
         {
            conv_factor = resnorm;
         }

         if (rhs_norm > DBL_EPSILON)
         {
            rel_resnorm = resnorm / rhs_norm;
         }
         else
         {
            rel_resnorm = resnorm;
         }

         norms[iter] = rel_resnorm;
      }

      ++iter;
      jxf_ParILUDataNumIterations(ilu_data) = iter;
      jxf_ParILUDataFinalRelResidualNorm(ilu_data) = rel_resnorm;

      if (my_id == 0 && print_level > 1)
      {
         jxf_printf("    BILUSolve %2d   %e    %f     %e \n", iter,
                      resnorm, conv_factor, rel_resnorm);
      }
   }

   /* check convergence within max_iter */
   if (iter == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      // jxf_error(JXF_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Print closing statistics
    *    Add operator and grid complexity stats
    *-----------------------------------------------------------------------*/

   if (iter > 0 && init_resnorm)
   {
      conv_factor = pow((resnorm / init_resnorm), (1.0 / (JXF_Real) iter));
   }
   else
   {
      conv_factor = 1.;
   }

   if (print_level > 1)
   {
      /*** compute operator and grid complexity (fill factor) here ?? ***/
      if (my_id == 0)
      {
         if (Solve_err_flag == 1)
         {
            jxf_printf("\n\n==============================================");
            jxf_printf("\n NOTE: Convergence tolerance was not achieved\n");
            jxf_printf("      within the allowed %d iterations\n", max_iter);
            jxf_printf("==============================================");
         }
         jxf_printf("\n\n Average Convergence Factor = %f \n", conv_factor);
         jxf_printf("                operator = %f\n", operat_cmplxty);
      }
   }

   return jxf_error_flag;
}

/*--------------------------------------------------------------------
 * jxf_ILUSolveLUIter
 *
 * Iterative incomplete LU solve
 *
 * L, D and U factors only have local scope (no off-diag terms)
 *  so apart from the residual calculation (which uses A), the solves
 *  with the L and U factors are local.
 *
 * Note: perm contains the permutation of indexes corresponding to
 * user-prescribed reordering strategy. In the block Jacobi case, perm
 * may be NULL if no reordering is done (for performance, (perm == NULL)
 * assumes identity mapping of indexes). Hence we need to check the local
 * solves for this case and avoid segfaults. - DOK
 *--------------------------------------------------------------------*/

// JXF_Int
// jxf_ILUSolveLUIter(jxf_ParCSRMatrix *A,
//                      jxf_ParVector    *f,
//                      jxf_ParVector    *u,
//                      JXF_Int          *perm,
//                      JXF_Int           nLU,
//                      jxf_ParCSRMatrix *L,
//                      JXF_Real         *D,
//                      jxf_ParCSRMatrix *U,
//                      jxf_ParVector    *ftemp,
//                      jxf_ParVector    *utemp,
//                      JXF_Int           lower_jacobi_iters,
//                      JXF_Int           upper_jacobi_iters)
// {
//    /* Data objects for L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* Local variables */
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Real       sum;
//    JXF_Int        i, j, k1, k2, kk;

//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* L solve - Forward solve */
//    /* copy rhs to account for diagonal of L (which is identity) */

//    /* Initialize iteration to 0 */
//    if (perm)
//    {
//       for ( i = 0; i < nLU; i++ )
//       {
//          utemp_data[perm[i]] = 0.0;
//       }
//    }
//    else
//    {
//       for ( i = 0; i < nLU; i++ )
//       {
//          utemp_data[i] = 0.0;
//       }
//    }
//    /* Jacobi iteration loop */
//    for ( kk = 0; kk < lower_jacobi_iters; kk++ )
//    {
//       /* u^{k+1} = f - Lu^k */

//       /* Do a SpMV with L and save the results in xtemp */
//       if (perm)
//       {
//          for ( i = nLU - 1; i >= 0; i-- )
//          {
//             sum = 0.0;
//             k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += L_diag_data[j] * utemp_data[perm[L_diag_j[j]]];
//             }
//             utemp_data[perm[i]] = ftemp_data[perm[i]] - sum;
//          }
//       }
//       else
//       {
//          for ( i = nLU - 1; i >= 0; i-- )
//          {
//             sum = 0.0;
//             k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += L_diag_data[j] * utemp_data[L_diag_j[j]];
//             }
//             utemp_data[i] = ftemp_data[i] - sum;
//          }
//       }
//    } /* end jacobi loop */

//    /* Initialize iteration to 0 */
//    if (perm)
//    {
//       for ( i = 0; i < nLU; i++ )
//       {
//          ftemp_data[perm[i]] = 0.0;
//       }
//    }
//    else
//    {
//       for ( i = 0; i < nLU; i++ )
//       {
//          ftemp_data[i] = 0.0;
//       }
//    }

//    /* Jacobi iteration loop */
//    for ( kk = 0; kk < upper_jacobi_iters; kk++ )
//    {
//       /* u^{k+1} = f - Uu^k */

//       /* Do a SpMV with U and save the results in xtemp */
//       if (perm)
//       {
//          for ( i = 0; i < nLU; ++i )
//          {
//             sum = 0.0;
//             k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += U_diag_data[j] * ftemp_data[perm[U_diag_j[j]]];
//             }
//             ftemp_data[perm[i]] = D[i] * (utemp_data[perm[i]] - sum);
//          }
//       }
//       else
//       {
//          for ( i = 0; i < nLU; ++i )
//          {
//             sum = 0.0;
//             k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += U_diag_data[j] * ftemp_data[U_diag_j[j]];
//             }
//             ftemp_data[i] = D[i] * (utemp_data[i] - sum);
//          }
//       }
//    } /* end jacobi loop */

//    /* Update solution */
//    jxf_ParVectorAxpy(beta, ftemp, u);

//    return jxf_error_flag;
// }

// JXF_Int
// jxf_ILUSolveLUIter(jxf_ParCSRMatrix *A,
//                      jxf_ParVector    *f,
//                      jxf_ParVector    *u,
//                      JXF_Int          *perm,
//                      JXF_Int           nLU,
//                      jxf_ParCSRMatrix *L,
//                      JXF_Real         *D,
//                      jxf_ParCSRMatrix *U,
//                      jxf_ParVector    *ftemp,
//                      jxf_ParVector    *utemp,
//                      JXF_Int           lower_jacobi_iters,
//                      JXF_Int           upper_jacobi_iters)
// {
//    /* Data objects for L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* Local variables */
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Real       sum;
//    JXF_Int        i, j, k1, k2, kk;

//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* Initialize iteration to 0 */
//    JXF_Real  *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    JXF_Real  *y_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//    #pragma omp parallel for
//    for ( i = 0; i < nLU; i++ )
//    {
//       utemp_data[i] = 0.0;
//    }
//    /* Jacobi iteration loop */
//    for ( kk = 0; kk < lower_jacobi_iters; kk++ )
//    {
//       #pragma omp parallel for private(j, k1, k2, sum)
//       for ( i = nLU - 1; i >= 0; i-- )
//       {
//          sum = 0.0;
//          k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//          for (j = k1; j < k2; j++)
//          {
//             sum += L_diag_data[j] * utemp_data[L_diag_j[j]];
//          }
//          y_new[i] = ftemp_data[i] - sum;
//       }
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) utemp_data[i] = y_new[i];
//    } /* end jacobi loop */

//    #pragma omp parallel for
//    for ( i = 0; i < nLU; i++ )
//    {
//       utemp_data[i] = 0.0;
//    }

//    /* Jacobi iteration loop */
//    for ( kk = 0; kk < upper_jacobi_iters; kk++ )
//    {
//       #pragma omp parallel for private(j, k1, k2, sum)
//       /* u^{k+1} = f - Uu^k */
//       for ( i = 0; i < nLU; ++i )
//       {
//          sum = 0.0;
//          k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//          for (j = k1; j < k2; j++)
//          {
//             sum += U_diag_data[j] * utemp_data[U_diag_j[j]];
//          }
//          u_new[i] = D[i] * (y_new[i] - sum);
//       }
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) utemp_data[i] = u_new[i];
//    } /* end jacobi loop */

//    /* Update solution */
//    jxf_ParVectorAxpy(beta, utemp, u);

//    free(u_new);

//    return jxf_error_flag;
// }

// JXF_Int
// jxf_ILUSolveLUIter_v2(jxf_ParCSRMatrix *A,
//                      jxf_ParVector    *f,
//                      jxf_ParVector    *u,
//                      JXF_Int          *perm,
//                      JXF_Int           nLU,
//                      jxf_ParCSRMatrix *L,
//                      JXF_Real         *D,
//                      jxf_ParCSRMatrix *U,
//                      jxf_ParVector    *ftemp,
//                      jxf_ParVector    *utemp,
//                      JXF_Int           lower_jacobi_iters,
//                      JXF_Int           upper_jacobi_iters)
// {
//    /* Data objects for L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* Local variables */
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Real       sum;
//    JXF_Int        i, j, k1, k2, kk;

//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* Initialize iteration to 0 */
//    JXF_Real  *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    JXF_Real  *ytemp = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    JXF_Real  *y_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//    #pragma omp parallel for
//    for ( i = 0; i < nLU; i++ )
//    {
//       utemp_data[i] = 0.0;
//       ytemp[i] = 0.0;
//    }
//    /* Jacobi iteration loop */
//    for ( kk = 0; kk < lower_jacobi_iters; kk++ )
//    {
//       #pragma omp parallel for private(j, k1, k2, sum)
//       for ( i = nLU - 1; i >= 0; i-- )
//       {
//          sum = 0.0;
//          k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//          for (j = k1; j < k2; j++)
//          {
//             sum += L_diag_data[j] * ytemp[L_diag_j[j]];
//          }
//          y_new[i] = ftemp_data[i] - sum;
//       }

//       #pragma omp parallel for private(j, k1, k2, sum)
//       /* u^{k+1} = f - Uu^k */
//       for ( i = 0; i < nLU; ++i )
//       {
//          sum = 0.0;
//          k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//          for (j = k1; j < k2; j++)
//          {
//             sum += U_diag_data[j] * utemp_data[U_diag_j[j]];
//          }
//          u_new[i] = D[i] * (y_new[i] - sum);
//       }

//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = u_new[i];
//          ytemp[i] = y_new[i];
//       }
//    } /* end jacobi loop */

//    /* Update solution */
//    jxf_ParVectorAxpy(beta, utemp, u);

//    free(u_new);

//    return jxf_error_flag;
// }

// JXF_Int
// jxf_ILUSolveLU(jxf_ParCSRMatrix *A,
//                  jxf_ParVector    *f,
//                  jxf_ParVector    *u,
//                  JXF_Int          *perm,
//                  JXF_Int           nLU,
//                  jxf_ParCSRMatrix *L,
//                  JXF_Real         *D,
//                  jxf_ParCSRMatrix *U,
//                  jxf_ParVector    *ftemp,
//                  jxf_ParVector    *utemp)
// {
//    /* data objects for L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Int        i, j, k1, k2;

//    // char FileNameCoaMat[256];
//    // printf("%d",nLU);
//    // jxf_sprintf(FileNameCoaMat, "L_CSR_%d", 2);
//    // jxf_ParCSRMatrixPrint(L, FileNameCoaMat);
//    // jxf_sprintf(FileNameCoaMat, "U_CSR_%d", 2);
//    // jxf_ParCSRMatrixPrint(U, FileNameCoaMat);
   
//    // FILE *fp;
//    // jxf_sprintf(FileNameCoaMat, "D_%d", 2);
//    // fp = fopen(FileNameCoaMat, "w");
//    // for(j = 0; j < nLU; j++){
//    // jxf_fprintf(fp, "%.14e\n", D[j]);
//    // }
//    // fclose(fp);


//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta,ftemp);

//    // jxf_sprintf(FileNameCoaMat, "F_temp_%d", 2);
//    // fp = fopen(FileNameCoaMat, "w");
//    // for(j = 0; j < nLU; j++){
//    // jxf_fprintf(fp, "%.14e\n", ftemp_data[j]);
//    // }
//    // fclose(fp);

//    /* L solve - Forward solve */
//    /* copy rhs to account for diagonal of L (which is identity) */
//    if (perm)
//    {
//       for (i = 0; i < nLU; i++)
//       {
//          utemp_data[perm[i]] = ftemp_data[perm[i]];
//       }
//    }
//    else
//    {
//       for (i = 0; i < nLU; i++)
//       {
//          utemp_data[i] = ftemp_data[i];
//       }
//    }

//    /* Update with remaining (off-diagonal) entries of L */
//    if (perm)
//    {
//       for ( i = 0; i < nLU; i++ )
//       {
//          k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//          for (j = k1; j < k2; j++)
//          {
//             utemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[L_diag_j[j]]];
//          }
//       }
//    }
//    else
//    {
//       for ( i = 0; i < nLU; i++ )
//       {
//          k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//          for (j = k1; j < k2; j++)
//          {
//             utemp_data[i] -= L_diag_data[j] * utemp_data[L_diag_j[j]];
//          }
//       }
//    }
//    /*-------------------- U solve - Backward substitution */
//    if (perm)
//    {
//       for ( i = nLU - 1; i >= 0; i-- )
//       {
//          /* first update with the remaining (off-diagonal) entries of U */
//          k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//          for (j = k1; j < k2; j++)
//          {
//             utemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[U_diag_j[j]]];
//          }

//          /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
//          utemp_data[perm[i]] *= D[i];
//       }
//    }
//    else
//    {
//       for ( i = nLU - 1; i >= 0; i-- )
//       {
//          /* first update with the remaining (off-diagonal) entries of U */
//          k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//          for (j = k1; j < k2; j++)
//          {
//             utemp_data[i] -= U_diag_data[j] * utemp_data[U_diag_j[j]];
//          }

//          /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
//          utemp_data[i] *= D[i];
//       }
//    }
//    /* Update solution */
//    jxf_ParVectorAxpy(beta, utemp, u);

//    //char FileNameCoaMat[256];
//    // FILE  *fp;
//    // jxf_sprintf(FileNameCoaMat, "rhs_%d", 1);
//    // fp = fopen(FileNameCoaMat, "w");
//    // for(j = 0; j < nLU; j++){
//    // jxf_fprintf(fp, "%.14e\n", utemp_data[j]);
//    // }
//    // fclose(fp);
   

//    return jxf_error_flag;
// }

// JXF_Int jxf_TriangularSolveLevel_v1(jxf_ParCSRMatrix *A,
//                                jxf_ParVector    *f,
//                                jxf_ParVector    *u,
//                                JXF_Int           nLU,
//                                jxf_ParCSRMatrix *L,
//                                JXF_Real         *D,
//                                jxf_ParCSRMatrix *U,
//                                jxf_ParVector    *ftemp,
//                                jxf_ParVector    *utemp,
//                                JXF_Int         **L_levels,      // L 的层次调度
//                                JXF_Int          *L_level_sizes, // L 的层次大小
//                                JXF_Int          L_num_levels,   // L 的层次数
//                                JXF_Int         **U_levels,      // U 的层次调度
//                                JXF_Int          *U_level_sizes, // U 的层次大小
//                                JXF_Int          U_num_levels)   // U 的层次数
// {
//    /* Extract diagonal blocks of L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);

//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Get vector data */
//    jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real   *utemp_data  = jxf_VectorData(utemp_local);

//    jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real   *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* Residual: ftemp = f - A * u */
//    JXF_Real alpha = -1.0;
//    JXF_Real beta  = 1.0;
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* L solve - Forward substitution (L assumed unit diagonal) */
//    //#pragma omp parallel for schedule(static)
//    for (JXF_Int i = 0; i < nLU; i++) {
//       utemp_data[i] = ftemp_data[i];
//    }

//    for (JXF_Int l = 0; l < L_num_levels; l++) {
//       //#pragma omp parallel for schedule(static)
//       for (JXF_Int k = 0; k < L_level_sizes[l]; k++) {
//          JXF_Int row = L_levels[l][k];
//          JXF_Real sum = utemp_data[row];

//          for (JXF_Int jj = L_diag_i[row]; jj < L_diag_i[row + 1]; jj++) {
//             JXF_Int col = L_diag_j[jj];
//             sum -= L_diag_data[jj] * utemp_data[col];
//          }

//          utemp_data[row] = sum;
//       }
//    }

//    /* U solve - Backward substitution */
//    for (JXF_Int l = U_num_levels - 1; l >= 0; l--) {
//       //#pragma omp parallel for schedule(static)
//       for (JXF_Int k = 0; k < U_level_sizes[l]; k++) {
//          JXF_Int row = U_levels[l][k];
//          JXF_Real sum = utemp_data[row];

//          for (JXF_Int jj = U_diag_i[row]; jj < U_diag_i[row + 1]; jj++) {
//             JXF_Int col = U_diag_j[jj];
//             sum -= U_diag_data[jj] * utemp_data[col];
//          }

//          // Diagonal inverse scaling: u[row] = sum / U(row,row)
//          utemp_data[row] = sum * D[row];
//       }
//    }

//    /* Final update: u = u + utemp */
//    jxf_ParVectorAxpy(beta, utemp, u);

//    return jxf_error_flag;
// }


// JXF_Int jxf_TriangularSolveLevel(jxf_ParCSRMatrix *A,
//                               jxf_ParVector    *f,
//                               jxf_ParVector    *u,
//                               JXF_Int           nLU,
//                               jxf_ParCSRMatrix *L,
//                               JXF_Real         *D,
//                               jxf_ParCSRMatrix *U,
//                               jxf_ParVector    *ftemp,
//                               jxf_ParVector    *utemp,
//                               JXF_Int           nlevL,
//                               JXF_Int          *jlevL,
//                               JXF_Int          *ilevL,
//                               JXF_Int           nlevU,
//                               JXF_Int          *jlevU,
//                               JXF_Int          *ilevU)
// {
//    /* 提取 L 和 U 的对角块 */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);

//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* 获取向量数据 */
//    jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real   *utemp_data  = jxf_VectorData(utemp_local);

//    jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real   *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* 计算残差：ftemp = f - A * u */
//    JXF_Real alpha = -1.0;
//    JXF_Real beta  = 1.0;
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    JXF_Real sum;
//    JXF_Int begin_row ,end_row,i ,ii ,k, jj, j;

//    // char FileNameCoaMat[256];
//    // jxf_sprintf(FileNameCoaMat, "L_CSR_%d", 2);
//    // jxf_ParCSRMatrixPrint(L, FileNameCoaMat);
//    // jxf_sprintf(FileNameCoaMat, "U_CSR_%d", 2);
//    // jxf_ParCSRMatrixPrint(U, FileNameCoaMat);
   
//    // FILE *fp;
//    // jxf_sprintf(FileNameCoaMat, "D_%d", 2);
//    // fp = fopen(FileNameCoaMat, "w");
//    // for(j = 0; j < nLU; j++){
//    // jxf_fprintf(fp, "%.14e\n", D[j]);
//    // }
//    // fclose(fp);

//    // printf("Lower Triangular Matrix L:\n");
//    //  printf("Number of levels (nlevL): %d\n", nlevL);
//    //  printf("jlevL (row indices per level): ");
//    //  for (i = 0; i < nLU; i++) printf("%d ", jlevL[i]);
//    //  printf("\n");
//    //  printf("ilevL (level offsets): ");
//    //  for (i = 0; i < nlevL; i++) printf("%d ", ilevL[i]);
//    //  printf("\n\n");

//    /* 初始化 utemp = ftemp */
//    for ( i = 0; i < nLU; i++) {
//       utemp_data[i] = ftemp_data[i];
//    }

//    /* 前向替换：解单位下三角矩阵方程 L * utemp = ftemp */
//    for ( k = 0; k < nlevL-1; k++) {
//  #pragma omp parallel for private(i, ii, begin_row, end_row, j, jj, sum)
//       for ( ii = ilevL[k]; ii < ilevL[k+1]; ii++) {
//          i = jlevL[ii];
//          sum = utemp_data[i];
//          begin_row = L_diag_i[i];
//          end_row = L_diag_i[i+1] ;

//          for (j = begin_row; j < end_row; j++) 
//          {
//             jj = L_diag_j[j]; 
//             sum -= L_diag_data[j] * utemp_data[jj];
//          }
//          utemp_data[i] = sum;
//       }
//       //  #pragma omp barrier
//    }

//    /* 后向替换：解上三角矩阵方程 U * u_temp = D⁻¹ * ftemp */
//    for (k = nlevU - 1; k >= 0; k--) {
//       #pragma omp parallel for private(i, ii, begin_row, end_row, j, jj, sum)
//       for (ii = ilevU[k]; ii < ilevU[k + 1]; ii++) {
//             i = jlevU[ii];
//             sum = ftemp_data[i];
//             begin_row = U_diag_i[i];
//             end_row = U_diag_i[i + 1];

//             for (j = begin_row; j < end_row; j++) {
//                jj = U_diag_j[j];
//                if (jj > i) { /* 严格上三角 */
//                   sum -= U_diag_data[j] *utemp_data[jj];
//                }
//             }
//             utemp_data[i] = sum *D[i]; /* 应用对角逆缩放 */
//       }


//    }

//    /* 最终更新：u = u + utemp */
//    jxf_ParVectorAxpy(beta, utemp, u);

//    return jxf_error_flag;
// }

// JXF_Int jxf_MultiL_TriangularSolve(jxf_ParCSRMatrix *A,
//                                     jxf_ParVector *f,
//                                     jxf_ParVector *u,
//                                     JXF_Int nLU,
//                                     jxf_ParCSRMatrix *L,
//                                     JXF_Real *D,
//                                     jxf_ParCSRMatrix *U,
//                                     jxf_ParVector *ftemp,
//                                     jxf_ParVector *utemp,
//                                     JXF_Int nlevL,
//                                     JXF_Int *jlevL,
//                                     JXF_Int *ilevL,
//                                     JXF_Int nlevU,
//                                     JXF_Int *jlevU,
//                                     JXF_Int *ilevU,
//                                     JXF_Int maxIter) 
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);
//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);
//     JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Int i, jj, j, iter, c, k, ii;
//     JXF_Int begin_row, end_row;
//     JXF_Real sum;

//     JXF_Real alpha = -1.0;
//     JXF_Real beta = 1.0;

//     // 初始残差 ftemp = f - A * u
//     jxf_ParVectorCopy(f, ftemp);
//     jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//     for (iter = 0; iter < maxIter; iter++) 
//     {
//         // 清空 utemp/ytemp
//         #pragma omp parallel for
//         for (i = 0; i < nLU; i++) 
//         {
//             utemp_data[i] = 0.0;
//             ytemp_data[i] = 0.0;
//         }

//         // === 前向替代：解 L * y = ftemp ===
//         for (k = 0; k < nlevL; k++) 
//         {
//             #pragma omp parallel for private(ii, i, j, jj, sum, begin_row, end_row)
//             for (ii = ilevL[k]; ii < ilevL[k + 1]; ii++) {
//                 i = jlevL[ii];
//                 sum = ftemp_data[i];
//                 begin_row = L_diag_i[i];
//                 end_row = L_diag_i[i + 1];

//                 for (j = begin_row; j < end_row; j++) {
//                     jj = L_diag_j[j];
//                     if (jj < i) {
//                         sum -= L_diag_data[j] * ytemp_data[jj];
//                     }
//                 }
//                 ytemp_data[i] = sum; // L 为单位对角矩阵
//             }
//         }

//         // === 后向替代：解 U * utemp = D⁻¹ * ytemp ===
//         for (k = 0; k < nlevU; k++) 
//         {
//             #pragma omp parallel for private(ii, i, j, jj, sum, begin_row, end_row)
//             for (ii = ilevU[k]; ii < ilevU[k + 1]; ii++) {
//                 i = jlevU[ii];
//                 sum = ytemp_data[i];
//                 begin_row = U_diag_i[i];
//                 end_row = U_diag_i[i + 1];

//                 for (j = begin_row; j < end_row; j++) {
//                     jj = U_diag_j[j];
//                     if (jj > i) {
//                         sum -= U_diag_data[j] * utemp_data[jj];
//                     }
//                 }
//                 utemp_data[i] = sum * D[i]; // U 对角不为 1，需要乘 D[i]
//             }
//         }

//         // 更新残差 ftemp = ftemp - A * utemp
//         jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//         jxf_ParVectorAxpy(beta, utemp, u);
//     }

//     free(ytemp_data);
//     return jxf_error_flag;
// }

// JXF_Int jxf_MultiL_TriangularSolve1(jxf_ParCSRMatrix *A, jxf_ParVector *f,
//                                     jxf_ParVector *u, JXF_Int nLU,
//                                     jxf_ParCSRMatrix *L, JXF_Real *D,
//                                     jxf_ParCSRMatrix *U, jxf_ParVector *ftemp,
//                                     jxf_ParVector *utemp, JXF_Int nlevL,
//                                     JXF_Int *jlevL, JXF_Int *ilevL,
//                                     JXF_Int nlevU, JXF_Int *jlevU,
//                                     JXF_Int *ilevU,JXF_Int maxIter,
//                                     JXF_Int *L_perm, JXF_Int *L_iperm,
//                                     JXF_Int *U_perm, JXF_Int *U_iperm)
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);
//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);
//     JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//     JXF_Real *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Int i, jj, j, iter, c, k, ii;
//     JXF_Int begin_row, end_row;
//     JXF_Real sum;

//     JXF_Real alpha = -1.0;
//     JXF_Real beta = 1.0;

//     // 初始残差 ftemp = f - A * u
//     jxf_ParVectorCopy(f, ftemp);
//     jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    for (iter = 0; iter < maxIter; iter++)
//     {
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++)
//       {
//          ytemp_data[i] = ftemp_data[i];
//          u_new[i] = 0;
//       }

//         // ====== 前向替代：解 L * y = ftemp，Jacobi迭代 ======
//          for (int lev = 0; lev < nlevL; lev++)
//             {
//                #pragma omp parallel for private( i, j, jj, sum, begin_row, end_row)
//                for (i = ilevL[lev]; i < ilevL[lev + 1]; i++)
//                 {
//                     sum = 0.0;
//                     begin_row = L_diag_i[i];
//                     end_row = L_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = L_diag_j[j];
//                         sum += L_diag_data[j] * u_new[jj];
//                     }
//                     u_new[L_iperm[i]] = ytemp_data[L_iperm[i]] - sum;
//                 }
//             }

//          #pragma omp parallel for
//             for (i = 0; i < nLU; i++)
//             {
//                ytemp_data[i] = u_new[i];
//                u_new[i] = 0;
//             }

//         // ====== 后向替代：解 U * utemp = D⁻¹ * ytemp ======
//             for (int lev = 0; lev < nlevU; lev++)
//             {
//                 #pragma omp parallel for private(ii,j, jj, sum, begin_row, end_row)
//                for (ii = ilevU[lev] ; ii < ilevU[lev + 1]; ii++)
//                 {
//                     i = ii;
//                     sum = 0.0;
//                     begin_row = U_diag_i[i];
//                     end_row = U_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = U_diag_j[j];
//                         sum += U_diag_data[j] * u_new[jj];
//                     }
//                     u_new[U_iperm[i]] = D[U_iperm[i]] * (ytemp_data[U_iperm[i]] - sum);
//                 }
//         }

//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++)
//          {
//             utemp_data[i] = u_new[i];
//          }

//         // 累加 u ← u + utemp，更新残差
//         jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//         jxf_ParVectorAxpy(beta, utemp, u);
//     }

//     free(ytemp_data);
//     free(u_new);

//     return jxf_error_flag;
// }

// JXF_Int jxf_MultiL_TriangularSolve1(jxf_ParCSRMatrix *A, jxf_ParVector *f,
//                                     jxf_ParVector *u, JXF_Int nLU,
//                                     jxf_ParCSRMatrix *L, JXF_Real *D,
//                                     jxf_ParCSRMatrix *U, jxf_ParVector *ftemp,
//                                     jxf_ParVector *utemp, JXF_Int nlevL,
//                                     JXF_Int *jlevL, JXF_Int *ilevL,
//                                     JXF_Int nlevU, JXF_Int *jlevU,
//                                     JXF_Int *ilevU,JXF_Int maxIter,
//                                     JXF_Int *L_perm, JXF_Int *L_iperm,
//                                     JXF_Int *U_perm, JXF_Int *U_iperm)
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);
//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);
//     JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    //  JXF_Real *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Int i, jj, j, iter, c, k, ii;
//     JXF_Int begin_row, end_row;
//     JXF_Real sum;

//     JXF_Real alpha = -1.0;
//     JXF_Real beta = 1.0;

//     // 初始残差 ftemp = f - A * u
//     jxf_ParVectorCopy(f, ftemp);
//     jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    for (iter = 0; iter < maxIter; iter++)
//     {
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++)
//       {
//          //ytemp_data[i] = ftemp_data[i];
//          ytemp_data[i] = 0;
//           utemp_data[i] = 0;
//       }

//         // ====== 前向替代：解 L * y = ftemp，Jacobi迭代 ======
//          for (int lev = 0; lev < nlevL; lev++)
//             {
//                #pragma omp parallel for private( i, j, jj, sum, begin_row, end_row)
//                for (i = ilevL[lev]; i < ilevL[lev + 1]; i++)
//                 {
//                     sum = 0.0;
//                     begin_row = L_diag_i[i];
//                     end_row = L_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = L_diag_j[j];
//                         sum += L_diag_data[j] * ytemp_data[jj];
//                     }
//                     ytemp_data[L_iperm[i]] = ftemp_data[L_iperm[i]] - sum;
//                 }
//             }

//         // ====== 后向替代：解 U * utemp = D⁻¹ * ytemp ======
//             for (int lev = 0; lev < nlevU; lev++)
//             {
//                 #pragma omp parallel for private(ii,j, jj, sum, begin_row, end_row)
//                for (ii = ilevU[lev] ; ii < ilevU[lev + 1]; ii++)
//                 {
//                     i = ii;
//                     sum = 0.0;
//                     begin_row = U_diag_i[i];
//                     end_row = U_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = U_diag_j[j];
//                         sum += U_diag_data[j] * utemp_data[jj];
//                     }
//                     utemp_data[U_iperm[i]] = D[U_iperm[i]] * (ytemp_data[U_iperm[i]] - sum);
//                 }
//         }

//          // #pragma omp parallel for
//          // for (i = 0; i < nLU; i++)
//          // {
//          //    utemp_data[i] = u_new[i];
//          // }

//         // 累加 u ← u + utemp，更新残差
//         jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//         jxf_ParVectorAxpy(beta, utemp, u);
//     }

//     free(ytemp_data);
//    //  free(u_new);

//     return jxf_error_flag;
// }

//多色GS逼近L，多色GS逼近U
// JXF_Int jxf_MultiColor_baseTriangularSolve(jxf_ParCSRMatrix *A,
//                                     jxf_ParVector *f,
//                                     jxf_ParVector *u,
//                                     JXF_Int nLU,
//                                     jxf_ParCSRMatrix *L,
//                                     JXF_Real *D,
//                                     jxf_ParCSRMatrix *U,
//                                     jxf_ParVector *ftemp,
//                                     jxf_ParVector *utemp,
//                                     JXF_Int nlevL,
//                                     JXF_Int *jlevL,
//                                     JXF_Int *ilevL,
//                                     JXF_Int nlevU,
//                                     JXF_Int *jlevU,
//                                     JXF_Int *ilevU,
//                                     JXF_Int maxIter)
// {
//     /* 提取 L 和 U 的对角块 */
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//     /* 获取向量数据 */
//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);
//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);
//     JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Int i, jj, j, iter, c, k, ii;
//     JXF_Int begin_row, end_row;
//     JXF_Real sum;

//     /* 计算初始残差：ftemp = f - A * u */
//     JXF_Real alpha = -1.0;
//     JXF_Real beta = 1.0;
//     jxf_ParVectorCopy(f, ftemp);
//     jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//     /* 初始化 utemp 和 ytemp 为零 */
//     #pragma omp parallel for
//     for (i = 0; i < nLU; i++) 
//     {
//         utemp_data[i] = 0.0;
//         ytemp_data[i] = 0.0;
//     }
//     /* 主迭代循环 */
//     for (iter = 0; iter < maxIter; iter++) 
//     {
//          for ( k = 0; k < nlevL; k++) {
//             #pragma omp parallel for private(ii, j, jj, sum, begin_row, end_row) shared(ytemp_data, ftemp_data, L_diag_i, L_diag_j, L_diag_data)
//             for ( ii = ilevL[k]; ii < ilevL[k+1]; ii++) {
//                i = jlevL[ii];
//                sum = ftemp_data[i];
//                begin_row = L_diag_i[i];
//                end_row = L_diag_i[i+1];

//                for (j = begin_row; j < end_row; j++) 
//                {
//                   jj = L_diag_j[j]; 
//                   sum -= L_diag_data[j] * ytemp_data[jj];
//                }
//                ytemp_data[i] = sum;
//             }
//             #pragma omp barrier
//          }
//    }

//    for (iter = 0; iter < maxIter; iter++) 
//    {
//       for (k =0; k < nlevU; k++) {
//          #pragma omp parallel for private(ii, j, jj, sum, begin_row, end_row) shared(utemp_data, ytemp_data, U_diag_i, U_diag_j, U_diag_data, D)
//          for (ii = ilevU[k]; ii < ilevU[k+1]; ii++) {
//             i = jlevU[ii];
//             sum = ytemp_data[i];
//             begin_row = U_diag_i[i];
//             end_row = U_diag_i[i+1];

//             for (j = begin_row; j < end_row; j++) 
//             {
//                jj = U_diag_j[j]; /* 修正：使用 j 而非 jj 作为索引 */
//                sum -= U_diag_data[j] * utemp_data[jj];
//             }
//             utemp_data[i] = sum * D[i]; /* 应用对角逆缩放 */
//          }
//          #pragma omp barrier
//       }
//    }

//    jxf_ParVectorAxpy(beta, utemp, u);
//     /* 清理内存 */
//     free(ytemp_data);
//     return jxf_error_flag;
// }


// //多次GS
// JXF_Int jxf_MultiGS_TriangularSolve(jxf_ParCSRMatrix *A,
//                                     jxf_ParVector *f,
//                                     jxf_ParVector *u,
//                                     JXF_Int nLU,
//                                     jxf_ParCSRMatrix *L,
//                                     JXF_Real *D,
//                                     jxf_ParCSRMatrix *U,
//                                     jxf_ParVector *ftemp,
//                                     jxf_ParVector *utemp,
//                                     JXF_Int maxIter) 
// {
//     /* 提取 L 和 U 的对角块 */
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//     /* 获取向量数据 */
//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);
//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);
//     JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Real sum;
//     JXF_Int begin_row, end_row, i, jj, j, iter, k;
//    JXF_Real alpha = -1.0;
//    JXF_Real beta = 1.0;
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//     /* 迭代 maxIter 次 */
//     for (iter = 0; iter < maxIter; iter++) 
//     {
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = 0.0;
//          ytemp_data[i] = 0.0;
//       }
//       // === 前向替代：解 L * y = ftemp ===
//       for (k = 0; k < 1; k++) {
//             for (i = k; i < nLU; i += 1) {
//                sum = ftemp_data[i];
//                begin_row = L_diag_i[i];
//                end_row = L_diag_i[i + 1];
//                for (j = begin_row; j < end_row; j++) {
//                   jj = L_diag_j[j];
//                   if (jj < i) {
//                         sum -= L_diag_data[j] * ytemp_data[jj];
//                   }
//                }
//                ytemp_data[i] = sum; // L 为单位下三角
//             }
//          }

//         /* 后向替换：解上三角矩阵方程 U * u = D⁻¹ * ftemp */
//       for (k = 0; k < 1; k++) {
//          for (i = nLU - 1 - k; i >= 0; i -= 1) {
//                sum = ytemp_data[i];
//                begin_row = U_diag_i[i];
//                end_row = U_diag_i[i + 1];
//                for (j = begin_row; j < end_row; j++) {
//                   jj = U_diag_j[j];
//                   if (jj > i) {
//                      sum -= U_diag_data[j] * utemp_data[jj];
//                   }
//                }
//                utemp_data[i] = sum * D[i];
//          }
//       }
//       jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//       jxf_ParVectorAxpy(beta, utemp, u);
//    }

//    free(ytemp_data);
//    return jxf_error_flag;
// }

// JXF_Int jxf_MultiColorTriangularSolve(jxf_ParCSRMatrix *A,jxf_ParVector *f,
//                            jxf_ParVector *u,JXF_Int nLU,jxf_ParCSRMatrix *L,
//                            JXF_Real *D,jxf_ParCSRMatrix *U,jxf_ParVector *ftemp,
//                            jxf_ParVector *utemp,JXF_Int nlevL, JXF_Int *jlevL,
//                            JXF_Int *ilevL,JXF_Int nlevU,JXF_Int *jlevU,
//                            JXF_Int *ilevU,JXF_Int maxIter,JXF_Int lower_jacobi_iters,
//                            JXF_Int upper_jacobi_iters)
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);

//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);

//     JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//     JXF_Real *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Int i, j, ii, jj, k, kk;
//     JXF_Int begin_row, end_row;
//     JXF_Real sum;
//     JXF_Real alpha = -1.0;
//     JXF_Real beta = 1.0;
//    //  JXF_Int lower_jacobi_iters = 3;
//    //  JXF_Int upper_jacobi_iters = 3;

//     // 初始残差 ftemp = f - A * u
//     jxf_ParVectorCopy(f, ftemp);
//     jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//     for (kk = 0; kk < maxIter; kk++)
//     {
//         // 清空中间变量
//         #pragma omp parallel for
//         for (i = 0; i < nLU; i++) {
//             utemp_data[i] = 0.0;
//             ytemp_data[i] = 0.0;
//         }

//         // ====== 前向替代：解 L * y = ftemp，Jacobi迭代 ======
//         for (k = 0; k < lower_jacobi_iters; k++)
//         {
//             for (int lev = 0; lev < nlevL; lev++)
//             {
//                 #pragma omp parallel for private(i, j, jj, sum, begin_row, end_row)
//                 for (ii = ilevL[lev]; ii < ilevL[lev + 1]; ii++)
//                 {
//                     i = jlevL[ii];
//                     sum = 0.0;
//                     begin_row = L_diag_i[i];
//                     end_row = L_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = L_diag_j[j];
//                         if (jj != i) // 单位对角，跳过
//                             sum += L_diag_data[j] * ytemp_data[jj];
//                     }
//                     ytemp_data[i] = ftemp_data[i] - sum;
//                 }
//             }
//         }

//         // ====== 后向替代：解 U * utemp = D⁻¹ * ytemp，Jacobi迭代 ======
//         for (k = 0; k < upper_jacobi_iters; k++)
//         {
//             for (int lev = 0; lev < nlevU; lev++)
//             {
//                 #pragma omp parallel for private(i, j, jj, sum, begin_row, end_row)
//                 for (ii = ilevU[lev]; ii < ilevU[lev + 1]; ii++)
//                 {
//                     i = jlevU[ii];
//                     sum = 0.0;
//                     begin_row = U_diag_i[i];
//                     end_row = U_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = U_diag_j[j];
//                         if (jj != i)
//                             sum += U_diag_data[j] * utemp_data[jj];
//                     }

//                     utemp_data[i] = D[i] * (ytemp_data[i] - sum);
//                 }
//             }
//         }

//         // 累加 u ← u + utemp，更新残差
//         jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//         jxf_ParVectorAxpy(beta, utemp, u);
//     }

//     free(ytemp_data);
//     free(u_new);
//     return jxf_error_flag;
// }


// JXF_Int jxf_MultiColorTriangularSolve1(jxf_ParCSRMatrix *A,jxf_ParVector *f,
//                                     jxf_ParVector *u,JXF_Int nLU,jxf_ParCSRMatrix *L,
//                                     JXF_Real *D,jxf_ParCSRMatrix *U,jxf_ParVector *ftemp,
//                                     jxf_ParVector *utemp,JXF_Int nlevL, JXF_Int *jlevL,
//                                     JXF_Int *ilevL,JXF_Int nlevU,JXF_Int *jlevU,
//                                     JXF_Int *ilevU,JXF_Int maxIter,JXF_Int lower_jacobi_iters,
//                                     JXF_Int upper_jacobi_iters,JXF_Int *L_perm, JXF_Int *L_iperm,
//                                     JXF_Int *U_perm, JXF_Int *U_iperm)
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//       jxf_Vector *f_local = jxf_ParVectorLocalVector(f);
//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);

//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);

//     JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    JXF_Real *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Int i, j, ii, jj, k, kk;
//     JXF_Int begin_row, end_row;
//     JXF_Real sum;
//     JXF_Real alpha = -1.0;
//     JXF_Real beta = 1.0;

//     // 初始残差 ftemp = f - A * u
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//     for (kk = 0; kk < maxIter; kk++)
//     {
//       //   // 清空中间变量
//       //   #pragma omp parallel for
//       //   for (i = 0; i < nLU; i++) {
//       //       utemp_data[i] = 0.0;
//       //       ytemp_data[i] = 0.0;
//       //   }
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++)
//       {
//          ytemp_data[L_perm[i]] = ftemp_data[i];
//          u_new[i] = 0;
//       }

//         // ====== 前向替代：解 L * y = ftemp，Jacobi迭代 ======
//         for (k = 0; k < lower_jacobi_iters; k++)
//         {
//             for (int lev = 0; lev < nlevL; lev++)
//             {
//                 #pragma omp parallel for private(j, jj, sum, begin_row, end_row)
//                for (ii = ilevL[lev]; ii < ilevL[lev + 1]; ii++)
//                 {
//                      i = jlevL[ii];
//                     sum = 0.0;
//                     begin_row = L_diag_i[i];
//                     end_row = L_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = L_diag_j[j];
//                         if (jj != i) // 单位对角，跳过
//                             sum += L_diag_data[j] * u_new[jj];
//                     }
//                     u_new[i] = ytemp_data[i] - sum;
//                 }
//             }
//         }
//         #pragma omp parallel for
//          for (i = 0; i < nLU; i++)
//          {
//             utemp_data[L_iperm[i]] = u_new[i];
//          }
//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++)
//          {
//             ytemp_data[U_perm[i]] = utemp_data[i];
//             u_new[i] = 0;
//          }

//         // ====== 后向替代：解 U * utemp = D⁻¹ * ytemp，Jacobi迭代 ======
//         for (k = 0; k < upper_jacobi_iters; k++)
//         {
//             for (int lev = 0; lev < nlevU; lev++)
//             {
//                 #pragma omp parallel for private(j, jj, sum, begin_row, end_row)
//                //  for (ii = ilevU[lev]; ii < ilevU[lev + 1]; ii++)
//                for (ii = ilevU[lev]; ii < ilevU[lev + 1]; ii++)
//                 {
//                      // i = nLU-1 -i;
//                    i = jlevU[ii];
//                   // i = U_perm[i];
//                     sum = 0.0;
//                     begin_row = U_diag_i[i];
//                     end_row = U_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = U_diag_j[j];
//                         if (jj != i)
//                             sum += U_diag_data[j] * u_new[jj];
//                     }
//                     u_new[i] = D[U_iperm[i]] * (ytemp_data[i] - sum);
//                 }
//             }
//         }

//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++)
//          {
//             utemp_data[U_iperm[i]] = u_new[i];
//          }

//         // 累加 u ← u + utemp，更新残差
//         jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//         jxf_ParVectorAxpy(beta, utemp, u);
//     }

//     free(ytemp_data);
//     free(u_new);
//     return jxf_error_flag;
// }

// JXF_Int jxf_MultiColorTriangularSolve1(jxf_ParCSRMatrix *A,jxf_ParVector *f,
//                                     jxf_ParVector *u,JXF_Int nLU,jxf_ParCSRMatrix *L,
//                                     JXF_Real *D,jxf_ParCSRMatrix *U,jxf_ParVector *ftemp,
//                                     jxf_ParVector *utemp,JXF_Int nlevL, JXF_Int *jlevL,
//                                     JXF_Int *ilevL,JXF_Int nlevU,JXF_Int *jlevU,
//                                     JXF_Int *ilevU,JXF_Int maxIter,JXF_Int lower_jacobi_iters,
//                                     JXF_Int upper_jacobi_iters,JXF_Int *L_perm, JXF_Int *L_iperm,
//                                     JXF_Int *U_perm, JXF_Int *U_iperm)
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//       jxf_Vector *f_local = jxf_ParVectorLocalVector(f);
//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);

//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);

//     JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    JXF_Real *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Int i, j, ii, jj, k, kk;
//     JXF_Int begin_row, end_row;
//     JXF_Real sum;
//     JXF_Real alpha = -1.0;
//     JXF_Real beta = 1.0;

//     // 初始残差 ftemp = f - A * u
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//     for (kk = 0; kk < maxIter; kk++)
//     {
//       //   // 清空中间变量
//       //   #pragma omp parallel for
//       //   for (i = 0; i < nLU; i++) {
//       //       utemp_data[i] = 0.0;
//       //       ytemp_data[i] = 0.0;
//       //   }
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++)
//       {
//          ytemp_data[i] = ftemp_data[i];
//          u_new[i] = 0;
//       }

//         // ====== 前向替代：解 L * y = ftemp，Jacobi迭代 ======
//         for (k = 0; k < lower_jacobi_iters; k++)
//         {
//             for (int lev = 0; lev < nlevL; lev++)
//             {
//                 #pragma omp parallel for private(j, jj, sum, begin_row, end_row)
//                for (i = ilevL[lev]; i < ilevL[lev + 1]; i++)
//                 {
//                     sum = 0.0;
//                     begin_row = L_diag_i[i];
//                     end_row = L_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = L_diag_j[j];
//                         sum += L_diag_data[j] * u_new[jj];
//                     }
//                     u_new[L_iperm[i]] = ytemp_data[L_iperm[i]] - sum;
//                 }
//             }
//         }
//         #pragma omp parallel for
//          for (i = 0; i < nLU; i++)
//          {
//             ytemp_data[i] = u_new[i];
//             u_new[i] = 0;
//          }

//         // ====== 后向替代：解 U * utemp = D⁻¹ * ytemp，Jacobi迭代 ======
//         for (k = 0; k < upper_jacobi_iters; k++)
//         {
//             for (int lev = 0; lev < nlevU; lev++)
//             {
//                 #pragma omp parallel for private(j, jj, sum, begin_row, end_row)
//                //  for (ii = ilevU[lev]; ii < ilevU[lev + 1]; ii++)
//                for (ii = ilevU[lev] ; ii < ilevU[lev + 1]; ii++)
//                 {
//                     i = ii;  //nLU - ii -1;
//                     sum = 0.0;
//                     begin_row = U_diag_i[i];
//                     end_row = U_diag_i[i + 1];

//                     for (j = begin_row; j < end_row; j++)
//                     {
//                         jj = U_diag_j[j];
//                         sum += U_diag_data[j] * u_new[jj];
//                     }
//                     u_new[U_iperm[i]] = D[U_iperm[i]] * (ytemp_data[U_iperm[i]] - sum);
//                 }
//             }
//         }

//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++)
//          {
//             utemp_data[i] = u_new[i];
//          }

//         // 累加 u ← u + utemp，更新残差
//         jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//         jxf_ParVectorAxpy(beta, utemp, u);
//     }

//     free(ytemp_data);
//     free(u_new);
//     return jxf_error_flag;
// }

// JXF_Int jxf_MultiColorTriangularSolve(jxf_ParCSRMatrix *A,
//                                     jxf_ParVector *f,
//                                     jxf_ParVector *u,
//                                     JXF_Int nLU,
//                                     jxf_ParCSRMatrix *L,
//                                     JXF_Real *D,
//                                     jxf_ParCSRMatrix *U,
//                                     jxf_ParVector *ftemp,
//                                     jxf_ParVector *utemp,
//                                     JXF_Int nlevL,
//                                     JXF_Int *jlevL,
//                                     JXF_Int *ilevL,
//                                     JXF_Int nlevU,
//                                     JXF_Int *jlevU,
//                                     JXF_Int *ilevU,
//                                     JXF_Int maxIter) 
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);
//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);
//     JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Int i, jj, j, iter, k;
//     JXF_Int begin_row, end_row;
//     JXF_Real sum;

//     JXF_Real alpha = -1.0;
//     JXF_Real beta = 1.0;
//     jxf_ParVectorCopy(f, ftemp);
//     jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//     for (i = 0; i < nLU; i++) {
//         utemp_data[i] = 0;
//         ytemp_data[i] = 0;
//     }

//     for (iter = 0; iter < maxIter; iter++) 
//     {
//         // === 前向替代：解 L * y = ftemp ===
//         for (k = 0; k < 1; k++) {
//             for (i = k; i < nLU; i += 1) {
//                 sum = ftemp_data[i];
//                 begin_row = L_diag_i[i];
//                 end_row = L_diag_i[i + 1];
//                 for (j = begin_row; j < end_row; j++) {
//                     jj = L_diag_j[j];
//                     if (jj < i) {
//                         sum -= L_diag_data[j] * ytemp_data[jj];
//                     }
//                 }
//                 ytemp_data[i] = sum; // L 为单位下三角
//             }
//         }
//     }

//     for (iter = 0; iter < maxIter; iter++) 
//     {
//         // === 后向替代：解 U * utemp = D⁻¹ * y ===
//         for (k = 0; k < 1; k++) {
//             for (i = nLU - 1 - k; i >= 0; i -= 1) {
//                 sum = ytemp_data[i];
//                 begin_row = U_diag_i[i];
//                 end_row = U_diag_i[i + 1];
//                 for (j = begin_row; j < end_row; j++) {
//                     jj = U_diag_j[j];
//                     if (jj > i) {
//                         sum -= U_diag_data[j] * utemp_data[jj];
//                     }
//                 }
//                 utemp_data[i] = sum * D[i];
//             }
//         }
//     }

//     jxf_ParVectorAxpy(beta, utemp, u);
//     free(ytemp_data);
//     return jxf_error_flag;
// }

// //单次GS即直接三角求解
// JXF_Int jxf_GSTriangularSolve(jxf_ParCSRMatrix *A,
//                                     jxf_ParVector *f,
//                                     jxf_ParVector *u,
//                                     JXF_Int nLU,
//                                     jxf_ParCSRMatrix *L,
//                                     JXF_Real *D,
//                                     jxf_ParCSRMatrix *U,
//                                     jxf_ParVector *ftemp,
//                                     jxf_ParVector *utemp) 
// {
//     /* 提取 L 和 U 的对角块 */
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//     /* 获取向量数据 */
//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);
//    jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real   *utemp_data  = jxf_VectorData(utemp_local);

//     JXF_Real sum;
//     JXF_Int begin_row, end_row, i, jj, j, iter;
   
//       /* 计算残差：ftemp = f - A * u */
//    JXF_Real alpha = -1.0;
//    JXF_Real beta = 1.0;
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);
   
//       /* 初始化 utemp = ftemp */
//    for ( i = 0; i < nLU; i++) {
//       utemp_data[i] = ftemp_data[i];
//    }

//    /* 前向替换：解单位下三角矩阵方程 L * ftemp = ftemp */
//    for (i = 0; i < nLU; i++) {
//       sum = utemp_data[i]; // 初始化 sum
//       begin_row = L_diag_i[i];
//       end_row = L_diag_i[i + 1];
//       for (j = begin_row; j < end_row; j++) {
//             jj = L_diag_j[j];
//             if (jj < i) { /* 严格下三角 */
//                sum -= L_diag_data[j] * utemp_data[jj];
//             }
//       }
//       utemp_data[i] = sum; /* L 对角线为 1 */
//    }

//    /* 后向替换：解上三角矩阵方程 U * u = D⁻¹ * ftemp */
//    for (i = nLU - 1; i >= 0; i--) {
//       sum = utemp_data[i]; // 初始化 sum
//       begin_row = U_diag_i[i];
//       end_row = U_diag_i[i + 1];
//       for (j = begin_row; j < end_row; j++) {
//             jj = U_diag_j[j];
//             if (jj > i) { /* 严格上三角 */
//                sum -= U_diag_data[j] * utemp_data[jj];
//             }
//       }
//       utemp_data[i] = sum * D[i]; /* 应用对角逆缩放 */
//    }

//    jxf_ParVectorAxpy(beta, utemp, u);
   
//    return jxf_error_flag;
// }

// JXF_Int jxf_JacobiTriangularSolve(jxf_ParCSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u,
//                                 JXF_Int nLU, jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
//                                 jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int maxIter) 
// {
//     /* 提取 L 和 U 的对角块 */
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);

//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Real *U_diag_data = jxf_CSRMatrixData(U_diag);
//     JXF_Int *U_diag_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_diag_j = jxf_CSRMatrixJ(U_diag);

//     /* 获取向量数据 */
//     jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//     JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);
//     jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//     JXF_Real *utemp_data = jxf_VectorData(utemp_local);

//     /* 创建中间数组用于存储新迭代的解 */
//     JXF_Real *utemp_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     JXF_Real sum;
//     JXF_Int i, j, jj, iter, begin_row, end_row;

//     /* 计算初始残差：ftemp = f - A * u */
//     JXF_Real alpha = -1.0, beta = 1.0;
//     jxf_ParVectorCopy(f, ftemp);
//     jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//     /* 初始化 utemp = ftemp */
//     for (i = 0; i < nLU; i++) {
//         utemp_data[i] = ftemp_data[i];
//     }

//     /* Jacobi 迭代 */
//     for (iter = 0; iter < maxIter; iter++) {
//         /* 计算新解：utemp_new = D⁻¹ * (ftemp - (L + U) * utemp) */
//         for (i = 0; i < nLU; i++) {
//             sum = ftemp_data[i]; // 残差项

//             /* 计算 L * utemp 的贡献（严格下三角） */
//             begin_row = L_diag_i[i];
//             end_row = L_diag_i[i + 1];
//             for (j = begin_row; j < end_row; j++) {
//                 jj = L_diag_j[j];
//                 if (jj != i) { /* 排除对角元素 */
//                     sum -= L_diag_data[j] * utemp_data[jj];
//                 }
//             }

//             /* 计算 U * utemp 的贡献（严格上三角） */
//             begin_row = U_diag_i[i];
//             end_row = U_diag_i[i + 1];
//             for (j = begin_row; j < end_row; j++) {
//                 jj = U_diag_j[j];
//                 if (jj != i) { /* 排除对角元素 */
//                     sum -= U_diag_data[j] * utemp_data[jj];
//                 }
//             }

//             /* 应用对角逆缩放 */
//             utemp_new[i] = sum * D[i];
//         }

//         /* 更新 utemp 为新解 */
//         for (i = 0; i < nLU; i++) {
//             utemp_data[i] = utemp_new[i];
//         }
//     }

//     /* 更新最终解：u = u + utemp */
//     jxf_ParVectorAxpy(beta, utemp, u);

//     /* 释放中间数组 */
//     free(utemp_new);

//    return jxf_error_flag;
// }

// JXF_Int
// jxf_JacobiTriangularSolve(jxf_ParCSRMatrix *A,
//                      jxf_ParVector    *f,
//                      jxf_ParVector    *u,
//                      JXF_Int          *perm,
//                      JXF_Int           nLU,
//                      jxf_ParCSRMatrix *L,
//                      JXF_Real         *D,
//                      jxf_ParCSRMatrix *U,
//                      jxf_ParVector    *ftemp,
//                      jxf_ParVector    *utemp,
//                      JXF_Int           maxIter,
//                      JXF_Int           lower_jacobi_iters,
//                      JXF_Int           upper_jacobi_iters)
// {
//    /* Data objects for L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* Local variables */
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Real       sum;
//    JXF_Int        i, j, k1, k2, kk,iter;

//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* Initialize iteration to 0 */
//    JXF_Real  *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    JXF_Real  *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//        /* 迭代 maxIter 次 */
//     for (iter = 0; iter < maxIter; iter++) 
//     {
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = 0.0;
//       }
//       /* Jacobi iteration loop */
//       for ( kk = 0; kk < lower_jacobi_iters; kk++ )  //lower_jacobi_iters
//       {
//          #pragma omp parallel for private(i,j, k1, k2, sum)
//          for ( i = 0; i< nLU; i++ )
//          {
//             sum = 0.0;
//             k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += L_diag_data[j] * utemp_data[L_diag_j[j]];
//             }
//             ytemp_data[i] = ftemp_data[i] - sum;
//          }
//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++) utemp_data[i] = ytemp_data[i];
//       } /* end jacobi loop */

//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = 0.0;
//       }

//       /* Jacobi iteration loop */
//       for ( kk = 0; kk < upper_jacobi_iters; kk++ )  //upper_jacobi_iters
//       {
//          #pragma omp parallel for private(i,j, k1, k2, sum)
//          /* u^{k+1} = f - Uu^k */
//          for ( i = 0; i < nLU; ++i )
//          {
//             sum = 0.0;
//             k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += U_diag_data[j] * utemp_data[U_diag_j[j]];
//             }
//             u_new[i] = D[i] * (ytemp_data[i] - sum);
//          }
//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++) utemp_data[i] = u_new[i];
//       } /* end jacobi loop */

//       jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//       jxf_ParVectorAxpy(beta, utemp, u);
//    }

//    free(u_new);
//    free(ytemp_data);

//    return jxf_error_flag;
// }

// JXF_Int
// jxf_JacobiTriangularSolve22(jxf_ParCSRMatrix *A,
//                      jxf_ParVector    *f,
//                      jxf_ParVector    *u,
//                      JXF_Int          *perm,
//                      JXF_Int           nLU,
//                      jxf_ParCSRMatrix *L,
//                      JXF_Real         *D,
//                      jxf_ParCSRMatrix *U,
//                      jxf_ParVector    *ftemp,
//                      jxf_ParVector    *utemp,
//                      JXF_Int           maxIter,
//                      JXF_Int           lower_jacobi_iters,
//                      JXF_Int           upper_jacobi_iters)
// {
//    /* Data objects for L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* Local variables */
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Real       sum;
//    JXF_Int        i, j, k1, k2, kk,iter;

//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* Initialize iteration to 0 */
//    JXF_Real  *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    JXF_Real  *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//        /* 迭代 maxIter 次 */
//     for (iter = 0; iter < maxIter; iter++) 
//     {
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = 0.0;
//       }
//       /* Jacobi iteration loop */
//       for ( kk = 0; kk < lower_jacobi_iters; kk++ )  //lower_jacobi_iters
//       {
//          #pragma omp parallel for private(i,j, k1, k2, sum)
//          for ( i = 0; i< nLU; i++ )
//          {
//             sum = 0.0;
//             k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += L_diag_data[j] * utemp_data[L_diag_j[j]];
//             }
//             ytemp_data[i] = ftemp_data[i] - sum;
//          }

//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++) 
//          {
//             u_new[i] = 0.0;
//          }

//          #pragma omp parallel for private(i,j, k1, k2, sum)
//          /* u^{k+1} = f - Uu^k */
//          for ( i = 0; i < nLU; ++i )
//          {
//             sum = 0.0;
//             k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += U_diag_data[j] * u_new[U_diag_j[j]];
//             }
//             utemp_data[i] = D[i] * (ytemp_data[i] - sum);
//          }
//       } /* end jacobi loop */

//       jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//       jxf_ParVectorAxpy(beta, utemp, u);
//    }

//    free(u_new);
//    free(ytemp_data);

//    return jxf_error_flag;
// }

// JXF_Int
// jxf_JacobiTriangularSolve33(jxf_ParCSRMatrix *A,
//                      jxf_ParVector    *f,
//                      jxf_ParVector    *u,
//                      JXF_Int          *perm,
//                      JXF_Int           nLU,
//                      jxf_ParCSRMatrix *L,
//                      JXF_Real         *D,
//                      jxf_ParCSRMatrix *U,
//                      jxf_ParVector    *ftemp,
//                      jxf_ParVector    *utemp,
//                      JXF_Int           maxIter,
//                      JXF_Int           lower_jacobi_iters,
//                      JXF_Int           upper_jacobi_iters)
// {
//    /* Data objects for L and U */
//    jxf_CSRMatrix *A_diag      = jxf_ParCSRMatrixDiag(A);
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *u_local = jxf_ParVectorLocalVector(u);
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* Local variables */
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Real       sum;
//    JXF_Int        i, j, k1, k2, kk,iter;

//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* Initialize iteration to 0 */
//    JXF_Real  *u_new = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    JXF_Real  *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//        /* 迭代 maxIter 次 */
//     for (iter = 0; iter < maxIter; iter++) 
//     {
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = 0.0;
//       }

//       for ( kk = 0; kk < lower_jacobi_iters; kk++ )  //lower_jacobi_iters
//       {
//          #pragma omp parallel for private(i,j, k1, k2, sum)
//          for ( i = 0; i< nLU; i++ )
//          {
//             sum = 0.0;
//             k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += L_diag_data[j] * utemp_data[L_diag_j[j]];
//             }
//             ytemp_data[i] = ftemp_data[i] - sum;
//          }
//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++) 
//          {
//             u_new[i] = 0.0;
//          }
//          #pragma omp parallel for private(i,j, k1, k2, sum)
//          for ( i = 0; i < nLU; ++i )
//          {
//             sum = 0.0;
//             k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += U_diag_data[j] * u_new[U_diag_j[j]];
//             }
//             utemp_data[i] = D[i] * (ytemp_data[i] - sum);
//          }
         
//          jxf_CSRMatrixMatvec(alpha,A_diag,utemp_local, beta, ftemp_local);
//          jxf_ParVectorAxpy(beta, utemp, u);
//       }
//       jxf_ParVectorCopy(f, ftemp);
//       jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);
//    }

//    free(u_new);
//    free(ytemp_data);

//    return jxf_error_flag;
// }


// JXF_Int
// jxf_JacobiTriangularSolve_spmv(jxf_ParCSRMatrix *A,
//                      jxf_ParVector    *f,
//                      jxf_ParVector    *u,
//                      JXF_Int          *perm,
//                      JXF_Int           nLU,
//                      jxf_ParCSRMatrix *L,
//                      JXF_Real         *D,
//                      jxf_ParCSRMatrix *U,
//                      jxf_ParVector    *ftemp,
//                      jxf_ParVector    *utemp,
//                      JXF_Int           maxIter,
//                      JXF_Int           lower_jacobi_iters,
//                      JXF_Int           upper_jacobi_iters)
// {
//    /* Data objects for L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* Local variables */
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Real       sum;
//    JXF_Int        i, j, k1, k2, kk,iter;

//    // char FileNameCoaMat[256];
//    //  jxf_sprintf(FileNameCoaMat, "A_CSR_%d", 1);
//    //  jxf_ParCSRMatrixPrint(A, FileNameCoaMat);
//    //  jxf_sprintf(FileNameCoaMat, "L_CSR_%d", 1);
//    //  jxf_ParCSRMatrixPrint(L, FileNameCoaMat);
//    //  jxf_sprintf(FileNameCoaMat, "U_CSR_%d", 1);
//    //  jxf_ParCSRMatrixPrint(U, FileNameCoaMat);

//    // FILE *fp;
//    // jxf_sprintf(FileNameCoaMat, "D_%d", 1);
//    // fp = fopen(FileNameCoaMat, "w");
//    // for(j = 0; j < nLU; j++){
//    // jxf_fprintf(fp, "%.14e\n", D[j]);
//    // }
//    // fclose(fp);

//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* Initialize iteration to 0 */
//    // JXF_Real  *unew = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//    jxf_Vector  *ytemp = jxf_SeqVectorCreate(nLU);
//    jxf_SeqVectorInitialize(ytemp);
//    JXF_Real  *ytemp_data= jxf_VectorData(ytemp);

//    jxf_Vector  *unew = jxf_SeqVectorCreate(nLU);
//    jxf_SeqVectorInitialize(unew);
//    JXF_Real  *unew_data= jxf_VectorData(unew);

//    // JXF_Real  *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//     /* 迭代 maxIter 次 */
//     for (iter = 0; iter < maxIter; iter++) 
//     {
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = ftemp_data[i];
//          ytemp_data[i] = 0;
//       }
//       /* Jacobi iteration loop */
//       for ( kk = 0; kk < lower_jacobi_iters; kk++ )  //lower_jacobi_iters
//       {
//          //jxf_CSRMatrixMatvec_LU(alpha,L_diag, ytemp, beta, ftemp_local);
//          jxf_CSRMatrixMatvec(alpha,L_diag, ytemp, 0, utemp_local);
//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++) 
//          {
//             ytemp_data[i] = ftemp_data[i]+utemp_data[i];
//          }

//       } /* end jacobi loop */

//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = 0;
//          unew_data[i] = 0;
//       }

//       /* Jacobi iteration loop */
//       for ( kk = 0; kk < upper_jacobi_iters; kk++ )  //upper_jacobi_iters
//       {
//          //jxf_CSRMatrixMatvec_LU(alpha,L_diag, utemp_local, beta, ytemp);
//          jxf_CSRMatrixMatvec(alpha,U_diag, utemp_local, 0, unew);
//          #pragma omp parallel for
//          for (i = 0; i < nLU; i ++)
//          {
//             utemp_data[i] = D[i] *( ytemp_data[i]+unew_data[i]);
//          }
//       } /* end jacobi loop */

//       jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//       jxf_ParVectorAxpy(beta, utemp, u);
//    }

//    // free(u_new);
//    //free(ytemp_data);
//    jxf_SeqVectorDestroy(ytemp); 
//    jxf_SeqVectorDestroy(unew); 

//    return jxf_error_flag;
// }

// JXF_Int
// jxf_JacobiTriangularSolve_spmv1(jxf_ParCSRMatrix *A,
//                      jxf_ParVector    *f,
//                      jxf_ParVector    *u,
//                      JXF_Int          *perm,
//                      JXF_Int           nLU,
//                      jxf_ParCSRMatrix *L,
//                      JXF_Real         *D,
//                      jxf_ParVector     *D_array,
//                      jxf_ParCSRMatrix *U,
//                      jxf_ParVector    *ftemp,
//                      jxf_ParVector    *utemp,
//                      JXF_Int           maxIter,
//                      JXF_Int           lower_jacobi_iters,
//                      JXF_Int           upper_jacobi_iters)
// {
//    /* Data objects for L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);
//    jxf_Vector    *D_local = jxf_ParVectorLocalVector(D_array);
//    JXF_Real      *D_data  = jxf_VectorData(D_local);
   

//    /* Local variables */
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Real       sum;
//    JXF_Int        i, j, k1, k2, kk,iter;


//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* Initialize iteration to 0 */
//    // JXF_Real  *unew = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//    jxf_Vector  *ytemp = jxf_SeqVectorCreate(nLU);
//    jxf_SeqVectorInitialize(ytemp);
//    JXF_Real  *ytemp_data= jxf_VectorData(ytemp);

//    jxf_Vector  *unew = jxf_SeqVectorCreate(nLU);
//    jxf_SeqVectorInitialize(unew);
//    JXF_Real  *unew_data= jxf_VectorData(unew);

//    // JXF_Real  *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    // memcpy(D_data, D, nLU * sizeof(JXF_Real)); // 只执行一次
//     /* 迭代 maxIter 次 */
//     for (iter = 0; iter < maxIter; iter++) 
//     {
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = ftemp_data[i];
//          ytemp_data[i] = 0;
//       }
//       /* Jacobi iteration loop */
//       for ( kk = 0; kk < lower_jacobi_iters; kk++ )  //lower_jacobi_iters
//       {
//          //jxf_CSRMatrixMatvec_LU(alpha,L_diag, ytemp, beta, ftemp_local);
//          jxf_CSRMatrixMatvec(alpha,L_diag, ytemp, 0, utemp_local);
//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++) 
//          {
//             ytemp_data[i] = ftemp_data[i]+utemp_data[i];
//          }

//       } /* end jacobi loop */

//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = 0;
//          unew_data[i] = 0;
//       }

//       /* Jacobi iteration loop */
//       for ( kk = 0; kk < upper_jacobi_iters; kk++ )  //upper_jacobi_iters
//       {
//          //jxf_CSRMatrixMatvec_LU(alpha,L_diag, utemp_local, beta, ytemp);
//          jxf_CSRMatrixMatvec(alpha,U_diag, utemp_local, 0, unew);
//          #pragma omp parallel for
//          for (i = 0; i < nLU; i ++)
//          {
//             utemp_data[i] = D_data[i] *( ytemp_data[i]+unew_data[i]);
//          }
//       } /* end jacobi loop */

//       jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//       jxf_ParVectorAxpy(beta, utemp, u);
//    }

//    // free(u_new);
//    //free(ytemp_data);
//    jxf_SeqVectorDestroy(ytemp); 
//    jxf_SeqVectorDestroy(unew); 

//    return jxf_error_flag;
// }


JXF_Int
jxf_par_Block_Jacobi_TriangularSolve_B(jxf_ParBSRMatrix *A,
                     jxf_ParVector    *f,
                     jxf_ParVector    *u,
                     JXF_Int           nLU,
                     jxf_ParBSRMatrix *L,
                     jxf_ParVector     *D_vector,
                     jxf_ParBSRMatrix *U,
                     jxf_ParVector    *ftemp,
                     jxf_ParVector    *utemp,
                     jxf_ParVector    *unew,
                     jxf_ParVector    *ytemp,
                     JXF_Int           maxIter,
                     JXF_Int           lower_jacobi_iters,
                     JXF_Int           upper_jacobi_iters)
{
   MPI_Comm         comm     = jxf_ParBSRMatrixComm(A); 

   JXF_Real       alpha       = -1.0;
   JXF_Real       beta        = 1.0;
   JXF_Int        i, kk, iter, r, j, k;
   JXF_Int        my_id_tri;
   MPI_Comm_rank(comm, &my_id_tri);

   JXF_Int block_size = jxf_ParBSRMatrixBlockSize(A);
   JXF_Int bnnz = block_size * block_size;
   JXF_Int n_scalar = nLU * block_size;

   /* compute residual: ftemp = f - A*u */
   jxf_ParVectorCopy(f, ftemp);
   jxf_ParBSRMatrixMatvec(alpha, A, u, beta, ftemp);

   jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
   jxf_Vector *D_local     = jxf_ParVectorLocalVector(D_vector);
   jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
   jxf_Vector *ytemp_local = jxf_ParVectorLocalVector(ytemp);
   jxf_Vector *unew_local  = jxf_ParVectorLocalVector(unew);

   JXF_Real *D_data     = jxf_VectorData(D_local);
   JXF_Real *ftemp_data = jxf_VectorData(ftemp_local);
   JXF_Real *utemp_data = jxf_VectorData(utemp_local);
   JXF_Real *ytemp_data = jxf_VectorData(ytemp_local);
   JXF_Real *unew_data  = jxf_VectorData(unew_local);

    for (iter = 0; iter < maxIter; iter++) 
    {
      #pragma omp parallel for
      for (i = 0; i < n_scalar; i++) 
      {
         utemp_data[i] = 0;
         ytemp_data[i] = 0;
         unew_data[i] = 0;
      }

      /* Lower Jacobi: solve L*y = ftemp (local diag only - block-Jacobi) */
      for (kk = 0; kk < lower_jacobi_iters; kk++)
      {
         jxf_BSRMatrixMatvec(alpha, jxf_ParBSRMatrixDiag(L), ytemp_local, 0, utemp_local);
         #pragma omp parallel for
         for (i = 0; i < n_scalar; i++) 
         {
            ytemp_data[i] = utemp_data[i] + ftemp_data[i];
         }
      }

      #pragma omp parallel for
      for (i = 0; i < n_scalar; i++) 
      {
         utemp_data[i] = 0;
      }
      
      /* Upper Jacobi: solve (I+U)*utemp = D⁻¹*ytemp (local diag only - block-Jacobi) */
      for (kk = 0; kk < upper_jacobi_iters; kk++)
      {
          jxf_BSRMatrixMatvec(alpha, jxf_ParBSRMatrixDiag(U), utemp_local, 0, unew_local);
          /* unew = D⁻¹ * (-U * utemp) */
          #pragma omp parallel for private(r, k, j)
          for (r = 0; r < nLU; r++)
          {
              JXF_Real *D_blk = D_data + r * bnnz;
              JXF_Real *unew_blk = unew_data + r * block_size;
              JXF_Real tmp[5];
              for (k = 0; k < block_size; k++)
              {
                  JXF_Real s = 0.0;
                  for (j = 0; j < block_size; j++)
                      s += D_blk[k * block_size + j] * unew_blk[j];
                  tmp[k] = s;
              }
              for (k = 0; k < block_size; k++)
                  unew_blk[k] = tmp[k];
          }
          #pragma omp parallel for private(r, j, k)
          for (r = 0; r < nLU; r++) 
          {
              JXF_Real *D_blk = D_data + r * bnnz;
              for (k = 0; k < block_size; k++)
              {
                  JXF_Real s = 0.0;
                  for (j = 0; j < block_size; j++)
                      s += D_blk[k * block_size + j] * ytemp_data[r * block_size + j];
                  utemp_data[r * block_size + k] = s + unew_data[r * block_size + k];
              }
          }
      }

      jxf_ParBSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
      jxf_ParVectorAxpy(beta, utemp, u);
    }

   return jxf_error_flag;
}


// JXF_Int
// jxf_gsJacobiTriangularSolve(jxf_ParCSRMatrix *A,
//                      jxf_ParVector    *f,
//                      jxf_ParVector    *u,
//                      JXF_Int          *perm,
//                      JXF_Int           nLU,
//                      jxf_ParCSRMatrix *L,
//                      JXF_Real         *D,
//                      jxf_ParCSRMatrix *U,
//                      jxf_ParVector    *ftemp,
//                      jxf_ParVector    *utemp,
//                      JXF_Int           maxIter,
//                      JXF_Int           lower_jacobi_iters,
//                      JXF_Int           upper_jacobi_iters)
// {
//    /* Data objects for L and U */
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);
//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    /* Vectors */
//    jxf_Vector    *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real      *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector    *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real      *ftemp_data  = jxf_VectorData(ftemp_local);

//    /* Local variables */
//    JXF_Real       alpha       = -1.0;
//    JXF_Real       beta        = 1.0;
//    JXF_Real       sum;
//    JXF_Int        i, j, k1, k2, kk,iter;

//    /* Initialize Utemp to zero.
//     * This is necessary for correctness, when we use optimized
//     * vector operations in the case where sizeof(L, D or U) < sizeof(A)
//     */
//    //jxf_ParVectorSetConstantValues( utemp, 0.);
//    /* compute residual */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    /* Initialize iteration to 0 */
//    JXF_Real  *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//        /* 迭代 maxIter 次 */
//     for (iter = 0; iter < maxIter; iter++) 
//     {
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) 
//       {
//          utemp_data[i] = 0.0;
//          ytemp_data[i] = 0.0;
//       }
//       /* Jacobi iteration loop */
//       for ( kk = 0; kk < lower_jacobi_iters; kk++ )  //lower_jacobi_iters
//       {
//          #pragma omp parallel for private(i,j, k1, k2, sum)
//          for ( i = 0; i< nLU; i++ )
//          {
//             sum = 0.0;
//             k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += L_diag_data[j] * ytemp_data[L_diag_j[j]];
//             }
//             ytemp_data[i] = ftemp_data[i] - sum;
//          }
//       } /* end jacobi loop */
//       /* Jacobi iteration loop */
//       for ( kk = 0; kk < upper_jacobi_iters; kk++ )  //upper_jacobi_iters
//       {
//          #pragma omp parallel for private(i,j, k1, k2, sum)
//          /* u^{k+1} = f - Uu^k */
//          for ( i = nLU-1; i >= 0; i-- )
//          {
//             sum = 0.0;
//             k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
//             for (j = k1; j < k2; j++)
//             {
//                sum += U_diag_data[j] * utemp_data[U_diag_j[j]];
//             }
//             utemp_data[i] = D[i] * (ytemp_data[i] - sum);
//          }
//       } /* end jacobi loop */

//       jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//       jxf_ParVectorAxpy(beta, utemp, u);
//    }

//    free(ytemp_data);

//    return jxf_error_flag;
// }

// JXF_Int
// jxf_JGS_TriangularSolve(jxf_ParCSRMatrix *A,
//                        jxf_ParVector    *f,
//                        jxf_ParVector    *u,
//                        JXF_Int          *perm,
//                        JXF_Int           nLU,
//                        jxf_ParCSRMatrix *L,
//                        JXF_Real         *D,
//                        jxf_ParCSRMatrix *U,
//                        jxf_ParVector    *ftemp,
//                        jxf_ParVector    *utemp,
//                        JXF_Int           maxIter,
//                        JXF_Int           lower_jacobi_iters,
//                        JXF_Int           upper_jacobi_iters)
// {
//    jxf_CSRMatrix *L_diag      = jxf_ParCSRMatrixDiag(L);
//    JXF_Real      *L_diag_data = jxf_CSRMatrixData(L_diag);
//    JXF_Int       *L_diag_i    = jxf_CSRMatrixI(L_diag);
//    JXF_Int       *L_diag_j    = jxf_CSRMatrixJ(L_diag);

//    jxf_CSRMatrix *U_diag      = jxf_ParCSRMatrixDiag(U);
//    JXF_Real      *U_diag_data = jxf_CSRMatrixData(U_diag);
//    JXF_Int       *U_diag_i    = jxf_CSRMatrixI(U_diag);
//    JXF_Int       *U_diag_j    = jxf_CSRMatrixJ(U_diag);

//    jxf_Vector *utemp_local = jxf_ParVectorLocalVector(utemp);
//    JXF_Real   *utemp_data  = jxf_VectorData(utemp_local);
//    jxf_Vector *ftemp_local = jxf_ParVectorLocalVector(ftemp);
//    JXF_Real   *ftemp_data  = jxf_VectorData(ftemp_local);

//    JXF_Real alpha = -1.0, beta = 1.0;
//    JXF_Int iter, i, j, k1, k2, tid, T;

//    JXF_Real *u_new      = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));
//    JXF_Real *ytemp_data = (JXF_Real *)malloc(nLU * sizeof(JXF_Real));

//    /* 初始化 */
//    jxf_ParVectorCopy(f, ftemp);
//    jxf_ParCSRMatrixMatvec(alpha, A, u, beta, ftemp);

//    T = jxf_NumThreads();
//    JXF_Int *row_start = (JXF_Int *)malloc((T + 1) * sizeof(JXF_Int));

//    // 行划分
//    JXF_Int n1 = nLU / T;      // 每个线程最少的行数
//    JXF_Int m  = nLU % T;      // 前 m 个线程多一行

//    row_start[0] = 0;
//    for (tid = 0; tid < T; tid++) 
//    {
//        row_start[tid + 1] = row_start[tid] + (tid < m ? n1 + 1 : n1);
//    }

//    for (iter = 0; iter < maxIter; iter++)
//    {
//                // 每轮Jacobi，临时缓冲清零
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) ytemp_data[i] = 0.0;
//       /* 下三角：Jacobi-GS */
//       for (JXF_Int kk = 0; kk < lower_jacobi_iters; kk++)
//       {
//          #pragma omp parallel private( i, j, k1, k2)
//          {
//          // for (tid = 0; tid < T; tid++) {
//             tid = omp_get_thread_num();
//             JXF_Int row_s = row_start[tid];
//             JXF_Int row_e = row_start[tid + 1];

//             for (i = row_s; i < row_e; i++)
//             {
//                JXF_Real sum = 0.0;
//                k1 = L_diag_i[i];
//                k2 = L_diag_i[i + 1];

//                for (j = k1; j < k2; j++)
//                {
//                   JXF_Int col = L_diag_j[j];
//                      sum += L_diag_data[j] * ytemp_data[col]; // GS更新
//                }
//                ytemp_data[i] = ftemp_data[i] - sum;
//             }
//          }

//          #pragma omp parallel for
//          for (i = 0; i < nLU; i++)
//             utemp_data[i] = ytemp_data[i];
//       }

//       /* 上三角 Jacobi */
//                // 每轮Jacobi，临时缓冲清零
//       #pragma omp parallel for
//       for (i = 0; i < nLU; i++) utemp_data[i] = 0.0;

//       for (JXF_Int kk = 0; kk < upper_jacobi_iters; kk++)
//       {
//          #pragma omp parallel private( i, j, k1, k2)
//          {
//          // {
//          // for (tid = 0; tid < T; tid++) {
//             tid = omp_get_thread_num();
//             JXF_Int row_s = row_start[tid];
//             JXF_Int row_e = row_start[tid + 1];

//             for (i = row_e - 1; i >= row_s; i--)
//             {
//                JXF_Real sum = 0.0;
//                k1 = U_diag_i[i];
//                k2 = U_diag_i[i + 1];

//                for (j = k1; j < k2; j++)
//                {
//                   JXF_Int col = U_diag_j[j];
//                   // if (col < row_start[tid])
//                      sum += U_diag_data[j] * utemp_data[col]; // GS更新
//                   // else
//                   //    sum += U_diag_data[j] * utemp_data[col]; // Jacobi部分
//                }
//                utemp_data[i] = D[i] * (ytemp_data[i] - sum);
//             }
//          }

//          // #pragma omp parallel for
//          // for (i = 0; i < nLU; i++)
//          //    utemp_data[i] = u_new[i];
//       }

//       // 残差更新
//       jxf_ParCSRMatrixMatvec(alpha, A, utemp, beta, ftemp);
//       jxf_ParVectorAxpy(beta, utemp, u);
//    }

//    free(u_new);
//    free(ytemp_data);
//    free(row_start);
//    return jxf_error_flag;
// }