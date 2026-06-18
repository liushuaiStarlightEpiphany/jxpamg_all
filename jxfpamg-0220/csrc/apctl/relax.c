//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  relax.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

/*!
 * \fn JXF_Int jxf_3tAPCTLRelax_GSType_mp
 * \brief (Gauss-Seidel type) Block Relaxation of PCTL iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLRelax_GSType_mp( jxf_3tAPCTLData    *pre_3tapctl_data,
                           jxf_ParCSRMatrix   *A,
                           jxf_ParVector      *g,
                           jxf_ParVector      *w )
{
   MPI_Comm  comm    = jxf_ParCSRMatrixComm(A);
   MPI_Comm  comm_x  = jxf_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm  comm_y  = jxf_3tAPCTLDataCommY(pre_3tapctl_data);
   
   JXF_Int  groupid_x   = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JXF_Int  print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   
   JXF_Int ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JXF_Int AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JXF_Int AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);   

   JXF_Int ARR_relax_maxit  = jxf_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JXF_Int AEE_relax_maxit  = jxf_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JXF_Int AII_relax_maxit  = jxf_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data); 
         
   jxf_ParAMGData   *ARR_amg_solver = jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData   *AEE_amg_solver = jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData   *AII_amg_solver = jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);      
   
   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data); 
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);

   jxf_ParVector    *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector    *VER = jxf_3tAPCTLDataVER(pre_3tapctl_data);
   jxf_ParVector    *VEI = jxf_3tAPCTLDataVEI(pre_3tapctl_data);
   jxf_ParVector    *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);

   jxf_ParVector    *WRR = jxf_3tAPCTLDataWRR(pre_3tapctl_data);
   jxf_ParVector    *WEE = jxf_3tAPCTLDataWEE(pre_3tapctl_data);
   jxf_ParVector    *WII = jxf_3tAPCTLDataWII(pre_3tapctl_data);
   
   jxf_ParVector    *JAC = jxf_3tAPCTLDataJAC(pre_3tapctl_data);
   jxf_ParVector    *RHS = jxf_3tAPCTLDataRHS(pre_3tapctl_data);
   
   /* newly added, peghoty, 2012/02/16 */
   JXF_Int use_fixedmode_R = jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JXF_Int use_fixedmode_E = jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JXF_Int use_fixedmode_I = jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);     
   
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);
   
   /* local variables */
   JXF_Int sweep;
   JXF_Int myid, myid_x;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);

   if (groupid_x == 0)
   {
      WRR->local_vector->data = w->local_vector->data;
   }
   else if (groupid_x == 1)
   {
      WEE->local_vector->data = w->local_vector->data;
   }
   else if (groupid_x == 2)
   {
      WII->local_vector->data = w->local_vector->data;
   }  

   jxf_MPI_Bcast(WRR->local_vector->data, WRR->local_vector->size, JXF_MPI_REAL, 0, comm_y);  
   jxf_MPI_Bcast(WII->local_vector->data, WII->local_vector->size, JXF_MPI_REAL, 2, comm_y); 

   if (groupid_x == 1) 
   {   
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);

      jxf_ParVecZXY(RHS, -1.0, VEI, WII);    
      jxf_ParVecZXY(RHS, -1.0, VER, WRR);  
         
      if (AEE_relax_type == RELAX_AMG)
      {      
         jxf_ParVectorSetConstantValues(WEE, 0.0); 
         
         if (use_fixedmode_E)
            jxf_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
         else     
            jxf_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            if (use_fixedmode_E)
               jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
            else
               jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
         }                 
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
         {
            jxf_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
         }
         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            jxf_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
         } 
      }

      if (test_subls_iter)
      {
         if (AEE_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_E)
               jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            else
               jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         }
      }
   }
   
   jxf_MPI_Barrier(comm);

   
   jxf_MPI_Bcast(WEE->local_vector->data, WEE->local_vector->size, JXF_MPI_REAL, 1, comm_y);  
   
   if (groupid_x == 0)
   {
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);
      jxf_ParVecZXY(RHS, -1.0, VRE, WEE);
      
      if (ARR_relax_type == RELAX_AMG)
      {  
         jxf_ParVectorSetConstantValues(WRR, 0.0);
 
         if (use_fixedmode_R)
            jxf_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
         else          
            jxf_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            
         if ((print_level == 2 || print_level == 3) && myid_x == 0) 
         {
            if (use_fixedmode_R)
               jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
            else
               jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations); 
         }         
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {  
         for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
         {      
            jxf_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
         }
         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            jxf_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
         } 
      }

      if (test_subls_iter)
      {
         if (ARR_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_R)
               jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            else
               jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         }
      }     
   }
   else if (groupid_x == 2)
   {
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);
      jxf_ParVecZXY(RHS, -1.0, VIE, WEE);
      
      if (AII_relax_type == RELAX_AMG)
      {
         jxf_ParVectorSetConstantValues(WII, 0.0);
         
         if (use_fixedmode_I)
            jxf_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
         else           
            jxf_PAMGSolve(AII_amg_solver, AII, RHS, WII);
             
         if ((print_level == 2 || print_level == 3) && myid_x == 0) 
         {
            if (use_fixedmode_I)
               jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
            else
               jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
         }         
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
         {      
            jxf_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC); 
         }
         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            jxf_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
         } 
      }

      if (test_subls_iter)
      {
         if (AII_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_I)
               jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            else
               jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         }
      }       
   }

   jxf_MPI_Barrier(comm);

   return (0);
}

JXF_Int
jxf_3tAPCTLmgRelax_GSType_mp( jxf_3tAPCTLData    *pre_3tapctl_data,
                             jxf_ParCSRMatrix   *A,
                             jxf_ParVector      *g,
                             jxf_ParVector      *w )
{
   MPI_Comm comm   = jxf_ParCSRMatrixComm(A);
   MPI_Comm comm_x = jxf_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jxf_3tAPCTLDataCommY(pre_3tapctl_data);

   JXF_Int ng = jxf_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JXF_Int groupid_x   = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JXF_Int reset_zero  = jxf_3tAPCTLDataResetZero(pre_3tapctl_data);
   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);
 
   JXF_Int ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JXF_Int AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JXF_Int AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   JXF_Int ARR_relax_maxit = jxf_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JXF_Int AEE_relax_maxit = jxf_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JXF_Int AII_relax_maxit = jxf_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data);

   jxf_ParAMGData *ARR_amg_solver = jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData *AEE_amg_solver = jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData *AII_amg_solver = jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);

   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data);
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);

   jxf_ParVector *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);

   jxf_ParVector *WRR = jxf_3tAPCTLDataWRR(pre_3tapctl_data);
   jxf_ParVector *WEE = jxf_3tAPCTLDataWEE(pre_3tapctl_data);
   jxf_ParVector *WII = jxf_3tAPCTLDataWII(pre_3tapctl_data);

   jxf_ParVector *JAC = jxf_3tAPCTLDataJAC(pre_3tapctl_data);
   jxf_ParVector *RHS = jxf_3tAPCTLDataRHS(pre_3tapctl_data);

   /* newly added, peghoty, 2012/02/16 */
   JXF_Int use_fixedmode_R = jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JXF_Int use_fixedmode_E = jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JXF_Int use_fixedmode_I = jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);

   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);
 
   /* local variables */
   JXF_Int sweep, myid, myid_x;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);

   if (groupid_x == ng)
   {
      WEE->local_vector->data = w->local_vector->data;
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);
      if (AEE_relax_type == RELAX_AMG)
      {
         if (reset_zero) jxf_ParVectorSetConstantValues(WEE, 0.0);
         if (use_fixedmode_E)
         {
            jxf_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
         }
         else
         {
            jxf_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            if (use_fixedmode_E)
            {
               jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
            }
            else
            {
               jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
            }
         }
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
         {
            jxf_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            jxf_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
         }
      }

      if (test_subls_iter)
      {
         if (AEE_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_E)
            {
               jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            }
            else
            {
               jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         }
      }
   }

   jxf_MPI_Barrier(comm);

   jxf_MPI_Bcast(WEE->local_vector->data, WEE->local_vector->size, JXF_MPI_REAL, ng, comm_y);

   if (groupid_x < ng)
   {
      WRR->local_vector->data = w->local_vector->data;
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);
      jxf_ParVecZXY(RHS, -1.0, VRE, WEE);
      if (ARR_relax_type == RELAX_AMG)
      {
         if (reset_zero) jxf_ParVectorSetConstantValues(WRR, 0.0);
         if (use_fixedmode_R)
         {
            jxf_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
         }
         else
         {
            jxf_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            if (use_fixedmode_R)
            {
               jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
            }
            else
            {
               jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
            }
         }
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
         {
            jxf_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            jxf_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
         }
      }

      if (test_subls_iter)
      {
         if (ARR_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_R)
            {
               jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            }
            else
            {
               jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
            }
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         }
      }
   }
   else if (groupid_x == ng+1)
   {
      WII->local_vector->data = w->local_vector->data;
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);
      jxf_ParVecZXY(RHS, -1.0, VIE, WEE);
      if (AII_relax_type == RELAX_AMG)
      {
         if (reset_zero) jxf_ParVectorSetConstantValues(WII, 0.0);
         if (use_fixedmode_I)
         {
            jxf_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
         }
         else
         {
            jxf_PAMGSolve(AII_amg_solver, AII, RHS, WII);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            if (use_fixedmode_I)
            {
               jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
            }
            else
            {
               jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
            }
         }
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
         {
            jxf_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            jxf_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
         }
      }

      if (test_subls_iter)
      {
         if (AII_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_I)
            {
               jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            }
            else
            {
               jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
            }
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         }
      }
   }

   jxf_MPI_Barrier(comm);

   return (0);
}

/*!
 * \fn JXF_Int jxf_3tAPCTLRelax_BDType_mp
 * \brief (Block Diagonal type) Block Relaxation of PCTL iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/22
 */
JXF_Int
jxf_3tAPCTLRelax_BDType_mp( jxf_3tAPCTLData    *pre_3tapctl_data,
                           jxf_ParCSRMatrix   *A,
                           jxf_ParVector      *g,
                           jxf_ParVector      *w )
{
   MPI_Comm  comm    = jxf_ParCSRMatrixComm(A);
   MPI_Comm  comm_x  = jxf_3tAPCTLDataCommX(pre_3tapctl_data);
   
   JXF_Int  groupid_x   = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JXF_Int  print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   
   JXF_Int ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JXF_Int AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JXF_Int AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);   

   JXF_Int ARR_relax_maxit  = jxf_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JXF_Int AEE_relax_maxit  = jxf_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JXF_Int AII_relax_maxit  = jxf_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data); 
         
   jxf_ParAMGData   *ARR_amg_solver = jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData   *AEE_amg_solver = jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData   *AII_amg_solver = jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);      
   
   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data); 
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);

   jxf_ParVector    *WRR = jxf_3tAPCTLDataWRR(pre_3tapctl_data);
   jxf_ParVector    *WEE = jxf_3tAPCTLDataWEE(pre_3tapctl_data);
   jxf_ParVector    *WII = jxf_3tAPCTLDataWII(pre_3tapctl_data);
   
   jxf_ParVector    *JAC = jxf_3tAPCTLDataJAC(pre_3tapctl_data);
   jxf_ParVector    *RHS = jxf_3tAPCTLDataRHS(pre_3tapctl_data);
   
   /* newly added, peghoty, 2012/02/16 */
   JXF_Int use_fixedmode_R = jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JXF_Int use_fixedmode_E = jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JXF_Int use_fixedmode_I = jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data); 
   
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);   

   /* local variables */
   JXF_Int sweep;
   JXF_Int myid, myid_x;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);

   if (groupid_x == 0)
   {
      WRR->local_vector->data = w->local_vector->data;
   }
   else if (groupid_x == 1)
   {
      WEE->local_vector->data = w->local_vector->data;
   }
   else if (groupid_x == 2)
   {
      WII->local_vector->data = w->local_vector->data;
   }  


   if (groupid_x == 1) 
   {   
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);

      if (AEE_relax_type == RELAX_AMG)
      {      
         jxf_ParVectorSetConstantValues(WEE, 0.0); 
         
         if (use_fixedmode_E)
            jxf_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
         else     
            jxf_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            if (use_fixedmode_E)
               jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
            else
               jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
         }                 
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
         {
            jxf_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
         }
         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            jxf_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
         } 
      }

      if (test_subls_iter)
      {
         if (AEE_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_E)
                jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            else
                jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
             jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         }
      }
   }
   else if (groupid_x == 0)
   {
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);

      if (ARR_relax_type == RELAX_AMG)
      {  
         jxf_ParVectorSetConstantValues(WRR, 0.0);
 
         if (use_fixedmode_R)
            jxf_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
         else          
            jxf_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            
         if ((print_level == 2 || print_level == 3) && myid_x == 0) 
         {
            if (use_fixedmode_R)
               jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
            else
               jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations); 
         }         
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {  
         for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
         {      
            jxf_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
         }
         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            jxf_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
         } 
      }

      if (test_subls_iter)
      {
         if (ARR_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_R)
               jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            else
               jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         }
      }      
   } 
   else if (groupid_x == 2)
   {
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);
      
      if (AII_relax_type == RELAX_AMG)
      {
         jxf_ParVectorSetConstantValues(WII, 0.0);
         
         if (use_fixedmode_I)
            jxf_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
         else           
            jxf_PAMGSolve(AII_amg_solver, AII, RHS, WII);
             
         if ((print_level == 2 || print_level == 3) && myid_x == 0) 
         {
            if (use_fixedmode_I)
               jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
            else
               jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
         }         
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
         {      
            jxf_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC); 
         }
         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            jxf_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
         } 
      }

      if (test_subls_iter)
      {
         if (AII_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_I)
               jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            else
               jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         }
      }        
   }

   jxf_MPI_Barrier(comm);

   return (0);
}

JXF_Int
jxf_3tAPCTLmgRelax_BDType_mp( jxf_3tAPCTLData    *pre_3tapctl_data,
                             jxf_ParCSRMatrix   *A,
                             jxf_ParVector      *g,
                             jxf_ParVector      *w )
{
   MPI_Comm comm   = jxf_ParCSRMatrixComm(A);
   MPI_Comm comm_x = jxf_3tAPCTLDataCommX(pre_3tapctl_data);

   JXF_Int ng = jxf_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JXF_Int groupid_x   = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JXF_Int reset_zero  = jxf_3tAPCTLDataResetZero(pre_3tapctl_data);
   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);

   JXF_Int ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JXF_Int AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JXF_Int AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   JXF_Int ARR_relax_maxit = jxf_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JXF_Int AEE_relax_maxit = jxf_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JXF_Int AII_relax_maxit = jxf_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data);

   jxf_ParAMGData *ARR_amg_solver = jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData *AEE_amg_solver = jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData *AII_amg_solver = jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);

   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data);
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);

   jxf_ParVector *WRR = jxf_3tAPCTLDataWRR(pre_3tapctl_data);
   jxf_ParVector *WEE = jxf_3tAPCTLDataWEE(pre_3tapctl_data);
   jxf_ParVector *WII = jxf_3tAPCTLDataWII(pre_3tapctl_data);

   jxf_ParVector *JAC = jxf_3tAPCTLDataJAC(pre_3tapctl_data);
   jxf_ParVector *RHS = jxf_3tAPCTLDataRHS(pre_3tapctl_data);

   /* newly added, peghoty, 2012/02/16 */
   JXF_Int use_fixedmode_R = jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JXF_Int use_fixedmode_E = jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JXF_Int use_fixedmode_I = jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);

   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   /* local variables */
   JXF_Int sweep, myid, myid_x;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);

   if (groupid_x == ng)
   {
      WEE->local_vector->data = w->local_vector->data;
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);

      if (AEE_relax_type == RELAX_AMG)
      {
         if (reset_zero) jxf_ParVectorSetConstantValues(WEE, 0.0);

         if (use_fixedmode_E)
         {
            jxf_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
         }
         else
         {
            jxf_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            if (use_fixedmode_E)
            {
               jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
            }
            else
            {
               jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
            }
         }
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
         {
            jxf_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            jxf_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
         }
      }

      if (test_subls_iter)
      {
         if (AEE_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_E)
            {
               jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            }
            else
            {
               jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         }
      }
   }
   else if (groupid_x < ng)
   {
      WRR->local_vector->data = w->local_vector->data;
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);

      if (ARR_relax_type == RELAX_AMG)
      {
         if (reset_zero) jxf_ParVectorSetConstantValues(WRR, 0.0);
 
         if (use_fixedmode_R)
         {
            jxf_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
         }
         else
         {
            jxf_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            if (use_fixedmode_R)
            {
               jxf_printf(" APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
            }
            else
            {
               jxf_printf(" APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
            }
         }
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
         {
            jxf_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            jxf_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
         }
      }

      if (test_subls_iter)
      {
         if (ARR_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_R)
            {
               jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            }
            else
            {
               jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
            }
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         }
      }
   }
   else if (groupid_x == ng+1)
   {
      WII->local_vector->data = w->local_vector->data;
      jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);

      if (AII_relax_type == RELAX_AMG)
      {
         if (reset_zero) jxf_ParVectorSetConstantValues(WII, 0.0);

         if (use_fixedmode_I)
         {
            jxf_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
         }
         else
         {
            jxf_PAMGSolve(AII_amg_solver, AII, RHS, WII);
         }

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            if (use_fixedmode_I)
            {
               jxf_printf(" APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
            }
            else
            {
               jxf_printf(" APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
            }
         }
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
         {
            jxf_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC);
         }
         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         {
            jxf_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
         }
      }

      if (test_subls_iter)
      {
         if (AII_relax_type == RELAX_AMG)
         {
            if (use_fixedmode_I)
            {
               jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            }
            else
            {
               jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
            }
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         }
      }
   }

   jxf_MPI_Barrier(comm);

   return (0);
}

/*!
 * \fn JXF_Int jxf_3tAPCTLRelax_GSType_sp
 * \brief (Gauss-Seidel type) Block Relaxation of PCTL iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLRelax_GSType_sp( jxf_3tAPCTLData    *pre_3tapctl_data,
                           jxf_ParCSRMatrix   *A,
                           jxf_ParVector      *g,
                           jxf_ParVector      *w )
{
   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   
   JXF_Int ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JXF_Int AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JXF_Int AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);   

   JXF_Int ARR_relax_maxit  = jxf_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JXF_Int AEE_relax_maxit  = jxf_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JXF_Int AII_relax_maxit  = jxf_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data); 
         
   jxf_ParAMGData   *ARR_amg_solver = jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData   *AEE_amg_solver = jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData   *AII_amg_solver = jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);      
   
   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data); 
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);

   jxf_ParVector    *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector    *VER = jxf_3tAPCTLDataVER(pre_3tapctl_data);
   jxf_ParVector    *VEI = jxf_3tAPCTLDataVEI(pre_3tapctl_data);
   jxf_ParVector    *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);

   jxf_ParVector    *WRR = jxf_3tAPCTLDataWRR(pre_3tapctl_data);
   jxf_ParVector    *WEE = jxf_3tAPCTLDataWEE(pre_3tapctl_data);
   jxf_ParVector    *WII = jxf_3tAPCTLDataWII(pre_3tapctl_data);
   
   jxf_ParVector    *JAC = jxf_3tAPCTLDataJAC(pre_3tapctl_data);
   jxf_ParVector    *RHS = jxf_3tAPCTLDataRHS(pre_3tapctl_data);
   
   JXF_Real *vre_data = jxf_VectorData(jxf_ParVectorLocalVector(VRE));
   JXF_Real *ver_data = jxf_VectorData(jxf_ParVectorLocalVector(VER));
   JXF_Real *vei_data = jxf_VectorData(jxf_ParVectorLocalVector(VEI));
   JXF_Real *vie_data = jxf_VectorData(jxf_ParVectorLocalVector(VIE));
   JXF_Real *rhs_data = jxf_VectorData(jxf_ParVectorLocalVector(RHS));
   
   JXF_Real *g_data   = jxf_VectorData(jxf_ParVectorLocalVector(g));
   JXF_Real *w_data   = jxf_VectorData(jxf_ParVectorLocalVector(w));
   
   JXF_Real *wrr_data = NULL;
   JXF_Real *wee_data = NULL;
   JXF_Real *wii_data = NULL;
   
   /* newly added, peghoty, 2012/02/16 */
   JXF_Int use_fixedmode_R = jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JXF_Int use_fixedmode_E = jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JXF_Int use_fixedmode_I = jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);      

   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   /* local variables */
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / 3;
   JXF_Int nplusn = 2*n;
   JXF_Int i, sweep;

   jxf_VectorData(jxf_ParVectorLocalVector(WRR)) = w_data;
   jxf_VectorData(jxf_ParVectorLocalVector(WEE)) = w_data + n;
   jxf_VectorData(jxf_ParVectorLocalVector(WII)) = w_data + nplusn;
   
   wrr_data = jxf_VectorData(jxf_ParVectorLocalVector(WRR));
   wee_data = jxf_VectorData(jxf_ParVectorLocalVector(WEE));
   wii_data = jxf_VectorData(jxf_ParVectorLocalVector(WII));
 
   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i+n] - vei_data[i]*wii_data[i] - ver_data[i]*wrr_data[i];
   }
    
   jxf_ParVectorSetConstantValues(WEE, 0.0);
         
   if (AEE_relax_type == RELAX_AMG)
   {      
      if (use_fixedmode_E)
         jxf_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
      else     
         jxf_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);

      if (print_level == 2 || print_level == 3)
      { 
         if (use_fixedmode_E)
            jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
         else
            jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
      }                 
   }
   else if (AEE_relax_type == RELAX_WJACOBI)
   {
      for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
      {
         jxf_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
      }
      if (print_level == 2 || print_level == 3)
      { 
         jxf_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (AEE_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_E)
            jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         else
            jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
      }
   }

   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i] - vre_data[i]*wee_data[i];
   }      
   
   jxf_ParVectorSetConstantValues(WRR, 0.0); 
     
   if (ARR_relax_type == RELAX_AMG)
   {  
      if (use_fixedmode_R)
         jxf_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
      else          
         jxf_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            
      if (print_level == 2 || print_level == 3) 
      {
         if (use_fixedmode_R)
            jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
         else
            jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations); 
      }         
   }
   else if (ARR_relax_type == RELAX_WJACOBI)
   {  
      for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
      {      
         jxf_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
      }
      if (print_level == 2 || print_level == 3)
      { 
         jxf_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (ARR_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_R)
            jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         else
            jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
      }
   }      

   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i+nplusn] - vie_data[i]*wee_data[i];
   } 
           
   jxf_ParVectorSetConstantValues(WII, 0.0);
     
   if (AII_relax_type == RELAX_AMG)
   {
      if (use_fixedmode_I)
         jxf_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
      else           
         jxf_PAMGSolve(AII_amg_solver, AII, RHS, WII);
             
      if (print_level == 2 || print_level == 3) 
      {
         if (use_fixedmode_I)
            jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
         else
            jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
      }         
   }
   else if (AII_relax_type == RELAX_WJACOBI)
   {
      for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
      {      
         jxf_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC); 
      }
      if (print_level == 2 || print_level == 3)
      { 
         jxf_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (AII_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_I)
            jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         else
            jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations; 
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
      }
   }        

   return (0);
}

/*!
 * \fn JXF_Int jxf_3tAPCTLRelax_BDType_sp
 * \brief (Block-Diagonal type) Block Relaxation of PCTL iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/27
 */
JXF_Int
jxf_3tAPCTLRelax_BDType_sp( jxf_3tAPCTLData    *pre_3tapctl_data,
                           jxf_ParCSRMatrix   *A,
                           jxf_ParVector      *g,
                           jxf_ParVector      *w )
{
   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   
   JXF_Int ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JXF_Int AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JXF_Int AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);   

   JXF_Int ARR_relax_maxit  = jxf_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JXF_Int AEE_relax_maxit  = jxf_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JXF_Int AII_relax_maxit  = jxf_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data); 
         
   jxf_ParAMGData   *ARR_amg_solver = jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData   *AEE_amg_solver = jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jxf_ParAMGData   *AII_amg_solver = jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);      
   
   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data); 
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);

   jxf_ParVector    *WRR = jxf_3tAPCTLDataWRR(pre_3tapctl_data);
   jxf_ParVector    *WEE = jxf_3tAPCTLDataWEE(pre_3tapctl_data);
   jxf_ParVector    *WII = jxf_3tAPCTLDataWII(pre_3tapctl_data);
   
   jxf_ParVector    *JAC = jxf_3tAPCTLDataJAC(pre_3tapctl_data);
   jxf_ParVector    *RHS = jxf_3tAPCTLDataRHS(pre_3tapctl_data);

   JXF_Real *rhs_data = jxf_VectorData(jxf_ParVectorLocalVector(RHS));
   
   JXF_Real *g_data   = jxf_VectorData(jxf_ParVectorLocalVector(g));
   JXF_Real *w_data   = jxf_VectorData(jxf_ParVectorLocalVector(w));
   
   /* newly added, peghoty, 2012/02/16 */
   JXF_Int use_fixedmode_R = jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JXF_Int use_fixedmode_E = jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JXF_Int use_fixedmode_I = jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);
   
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);      

   /* local variables */
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / 3;
   JXF_Int nplusn = 2*n;
   JXF_Int i, sweep;

   jxf_VectorData(jxf_ParVectorLocalVector(WRR)) = w_data;
   jxf_VectorData(jxf_ParVectorLocalVector(WEE)) = w_data + n;
   jxf_VectorData(jxf_ParVectorLocalVector(WII)) = w_data + nplusn;
   
   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i+n];
   }
    
   jxf_ParVectorSetConstantValues(WEE, 0.0);
         
   if (AEE_relax_type == RELAX_AMG)
   {      
      if (use_fixedmode_E)
         jxf_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
      else     
         jxf_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);

      if (print_level == 2 || print_level == 3)
      { 
         if (use_fixedmode_E)
            jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
         else
            jxf_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
      }                 
   }
   else if (AEE_relax_type == RELAX_WJACOBI)
   {
      for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
      {
         jxf_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
      }
      if (print_level == 2 || print_level == 3)
      { 
         jxf_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (AEE_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_E)
            jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         else
            jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
      }
   }

   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i];
   }      
   
   jxf_ParVectorSetConstantValues(WRR, 0.0); 
     
   if (ARR_relax_type == RELAX_AMG)
   {  
      if (use_fixedmode_R)
         jxf_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
      else          
         jxf_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            
      if (print_level == 2 || print_level == 3) 
      {
         if (use_fixedmode_R)
            jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
         else
            jxf_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations); 
      }         
   }
   else if (ARR_relax_type == RELAX_WJACOBI)
   {  
      for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
      {      
         jxf_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
      }
      if (print_level == 2 || print_level == 3)
      { 
         jxf_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (ARR_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_R)
            jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         else
            jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
      }
   }      

   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i+nplusn];
   } 
           
   jxf_ParVectorSetConstantValues(WII, 0.0);
     
   if (AII_relax_type == RELAX_AMG)
   {
      if (use_fixedmode_I)
         jxf_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
      else           
         jxf_PAMGSolve(AII_amg_solver, AII, RHS, WII);
             
      if (print_level == 2 || print_level == 3) 
      {
         if (use_fixedmode_I)
            jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
         else
            jxf_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
      }         
   }
   else if (AII_relax_type == RELAX_WJACOBI)
   {
      for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
      {      
         jxf_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC); 
      }
      if (print_level == 2 || print_level == 3)
      { 
         jxf_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (AII_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_I)
            jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         else
            jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
      }
   }        

   return (0);
}
