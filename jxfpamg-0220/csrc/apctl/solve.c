//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  solve.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

JXF_Int jxf_total_iter_index = 0;

/*!
 * \fn JXF_Int JXF_3tAPCTLSolve
 * \brief Solve phase of PCTL iteration.
 * \author peghoty
 * \date 2011/09/17
 */
JXF_Int 
JXF_3tAPCTLSolve( JXF_Solver        solver,
                 JXF_ParCSRMatrix  A,
                 JXF_ParVector     b,
                 JXF_ParVector     x )
{
   return( jxf_3tAPCTLSolve( (void *) solver,
                            (jxf_ParCSRMatrix *) A,
                            (jxf_ParVector *) b,
                            (jxf_ParVector *) x ) );
}

/*!
 * \fn JXF_Int JXF_3tAPCTLPrecond
 * \brief PCTL as preconditioner.
 * \author peghoty
 * \date 2011/09/17
 */
JXF_Int 
JXF_3tAPCTLPrecond( JXF_Solver        solver,
                   JXF_ParCSRMatrix  A,
                   JXF_ParVector     b,
                   JXF_ParVector     x  )
{
   return( jxf_3tAPCTLPrecond( (void *) solver,
                              (jxf_ParCSRMatrix *) A,
                              (jxf_ParVector *) b,
                              (jxf_ParVector *) x ) );
} 

JXF_Int 
JXF_3tAPCTLmgPrecond( JXF_Solver        solver,
                     JXF_ParCSRMatrix  A,
                     JXF_ParVector     b,
                     JXF_ParVector     x  )
{
   return( jxf_3tAPCTLmgPrecond( (void *) solver,
                                (jxf_ParCSRMatrix *) A,
                                (jxf_ParVector *) b,
                                (jxf_ParVector *) x ) );
}

JXF_Int 
JXF_3tABSC1mgPrecond( JXF_Solver        solver,
                     JXF_ParCSRMatrix  A,
                     JXF_ParVector     b,
                     JXF_ParVector     x  )
{
   return( jxf_3tABSC1mgPrecond( (void *) solver,
                                (jxf_ParCSRMatrix *) A,
                                (jxf_ParVector *) b,
                                (jxf_ParVector *) x ) );
}

JXF_Int 
JXF_3tABSC2mgPrecond( JXF_Solver        solver,
                     JXF_ParCSRMatrix  A,
                     JXF_ParVector     b,
                     JXF_ParVector     x  )
{
   return( jxf_3tABSC2mgPrecond( (void *) solver,
                                (jxf_ParCSRMatrix *) A,
                                (jxf_ParVector *) b,
                                (jxf_ParVector *) x ) );
}

/*!
 * \fn JXF_Int jxf_3tAPCTLOneIteration
 * \brief One Iteration of PCTL. 
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLOneIteration( jxf_3tAPCTLData    *pre_3tapctl_data,
                        jxf_ParCSRMatrix   *A,
                        jxf_ParVector      *g,
                        jxf_ParVector      *w )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);
   
   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jxf_3tAPCTLOneIteration_sp(pre_3tapctl_data, A, g, w);
   }
   else if (nprocs > 1)
   {
      jxf_3tAPCTLOneIteration_mp(pre_3tapctl_data, A, g, w);
   }
   
   return 0;
}

JXF_Int
jxf_3tAPCTLmgOneIteration( jxf_3tAPCTLData    *pre_3tapctl_data,
                          jxf_ParCSRMatrix   *A,
                          jxf_ParVector      *g,
                          jxf_ParVector      *w )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);

   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jxf_3tAPCTLmgOneIteration_mp(pre_3tapctl_data, A, g, w);
   }

   return 0;
}

JXF_Int
jxf_3tABSC1mgOneIteration( jxf_3tAPCTLData    *pre_3tapctl_data,
                          jxf_ParCSRMatrix   *A,
                          jxf_ParVector      *g,
                          jxf_ParVector      *w )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);

   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jxf_3tABSC1mgOneIteration_mp(pre_3tapctl_data, A, g, w);
   }

   return 0;
}

JXF_Int
jxf_3tABSC2mgOneIteration( jxf_3tAPCTLData    *pre_3tapctl_data,
                          jxf_ParCSRMatrix   *A,
                          jxf_ParVector      *g,
                          jxf_ParVector      *w )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);

   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jxf_3tABSC2mgOneIteration_mp(pre_3tapctl_data, A, g, w);
   }

   return 0;
}

/*!
 * \fn JXF_Int jxf_3tAPCTLOneIteration_mp
 * \brief One Iteration of PCTL. 
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLOneIteration_mp( jxf_3tAPCTLData    *pre_3tapctl_data,
                           jxf_ParCSRMatrix   *A,
                           jxf_ParVector      *g,
                           jxf_ParVector      *w )
{
   MPI_Comm  comm    = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm  comm_x  = jxf_3tAPCTLDataCommX(pre_3tapctl_data);
   
   JXF_Int blocksmooth_type = jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);   

   JXF_Int  Need_CC        = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JXF_Int  debug_flag     = jxf_3tAPCTLDataDebugFlag(pre_3tapctl_data);
   JXF_Int  groupid_x      = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JXF_Int  print_level    = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   JXF_Int  num_relax_pre  = jxf_3tAPCTLDataNumRlxPre(pre_3tapctl_data);
   JXF_Int  num_relax_post = jxf_3tAPCTLDataNumRlxPost(pre_3tapctl_data);
   
   JXF_Int  ACC_solver_id  = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);
   
   jxf_ParAMGData   *ACC_amg_solver   = jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data); 
   jxf_GMRESData    *ACC_gmres_solver = jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data);     

   jxf_ParCSRMatrix *P   = jxf_3tAPCTLDataP(pre_3tapctl_data);
   jxf_ParCSRMatrix *ACC = jxf_3tAPCTLDataACC(pre_3tapctl_data);
   
   jxf_ParVector    *WRR = jxf_3tAPCTLDataWRR(pre_3tapctl_data);
   jxf_ParVector    *WEE = jxf_3tAPCTLDataWEE(pre_3tapctl_data);
   jxf_ParVector    *WII = jxf_3tAPCTLDataWII(pre_3tapctl_data);
   jxf_ParVector    *WCC = jxf_3tAPCTLDataWCC(pre_3tapctl_data);

   jxf_ParVector    *RES = jxf_3tAPCTLDataRES(pre_3tapctl_data);
   jxf_ParVector    *GCC = jxf_3tAPCTLDataGCC(pre_3tapctl_data);
   
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   /* local variables */
   JXF_Int i, myid, myid_x;
   JXF_Int np;
   char MatFile[255];

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &np);
   jxf_MPI_Comm_rank(comm_x, &myid_x);

  
   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================
   
   if (Need_CC == 1)
   {
      for (i = 0; i < num_relax_pre; i ++)
      {
         jxf_3tAPCTLRelax_GSType_mp(pre_3tapctl_data, A, g, w);
         if ((debug_flag == 2) || (debug_flag == 3))
         {
            jxf_sprintf(MatFile,"%s.%02d", "./3t.w", jxf_total_iter_index);
            jxf_ParVectorPrint(w, MatFile);
            jxf_total_iter_index ++;
         }
      }
   }
   else // if (Need_CC == 0)
   {  

      if (blocksmooth_type == BLOCKSMOOTH_GS)
      {
         for (i = 0; i < num_relax_pre; i ++)
         {
            jxf_3tAPCTLRelax_GSType_mp(pre_3tapctl_data, A, g, w);
         }
      }
      else // if (blocksmooth_type == BLOCKSMOOTH_BD)
      {
         for (i = 0; i < num_relax_pre; i ++)
         {
            jxf_3tAPCTLRelax_BDType_mp(pre_3tapctl_data, A, g, w);
         }
      }
   }
   

   if (Need_CC == 1)
   {
      //=============================================================
      //  计算细网格层上的残量，并将其限制到粗网格层
      //=============================================================  
   
      jxf_ParVectorCopy(g, RES);                      // RES = g
      jxf_ParCSRMatrixMatvec(-1.0, A, w, 1.0, RES);   // RES = g - A*w
      jxf_ParCSRMatrixMatvecT(1.0, P, RES, 0.0, GCC); // GCC = P^T*RES
 
 
      //================================================================
      // 求解粗网格方程 ACC*WCC = GCC
      //================================================================  
   
      jxf_ParVectorSetConstantValues(WCC, 0.0); 
      
      if (ACC_solver_id == SOLVER_AMG)
      {  
         jxf_PAMGSolve(ACC_amg_solver, ACC, GCC, WCC);

         if ((print_level == 2 || print_level == 3) && myid == 0)
         { 
            jxf_printf(" APCTL-Solve == ACC AMG-Iter: %d\n", ACC_amg_solver->num_iterations);
         } 
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         jxf_GMRESSolve(ACC_gmres_solver, ACC, ACC, GCC, WCC);
         if ((print_level == 2 || print_level == 3) && myid == 0)
         { 
            jxf_printf(" APCTL-Solve == ACC AMGGMRES-Iter: %d\n", ACC_gmres_solver->num_iterations);
         } 
      }  
      
      if (test_subls_iter)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_amg_solver->num_iterations;
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_gmres_solver->num_iterations;
         }
      }
 
      //================================================================
      //  提升与校正: w = w + P*WCC
      //================================================================  
       
      jxf_ParCSRMatrixMatvec(1.0, P, WCC, 1.0, w);  
   
  
      //=============================================================
      //  后（块）磨光 （磨光序为先电子，后离子和光子）
      //=============================================================  

      for (i = 0; i < num_relax_post; i ++)
      {
         jxf_3tAPCTLRelax_GSType_mp(pre_3tapctl_data, A, g, w);
      } 
   }

   //=============================================================
   //  将辅助数组的数据部分指向空指针，避免 JXF_Real free memory. 
   //============================================================= 
   
   if (groupid_x == 0)
   {
      WRR->local_vector->data = NULL;
   }
   else if (groupid_x == 1)
   {
      WEE->local_vector->data = NULL;
   }
   else if (groupid_x == 2)
   {
      WII->local_vector->data = NULL;
   } 
  
   return (0);
}

JXF_Int
jxf_3tAPCTLmgOneIteration_mp( jxf_3tAPCTLData  *pre_3tapctl_data,
                             jxf_ParCSRMatrix *A,
                             jxf_ParVector    *g,
                             jxf_ParVector    *w )
{
   MPI_Comm  comm    = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm  comm_x  = jxf_3tAPCTLDataCommX(pre_3tapctl_data);

   JXF_Int blocksmooth_type = jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);

   JXF_Int  Need_CC     = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JXF_Int  debug_flag  = jxf_3tAPCTLDataDebugFlag(pre_3tapctl_data);
   JXF_Int  print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);

   JXF_Int  ACC_solver_id  = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   jxf_ParAMGData *ACC_amg_solver   = jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data);
   jxf_GMRESData  *ACC_gmres_solver = jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data);

   jxf_ParCSRMatrix *P   = jxf_3tAPCTLDataP(pre_3tapctl_data);
   jxf_ParCSRMatrix *ACC = jxf_3tAPCTLDataACC(pre_3tapctl_data);
 
   jxf_ParVector *WCC = jxf_3tAPCTLDataWCC(pre_3tapctl_data);

   jxf_ParVector *RES = jxf_3tAPCTLDataRES(pre_3tapctl_data);
   jxf_ParVector *GCC = jxf_3tAPCTLDataGCC(pre_3tapctl_data);

   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   /* local variables */
   JXF_Int myid, myid_x;
   char MatFile[255];

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);

   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================

   if (Need_CC == 1)
   {
      jxf_3tAPCTLmgRelax_GSType_mp(pre_3tapctl_data, A, g, w);
      if ((debug_flag == 2) || (debug_flag == 3))
      {
         jxf_sprintf(MatFile,"%s.%02d", "./mg.w", jxf_total_iter_index);
         jxf_ParVectorPrint(w, MatFile);
         jxf_total_iter_index ++;
      }
   }
   else // if (Need_CC == 0)
   {
      if (blocksmooth_type == BLOCKSMOOTH_GS)
      {
         jxf_3tAPCTLmgRelax_GSType_mp(pre_3tapctl_data, A, g, w);
      }
      else // if (blocksmooth_type == BLOCKSMOOTH_BD)
      {
         jxf_3tAPCTLmgRelax_BDType_mp(pre_3tapctl_data, A, g, w);
      }
   }

   if (Need_CC == 1)
   {
      //=============================================================
      //  计算细网格层上的残量，并将其限制到粗网格层
      //=============================================================

      jxf_ParVectorCopy(g, RES);                      // RES = g
      jxf_ParCSRMatrixMatvec(-1.0, A, w, 1.0, RES);   // RES = g - A*w
      jxf_ParCSRMatrixMatvecT(1.0, P, RES, 0.0, GCC); // GCC = P^T*RES
 
      //================================================================
      // 求解粗网格方程 ACC*WCC = GCC
      //================================================================

      jxf_ParVectorSetConstantValues(WCC, 0.0);

      if (ACC_solver_id == SOLVER_AMG)
      {
         jxf_PAMGSolve(ACC_amg_solver, ACC, GCC, WCC);

         if ((print_level == 2 || print_level == 3) && myid == 0)
         {
            jxf_printf(" APCTL-Solve == ACC AMG-Iter: %d\n", ACC_amg_solver->num_iterations);
         }
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         jxf_GMRESSolve(ACC_gmres_solver, ACC, ACC, GCC, WCC);
         if ((print_level == 2 || print_level == 3) && myid == 0)
         {
            jxf_printf(" APCTL-Solve == ACC AMGGMRES-Iter: %d\n", ACC_gmres_solver->num_iterations);
         }
      }
 
      if (test_subls_iter)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_amg_solver->num_iterations;
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_gmres_solver->num_iterations;
         }
      }
 
      //================================================================
      //  提升与校正: w = w + P*WCC
      //================================================================

      jxf_ParCSRMatrixMatvec(1.0, P, WCC, 1.0, w);
   }

   return (0);
}

JXF_Int
jxf_3tABSC1mgOneIteration_mp( jxf_3tAPCTLData  *pre_3tapctl_data,
                             jxf_ParCSRMatrix *A,
                             jxf_ParVector    *g,
                             jxf_ParVector    *w )
{
   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jxf_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jxf_3tAPCTLDataCommY(pre_3tapctl_data);

   JXF_Int ng = jxf_3tAPCTLDataNumGroup(pre_3tapctl_data);
   JXF_Int groupid_x = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JXF_Int Need_CC = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JXF_Int debug_flag = jxf_3tAPCTLDataDebugFlag(pre_3tapctl_data);
   JXF_Int reset_zero = jxf_3tAPCTLDataResetZero(pre_3tapctl_data);
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

   jxf_ParVector **VER = jxf_3tAPCTLDataVER2(pre_3tapctl_data);
   jxf_ParVector  *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector  *VEI = jxf_3tAPCTLDataVEI(pre_3tapctl_data);
   jxf_ParVector  *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);

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

   JXF_Int np_R = jxf_3tAPCTLDataNpR(pre_3tapctl_data);
   //JXF_Int rootid_E = np_R * ng;

   JXF_Real *tTEMP = jxf_3tAPCTLDatatTEMP(pre_3tapctl_data);

   MPI_Status status;

#if 0
   MPI_Comm ycomm, yycomm;
   MPI_Group world_group, select_group;
   JXF_Int *yrank = NULL;
   JXF_Int gidx, ymyid;
   JXF_Int ygroupid = MPI_UNDEFINED;
#endif

   /* local variables */
   JXF_Int myid, myid_x, sweep, dest, gidx;
   char MatFile[255];

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);

   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================

   if (Need_CC == 1)
   {
      if (groupid_x == ng+1)
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

         jxf_MPI_Send(WII->local_vector->data, WII->local_vector->size, JXF_MPI_REAL, myid-np_R, 123, comm);
      }
      else if (groupid_x == ng)
      {
         jxf_MPI_Recv(WII->local_vector->data, WII->local_vector->size, JXF_MPI_REAL, myid+np_R, 123, comm, &status);

         WEE->local_vector->data = w->local_vector->data;
         jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);
         jxf_ParVecZXY(RHS, -1.0, VEI, WII);

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

#if 0
         dest = myid - rootid_E;
         for (gidx = 0; gidx < ng; gidx ++)
         {
            jxf_MPI_Send(WEE->local_vector->data, WEE->local_vector->size, JXF_MPI_REAL, dest, dest*321, comm);
            dest += np_R;
         }
#endif
      }

   jxf_MPI_Barrier(comm); //  各进程(组)同步
      jxf_MPI_Bcast(WEE->local_vector->data, WEE->local_vector->size, JXF_MPI_REAL, ng, comm_y); // yue: invalid for comm_I

#if 0
      jxf_MPI_Comm_group(comm, &world_group);
      sweep = (ng + 1) * np_R;
      yrank = jxf_CTAlloc(JXF_Int, sweep);
      for (gidx = 0; gidx < sweep; gidx ++) yrank[gidx] = gidx;
      jxf_MPI_Group_incl(world_group, sweep, yrank, &select_group);
      jxf_MPI_Comm_create(comm, select_group, &ycomm);

      jxf_MPI_Comm_rank(ycomm, &ymyid);
      ygroupid = ymyid % np_R;
      jxf_MPI_Comm_split(ycomm, ygroupid, ymyid, &yycomm);
      jxf_MPI_Bcast(WEE->local_vector->data, WEE->local_vector->size, JXF_MPI_REAL, ng, yycomm);

      if (select_group != MPI_GROUP_NULL) jxf_MPI_Group_free(&select_group);
      if (world_group != MPI_GROUP_NULL) jxf_MPI_Group_free(&world_group);
      if (ycomm != MPI_COMM_NULL) jxf_MPI_Comm_free(&ycomm);
      if (yycomm != MPI_COMM_NULL) jxf_MPI_Comm_free(&yycomm);
      jxf_TFree(yrank);
#endif

      if (groupid_x < ng)
      {
         //jxf_MPI_Recv(WEE->local_vector->data, WEE->local_vector->size, JXF_MPI_REAL, myid, myid*321, comm, &status);

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

         //jxf_MPI_Send(WRR->local_vector->data, WRR->local_vector->size, JXF_MPI_REAL, myid, myid*321, comm);
      }

   jxf_MPI_Barrier(comm); //  各进程(组)同步
      jxf_MPI_Gather(WRR->local_vector->data, WRR->local_vector->size, JXF_MPI_REAL,
                                     tTEMP, WRR->local_vector->size, JXF_MPI_REAL, ng, comm_y); // yue: invalid from comm_E and comm_I

      if (groupid_x == ng)
      {
         jxf_ParVectorSetConstantValues(RHS, 0.0);
         dest = 0;
         for (gidx = 0; gidx < ng; gidx ++)
         {
            for (sweep = 0; sweep < WRR->local_vector->size; sweep ++)
            {
               RHS->local_vector->data[sweep] += (VER[gidx]->local_vector->data[sweep] * tTEMP[dest++]);
            }
         }
#if 0
         dest = myid - rootid_E;
         for (gidx = 0; gidx < ng; gidx ++)
         {
            jxf_MPI_Recv(WRR->local_vector->data, WRR->local_vector->size, JXF_MPI_REAL, dest, dest*321, comm, &status);
            for (sweep = 0; sweep < WRR->local_vector->size; sweep ++)
            {
               RHS->local_vector->data[sweep] += (VER[gidx]->local_vector->data[sweep] * WRR->local_vector->data[sweep]);
            }
            dest += np_R;
         }
#endif

         if (AEE_relax_type == RELAX_AMG)
         {
            if (reset_zero) jxf_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_E)
            {
               jxf_PAMGPrecond(AEE_amg_solver, AEE, RHS, WRR);
            }
            else
            {
               jxf_PAMGSolve(AEE_amg_solver, AEE, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_E)
               {
                  jxf_printf(" APCTL-Solve == 2AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jxf_printf(" APCTL-Solve == 2AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
               }
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
            {
               jxf_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jxf_printf(" APCTL-Solve == 2AEE JAC-Iter: %d\n", AEE_relax_maxit);
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

         jxf_ParVectorAxpy(-1.0, WRR, WEE);

         jxf_MPI_Send(WEE->local_vector->data, WEE->local_vector->size, JXF_MPI_REAL, myid+np_R, 113, comm);
      }
      else if (groupid_x == ng+1)
      {
         jxf_MPI_Recv(WEE->local_vector->data, WEE->local_vector->size, JXF_MPI_REAL, myid-np_R, 113, comm, &status);

         jxf_ParVecMul(WEE, VIE, RHS);

         if (AII_relax_type == RELAX_AMG)
         {
            if (reset_zero) jxf_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_I)
            {
               jxf_PAMGPrecond(AII_amg_solver, AII, RHS, WRR);
            }
            else
            {
               jxf_PAMGSolve(AII_amg_solver, AII, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_I)
               {
                  jxf_printf( " APCTL-Solve == 2AII AMG-Iter: %d\n", AII_relax_maxit);
               }
               else
               {
                  jxf_printf( " APCTL-Solve == 2AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
               }
            }
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
            {
               jxf_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jxf_printf(" APCTL-Solve == 2AII JAC-Iter: %d\n", AII_relax_maxit);
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

         jxf_ParVectorAxpy(-1.0, WRR, WII);
      }

      if ((debug_flag == 2) || (debug_flag == 3))
      {
         jxf_sprintf(MatFile,"%s.%02d", "./mg.w", jxf_total_iter_index);
         jxf_ParVectorPrint(w, MatFile);
         jxf_total_iter_index ++;
      }
   }
   else
   {
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
                  jxf_printf(" BD-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jxf_printf(" BD-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
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
               jxf_printf(" BD-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
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
                  jxf_printf(" BD-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
               }
               else
               {
                  jxf_printf(" BD-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
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
               jxf_printf(" BD-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
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
                  jxf_printf(" BD-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
               }
               else
               {
                  jxf_printf(" BD-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
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
               jxf_printf(" BD-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
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
   }

   return (0);
}

JXF_Int
jxf_3tABSC2mgOneIteration_mp( jxf_3tAPCTLData  *pre_3tapctl_data,
                             jxf_ParCSRMatrix *A,
                             jxf_ParVector    *g,
                             jxf_ParVector    *w )
{
   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jxf_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jxf_3tAPCTLDataCommY(pre_3tapctl_data);

   JXF_Int ng = jxf_3tAPCTLDataNumGroup(pre_3tapctl_data);
   JXF_Int groupid_x = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JXF_Int Need_CC = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JXF_Int debug_flag = jxf_3tAPCTLDataDebugFlag(pre_3tapctl_data);
   JXF_Int reset_zero = jxf_3tAPCTLDataResetZero(pre_3tapctl_data);
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

   jxf_ParVector **VER = jxf_3tAPCTLDataVER2(pre_3tapctl_data);
   jxf_ParVector  *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector  *VEI = jxf_3tAPCTLDataVEI(pre_3tapctl_data);
   jxf_ParVector  *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);

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

   JXF_Real *tTEMP = jxf_3tAPCTLDatatTEMP(pre_3tapctl_data);

   /* local variables */
   JXF_Int myid, myid_x, sweep, dest, gidx;
   char MatFile[255];

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);

   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================

   if (Need_CC == 1)
   {
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

   jxf_MPI_Barrier(comm); //  各进程(组)同步
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
         WRR->local_vector->data = w->local_vector->data;
         jxf_SeqVectorCopy(g->local_vector, RHS->local_vector);
         jxf_ParVecZXY(RHS, -1.0, VIE, WEE);

         if (AII_relax_type == RELAX_AMG)
         {
            if (reset_zero) jxf_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_I)
            {
               jxf_PAMGPrecond(AII_amg_solver, AII, RHS, WRR);
            }
            else
            {
               jxf_PAMGSolve(AII_amg_solver, AII, RHS, WRR);
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
               jxf_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
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

   jxf_MPI_Barrier(comm); //  各进程(组)同步
      jxf_MPI_Gather(WRR->local_vector->data, WRR->local_vector->size, JXF_MPI_REAL,
                                     tTEMP, WRR->local_vector->size, JXF_MPI_REAL, ng, comm_y); // yue: invalid from comm_E

      if (groupid_x == ng)
      {
         jxf_ParVectorSetConstantValues(RHS, 0.0);
         dest = 0;
         for (gidx = 0; gidx < ng; gidx ++)
         {
            for (sweep = 0; sweep < WRR->local_vector->size; sweep ++)
            {
               RHS->local_vector->data[sweep] += (VER[gidx]->local_vector->data[sweep] * tTEMP[dest++]);
            }
         }
         dest += WRR->local_vector->size; // skip comm_E
         for (sweep = 0; sweep < WRR->local_vector->size; sweep ++)
         {
            RHS->local_vector->data[sweep] += (VEI->local_vector->data[sweep] * tTEMP[dest++]);
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            if (reset_zero) jxf_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_E)
            {
               jxf_PAMGPrecond(AEE_amg_solver, AEE, RHS, WRR);
            }
            else
            {
               jxf_PAMGSolve(AEE_amg_solver, AEE, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_E)
               {
                  jxf_printf(" APCTL-Solve == 2AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jxf_printf(" APCTL-Solve == 2AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
               }
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
            {
               jxf_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jxf_printf(" APCTL-Solve == 2AEE JAC-Iter: %d\n", AEE_relax_maxit);
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

         jxf_ParVectorAxpy(-1.0, WRR, WEE);
      }

      if ((debug_flag == 2) || (debug_flag == 3))
      {
         jxf_sprintf(MatFile,"%s.%02d", "./mg.w", jxf_total_iter_index);
         jxf_ParVectorPrint(w, MatFile);
         jxf_total_iter_index ++;
      }
   }
   else
   {
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
                  jxf_printf(" BD-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jxf_printf(" BD-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
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
               jxf_printf(" BD-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
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
                  jxf_printf(" BD-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
               }
               else
               {
                  jxf_printf(" BD-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
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
               jxf_printf(" BD-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
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
                  jxf_printf(" BD-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
               }
               else
               {
                  jxf_printf(" BD-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
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
               jxf_printf(" BD-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
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
   }

   return (0);
}

/*!
 * \fn JXF_Int jxf_3tAPCTLOneIteration_sp
 * \brief One Iteration of PCTL(single-processor case). 
 * \author peghoty 
 * \date 2011/09/27
 */
JXF_Int
jxf_3tAPCTLOneIteration_sp( jxf_3tAPCTLData    *pre_3tapctl_data,
                           jxf_ParCSRMatrix   *A,
                           jxf_ParVector      *g,
                           jxf_ParVector      *w )
{
   JXF_Int blocksmooth_type = jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);   

   JXF_Int  Need_CC        = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JXF_Int  print_level    = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   JXF_Int  num_relax_pre  = jxf_3tAPCTLDataNumRlxPre(pre_3tapctl_data);
   JXF_Int  num_relax_post = jxf_3tAPCTLDataNumRlxPost(pre_3tapctl_data);
   
   JXF_Int  ACC_solver_id = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   jxf_ParAMGData   *ACC_amg_solver   = jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data); 
   jxf_GMRESData    *ACC_gmres_solver = jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data);    

   jxf_ParCSRMatrix *ACC = jxf_3tAPCTLDataACC(pre_3tapctl_data);
   jxf_ParVector    *WCC = jxf_3tAPCTLDataWCC(pre_3tapctl_data);
   jxf_ParVector    *RHS = jxf_3tAPCTLDataRHS(pre_3tapctl_data);
   
   jxf_ParVector    *RES = jxf_3tAPCTLDataRES(pre_3tapctl_data);
   
   jxf_ParVector    *PRR = jxf_3tAPCTLDataPRR(pre_3tapctl_data);
   jxf_ParVector    *PII = jxf_3tAPCTLDataPII(pre_3tapctl_data);
    
   JXF_Real *rhs_data = NULL;   
   JXF_Real *prr_data = NULL;
   JXF_Real *pii_data = NULL;
   JXF_Real *res_data = NULL;
   JXF_Real *wcc_data = NULL;
   JXF_Real *w_data   = NULL;

   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);
     
   /* local variables */
   JXF_Int i;
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / 3;
   JXF_Int nplusn = 2*n;   


   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================
   
   if (Need_CC == 1)
   {
      for (i = 0; i < num_relax_pre; i ++)
      {
         jxf_3tAPCTLRelax_GSType_sp(pre_3tapctl_data, A, g, w);
      }
   }
   else // if (Need_CC == 0)
   {  
      if (blocksmooth_type == BLOCKSMOOTH_GS)
      {
         for (i = 0; i < num_relax_pre; i ++)
         {
            jxf_3tAPCTLRelax_GSType_sp(pre_3tapctl_data, A, g, w);
         }
      }
      else // if (blocksmooth_type == BLOCKSMOOTH_BD)
      {
         for (i = 0; i < num_relax_pre; i ++)
         {
            jxf_3tAPCTLRelax_BDType_sp(pre_3tapctl_data, A, g, w);
         }
      }
   }
   
   if (Need_CC == 1)
   {
      //=============================================================
      //  计算细网格层上的残量
      //=============================================================  
   
      jxf_ParVectorCopy(g, RES);                      // RES = g
      jxf_ParCSRMatrixMatvec(-1.0, A, w, 1.0, RES);   // RES = g - A*w
      

      //=============================================================
      //  将残量限制到粗网格层
      //=============================================================  

      rhs_data = jxf_VectorData(jxf_ParVectorLocalVector(RHS));   
      prr_data = jxf_VectorData(jxf_ParVectorLocalVector(PRR));
      pii_data = jxf_VectorData(jxf_ParVectorLocalVector(PII)); 
      res_data = jxf_VectorData(jxf_ParVectorLocalVector(RES));     
      for (i = 0; i < n; i ++)
      {
         rhs_data[i] = res_data[i]*prr_data[i] + res_data[i+n] + res_data[i+nplusn]*pii_data[i];
      }      
 

      //================================================================
      // 求解粗网格方程 ACC*WCC = GCC
      //================================================================  
   
      jxf_ParVectorSetConstantValues(WCC, 0.0);
      
      if (ACC_solver_id == SOLVER_AMG)
      {   
         jxf_PAMGSolve(ACC_amg_solver, ACC, RHS, WCC);

         if (print_level == 2 || print_level == 3)
         { 
            jxf_printf(" APCTL-Solve == ACC AMG-Iter: %d\n", ACC_amg_solver->num_iterations);
         }
      } 
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         jxf_GMRESSolve(ACC_gmres_solver, ACC, ACC, RHS, WCC);
         if (print_level == 2 || print_level == 3)
         { 
            jxf_printf(" APCTL-Solve == ACC AMGGMRES-Iter: %d\n", ACC_gmres_solver->num_iterations);
         } 
      }
               
      if (test_subls_iter)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_amg_solver->num_iterations;
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_gmres_solver->num_iterations;
         }
      }


      //================================================================
      // 提升与校正: w = w + P*WCC
      //================================================================  
 
      wcc_data = jxf_VectorData(jxf_ParVectorLocalVector(WCC));
      w_data   = jxf_VectorData(jxf_ParVectorLocalVector(w));
      for (i = 0; i < n; i ++) 
      {
         w_data[i] += prr_data[i]*wcc_data[i];
         w_data[i+n] += wcc_data[i];
         w_data[i+nplusn] += pii_data[i]*wcc_data[i]; 
      } 
      

      //=============================================================
      //  后（块）磨光 （磨光序为先电子，后离子和光子）
      //=============================================================  

      for (i = 0; i < num_relax_post; i ++)
      {
         jxf_3tAPCTLRelax_GSType_sp(pre_3tapctl_data, A, g, w);
      }
   }
   
   return (0);
}

/*!
 * \fn JXF_Int jxf_3tAPCTLPrecond
 * \brief Solution phase of PCTL as preconditioner. 
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLPrecond( jxf_3tAPCTLData    *pre_3tapctl_data,
                   jxf_ParCSRMatrix   *A,
                   jxf_ParVector      *f,
                   jxf_ParVector      *u )
{   
   JXF_Int  max_iter = jxf_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JXF_Int  num_iter = 0;

   while (num_iter < max_iter)
   {
      /* One iteration of APCTL as preconditioner */
      jxf_3tAPCTLOneIteration(pre_3tapctl_data, A, f, u);
      num_iter ++;
   }

   return (0);
}

JXF_Int
jxf_3tAPCTLmgPrecond( jxf_3tAPCTLData    *pre_3tapctl_data,
                     jxf_ParCSRMatrix   *A,
                     jxf_ParVector      *f,
                     jxf_ParVector      *u )
{   
   JXF_Int  max_iter = jxf_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JXF_Int  num_iter = 0;

   while (num_iter < max_iter)
   {
      /* One iteration of APCTL as preconditioner */
      jxf_3tAPCTLmgOneIteration(pre_3tapctl_data, A, f, u);
      num_iter ++;
   }

   return (0);
}

JXF_Int
jxf_3tABSC1mgPrecond( jxf_3tAPCTLData    *pre_3tapctl_data,
                     jxf_ParCSRMatrix   *A,
                     jxf_ParVector      *f,
                     jxf_ParVector      *u )
{   
   JXF_Int  max_iter = jxf_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JXF_Int  num_iter = 0;

   while (num_iter < max_iter)
   {
      /* One iteration of APCTL as preconditioner */
      jxf_3tABSC1mgOneIteration(pre_3tapctl_data, A, f, u);
      num_iter ++;
   }

   return (0);
}

JXF_Int
jxf_3tABSC2mgPrecond( jxf_3tAPCTLData    *pre_3tapctl_data,
                     jxf_ParCSRMatrix   *A,
                     jxf_ParVector      *f,
                     jxf_ParVector      *u )
{   
   JXF_Int  max_iter = jxf_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JXF_Int  num_iter = 0;

   while (num_iter < max_iter)
   {
      /* One iteration of APCTL as preconditioner */
      jxf_3tABSC2mgOneIteration(pre_3tapctl_data, A, f, u);
      num_iter ++;
   }

   return (0);
}

/*!
 * \fn JXF_Int jxf_3tAPCTLSolve
 * \brief SOLVE phase of PCTL iteration.
 * \author peghoty 
 * \date 2011/09/17
 */ 
JXF_Int
jxf_3tAPCTLSolve( jxf_3tAPCTLData    *pre_3tapctl_data,
                 jxf_ParCSRMatrix   *A,
                 jxf_ParVector      *f,
                 jxf_ParVector      *u )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);
     
   /* members of pre_3tapctl_data */ 
   JXF_Real        tol         = jxf_3tAPCTLDataTol(pre_3tapctl_data);
   JXF_Int           max_iter    = jxf_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JXF_Int           print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);
   jxf_ParVector *R           = jxf_3tAPCTLDataRES(pre_3tapctl_data);

   /* local variables */
   JXF_Int      num_iterations   = 0;
   JXF_Real   conv_factor      = 0.0;
   JXF_Real   res_nrm          = 0.0;
   JXF_Real   res_nrm_init     = 0.0;
   JXF_Real   relative_res_nrm = 0.0;
   JXF_Real   rhs_nrm          = 0.0; 
   JXF_Real   res_nrm_old      = 0.0;
   
   JXF_Int Solve_err_flag = 0;
   JXF_Int myid; 

   jxf_MPI_Comm_rank(comm, &myid);

  /*-----------------------------------------------------------------------
   *  Compute rhs vector and initial fine-grid residual
   *----------------------------------------------------------------------*/
   rhs_nrm = jxf_ParVectorNorm2(f);            // rhs_nrm = ||f||_2
   jxf_ParVectorCopy(f, R);                    // R = f
   jxf_ParCSRMatrixMatvec(-1.0, A, u, 1.0, R); // R = f - Au
   res_nrm = jxf_ParVectorNorm2(R);            // res_nrm = ||f - A*u||_2
   res_nrm_init = res_nrm; 

   if (rhs_nrm)
      relative_res_nrm = res_nrm_init / rhs_nrm;
   else
      relative_res_nrm = res_nrm_init;

   if (print_level && myid == 0)
   {  
      jxf_printf("\n");   
      jxf_printf(" ----------------------------------------------------\n");     
      jxf_printf("    iters    residual      factor      rel_res_nrm   \n");
      jxf_printf(" ----------------------------------------------------\n");
      jxf_printf("      0    %e                %e\n", res_nrm_init,relative_res_nrm);
   }

  /*---------------------------------------------------------------*
   *                 Main Iterative loop                           *
   *---------------------------------------------------------------*/

   while ( relative_res_nrm >= tol && num_iterations < max_iter )
   {
     /*-----------------------------------------------------------
      * One iteration of PCTL
      *---------------------------------------------------------*/
      jxf_3tAPCTLOneIteration(pre_3tapctl_data, A, f, u);

     /*-----------------------------------------------------------
      * Save the res_nrm
      *---------------------------------------------------------*/
      res_nrm_old = res_nrm;

     /*--------------------------------------------------------------------------
      * Compute fine-grid residual and residual norm
      *------------------------------------------------------------------------*/
      jxf_ParVectorCopy(f, R);                      // R:= f           
      jxf_ParCSRMatrixMatvec(-1.0, A, u, 1.0, R);   // R:= R - Au ( = f - Au )
      res_nrm = jxf_ParVectorNorm2(R);              // res_nrm = ||R||_2  

     /*-----------------------------------------------------------
      * Compute convergence factor
      *---------------------------------------------------------*/
      if (res_nrm_old)
         conv_factor = res_nrm / res_nrm_old;
      else
         conv_factor = res_nrm;   

     /*---------------------------------------------
      * Compute relative_res_nrm
      *-------------------------------------------*/ 
      if (rhs_nrm)
         relative_res_nrm = res_nrm / rhs_nrm;
      else
         relative_res_nrm = res_nrm;

      num_iterations ++;

      if (print_level && myid == 0)
      { 
         jxf_printf("    %3d    %e   %f     %e \n",num_iterations,res_nrm,conv_factor,relative_res_nrm);
      }
   } // end while

   if (num_iterations == max_iter && relative_res_nrm >= tol) Solve_err_flag = 1;

   jxf_3tAPCTLSetNumIterations(pre_3tapctl_data, num_iterations);
   jxf_3tAPCTLSetLastRelNrm(pre_3tapctl_data, relative_res_nrm);

   /* Compute average convergence factor */
   if (res_nrm_init)
   {
      conv_factor = pow( (res_nrm / res_nrm_init), (1.0 / ((JXF_Real) num_iterations)) );
   }
   jxf_3tAPCTLSetAveConvFactor(pre_3tapctl_data, conv_factor);

   if (print_level && myid == 0)
   {
      if (Solve_err_flag == 1)
      {
         jxf_printf("\n\n==============================================");
         jxf_printf("\n NOTE: Convergence tolerance was not achieved\n");
         jxf_printf("       within the allowed %d iterations\n",max_iter);
         jxf_printf("==============================================\n");
      }
      jxf_printf("\n >>> \033[34mAverage Convergence Factor\033[00m = %f\n\n", conv_factor);
   }

   return (Solve_err_flag);
}
