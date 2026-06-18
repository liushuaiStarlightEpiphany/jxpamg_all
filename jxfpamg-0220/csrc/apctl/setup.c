//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  setup.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

/*!
 * \fn JXF_Int JXF_3tAPCTLSetup
 * \brief Setup phase of PCTL iteration or preconditioner.
 * \author peghoty
 * \date 2011/09/17
 */
JXF_Int 
JXF_3tAPCTLSetup( JXF_Solver solver, JXF_ParCSRMatrix A )
{
   return( jxf_3tAPCTLSetup( (void *) solver, (jxf_ParCSRMatrix *) A ) );
}

/*!
 * \fn JXF_Int jxf_3tAPCTLSetup
 * \brief Setup phase of PCTL Iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLSetup( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);
   
   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jxf_3tAPCTLSetup_sp(pre_3tapctl_data, A);
   }
   else if (nprocs > 1)
   { 
      jxf_3tAPCTLSetup_mp(pre_3tapctl_data, A); 
   }
   
   return 0;
}

/*!
 * \fn JXF_Int jxf_3tAPCTLSetup_mp
 * \brief Setup phase of PCTL Iteration or preconditioner(for multi-processor case). 
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLSetup_mp( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{   
   JXF_Real starttime = 0.0, endtime = 0.0;

   MPI_Comm comm = jxf_ParCSRMatrixComm(A);
        
   JXF_Int fixit_pctl_R = jxf_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JXF_Int fixit_pctl_E = jxf_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JXF_Int fixit_pctl_I = jxf_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JXF_Int fixit_brlx_R = jxf_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JXF_Int fixit_brlx_E = jxf_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JXF_Int fixit_brlx_I = jxf_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JXF_Real theta_wc_E     = jxf_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JXF_Real threshold_wc_E = jxf_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JXF_Real theta_dd_R     = jxf_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JXF_Real theta_dd_E     = jxf_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JXF_Real theta_dd_I     = jxf_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JXF_Real threshold_dd_R = jxf_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JXF_Real threshold_dd_E = jxf_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JXF_Real threshold_dd_I = jxf_3tAPCTLDataThresholdDDI(pre_3tapctl_data);  

   JXF_Int IS_DD_R;
   JXF_Int IS_DD_E;    
   JXF_Int IS_DD_I;
   
   /* max_iter for each subblock */
   JXF_Int ARR_interp_maxit = jxf_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);     
   JXF_Int AII_interp_maxit = jxf_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);    
   JXF_Int ACC_relax_maxit  = jxf_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data);
   
   /* 需要根据参数 Need_CC 来待定 */ 
   JXF_Int ARR_relax_maxit;
   JXF_Int AEE_relax_maxit;
   JXF_Int AII_relax_maxit;
      
   /* tolerace for each subblock */
   JXF_Real  ARR_interp_tol = jxf_3tAPCTLDataARRInterpTol(pre_3tapctl_data);   
   JXF_Real  AII_interp_tol = jxf_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JXF_Real  ARR_relax_tol  = jxf_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JXF_Real  AEE_relax_tol  = jxf_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JXF_Real  AII_relax_tol  = jxf_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JXF_Real  ACC_relax_tol  = jxf_3tAPCTLDataACCRelaxTol(pre_3tapctl_data); 

   /* solver type for interpolation-building of PRR.  peghoty,2011/10/29 */
   JXF_Int ARR_solver_id = jxf_3tAPCTLDataARRSolverID(pre_3tapctl_data);
   
   /* solver type for PCTL iteration.  peghoty,2011/10/29 */
   JXF_Int ACC_solver_id = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);
   
   /* restart parameters for GMRES solver. peghoty,2011/10/29 */
   JXF_Int ARR_kdim = jxf_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JXF_Int ACC_kdim = jxf_3tAPCTLDataACCKDim(pre_3tapctl_data);  

   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);  
   JXF_Int blocksmooth_type = jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
  
   jxf_ParAMGData   *ARR_amg_solver = NULL;
   jxf_ParAMGData   *AEE_amg_solver = NULL;
   jxf_ParAMGData   *AII_amg_solver = NULL;
   jxf_ParAMGData   *ACC_amg_solver = NULL;

   /* gmres solver data for ARR and ACC. peghoty,2011/10/29 */
   jxf_GMRESData    *ARR_gmres_solver = NULL;
   jxf_GMRESData    *ACC_gmres_solver = NULL; 
   
   jxf_ParCSRMatrix *ARR = NULL;
   jxf_ParCSRMatrix *AEE = NULL;
   jxf_ParCSRMatrix *AII = NULL;
   
   jxf_ParVector    *VRE = NULL;
   jxf_ParVector    *VER = NULL;
   jxf_ParVector    *VEI = NULL;
   jxf_ParVector    *VIE = NULL;      

   jxf_ParCSRMatrix *P   = NULL;  
   jxf_ParCSRMatrix *ACC = NULL; 
     
   jxf_ParVector    *WRR = NULL;
   jxf_ParVector    *WEE = NULL;
   jxf_ParVector    *WII = NULL;
   jxf_ParVector    *WCC = NULL;

   jxf_ParVector    *GCC = NULL;
 
   jxf_ParVector    *RES = NULL;
   jxf_ParVector    *RHS = NULL;
   jxf_ParVector    *JAC = NULL;
   
   jxf_ParCSRMatrix *ARR_all = NULL;
   jxf_ParCSRMatrix *AEE_all = NULL;
   jxf_ParCSRMatrix *AII_all = NULL;  
   jxf_ParVector    *VRE_all = NULL;
   jxf_ParVector    *VER_all = NULL;
   jxf_ParVector    *VEI_all = NULL;
   jxf_ParVector    *VIE_all = NULL;   

   JXF_Int Need_CC; 
   
   JXF_Int ARR_relax_type = RELAX_AMG;
   JXF_Int AEE_relax_type = RELAX_AMG;
   JXF_Int AII_relax_type = RELAX_AMG;      

   jxf_ParVector *PRR = NULL;
   jxf_ParVector *PII = NULL; 

   JXF_Int *row_starts = NULL;
   JXF_Int *col_starts = NULL; 
   
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);
      
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / 3;
   JXF_Int rootid_R, rootid_E, rootid_I;
   JXF_Real temp_adrress = 0.0;

   JXF_Int myid, nprocs;
   JXF_Int np_R = jxf_3tAPCTLDataNpR(pre_3tapctl_data);
   JXF_Int np_E = jxf_3tAPCTLDataNpE(pre_3tapctl_data); 
   JXF_Int np_I = jxf_3tAPCTLDataNpI(pre_3tapctl_data);

   JXF_Int group_num_x = 3;
   JXF_Int group_num_y;
   JXF_Int groupid_x = MPI_UNDEFINED;
   JXF_Int groupid_y = MPI_UNDEFINED; 
   MPI_Comm comm_x, comm_y;    

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (myid < np_R)
   {
      groupid_x = 0;
   }
   else if (myid < np_R + np_E)
   {
      groupid_x = 1;
   }
   else if (myid < np_R + np_E + np_I)
   {
      groupid_x = 2;
   }
   rootid_R = 0; 
   rootid_E = np_R;
   rootid_I = np_R + np_E;    
   jxf_MPI_Comm_split(comm, groupid_x, myid, &comm_x); 
   
   group_num_y = nprocs / group_num_x; 
   groupid_y   = myid % group_num_y;
   jxf_MPI_Comm_split(comm, groupid_y, myid, &comm_y);

   if (print_level == 1 || print_level == 3)
   {       
      starttime = jxf_MPI_Wtime();
   }
      
   if (groupid_x == 1)
   {  
      jxf_MatVecGroupE(comm_x, A, &AEE, &VER, &VEI); 
      Need_CC = jxf_3tAPCTLWeakCouplingE(theta_wc_E, threshold_wc_E, AEE, VER, VEI); 
   }
   jxf_MPI_Bcast(&Need_CC, 1, JXF_MPI_INT, rootid_E, comm); /* 将标志变量广播给所有进程 */
   jxf_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
   if (print_level && myid == 0) jxf_printf(" >> Need_CC = %d\n", Need_CC);
      
  
   if (groupid_x == 0)
   {  
      if ( Need_CC == 1 || (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_GS) )
      {
         jxf_MatVecGroupR(comm_x, A, &ARR, &VRE);
      }
      else
      {
         jxf_MatGroupR(comm_x, A, &ARR);
      }  
   }
   else if (groupid_x == 1)
   {
      if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
      {
         jxf_ParVectorDestroy(VER);
         jxf_ParVectorDestroy(VEI);
      }     
   }
   else if (groupid_x == 2)
   {
      if ( Need_CC == 1 || (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_GS) )
      {
         jxf_MatVecGroupI(comm_x, A, &AII, &VIE);
      }
      else
      {
         jxf_MatGroupI(comm_x, A, &AII);
      } 
   }
   
   if (Need_CC == 1)
   {
      jxf_MatVecGroup2All( comm, groupid_x,  
                          ARR, AEE, AII, VRE, VER, VEI, VIE,
                          &ARR_all, &AEE_all, &AII_all, 
                          &VRE_all, &VER_all, &VEI_all, &VIE_all );    
   }
   
   if (print_level == 1 || print_level == 3)
   {                       
      endtime = jxf_MPI_Wtime();
      jxf_GetWallTime(comm, "APCTL == Get Seven Blocks", starttime, endtime, 0, 3);
   }  

   
   if (groupid_x == 0)
   {  
      IS_DD_R = jxf_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
      jxf_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
      if (print_level && myid == rootid_R) jxf_printf(" >> IS_DD_R = %d\n", IS_DD_R);
      if (IS_DD_R)
      {      
         ARR_relax_type = RELAX_WJACOBI;        
      }
      jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;     
   }
   else if (groupid_x == 1)
   {
      IS_DD_E = jxf_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
      jxf_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
      if (print_level && myid == rootid_E) jxf_printf(" >> IS_DD_E = %d\n", IS_DD_E);
      if (IS_DD_E)
      {      
         AEE_relax_type = RELAX_WJACOBI;        
      }
      jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;        
   }
   else if (groupid_x == 2)
   { 
      IS_DD_I = jxf_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
      jxf_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
      if (print_level && myid == rootid_I) jxf_printf(" >> IS_DD_I = %d\n", IS_DD_I);
      if (IS_DD_I)
      {      
         AII_relax_type = RELAX_WJACOBI;        
      }
      jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;  
   }


   if (Need_CC)
   {
      ARR_relax_maxit = fixit_pctl_R;
      AEE_relax_maxit = fixit_pctl_E;
      AII_relax_maxit = fixit_pctl_I;
   }
   else
   {
      ARR_relax_maxit = fixit_brlx_R;
      AEE_relax_maxit = fixit_brlx_E;
      AII_relax_maxit = fixit_brlx_I;
   }
   jxf_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
   jxf_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
   jxf_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);



   if (Need_CC == 1)
   {

      //======================================================
      //  0 号进程组处理光子系统
      //====================================================== 
   
      if (groupid_x == 0)
      {  
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         WRR->local_vector->data = &temp_adrress;
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetDataOwner(WRR, 0);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WII); 
         jxf_ParVectorSetConstantValues(WII, -1.0);
         jxf_ParVectorSetPartitioningOwner(WII, 0);        
                
         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         if (ARR_solver_id == SOLVER_AMG)
         {
            ARR_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_interp_maxit); 
            jxf_PAMGSetTol(ARR_amg_solver, ARR_interp_tol); 
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout            
            jxf_PAMGSetPrintLevel(ARR_amg_solver, 0);
            jxf_PAMGSetup(ARR_amg_solver, ARR);
  
            if (print_level == 1 || print_level == 3)
            {                    
               endtime = jxf_MPI_Wtime();
               jxf_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
            }
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            ARR_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(ARR_amg_solver, 1); 
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, 0.25); 
            jxf_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout   
            jxf_PAMGSetPrintLevel(ARR_amg_solver, 0);
         
            ARR_gmres_solver = jxf_ParCSRGMRESCreate(comm_x);
            jxf_GMRESSetKDim(ARR_gmres_solver, ARR_kdim);
            jxf_GMRESSetMaxIter(ARR_gmres_solver, ARR_interp_maxit);
            jxf_GMRESSetTol(ARR_gmres_solver, ARR_interp_tol);
            jxf_GMRESSetPrintLevel(ARR_gmres_solver, 0); 
         
            jxf_GMRESSetPrecond( ARR_gmres_solver,
                                jxf_PAMGPrecond,
                                jxf_PAMGSetup,
                                ARR_amg_solver );
                             
            jxf_PAMGSetup(ARR_amg_solver, ARR);
         
            jxf_GMRESSetup(ARR_gmres_solver, ARR, RHS, RHS);
         
            if (print_level == 1 || print_level == 3)
            {                    
               endtime = jxf_MPI_Wtime();
               jxf_GetWallTime(comm_x, "PPCTL == AMGGMRESSetup for ARR", starttime, endtime, 0, 3);
            } 
         } 

         jxf_ParVectorSetConstantValues(WEE, 0.0); 
         jxf_ParVecMul(VRE, WII, RHS);
         
         if (ARR_solver_id == SOLVER_AMG)
         {
            jxf_PAMGSolve(ARR_amg_solver, ARR, RHS, WEE);  
            if ((print_level == 2 || print_level == 3) && myid == rootid_R)
            { 
               jxf_printf(" PCTL-Setup == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
            }
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jxf_GMRESSolve(ARR_gmres_solver, ARR, ARR, RHS, WEE);
            if ((print_level == 2 || print_level == 3) && myid == rootid_R)
            {
               jxf_printf(" PCTL-Setup == ARR AMGGMRES-Iter: %d\n", ARR_gmres_solver->num_iterations);
            }  
         }

         if (test_subls_iter)
         {
            if (ARR_solver_id == SOLVER_AMG)
            {
               jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_amg_solver->num_iterations;
            }
            else if (ARR_solver_id == SOLVER_AMGGMRES)
            {
               jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_gmres_solver->num_iterations;
            }
         }     

         if (ARR_relax_type == RELAX_AMG)
         {
            /* modify parameters for Relaxation */
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
            jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 0.0);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout
            jxf_PAMGSetPrintLevel(ARR_amg_solver, 0);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jxf_PAMGDestroy(ARR_amg_solver);
            ARR_amg_solver = NULL;
            JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
            jxf_ParVectorSetPartitioningOwner(JAC, 0);            
         }
      }
        
      //======================================================
      //  1 号进程组处理电子系统
      //====================================================== 
     
      else if (groupid_x == 1)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetDataOwner(WEE, 0);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         jxf_ParVectorInitialize(WII);
         jxf_ParVectorSetPartitioningOwner(WII, 0);         
   
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jxf_MPI_Wtime();
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
            jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
            jxf_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
            jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jxf_PAMGSetPrintLevel(AEE_amg_solver, 0);
            jxf_PAMGSetup(AEE_amg_solver, AEE); 
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {                      
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }
 
      //======================================================
      //  2 号进程组处理离子系统
      //======================================================  
  
      else if (groupid_x == 2)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetConstantValues(WRR, -1.0);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         WII->local_vector->data = &temp_adrress;
         jxf_ParVectorInitialize(WII);   
         jxf_ParVectorSetDataOwner(WII, 0);
         jxf_ParVectorSetPartitioningOwner(WII, 0);      
                 
         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }   

         AII_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(AII_amg_solver, AII_interp_maxit); 
         jxf_PAMGSetTol(AII_amg_solver, AII_interp_tol); 
         jxf_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
         jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
         jxf_PAMGSetPrintLevel(AII_amg_solver, 0);
         jxf_PAMGSetup(AII_amg_solver, AII); 
  
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }

         jxf_ParVectorSetConstantValues(WEE, 0.0);
        
         jxf_ParVecMul(VIE, WRR, RHS);
         
         jxf_PAMGSolve(AII_amg_solver, AII, RHS, WEE);  
           
         if ((print_level == 2 || print_level == 3) && myid == rootid_I) 
         {
            jxf_printf(" PCTL-Setup == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);   
         }
 
         if (test_subls_iter)
         {
            jxf_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data) = AII_amg_solver->num_iterations;
         }

         if (AII_relax_type == RELAX_AMG)
         {
            jxf_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
            jxf_PAMGSetTol(AII_amg_solver, AII_relax_tol);
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jxf_PAMGDestroy(AII_amg_solver);
            AII_amg_solver = NULL;  

            JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
            jxf_ParVectorInitialize(JAC); 
            jxf_ParVectorSetPartitioningOwner(JAC, 0);           
         }
      }   
 
   }
   else // if (Need_CC == 0)
   {
   
      //======================================================
      //  0 号进程组处理光子系统
      //====================================================== 
   
      if (groupid_x == 0)
      {         
            RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
            jxf_ParVectorInitialize(RHS);
            jxf_ParVectorSetPartitioningOwner(RHS, 0); 

            WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
            WRR->local_vector->data = &temp_adrress;
            jxf_ParVectorInitialize(WRR);
            jxf_ParVectorSetDataOwner(WRR, 0);
            jxf_ParVectorSetPartitioningOwner(WRR, 0);

            WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
            jxf_ParVectorInitialize(WEE);
            jxf_ParVectorSetPartitioningOwner(WEE, 0);

            WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
            jxf_ParVectorInitialize(WII); 
            jxf_ParVectorSetConstantValues(WII, -1.0); 
            jxf_ParVectorSetPartitioningOwner(WII, 0);       
                
            if (print_level == 1 || print_level == 3)
            {
               starttime = jxf_MPI_Wtime();
            }

            if (ARR_relax_type == RELAX_AMG)
            {
               ARR_amg_solver = jxf_PAMGCreate();
               jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
               jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol); 
               jxf_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
               jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
               jxf_PAMGSetPrintLevel(ARR_amg_solver, 0);
               jxf_PAMGSetup(ARR_amg_solver, ARR);
            } 
            else if (ARR_relax_type == RELAX_WJACOBI)
            {
               JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
               jxf_ParVectorInitialize(JAC); 
               jxf_ParVectorSetPartitioningOwner(JAC, 0);         
            }
  
            if (print_level == 1 || print_level == 3)
            {                    
               endtime = jxf_MPI_Wtime();
               jxf_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
            } 
      }
   
      //======================================================
      //  1 号进程组处理电子系统
      //====================================================== 
     
      else if (groupid_x == 1)
      {
            RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
            jxf_ParVectorInitialize(RHS);
            jxf_ParVectorSetPartitioningOwner(RHS, 0);

            WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
            jxf_ParVectorInitialize(WRR);
            jxf_ParVectorSetPartitioningOwner(WRR, 0);

            WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
            WEE->local_vector->data = &temp_adrress;
            jxf_ParVectorInitialize(WEE);
            jxf_ParVectorSetDataOwner(WEE, 0);
            jxf_ParVectorSetPartitioningOwner(WEE, 0);

            WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
            jxf_ParVectorInitialize(WII);
            jxf_ParVectorSetPartitioningOwner(WII, 0);         
   
            if (print_level == 1 || print_level == 3)
            {      
               starttime = jxf_MPI_Wtime();
            }

            if (AEE_relax_type == RELAX_AMG)
            {
               AEE_amg_solver = jxf_PAMGCreate();
               jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
               jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
               jxf_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
               jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
               jxf_PAMGSetPrintLevel(AEE_amg_solver, 0);
               jxf_PAMGSetup(AEE_amg_solver, AEE); 
            }
            else if (AEE_relax_type == RELAX_WJACOBI)
            {
               JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
               jxf_ParVectorInitialize(JAC);
               jxf_ParVectorSetPartitioningOwner(JAC, 0);          
            }
         
            if (print_level == 1 || print_level == 3)
            {                      
               endtime = jxf_MPI_Wtime();
               jxf_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
            }
      }
   
      //======================================================
      //  2 号进程组处理离子系统
      //======================================================  
    
      else if (groupid_x == 2)
      {
            RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
            jxf_ParVectorInitialize(RHS);
            jxf_ParVectorSetPartitioningOwner(RHS, 0);

            WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
            jxf_ParVectorInitialize(WRR);
            jxf_ParVectorSetConstantValues(WRR, -1.0);
            jxf_ParVectorSetPartitioningOwner(WRR, 0);

            WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
            jxf_ParVectorInitialize(WEE);
            jxf_ParVectorSetPartitioningOwner(WEE, 0);

            WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
            WII->local_vector->data = &temp_adrress;
            jxf_ParVectorInitialize(WII);   
            jxf_ParVectorSetDataOwner(WII, 0);
            jxf_ParVectorSetPartitioningOwner(WII, 0);      
                  
            if (print_level == 1 || print_level == 3)
            {
               starttime = jxf_MPI_Wtime();
            }   

            if (AII_relax_type == RELAX_AMG)
            {         
               AII_amg_solver = jxf_PAMGCreate();
               jxf_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
               jxf_PAMGSetTol(AII_amg_solver, AII_relax_tol); 
               jxf_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
               jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
               jxf_PAMGSetPrintLevel(AII_amg_solver, 0);
               jxf_PAMGSetup(AII_amg_solver, AII); 
            }
            else if (AII_relax_type == RELAX_WJACOBI)
            {
               JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
               jxf_ParVectorInitialize(JAC);
               jxf_ParVectorSetPartitioningOwner(JAC, 0);          
            }         
         
            if (print_level == 1 || print_level == 3)
            {                       
               endtime = jxf_MPI_Wtime();
               jxf_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
            }
      }

   } // end if Need_CC  
    
   jxf_MPI_Barrier(comm); //  各进程(组)同步
    

   if (Need_CC == 1)
   {           
      if (print_level == 1 || print_level == 3)
      {       
         starttime = jxf_MPI_Wtime();
      }
  
      jxf_3tAPCTLParaInterpVec(comm, comm_x, groupid_x, NULL, WEE, &PRR, &PII);

      ACC = jxf_3tApctlCoarseOperator_mp(comm, ARR_all, AEE_all, AII_all, 
                                        VRE_all, VER_all, VEI_all, VIE_all, PRR, PII);

      /* 释放 PRR 和 PII */                                   
      jxf_ParVectorDestroy(PRR);
      jxf_ParVectorDestroy(PII); 
                                        
      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "APCTL ==  Coarse Operator", starttime, endtime, 0, 3);
      } 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jxf_MPI_Wtime();
      }
  
      row_starts = jxf_ParCSRMatrixRowStarts(A);
      col_starts = jxf_ParCSRMatrixRowStarts(ACC); 
      P = jxf_3tApctlInterpolation(comm, comm_x, N, groupid_x, row_starts, col_starts, WEE);                   

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "APCTL ==  Interp Operator", starttime, endtime, 0, 3);
      }

      /* 为生成 ACC 开设的辅助矩阵和向量释放内存 */
      jxf_ParCSRMatrixDestroy(ARR_all);
      jxf_ParCSRMatrixDestroy(AEE_all);
      jxf_ParCSRMatrixDestroy(AII_all);
      jxf_ParVectorDestroy(VRE_all);
      jxf_ParVectorDestroy(VER_all);
      jxf_ParVectorDestroy(VEI_all);
      jxf_ParVectorDestroy(VIE_all);            

      //=============================================================
      //  生成表示粗网格系统右端和解向量的两个并行向量
      //=============================================================

      GCC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ACC));
      jxf_ParVectorInitialize(GCC); 
      jxf_ParVectorSetPartitioningOwner(GCC, 0);    

      WCC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ACC));
      jxf_ParVectorInitialize(WCC);
      jxf_ParVectorSetPartitioningOwner(WCC, 0);

      RES = jxf_ParVectorCreate(comm, N, jxf_ParCSRMatrixRowStarts(A));
      jxf_ParVectorInitialize(RES); 
      jxf_ParVectorSetPartitioningOwner(RES, 0); 

      //================================================================
      // 对粗网格矩阵 ACC 进行 AMG 或 AMGGMRES 的 Setup 过程
      //================================================================ 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jxf_MPI_Wtime();
      }

      if (ACC_solver_id == SOLVER_AMG)
      {
         ACC_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(ACC_amg_solver, ACC_relax_maxit); 
         jxf_PAMGSetTol(ACC_amg_solver, ACC_relax_tol); 
         jxf_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jxf_PAMGSetRhsNrmThreshold(ACC_amg_solver, 1.0); 
         jxf_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jxf_PAMGSetPrintLevel(ACC_amg_solver, 0);
         jxf_PAMGSetup(ACC_amg_solver, ACC); 
         
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGSetup for ACC", starttime, endtime, 0, 3);
         }   
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         ACC_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(ACC_amg_solver, 1); 
         jxf_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jxf_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jxf_PAMGSetPrintLevel(ACC_amg_solver, 0);
         
         ACC_gmres_solver = jxf_ParCSRGMRESCreate(comm);
         jxf_GMRESSetKDim(ACC_gmres_solver, ACC_kdim);
         jxf_GMRESSetMaxIter(ACC_gmres_solver, ACC_relax_maxit);
         jxf_GMRESSetTol(ACC_gmres_solver, ACC_relax_tol);
         jxf_GMRESSetPrintLevel(ACC_gmres_solver, 0); 
         
         jxf_GMRESSetPrecond( ACC_gmres_solver,
                             jxf_PAMGPrecond,
                             jxf_PAMGSetup,
                             ACC_amg_solver );
                             
         jxf_PAMGSetup(ACC_amg_solver, ACC);
         
         jxf_GMRESSetup(ACC_gmres_solver, ACC, WCC, WCC);
      
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGGMRESSetup for ACC", starttime, endtime, 0, 3);
         }   
      }      
   
   } // end if Need_CC

   //=============================================================
   //  填充结构体 pre_3tapctl_data 的部分成员
   //=============================================================
     
   jxf_3tAPCTLDataComm(pre_3tapctl_data)           = comm;  
   jxf_3tAPCTLDataCommX(pre_3tapctl_data)          = comm_x; 
   jxf_3tAPCTLDataCommY(pre_3tapctl_data)          = comm_y;     
   jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data)       = groupid_x;
   jxf_3tAPCTLDataGroupIdY(pre_3tapctl_data)       = groupid_y;

   jxf_3tAPCTLDataA(pre_3tapctl_data)              = A;
   jxf_3tAPCTLDataARR(pre_3tapctl_data)            = ARR;
   jxf_3tAPCTLDataAEE(pre_3tapctl_data)            = AEE;
   jxf_3tAPCTLDataAII(pre_3tapctl_data)            = AII;

   jxf_3tAPCTLDataVRE(pre_3tapctl_data)            = VRE;
   jxf_3tAPCTLDataVER(pre_3tapctl_data)            = VER;
   jxf_3tAPCTLDataVEI(pre_3tapctl_data)            = VEI;
   jxf_3tAPCTLDataVIE(pre_3tapctl_data)            = VIE;
   
   jxf_3tAPCTLDataP(pre_3tapctl_data)              = P;
   jxf_3tAPCTLDataACC(pre_3tapctl_data)            = ACC;

   jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data)   = ARR_amg_solver;  
   jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data)   = AEE_amg_solver;
   jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data)   = AII_amg_solver;
   jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data)   = ACC_amg_solver;
   
   jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = ARR_gmres_solver;
   jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = ACC_gmres_solver;
   
   jxf_3tAPCTLDataWRR(pre_3tapctl_data)            = WRR;
   jxf_3tAPCTLDataWEE(pre_3tapctl_data)            = WEE;
   jxf_3tAPCTLDataWII(pre_3tapctl_data)            = WII;
   jxf_3tAPCTLDataWCC(pre_3tapctl_data)            = WCC;
   jxf_3tAPCTLDataGCC(pre_3tapctl_data)            = GCC;

   jxf_3tAPCTLDataRES(pre_3tapctl_data)            = RES;
   jxf_3tAPCTLDataRHS(pre_3tapctl_data)            = RHS;
   jxf_3tAPCTLDataJAC(pre_3tapctl_data)            = JAC;
                  
   return (0); 
}

/*!
 * \fn JXF_Int jxf_3tAPCTLSetup_sp
 * \brief Setup phase of PCTL Iteration or preconditioner(for single-processor case). 
 * \author peghoty 
 * \date 2011/09/26
 */
JXF_Int
jxf_3tAPCTLSetup_sp( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   JXF_Real starttime = 0.0, endtime = 0.0;

   MPI_Comm comm = jxf_ParCSRMatrixComm(A);

   JXF_Int fixit_pctl_R = jxf_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JXF_Int fixit_pctl_E = jxf_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JXF_Int fixit_pctl_I = jxf_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JXF_Int fixit_brlx_R = jxf_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JXF_Int fixit_brlx_E = jxf_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JXF_Int fixit_brlx_I = jxf_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);
      
   /* max_iter for each subblock */
   JXF_Int ARR_interp_maxit = jxf_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);     
   JXF_Int AII_interp_maxit = jxf_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);     
   JXF_Int ACC_relax_maxit  = jxf_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data); 
   
   /* 需要根据参数 Need_CC 来待定 */ 
   JXF_Int ARR_relax_maxit;
   JXF_Int AEE_relax_maxit;
   JXF_Int AII_relax_maxit;
      
   /* tolerace for each subblock */
   JXF_Real  ARR_interp_tol = jxf_3tAPCTLDataARRInterpTol(pre_3tapctl_data);   
   JXF_Real  AII_interp_tol = jxf_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JXF_Real  ARR_relax_tol  = jxf_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JXF_Real  AEE_relax_tol  = jxf_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JXF_Real  AII_relax_tol  = jxf_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JXF_Real  ACC_relax_tol  = jxf_3tAPCTLDataACCRelaxTol(pre_3tapctl_data); 

   /* solver type for interpolation-building of PRR.  peghoty,2011/10/29 */
   JXF_Int ARR_solver_id = jxf_3tAPCTLDataARRSolverID(pre_3tapctl_data);
   
   /* solver type for PCTL iteration.  peghoty,2011/10/29 */
   JXF_Int ACC_solver_id = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);
   
   /* restart parameters for GMRES solver. peghoty,2011/10/29 */
   JXF_Int ARR_kdim = jxf_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JXF_Int ACC_kdim = jxf_3tAPCTLDataACCKDim(pre_3tapctl_data);  
    
   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);  
   JXF_Int blocksmooth_type = jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
  
   jxf_ParAMGData   *ARR_amg_solver = NULL;
   jxf_ParAMGData   *AEE_amg_solver = NULL;
   jxf_ParAMGData   *AII_amg_solver = NULL;
   jxf_ParAMGData   *ACC_amg_solver = NULL;

   /* gmres solver data for ARR and ACC. peghoty,2011/10/29 */
   jxf_GMRESData    *ARR_gmres_solver = NULL;
   jxf_GMRESData    *ACC_gmres_solver = NULL; 
   
   jxf_ParCSRMatrix *ARR = NULL;
   jxf_ParCSRMatrix *AEE = NULL;
   jxf_ParCSRMatrix *AII = NULL;
   
   jxf_ParVector    *VRE = NULL;
   jxf_ParVector    *VER = NULL;
   jxf_ParVector    *VEI = NULL;
   jxf_ParVector    *VIE = NULL;      

   jxf_ParVector    *PRR = NULL;
   jxf_ParVector    *PII = NULL;
   jxf_ParCSRMatrix *ACC = NULL; 
     
   jxf_ParVector    *WRR = NULL;
   jxf_ParVector    *WEE = NULL;
   jxf_ParVector    *WII = NULL;
   jxf_ParVector    *WCC = NULL;
 
   jxf_ParVector    *RES = NULL;
   jxf_ParVector    *RHS = NULL;
   jxf_ParVector    *JAC = NULL;

   JXF_Int Need_CC; 
   
   JXF_Int ARR_relax_type;
   JXF_Int AEE_relax_type;
   JXF_Int AII_relax_type;      
     
   /* local variables */
   jxf_CSRMatrix  *ARR_s = NULL; 
   jxf_CSRMatrix  *AEE_s = NULL;
   jxf_CSRMatrix  *AII_s = NULL;
   jxf_Vector     *VRE_s = NULL;
   jxf_Vector     *VER_s = NULL;
   jxf_Vector     *VEI_s = NULL;
   jxf_Vector     *VIE_s = NULL;
   
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);   

   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / 3;
   JXF_Int i;
   JXF_Real temp_adrress = 0.0;


   //======================================================
   //  将并行矩阵 A 转化为串行矩阵 A_s, 并从中抓取各子块
   //======================================================

   if (print_level == 1 || print_level == 3)
   {       
      starttime = jxf_MPI_Wtime();
   }
   
   jxf_CSRMatrix *A_s = NULL;
   A_s = jxf_ParCSRMatrixDiag(A);
   jxf_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s); 
   
   if (print_level == 1 || print_level == 3)
   {                       
      endtime = jxf_MPI_Wtime();
      jxf_GetWallTime(comm, "APCTL == Get Seven Blocks", starttime, endtime, 0, 3);
   }  
   

   //===================================================================
   //  检查三个物理量之间的耦合强弱，以及三对角块的强对角占优性
   //===================================================================

   jxf_3tAPCTLWCDD( pre_3tapctl_data, comm, 
                   ARR_s, AEE_s, AII_s, VRE_s, VER_s, VEI_s, VIE_s );

   ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);
   Need_CC        = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);

   if (Need_CC)
   {
      ARR_relax_maxit = fixit_pctl_R;
      AEE_relax_maxit = fixit_pctl_E;
      AII_relax_maxit = fixit_pctl_I;
   }
   else
   {
      ARR_relax_maxit = fixit_brlx_R;
      AEE_relax_maxit = fixit_brlx_E;
      AII_relax_maxit = fixit_brlx_I;
   }
   jxf_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
   jxf_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
   jxf_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);

   ARR = jxf_CSRMatrixToParCSRMatrix_sp(comm, ARR_s);
   AEE = jxf_CSRMatrixToParCSRMatrix_sp(comm, AEE_s);
   AII = jxf_CSRMatrixToParCSRMatrix_sp(comm, AII_s);    

   if ( Need_CC == 1 || (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_GS) )
   { 
      VRE = jxf_VectorToParVector_sp(comm, VRE_s);
      VER = jxf_VectorToParVector_sp(comm, VER_s);
      VEI = jxf_VectorToParVector_sp(comm, VEI_s);
      VIE = jxf_VectorToParVector_sp(comm, VIE_s);      
   }

   RHS = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ARR));
   jxf_ParVectorInitialize(RHS);
   jxf_ParVectorSetPartitioningOwner(RHS, 0);

   WRR = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ARR));
   WRR->local_vector->data = &temp_adrress;
   jxf_ParVectorInitialize(WRR);
   jxf_ParVectorSetDataOwner(WRR, 0);
   jxf_ParVectorSetPartitioningOwner(WRR, 0);

   WEE = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(AEE));
   WEE->local_vector->data = &temp_adrress;
   jxf_ParVectorInitialize(WEE);
   jxf_ParVectorSetDataOwner(WEE, 0);
   jxf_ParVectorSetPartitioningOwner(WEE, 0);

   WII = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(AII));
   WII->local_vector->data = &temp_adrress;
   jxf_ParVectorInitialize(WII);   
   jxf_ParVectorSetDataOwner(WII, 0);
   jxf_ParVectorSetPartitioningOwner(WII, 0);  


   //======================================================
   //                需要进行粗网格校正
   //======================================================
   
   if (Need_CC == 1)
   {      
      PRR = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ARR));
      jxf_ParVectorInitialize(PRR);
      jxf_ParVectorSetPartitioningOwner(PRR, 0); 

      PII = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(AII));
      jxf_ParVectorInitialize(PII);
      jxf_ParVectorSetPartitioningOwner(PII, 0); 

      //----------------------------------------------------//
      //                 Step 1: 处理光子系统                 //
      //----------------------------------------------------// 
       
      if (ARR_solver_id == SOLVER_AMG)
      {  
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jxf_MPI_Wtime();
         } 

         ARR_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_interp_maxit); 
         jxf_PAMGSetTol(ARR_amg_solver, ARR_interp_tol); 
         jxf_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
         jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
         jxf_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout      
         jxf_PAMGSetPrintLevel(ARR_amg_solver, 0);
         jxf_PAMGSetup(ARR_amg_solver, ARR); 

         if (print_level == 1 || print_level == 3)
         {                    
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         } 
      }
      else if (ARR_solver_id == SOLVER_AMGGMRES)
      {
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jxf_MPI_Wtime();
         } 

         ARR_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(ARR_amg_solver, 1); 
         jxf_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
         jxf_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jxf_PAMGSetPrintLevel(ARR_amg_solver, 0);
         
         ARR_gmres_solver = jxf_ParCSRGMRESCreate(comm);
         jxf_GMRESSetKDim(ARR_gmres_solver, ARR_kdim);
         jxf_GMRESSetMaxIter(ARR_gmres_solver, ARR_interp_maxit);
         jxf_GMRESSetTol(ARR_gmres_solver, ARR_interp_tol);
         jxf_GMRESSetPrintLevel(ARR_gmres_solver, 0); 
         
         jxf_GMRESSetPrecond( ARR_gmres_solver,
                             jxf_PAMGPrecond,
                             jxf_PAMGSetup,
                             ARR_amg_solver );
                             
         jxf_PAMGSetup(ARR_amg_solver, ARR);
         
         jxf_GMRESSetup(ARR_gmres_solver, ARR, RHS, RHS);
         
         if (print_level == 1 || print_level == 3)
         {                    
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PPCTL == AMGGMRESSetup for ARR", starttime, endtime, 0, 3);
         } 
      } 

      for (i = 0; i < n; i ++)
      {
         RHS->local_vector->data[i] = - VRE->local_vector->data[i];   
      }
      
      jxf_ParVectorSetConstantValues(PRR, 0.0);
      
      if (ARR_solver_id == SOLVER_AMG)
      { 
         jxf_PAMGSolve(ARR_amg_solver, ARR, RHS, PRR);  
         if (print_level == 2 || print_level == 3)
         { 
            jxf_printf(" PCTL-Setup == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
         }   
      } 
      else if (ARR_solver_id == SOLVER_AMGGMRES)
      {
         jxf_GMRESSolve(ARR_gmres_solver, ARR, ARR, RHS, PRR);
         if (print_level == 2 || print_level == 3)
         {
            jxf_printf(" PCTL-Setup == ARR AMGGMRES-Iter: %d\n", ARR_gmres_solver->num_iterations);
         }  
      }
           
      if (test_subls_iter)
      {
         if (ARR_solver_id == SOLVER_AMG)
         {
            jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_amg_solver->num_iterations;
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_gmres_solver->num_iterations;
         }
      }        

      if (ARR_relax_type == RELAX_AMG)
      {
         jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
         jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
         jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 0.0);
         jxf_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         jxf_PAMGDestroy(ARR_amg_solver);
         ARR_amg_solver = NULL;
         if (!JAC)
         {  
            JAC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ARR));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);  
         }           
      }
  
      //----------------------------------------------------//
      //                 Step 2: 处理电子系统                 //
      //----------------------------------------------------// 

      if (AEE_relax_type == RELAX_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jxf_MPI_Wtime();
         }      
      
         AEE_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
         jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
         jxf_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
         jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
         jxf_PAMGSetPrintLevel(AEE_amg_solver, 0);
         jxf_PAMGSetup(AEE_amg_solver, AEE); 

         if (print_level == 1 || print_level == 3)
         {                      
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }         
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(AEE));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);  
         }
      }

      //----------------------------------------------------//
      //                 Step 3: 处理离子系统                 //
      //----------------------------------------------------// 
                  
      if (print_level == 1 || print_level == 3)
      {
         starttime = jxf_MPI_Wtime();
      }   

      AII_amg_solver = jxf_PAMGCreate();
      jxf_PAMGSetMaxIter(AII_amg_solver, AII_interp_maxit); 
      jxf_PAMGSetTol(AII_amg_solver, AII_interp_tol); 
      jxf_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
      jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
      jxf_PAMGSetPrintLevel(AII_amg_solver, 0);
      jxf_PAMGSetup(AII_amg_solver, AII); 

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
      }

      for (i = 0; i < n; i ++)
      {
         RHS->local_vector->data[i] = - VIE->local_vector->data[i];   
      }
      jxf_ParVectorSetConstantValues(PII, 0.0);
      jxf_PAMGSolve(AII_amg_solver, AII, RHS, PII);  
      if (print_level == 2 || print_level == 3) 
      {
         jxf_printf(" PCTL-Setup == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);   
      }

      if (test_subls_iter)
      {
         jxf_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data) = AII_amg_solver->num_iterations;
      } 

      if (AII_relax_type == RELAX_AMG)
      {
         jxf_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
         jxf_PAMGSetTol(AII_amg_solver, AII_relax_tol);
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         jxf_PAMGDestroy(AII_amg_solver);
         AII_amg_solver = NULL;  
         if (!JAC)
         {
            JAC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(AII));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);  
         }           
      }  
  
   }

   //======================================================
   //                不需要进行粗网格校正
   //======================================================
      
   else // if (Need_CC == 0)
   {
     
      //----------------------------------------------------//
      //                 Step 1: 处理光子系统                 //
      //----------------------------------------------------// 

      if (ARR_relax_type == RELAX_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }
         
         ARR_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
         jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol); 
         jxf_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
         jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
         jxf_PAMGSetPrintLevel(ARR_amg_solver, 0);
         jxf_PAMGSetup(ARR_amg_solver, ARR);

         if (print_level == 1 || print_level == 3)
         {                    
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         }
      } 
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ARR));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);  
         }          
      }
  
      //----------------------------------------------------//
      //                 Step 2: 处理电子系统                 //
      //----------------------------------------------------// 

      if (AEE_relax_type == RELAX_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jxf_MPI_Wtime();
         }
                  
         AEE_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
         jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
         jxf_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
         jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
         jxf_PAMGSetPrintLevel(AEE_amg_solver, 0);
         jxf_PAMGSetup(AEE_amg_solver, AEE); 
         
         if (print_level == 1 || print_level == 3)
         {                      
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }            
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(AEE));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);  
         }          
      }

      //----------------------------------------------------//
      //                 Step 3: 处理离子系统                 //
      //----------------------------------------------------// 

      if (AII_relax_type == RELAX_AMG)
      {  
         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }   
             
         AII_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
         jxf_PAMGSetTol(AII_amg_solver, AII_relax_tol); 
         jxf_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
         jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
         jxf_PAMGSetPrintLevel(AII_amg_solver, 0);
         jxf_PAMGSetup(AII_amg_solver, AII); 
            
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(AII));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);   
         }        
      }         
          
   } // end if Need_CC  


   if (Need_CC == 1)
   {
      //================================================================
      // 生成并行粗网格矩阵 ACC = P^T*A*P
      //================================================================ 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jxf_MPI_Wtime();
      }
         
      ACC = jxf_3tApctlCoarseOperator_sp( comm, ARR_s, AEE_s, AII_s, VRE_s, 
                                         VER_s, VEI_s, VIE_s, PRR, PII );

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "APCTL ==  Coarse Operator", starttime, endtime, 0, 3);
      } 

      //=============================================================
      //  创建 WCC - 粗网格系统(并行)解向量
      //=============================================================   

      WCC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ACC));
      jxf_ParVectorInitialize(WCC);
      jxf_ParVectorSetPartitioningOwner(WCC, 0);

      //=============================================================
      //  创建 RES - 残量
      //=============================================================

      RES = jxf_ParVectorCreate(comm, N, jxf_ParCSRMatrixRowStarts(A));
      jxf_ParVectorInitialize(RES);
      jxf_ParVectorSetPartitioningOwner(RES, 0);
            
      //================================================================
      // 对粗网格矩阵 ACC 进行 AMG 或 AMGGMRES 的 Setup 过程
      //================================================================ 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jxf_MPI_Wtime();
      }
     
      if (ACC_solver_id == SOLVER_AMG)
      {
         ACC_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(ACC_amg_solver, ACC_relax_maxit); 
         jxf_PAMGSetTol(ACC_amg_solver, ACC_relax_tol); 
         jxf_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jxf_PAMGSetRhsNrmThreshold(ACC_amg_solver, 1.0);
         jxf_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout      
         jxf_PAMGSetPrintLevel(ACC_amg_solver, 0);
         jxf_PAMGSetup(ACC_amg_solver, ACC); 
         
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGSetup for ACC", starttime, endtime, 0, 3);
         }  
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         ACC_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(ACC_amg_solver, 1); 
         jxf_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jxf_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jxf_PAMGSetPrintLevel(ACC_amg_solver, 0);
         
         ACC_gmres_solver = jxf_ParCSRGMRESCreate(comm);
         jxf_GMRESSetKDim(ACC_gmres_solver, ACC_kdim);
         jxf_GMRESSetMaxIter(ACC_gmres_solver, ACC_relax_maxit);
         jxf_GMRESSetTol(ACC_gmres_solver, ACC_relax_tol);
         jxf_GMRESSetPrintLevel(ACC_gmres_solver, 0); 
         
         jxf_GMRESSetPrecond( ACC_gmres_solver,
                             jxf_PAMGPrecond,
                             jxf_PAMGSetup,
                             ACC_amg_solver );
                             
         jxf_PAMGSetup(ACC_amg_solver, ACC);
         
         jxf_GMRESSetup(ACC_gmres_solver, ACC, WCC, WCC);
      
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGGMRESSetup for ACC", starttime, endtime, 0, 3);
         }   
      }      

   } // end if Need_CC


   //=============================================================
   //  填充结构体 pre_3tapctl_data 的部分成员
   //=============================================================
     
   jxf_3tAPCTLDataComm(pre_3tapctl_data)           = comm;  

   jxf_3tAPCTLDataA(pre_3tapctl_data)              = A;
   jxf_3tAPCTLDataARR(pre_3tapctl_data)            = ARR;
   jxf_3tAPCTLDataAEE(pre_3tapctl_data)            = AEE;
   jxf_3tAPCTLDataAII(pre_3tapctl_data)            = AII;

   jxf_3tAPCTLDataVRE(pre_3tapctl_data)            = VRE;
   jxf_3tAPCTLDataVER(pre_3tapctl_data)            = VER;
   jxf_3tAPCTLDataVEI(pre_3tapctl_data)            = VEI;
   jxf_3tAPCTLDataVIE(pre_3tapctl_data)            = VIE;
   
   jxf_3tAPCTLDataPRR(pre_3tapctl_data)            = PRR;
   jxf_3tAPCTLDataPII(pre_3tapctl_data)            = PII;
   jxf_3tAPCTLDataACC(pre_3tapctl_data)            = ACC;

   jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data)   = ARR_amg_solver;  
   jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data)   = AEE_amg_solver;
   jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data)   = AII_amg_solver;
   jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data)   = ACC_amg_solver;

   jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = ARR_gmres_solver;
   jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = ACC_gmres_solver;

   jxf_3tAPCTLDataWRR(pre_3tapctl_data)            = WRR;
   jxf_3tAPCTLDataWEE(pre_3tapctl_data)            = WEE;
   jxf_3tAPCTLDataWII(pre_3tapctl_data)            = WII;
   jxf_3tAPCTLDataWCC(pre_3tapctl_data)            = WCC;

   jxf_3tAPCTLDataRES(pre_3tapctl_data)            = RES;
   jxf_3tAPCTLDataRHS(pre_3tapctl_data)            = RHS;
   jxf_3tAPCTLDataJAC(pre_3tapctl_data)            = JAC;
   
   /* 释放辅助矩阵和向量 */
   jxf_TFree(ARR_s);
   jxf_TFree(AEE_s);
   jxf_TFree(AII_s);

   /* Attention: VRE_s, VER_s, VEI_s, VIE_s should be treated accordingly. peghoty, 2011/11/03 */
   if ( Need_CC == 1 || (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_GS) )
   { 
      /* only free (not destroy) VRE_s, VER_s, VEI_s, VIE_s in this case, 
         because their data parts have been contained in VRE, VER, VEI, VIE.
         See the function "jxf_VectorToParVector_sp" for details. */
      jxf_TFree(VRE_s);
      jxf_TFree(VER_s);
      jxf_TFree(VEI_s);
      jxf_TFree(VIE_s);    
   }
   else // if (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD)
   {
      /* destroy VRE_s, VER_s, VEI_s, VIE_s completely in this case,
         because they won't be used any more. */
      jxf_SeqVectorDestroy(VRE_s);
      jxf_SeqVectorDestroy(VER_s);
      jxf_SeqVectorDestroy(VEI_s);
      jxf_SeqVectorDestroy(VIE_s);
   }  
             
   return (0); 
}
