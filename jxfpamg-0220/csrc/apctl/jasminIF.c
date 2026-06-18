//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jasminIF.c -- Adaptive PCTL iterattion or preconditioner for 3t linear systems.
 *                PCTL, Physical-variable based Coarsening Two Level iterative Method,
 *                is proposed by Xu Xiaowen, Mo Zeyao etc.
 *
 *  These functions are specially designed for JASMIN interface.  
 *
 *  Date: 2012/02/26
 *
 *  The 3T matrices are of the following form:
 *    /           \      /           \  
 *   | A11 A12  0  |    | Arr Are  0  |  
 *   | A21 A22 A23 | or | Aer Aee Aei |
 *   |  0  A32 A33 |    |  0  Aie Aii |
 *    \           /      \           /   
 * 
 *  Created by peghoty
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

/*!
 * \fn JXF_Int jxf_ApctlKrylov_JASMIN
 * \brief APCTL-Krylov 法的接口程序
 * \note 湘潭大学课题组和北京应用物理与计算数学研究所合作研制
 * \ref [1] 周志阳,徐小文,舒适,冯春生,莫则尧，二维三温辐射扩散方程组两层预条件子的自适应求解，计算物理，2012，已接收.
 *      [2] 周志阳,几种求解辐射扩散问题和弹性问题的代数多层网格法与区域分解法，博士学位论文，湘潭：湘潭大学，2012. 
 * \date 2012/02/25
 */
JXF_Int 
jxf_ApctlKrylov_JASMIN( jxf_ParCSRMatrix     *ARR_p, 
                       jxf_ParCSRMatrix     *AEE_p, 
                       jxf_ParCSRMatrix     *AII_p, 
                       jxf_ParVector        *VRE_p, 
                       jxf_ParVector        *VER_p, 
                       jxf_ParVector        *VEI_p, 
                       jxf_ParVector        *VIE_p, 
                       jxf_ParVector        *fR_p, 
                       jxf_ParVector        *fE_p, 
                       jxf_ParVector        *fI_p,
                       jxf_ParVector        *uR_p, 
                       jxf_ParVector        *uE_p, 
                       jxf_ParVector        *uI_p,                      
                       jxf_APCTLKrylovParam *apctlkrylov_param )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(ARR_p);
   JXF_Int myid, nprocs;
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs > 1 && nprocs % 3 == 0)
   {
      if (!jxf_APCTLKrylovParamISDiagElmFirst(apctlkrylov_param))
      {
         jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(ARR_p));
         jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(AEE_p));
         jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(AII_p));
      }  
      jxf_ApctlKrylov_JASMIN_mp( ARR_p, AEE_p, AII_p, 
                                VRE_p, VER_p, VEI_p, VIE_p, 
                                fR_p, fE_p, fI_p,
                                uR_p, uE_p, uI_p,
                                apctlkrylov_param );
   }
   else if (nprocs == 1)
   {
      if (!jxf_APCTLKrylovParamISDiagElmFirst(apctlkrylov_param))
      {
         jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(ARR_p));
         jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(AEE_p));
         jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(AII_p));
      } 
      
      jxf_ApctlKrylov_JASMIN_sp( ARR_p, AEE_p, AII_p, 
                                VRE_p, VER_p, VEI_p, VIE_p, 
                                fR_p, fE_p, fI_p,
                                uR_p, uE_p, uI_p,
                                apctlkrylov_param );
   }
   else
   {
      if (myid == 0)
      {
         jxf_printf(" Warning: np should be 1 or a multiple of 3, please try again!\n\n");
      }
   } 
  
   return (0);  
}

JXF_Int
jxf_ApctlKrylov_mgJASMIN( jxf_ParCSRMatrix    **ARR_p,
                         jxf_ParCSRMatrix     *AEE_p,
                         jxf_ParCSRMatrix     *AII_p,
                         jxf_ParVector       **VRE_p,
                         jxf_ParVector       **VER_p,
                         jxf_ParVector        *VEI_p,
                         jxf_ParVector        *VIE_p,
                         jxf_ParVector       **fR_p,
                         jxf_ParVector        *fE_p,
                         jxf_ParVector        *fI_p,
                         jxf_ParVector       **uR_p,
                         jxf_ParVector        *uE_p,
                         jxf_ParVector        *uI_p,
                         jxf_APCTLKrylovParam *apctlkrylov_param )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(ARR_p[0]);
   JXF_Int ng = jxf_APCTLKrylovParamNumGroup(apctlkrylov_param);
   JXF_Int myid, nprocs, idx;
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (!jxf_APCTLKrylovParamISDiagElmFirst(apctlkrylov_param))
   {
      for (idx = 0; idx < ng; idx ++) jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(ARR_p[idx]));
      jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(AEE_p));
      jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(AII_p));
   }

   if (nprocs > 1 && nprocs % (ng+2) == 0)
   {
      jxf_ApctlKrylov_mgJASMIN_mp(ARR_p, AEE_p, AII_p, VRE_p, VER_p, VEI_p, VIE_p, fR_p, fE_p, fI_p, uR_p, uE_p, uI_p, apctlkrylov_param);
   }
   else
   {
      if (myid == 0)
      {
         jxf_printf(" Warning: np should be a multiple of %d, please try again!\n\n", ng+2);
      }
   }

   return (0);
}

/*!
 * \fn JXF_Int jxf_ApctlKrylov_JASMIN_mp
 * \brief APCTL-GMRES solvers for multi-processor case.
 * \author peghoty
 * \date 2012/02/25 
 */
JXF_Int 
jxf_ApctlKrylov_JASMIN_mp( jxf_ParCSRMatrix     *ARR_p, 
                          jxf_ParCSRMatrix     *AEE_p, 
                          jxf_ParCSRMatrix     *AII_p, 
                          jxf_ParVector        *VRE_p, 
                          jxf_ParVector        *VER_p, 
                          jxf_ParVector        *VEI_p, 
                          jxf_ParVector        *VIE_p, 
                          jxf_ParVector        *fR_p, 
                          jxf_ParVector        *fE_p, 
                          jxf_ParVector        *fI_p,
                          jxf_ParVector        *uR_p, 
                          jxf_ParVector        *uE_p, 
                          jxf_ParVector        *uI_p,
                          jxf_APCTLKrylovParam *apctlkrylov_param )
{                    
   JXF_Int    solver_id          = jxf_APCTLKrylovParamSolverID(apctlkrylov_param);
   JXF_Real tol                = jxf_APCTLKrylovParamTol(apctlkrylov_param);
   JXF_Int    max_iter           = jxf_APCTLKrylovParamMaxIter(apctlkrylov_param);
   JXF_Int    k_dim              = jxf_APCTLKrylovParamKDim(apctlkrylov_param);
   JXF_Int    is_check_restarted = jxf_APCTLKrylovParamISCheckRestarted(apctlkrylov_param);
   JXF_Int    two_norm           = jxf_APCTLKrylovParamTwoNorm(apctlkrylov_param);
   JXF_Int    print_level        = jxf_APCTLKrylovParamPrintLevel(apctlkrylov_param);  
   JXF_Int    TTest              = jxf_APCTLKrylovParamTTest(apctlkrylov_param);       
   JXF_Int    keepsol            = jxf_APCTLKrylovParamKeepSol(apctlkrylov_param);
   
   JXF_Int    print_level_apctl  = jxf_APCTLKrylovParamPrintLevelAPCTL(apctlkrylov_param);   
   JXF_Int    blocksmooth_type   = jxf_APCTLKrylovParamBlockSmoothType(apctlkrylov_param);   
   JXF_Int    num_relax_pre      = jxf_APCTLKrylovParamNumRelaxPre(apctlkrylov_param);       
   JXF_Int    num_relax_post     = jxf_APCTLKrylovParamNumRelaxPost(apctlkrylov_param);      
   JXF_Int    interp_solver_ARR  = jxf_APCTLKrylovParamInterpSolverARR(apctlkrylov_param);   
   JXF_Int    interp_kdim_ARR    = jxf_APCTLKrylovParamInterpKdimARR(apctlkrylov_param);     
   JXF_Int    interp_maxit_ARR   = jxf_APCTLKrylovParamInterpMaxitARR(apctlkrylov_param);    
   JXF_Int    interp_maxit_AII   = jxf_APCTLKrylovParamInterpMaxitAII(apctlkrylov_param);    
   JXF_Real interp_tol_ARR     = jxf_APCTLKrylovParamInterpTolARR(apctlkrylov_param);      
   JXF_Real interp_tol_AII     = jxf_APCTLKrylovParamInterpTolAII(apctlkrylov_param);      
   JXF_Int    fixit_pctl_R       = jxf_APCTLKrylovParamFixitPCTLR(apctlkrylov_param);        
   JXF_Int    fixit_pctl_E       = jxf_APCTLKrylovParamFixitPCTLE(apctlkrylov_param);        
   JXF_Int    fixit_pctl_I       = jxf_APCTLKrylovParamFixitPCTLI(apctlkrylov_param);        
   JXF_Int    fixit_brlx_R       = jxf_APCTLKrylovParamFixitBrlxR(apctlkrylov_param);        
   JXF_Int    fixit_brlx_E       = jxf_APCTLKrylovParamFixitBrlxE(apctlkrylov_param);        
   JXF_Int    fixit_brlx_I       = jxf_APCTLKrylovParamFixitBrlxI(apctlkrylov_param);        
   JXF_Int    use_fixedmode_R    = jxf_APCTLKrylovParamUseFixedModeR(apctlkrylov_param);     
   JXF_Int    use_fixedmode_E    = jxf_APCTLKrylovParamUseFixedModeE(apctlkrylov_param);     
   JXF_Int    use_fixedmode_I    = jxf_APCTLKrylovParamUseFixedModeI(apctlkrylov_param);     
   JXF_Real theta_wc_E         = jxf_APCTLKrylovParamThetaWCE(apctlkrylov_param);          
   JXF_Real threshold_wc_E     = jxf_APCTLKrylovParamThresholdWCE(apctlkrylov_param);      
   JXF_Real theta_dd_R         = jxf_APCTLKrylovParamThetaDDR(apctlkrylov_param);          
   JXF_Real theta_dd_E         = jxf_APCTLKrylovParamThetaDDE(apctlkrylov_param);          
   JXF_Real theta_dd_I         = jxf_APCTLKrylovParamThetaDDI(apctlkrylov_param);          
   JXF_Real threshold_dd_R     = jxf_APCTLKrylovParamThresholdDDR(apctlkrylov_param);      
   JXF_Real threshold_dd_E     = jxf_APCTLKrylovParamThresholdDDE(apctlkrylov_param);      
   JXF_Real threshold_dd_I     = jxf_APCTLKrylovParamThresholdDDI(apctlkrylov_param); 
   JXF_Int    use_ppctl          = jxf_APCTLKrylovParamUsePPCTL(apctlkrylov_param);
   JXF_Int    test_subls_iter    = jxf_APCTLKrylovParamTestSubLSIter(apctlkrylov_param);     
   JXF_Real tol_default        = 1.0e-6;
   JXF_Int    debug_flag         = jxf_APCTLKrylovParamDebugFlag(apctlkrylov_param);

   JXF_Real strong_threshold = jxf_APCTLKrylovParamStrongThreshold(apctlkrylov_param);
   JXF_Int interp_type = jxf_APCTLKrylovParamInterpType(apctlkrylov_param);
   JXF_Int coarsen_type = jxf_APCTLKrylovParamCoarsenType(apctlkrylov_param);
   JXF_Int agg_num_levels = jxf_APCTLKrylovParamAggNumLevels(apctlkrylov_param);
   JXF_Int coarse_threshold = jxf_APCTLKrylovParamCoarseThreshold(apctlkrylov_param);
   JXF_Int print_level_amg = jxf_APCTLKrylovParamPrintLevelAMG(apctlkrylov_param);
   
   JXF_Int    num_iterations;
   JXF_Real final_res_norm; 
   JXF_Real starttime  = 0.0;
   JXF_Real endtime    = 0.0;  
   JXF_Real starttimeT = 0.0;
   JXF_Real endtimeT   = 0.0; 
   
   JXF_Real cpu_trans_in  = 0.0;
   JXF_Real cpu_trans_out = 0.0;
   JXF_Real cpu_setup     = 0.0;
   JXF_Real cpu_solve     = 0.0;
   JXF_Real cpu_total     = 0.0;         
   
   jxf_ParCSRMatrix  *ARR = NULL; 
   jxf_ParCSRMatrix  *AEE = NULL; 
   jxf_ParCSRMatrix  *AII = NULL; 
   jxf_ParVector     *VRE = NULL; 
   jxf_ParVector     *VER = NULL; 
   jxf_ParVector     *VEI = NULL; 
   jxf_ParVector     *VIE = NULL; 
                   
   jxf_ParCSRMatrix  *par_mat = NULL;
   jxf_ParVector     *par_rhs = NULL;
   jxf_ParVector     *par_sol = NULL;

   JXF_Solver solver  = NULL;
   JXF_Solver precond = NULL;

   JXF_Int myid, nprocs;
   JXF_Int np_R, np_E, np_I;
   JXF_Int group_num_x = 3;
   JXF_Int group_num_y;
   JXF_Int groupid_x = MPI_UNDEFINED;
   JXF_Int groupid_y = MPI_UNDEFINED; 
   MPI_Comm comm = jxf_ParCSRMatrixComm(ARR_p);
   MPI_Comm comm_x, comm_y;
   MPI_Comm comm_bak; 
         
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   np_R = nprocs / 3;
   np_E = np_R;
   np_I = np_R;  

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
   jxf_MPI_Comm_split(comm, groupid_x, myid, &comm_x); 
   jxf_MPI_Comm_dup(comm_x, &comm_bak);

   group_num_y = nprocs / group_num_x; 
   groupid_y   = myid % group_num_y;
   jxf_MPI_Comm_split(comm, groupid_y, myid, &comm_y);

   if (TTest) starttime = jxf_MPI_Wtime();
                
   jxf_ParaDataTrans4ApctlKrylov( comm, comm_x, groupid_x, 
                                 ARR_p, AEE_p, AII_p, 
                                 VRE_p, VER_p, VEI_p, VIE_p,
                                 fR_p, fE_p, fI_p,
                                 uR_p, uE_p, uI_p,
                                 &ARR, &AEE, &AII, &VRE, &VER, &VEI, &VIE,
                                 &par_mat, &par_rhs, &par_sol ); 

   if (TTest)
   {
      endtime = jxf_MPI_Wtime();
      cpu_trans_in = jxf_GetWallTimeMax(comm, starttime, endtime);  
   }     

   if (TTest) starttimeT = jxf_MPI_Wtime();
                                          
   switch (solver_id)
   {

      case 1: 
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jxf_printf("\n Solver: \033[31mAPCTL-CG\033[00m\n\n");
            else
               jxf_printf("\n Solver: \033[31mPPCTL-CG\033[00m\n\n");
         }
         
         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_3tAPCTLDataInitialize(&precond);

         JXF_3tAPCTLSetA(precond, par_mat);
         JXF_3tAPCTLSetARR(precond, ARR);
         JXF_3tAPCTLSetAEE(precond, AEE);
         JXF_3tAPCTLSetAII(precond, AII);
         JXF_3tAPCTLSetVRE(precond, VRE);
         JXF_3tAPCTLSetVER(precond, VER);
         JXF_3tAPCTLSetVEI(precond, VEI);
         JXF_3tAPCTLSetVIE(precond, VIE);
         JXF_3tAPCTLSetARRAll(precond, ARR_p);
         JXF_3tAPCTLSetAEEAll(precond, AEE_p);
         JXF_3tAPCTLSetAIIAll(precond, AII_p);
         JXF_3tAPCTLSetVREAll(precond, VRE_p);
         JXF_3tAPCTLSetVERAll(precond, VER_p);
         JXF_3tAPCTLSetVEIAll(precond, VEI_p);
         JXF_3tAPCTLSetVIEAll(precond, VIE_p);         
         JXF_3tAPCTLSetMaxiter(precond, 1);
         JXF_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JXF_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JXF_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JXF_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JXF_3tAPCTLSetARRSolverID(precond, interp_solver_ARR); /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetACCSolverID(precond, SOLVER_AMG); /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetARRKDim(precond, interp_kdim_ARR); /* peghoty 2011/10/30 */          
         JXF_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JXF_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JXF_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JXF_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JXF_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JXF_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetACCRelaxTol(precond, tol_default);  
         
         JXF_3tAPCTLSetNpR(precond, np_R);
         JXF_3tAPCTLSetNpE(precond, np_E);
         JXF_3tAPCTLSetNpI(precond, np_I); 
         JXF_3tAPCTLSetComm(precond, comm);          
         JXF_3tAPCTLSetCommX(precond, comm_x);
         JXF_3tAPCTLSetCommY(precond, comm_y);
         JXF_3tAPCTLSetGroupIdX(precond, groupid_x);
         JXF_3tAPCTLSetGroupIdY(precond, groupid_y);
  
         JXF_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JXF_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         JXF_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JXF_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JXF_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JXF_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JXF_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JXF_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JXF_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JXF_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JXF_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JXF_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JXF_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JXF_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JXF_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JXF_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JXF_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */ 

         JXF_3tAPCTLSetDebugFlag(precond, debug_flag);
         JXF_3tAPCTLSetStrongThreshold(precond, strong_threshold);
         JXF_3tAPCTLSetInterpType(precond, interp_type);
         JXF_3tAPCTLSetCoarsenType(precond, coarsen_type);
         JXF_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
         JXF_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
         JXF_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);

         JXF_ParCSRPCGCreate(comm, &solver); 
         JXF_PCGSetMaxIter(solver, max_iter);
         JXF_PCGSetTol(solver, tol);
         JXF_PCGSetTwoNorm(solver, two_norm);  // 0: B 范数； 1：l2 范数 
         JXF_PCGSetLogging(solver, 1);
         JXF_PCGSetPrintLevel(solver, print_level);
         
         JXF_PCGSetPrecond( solver,
                           (JXF_PtrToSolverFcn) JXF_3tAPCTLPrecond,
                           (JXF_PtrToSolverFcn) NULL,
                           precond ); 
         
         JXF_3tAPCTLSetup4Jasmin( precond, (JXF_ParCSRMatrix)par_mat );                   
         
         JXF_PCGSetup ( solver, 
                       (JXF_Matrix) par_mat, 
                       (JXF_Vector) par_rhs, 
                       (JXF_Vector) par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_setup = jxf_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jxf_MPI_Wtime();
         
         JXF_PCGSolve ( solver, 
                       (JXF_Matrix) par_mat, // preOperater
                       (JXF_Matrix) par_mat, 
                       (JXF_Vector) par_rhs, 
                       (JXF_Vector) par_sol );  

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_solve = jxf_GetWallTimeMax(comm, starttime, endtime); 
         }

         JXF_PCGGetNumIterations(solver, &num_iterations);
         JXF_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jxf_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jxf_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);

         if (print_level_apctl)
         { 
            JXF_3tAPCTLIterCount(precond);
         }

         JXF_ParCSRPCGDestroy(solver);
      }
      break;
      

      case 2:  // APCTL-GMRES(m)
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jxf_printf("\n Solver: \033[31mAPCTL-GMRES(%d)\033[00m\n\n", k_dim);
            else
               jxf_printf("\n Solver: \033[31mPPCTL-GMRES(%d)\033[00m\n\n", k_dim);
         }
         
         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_3tAPCTLDataInitialize(&precond);
         JXF_3tAPCTLSetA(precond, par_mat);
         JXF_3tAPCTLSetARR(precond, ARR);
         JXF_3tAPCTLSetAEE(precond, AEE);
         JXF_3tAPCTLSetAII(precond, AII);
         JXF_3tAPCTLSetVRE(precond, VRE);
         JXF_3tAPCTLSetVER(precond, VER);
         JXF_3tAPCTLSetVEI(precond, VEI);
         JXF_3tAPCTLSetVIE(precond, VIE);
         JXF_3tAPCTLSetARRAll(precond, ARR_p);
         JXF_3tAPCTLSetAEEAll(precond, AEE_p);
         JXF_3tAPCTLSetAIIAll(precond, AII_p);
         JXF_3tAPCTLSetVREAll(precond, VRE_p);
         JXF_3tAPCTLSetVERAll(precond, VER_p);
         JXF_3tAPCTLSetVEIAll(precond, VEI_p);
         JXF_3tAPCTLSetVIEAll(precond, VIE_p);            
         JXF_3tAPCTLSetMaxiter(precond, 1);
         JXF_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JXF_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JXF_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JXF_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JXF_3tAPCTLSetARRSolverID(precond, interp_solver_ARR); /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetACCSolverID(precond, SOLVER_AMG); /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetARRKDim(precond, interp_kdim_ARR); /* peghoty 2011/10/30 */          
         JXF_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JXF_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JXF_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JXF_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JXF_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JXF_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetACCRelaxTol(precond, tol_default);  
                  
         JXF_3tAPCTLSetNpR(precond, np_R);
         JXF_3tAPCTLSetNpE(precond, np_E);
         JXF_3tAPCTLSetNpI(precond, np_I);
         JXF_3tAPCTLSetComm(precond, comm);           
         JXF_3tAPCTLSetCommX(precond, comm_x);
         JXF_3tAPCTLSetCommY(precond, comm_y);
         JXF_3tAPCTLSetGroupIdX(precond, groupid_x);
         JXF_3tAPCTLSetGroupIdY(precond, groupid_y);
         
         JXF_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JXF_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         
         JXF_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JXF_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JXF_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JXF_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JXF_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JXF_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JXF_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JXF_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JXF_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JXF_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JXF_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JXF_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JXF_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JXF_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JXF_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

         JXF_3tAPCTLSetDebugFlag(precond, debug_flag);
         JXF_3tAPCTLSetStrongThreshold(precond, strong_threshold);
         JXF_3tAPCTLSetInterpType(precond, interp_type);
         JXF_3tAPCTLSetCoarsenType(precond, coarsen_type);
         JXF_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
         JXF_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
         JXF_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);
                  
         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level);

         JXF_GMRESSetPrecond( solver,
                             (JXF_PtrToSolverFcn) JXF_3tAPCTLPrecond,
                             (JXF_PtrToSolverFcn) NULL,
                             precond );            

         JXF_3tAPCTLSetup4Jasmin( precond, (JXF_ParCSRMatrix)par_mat );

         JXF_GMRESSetup( solver, 
                        (JXF_Matrix)par_mat,
                        (JXF_Vector)par_rhs,
                        (JXF_Vector)par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_setup = jxf_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jxf_MPI_Wtime();
         
         JXF_GMRESSolve( solver,
                        (JXF_Matrix)par_mat, // preOperator
                        (JXF_Matrix)par_mat,
                        (JXF_Vector)par_rhs,
                        (JXF_Vector)par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_solve = jxf_GetWallTimeMax(comm, starttime, endtime); 
         }
         
         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jxf_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jxf_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
         if (print_level_apctl)
         { 
            JXF_3tAPCTLIterCount(precond);
         }

         JXF_ParCSRGMRESDestroy(solver);
      }
      break;


      case 3:  // APCTL-BiCGSTab
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jxf_printf("\n Solver: \033[31mAPCTL-BiCGSTab\033[00m\n\n");
            else
               jxf_printf("\n Solver: \033[31mPPCTL-BiCGSTab\033[00m\n\n");
         }
         
         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_3tAPCTLDataInitialize(&precond);

         JXF_3tAPCTLSetA(precond, par_mat);
         JXF_3tAPCTLSetARR(precond, ARR);
         JXF_3tAPCTLSetAEE(precond, AEE);
         JXF_3tAPCTLSetAII(precond, AII);
         JXF_3tAPCTLSetVRE(precond, VRE);
         JXF_3tAPCTLSetVER(precond, VER);
         JXF_3tAPCTLSetVEI(precond, VEI);
         JXF_3tAPCTLSetVIE(precond, VIE);
         JXF_3tAPCTLSetARRAll(precond, ARR_p);
         JXF_3tAPCTLSetAEEAll(precond, AEE_p);
         JXF_3tAPCTLSetAIIAll(precond, AII_p);
         JXF_3tAPCTLSetVREAll(precond, VRE_p);
         JXF_3tAPCTLSetVERAll(precond, VER_p);
         JXF_3tAPCTLSetVEIAll(precond, VEI_p);
         JXF_3tAPCTLSetVIEAll(precond, VIE_p);         
         JXF_3tAPCTLSetMaxiter(precond, 1);
         JXF_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JXF_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JXF_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JXF_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JXF_3tAPCTLSetARRSolverID(precond, interp_solver_ARR); /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetACCSolverID(precond, SOLVER_AMG); /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetARRKDim(precond, interp_kdim_ARR); /* peghoty 2011/10/30 */          
         JXF_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JXF_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JXF_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JXF_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JXF_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JXF_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetACCRelaxTol(precond, tol_default);  
         
         JXF_3tAPCTLSetNpR(precond, np_R);
         JXF_3tAPCTLSetNpE(precond, np_E);
         JXF_3tAPCTLSetNpI(precond, np_I); 
         JXF_3tAPCTLSetComm(precond, comm);          
         JXF_3tAPCTLSetCommX(precond, comm_x);
         JXF_3tAPCTLSetCommY(precond, comm_y);
         JXF_3tAPCTLSetGroupIdX(precond, groupid_x);
         JXF_3tAPCTLSetGroupIdY(precond, groupid_y);
         
         JXF_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JXF_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         
         JXF_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JXF_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JXF_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JXF_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JXF_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JXF_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JXF_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JXF_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JXF_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JXF_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JXF_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JXF_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JXF_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JXF_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JXF_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

         JXF_3tAPCTLSetDebugFlag(precond, debug_flag);
         JXF_3tAPCTLSetStrongThreshold(precond, strong_threshold);
         JXF_3tAPCTLSetInterpType(precond, interp_type);
         JXF_3tAPCTLSetCoarsenType(precond, coarsen_type);
         JXF_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
         JXF_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
         JXF_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);
                  
         JXF_ParCSRBiCGSTABCreate(comm, &solver);
         JXF_BiCGSTABSetMaxIter(solver, max_iter);
         JXF_BiCGSTABSetTol(solver, tol);
         JXF_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JXF_BiCGSTABSetConvCriteria(solver, 0);
         JXF_BiCGSTABSetLogging(solver, 1);
         JXF_BiCGSTABSetPrintLevel(solver, print_level);

         JXF_BiCGSTABSetPrecond ( solver,
                                 (JXF_PtrToSolverFcn) JXF_3tAPCTLPrecond,
                                 (JXF_PtrToSolverFcn) NULL,
                                 precond ); 

         JXF_3tAPCTLSetup4Jasmin( precond, (JXF_ParCSRMatrix)par_mat );

         JXF_BiCGSTABSetup ( solver, 
                            (JXF_Matrix) par_mat, 
		            (JXF_Vector) par_rhs, 
		            (JXF_Vector) par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_setup = jxf_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jxf_MPI_Wtime();
         
         JXF_BiCGSTABSolve ( solver, 
                            (JXF_Matrix) par_mat, // preOperater
                            (JXF_Matrix) par_mat, 
		            (JXF_Vector) par_rhs, 
		            (JXF_Vector) par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_solve = jxf_GetWallTimeMax(comm, starttime, endtime); 
         }

         JXF_BiCGSTABGetNumIterations(solver, &num_iterations);
         JXF_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jxf_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jxf_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
 
         if (print_level_apctl)
         { 
            JXF_3tAPCTLIterCount(precond);
         }
         
         JXF_ParCSRBiCGSTABDestroy(solver);
      }
      break;

   }  // end switch
   
    
   if (TTest)
   {
      endtimeT = jxf_MPI_Wtime();
      cpu_total = jxf_GetWallTimeMax(comm, starttimeT, endtimeT);   
   }
   
   
   //=========================================================
   //  Step 4: 获取子系统迭代次数统计信息
   //=========================================================
   if (jxf_APCTLKrylovParamTestSubLSIter(apctlkrylov_param))
   {   
      jxf_GetAPCTLNumIterOfSubLS(precond, apctlkrylov_param);
   }
   JXF_3tAPCTLDestroy4Jasmin(precond);     

  
   //=========================================================
   //  Step 5: 将近似解向量保存到文件中
   //=========================================================
   
   if (keepsol)
   {
      jxf_Vector *ser_sol = NULL;
      ser_sol = jxf_ParVectorToVectorAll(par_sol);
      if (myid == 0) 
      {
#if 1      
         jxf_SeqVectorPrint(ser_sol, "./app");
#else       
         jxf_Vector *solR = NULL;
         jxf_Vector *solE = NULL;
         jxf_Vector *solI = NULL;
         jxf_3tGetSubVecs(ser_sol, &solR, &solE, &solI);
         jxf_SeqVectorPrint(solR, "./uR-1");
         jxf_SeqVectorPrint(solE, "./uE-1");
         jxf_SeqVectorPrint(solI, "./uI-1");
         jxf_SeqVectorDestroy(solR);
         jxf_SeqVectorDestroy(solE);
         jxf_SeqVectorDestroy(solI);               
#endif
      }
      jxf_SeqVectorDestroy(ser_sol);
   }


   //=========================================================
   //  Step 6: 将近似解向量从 par_sol 还原到 uR_p, uE_p, uI_p
   //=========================================================
   
   if (TTest) starttime = jxf_MPI_Wtime();
      
   jxf_APCTLKrylovSolBack4Jasmin(comm, comm_bak, groupid_x, par_sol, uR_p, uE_p, uI_p);

   if (TTest)
   {
      endtime = jxf_MPI_Wtime();
      cpu_trans_out = jxf_GetWallTimeMax(comm, starttime, endtime); 
   }
   
   
   //==================================================================================
   //  Step 7: 获取数据转换，Setup 和 Solve 过程的 CPU 时间
   //==================================================================================  
   
   jxf_APCTLKrylovParamSetCPUTrans(apctlkrylov_param, cpu_trans_in + cpu_trans_out);
   jxf_APCTLKrylovParamSetCPUSetup(apctlkrylov_param, cpu_setup);
   jxf_APCTLKrylovParamSetCPUSolve(apctlkrylov_param, cpu_solve);
   jxf_APCTLKrylovParamSetCPUTotal(apctlkrylov_param, cpu_total);    
   
   
   //=========================================================
   //  Step 8: 释放内存
   //=========================================================
   
   jxf_ParCSRMatrixDestroy(par_mat);
   jxf_ParVectorDestroy(par_rhs);
   jxf_ParVectorDestroy(par_sol);
   jxf_MPI_Comm_free(&comm_bak);
   
   return (0);
}                      

JXF_Int
jxf_ApctlKrylov_mgJASMIN_mp( jxf_ParCSRMatrix    **ARR_p,
                            jxf_ParCSRMatrix     *AEE_p,
                            jxf_ParCSRMatrix     *AII_p,
                            jxf_ParVector       **VRE_p,
                            jxf_ParVector       **VER_p,
                            jxf_ParVector        *VEI_p,
                            jxf_ParVector        *VIE_p,
                            jxf_ParVector       **fR_p,
                            jxf_ParVector        *fE_p,
                            jxf_ParVector        *fI_p,
                            jxf_ParVector       **uR_p,
                            jxf_ParVector        *uE_p,
                            jxf_ParVector        *uI_p,
                            jxf_APCTLKrylovParam *apctlkrylov_param )
{
   JXF_Int    ng                 = jxf_APCTLKrylovParamNumGroup(apctlkrylov_param);
   JXF_Int    solver_id          = jxf_APCTLKrylovParamSolverID(apctlkrylov_param);
   JXF_Int    precond_id         = jxf_APCTLKrylovParamPrecondID(apctlkrylov_param);
   JXF_Real tol                = jxf_APCTLKrylovParamTol(apctlkrylov_param);
   JXF_Int    max_iter           = jxf_APCTLKrylovParamMaxIter(apctlkrylov_param);
   JXF_Int    k_dim              = jxf_APCTLKrylovParamKDim(apctlkrylov_param);
   JXF_Int    is_check_restarted = jxf_APCTLKrylovParamISCheckRestarted(apctlkrylov_param);
   JXF_Int    print_level        = jxf_APCTLKrylovParamPrintLevel(apctlkrylov_param);
   JXF_Int    TTest              = jxf_APCTLKrylovParamTTest(apctlkrylov_param);
   JXF_Int    keepsol            = jxf_APCTLKrylovParamKeepSol(apctlkrylov_param);

   JXF_Int    print_level_apctl  = jxf_APCTLKrylovParamPrintLevelAPCTL(apctlkrylov_param);
   JXF_Int    blocksmooth_type   = jxf_APCTLKrylovParamBlockSmoothType(apctlkrylov_param);
   JXF_Int    interp_solver_ARR  = jxf_APCTLKrylovParamInterpSolverARR(apctlkrylov_param);
   JXF_Int    interp_kdim_ARR    = jxf_APCTLKrylovParamInterpKdimARR(apctlkrylov_param);
   JXF_Int    interp_maxit_ARR   = jxf_APCTLKrylovParamInterpMaxitARR(apctlkrylov_param);
   JXF_Int    interp_maxit_AII   = jxf_APCTLKrylovParamInterpMaxitAII(apctlkrylov_param);
   JXF_Real interp_tol_ARR     = jxf_APCTLKrylovParamInterpTolARR(apctlkrylov_param);
   JXF_Real interp_tol_AII     = jxf_APCTLKrylovParamInterpTolAII(apctlkrylov_param);
   JXF_Int    fixit_pctl_R       = jxf_APCTLKrylovParamFixitPCTLR(apctlkrylov_param);
   JXF_Int    fixit_pctl_E       = jxf_APCTLKrylovParamFixitPCTLE(apctlkrylov_param);
   JXF_Int    fixit_pctl_I       = jxf_APCTLKrylovParamFixitPCTLI(apctlkrylov_param);
   JXF_Int    fixit_brlx_R       = jxf_APCTLKrylovParamFixitBrlxR(apctlkrylov_param);
   JXF_Int    fixit_brlx_E       = jxf_APCTLKrylovParamFixitBrlxE(apctlkrylov_param);
   JXF_Int    fixit_brlx_I       = jxf_APCTLKrylovParamFixitBrlxI(apctlkrylov_param);
   JXF_Int    ARR_relax_type     = jxf_APCTLKrylovParamARRRelaxType(apctlkrylov_param);
   JXF_Int    AEE_relax_type     = jxf_APCTLKrylovParamAEERelaxType(apctlkrylov_param);
   JXF_Int    AII_relax_type     = jxf_APCTLKrylovParamAIIRelaxType(apctlkrylov_param);
   JXF_Int    use_fixedmode_R    = jxf_APCTLKrylovParamUseFixedModeR(apctlkrylov_param);
   JXF_Int    use_fixedmode_E    = jxf_APCTLKrylovParamUseFixedModeE(apctlkrylov_param);
   JXF_Int    use_fixedmode_I    = jxf_APCTLKrylovParamUseFixedModeI(apctlkrylov_param);
   JXF_Real theta_wc_E         = jxf_APCTLKrylovParamThetaWCE(apctlkrylov_param);
   JXF_Real threshold_wc_E     = jxf_APCTLKrylovParamThresholdWCE(apctlkrylov_param);
   JXF_Real theta_dd_R         = jxf_APCTLKrylovParamThetaDDR(apctlkrylov_param);
   JXF_Real theta_dd_E         = jxf_APCTLKrylovParamThetaDDE(apctlkrylov_param);
   JXF_Real theta_dd_I         = jxf_APCTLKrylovParamThetaDDI(apctlkrylov_param);
   JXF_Real threshold_dd_R     = jxf_APCTLKrylovParamThresholdDDR(apctlkrylov_param);
   JXF_Real threshold_dd_E     = jxf_APCTLKrylovParamThresholdDDE(apctlkrylov_param);
   JXF_Real threshold_dd_I     = jxf_APCTLKrylovParamThresholdDDI(apctlkrylov_param);
   JXF_Int    use_ppctl          = jxf_APCTLKrylovParamUsePPCTL(apctlkrylov_param);
   JXF_Int    test_subls_iter    = jxf_APCTLKrylovParamTestSubLSIter(apctlkrylov_param);
   JXF_Int    debug_flag         = jxf_APCTLKrylovParamDebugFlag(apctlkrylov_param);
   JXF_Int    reset_zero         = jxf_APCTLKrylovParamResetZero(apctlkrylov_param);
   JXF_Real ARR_relax_tol      = jxf_APCTLKrylovParamARRRelaxTol(apctlkrylov_param);
   JXF_Real AEE_relax_tol      = jxf_APCTLKrylovParamAEERelaxTol(apctlkrylov_param);
   JXF_Real AII_relax_tol      = jxf_APCTLKrylovParamAIIRelaxTol(apctlkrylov_param);
   JXF_Real ACC_relax_tol      = jxf_APCTLKrylovParamACCRelaxTol(apctlkrylov_param);

   JXF_Real strong_threshold = jxf_APCTLKrylovParamStrongThreshold(apctlkrylov_param);
   JXF_Int interp_type = jxf_APCTLKrylovParamInterpType(apctlkrylov_param);
   JXF_Int coarsen_type = jxf_APCTLKrylovParamCoarsenType(apctlkrylov_param);
   JXF_Int agg_num_levels = jxf_APCTLKrylovParamAggNumLevels(apctlkrylov_param);
   JXF_Int coarse_threshold = jxf_APCTLKrylovParamCoarseThreshold(apctlkrylov_param);
   JXF_Int print_level_amg = jxf_APCTLKrylovParamPrintLevelAMG(apctlkrylov_param);

   JXF_Int    num_iterations;
   JXF_Real final_res_norm;
   JXF_Real starttime  = 0.0;
   JXF_Real endtime    = 0.0;
   JXF_Real starttimeT = 0.0;
   JXF_Real endtimeT   = 0.0;

   JXF_Real cpu_trans_in  = 0.0;
   JXF_Real cpu_trans_out = 0.0;
   JXF_Real cpu_setup     = 0.0;
   JXF_Real cpu_solve     = 0.0;
   JXF_Real cpu_total     = 0.0;

   jxf_ParCSRMatrix  *ARR = NULL;
   jxf_ParCSRMatrix  *AEE = NULL;
   jxf_ParCSRMatrix  *AII = NULL;
   jxf_ParVector     *VRE = NULL;
   jxf_ParVector    **VER = NULL;
   jxf_ParVector     *VEI = NULL;
   jxf_ParVector     *VIE = NULL;

   jxf_ParCSRMatrix  *par_mat = NULL;
   jxf_ParVector     *par_rhs = NULL;
   jxf_ParVector     *par_sol = NULL;

   JXF_Solver solver  = NULL;
   JXF_Solver precond = NULL;

   JXF_Int myid, nprocs, np_R;
   JXF_Int groupid_x = MPI_UNDEFINED;
   JXF_Int groupid_y = MPI_UNDEFINED;
   MPI_Comm comm = jxf_ParCSRMatrixComm(ARR_p[0]);
   MPI_Comm comm_x, comm_bak, comm_y;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   np_R = nprocs / (ng + 2);
   groupid_x = myid / np_R;
   jxf_MPI_Comm_split(comm, groupid_x, myid, &comm_x);
   jxf_MPI_Comm_dup(comm_x, &comm_bak);

   groupid_y = myid % np_R;
   jxf_MPI_Comm_split(comm, groupid_y, myid, &comm_y);

   if (TTest) starttime = jxf_MPI_Wtime();

   jxf_ParaDataTrans4ApctlmgKrylov( comm, comm_x, groupid_x,
                                   ARR_p, AEE_p, AII_p, VRE_p, VER_p, VEI_p, VIE_p,
                                   fR_p, fE_p, fI_p, uR_p, uE_p, uI_p, ng,
                                  &ARR, &AEE, &AII, &VRE, &VER, &VEI, &VIE,
                                  &par_mat, &par_rhs, &par_sol );

   if (TTest)
   {
      endtime = jxf_MPI_Wtime();
      cpu_trans_in = jxf_GetWallTimeMax(comm, starttime, endtime);
   }

   if (TTest) starttimeT = jxf_MPI_Wtime();

   if (precond_id == 1) // APCTL
   {
      switch (solver_id)
      {
         case 2:  // APCTL-GMRES(m)
         {
            if (myid == 0 && print_level)
            {
               if (!use_ppctl)
               {
                  jxf_printf("\n Solver: \033[31mAPCTL-GMRES(%d)\033[00m\n\n", k_dim);
               }
               else
               {
                  jxf_printf("\n Solver: \033[31mPPCTL-GMRES(%d)\033[00m\n\n", k_dim);
               }
            }

            if (TTest) starttime = jxf_MPI_Wtime();

            JXF_3tAPCTLDataInitialize(&precond);

            JXF_3tAPCTLSetNumGroup(precond, ng);
            JXF_3tAPCTLSetA(precond, par_mat);
            JXF_3tAPCTLSetARR(precond, ARR);
            JXF_3tAPCTLSetAEE(precond, AEE);
            JXF_3tAPCTLSetAII(precond, AII);
            JXF_3tAPCTLSetVRE(precond, VRE);
            JXF_3tAPCTLSetVER2(precond, VER);
            JXF_3tAPCTLSetVEI(precond, VEI);
            JXF_3tAPCTLSetVIE(precond, VIE);
            JXF_3tAPCTLSetARRAll2(precond, ARR_p);
            JXF_3tAPCTLSetAEEAll(precond, AEE_p);
            JXF_3tAPCTLSetAIIAll(precond, AII_p);
            JXF_3tAPCTLSetVREAll2(precond, VRE_p);
            JXF_3tAPCTLSetVERAll2(precond, VER_p);
            JXF_3tAPCTLSetVEIAll(precond, VEI_p);
            JXF_3tAPCTLSetVIEAll(precond, VIE_p);
            JXF_3tAPCTLSetMaxiter(precond, 1);
            JXF_3tAPCTLSetPrintLevel(precond, print_level_apctl);
            JXF_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type);
            JXF_3tAPCTLSetARRSolverID(precond, interp_solver_ARR); /* peghoty 2011/10/30 */
            JXF_3tAPCTLSetACCSolverID(precond, SOLVER_AMG); /* peghoty 2011/10/30 */
            JXF_3tAPCTLSetARRKDim(precond, interp_kdim_ARR); /* peghoty 2011/10/30 */
            JXF_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
            JXF_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
            JXF_3tAPCTLSetACCRelaxMaxIt(precond, 1);
            JXF_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
            JXF_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
            JXF_3tAPCTLSetARRRelaxTol(precond, ARR_relax_tol);
            JXF_3tAPCTLSetAEERelaxTol(precond, AEE_relax_tol);
            JXF_3tAPCTLSetAIIRelaxTol(precond, AII_relax_tol);
            JXF_3tAPCTLSetACCRelaxTol(precond, ACC_relax_tol);
            JXF_3tAPCTLSetARRRelaxType(precond, ARR_relax_type);
            JXF_3tAPCTLSetAEERelaxType(precond, AEE_relax_type);
            JXF_3tAPCTLSetAIIRelaxType(precond, AII_relax_type);

            JXF_3tAPCTLSetNpR(precond, np_R);
            JXF_3tAPCTLSetComm(precond, comm);
            JXF_3tAPCTLSetCommX(precond, comm_x);
            JXF_3tAPCTLSetCommY(precond, comm_y);
            JXF_3tAPCTLSetGroupIdX(precond, groupid_x);

            JXF_3tAPCTLSetThetaWCE(precond, theta_wc_E);
            JXF_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);

            JXF_3tAPCTLSetThetaDDR(precond, theta_dd_R);
            JXF_3tAPCTLSetThetaDDE(precond, theta_dd_E);
            JXF_3tAPCTLSetThetaDDI(precond, theta_dd_I);
            JXF_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
            JXF_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
            JXF_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);

            JXF_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
            JXF_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
            JXF_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
            JXF_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
            JXF_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
            JXF_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

            JXF_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
            JXF_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
            JXF_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */

            JXF_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
            JXF_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

            JXF_3tAPCTLSetDebugFlag(precond, debug_flag);
            JXF_3tAPCTLSetResetZero(precond, reset_zero);
            JXF_3tAPCTLSetStrongThreshold(precond, strong_threshold);
            JXF_3tAPCTLSetInterpType(precond, interp_type);
            JXF_3tAPCTLSetCoarsenType(precond, coarsen_type);
            JXF_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
            JXF_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
            JXF_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);

            JXF_ParCSRGMRESCreate(comm, &solver);
            JXF_GMRESSetKDim(solver, k_dim);
            JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
            JXF_GMRESSetMaxIter(solver, max_iter);
            JXF_GMRESSetTol(solver, tol);
            JXF_GMRESSetLogging(solver, 1);
            JXF_GMRESSetPrintLevel(solver, print_level);

            JXF_GMRESSetPrecond( solver,
                                (JXF_PtrToSolverFcn) JXF_3tAPCTLmgPrecond,
                                (JXF_PtrToSolverFcn) NULL,
                                precond );

            JXF_3tAPCTLSetup4mgJasmin(precond, (JXF_ParCSRMatrix)par_mat);

            JXF_GMRESSetup( solver, 
                           (JXF_Matrix)par_mat,
                           (JXF_Vector)par_rhs,
                           (JXF_Vector)par_sol );

            if (TTest)
            {
               endtime = jxf_MPI_Wtime();
               cpu_setup = jxf_GetWallTimeMax(comm, starttime, endtime);
            }

            if (TTest) starttime = jxf_MPI_Wtime();

            JXF_GMRESSolve( solver,
                           (JXF_Matrix)par_mat, // preOperator
                           (JXF_Matrix)par_mat,
                           (JXF_Vector)par_rhs,
                           (JXF_Vector)par_sol );

            if (TTest)
            {
               endtime = jxf_MPI_Wtime();
               cpu_solve = jxf_GetWallTimeMax(comm, starttime, endtime);
            }

            JXF_GMRESGetNumIterations(solver, &num_iterations);
            JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
            jxf_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
            jxf_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
            if (print_level_apctl)
            {
               JXF_3tAPCTLIterCount(precond);
            }

            JXF_ParCSRGMRESDestroy(solver);
         }
         break;
      }
   }
   else if (precond_id == 2) // ABSC1
   {
      switch (solver_id)
      {
         case 2:  // ABSC1-GMRES(m)
         {
            if (myid == 0 && print_level)
            {
               jxf_printf("\n Solver: \033[31mABSC1-GMRES(%d)\033[00m\n\n", k_dim);
            }

            if (TTest) starttime = jxf_MPI_Wtime();

            JXF_3tAPCTLDataInitialize(&precond);

            JXF_3tAPCTLSetNumGroup(precond, ng);
            JXF_3tAPCTLSetA(precond, par_mat);
            JXF_3tAPCTLSetARR(precond, ARR);
            JXF_3tAPCTLSetAEE(precond, AEE);
            JXF_3tAPCTLSetAII(precond, AII);
            JXF_3tAPCTLSetVRE(precond, VRE);
            JXF_3tAPCTLSetVER2(precond, VER);
            JXF_3tAPCTLSetVEI(precond, VEI);
            JXF_3tAPCTLSetVIE(precond, VIE);
            JXF_3tAPCTLSetMaxiter(precond, 1);
            JXF_3tAPCTLSetPrintLevel(precond, print_level_apctl);
            JXF_3tAPCTLSetARRRelaxTol(precond, ARR_relax_tol);
            JXF_3tAPCTLSetAEERelaxTol(precond, AEE_relax_tol);
            JXF_3tAPCTLSetAIIRelaxTol(precond, AII_relax_tol);
            JXF_3tAPCTLSetARRRelaxType(precond, ARR_relax_type);
            JXF_3tAPCTLSetAEERelaxType(precond, AEE_relax_type);
            JXF_3tAPCTLSetAIIRelaxType(precond, AII_relax_type);

            JXF_3tAPCTLSetNpR(precond, np_R);
            JXF_3tAPCTLSetComm(precond, comm);
            JXF_3tAPCTLSetCommX(precond, comm_x);
            JXF_3tAPCTLSetCommY(precond, comm_y);
            JXF_3tAPCTLSetGroupIdX(precond, groupid_x);

            JXF_3tAPCTLSetThetaWCE(precond, theta_wc_E);
            JXF_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);

            JXF_3tAPCTLSetThetaDDR(precond, theta_dd_R);
            JXF_3tAPCTLSetThetaDDE(precond, theta_dd_E);
            JXF_3tAPCTLSetThetaDDI(precond, theta_dd_I);
            JXF_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
            JXF_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
            JXF_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);

            JXF_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
            JXF_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
            JXF_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
            JXF_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
            JXF_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
            JXF_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

            JXF_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
            JXF_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
            JXF_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */

            JXF_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

            JXF_3tAPCTLSetDebugFlag(precond, debug_flag);
            JXF_3tAPCTLSetResetZero(precond, reset_zero);
            JXF_3tAPCTLSetStrongThreshold(precond, strong_threshold);
            JXF_3tAPCTLSetInterpType(precond, interp_type);
            JXF_3tAPCTLSetCoarsenType(precond, coarsen_type);
            JXF_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
            JXF_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
            JXF_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);

            JXF_ParCSRGMRESCreate(comm, &solver);
            JXF_GMRESSetKDim(solver, k_dim);
            JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
            JXF_GMRESSetMaxIter(solver, max_iter);
            JXF_GMRESSetTol(solver, tol);
            JXF_GMRESSetLogging(solver, 1);
            JXF_GMRESSetPrintLevel(solver, print_level);

            JXF_GMRESSetPrecond( solver,
                                (JXF_PtrToSolverFcn) JXF_3tABSC1mgPrecond,
                                (JXF_PtrToSolverFcn) NULL,
                                precond );

            JXF_3tABSC1Setup4mgJasmin(precond, (JXF_ParCSRMatrix)par_mat);

            JXF_GMRESSetup( solver, 
                           (JXF_Matrix)par_mat,
                           (JXF_Vector)par_rhs,
                           (JXF_Vector)par_sol );

            if (TTest)
            {
               endtime = jxf_MPI_Wtime();
               cpu_setup = jxf_GetWallTimeMax(comm, starttime, endtime);
            }

            if (TTest) starttime = jxf_MPI_Wtime();

            JXF_GMRESSolve( solver,
                           (JXF_Matrix)par_mat, // preOperator
                           (JXF_Matrix)par_mat,
                           (JXF_Vector)par_rhs,
                           (JXF_Vector)par_sol );

            if (TTest)
            {
               endtime = jxf_MPI_Wtime();
               cpu_solve = jxf_GetWallTimeMax(comm, starttime, endtime);
            }

            JXF_GMRESGetNumIterations(solver, &num_iterations);
            JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
            jxf_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
            jxf_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
            if (print_level_apctl)
            {
               JXF_3tAPCTLIterCount(precond);
            }

            JXF_ParCSRGMRESDestroy(solver);
         }
         break;
      }
   }
   else if (precond_id == 3) // ABSC2
   {
      switch (solver_id)
      {
         case 2:  // ABSC2-GMRES(m)
         {
            if (myid == 0 && print_level)
            {
               jxf_printf("\n Solver: \033[31mABSC2-GMRES(%d)\033[00m\n\n", k_dim);
            }

            if (TTest) starttime = jxf_MPI_Wtime();

            JXF_3tAPCTLDataInitialize(&precond);

            JXF_3tAPCTLSetNumGroup(precond, ng);
            JXF_3tAPCTLSetA(precond, par_mat);
            JXF_3tAPCTLSetARR(precond, ARR);
            JXF_3tAPCTLSetAEE(precond, AEE);
            JXF_3tAPCTLSetAII(precond, AII);
            JXF_3tAPCTLSetVRE(precond, VRE);
            JXF_3tAPCTLSetVER2(precond, VER);
            JXF_3tAPCTLSetVEI(precond, VEI);
            JXF_3tAPCTLSetVIE(precond, VIE);
            JXF_3tAPCTLSetMaxiter(precond, 1);
            JXF_3tAPCTLSetPrintLevel(precond, print_level_apctl);
            JXF_3tAPCTLSetARRRelaxTol(precond, ARR_relax_tol);
            JXF_3tAPCTLSetAEERelaxTol(precond, AEE_relax_tol);
            JXF_3tAPCTLSetAIIRelaxTol(precond, AII_relax_tol);
            JXF_3tAPCTLSetARRRelaxType(precond, ARR_relax_type);
            JXF_3tAPCTLSetAEERelaxType(precond, AEE_relax_type);
            JXF_3tAPCTLSetAIIRelaxType(precond, AII_relax_type);

            JXF_3tAPCTLSetNpR(precond, np_R);
            JXF_3tAPCTLSetComm(precond, comm);
            JXF_3tAPCTLSetCommX(precond, comm_x);
            JXF_3tAPCTLSetCommY(precond, comm_y);
            JXF_3tAPCTLSetGroupIdX(precond, groupid_x);

            JXF_3tAPCTLSetThetaWCE(precond, theta_wc_E);
            JXF_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);

            JXF_3tAPCTLSetThetaDDR(precond, theta_dd_R);
            JXF_3tAPCTLSetThetaDDE(precond, theta_dd_E);
            JXF_3tAPCTLSetThetaDDI(precond, theta_dd_I);
            JXF_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
            JXF_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
            JXF_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);

            JXF_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
            JXF_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
            JXF_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
            JXF_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
            JXF_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
            JXF_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

            JXF_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
            JXF_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
            JXF_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */

            JXF_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

            JXF_3tAPCTLSetDebugFlag(precond, debug_flag);
            JXF_3tAPCTLSetResetZero(precond, reset_zero);
            JXF_3tAPCTLSetStrongThreshold(precond, strong_threshold);
            JXF_3tAPCTLSetInterpType(precond, interp_type);
            JXF_3tAPCTLSetCoarsenType(precond, coarsen_type);
            JXF_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
            JXF_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
            JXF_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);

            JXF_ParCSRGMRESCreate(comm, &solver);
            JXF_GMRESSetKDim(solver, k_dim);
            JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
            JXF_GMRESSetMaxIter(solver, max_iter);
            JXF_GMRESSetTol(solver, tol);
            JXF_GMRESSetLogging(solver, 1);
            JXF_GMRESSetPrintLevel(solver, print_level);

            JXF_GMRESSetPrecond( solver,
                                (JXF_PtrToSolverFcn) JXF_3tABSC2mgPrecond,
                                (JXF_PtrToSolverFcn) NULL,
                                precond );

            JXF_3tABSC2Setup4mgJasmin(precond, (JXF_ParCSRMatrix)par_mat);

            JXF_GMRESSetup( solver, 
                           (JXF_Matrix)par_mat,
                           (JXF_Vector)par_rhs,
                           (JXF_Vector)par_sol );

            if (TTest)
            {
               endtime = jxf_MPI_Wtime();
               cpu_setup = jxf_GetWallTimeMax(comm, starttime, endtime);
            }

            if (TTest) starttime = jxf_MPI_Wtime();

            JXF_GMRESSolve( solver,
                           (JXF_Matrix)par_mat, // preOperator
                           (JXF_Matrix)par_mat,
                           (JXF_Vector)par_rhs,
                           (JXF_Vector)par_sol );

            if (TTest)
            {
               endtime = jxf_MPI_Wtime();
               cpu_solve = jxf_GetWallTimeMax(comm, starttime, endtime);
            }

            JXF_GMRESGetNumIterations(solver, &num_iterations);
            JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
            jxf_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
            jxf_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
            if (print_level_apctl)
            {
               JXF_3tAPCTLIterCount(precond);
            }

            JXF_ParCSRGMRESDestroy(solver);
         }
         break;
      }
   }

   if (TTest)
   {
      endtimeT = jxf_MPI_Wtime();
      cpu_total = jxf_GetWallTimeMax(comm, starttimeT, endtimeT);
   }

   //=========================================================
   //  Step 4: 获取子系统迭代次数统计信息
   //=========================================================
   if (jxf_APCTLKrylovParamTestSubLSIter(apctlkrylov_param))
   {
      jxf_GetAPCTLmgNumIterOfSubLS(precond, groupid_x, ng, apctlkrylov_param);
   }
   JXF_3tAPCTLDestroy4mgJasmin(precond);

   //=========================================================
   //  Step 5: 将近似解向量保存到文件中
   //=========================================================

   if (keepsol)
   {
      jxf_Vector *ser_sol = NULL;
      ser_sol = jxf_ParVectorToVectorAll(par_sol);
      if (myid == 0)
      {
         jxf_SeqVectorPrint(ser_sol, "./app");
      }
      jxf_SeqVectorDestroy(ser_sol);
   }

   //=========================================================
   //  Step 6: 将近似解向量从 par_sol 还原到 uR_p, uE_p, uI_p
   //=========================================================

   if (TTest) starttime = jxf_MPI_Wtime();
 
   jxf_APCTLKrylovSolBack4mgJasmin(comm, comm_bak, groupid_x, ng, par_sol, uR_p, uE_p, uI_p);

   if (TTest)
   {
      endtime = jxf_MPI_Wtime();
      cpu_trans_out = jxf_GetWallTimeMax(comm, starttime, endtime);
   }

   //==================================================================================
   //  Step 7: 获取数据转换，Setup 和 Solve 过程的 CPU 时间
   //==================================================================================

   jxf_APCTLKrylovParamSetCPUTrans(apctlkrylov_param, cpu_trans_in + cpu_trans_out);
   jxf_APCTLKrylovParamSetCPUSetup(apctlkrylov_param, cpu_setup);
   jxf_APCTLKrylovParamSetCPUSolve(apctlkrylov_param, cpu_solve);
   jxf_APCTLKrylovParamSetCPUTotal(apctlkrylov_param, cpu_total);

   //=========================================================
   //  Step 8: 释放内存
   //=========================================================

   jxf_ParCSRMatrixDestroy(par_mat);
   jxf_ParVectorDestroy(par_rhs);
   jxf_ParVectorDestroy(par_sol);
   jxf_MPI_Comm_free(&comm_bak);

   return (0);
}

/*!
 * \fn JXF_Int jxf_ApctlKrylov_JASMIN_sp
 * \brief APCTL-GMRES solvers for single-processor case.
 * \author peghoty
 * \date 2012/02/25 
 */
JXF_Int 
jxf_ApctlKrylov_JASMIN_sp( jxf_ParCSRMatrix     *ARR_p, 
                          jxf_ParCSRMatrix     *AEE_p, 
                          jxf_ParCSRMatrix     *AII_p, 
                          jxf_ParVector        *VRE_p, 
                          jxf_ParVector        *VER_p, 
                          jxf_ParVector        *VEI_p, 
                          jxf_ParVector        *VIE_p, 
                          jxf_ParVector        *fR_p, 
                          jxf_ParVector        *fE_p, 
                          jxf_ParVector        *fI_p,
                          jxf_ParVector        *uR_p, 
                          jxf_ParVector        *uE_p, 
                          jxf_ParVector        *uI_p,
                          jxf_APCTLKrylovParam *apctlkrylov_param )
{                   
   JXF_Int    solver_id          = jxf_APCTLKrylovParamSolverID(apctlkrylov_param);
   JXF_Real tol                = jxf_APCTLKrylovParamTol(apctlkrylov_param);
   JXF_Int    max_iter           = jxf_APCTLKrylovParamMaxIter(apctlkrylov_param);
   JXF_Int    k_dim              = jxf_APCTLKrylovParamKDim(apctlkrylov_param);
   JXF_Int    is_check_restarted = jxf_APCTLKrylovParamISCheckRestarted(apctlkrylov_param);
   JXF_Int    two_norm           = jxf_APCTLKrylovParamTwoNorm(apctlkrylov_param);
   JXF_Int    print_level        = jxf_APCTLKrylovParamPrintLevel(apctlkrylov_param);  
   JXF_Int    TTest              = jxf_APCTLKrylovParamTTest(apctlkrylov_param);       
   JXF_Int    keepsol            = jxf_APCTLKrylovParamKeepSol(apctlkrylov_param);

   JXF_Int    print_level_apctl  = jxf_APCTLKrylovParamPrintLevelAPCTL(apctlkrylov_param);   
   JXF_Int    blocksmooth_type   = jxf_APCTLKrylovParamBlockSmoothType(apctlkrylov_param);   
   JXF_Int    num_relax_pre      = jxf_APCTLKrylovParamNumRelaxPre(apctlkrylov_param);       
   JXF_Int    num_relax_post     = jxf_APCTLKrylovParamNumRelaxPost(apctlkrylov_param);      
   JXF_Int    interp_solver_ARR  = jxf_APCTLKrylovParamInterpSolverARR(apctlkrylov_param);   
   JXF_Int    interp_kdim_ARR    = jxf_APCTLKrylovParamInterpKdimARR(apctlkrylov_param);     
   JXF_Int    interp_maxit_ARR   = jxf_APCTLKrylovParamInterpMaxitARR(apctlkrylov_param);    
   JXF_Int    interp_maxit_AII   = jxf_APCTLKrylovParamInterpMaxitAII(apctlkrylov_param);    
   JXF_Real interp_tol_ARR     = jxf_APCTLKrylovParamInterpTolARR(apctlkrylov_param);      
   JXF_Real interp_tol_AII     = jxf_APCTLKrylovParamInterpTolAII(apctlkrylov_param);      
   JXF_Int    fixit_pctl_R       = jxf_APCTLKrylovParamFixitPCTLR(apctlkrylov_param);        
   JXF_Int    fixit_pctl_E       = jxf_APCTLKrylovParamFixitPCTLE(apctlkrylov_param);        
   JXF_Int    fixit_pctl_I       = jxf_APCTLKrylovParamFixitPCTLI(apctlkrylov_param);        
   JXF_Int    fixit_brlx_R       = jxf_APCTLKrylovParamFixitBrlxR(apctlkrylov_param);        
   JXF_Int    fixit_brlx_E       = jxf_APCTLKrylovParamFixitBrlxE(apctlkrylov_param);        
   JXF_Int    fixit_brlx_I       = jxf_APCTLKrylovParamFixitBrlxI(apctlkrylov_param);        
   JXF_Int    use_fixedmode_R    = jxf_APCTLKrylovParamUseFixedModeR(apctlkrylov_param);     
   JXF_Int    use_fixedmode_E    = jxf_APCTLKrylovParamUseFixedModeE(apctlkrylov_param);     
   JXF_Int    use_fixedmode_I    = jxf_APCTLKrylovParamUseFixedModeI(apctlkrylov_param);     
   JXF_Real theta_wc_E         = jxf_APCTLKrylovParamThetaWCE(apctlkrylov_param);          
   JXF_Real threshold_wc_E     = jxf_APCTLKrylovParamThresholdWCE(apctlkrylov_param);      
   JXF_Real theta_dd_R         = jxf_APCTLKrylovParamThetaDDR(apctlkrylov_param);          
   JXF_Real theta_dd_E         = jxf_APCTLKrylovParamThetaDDE(apctlkrylov_param);          
   JXF_Real theta_dd_I         = jxf_APCTLKrylovParamThetaDDI(apctlkrylov_param);          
   JXF_Real threshold_dd_R     = jxf_APCTLKrylovParamThresholdDDR(apctlkrylov_param);      
   JXF_Real threshold_dd_E     = jxf_APCTLKrylovParamThresholdDDE(apctlkrylov_param);      
   JXF_Real threshold_dd_I     = jxf_APCTLKrylovParamThresholdDDI(apctlkrylov_param);      
   JXF_Int    use_ppctl          = jxf_APCTLKrylovParamUsePPCTL(apctlkrylov_param);
   JXF_Int    test_subls_iter    = jxf_APCTLKrylovParamTestSubLSIter(apctlkrylov_param);
   JXF_Real tol_default        = 1.0e-6;

   JXF_Int    num_iterations;
   JXF_Real final_res_norm; 
   JXF_Real starttime  = 0.0;
   JXF_Real endtime    = 0.0;  
   JXF_Real starttimeT = 0.0;
   JXF_Real endtimeT   = 0.0;      

   JXF_Real cpu_trans_in  = 0.0;
   JXF_Real cpu_trans_out = 0.0;
   JXF_Real cpu_setup     = 0.0;
   JXF_Real cpu_solve     = 0.0;
   JXF_Real cpu_total     = 0.0;  
                 
   jxf_ParCSRMatrix  *par_mat = NULL;
   jxf_ParVector     *par_rhs = NULL;
   jxf_ParVector     *par_sol = NULL;

   JXF_Solver solver  = NULL;
   JXF_Solver precond = NULL;

   JXF_Int myid, nprocs;
   MPI_Comm comm = jxf_ParCSRMatrixComm(ARR_p);
         
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (TTest) starttime = jxf_MPI_Wtime();
             
   jxf_DataCombine4ApctlKrylov( ARR_p, AEE_p, AII_p, 
                               VRE_p, VER_p, VEI_p, VIE_p,
                               fR_p, fE_p, fI_p,
                               uR_p, uE_p, uI_p,
                               &par_mat, &par_rhs, &par_sol ); 

   if (TTest)
   {
      endtime = jxf_MPI_Wtime();
      cpu_trans_in = jxf_GetWallTimeMax(comm, starttime, endtime);  
   }     


   if (TTest) starttimeT = jxf_MPI_Wtime();
                                          
   switch (solver_id)
   {

      case 1:  // APCTL-CG
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jxf_printf("\n Solver: \033[31mAPCTL-CG\033[00m\n\n");
            else
               jxf_printf("\n Solver: \033[31mPPCTL-CG\033[00m\n\n");
         }
         
         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_3tAPCTLDataInitialize(&precond);

         JXF_3tAPCTLSetA(precond, par_mat);
         JXF_3tAPCTLSetComm(precond, comm);
         JXF_3tAPCTLSetARR(precond, ARR_p);
         JXF_3tAPCTLSetAEE(precond, AEE_p);
         JXF_3tAPCTLSetAII(precond, AII_p);
         JXF_3tAPCTLSetVRE(precond, VRE_p);
         JXF_3tAPCTLSetVER(precond, VER_p);
         JXF_3tAPCTLSetVEI(precond, VEI_p);
         JXF_3tAPCTLSetVIE(precond, VIE_p);        
         JXF_3tAPCTLSetMaxiter(precond, 1);
         JXF_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JXF_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JXF_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JXF_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JXF_3tAPCTLSetARRSolverID(precond, interp_solver_ARR);  /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetACCSolverID(precond, SOLVER_AMG);         /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetARRKDim(precond, interp_kdim_ARR);        /* peghoty 2011/10/30 */          
         JXF_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JXF_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JXF_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JXF_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JXF_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JXF_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetACCRelaxTol(precond, tol_default);   

         JXF_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JXF_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         JXF_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JXF_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JXF_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JXF_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JXF_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JXF_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JXF_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JXF_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JXF_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JXF_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JXF_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JXF_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JXF_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JXF_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JXF_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

         JXF_ParCSRPCGCreate(comm, &solver); 
         JXF_PCGSetMaxIter(solver, max_iter);
         JXF_PCGSetTol(solver, tol);
         JXF_PCGSetTwoNorm(solver, two_norm);  // 0: B 范数； 1：l2 范数 
         JXF_PCGSetLogging(solver, 1);
         JXF_PCGSetPrintLevel(solver, print_level);
         
         JXF_PCGSetPrecond( solver,
                           (JXF_PtrToSolverFcn) JXF_3tAPCTLPrecond,
                           (JXF_PtrToSolverFcn) NULL,
                           precond ); 
         
         JXF_3tAPCTLSetup4Jasmin( precond, (JXF_ParCSRMatrix)par_mat );                   
         
         JXF_PCGSetup ( solver, 
                       (JXF_Matrix) par_mat, 
                       (JXF_Vector) par_rhs, 
                       (JXF_Vector) par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_setup = jxf_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jxf_MPI_Wtime();
         
         JXF_PCGSolve ( solver, 
                       (JXF_Matrix) par_mat, // preOperater
                       (JXF_Matrix) par_mat, 
                       (JXF_Vector) par_rhs, 
                       (JXF_Vector) par_sol );  

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_solve = jxf_GetWallTimeMax(comm, starttime, endtime); 
         }

         JXF_PCGGetNumIterations(solver, &num_iterations);
         JXF_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jxf_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jxf_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);

         if (print_level_apctl)
         { 
            JXF_3tAPCTLIterCount(precond);
         }

         JXF_ParCSRPCGDestroy(solver);
      }
      break;
      

      case 2:  // APCTL-GMRES(m)
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jxf_printf("\n Solver: \033[31mAPCTL-GMRES(%d)\033[00m\n\n", k_dim);
            else
               jxf_printf("\n Solver: \033[31mPPCTL-GMRES(%d)\033[00m\n\n", k_dim);
         }
         
         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_3tAPCTLDataInitialize(&precond);

         JXF_3tAPCTLSetA(precond, par_mat);
         JXF_3tAPCTLSetComm(precond, comm);
         JXF_3tAPCTLSetARR(precond, ARR_p);
         JXF_3tAPCTLSetAEE(precond, AEE_p);
         JXF_3tAPCTLSetAII(precond, AII_p);
         JXF_3tAPCTLSetVRE(precond, VRE_p);
         JXF_3tAPCTLSetVER(precond, VER_p);
         JXF_3tAPCTLSetVEI(precond, VEI_p);
         JXF_3tAPCTLSetVIE(precond, VIE_p);        
         JXF_3tAPCTLSetMaxiter(precond, 1);
         JXF_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JXF_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JXF_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JXF_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JXF_3tAPCTLSetARRSolverID(precond, interp_solver_ARR);  /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetACCSolverID(precond, SOLVER_AMG);         /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetARRKDim(precond, interp_kdim_ARR);        /* peghoty 2011/10/30 */          
         JXF_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JXF_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JXF_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JXF_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JXF_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JXF_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetACCRelaxTol(precond, tol_default);   

         JXF_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JXF_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         JXF_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JXF_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JXF_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JXF_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JXF_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JXF_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JXF_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JXF_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JXF_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JXF_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JXF_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JXF_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JXF_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JXF_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JXF_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */
                  
         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level);

         JXF_GMRESSetPrecond( solver,
                             (JXF_PtrToSolverFcn) JXF_3tAPCTLPrecond,
                             (JXF_PtrToSolverFcn) NULL,
                             precond );            
 
         JXF_3tAPCTLSetup4Jasmin( precond, (JXF_ParCSRMatrix)par_mat );
 
         JXF_GMRESSetup( solver, 
                        (JXF_Matrix)par_mat,
                        (JXF_Vector)par_rhs,
                        (JXF_Vector)par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_setup = jxf_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jxf_MPI_Wtime();
         
         JXF_GMRESSolve( solver,
                        (JXF_Matrix)par_mat, // preOperator
                        (JXF_Matrix)par_mat,
                        (JXF_Vector)par_rhs,
                        (JXF_Vector)par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_solve = jxf_GetWallTimeMax(comm, starttime, endtime); 
         }
         
         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jxf_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jxf_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
         if (print_level_apctl)
         { 
            JXF_3tAPCTLIterCount(precond);
         }

         JXF_ParCSRGMRESDestroy(solver);
      }
      break;


      case 3:  // APCTL-BiCGSTab
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jxf_printf("\n Solver: \033[31mAPCTL-BiCGSTab\033[00m\n\n");
            else
               jxf_printf("\n Solver: \033[31mPPCTL-BiCGSTab\033[00m\n\n");
         }
         
         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_3tAPCTLDataInitialize(&precond);

         JXF_3tAPCTLSetA(precond, par_mat);
         JXF_3tAPCTLSetComm(precond, comm);
         JXF_3tAPCTLSetARR(precond, ARR_p);
         JXF_3tAPCTLSetAEE(precond, AEE_p);
         JXF_3tAPCTLSetAII(precond, AII_p);
         JXF_3tAPCTLSetVRE(precond, VRE_p);
         JXF_3tAPCTLSetVER(precond, VER_p);
         JXF_3tAPCTLSetVEI(precond, VEI_p);
         JXF_3tAPCTLSetVIE(precond, VIE_p);        
         JXF_3tAPCTLSetMaxiter(precond, 1);
         JXF_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JXF_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JXF_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JXF_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JXF_3tAPCTLSetARRSolverID(precond, interp_solver_ARR);  /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetACCSolverID(precond, SOLVER_AMG);         /* peghoty 2011/10/30 */
         JXF_3tAPCTLSetARRKDim(precond, interp_kdim_ARR);        /* peghoty 2011/10/30 */          
         JXF_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JXF_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JXF_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JXF_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JXF_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JXF_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JXF_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JXF_3tAPCTLSetACCRelaxTol(precond, tol_default);   

         JXF_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JXF_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         JXF_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JXF_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JXF_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JXF_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JXF_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JXF_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JXF_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JXF_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JXF_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JXF_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JXF_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JXF_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JXF_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JXF_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JXF_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JXF_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */
                  
         JXF_ParCSRBiCGSTABCreate(comm, &solver);
         JXF_BiCGSTABSetMaxIter(solver, max_iter);
         JXF_BiCGSTABSetTol(solver, tol);
         JXF_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JXF_BiCGSTABSetConvCriteria(solver, 0);
         JXF_BiCGSTABSetLogging(solver, 1);
         JXF_BiCGSTABSetPrintLevel(solver, print_level);

         JXF_BiCGSTABSetPrecond ( solver,
                                 (JXF_PtrToSolverFcn) JXF_3tAPCTLPrecond,
                                 (JXF_PtrToSolverFcn) NULL,
                                 precond ); 

         JXF_3tAPCTLSetup4Jasmin( precond, (JXF_ParCSRMatrix)par_mat );

         JXF_BiCGSTABSetup ( solver, 
                            (JXF_Matrix) par_mat, 
		            (JXF_Vector) par_rhs, 
		            (JXF_Vector) par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_setup = jxf_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jxf_MPI_Wtime();
         
         JXF_BiCGSTABSolve ( solver, 
                            (JXF_Matrix) par_mat, // preOperater
                            (JXF_Matrix) par_mat, 
		            (JXF_Vector) par_rhs, 
		            (JXF_Vector) par_sol );

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            cpu_solve = jxf_GetWallTimeMax(comm, starttime, endtime); 
         }

         JXF_BiCGSTABGetNumIterations(solver, &num_iterations);
         JXF_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jxf_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jxf_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
 
         if (print_level_apctl)
         { 
            JXF_3tAPCTLIterCount(precond);
         }

         JXF_ParCSRBiCGSTABDestroy(solver);
      }
      break;

   }  // end switch
       
   if (TTest)
   {
      endtimeT = jxf_MPI_Wtime();
      cpu_total = jxf_GetWallTimeMax(comm, starttimeT, endtimeT);      
   }
   
  
   //=========================================================
   //  Step 3: 获取子系统迭代次数统计信息
   //=========================================================
      
   if (jxf_APCTLKrylovParamTestSubLSIter(apctlkrylov_param))
   {   
      jxf_GetAPCTLNumIterOfSubLS(precond, apctlkrylov_param);
   }
   JXF_3tAPCTLDestroy4Jasmin(precond); 
      
  
   //=========================================================
   //  Step 4: 将近似解向量保存到文件中
   //=========================================================
    
   if (keepsol)
   {
      jxf_Vector *ser_sol = jxf_ParVectorLocalVector(par_sol);
      if (myid == 0) 
      {
#if 1     
         jxf_SeqVectorPrint(ser_sol, "./app");
#else       
         jxf_Vector *solR = NULL;
         jxf_Vector *solE = NULL;
         jxf_Vector *solI = NULL;
         jxf_3tGetSubVecs(ser_sol, &solR, &solE, &solI);
         jxf_SeqVectorPrint(solR, "./uR-1");
         jxf_SeqVectorPrint(solE, "./uE-1");
         jxf_SeqVectorPrint(solI, "./uI-1");
         jxf_SeqVectorDestroy(solR);
         jxf_SeqVectorDestroy(solE);
         jxf_SeqVectorDestroy(solI);      
#endif
      }
   }


   //=========================================================
   //  Step 5: 将近似解向量从 par_sol 还原到 uR_p, uE_p, uI_p
   //=========================================================
   
   if (TTest) starttime = jxf_MPI_Wtime();
   
   jxf_Vector *uR = jxf_ParVectorLocalVector(uR_p);
   jxf_Vector *uE = jxf_ParVectorLocalVector(uE_p);
   jxf_Vector *uI = jxf_ParVectorLocalVector(uI_p);
   jxf_Vector *sol = jxf_ParVectorLocalVector(par_sol);
   JXF_Int n = jxf_ParCSRMatrixGlobalNumRows(ARR_p);
   
   memcpy(jxf_VectorData(uR), jxf_VectorData(sol), n*sizeof(JXF_Real));
   memcpy(jxf_VectorData(uE), jxf_VectorData(sol)+n, n*sizeof(JXF_Real));
   memcpy(jxf_VectorData(uI), jxf_VectorData(sol)+2*n, n*sizeof(JXF_Real));

   if (TTest)
   {
      endtime = jxf_MPI_Wtime();
      cpu_trans_out = jxf_GetWallTimeMax(comm, starttime, endtime);   
   } 

   //===============================================================================
   //  Step 6: 获取数据转换，Setup 和 Solve 过程的 CPU 时间
   //===============================================================================  
   
   jxf_APCTLKrylovParamSetCPUTrans(apctlkrylov_param, cpu_trans_in + cpu_trans_out);
   jxf_APCTLKrylovParamSetCPUSetup(apctlkrylov_param, cpu_setup);
   jxf_APCTLKrylovParamSetCPUSolve(apctlkrylov_param, cpu_solve);
   jxf_APCTLKrylovParamSetCPUTotal(apctlkrylov_param, cpu_total); 
   
      
   //=========================================================
   //  Step 7: 释放内存
   //=========================================================
   
   jxf_ParCSRMatrixDestroy(par_mat);
   jxf_ParVectorDestroy(par_rhs);
   jxf_ParVectorDestroy(par_sol);

   return (0);
}                      

/*!
 * \fn JXF_Int JXF_3tAPCTLSetup4Jasmin
 * \brief Setup phase of PCTL iteration or preconditioner for JASMIN interface.
 * \author peghoty
 * \date 2012/02/26
 */
JXF_Int 
JXF_3tAPCTLSetup4Jasmin( JXF_Solver solver, JXF_ParCSRMatrix A )
{   
   return( jxf_3tAPCTLSetup4Jasmin( (void *) solver, (jxf_ParCSRMatrix *) A ) );
}

JXF_Int 
JXF_3tAPCTLSetup4mgJasmin( JXF_Solver solver, JXF_ParCSRMatrix A )
{
   return( jxf_3tAPCTLSetup4mgJasmin( (void *) solver, (jxf_ParCSRMatrix *) A ) );
}

JXF_Int 
JXF_3tABSC1Setup4mgJasmin( JXF_Solver solver, JXF_ParCSRMatrix A )
{
   return( jxf_3tABSC1Setup4mgJasmin( (void *) solver, (jxf_ParCSRMatrix *) A ) );
}

JXF_Int 
JXF_3tABSC2Setup4mgJasmin( JXF_Solver solver, JXF_ParCSRMatrix A )
{
   return( jxf_3tABSC2Setup4mgJasmin( (void *) solver, (jxf_ParCSRMatrix *) A ) );
}

/*!
 * \fn JXF_Int jxf_3tAPCTLSetup4Jasmin
 * \brief Setup phase of PCTL Iteration or preconditioner for JASMIN interface. 
 * \author peghoty 
 * \date 2012/02/26
 */
JXF_Int
jxf_3tAPCTLSetup4Jasmin( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{  
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);
   
   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);
      
   if (nprocs == 1)
   {
      jxf_3tAPCTLSetup4Jasmin_sp(pre_3tapctl_data, A);
   }
   else if (nprocs > 1)
   { 
      jxf_3tAPCTLSetup4Jasmin_mp(pre_3tapctl_data, A); 
   }
   
   return 0;
}

JXF_Int
jxf_3tAPCTLSetup4mgJasmin( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);
 
   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jxf_3tAPCTLSetup4mgJasmin_mp(pre_3tapctl_data, A);
   }

   return 0;
}

JXF_Int
jxf_3tABSC1Setup4mgJasmin( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);
   JXF_Int nprocs;

   jxf_MPI_Comm_size(comm, &nprocs);

   if ((nprocs > 1) && (nprocs % (jxf_3tAPCTLDataNumGroup(pre_3tapctl_data)+2) == 0))
   {
      jxf_3tABSC1Setup4mgJasmin_mp(pre_3tapctl_data, A);
   }

   return 0;
}

JXF_Int
jxf_3tABSC2Setup4mgJasmin( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);
   JXF_Int nprocs;

   jxf_MPI_Comm_size(comm, &nprocs);

   if ((nprocs > 1) && (nprocs % (jxf_3tAPCTLDataNumGroup(pre_3tapctl_data)+2) == 0))
   {
      jxf_3tABSC2Setup4mgJasmin_mp(pre_3tapctl_data, A);
   }

   return 0;
}

/*!
 * \fn JXF_Int jxf_3tAPCTLSetup4Jasmin_mp
 * \brief Setup phase of PCTL Iteration or preconditioner(for multi-processor case). 
 * \author peghoty 
 * \date 2012/02/25
 */
JXF_Int
jxf_3tAPCTLSetup4Jasmin_mp( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   JXF_Real starttime = 0.0, endtime = 0.0;
     
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

   JXF_Int ARR_interp_maxit = jxf_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);     
   JXF_Int AII_interp_maxit = jxf_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);    
   JXF_Int ACC_relax_maxit  = jxf_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data);

   JXF_Real  ARR_interp_tol = jxf_3tAPCTLDataARRInterpTol(pre_3tapctl_data);   
   JXF_Real  AII_interp_tol = jxf_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JXF_Real  ARR_relax_tol  = jxf_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JXF_Real  AEE_relax_tol  = jxf_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JXF_Real  AII_relax_tol  = jxf_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JXF_Real  ACC_relax_tol  = jxf_3tAPCTLDataACCRelaxTol(pre_3tapctl_data); 

   JXF_Int ARR_solver_id = jxf_3tAPCTLDataARRSolverID(pre_3tapctl_data);

   JXF_Int ACC_solver_id = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   JXF_Int ARR_kdim = jxf_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JXF_Int ACC_kdim = jxf_3tAPCTLDataACCKDim(pre_3tapctl_data);  

   JXF_Int debug_flag = jxf_3tAPCTLDataDebugFlag(pre_3tapctl_data);

   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);  
   JXF_Int blocksmooth_type = jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
   JXF_Int use_ppctl = jxf_3tAPCTLDataUsePPCTL(pre_3tapctl_data);
  
   jxf_ParAMGData   *ARR_amg_solver = NULL;
   jxf_ParAMGData   *AEE_amg_solver = NULL;
   jxf_ParAMGData   *AII_amg_solver = NULL;
   jxf_ParAMGData   *ACC_amg_solver = NULL;

   jxf_GMRESData    *ARR_gmres_solver = NULL;
   jxf_GMRESData    *ACC_gmres_solver = NULL; 

   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data);
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);  
   jxf_ParVector    *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector    *VER = jxf_3tAPCTLDataVER(pre_3tapctl_data);
   jxf_ParVector    *VEI = jxf_3tAPCTLDataVEI(pre_3tapctl_data);
   jxf_ParVector    *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);  

   jxf_ParCSRMatrix *ARR_all = jxf_3tAPCTLDataARRAll(pre_3tapctl_data);
   jxf_ParCSRMatrix *AEE_all = jxf_3tAPCTLDataAEEAll(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII_all = jxf_3tAPCTLDataAIIAll(pre_3tapctl_data);  
   jxf_ParVector    *VRE_all = jxf_3tAPCTLDataVREAll(pre_3tapctl_data);
   jxf_ParVector    *VER_all = jxf_3tAPCTLDataVERAll(pre_3tapctl_data);
   jxf_ParVector    *VEI_all = jxf_3tAPCTLDataVEIAll(pre_3tapctl_data);
   jxf_ParVector    *VIE_all = jxf_3tAPCTLDataVIEAll(pre_3tapctl_data);  
      
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
   
   jxf_ParVector    *PRR = NULL;
   jxf_ParVector    *PII = NULL;

   JXF_Int    IS_DD_R;
   JXF_Int    IS_DD_E;    
   JXF_Int    IS_DD_I;

   JXF_Int ARR_relax_maxit;
   JXF_Int AEE_relax_maxit;
   JXF_Int AII_relax_maxit;
   
   JXF_Int Need_CC;
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data); 

   JXF_Real strong_threshold = jxf_3tAPCTLDataStrongThreshold(pre_3tapctl_data);
   JXF_Int interp_type = jxf_3tAPCTLDataInterpType(pre_3tapctl_data);
   JXF_Int coarsen_type = jxf_3tAPCTLDataCoarsenType(pre_3tapctl_data);
   JXF_Int agg_num_levels = jxf_3tAPCTLDataAggNumLevels(pre_3tapctl_data);
   JXF_Int coarse_threshold = jxf_3tAPCTLDataCoarseThreshold(pre_3tapctl_data);
   JXF_Int print_level_amg = jxf_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data);

   JXF_Int ARR_relax_type = RELAX_AMG;
   JXF_Int AEE_relax_type = RELAX_AMG;
   JXF_Int AII_relax_type = RELAX_AMG;      
   
   JXF_Int *row_starts = NULL;
   JXF_Int *col_starts = NULL;   

   JXF_Int    maxit_default = 200;
   JXF_Real tol_default   = 1.0e-6;
 
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / 3;
   JXF_Real temp_adrress = 0.0;

   JXF_Int np_R = jxf_3tAPCTLDataNpR(pre_3tapctl_data);
   JXF_Int np_E = jxf_3tAPCTLDataNpE(pre_3tapctl_data); 
   
   JXF_Int rootid_R = 0; 
   JXF_Int rootid_E = np_R;
   JXF_Int rootid_I = np_R + np_E; 
   
   MPI_Comm comm   = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jxf_3tAPCTLDataCommX(pre_3tapctl_data); 

   char MatFile[255];
   JXF_Int myid, nprocs;
   JXF_Int groupid_x = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);
      
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);
   
   if (!use_ppctl)
   {
      if (print_level == 1 || print_level == 3)
      {       
         starttime = jxf_MPI_Wtime();
      }
     

      if (groupid_x == 1)
      {  
         Need_CC = jxf_3tAPCTLWeakCouplingE(theta_wc_E, threshold_wc_E, AEE, VER, VEI); 
      }
      jxf_MPI_Bcast(&Need_CC, 1, JXF_MPI_INT, rootid_E, comm);
      jxf_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      if (print_level && myid == 0) jxf_printf(" >> Need_CC = %d\n", Need_CC);
  
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

      if (groupid_x == 0)
      {  
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jxf_ParVectorDestroy(VRE);
         }
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
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jxf_ParVectorDestroy(VER);
            jxf_ParVectorDestroy(VEI);
         }
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
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jxf_ParVectorDestroy(VIE);
         }
         IS_DD_I = jxf_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jxf_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jxf_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I)
         {      
            AII_relax_type = RELAX_WJACOBI;        
         }
         jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;  
      }
   
      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "APCTL == Check WC-DD cond", starttime, endtime, 0, 3);
      } 
   }
   else
   {
      Need_CC = 1;
      ARR_solver_id = SOLVER_AMG;
      ACC_solver_id = SOLVER_AMG;

      ARR_relax_type = RELAX_AMG;
      AEE_relax_type = RELAX_AMG;
      AII_relax_type = RELAX_AMG;

      ARR_relax_maxit = maxit_default;
      AEE_relax_maxit = maxit_default;
      AII_relax_maxit = maxit_default;
      ACC_relax_maxit = maxit_default;

      ARR_relax_tol = tol_default;
      AEE_relax_tol = tol_default;
      AII_relax_tol = tol_default;
      ACC_relax_tol = tol_default; 
    
      jxf_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;
      jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;
      jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;
      jxf_3tAPCTLSetACCSolverID(pre_3tapctl_data, ACC_solver_id); 
      jxf_3tAPCTLSetARRSolverID(pre_3tapctl_data, ARR_solver_id);     
      jxf_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
      jxf_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
      jxf_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);
      jxf_3tAPCTLSetACCRelaxMaxIt(pre_3tapctl_data, ACC_relax_maxit);
      jxf_3tAPCTLSetARRRelaxTol(pre_3tapctl_data, tol_default);
      jxf_3tAPCTLSetAEERelaxTol(pre_3tapctl_data, tol_default);
      jxf_3tAPCTLSetAIIRelaxTol(pre_3tapctl_data, tol_default);
      jxf_3tAPCTLSetACCRelaxTol(pre_3tapctl_data, tol_default);
      
      jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data) = 0;
      jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data) = 0;
      jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data) = 0;            
   }    


   if (Need_CC == 1)
   {

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
         jxf_ParVectorSetPartitioningOwner(WII, 0); 
         jxf_ParVectorSetConstantValues(WII, -1.0);
  
         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         if (ARR_solver_id == SOLVER_AMG)
         {
            ARR_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_interp_maxit); 
            jxf_PAMGSetTol(ARR_amg_solver, ARR_interp_tol); 
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
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
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
         
            ARR_gmres_solver = jxf_ParCSRGMRESCreate(comm_x);
            jxf_GMRESSetKDim(ARR_gmres_solver, ARR_kdim);
            jxf_GMRESSetMaxIter(ARR_gmres_solver, ARR_interp_maxit);
            jxf_GMRESSetTol(ARR_gmres_solver, ARR_interp_tol);
            jxf_GMRESSetPrintLevel(ARR_gmres_solver, 0);
            jxf_GMRESSetIsCheckRestarted(ARR_gmres_solver, 0); 
         
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
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
            jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 0.0);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jxf_PAMGDestroy(ARR_amg_solver);
            ARR_amg_solver = NULL;
            JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);            
         }      
      } 
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
            jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
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
         jxf_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
         jxf_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
         jxf_PAMGSetInterpType(AII_amg_solver, interp_type);
         jxf_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
         jxf_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
         jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
         jxf_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
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
   else
   {

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
            jxf_ParVectorSetPartitioningOwner(WII, 0);
            jxf_ParVectorSetConstantValues(WII, -1.0); 
              
            if (print_level == 1 || print_level == 3)
            {
               starttime = jxf_MPI_Wtime();
            }

            if (ARR_relax_type == RELAX_AMG)
            {
               ARR_amg_solver = jxf_PAMGCreate();
               jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
               jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol); 
               jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
               jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
               jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
               jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
               jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
               jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
               jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
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
               jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
               jxf_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
               jxf_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
               jxf_PAMGSetInterpType(AEE_amg_solver, interp_type);
               jxf_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
               jxf_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
               jxf_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
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
      else if (groupid_x == 2)
      {
            RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
            jxf_ParVectorInitialize(RHS);
            jxf_ParVectorSetPartitioningOwner(RHS, 0);

            WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
            jxf_ParVectorInitialize(WRR);
            jxf_ParVectorSetPartitioningOwner(WRR, 0);
            jxf_ParVectorSetConstantValues(WRR, -1.0);

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
               jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
               jxf_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
               jxf_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
               jxf_PAMGSetInterpType(AII_amg_solver, interp_type);
               jxf_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
               jxf_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
               jxf_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
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

   } 
    
   jxf_MPI_Barrier(comm);
    
  
   if (Need_CC == 1)
   {
      if (print_level == 1 || print_level == 3)
      {       
         starttime = jxf_MPI_Wtime();
      }
  
      row_starts = jxf_ParCSRMatrixRowStarts(ARR_all);
      jxf_3tAPCTLParaInterpVec(comm, comm_x, groupid_x, row_starts, WEE, &PRR, &PII);

      ACC = jxf_3tApctlCoarseOperator_mp(comm, ARR_all, AEE_all, AII_all, 
                                        VRE_all, VER_all, VEI_all, VIE_all, PRR, PII);
                                 
      jxf_ParVectorDestroy(PRR);
      jxf_ParVectorDestroy(PII);                                  

      if ((debug_flag == 1) || (debug_flag == 3))
      {
          jxf_sprintf(MatFile,"%s.%02d", "./3t.acc", jxf_total_iter_index);
          jxf_ParCSRMatrixPrint(ACC, MatFile);
          jxf_total_iter_index ++;
      }

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
      col_starts = jxf_ParCSRMatrixRowStarts(ARR_all); 
      P = jxf_3tApctlInterpolation(comm, comm_x, N, groupid_x, row_starts, col_starts, WEE);

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "APCTL ==  Interp Operator", starttime, endtime, 0, 3);
      }

      GCC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ACC));
      jxf_ParVectorInitialize(GCC);
      jxf_ParVectorSetPartitioningOwner(GCC, 0);  

      WCC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ACC));
      jxf_ParVectorInitialize(WCC);
      jxf_ParVectorSetPartitioningOwner(WCC, 0);        

      RES = jxf_ParVectorCreate(comm, N, jxf_ParCSRMatrixRowStarts(A));
      jxf_ParVectorInitialize(RES);
      jxf_ParVectorSetPartitioningOwner(RES, 0); 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jxf_MPI_Wtime();
      }

      if (ACC_solver_id == SOLVER_AMG)
      {
         ACC_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(ACC_amg_solver, ACC_relax_maxit); 
         jxf_PAMGSetTol(ACC_amg_solver, ACC_relax_tol); 
         jxf_PAMGSetRhsNrmThreshold(ACC_amg_solver, 1.0); 
         jxf_PAMGSetStrongThreshold(ACC_amg_solver, strong_threshold);
         jxf_PAMGSetCoarsenType(ACC_amg_solver, coarsen_type);
         jxf_PAMGSetInterpType(ACC_amg_solver, interp_type);
         jxf_PAMGSetAggNumLevels(ACC_amg_solver, agg_num_levels);
         jxf_PAMGSetCoarseThreshold(ACC_amg_solver, coarse_threshold);
         jxf_PAMGSetPrintLevel(ACC_amg_solver, print_level_amg);
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
         jxf_PAMGSetStrongThreshold(ACC_amg_solver, strong_threshold);
         jxf_PAMGSetCoarsenType(ACC_amg_solver, coarsen_type);
         jxf_PAMGSetInterpType(ACC_amg_solver, interp_type);
         jxf_PAMGSetAggNumLevels(ACC_amg_solver, agg_num_levels);
         jxf_PAMGSetCoarseThreshold(ACC_amg_solver, coarse_threshold);
         jxf_PAMGSetPrintLevel(ACC_amg_solver, print_level_amg);
         
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
   
   }

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

JXF_Int
jxf_3tAPCTLSetup4mgJasmin_mp( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   JXF_Real starttime = 0.0, endtime = 0.0;

   JXF_Int ng = jxf_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JXF_Int fixit_pctl_R = jxf_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JXF_Int fixit_pctl_E = jxf_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JXF_Int fixit_pctl_I = jxf_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JXF_Int fixit_brlx_R = jxf_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JXF_Int fixit_brlx_E = jxf_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JXF_Int fixit_brlx_I = jxf_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JXF_Real theta_wc_E = jxf_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JXF_Real threshold_wc_E = jxf_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JXF_Real theta_dd_R = jxf_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JXF_Real theta_dd_E = jxf_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JXF_Real theta_dd_I = jxf_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JXF_Real threshold_dd_R = jxf_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JXF_Real threshold_dd_E = jxf_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JXF_Real threshold_dd_I = jxf_3tAPCTLDataThresholdDDI(pre_3tapctl_data);

   JXF_Int ARR_interp_maxit = jxf_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);
   JXF_Int AII_interp_maxit = jxf_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);
   JXF_Int ACC_relax_maxit  = jxf_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data);

   JXF_Real ARR_interp_tol = jxf_3tAPCTLDataARRInterpTol(pre_3tapctl_data);
   JXF_Real AII_interp_tol = jxf_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JXF_Real ARR_relax_tol = jxf_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JXF_Real AEE_relax_tol = jxf_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JXF_Real AII_relax_tol = jxf_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JXF_Real ACC_relax_tol = jxf_3tAPCTLDataACCRelaxTol(pre_3tapctl_data);

   JXF_Int ARR_solver_id = jxf_3tAPCTLDataARRSolverID(pre_3tapctl_data);
   JXF_Int ACC_solver_id = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   JXF_Int ARR_kdim = jxf_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JXF_Int ACC_kdim = jxf_3tAPCTLDataACCKDim(pre_3tapctl_data);

   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);
   JXF_Int blocksmooth_type = jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
   JXF_Int use_ppctl = jxf_3tAPCTLDataUsePPCTL(pre_3tapctl_data);
   JXF_Int debug_flag = jxf_3tAPCTLDataDebugFlag(pre_3tapctl_data);

   JXF_Real strong_threshold = jxf_3tAPCTLDataStrongThreshold(pre_3tapctl_data);
   JXF_Int interp_type = jxf_3tAPCTLDataInterpType(pre_3tapctl_data);
   JXF_Int coarsen_type = jxf_3tAPCTLDataCoarsenType(pre_3tapctl_data);
   JXF_Int agg_num_levels = jxf_3tAPCTLDataAggNumLevels(pre_3tapctl_data);
   JXF_Int coarse_threshold = jxf_3tAPCTLDataCoarseThreshold(pre_3tapctl_data);
   JXF_Int print_level_amg = jxf_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data);

   JXF_Int maxit_default = jxf_3tAPCTLDataMaxItDefault(pre_3tapctl_data);
   JXF_Real tol_default = jxf_3tAPCTLDataTolDefault(pre_3tapctl_data);

   JXF_Int ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JXF_Int AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JXF_Int AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   jxf_ParAMGData *ARR_amg_solver = NULL;
   jxf_ParAMGData *AEE_amg_solver = NULL;
   jxf_ParAMGData *AII_amg_solver = NULL;
   jxf_ParAMGData *ACC_amg_solver = NULL;

   jxf_GMRESData *ARR_gmres_solver = NULL;
   jxf_GMRESData *ACC_gmres_solver = NULL;

   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data);
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);

   jxf_ParVector  *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector **VER = jxf_3tAPCTLDataVER2(pre_3tapctl_data);
   jxf_ParVector  *VEI = jxf_3tAPCTLDataVEI(pre_3tapctl_data);
   jxf_ParVector  *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);

   jxf_ParCSRMatrix **ARR_all = jxf_3tAPCTLDataARRAll2(pre_3tapctl_data);
   jxf_ParCSRMatrix  *AEE_all = jxf_3tAPCTLDataAEEAll(pre_3tapctl_data);
   jxf_ParCSRMatrix  *AII_all = jxf_3tAPCTLDataAIIAll(pre_3tapctl_data);

   jxf_ParVector **VRE_all = jxf_3tAPCTLDataVREAll2(pre_3tapctl_data);
   jxf_ParVector **VER_all = jxf_3tAPCTLDataVERAll2(pre_3tapctl_data);
   jxf_ParVector  *VEI_all = jxf_3tAPCTLDataVEIAll(pre_3tapctl_data);
   jxf_ParVector  *VIE_all = jxf_3tAPCTLDataVIEAll(pre_3tapctl_data);

   jxf_ParCSRMatrix *P   = NULL;
   jxf_ParCSRMatrix *ACC = NULL;

   jxf_ParVector *WRR = NULL;
   jxf_ParVector *WEE = NULL;
   jxf_ParVector *WII = NULL;
   jxf_ParVector *WCC = NULL;

   jxf_ParVector *GCC = NULL;

   jxf_ParVector *RES = NULL;
   jxf_ParVector *RHS = NULL;
   jxf_ParVector *JAC = NULL;

   jxf_ParVector **PRR = NULL;
   jxf_ParVector  *PII = NULL;

   JXF_Int IS_DD_R;
   JXF_Int IS_DD_E;
   JXF_Int IS_DD_I;

   JXF_Int ARR_relax_maxit;
   JXF_Int AEE_relax_maxit;
   JXF_Int AII_relax_maxit;

   JXF_Int Need_CC;
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   JXF_Int *row_starts = NULL;
   JXF_Int *col_starts = NULL;

   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / (ng + 2);
   JXF_Real temp_adrress = 0.0;

   MPI_Comm comm   = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jxf_3tAPCTLDataCommX(pre_3tapctl_data);

   char MatFile[255];
   JXF_Int myid, nprocs, gidx;
   JXF_Int groupid_x = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JXF_Int np_R = jxf_3tAPCTLDataNpR(pre_3tapctl_data);
   JXF_Int rootid_R = groupid_x * np_R;
   JXF_Int rootid_E = np_R * ng;
   JXF_Int rootid_I = np_R + rootid_E;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (!use_ppctl)
   {
      if (print_level == 1 || print_level == 3)
      {
         starttime = jxf_MPI_Wtime();
      }

      if (groupid_x == ng)
      {
         Need_CC = jxf_3tAPCTLmgWeakCouplingE(theta_wc_E, threshold_wc_E, ng, AEE, VER, VEI);
      }
      jxf_MPI_Bcast(&Need_CC, 1, JXF_MPI_INT, rootid_E, comm);
      jxf_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      if (print_level && myid == 0) jxf_printf(" >> Need_CC = %d\n", Need_CC);

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

      if (groupid_x < ng)
      {
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jxf_ParVectorDestroy(VRE);
         }
         IS_DD_R = jxf_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jxf_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jxf_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;
      }
      else if (groupid_x == ng)
      {
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            for (gidx = 0; gidx < ng; gidx ++) jxf_ParVectorDestroy(VER[gidx]);
            jxf_TFree(VER);
            jxf_ParVectorDestroy(VEI);
         }
         IS_DD_E = jxf_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jxf_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jxf_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;
      }
      else if (groupid_x == ng+1)
      {
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jxf_ParVectorDestroy(VIE);
         }
         IS_DD_I = jxf_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jxf_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jxf_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;
      }

      if (print_level == 1 || print_level == 3)
      {
         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "APCTL == Check WC-DD cond", starttime, endtime, 0, 3);
      }
   }
   else
   {
      Need_CC = 1;
      ARR_solver_id = SOLVER_AMG;
      ACC_solver_id = SOLVER_AMG;

      ARR_relax_type = RELAX_AMG;
      AEE_relax_type = RELAX_AMG;
      AII_relax_type = RELAX_AMG;

      ARR_relax_maxit = maxit_default;
      AEE_relax_maxit = maxit_default;
      AII_relax_maxit = maxit_default;
      ACC_relax_maxit = 1; //maxit_default;

      ARR_relax_tol = tol_default;
      AEE_relax_tol = tol_default;
      AII_relax_tol = tol_default;
      ACC_relax_tol = tol_default;

      jxf_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;
      jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;
      jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;
      jxf_3tAPCTLSetACCSolverID(pre_3tapctl_data, ACC_solver_id);
      jxf_3tAPCTLSetARRSolverID(pre_3tapctl_data, ARR_solver_id);
      jxf_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
      jxf_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
      jxf_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);
      jxf_3tAPCTLSetACCRelaxMaxIt(pre_3tapctl_data, ACC_relax_maxit);
      jxf_3tAPCTLSetARRRelaxTol(pre_3tapctl_data, tol_default);
      jxf_3tAPCTLSetAEERelaxTol(pre_3tapctl_data, tol_default);
      jxf_3tAPCTLSetAIIRelaxTol(pre_3tapctl_data, tol_default);
      jxf_3tAPCTLSetACCRelaxTol(pre_3tapctl_data, tol_default);

      jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data) = 0;
      jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data) = 0;
      jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data) = 0;
   }

   if (Need_CC == 1)
   {
      if (groupid_x < ng)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WII);
         jxf_ParVectorSetPartitioningOwner(WII, 0);
         jxf_ParVectorSetConstantValues(WII, -1.0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         if (ARR_solver_id == SOLVER_AMG)
         {
            ARR_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_interp_maxit);
            jxf_PAMGSetTol(ARR_amg_solver, ARR_interp_tol);
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
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
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
            ARR_gmres_solver = jxf_ParCSRGMRESCreate(comm_x);
            jxf_GMRESSetKDim(ARR_gmres_solver, ARR_kdim);
            jxf_GMRESSetMaxIter(ARR_gmres_solver, ARR_interp_maxit);
            jxf_GMRESSetTol(ARR_gmres_solver, ARR_interp_tol);
            jxf_GMRESSetPrintLevel(ARR_gmres_solver, 0);
            jxf_GMRESSetIsCheckRestarted(ARR_gmres_solver, 0);
            jxf_GMRESSetPrecond( ARR_gmres_solver, jxf_PAMGPrecond, jxf_PAMGSetup, ARR_amg_solver );
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

         jxf_ParVectorDestroy(WII);

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
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jxf_PAMGDestroy(ARR_amg_solver);
            ARR_amg_solver = NULL;
            JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);
         }
      }
      else if (groupid_x == ng)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetDataOwner(WEE, 0);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jxf_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jxf_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
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
      else if (groupid_x == ng+1)
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
         jxf_ParVectorInitialize(WII);
         jxf_ParVectorSetPartitioningOwner(WII, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         AII_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(AII_amg_solver, AII_interp_maxit);
         jxf_PAMGSetTol(AII_amg_solver, AII_interp_tol);
         jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
         jxf_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
         jxf_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
         jxf_PAMGSetInterpType(AII_amg_solver, interp_type);
         jxf_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
         jxf_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
         jxf_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
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

         jxf_ParVectorDestroy(WRR);

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
   else
   {
      if (groupid_x < ng)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
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
      else if (groupid_x == ng)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetDataOwner(WEE, 0);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
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
      else if (groupid_x == ng+1)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WII);
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
            jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AII_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
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
   }

   jxf_MPI_Barrier(comm); //  各进程(组)同步

   if (Need_CC == 1)
   {
      if (print_level == 1 || print_level == 3)
      {
         starttime = jxf_MPI_Wtime();
      }

      row_starts = jxf_ParCSRMatrixRowStarts(ARR_all[0]);
      jxf_3tAPCTLmgParaInterpVec(comm, comm_x, groupid_x, row_starts, ng, WEE, &PRR, &PII);
      ACC = jxf_3tApctlmgCoarseOperator_mp(comm, ng, ARR_all, AEE_all, AII_all, VRE_all, VER_all, VEI_all, VIE_all, PRR, PII);
      for (gidx = 0; gidx < ng; gidx ++) jxf_ParVectorDestroy(PRR[gidx]);
      jxf_TFree(PRR);
      jxf_ParVectorDestroy(PII);

      if ((debug_flag == 1) || (debug_flag == 3))
      {
          jxf_sprintf(MatFile,"%s.%02d", "./mg.acc", jxf_total_iter_index);
          jxf_ParCSRMatrixPrint(ACC, MatFile);
          jxf_total_iter_index ++;
      }

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
      col_starts = jxf_ParCSRMatrixRowStarts(ARR_all[0]);
      P = jxf_3tApctlmgInterpolation(comm, comm_x, N, groupid_x, ng, row_starts, col_starts, WEE);

      if (print_level == 1 || print_level == 3)
      {
         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "APCTL ==  Interp Operator", starttime, endtime, 0, 3);
      }

      GCC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ACC));
      jxf_ParVectorInitialize(GCC);
      jxf_ParVectorSetPartitioningOwner(GCC, 0);

      WCC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ACC));
      jxf_ParVectorInitialize(WCC);
      jxf_ParVectorSetPartitioningOwner(WCC, 0);

      RES = jxf_ParVectorCreate(comm, N, jxf_ParCSRMatrixRowStarts(A));
      jxf_ParVectorInitialize(RES);
      jxf_ParVectorSetPartitioningOwner(RES, 0);

      if (print_level == 1 || print_level == 3)
      {
         starttime = jxf_MPI_Wtime();
      }

      if (ACC_solver_id == SOLVER_AMG)
      {
         ACC_amg_solver = jxf_PAMGCreate();
         jxf_PAMGSetMaxIter(ACC_amg_solver, ACC_relax_maxit);
         jxf_PAMGSetTol(ACC_amg_solver, ACC_relax_tol);
         jxf_PAMGSetRhsNrmThreshold(ACC_amg_solver, 1.0);
         jxf_PAMGSetPrintLevel(ACC_amg_solver, print_level_amg);
         jxf_PAMGSetStrongThreshold(ACC_amg_solver, strong_threshold);
         jxf_PAMGSetCoarsenType(ACC_amg_solver, coarsen_type);
         jxf_PAMGSetInterpType(ACC_amg_solver, interp_type);
         jxf_PAMGSetAggNumLevels(ACC_amg_solver, agg_num_levels);
         jxf_PAMGSetCoarseThreshold(ACC_amg_solver, coarse_threshold);
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
         jxf_PAMGSetStrongThreshold(ACC_amg_solver, strong_threshold);
         jxf_PAMGSetCoarsenType(ACC_amg_solver, coarsen_type);
         jxf_PAMGSetInterpType(ACC_amg_solver, interp_type);
         jxf_PAMGSetAggNumLevels(ACC_amg_solver, agg_num_levels);
         jxf_PAMGSetCoarseThreshold(ACC_amg_solver, coarse_threshold);
         jxf_PAMGSetPrintLevel(ACC_amg_solver, print_level_amg);
         ACC_gmres_solver = jxf_ParCSRGMRESCreate(comm);
         jxf_GMRESSetKDim(ACC_gmres_solver, ACC_kdim);
         jxf_GMRESSetMaxIter(ACC_gmres_solver, ACC_relax_maxit);
         jxf_GMRESSetTol(ACC_gmres_solver, ACC_relax_tol);
         jxf_GMRESSetPrintLevel(ACC_gmres_solver, 0);
         jxf_GMRESSetPrecond( ACC_gmres_solver, jxf_PAMGPrecond, jxf_PAMGSetup, ACC_amg_solver );
         jxf_PAMGSetup(ACC_amg_solver, ACC);
         jxf_GMRESSetup(ACC_gmres_solver, ACC, WCC, WCC);

         if (print_level == 1 || print_level == 3)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "APCTL == AMGGMRESSetup for ACC", starttime, endtime, 0, 3);
         }
      }
   }

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

JXF_Int
jxf_3tABSC1Setup4mgJasmin_mp( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   JXF_Real starttime = 0.0, endtime = 0.0;

   JXF_Int ng = jxf_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JXF_Int fixit_pctl_R = jxf_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JXF_Int fixit_pctl_E = jxf_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JXF_Int fixit_pctl_I = jxf_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JXF_Int fixit_brlx_R = jxf_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JXF_Int fixit_brlx_E = jxf_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JXF_Int fixit_brlx_I = jxf_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JXF_Real theta_wc_E = jxf_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JXF_Real threshold_wc_E = jxf_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JXF_Real theta_dd_R = jxf_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JXF_Real theta_dd_E = jxf_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JXF_Real theta_dd_I = jxf_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JXF_Real threshold_dd_R = jxf_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JXF_Real threshold_dd_E = jxf_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JXF_Real threshold_dd_I = jxf_3tAPCTLDataThresholdDDI(pre_3tapctl_data);

   JXF_Real ARR_relax_tol = jxf_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JXF_Real AEE_relax_tol = jxf_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JXF_Real AII_relax_tol = jxf_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);

   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);

   JXF_Real strong_threshold = jxf_3tAPCTLDataStrongThreshold(pre_3tapctl_data);
   JXF_Int interp_type = jxf_3tAPCTLDataInterpType(pre_3tapctl_data);
   JXF_Int coarsen_type = jxf_3tAPCTLDataCoarsenType(pre_3tapctl_data);
   JXF_Int agg_num_levels = jxf_3tAPCTLDataAggNumLevels(pre_3tapctl_data);
   JXF_Int coarse_threshold = jxf_3tAPCTLDataCoarseThreshold(pre_3tapctl_data);
   JXF_Int print_level_amg = jxf_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data);

   JXF_Int ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JXF_Int AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JXF_Int AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   jxf_ParAMGData *ARR_amg_solver = NULL;
   jxf_ParAMGData *AEE_amg_solver = NULL;
   jxf_ParAMGData *AII_amg_solver = NULL;

   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data);
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);

   jxf_ParVector  *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector **VER = jxf_3tAPCTLDataVER2(pre_3tapctl_data);
   jxf_ParVector  *VEI = jxf_3tAPCTLDataVEI(pre_3tapctl_data);
   jxf_ParVector  *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);

   jxf_ParVector *WRR = NULL;
   jxf_ParVector *WEE = NULL;
   jxf_ParVector *WII = NULL;
   jxf_ParVector *RHS = NULL;
   jxf_ParVector *JAC = NULL;

   JXF_Int Need_CC, sr_hsize = 0;
   JXF_Int IS_DD_R, IS_DD_E, IS_DD_I;
   JXF_Int ARR_relax_maxit, AEE_relax_maxit, AII_relax_maxit;

   MPI_Status status;

   JXF_Real *rTEMP = NULL;
   JXF_Real *sTEMP = NULL;
   JXF_Real *vTEMP = NULL;
   JXF_Real *tTEMP = NULL;

   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / (ng + 2);
   JXF_Real temp_adrress = 0.0;

   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jxf_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jxf_3tAPCTLDataCommY(pre_3tapctl_data);

   JXF_Int myid, nprocs, gidx, send_cnt, j;
   JXF_Int groupid_x = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JXF_Int np_R = jxf_3tAPCTLDataNpR(pre_3tapctl_data);
   JXF_Int rootid_R = groupid_x * np_R;
   JXF_Int rootid_E = np_R * ng;
   JXF_Int rootid_I = np_R + rootid_E;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (groupid_x < ng)
   {
      sr_hsize = jxf_CSRMatrixNumRows(jxf_ParCSRMatrixDiag(ARR));
   }
   else if (groupid_x == ng)
   {
      sr_hsize = jxf_CSRMatrixNumRows(jxf_ParCSRMatrixDiag(AEE));
   }
   else if (groupid_x == ng+1)
   {
      sr_hsize = jxf_CSRMatrixNumRows(jxf_ParCSRMatrixDiag(AII));
   }

   rTEMP = jxf_CTAlloc(JXF_Real, 2*sr_hsize);
   sTEMP = jxf_CTAlloc(JXF_Real, sr_hsize);
   vTEMP = jxf_CTAlloc(JXF_Real, sr_hsize);

   if (print_level == 1 || print_level == 3)
   {
      starttime = jxf_MPI_Wtime();
   }

   if (groupid_x == ng)
   {
      Need_CC = jxf_3tAPCTLmgWeakCouplingE(theta_wc_E, threshold_wc_E, ng, AEE, VER, VEI);
   }
   jxf_MPI_Bcast(&Need_CC, 1, JXF_MPI_INT, rootid_E, comm);
   jxf_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
   if (print_level && myid == 0) jxf_printf(" >> Need_CC = %d\n", Need_CC);

   if (Need_CC)
   {
      ARR_relax_maxit = fixit_pctl_R;
      AEE_relax_maxit = fixit_pctl_E;
      AII_relax_maxit = fixit_pctl_I;
   }
   else
   {
      if (groupid_x < ng)
      {
         jxf_ParVectorDestroy(VRE);
      }
      else if (groupid_x == ng)
      {
         for (gidx = 0; gidx < ng; gidx ++) jxf_ParVectorDestroy(VER[gidx]);
         jxf_TFree(VER);
         jxf_ParVectorDestroy(VEI);
      }
      else if (groupid_x == ng+1)
      {
         jxf_ParVectorDestroy(VIE);
      }

      ARR_relax_maxit = fixit_brlx_R;
      AEE_relax_maxit = fixit_brlx_E;
      AII_relax_maxit = fixit_brlx_I;
   }
   jxf_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
   jxf_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
   jxf_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);

   if (print_level == 1 || print_level == 3)
   {
      endtime = jxf_MPI_Wtime();
      jxf_GetWallTime(comm, "APCTL == Check WC-DD cond", starttime, endtime, 0, 3);
   }

   if (Need_CC)
   {
      if (groupid_x == ng)
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

         jxf_MPI_Recv(rTEMP, 2*sr_hsize, JXF_MPI_REAL, myid+np_R, 123, comm, &status);

         jxf_BSCModifySubMat(AEE, VEI, rTEMP, sr_hsize); // update AEE

         IS_DD_E = jxf_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jxf_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jxf_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jxf_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jxf_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
            jxf_PAMGSetup(AEE_amg_solver, AEE);
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);
         }

#if 0
         dest = myid - rootid_E;
         for (j = 0; j < ng; j ++)
         {
            dTEMP[j] = jxf_CTAlloc(JXF_Real, 2*sr_hsize);
            for (gidx = 0; gidx < sr_hsize; gidx ++)
            {
               dTEMP[j][gidx] = jxf_CSRMatrixData(jxf_ParCSRMatrixDiag(AEE))[jxf_CSRMatrixI(jxf_ParCSRMatrixDiag(AEE))[gidx]];
            }
            send_cnt = sr_hsize;
            for (gidx = 0; gidx < sr_hsize; gidx ++)
            {
               dTEMP[j][send_cnt++] = jxf_VectorData(jxf_ParVectorLocalVector(VER[j]))[gidx];
            }
            jxf_MPI_Send(dTEMP[j], 2*sr_hsize, JXF_MPI_REAL, dest, dest*321, comm);
            dest += np_R;
         }
#else
         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            sTEMP[gidx] = jxf_CSRMatrixData(jxf_ParCSRMatrixDiag(AEE))[jxf_CSRMatrixI(jxf_ParCSRMatrixDiag(AEE))[gidx]];
         }
         tTEMP = jxf_CTAlloc(JXF_Real, (ng+2)*sr_hsize);
         send_cnt = 0;
         for (j = 0; j < ng; j ++)
         {
            for (gidx = 0; gidx < sr_hsize; gidx ++)
            {
               tTEMP[send_cnt++] = jxf_VectorData(jxf_ParVectorLocalVector(VER[j]))[gidx];
            }
         }
#endif

         if (print_level == 1 || print_level == 3)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng+1)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WII);
         jxf_ParVectorSetPartitioningOwner(WII, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         send_cnt = 0;
         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            rTEMP[send_cnt++] = jxf_CSRMatrixData(jxf_ParCSRMatrixDiag(AII))[jxf_CSRMatrixI(jxf_ParCSRMatrixDiag(AII))[gidx]];
         }
         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            rTEMP[send_cnt++] = jxf_VectorData(jxf_ParVectorLocalVector(VIE))[gidx];
         }
         jxf_MPI_Send(rTEMP, 2*sr_hsize, JXF_MPI_REAL, myid-np_R, 123, comm);

         IS_DD_I = jxf_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jxf_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jxf_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

         if (AII_relax_type == RELAX_AMG)
         {
            AII_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit);
            jxf_PAMGSetTol(AII_amg_solver, AII_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AII_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
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

   jxf_MPI_Barrier(comm); //  各进程(组)同步

      jxf_MPI_Bcast(sTEMP, sr_hsize, JXF_MPI_REAL, ng, comm_y); // yue: invalid for comm_I
      jxf_MPI_Scatter(tTEMP, sr_hsize, JXF_MPI_REAL, vTEMP, sr_hsize, JXF_MPI_REAL, ng, comm_y); // yue: invalid for comm_E and comm_I

      if (groupid_x < ng)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         //jxf_MPI_Recv(rTEMP, 2*sr_hsize, JXF_MPI_REAL, myid, myid*321, comm, &status);

         jxf_BSCModify2SubMat(ARR, VRE, sTEMP, vTEMP, sr_hsize); // update ARR

         IS_DD_R = jxf_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jxf_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jxf_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
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
   }
   else // BD
   {
      if (groupid_x < ng)
      {
         IS_DD_R = jxf_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jxf_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jxf_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
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
            jxf_GetWallTime(comm_x, "BD == AMGSetup for ARR", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng)
      {
         IS_DD_E = jxf_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jxf_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jxf_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetDataOwner(WEE, 0);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
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
            jxf_GetWallTime(comm_x, "BD == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng+1)
      {
         IS_DD_I = jxf_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jxf_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jxf_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WII);
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
            jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AII_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
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
            jxf_GetWallTime(comm_x, "BD == AMGSetup for AII", starttime, endtime, 0, 3);
         }
      }
   }

   jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) = ARR_amg_solver;
   jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) = AEE_amg_solver;
   jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) = AII_amg_solver;

   jxf_3tAPCTLDataWRR(pre_3tapctl_data) = WRR;
   jxf_3tAPCTLDataWEE(pre_3tapctl_data) = WEE;
   jxf_3tAPCTLDataWII(pre_3tapctl_data) = WII;

   jxf_3tAPCTLDataRHS(pre_3tapctl_data) = RHS;
   jxf_3tAPCTLDataJAC(pre_3tapctl_data) = JAC;

   jxf_3tAPCTLDatatTEMP(pre_3tapctl_data) = tTEMP;

   jxf_TFree(rTEMP);
   jxf_TFree(sTEMP);
   jxf_TFree(vTEMP);
#if 0
   for (j = 0; j < ng; j ++)
   {
      if (dTEMP[j]) jxf_TFree(dTEMP[j]);
   }
   jxf_TFree(dTEMP);
#endif

   return (0);
}

JXF_Int
jxf_3tABSC2Setup4mgJasmin_mp( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   JXF_Real starttime = 0.0, endtime = 0.0;

   JXF_Int ng = jxf_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JXF_Int fixit_pctl_R = jxf_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JXF_Int fixit_pctl_E = jxf_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JXF_Int fixit_pctl_I = jxf_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JXF_Int fixit_brlx_R = jxf_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JXF_Int fixit_brlx_E = jxf_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JXF_Int fixit_brlx_I = jxf_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JXF_Real theta_wc_E = jxf_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JXF_Real threshold_wc_E = jxf_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JXF_Real theta_dd_R = jxf_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JXF_Real theta_dd_E = jxf_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JXF_Real theta_dd_I = jxf_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JXF_Real threshold_dd_R = jxf_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JXF_Real threshold_dd_E = jxf_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JXF_Real threshold_dd_I = jxf_3tAPCTLDataThresholdDDI(pre_3tapctl_data);

   JXF_Real ARR_relax_tol = jxf_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JXF_Real AEE_relax_tol = jxf_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JXF_Real AII_relax_tol = jxf_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);

   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);

   JXF_Real strong_threshold = jxf_3tAPCTLDataStrongThreshold(pre_3tapctl_data);
   JXF_Int interp_type = jxf_3tAPCTLDataInterpType(pre_3tapctl_data);
   JXF_Int coarsen_type = jxf_3tAPCTLDataCoarsenType(pre_3tapctl_data);
   JXF_Int agg_num_levels = jxf_3tAPCTLDataAggNumLevels(pre_3tapctl_data);
   JXF_Int coarse_threshold = jxf_3tAPCTLDataCoarseThreshold(pre_3tapctl_data);
   JXF_Int print_level_amg = jxf_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data);

   JXF_Int ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JXF_Int AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JXF_Int AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   jxf_ParAMGData *ARR_amg_solver = NULL;
   jxf_ParAMGData *AEE_amg_solver = NULL;
   jxf_ParAMGData *AII_amg_solver = NULL;

   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data);
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);

   jxf_ParVector  *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector **VER = jxf_3tAPCTLDataVER2(pre_3tapctl_data);
   jxf_ParVector  *VEI = jxf_3tAPCTLDataVEI(pre_3tapctl_data);
   jxf_ParVector  *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);

   jxf_ParVector *WRR = NULL;
   jxf_ParVector *WEE = NULL;
   jxf_ParVector *WII = NULL;
   jxf_ParVector *RHS = NULL;
   jxf_ParVector *JAC = NULL;

   JXF_Int Need_CC, sr_hsize = 0;
   JXF_Int IS_DD_R, IS_DD_E, IS_DD_I;
   JXF_Int ARR_relax_maxit, AEE_relax_maxit, AII_relax_maxit;

   JXF_Real *sTEMP = NULL;
   JXF_Real *vTEMP = NULL;
   JXF_Real *tTEMP = NULL;

   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / (ng + 2);
   JXF_Real temp_adrress = 0.0;

   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jxf_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jxf_3tAPCTLDataCommY(pre_3tapctl_data);

   JXF_Int myid, nprocs, gidx, send_cnt, j;
   JXF_Int groupid_x = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JXF_Int np_R = jxf_3tAPCTLDataNpR(pre_3tapctl_data);
   JXF_Int rootid_R = groupid_x * np_R;
   JXF_Int rootid_E = np_R * ng;
   JXF_Int rootid_I = np_R + rootid_E;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (groupid_x < ng)
   {
      sr_hsize = jxf_CSRMatrixNumRows(jxf_ParCSRMatrixDiag(ARR));
   }
   else if (groupid_x == ng)
   {
      sr_hsize = jxf_CSRMatrixNumRows(jxf_ParCSRMatrixDiag(AEE));
   }
   else if (groupid_x == ng+1)
   {
      sr_hsize = jxf_CSRMatrixNumRows(jxf_ParCSRMatrixDiag(AII));
   }

   sTEMP = jxf_CTAlloc(JXF_Real, sr_hsize);
   vTEMP = jxf_CTAlloc(JXF_Real, sr_hsize);

   if (print_level == 1 || print_level == 3)
   {
      starttime = jxf_MPI_Wtime();
   }

   if (groupid_x == ng)
   {
      Need_CC = jxf_3tAPCTLmgWeakCouplingE(theta_wc_E, threshold_wc_E, ng, AEE, VER, VEI);
   }
   jxf_MPI_Bcast(&Need_CC, 1, JXF_MPI_INT, rootid_E, comm);
   jxf_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
   if (print_level && myid == 0) jxf_printf(" >> Need_CC = %d\n", Need_CC);

   if (Need_CC)
   {
      ARR_relax_maxit = fixit_pctl_R;
      AEE_relax_maxit = fixit_pctl_E;
      AII_relax_maxit = fixit_pctl_I;
   }
   else
   {
      if (groupid_x < ng)
      {
         jxf_ParVectorDestroy(VRE);
      }
      else if (groupid_x == ng)
      {
         for (gidx = 0; gidx < ng; gidx ++) jxf_ParVectorDestroy(VER[gidx]);
         jxf_TFree(VER);
         jxf_ParVectorDestroy(VEI);
      }
      else if (groupid_x == ng+1)
      {
         jxf_ParVectorDestroy(VIE);
      }

      ARR_relax_maxit = fixit_brlx_R;
      AEE_relax_maxit = fixit_brlx_E;
      AII_relax_maxit = fixit_brlx_I;
   }
   jxf_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
   jxf_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
   jxf_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);

   if (print_level == 1 || print_level == 3)
   {
      endtime = jxf_MPI_Wtime();
      jxf_GetWallTime(comm, "APCTL == Check WC-DD cond", starttime, endtime, 0, 3);
   }

   if (Need_CC)
   {
      if (groupid_x == ng)
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

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         IS_DD_E = jxf_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jxf_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jxf_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jxf_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jxf_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
            jxf_PAMGSetup(AEE_amg_solver, AEE);
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
            jxf_ParVectorInitialize(JAC);
            jxf_ParVectorSetPartitioningOwner(JAC, 0);
         }

         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            sTEMP[gidx] = jxf_CSRMatrixData(jxf_ParCSRMatrixDiag(AEE))[jxf_CSRMatrixI(jxf_ParCSRMatrixDiag(AEE))[gidx]];
         }
         tTEMP = jxf_CTAlloc(JXF_Real, (ng+2)*sr_hsize);
         send_cnt = 0;
         for (j = 0; j < ng; j ++)
         {
            for (gidx = 0; gidx < sr_hsize; gidx ++)
            {
               tTEMP[send_cnt++] = jxf_VectorData(jxf_ParVectorLocalVector(VER[j]))[gidx];
            }
         }
         send_cnt += sr_hsize;
         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            tTEMP[send_cnt++] = jxf_VectorData(jxf_ParVectorLocalVector(VEI))[gidx];
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }

   jxf_MPI_Barrier(comm); //  各进程(组)同步

      jxf_MPI_Bcast(sTEMP, sr_hsize, JXF_MPI_REAL, ng, comm_y);
      jxf_MPI_Scatter(tTEMP, sr_hsize, JXF_MPI_REAL, vTEMP, sr_hsize, JXF_MPI_REAL, ng, comm_y); // yue: invalid for comm_E

      if (groupid_x < ng)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         jxf_BSCModify2SubMat(ARR, VRE, sTEMP, vTEMP, sr_hsize); // update ARR

         IS_DD_R = jxf_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jxf_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jxf_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
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
      else if (groupid_x == ng+1)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         jxf_BSCModify2SubMat(AII, VIE, sTEMP, vTEMP, sr_hsize); // update AII

         IS_DD_I = jxf_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jxf_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jxf_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

         if (AII_relax_type == RELAX_AMG)
         {
            AII_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit);
            jxf_PAMGSetTol(AII_amg_solver, AII_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AII_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
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
   }
   else
   {
      if (groupid_x < ng)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(ARR));
         jxf_ParVectorInitialize(WRR);
         jxf_ParVectorSetPartitioningOwner(WRR, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         IS_DD_R = jxf_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jxf_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jxf_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jxf_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
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
      else if (groupid_x == ng)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WEE = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jxf_ParVectorInitialize(WEE);
         jxf_ParVectorSetDataOwner(WEE, 0);
         jxf_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         IS_DD_E = jxf_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jxf_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jxf_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jxf_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
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
      else if (groupid_x == ng+1)
      {
         RHS = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(RHS);
         jxf_ParVectorSetPartitioningOwner(RHS, 0);

         WII = jxf_ParVectorCreate(comm_x, n, jxf_ParCSRMatrixRowStarts(AII));
         jxf_ParVectorInitialize(WII);
         jxf_ParVectorSetPartitioningOwner(WII, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jxf_MPI_Wtime();
         }

         IS_DD_I = jxf_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jxf_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jxf_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

         if (AII_relax_type == RELAX_AMG)
         {
            AII_amg_solver = jxf_PAMGCreate();
            jxf_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit);
            jxf_PAMGSetTol(AII_amg_solver, AII_relax_tol);
            jxf_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jxf_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jxf_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jxf_PAMGSetInterpType(AII_amg_solver, interp_type);
            jxf_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jxf_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jxf_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
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
   }

   jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) = ARR_amg_solver;
   jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) = AEE_amg_solver;
   jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) = AII_amg_solver;

   jxf_3tAPCTLDataWRR(pre_3tapctl_data) = WRR;
   jxf_3tAPCTLDataWEE(pre_3tapctl_data) = WEE;
   jxf_3tAPCTLDataWII(pre_3tapctl_data) = WII;

   jxf_3tAPCTLDataRHS(pre_3tapctl_data) = RHS;
   jxf_3tAPCTLDataJAC(pre_3tapctl_data) = JAC;

   jxf_3tAPCTLDatatTEMP(pre_3tapctl_data) = tTEMP;

   jxf_TFree(sTEMP);
   jxf_TFree(vTEMP);

   return (0);
}

/*!
 * \fn JXF_Int jxf_3tAPCTLSetup4Jasmin_sp
 * \brief Setup phase of PCTL Iteration or preconditioner(for single-processor case). 
 * \author peghoty 
 * \date 2012/02/25
 */
JXF_Int
jxf_3tAPCTLSetup4Jasmin_sp( jxf_3tAPCTLData *pre_3tapctl_data, jxf_ParCSRMatrix *A )
{
   JXF_Real starttime = 0.0, endtime = 0.0;

   MPI_Comm comm = jxf_ParCSRMatrixComm(A);

   JXF_Int fixit_pctl_R = jxf_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JXF_Int fixit_pctl_E = jxf_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JXF_Int fixit_pctl_I = jxf_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JXF_Int fixit_brlx_R = jxf_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JXF_Int fixit_brlx_E = jxf_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JXF_Int fixit_brlx_I = jxf_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JXF_Int ARR_interp_maxit = jxf_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);     
   JXF_Int AII_interp_maxit = jxf_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);     
   JXF_Int ACC_relax_maxit  = jxf_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data); 

   JXF_Int ARR_relax_maxit;
   JXF_Int AEE_relax_maxit;
   JXF_Int AII_relax_maxit;

   JXF_Real  ARR_interp_tol = jxf_3tAPCTLDataARRInterpTol(pre_3tapctl_data);   
   JXF_Real  AII_interp_tol = jxf_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JXF_Real  ARR_relax_tol  = jxf_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JXF_Real  AEE_relax_tol  = jxf_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JXF_Real  AII_relax_tol  = jxf_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JXF_Real  ACC_relax_tol  = jxf_3tAPCTLDataACCRelaxTol(pre_3tapctl_data); 

   JXF_Int ARR_solver_id = jxf_3tAPCTLDataARRSolverID(pre_3tapctl_data);
   JXF_Int ACC_solver_id = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   JXF_Int ARR_kdim = jxf_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JXF_Int ACC_kdim = jxf_3tAPCTLDataACCKDim(pre_3tapctl_data); 
   
   JXF_Int use_ppctl = jxf_3tAPCTLDataUsePPCTL(pre_3tapctl_data);
    
   JXF_Int print_level = jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data);
  
   jxf_ParAMGData   *ARR_amg_solver = NULL;
   jxf_ParAMGData   *AEE_amg_solver = NULL;
   jxf_ParAMGData   *AII_amg_solver = NULL;
   jxf_ParAMGData   *ACC_amg_solver = NULL;

   jxf_GMRESData    *ARR_gmres_solver = NULL;
   jxf_GMRESData    *ACC_gmres_solver = NULL; 
   
   jxf_ParCSRMatrix *ARR = jxf_3tAPCTLDataARR(pre_3tapctl_data);
   jxf_ParCSRMatrix *AEE = jxf_3tAPCTLDataAEE(pre_3tapctl_data);
   jxf_ParCSRMatrix *AII = jxf_3tAPCTLDataAII(pre_3tapctl_data);  
   jxf_ParVector    *VRE = jxf_3tAPCTLDataVRE(pre_3tapctl_data);
   jxf_ParVector    *VER = jxf_3tAPCTLDataVER(pre_3tapctl_data);
   jxf_ParVector    *VEI = jxf_3tAPCTLDataVEI(pre_3tapctl_data);
   jxf_ParVector    *VIE = jxf_3tAPCTLDataVIE(pre_3tapctl_data);    

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
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data); 

   JXF_Int ARR_relax_type = RELAX_AMG;
   JXF_Int AEE_relax_type = RELAX_AMG;
   JXF_Int AII_relax_type = RELAX_AMG;         

   jxf_CSRMatrix  *ARR_s = jxf_ParCSRMatrixDiag(ARR); 
   jxf_CSRMatrix  *AEE_s = jxf_ParCSRMatrixDiag(AEE);
   jxf_CSRMatrix  *AII_s = jxf_ParCSRMatrixDiag(AII);
   jxf_Vector     *VRE_s = jxf_ParVectorLocalVector(VRE);
   jxf_Vector     *VER_s = jxf_ParVectorLocalVector(VER);
   jxf_Vector     *VEI_s = jxf_ParVectorLocalVector(VEI);
   jxf_Vector     *VIE_s = jxf_ParVectorLocalVector(VIE);
   
   JXF_Int    maxit_default = 200;
   JXF_Real tol_default   = 1.0e-6;

   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);
   JXF_Int n = N / 3;
   JXF_Int i;
   JXF_Real temp_adrress = 0.0;

   if (!use_ppctl)
   {
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
   }
   else
   {
      Need_CC = 1;
      ARR_solver_id = SOLVER_AMG;
      ACC_solver_id = SOLVER_AMG;

      ARR_relax_type = RELAX_AMG;
      AEE_relax_type = RELAX_AMG;
      AII_relax_type = RELAX_AMG;

      ARR_relax_maxit = maxit_default;
      AEE_relax_maxit = maxit_default;
      AII_relax_maxit = maxit_default;
      ACC_relax_maxit = maxit_default;

      ARR_relax_tol = tol_default;
      AEE_relax_tol = tol_default;
      AII_relax_tol = tol_default;
      ACC_relax_tol = tol_default; 
    
      jxf_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;
      jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;
      jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;
      jxf_3tAPCTLSetACCSolverID(pre_3tapctl_data, ACC_solver_id); 
      jxf_3tAPCTLSetARRSolverID(pre_3tapctl_data, ARR_solver_id);     
      jxf_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
      jxf_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
      jxf_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);
      jxf_3tAPCTLSetACCRelaxMaxIt(pre_3tapctl_data, ACC_relax_maxit);
      jxf_3tAPCTLSetARRRelaxTol(pre_3tapctl_data, tol_default);
      jxf_3tAPCTLSetAEERelaxTol(pre_3tapctl_data, tol_default);
      jxf_3tAPCTLSetAIIRelaxTol(pre_3tapctl_data, tol_default);
      jxf_3tAPCTLSetACCRelaxTol(pre_3tapctl_data, tol_default);
      
      jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data) = 0;
      jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data) = 0;
      jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data) = 0;            
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

   if (Need_CC == 1)
   {      
      PRR = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ARR));
      jxf_ParVectorInitialize(PRR); 
      jxf_ParVectorSetPartitioningOwner(PRR, 0);

      PII = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(AII));
      jxf_ParVectorInitialize(PII);
      jxf_ParVectorSetPartitioningOwner(PII, 0); 

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
         jxf_GMRESSetIsCheckRestarted(ARR_gmres_solver, 0); 

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
         jxf_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
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
   else
   {
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
          
   } 
   
   if (Need_CC == 1)
   {
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

      WCC = jxf_ParVectorCreate(comm, n, jxf_ParCSRMatrixRowStarts(ACC));
      jxf_ParVectorInitialize(WCC);
      jxf_ParVectorSetPartitioningOwner(WCC, 0);

      RES = jxf_ParVectorCreate(comm, N, jxf_ParCSRMatrixRowStarts(A));
      jxf_ParVectorInitialize(RES);
      jxf_ParVectorSetPartitioningOwner(RES, 0);

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

   }
      
   jxf_3tAPCTLDataPRR(pre_3tapctl_data) = PRR;
   jxf_3tAPCTLDataPII(pre_3tapctl_data) = PII;
   jxf_3tAPCTLDataACC(pre_3tapctl_data) = ACC;

   jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) = ARR_amg_solver;  
   jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) = AEE_amg_solver;
   jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) = AII_amg_solver;
   jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) = ACC_amg_solver;

   jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = ARR_gmres_solver;
   jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = ACC_gmres_solver;

   jxf_3tAPCTLDataWRR(pre_3tapctl_data) = WRR;
   jxf_3tAPCTLDataWEE(pre_3tapctl_data) = WEE;
   jxf_3tAPCTLDataWII(pre_3tapctl_data) = WII;
   jxf_3tAPCTLDataWCC(pre_3tapctl_data) = WCC;

   jxf_3tAPCTLDataRES(pre_3tapctl_data) = RES;
   jxf_3tAPCTLDataRHS(pre_3tapctl_data) = RHS;
   jxf_3tAPCTLDataJAC(pre_3tapctl_data) = JAC;
             
   return (0); 
}

/*!
 * \fn JXF_Int JXF_3tAPCTLDestroy4Jasmin
 * \brief Destroy a 3tAPCTLData object.
 * \author peghoty
 * \date 2012/02/27
 */
JXF_Int 
JXF_3tAPCTLDestroy4Jasmin( JXF_Solver solver )
{
   return( jxf_3tAPCTLDestroy4Jasmin( (void *) solver ) );
}

JXF_Int 
JXF_3tAPCTLDestroy4mgJasmin( JXF_Solver solver )
{
   return( jxf_3tAPCTLDestroy4mgJasmin( (void *) solver ) );
}

/*!
 * \fn JXF_Int jxf_3tAPCTLDestroy4Jasmin
 * \brief Destroy the jxf_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2012/02/27
 */
JXF_Int
jxf_3tAPCTLDestroy4Jasmin( void *vdata )
{
   jxf_3tAPCTLData  *pre_3tapctl_data = vdata;
   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);

   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jxf_3tAPCTLDestroy4Jasmin_sp(pre_3tapctl_data);
   }
   else if (nprocs > 1)
   {
      jxf_3tAPCTLDestroy4Jasmin_mp(pre_3tapctl_data);
   }
   
   return 0;
}

JXF_Int
jxf_3tAPCTLDestroy4mgJasmin( void *vdata )
{
   jxf_3tAPCTLData  *pre_3tapctl_data = vdata;
   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);

   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jxf_3tAPCTLDestroy4mgJasmin_mp(pre_3tapctl_data);
   }

   return 0;
}

/*!
 * \fn JXF_Int jxf_3tAPCTLDestroy4Jasmin_sp
 * \brief Destroy the jxf_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2012/02/27
 */
JXF_Int
jxf_3tAPCTLDestroy4Jasmin_sp( void *vdata )
{
   jxf_3tAPCTLData  *pre_3tapctl_data = vdata;
   
   if (pre_3tapctl_data)
   {  
      // ARR_amg_solver
      if ( jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) )
      {
         jxf_PAMGDestroy( jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) );
      }
             
      // ARR_gmres_solver  peghoty, 2011/10/29
      if ( jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) )
      {
         jxf_GMRESDestroy(jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data));
      } 
         
      // AEE_amg_solver
      if ( jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) )
      {
         jxf_PAMGDestroy( jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) );
      }        
   
      // AII_amg_solver
      if ( jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) )
      {
         jxf_PAMGDestroy( jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) );
      }       

      // RHS
      if ( jxf_3tAPCTLDataRHS(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataRHS(pre_3tapctl_data) );
      } 
       
      // WRR
      if ( jxf_3tAPCTLDataWRR(pre_3tapctl_data) )
      {
         jxf_TFree( jxf_ParVectorLocalVector(jxf_3tAPCTLDataWRR(pre_3tapctl_data)) );
         jxf_ParVectorDestroy( jxf_3tAPCTLDataWRR(pre_3tapctl_data) );
      }
            
      // WEE
      if ( jxf_3tAPCTLDataWEE(pre_3tapctl_data) )
      {
         jxf_TFree( jxf_ParVectorLocalVector(jxf_3tAPCTLDataWEE(pre_3tapctl_data)) );
         jxf_ParVectorDestroy( jxf_3tAPCTLDataWEE(pre_3tapctl_data) );
      }
      
      // WII
      if ( jxf_3tAPCTLDataWII(pre_3tapctl_data) )
      {
         jxf_TFree( jxf_ParVectorLocalVector(jxf_3tAPCTLDataWII(pre_3tapctl_data)) );
         jxf_ParVectorDestroy( jxf_3tAPCTLDataWII(pre_3tapctl_data) );
      } 

      // ACC
      if ( jxf_3tAPCTLDataACC(pre_3tapctl_data) )
      {
         jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataACC(pre_3tapctl_data) );
      }
      
      // PRR
      if ( jxf_3tAPCTLDataPRR(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataPRR(pre_3tapctl_data) );
      }
      
      // PII
      if ( jxf_3tAPCTLDataPII(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataPII(pre_3tapctl_data) );
      }   
                
      // ACC_amg_solver
      if ( jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) )
      {
         jxf_PAMGDestroy( jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) );
      } 

      // ACC_gmres_solver  peghoty, 2011/10/29
      if ( jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) )
      {
         jxf_GMRESDestroy(jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data));
      }  
      
      // WCC
      if ( jxf_3tAPCTLDataWCC(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataWCC(pre_3tapctl_data) );
      }    
        
      // RES
      if ( jxf_3tAPCTLDataRES(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataRES(pre_3tapctl_data) );
      }       
   
      // JAC
      if ( jxf_3tAPCTLDataJAC(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataJAC(pre_3tapctl_data) );
      }
          
      jxf_TFree(pre_3tapctl_data);
   }
 
   return jxf_error_flag;
}

/*!
 * \fn JXF_Int jxf_3tAPCTLDestroy4Jasmin_mp
 * \brief Destroy the jxf_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2012/03/01
 */
JXF_Int
jxf_3tAPCTLDestroy4Jasmin_mp( void *vdata )
{

   jxf_3tAPCTLData  *pre_3tapctl_data = vdata;
   JXF_Int blocksmooth_type;
   JXF_Int  Need_CC;      
   
   JXF_Int groupid_x;
   
   if (pre_3tapctl_data)
   {
      groupid_x = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);
      blocksmooth_type = jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
      Need_CC = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);   
    
      if (groupid_x == 0)
      {  
         // ARR
         if ( jxf_3tAPCTLDataARR(pre_3tapctl_data) )
         {
            jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataARR(pre_3tapctl_data) );
         }
         
         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            // VRE
            if ( jxf_3tAPCTLDataVRE(pre_3tapctl_data) )
            {
               jxf_ParVectorDestroy( jxf_3tAPCTLDataVRE(pre_3tapctl_data) );
            }
         }                  
         // ARR_amg_solver
         if ( jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) )
         {
            jxf_PAMGDestroy( jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) );
         }
          
         // ARR_gmres_solver  peghoty, 2011/10/29
         if ( jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) )
         {
            jxf_GMRESDestroy(jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data));
         } 

         // RHS
         if ( jxf_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
           
         // WRR
         if ( jxf_3tAPCTLDataWRR(pre_3tapctl_data) )
         {
            jxf_TFree( jxf_ParVectorLocalVector(jxf_3tAPCTLDataWRR(pre_3tapctl_data)) );
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWRR(pre_3tapctl_data) );
         }
               
         // WEE
         if ( jxf_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
         
         // WII
         if ( jxf_3tAPCTLDataWII(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWII(pre_3tapctl_data) );
         }    
      }
      else if (groupid_x == 1)
      {    
         // AEE
         if ( jxf_3tAPCTLDataAEE(pre_3tapctl_data) )
         {
            jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataAEE(pre_3tapctl_data) );
         }

         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            // VER
            if ( jxf_3tAPCTLDataVER(pre_3tapctl_data) )
            {
               jxf_ParVectorDestroy( jxf_3tAPCTLDataVER(pre_3tapctl_data) );
            }
                  
            // VEI
            if ( jxf_3tAPCTLDataVEI(pre_3tapctl_data) )
            {
               jxf_ParVectorDestroy( jxf_3tAPCTLDataVEI(pre_3tapctl_data) );
            }
         }
          
         // AEE_amg_solver
         if ( jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) )
         {
            jxf_PAMGDestroy( jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) );
         } 
         
         // RHS
         if ( jxf_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
           
         // WRR
         if ( jxf_3tAPCTLDataWRR(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWRR(pre_3tapctl_data) );
         } 
              
         // WEE
         if ( jxf_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jxf_TFree( jxf_ParVectorLocalVector(jxf_3tAPCTLDataWEE(pre_3tapctl_data)) );
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
         
         // WII
         if ( jxf_3tAPCTLDataWII(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWII(pre_3tapctl_data) );
         }     
      }
      else if (groupid_x == 2)
      {    
         // AII
         if ( jxf_3tAPCTLDataAII(pre_3tapctl_data) )
         {
            jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataAII(pre_3tapctl_data) );
         }

         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            // VIE
            if ( jxf_3tAPCTLDataVIE(pre_3tapctl_data) )
            {
               jxf_ParVectorDestroy( jxf_3tAPCTLDataVIE(pre_3tapctl_data) );
            }
         }
         
         // AII_amg_solver
         if ( jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) )
         {
            jxf_PAMGDestroy( jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) );
         }    
         
         // RHS
         if ( jxf_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataRHS(pre_3tapctl_data) );
         } 
          
         // WRR
         if ( jxf_3tAPCTLDataWRR(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWRR(pre_3tapctl_data) );
         } 
              
         // WEE
         if ( jxf_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
         
         // WII
         if ( jxf_3tAPCTLDataWII(pre_3tapctl_data) )
         {
            jxf_TFree( jxf_ParVectorLocalVector(jxf_3tAPCTLDataWII(pre_3tapctl_data)) );
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWII(pre_3tapctl_data) );
         }  
      }
     
      // ACC
      if ( jxf_3tAPCTLDataACC(pre_3tapctl_data) )
      {
         jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataACC(pre_3tapctl_data) );
      }
      
      // P
      if ( jxf_3tAPCTLDataP(pre_3tapctl_data) )
      {
         jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataP(pre_3tapctl_data) );
      } 
           
      // ACC_amg_solver
      if ( jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) )
      {
         jxf_PAMGDestroy( jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) );
      } 
      
      // ACC_gmres_solver  peghoty, 2011/10/29
      if ( jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) )
      {
         jxf_GMRESDestroy(jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data));
      } 
            
      // WCC
      if ( jxf_3tAPCTLDataWCC(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataWCC(pre_3tapctl_data) );
      }
            
      // GCC
      if ( jxf_3tAPCTLDataGCC(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataGCC(pre_3tapctl_data) );
      }
      
      // RES
      if ( jxf_3tAPCTLDataRES(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataRES(pre_3tapctl_data) );
      } 
            
      // PRR
      if ( jxf_3tAPCTLDataPRR(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataPRR(pre_3tapctl_data) );
      }
      
      // PII
      if ( jxf_3tAPCTLDataPII(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataPII(pre_3tapctl_data) );
      }
            
      // JAC
      if ( jxf_3tAPCTLDataJAC(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataJAC(pre_3tapctl_data) );
      }
      
      // comm_x and comm_y
      jxf_MPI_Comm_free( &jxf_3tAPCTLDataCommX(pre_3tapctl_data) ); 
      jxf_MPI_Comm_free( &jxf_3tAPCTLDataCommY(pre_3tapctl_data) );        
          
      jxf_TFree(pre_3tapctl_data);
   }
 
   return jxf_error_flag;
}

JXF_Int
jxf_3tAPCTLDestroy4mgJasmin_mp( void *vdata )
{
   jxf_3tAPCTLData *pre_3tapctl_data = vdata;
   JXF_Int blocksmooth_type, Need_CC, groupid_x, ng;

   if (pre_3tapctl_data)
   {
      groupid_x = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);
      blocksmooth_type = jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
      Need_CC = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);
      ng = jxf_3tAPCTLDataNumGroup(pre_3tapctl_data);

      if (groupid_x < ng)
      {
         if ( jxf_3tAPCTLDataARR(pre_3tapctl_data) )
         {
            jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataARR(pre_3tapctl_data) );
         }
         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            if ( jxf_3tAPCTLDataVRE(pre_3tapctl_data) )
            {
               jxf_ParVectorDestroy( jxf_3tAPCTLDataVRE(pre_3tapctl_data) );
            }
         }
         if ( jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) )
         {
            jxf_PAMGDestroy( jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) );
         }
         if ( jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) )
         {
            jxf_GMRESDestroy(jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data));
         }
         if ( jxf_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
         if ( jxf_3tAPCTLDataWRR(pre_3tapctl_data) )
         {
            jxf_TFree( jxf_ParVectorLocalVector(jxf_3tAPCTLDataWRR(pre_3tapctl_data)) );
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWRR(pre_3tapctl_data) );
         }
         if ( jxf_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
      }
      else if (groupid_x == ng)
      {
         if ( jxf_3tAPCTLDataAEE(pre_3tapctl_data) )
         {
            jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataAEE(pre_3tapctl_data) );
         }
         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            if ( jxf_3tAPCTLDataVER(pre_3tapctl_data) )
            {
               jxf_ParVectorDestroy( jxf_3tAPCTLDataVER(pre_3tapctl_data) );
            }
            if ( jxf_3tAPCTLDataVEI(pre_3tapctl_data) )
            {
               jxf_ParVectorDestroy( jxf_3tAPCTLDataVEI(pre_3tapctl_data) );
            }
         }
         if ( jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) )
         {
            jxf_PAMGDestroy( jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) );
         }
         if ( jxf_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
         if ( jxf_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jxf_TFree( jxf_ParVectorLocalVector(jxf_3tAPCTLDataWEE(pre_3tapctl_data)) );
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
         if ( jxf_3tAPCTLDatatTEMP(pre_3tapctl_data) ) jxf_TFree( jxf_3tAPCTLDatatTEMP(pre_3tapctl_data) );
      }
      else if (groupid_x == ng+1)
      {
         if ( jxf_3tAPCTLDataAII(pre_3tapctl_data) )
         {
            jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataAII(pre_3tapctl_data) );
         }
         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            if ( jxf_3tAPCTLDataVIE(pre_3tapctl_data) )
            {
               jxf_ParVectorDestroy( jxf_3tAPCTLDataVIE(pre_3tapctl_data) );
            }
         }
         if ( jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) )
         {
            jxf_PAMGDestroy( jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) );
         }
         if ( jxf_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
         if ( jxf_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
         if ( jxf_3tAPCTLDataWII(pre_3tapctl_data) )
         {
            jxf_TFree( jxf_ParVectorLocalVector(jxf_3tAPCTLDataWII(pre_3tapctl_data)) );
            jxf_ParVectorDestroy( jxf_3tAPCTLDataWII(pre_3tapctl_data) );
         }
      }

      if ( jxf_3tAPCTLDataACC(pre_3tapctl_data) )
      {
         jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataACC(pre_3tapctl_data) );
      }
      if ( jxf_3tAPCTLDataP(pre_3tapctl_data) )
      {
         jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataP(pre_3tapctl_data) );
      }
      if ( jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) )
      {
         jxf_PAMGDestroy( jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) );
      }
      if ( jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) )
      {
         jxf_GMRESDestroy(jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data));
      }
      if ( jxf_3tAPCTLDataWCC(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataWCC(pre_3tapctl_data) );
      }
      if ( jxf_3tAPCTLDataGCC(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataGCC(pre_3tapctl_data) );
      }
      if ( jxf_3tAPCTLDataRES(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataRES(pre_3tapctl_data) );
      }
      if ( jxf_3tAPCTLDataPRR(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataPRR(pre_3tapctl_data) );
      }
      if ( jxf_3tAPCTLDataPII(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataPII(pre_3tapctl_data) );
      }
      if ( jxf_3tAPCTLDataJAC(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataJAC(pre_3tapctl_data) );
      }
      jxf_MPI_Comm_free( &jxf_3tAPCTLDataCommX(pre_3tapctl_data) ); 
      jxf_MPI_Comm_free( &jxf_3tAPCTLDataCommY(pre_3tapctl_data) );
      jxf_TFree(pre_3tapctl_data);
   }
 
   return jxf_error_flag;
}
