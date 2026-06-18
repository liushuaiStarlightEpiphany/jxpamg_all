//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  param.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

/*!
 * \fn jxf_APCTLKrylovParam *jxf_APCTLKrylovParamInitialize
 * \brief Create a jxf_APCTLKrylovParam object. 
 * \author peghoty 
 * \date 2012/02/25
 */
jxf_APCTLKrylovParam *
jxf_APCTLKrylovParamInitialize()
{
   jxf_APCTLKrylovParam *apctlkrylov_param = jxf_CTAlloc(jxf_APCTLKrylovParam, 1);

   JXF_Int       solver_id;           // 1: PCG; 2: PGMRES; 3: PBicgStab
   JXF_Int       precond_id;          // 1: APCTL; 2: Schur1; 3: Schur2
   JXF_Real    tol;                 // tolerance of the APCTL-Krylov method
   JXF_Int       max_iter;            // maximal number of iteration
   JXF_Int       k_dim;               // number of restart
   JXF_Int       is_check_restarted;  // peghoty, 2011/11/08
   JXF_Int       two_norm;            // for PCG
   JXF_Int       print_level;         // how many info to be output?
   JXF_Int       TTest;               // whether timing the program?
   JXF_Int       keepsol;             // whether save the solution?

   JXF_Int    print_level_apctl; // how much info to be output in apctl?
                             // 1: CPU information
                             // 2: inner iteration information
                             // 3: both CPU and inner iteration information
   JXF_Int    blocksmooth_type;  // BD or GS type preconditioner when Coarse Correction is not needed?
                             // BLOCKSMOOTH_BD:
                             // BLOCKSMOOTH_GS:
   JXF_Int    num_relax_pre;     // number of sweeps for pre-block-smoothing
   JXF_Int    num_relax_post;    // number of sweeps for post-block-smoothing

   /* solver type and restart number for interpolation-building of PRR */
   JXF_Int    interp_solver_ARR; // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;   
   JXF_Int    interp_kdim_ARR;   // restart parameters for GMRES solver

   JXF_Int    interp_maxit_ARR;  // maximal number of iteration for ARR to build PRR
   JXF_Int    interp_maxit_AII;  // maximal number of iteration for AII to build PII
   JXF_Real interp_tol_ARR;    // tolerance for ARR to build PRR
   JXF_Real interp_tol_AII;    // tolerance for AII to build PII
   
   JXF_Int    fixit_pctl_R;      // fixed number of iterations for ARR in PCTL
   JXF_Int    fixit_pctl_E;      // fixed number of iterations for AEE in PCTL
   JXF_Int    fixit_pctl_I;      // fixed number of iterations for AII in PCTL
   JXF_Int    fixit_brlx_R;      // fixed number of iterations for ARR in Block Relaxation
   JXF_Int    fixit_brlx_E;      // fixed number of iterations for AEE in Block Relaxation
   JXF_Int    fixit_brlx_I;      // fixed number of iterations for AII in Block Relaxation     

   JXF_Int    ARR_relax_type;
   JXF_Int    AEE_relax_type;
   JXF_Int    AII_relax_type;

   /* whether employ the fixed-number-of-iterations mode? peghoty, 2012/02/15 */
   JXF_Int    use_fixedmode_R;
   JXF_Int    use_fixedmode_E;
   JXF_Int    use_fixedmode_I; 
   
   /* parameters to describe the weaking coupling between AEE and VER, VEI */
   JXF_Real theta_wc_E;
   JXF_Real threshold_wc_E;
   
   /* parameters to describe the diagonal dominance */
   JXF_Real theta_dd_R;
   JXF_Real theta_dd_E;
   JXF_Real theta_dd_I;
   JXF_Real threshold_dd_R;
   JXF_Real threshold_dd_E;
   JXF_Real threshold_dd_I;
   
   /* flag to indicate whether diagonal elements of the three DiagonalBlock
      matrices are firstly stored in each row for the CSR format. peghoty, 2012/03/06 */
   JXF_Int    is_diagelm_first;

   /* Whether use the pure PCTL? */
   JXF_Int    use_ppctl; 
 
   /* Whether test the number of iterations for each sub-linearsystem solution? */
   JXF_Int    test_subls_iter;

   /* CPU time of each phase */
   JXF_Real cpu_trans;
   JXF_Real cpu_setup;
   JXF_Real cpu_solve;
   JXF_Real cpu_total;   

   JXF_Int debug_flag;
      
   JXF_Int reset_zero;

   solver_id          = 2;
   precond_id         = 1;
   tol                = 1.0e-7;
   max_iter           = 200;
   k_dim              = 5;
   is_check_restarted = 0;
   two_norm           = 0;
   print_level        = 3;  
   TTest              = 1;
   keepsol            = 0;            

   print_level_apctl = 0; 
   blocksmooth_type  = BLOCKSMOOTH_BD; // BLOCKSMOOTH_GS;
   num_relax_pre     = 1;
   num_relax_post    = 0;  
   
   interp_solver_ARR = SOLVER_AMG;   // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   interp_kdim_ARR   = 20;
  
   interp_maxit_ARR  = 200;
   interp_maxit_AII  = 200;
   interp_tol_ARR    = 1.0e-6;   
   interp_tol_AII    = 1.0e-6; 
   
   fixit_pctl_R      = 1;      
   fixit_pctl_E      = 1;
   fixit_pctl_I      = 1;
   fixit_brlx_R      = 3;
   fixit_brlx_E      = 1;
   fixit_brlx_I      = 1;

   ARR_relax_type    = RELAX_AMG;
   AEE_relax_type    = RELAX_AMG;
   AII_relax_type    = RELAX_AMG;

   use_fixedmode_R   = 1;
   use_fixedmode_E   = 1;
   use_fixedmode_I   = 1;      
   
   theta_wc_E        = 1.0e-4;
   threshold_wc_E    = 0.5;
   theta_dd_R        = 0.9;
   theta_dd_E        = 0.9;
   theta_dd_I        = 0.9;
   threshold_dd_R    = 1.0;
   threshold_dd_E    = 1.0;
   threshold_dd_I    = 1.0; 
   
   is_diagelm_first  = 0; 
   use_ppctl         = 0;
   test_subls_iter   = 0; 
 
   cpu_trans         = 0.0;
   cpu_setup         = 0.0;
   cpu_solve         = 0.0;
   cpu_total         = 0.0;  

   debug_flag        = 0;

   reset_zero        = 1;
          
   jxf_APCTLKrylovParamSetSolverID(apctlkrylov_param, solver_id);
   jxf_APCTLKrylovParamSetPrecondID(apctlkrylov_param, precond_id);
   jxf_APCTLKrylovParamSetTol(apctlkrylov_param, tol);
   jxf_APCTLKrylovParamSetMaxIter(apctlkrylov_param, max_iter);
   jxf_APCTLKrylovParamSetKDim(apctlkrylov_param, k_dim);
   jxf_APCTLKrylovParamSetISCheckRestarted(apctlkrylov_param, is_check_restarted);
   jxf_APCTLKrylovParamSetTwoNorm(apctlkrylov_param, two_norm);
   jxf_APCTLKrylovParamSetPrintLevel(apctlkrylov_param, print_level);
   jxf_APCTLKrylovParamSetTTest(apctlkrylov_param, TTest);
   jxf_APCTLKrylovParamSetKeepSol(apctlkrylov_param, keepsol);
   
   jxf_APCTLKrylovParamSetPrintLevelAPCTL(apctlkrylov_param, print_level_apctl );
   jxf_APCTLKrylovParamSetBlockSmoothType(apctlkrylov_param, blocksmooth_type );
   jxf_APCTLKrylovParamSetNumRelaxPre(apctlkrylov_param, num_relax_pre );
   jxf_APCTLKrylovParamSetNumRelaxPost(apctlkrylov_param, num_relax_post );
   jxf_APCTLKrylovParamSetInterpSolverARR(apctlkrylov_param, interp_solver_ARR );
   jxf_APCTLKrylovParamSetInterpKdimARR(apctlkrylov_param, interp_kdim_ARR );
   jxf_APCTLKrylovParamSetInterpMaxitARR(apctlkrylov_param, interp_maxit_ARR );
   jxf_APCTLKrylovParamSetInterpMaxitAII(apctlkrylov_param, interp_maxit_AII );
   jxf_APCTLKrylovParamSetInterpTolARR(apctlkrylov_param, interp_tol_ARR );
   jxf_APCTLKrylovParamSetInterpTolAII(apctlkrylov_param, interp_tol_AII );
   jxf_APCTLKrylovParamSetFixitPCTLR(apctlkrylov_param, fixit_pctl_R );
   jxf_APCTLKrylovParamSetFixitPCTLE(apctlkrylov_param, fixit_pctl_E );
   jxf_APCTLKrylovParamSetFixitPCTLI(apctlkrylov_param, fixit_pctl_I );
   jxf_APCTLKrylovParamSetFixitBrlxR(apctlkrylov_param, fixit_brlx_R );
   jxf_APCTLKrylovParamSetFixitBrlxE(apctlkrylov_param, fixit_brlx_E );
   jxf_APCTLKrylovParamSetFixitBrlxI(apctlkrylov_param, fixit_brlx_I );
   jxf_APCTLKrylovParamSetARRRelaxType(apctlkrylov_param, ARR_relax_type);
   jxf_APCTLKrylovParamSetAEERelaxType(apctlkrylov_param, AEE_relax_type);
   jxf_APCTLKrylovParamSetAIIRelaxType(apctlkrylov_param, AII_relax_type);
   jxf_APCTLKrylovParamSetUseFixedModeR(apctlkrylov_param, use_fixedmode_R );
   jxf_APCTLKrylovParamSetUseFixedModeE(apctlkrylov_param, use_fixedmode_E );
   jxf_APCTLKrylovParamSetUseFixedModeI(apctlkrylov_param, use_fixedmode_I );
   jxf_APCTLKrylovParamSetThetaWCE(apctlkrylov_param, theta_wc_E );
   jxf_APCTLKrylovParamSetParamThresholdWCE(apctlkrylov_param, threshold_wc_E );
   jxf_APCTLKrylovParamSetThetaDDR(apctlkrylov_param, theta_dd_R );
   jxf_APCTLKrylovParamSetThetaDDE(apctlkrylov_param, theta_dd_E );
   jxf_APCTLKrylovParamSetThetaDDI(apctlkrylov_param, theta_dd_I );
   jxf_APCTLKrylovParamSetThresholdDDR(apctlkrylov_param, threshold_dd_R );
   jxf_APCTLKrylovParamSetThresholdDDE(apctlkrylov_param, threshold_dd_E );
   jxf_APCTLKrylovParamSetThresholdDDI(apctlkrylov_param, threshold_dd_I );
   jxf_APCTLKrylovParamSetISDiagElmFirst(apctlkrylov_param, is_diagelm_first);
   jxf_APCTLKrylovParamSetUsePPCTL(apctlkrylov_param, use_ppctl);
   jxf_APCTLKrylovParamSetTestSubLSIter(apctlkrylov_param, test_subls_iter);    
   // peghoty, 2012/03/25
   jxf_APCTLKrylovParamSetCPUTrans(apctlkrylov_param, cpu_trans);
   jxf_APCTLKrylovParamSetCPUSetup(apctlkrylov_param, cpu_setup);
   jxf_APCTLKrylovParamSetCPUSolve(apctlkrylov_param, cpu_solve);
   jxf_APCTLKrylovParamSetCPUTotal(apctlkrylov_param, cpu_total); 

   jxf_APCTLKrylovParamSetDebugFlag(apctlkrylov_param, debug_flag);
   jxf_APCTLKrylovParamSetResetZero(apctlkrylov_param, reset_zero);

   return (apctlkrylov_param);
}

/*!
 * \fn JXF_Int jxf_APCTLKrylovParamSetXXXX
 * \brief Set parameters for the jxf_APCTLKrylovParam object. 
 * \author peghoty 
 * \date 2012/02/25
 */
JXF_Int 
jxf_APCTLKrylovParamSetNumGroup( void *param, JXF_Int ng )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamNumGroup(apctlkrylov_param) = ng;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetSolverID( void *param, JXF_Int solver_id )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamSolverID(apctlkrylov_param) = solver_id;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetPrecondID( void *param, JXF_Int precond_id )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamPrecondID(apctlkrylov_param) = precond_id;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetTol( void *param, JXF_Real tol )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamTol(apctlkrylov_param) = tol;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetMaxIter( void *param, JXF_Int max_iter )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamMaxIter(apctlkrylov_param) = max_iter;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetKDim( void *param, JXF_Int k_dim )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamKDim(apctlkrylov_param) = k_dim;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetISCheckRestarted( void *param, JXF_Int is_check_restarted )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamISCheckRestarted(apctlkrylov_param) = is_check_restarted;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetTwoNorm( void *param, JXF_Int two_norm )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamTwoNorm(apctlkrylov_param) = two_norm;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetPrintLevel( void *param, JXF_Int print_level )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamPrintLevel(apctlkrylov_param) = print_level;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetTTest( void *param, JXF_Int TTest )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamTTest(apctlkrylov_param) = TTest;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetKeepSol( void *param, JXF_Int keepsol )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamKeepSol(apctlkrylov_param) = keepsol;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetNumIterations( void *param, JXF_Int num )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamNumIterations(apctlkrylov_param) = num;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetLastRelNrm( void *param, JXF_Real last_rel_nrm )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamLastRelNrm(apctlkrylov_param) = last_rel_nrm;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetPrintLevelAPCTL( void *param, JXF_Int print_level_apctl )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamPrintLevelAPCTL(apctlkrylov_param) = print_level_apctl;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetBlockSmoothType( void *param, JXF_Int blocksmooth_type )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamBlockSmoothType(apctlkrylov_param) = blocksmooth_type;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetNumRelaxPre( void *param, JXF_Int num_relax_pre )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamNumRelaxPre(apctlkrylov_param) = num_relax_pre;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetNumRelaxPost( void *param, JXF_Int num_relax_post )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamNumRelaxPost(apctlkrylov_param) = num_relax_post;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetInterpSolverARR( void *param, JXF_Int interp_solver_ARR )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamInterpSolverARR(apctlkrylov_param) = interp_solver_ARR;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetInterpKdimARR( void *param, JXF_Int interp_kdim_ARR )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamInterpKdimARR(apctlkrylov_param) = interp_kdim_ARR;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetInterpMaxitARR( void *param, JXF_Int interp_maxit_ARR )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamInterpMaxitARR(apctlkrylov_param) = interp_maxit_ARR;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetInterpMaxitAII( void *param, JXF_Int interp_maxit_AII )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamInterpMaxitAII(apctlkrylov_param) = interp_maxit_AII;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetInterpTolARR( void *param, JXF_Real interp_tol_ARR )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamInterpTolARR(apctlkrylov_param) = interp_tol_ARR;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetInterpTolAII( void *param, JXF_Real interp_tol_AII )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamInterpTolAII(apctlkrylov_param) = interp_tol_AII;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetARRRelaxTol( void *param, JXF_Real ARR_relax_tol )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamARRRelaxTol(apctlkrylov_param) = ARR_relax_tol;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetAEERelaxTol( void *param, JXF_Real AEE_relax_tol )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamAEERelaxTol(apctlkrylov_param) = AEE_relax_tol;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetAIIRelaxTol( void *param, JXF_Real AII_relax_tol )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamAIIRelaxTol(apctlkrylov_param) = AII_relax_tol;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetACCRelaxTol( void *param, JXF_Real ACC_relax_tol )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamACCRelaxTol(apctlkrylov_param) = ACC_relax_tol;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetFixitPCTLR( void *param, JXF_Int fixit_pctl_R )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamFixitPCTLR(apctlkrylov_param) = fixit_pctl_R;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetFixitPCTLE( void *param, JXF_Int fixit_pctl_E )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamFixitPCTLE(apctlkrylov_param) = fixit_pctl_E;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetFixitPCTLI( void *param, JXF_Int fixit_pctl_I )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamFixitPCTLI(apctlkrylov_param) = fixit_pctl_I;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetFixitBrlxR( void *param, JXF_Int fixit_brlx_R )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamFixitBrlxR(apctlkrylov_param) = fixit_brlx_R;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetFixitBrlxE( void *param, JXF_Int fixit_brlx_E )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamFixitBrlxE(apctlkrylov_param) = fixit_brlx_E;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetFixitBrlxI( void *param, JXF_Int fixit_brlx_I )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamFixitBrlxI(apctlkrylov_param) = fixit_brlx_I;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetARRRelaxType( void *param, JXF_Int ARR_relax_type )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamARRRelaxType(apctlkrylov_param) = ARR_relax_type;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetAEERelaxType( void *param, JXF_Int AEE_relax_type )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamAEERelaxType(apctlkrylov_param) = AEE_relax_type;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetAIIRelaxType( void *param, JXF_Int AII_relax_type )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamAIIRelaxType(apctlkrylov_param) = AII_relax_type;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetUseFixedModeR( void *param, JXF_Int use_fixedmode_R )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamUseFixedModeR(apctlkrylov_param) = use_fixedmode_R;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetUseFixedModeE( void *param, JXF_Int use_fixedmode_E )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamUseFixedModeE(apctlkrylov_param) = use_fixedmode_E;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetUseFixedModeI( void *param, JXF_Int use_fixedmode_I )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamUseFixedModeI(apctlkrylov_param) = use_fixedmode_I;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetThetaWCE( void *param, JXF_Real theta_wc_E )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamThetaWCE(apctlkrylov_param) = theta_wc_E;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetParamThresholdWCE( void *param, JXF_Real threshold_wc_E )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamThresholdWCE(apctlkrylov_param) = threshold_wc_E;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetThetaDDR( void *param, JXF_Real theta_dd_R )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamThetaDDR(apctlkrylov_param) = theta_dd_R;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetThetaDDE( void *param, JXF_Real theta_dd_E )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamThetaDDE(apctlkrylov_param) = theta_dd_E;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetThetaDDI( void *param, JXF_Real theta_dd_I )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamThetaDDI(apctlkrylov_param) = theta_dd_I;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetThresholdDDR( void *param, JXF_Real threshold_dd_R )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamThresholdDDR(apctlkrylov_param) = threshold_dd_R;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetThresholdDDE( void *param, JXF_Real threshold_dd_E )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamThresholdDDE(apctlkrylov_param) = threshold_dd_E;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetThresholdDDI( void *param, JXF_Real threshold_dd_I )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamThresholdDDI(apctlkrylov_param) = threshold_dd_I;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetISDiagElmFirst( void *param, JXF_Int is_diagelm_first )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamISDiagElmFirst(apctlkrylov_param) = is_diagelm_first;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetUsePPCTL( void *param, JXF_Int use_ppctl )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamUsePPCTL(apctlkrylov_param) = use_ppctl;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetDebugFlag( void *param, JXF_Int debug_flag )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamDebugFlag(apctlkrylov_param) = debug_flag;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetResetZero( void *param, JXF_Int reset_zero )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamResetZero(apctlkrylov_param) = reset_zero;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetStrongThreshold( void *param, JXF_Real strong_threshold )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamStrongThreshold(apctlkrylov_param) = strong_threshold;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetInterpType( void *param, JXF_Int interp_type )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamInterpType(apctlkrylov_param) = interp_type;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetCoarsenType( void *param, JXF_Int coarsen_type )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamCoarsenType(apctlkrylov_param) = coarsen_type;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetAggNumLevels( void *param, JXF_Int agg_num_levels )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamAggNumLevels(apctlkrylov_param) = agg_num_levels;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetCoarseThreshold( void *param, JXF_Int coarse_threshold )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamCoarseThreshold(apctlkrylov_param) = coarse_threshold;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetPrintLevelAMG( void *param, JXF_Int print_level_amg )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamPrintLevelAMG(apctlkrylov_param) = print_level_amg;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetTestSubLSIter( void *param, JXF_Int test_subls_iter )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamTestSubLSIter(apctlkrylov_param) = test_subls_iter;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetCPUTrans( void *param, JXF_Real cpu_trans )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamCPUTrans(apctlkrylov_param) = cpu_trans;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetCPUSetup( void *param, JXF_Real cpu_setup )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamCPUSetup(apctlkrylov_param) = cpu_setup;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetCPUSolve( void *param, JXF_Real cpu_solve )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamCPUSolve(apctlkrylov_param) = cpu_solve;
   return jxf_error_flag;
}

JXF_Int 
jxf_APCTLKrylovParamSetCPUTotal( void *param, JXF_Real cpu_total )
{
   jxf_APCTLKrylovParam *apctlkrylov_param = param;
   jxf_APCTLKrylovParamCPUTotal(apctlkrylov_param) = cpu_total;
   return jxf_error_flag;
}
