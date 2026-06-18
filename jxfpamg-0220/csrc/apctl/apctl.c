//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  apctl.c -- Adaptive PCTL iterattion or preconditioner for 3t linear systems.
 *             PCTL, Physical-variable based Coarsening Two Level iterative Method,
 *             is proposed by Xu Xiaowen, Mo Zeyao etc.
 *  Date: 2011/09/17
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
 * \fn JXF_Int JXF_3tAPCTLSetXXXXX
 * \brief Set parameters for jxf_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2011/09/17
 */ 
JXF_Int 
JXF_3tAPCTLSetTol( JXF_Solver solver, JXF_Real tol )
{
   return( jxf_3tAPCTLSetTol( (void *) solver, tol ) );
}

JXF_Int 
JXF_3tAPCTLSetMaxiter( JXF_Solver solver, JXF_Int max_iter )
{
   return( jxf_3tAPCTLSetMaxiter( (void *) solver, max_iter ) );
}

JXF_Int 
JXF_3tAPCTLSetNumRlxPre( JXF_Solver solver, JXF_Int num_relax )
{
   return( jxf_3tAPCTLSetNumRlxPre( (void *) solver, num_relax ) );
}

JXF_Int 
JXF_3tAPCTLSetNumRlxPost( JXF_Solver solver, JXF_Int num_relax )
{
   return( jxf_3tAPCTLSetNumRlxPost( (void *) solver, num_relax ) );
}

JXF_Int 
JXF_3tAPCTLSetPrintLevel( JXF_Solver solver, JXF_Int print_level )
{
   return( jxf_3tAPCTLSetPrintLevel( (void *) solver, print_level ) );
}

JXF_Int 
JXF_3tAPCTLSetIsDiagElmFirst( JXF_Solver solver, JXF_Int is_diagelm_first )
{
   return( jxf_3tAPCTLSetIsDiagElmFirst( (void *) solver, is_diagelm_first ) );
}

JXF_Int 
JXF_3tAPCTLSetNumGroup( JXF_Solver solver, JXF_Int num_group )
{
   return( jxf_3tAPCTLSetNumGroup( (void *) solver, num_group ) );
}

JXF_Int 
JXF_3tAPCTLSetNumIterations( JXF_Solver solver, JXF_Int num_iterations )
{
   return( jxf_3tAPCTLSetNumIterations( (void *) solver, num_iterations ) );
}

JXF_Int 
JXF_3tAPCTLSetLastRelNrm( JXF_Solver solver, JXF_Real last_rel_nrm )
{
   return( jxf_3tAPCTLSetLastRelNrm( (void *) solver, last_rel_nrm ) );
}

JXF_Int 
JXF_3tAPCTLSetAveConvFactor( JXF_Solver solver, JXF_Real ave_conv_factor )
{
   return( jxf_3tAPCTLSetAveConvFactor( (void *) solver, ave_conv_factor ) );
}

JXF_Int 
JXF_3tAPCTLGetNumIterations( JXF_Solver solver, JXF_Int *num_iterations )
{
   return( jxf_3tAPCTLGetNumIterations( (void *) solver, num_iterations ) );
}

JXF_Int 
JXF_3tAPCTLGetLastRelNrm( JXF_Solver solver, JXF_Real *rel_resid_norm )
{
   return( jxf_3tAPCTLGetLastRelNrm( (void *) solver, rel_resid_norm ) );
}

JXF_Int 
JXF_3tAPCTLSetNpR( JXF_Solver solver, JXF_Int np )
{
   return( jxf_3tAPCTLSetNpR( (void *) solver, np ) );
}

JXF_Int 
JXF_3tAPCTLSetNpE( JXF_Solver solver, JXF_Int np )
{
   return( jxf_3tAPCTLSetNpE( (void *) solver, np ) );
}

JXF_Int 
JXF_3tAPCTLSetNpI( JXF_Solver solver, JXF_Int np )
{
   return( jxf_3tAPCTLSetNpI( (void *) solver, np ) );
}

JXF_Int 
JXF_3tAPCTLSetARRInterpMaxIt( JXF_Solver solver, JXF_Int max_iter )
{
   return( jxf_3tAPCTLSetARRInterpMaxIt( (void *) solver, max_iter ) );
}

JXF_Int 
JXF_3tAPCTLSetAIIInterpMaxIt( JXF_Solver solver, JXF_Int max_iter )
{
   return( jxf_3tAPCTLSetAIIInterpMaxIt( (void *) solver, max_iter ) );
}

JXF_Int 
JXF_3tAPCTLSetARRRelaxMaxIt( JXF_Solver solver, JXF_Int max_iter )
{
   return( jxf_3tAPCTLSetARRRelaxMaxIt( (void *) solver, max_iter ) );
}

JXF_Int 
JXF_3tAPCTLSetAEERelaxMaxIt( JXF_Solver solver, JXF_Int max_iter )
{
   return( jxf_3tAPCTLSetAEERelaxMaxIt( (void *) solver, max_iter ) );
}

JXF_Int 
JXF_3tAPCTLSetAIIRelaxMaxIt( JXF_Solver solver, JXF_Int max_iter )
{
   return( jxf_3tAPCTLSetAIIRelaxMaxIt( (void *) solver, max_iter ) );
}

JXF_Int 
JXF_3tAPCTLSetACCRelaxMaxIt( JXF_Solver solver, JXF_Int max_iter )
{
   return( jxf_3tAPCTLSetACCRelaxMaxIt( (void *) solver, max_iter ) );
}

JXF_Int 
JXF_3tAPCTLSetARRInterpTol( JXF_Solver solver, JXF_Real tol )
{
   return( jxf_3tAPCTLSetARRInterpTol( (void *) solver, tol ) );
}

JXF_Int 
JXF_3tAPCTLSetAIIInterpTol( JXF_Solver solver, JXF_Real tol )
{
   return( jxf_3tAPCTLSetAIIInterpTol( (void *) solver, tol ) );
}

JXF_Int 
JXF_3tAPCTLSetARRRelaxTol( JXF_Solver solver, JXF_Real tol )
{
   return( jxf_3tAPCTLSetARRRelaxTol( (void *) solver, tol ) );
}

JXF_Int 
JXF_3tAPCTLSetAEERelaxTol( JXF_Solver solver, JXF_Real tol )
{
   return( jxf_3tAPCTLSetAEERelaxTol( (void *) solver, tol ) );
}

JXF_Int 
JXF_3tAPCTLSetAIIRelaxTol( JXF_Solver solver, JXF_Real tol )
{
   return( jxf_3tAPCTLSetAIIRelaxTol( (void *) solver, tol ) );
}

JXF_Int 
JXF_3tAPCTLSetACCRelaxTol( JXF_Solver solver, JXF_Real tol )
{
   return( jxf_3tAPCTLSetACCRelaxTol( (void *) solver, tol ) );
}

JXF_Int 
JXF_3tAPCTLSetARRRelaxType( JXF_Solver solver, JXF_Int ARR_relax_type )
{
   return( jxf_3tAPCTLSetARRRelaxType( (void *) solver, ARR_relax_type ) );
}

JXF_Int 
JXF_3tAPCTLSetAEERelaxType( JXF_Solver solver, JXF_Int AEE_relax_type )
{
   return( jxf_3tAPCTLSetAEERelaxType( (void *) solver, AEE_relax_type ) );
}

JXF_Int 
JXF_3tAPCTLSetAIIRelaxType( JXF_Solver solver, JXF_Int AII_relax_type )
{
   return( jxf_3tAPCTLSetAIIRelaxType( (void *) solver, AII_relax_type ) );
}

JXF_Int 
JXF_3tAPCTLSetThetaWCR( JXF_Solver solver, JXF_Real theta )
{
   return( jxf_3tAPCTLSetThetaWCR( (void *) solver, theta ) );
} 

JXF_Int 
JXF_3tAPCTLSetThetaWCE( JXF_Solver solver, JXF_Real theta )
{
   return( jxf_3tAPCTLSetThetaWCE( (void *) solver, theta ) );
} 
 
JXF_Int 
JXF_3tAPCTLSetThetaWCI( JXF_Solver solver, JXF_Real theta )
{
   return( jxf_3tAPCTLSetThetaWCI( (void *) solver, theta ) );
}
  
JXF_Int 
JXF_3tAPCTLSetThresholdWCR( JXF_Solver solver, JXF_Real threshold )
{
   return( jxf_3tAPCTLSetThresholdWCR( (void *) solver, threshold ) );
}
 
JXF_Int 
JXF_3tAPCTLSetThresholdWCE( JXF_Solver solver, JXF_Real threshold )
{
   return( jxf_3tAPCTLSetThresholdWCE( (void *) solver, threshold ) );
}
 
JXF_Int 
JXF_3tAPCTLSetThresholdWCI( JXF_Solver solver, JXF_Real threshold )
{
   return( jxf_3tAPCTLSetThresholdWCI( (void *) solver, threshold ) );
}
   
JXF_Int 
JXF_3tAPCTLSetISWCR( JXF_Solver solver, JXF_Int flag )
{
   return( jxf_3tAPCTLSetISWCR( (void *) solver, flag ) );
}

JXF_Int 
JXF_3tAPCTLSetISWCE( JXF_Solver solver, JXF_Int flag )
{
   return( jxf_3tAPCTLSetISWCE( (void *) solver, flag ) );
}  

JXF_Int 
JXF_3tAPCTLSetISWCI( JXF_Solver solver, JXF_Int flag )
{
   return( jxf_3tAPCTLSetISWCI( (void *) solver, flag ) );
}      

JXF_Int 
JXF_3tAPCTLSetThetaDDR( JXF_Solver solver, JXF_Real theta )
{
   return( jxf_3tAPCTLSetThetaDDR( (void *) solver, theta ) );
} 

JXF_Int 
JXF_3tAPCTLSetThetaDDE( JXF_Solver solver, JXF_Real theta )
{
   return( jxf_3tAPCTLSetThetaDDE( (void *) solver, theta ) );
} 
 
JXF_Int 
JXF_3tAPCTLSetThetaDDI( JXF_Solver solver, JXF_Real theta )
{
   return( jxf_3tAPCTLSetThetaDDI( (void *) solver, theta ) );
} 
 
JXF_Int 
JXF_3tAPCTLSetThresholdDDR( JXF_Solver solver, JXF_Real threshold )
{
   return( jxf_3tAPCTLSetThresholdDDR( (void *) solver, threshold ) );
}
 
JXF_Int 
JXF_3tAPCTLSetThresholdDDE( JXF_Solver solver, JXF_Real threshold )
{
   return( jxf_3tAPCTLSetThresholdDDE( (void *) solver, threshold ) );
} 

JXF_Int 
JXF_3tAPCTLSetThresholdDDI( JXF_Solver solver, JXF_Real threshold )
{
   return( jxf_3tAPCTLSetThresholdDDI( (void *) solver, threshold ) );
}
   
JXF_Int 
JXF_3tAPCTLSetISDDR( JXF_Solver solver, JXF_Int flag )
{
   return( jxf_3tAPCTLSetISDDR( (void *) solver, flag ) );
}

JXF_Int 
JXF_3tAPCTLSetISDDE( JXF_Solver solver, JXF_Int flag )
{
   return( jxf_3tAPCTLSetISDDE( (void *) solver, flag ) );
}  

JXF_Int 
JXF_3tAPCTLSetISDDI( JXF_Solver solver, JXF_Int flag )
{
   return( jxf_3tAPCTLSetISDDI( (void *) solver, flag ) );
}  
  
JXF_Int 
JXF_3tAPCTLSetNeedCC( JXF_Solver solver, JXF_Int flag )
{
   return( jxf_3tAPCTLSetNeedCC( (void *) solver, flag ) );
}

JXF_Int 
JXF_3tAPCTLSetBlockSmoothType( JXF_Solver solver, JXF_Int blocksmooth_type )
{
   return( jxf_3tAPCTLSetBlockSmoothType( (void *) solver, blocksmooth_type ) );
}
  
JXF_Int 
JXF_3tAPCTLSetFixItPCTLR( JXF_Solver solver, JXF_Int fixit )
{
   return( jxf_3tAPCTLSetFixItPCTLR( (void *) solver, fixit ) );
}

JXF_Int 
JXF_3tAPCTLSetFixItPCTLE( JXF_Solver solver, JXF_Int fixit )
{
   return( jxf_3tAPCTLSetFixItPCTLE( (void *) solver, fixit ) );
}

JXF_Int 
JXF_3tAPCTLSetFixItPCTLI( JXF_Solver solver, JXF_Int fixit )
{
   return( jxf_3tAPCTLSetFixItPCTLI( (void *) solver, fixit ) );
}

JXF_Int 
JXF_3tAPCTLSetFixItBRLXR( JXF_Solver solver, JXF_Int fixit )
{
   return( jxf_3tAPCTLSetFixItBRLXR( (void *) solver, fixit ) );
}

JXF_Int 
JXF_3tAPCTLSetFixItBRLXE( JXF_Solver solver, JXF_Int fixit )
{
   return( jxf_3tAPCTLSetFixItBRLXE( (void *) solver, fixit ) );
}

JXF_Int 
JXF_3tAPCTLSetFixItBRLXI( JXF_Solver solver, JXF_Int fixit )
{
   return( jxf_3tAPCTLSetFixItBRLXI( (void *) solver, fixit ) );
}

JXF_Int 
JXF_3tAPCTLSetUseFixedModeR( JXF_Solver solver, JXF_Int usefixedmode )
{
   return( jxf_3tAPCTLSetUseFixedModeR( (void *) solver, usefixedmode ) );
}

JXF_Int 
JXF_3tAPCTLSetUseFixedModeE( JXF_Solver solver, JXF_Int usefixedmode )
{
   return( jxf_3tAPCTLSetUseFixedModeE( (void *) solver, usefixedmode ) );
}   

JXF_Int 
JXF_3tAPCTLSetUseFixedModeI( JXF_Solver solver, JXF_Int usefixedmode )
{
   return( jxf_3tAPCTLSetUseFixedModeI( (void *) solver, usefixedmode ) );
}

JXF_Int 
JXF_3tAPCTLSetUsePPCTL( JXF_Solver solver, JXF_Int use_ppctl )
{
   return( jxf_3tAPCTLSetUsePPCTL( (void *) solver, use_ppctl ) );
}

JXF_Int 
JXF_3tAPCTLSetARRSolverID( JXF_Solver solver, JXF_Int solverid )
{
   return( jxf_3tAPCTLSetARRSolverID( (void *) solver, solverid ) );
}

JXF_Int 
JXF_3tAPCTLSetACCSolverID( JXF_Solver solver, JXF_Int solverid )
{
   return( jxf_3tAPCTLSetACCSolverID( (void *) solver, solverid ) );
}

JXF_Int 
JXF_3tAPCTLSetARRKDim( JXF_Solver solver, JXF_Int kdim )
{
   return( jxf_3tAPCTLSetARRKDim( (void *) solver, kdim ) );
}

JXF_Int 
JXF_3tAPCTLSetACCKDim( JXF_Solver solver, JXF_Int kdim )
{
   return( jxf_3tAPCTLSetACCKDim( (void *) solver, kdim ) );
}

JXF_Int 
JXF_3tAPCTLSetTestSubLSIter( JXF_Solver solver, JXF_Int test_subls_iter )
{
   return( jxf_3tAPCTLSetTestSubLSIter( (void *) solver, test_subls_iter ) );
}

JXF_Int 
JXF_3tAPCTLSetDebugFlag( JXF_Solver solver, JXF_Int debug_flag )
{
   return( jxf_3tAPCTLSetDebugFlag( (void *) solver, debug_flag ) );
}

JXF_Int 
JXF_3tAPCTLSetResetZero( JXF_Solver solver, JXF_Int reset_zero )
{
   return( jxf_3tAPCTLSetResetZero( (void *) solver, reset_zero ) );
}

JXF_Int 
JXF_3tAPCTLSetInterpType( JXF_Solver solver, JXF_Int interp_type )
{
   return( jxf_3tAPCTLSetInterpType( (void *) solver, interp_type ) );
}

JXF_Int 
JXF_3tAPCTLSetCoarsenType( JXF_Solver solver, JXF_Int coarsen_type )
{
   return( jxf_3tAPCTLSetCoarsenType( (void *) solver, coarsen_type ) );
}

JXF_Int 
JXF_3tAPCTLSetAggNumLevels( JXF_Solver solver, JXF_Int agg_num_levels )
{
   return( jxf_3tAPCTLSetAggNumLevels( (void *) solver, agg_num_levels ) );
}

JXF_Int 
JXF_3tAPCTLSetCoarseThreshold( JXF_Solver solver, JXF_Int coarse_threshold )
{
   return( jxf_3tAPCTLSetCoarseThreshold( (void *) solver, coarse_threshold ) );
}

JXF_Int 
JXF_3tAPCTLSetPrintLevelAMG( JXF_Solver solver, JXF_Int print_level_amg )
{
   return( jxf_3tAPCTLSetPrintLevelAMG( (void *) solver, print_level_amg ) );
}

JXF_Int 
JXF_3tAPCTLSetStrongThreshold( JXF_Solver solver, JXF_Real strong_threshold )
{
   return( jxf_3tAPCTLSetStrongThreshold( (void *) solver, strong_threshold ) );
}

JXF_Int 
JXF_3tAPCTLSetNumIterAiSetup( JXF_Solver solver, JXF_Int num_iter_Ai_pctl_setup )
{
   return( jxf_3tAPCTLSetNumIterAiSetup( (void *) solver, num_iter_Ai_pctl_setup ) );
}

JXF_Int 
JXF_3tAPCTLSetNumIterArSetup( JXF_Solver solver, JXF_Int num_iter_Ar_pctl_setup )
{
   return( jxf_3tAPCTLSetNumIterArSetup( (void *) solver, num_iter_Ar_pctl_setup ) );
}

JXF_Int 
JXF_3tAPCTLSetNumIterAePrecond( JXF_Solver solver, JXF_Int num_iter_Ae_pctl_precond )
{
   return( jxf_3tAPCTLSetNumIterAePrecond( (void *) solver, num_iter_Ae_pctl_precond ) );
}

JXF_Int 
JXF_3tAPCTLSetNumIterAiPrecond( JXF_Solver solver, JXF_Int num_iter_Ai_pctl_precond )
{
   return( jxf_3tAPCTLSetNumIterAiPrecond( (void *) solver, num_iter_Ai_pctl_precond ) );
}

JXF_Int 
JXF_3tAPCTLSetNumIterArPrecond( JXF_Solver solver, JXF_Int num_iter_Ar_pctl_precond )
{
   return( jxf_3tAPCTLSetNumIterArPrecond( (void *) solver, num_iter_Ar_pctl_precond ) );
}

JXF_Int 
JXF_3tAPCTLSetNumIterAcPrecond( JXF_Solver solver, JXF_Int num_iter_Ac_pctl_precond )
{
   return( jxf_3tAPCTLSetNumIterAcPrecond( (void *) solver, num_iter_Ac_pctl_precond ) );
}

JXF_Int 
JXF_3tAPCTLSetA( JXF_Solver solver, jxf_ParCSRMatrix *A )
{
   return( jxf_3tAPCTLSetA( (void *) solver, A ) );
}

JXF_Int 
JXF_3tAPCTLSetSubBlocks( JXF_Solver solver )
{
   return( jxf_3tAPCTLSetSubBlocks( (void *) solver ) );
}

JXF_Int 
JXF_3tAPCTLSetARR( JXF_Solver solver, jxf_ParCSRMatrix *ARR )
{
   return( jxf_3tAPCTLSetARR( (void *) solver, ARR ) );
}

JXF_Int 
JXF_3tAPCTLSetAEE( JXF_Solver solver, jxf_ParCSRMatrix *AEE )
{
   return( jxf_3tAPCTLSetAEE( (void *) solver, AEE ) );
}

JXF_Int 
JXF_3tAPCTLSetAII( JXF_Solver solver, jxf_ParCSRMatrix *AII )
{
   return( jxf_3tAPCTLSetAII( (void *) solver, AII ) );
}

JXF_Int 
JXF_3tAPCTLSetVRE( JXF_Solver solver, jxf_ParVector *VRE )
{
   return( jxf_3tAPCTLSetVRE( (void *) solver, VRE ) );
}

JXF_Int 
JXF_3tAPCTLSetVER( JXF_Solver solver, jxf_ParVector *VER )
{
   return( jxf_3tAPCTLSetVER( (void *) solver, VER ) );
}

JXF_Int 
JXF_3tAPCTLSetVER2( JXF_Solver solver, jxf_ParVector **VER )
{
   return( jxf_3tAPCTLSetVER2( (void *) solver, VER ) );
}

JXF_Int 
JXF_3tAPCTLSetVEI( JXF_Solver solver, jxf_ParVector *VEI )
{
   return( jxf_3tAPCTLSetVEI( (void *) solver, VEI ) );
}

JXF_Int 
JXF_3tAPCTLSetVIE( JXF_Solver solver, jxf_ParVector *VIE )
{
   return( jxf_3tAPCTLSetVIE( (void *) solver, VIE ) );
}

JXF_Int 
JXF_3tAPCTLSetARRAll( JXF_Solver solver, jxf_ParCSRMatrix *ARR )
{
   return( jxf_3tAPCTLSetARRAll( (void *) solver, ARR ) );
}

JXF_Int 
JXF_3tAPCTLSetARRAll2( JXF_Solver solver, jxf_ParCSRMatrix **ARR )
{
   return( jxf_3tAPCTLSetARRAll2( (void *) solver, ARR ) );
}

JXF_Int 
JXF_3tAPCTLSetAEEAll( JXF_Solver solver, jxf_ParCSRMatrix *AEE )
{
   return( jxf_3tAPCTLSetAEEAll( (void *) solver, AEE ) );
}

JXF_Int 
JXF_3tAPCTLSetAIIAll( JXF_Solver solver, jxf_ParCSRMatrix *AII )
{
   return( jxf_3tAPCTLSetAIIAll( (void *) solver, AII ) );
}

JXF_Int 
JXF_3tAPCTLSetVREAll( JXF_Solver solver, jxf_ParVector *VRE )
{
   return( jxf_3tAPCTLSetVREAll( (void *) solver, VRE ) );
}

JXF_Int 
JXF_3tAPCTLSetVREAll2( JXF_Solver solver, jxf_ParVector **VRE )
{
   return( jxf_3tAPCTLSetVREAll2( (void *) solver, VRE ) );
}

JXF_Int 
JXF_3tAPCTLSetVERAll( JXF_Solver solver, jxf_ParVector *VER )
{
   return( jxf_3tAPCTLSetVERAll( (void *) solver, VER ) );
}

JXF_Int 
JXF_3tAPCTLSetVERAll2( JXF_Solver solver, jxf_ParVector **VER )
{
   return( jxf_3tAPCTLSetVERAll2( (void *) solver, VER ) );
}

JXF_Int 
JXF_3tAPCTLSetVEIAll( JXF_Solver solver, jxf_ParVector *VEI )
{
   return( jxf_3tAPCTLSetVEIAll( (void *) solver, VEI ) );
}

JXF_Int 
JXF_3tAPCTLSetVIEAll( JXF_Solver solver, jxf_ParVector *VIE )
{
   return( jxf_3tAPCTLSetVIEAll( (void *) solver, VIE ) );
}

JXF_Int
JXF_3tAPCTLIterCount( JXF_Solver solver )
{
   return( jxf_3tAPCTLIterCount( (void *) solver) );
}

JXF_Int 
JXF_3tAPCTLSetCommX( JXF_Solver solver, MPI_Comm comm )
{
   return( jxf_3tAPCTLSetCommX( (void *) solver, comm ) );
}

JXF_Int 
JXF_3tAPCTLSetComm( JXF_Solver solver, MPI_Comm comm )
{
   return( jxf_3tAPCTLSetComm( (void *) solver, comm ) );
}

JXF_Int 
JXF_3tAPCTLSetCommY( JXF_Solver solver, MPI_Comm comm )
{
   return( jxf_3tAPCTLSetCommY( (void *) solver, comm ) );
}

JXF_Int 
JXF_3tAPCTLSetGroupIdX( JXF_Solver solver, JXF_Int groupid )
{
   return( jxf_3tAPCTLSetGroupIdX( (void *) solver, groupid ) );
}

JXF_Int 
JXF_3tAPCTLSetGroupIdY( JXF_Solver solver, JXF_Int groupid )
{
   return( jxf_3tAPCTLSetGroupIdY( (void *) solver, groupid ) );
}

/*!
 * \fn JXF_Int JXF_3tAPCTLDataInitialize
 * \brief Initialize a 3tAPCTLData object.
 * \author peghoty
 * \date 2011/09/17
 */
JXF_Int
JXF_3tAPCTLDataInitialize( JXF_Solver *solver )
{
   *solver = (JXF_Solver) jxf_3tAPCTLDataInitialize( );
   if (!solver)
   {
      jxf_error_in_arg(1);
   }
   return jxf_error_flag;
}

/*!
 * \fn JXF_Int JXF_3tAPCTLDestroy
 * \brief Destroy a 3tAPCTLData object.
 * \author peghoty
 * \date 2011/09/17
 */
JXF_Int 
JXF_3tAPCTLDestroy( JXF_Solver solver )
{
   return( jxf_3tAPCTLDestroy( (void *) solver ) );
}

/*!
 * \fn JXF_Int jxf_3tAPCTLSetXXXXXX
 * \brief Set parameters for PCTL.
 * \author peghoty
 * \date 2011/09/17
 */
JXF_Int 
jxf_3tAPCTLSetTol( void *solver, JXF_Real tol )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataTol(pre_3tapctl_data) = tol;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetMaxiter( void *solver, JXF_Int max_iter )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataMaxIter(pre_3tapctl_data) = max_iter;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNumRlxPre( void *solver, JXF_Int num_relax )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNumRlxPre(pre_3tapctl_data) = num_relax;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNumRlxPost( void *solver, JXF_Int num_relax )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNumRlxPost(pre_3tapctl_data) = num_relax;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetPrintLevel( void *solver, JXF_Int print_level )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataPrintLevel(pre_3tapctl_data) = print_level;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetIsDiagElmFirst( void *solver, JXF_Int is_diagelm_first )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataIsDiagElmFirst(pre_3tapctl_data) = is_diagelm_first;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNumGroup( void *solver, JXF_Int num_group )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   }

   jxf_3tAPCTLDataNumGroup(pre_3tapctl_data) = num_group;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNumIterations( void *solver, JXF_Int num_iterations )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNumIterations(pre_3tapctl_data) = num_iterations;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetLastRelNrm( void *solver, JXF_Real last_rel_nrm )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataLastRelNrm(pre_3tapctl_data) = last_rel_nrm;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAveConvFactor( void *solver, JXF_Real ave_conv_factor )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAveConvFactor(pre_3tapctl_data) = ave_conv_factor;

   return jxf_error_flag;
}

JXF_Int
jxf_3tAPCTLGetNumIterations( void *solver, JXF_Int *num_iterations )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 
   *num_iterations = jxf_3tAPCTLDataNumIterations(pre_3tapctl_data);

   return jxf_error_flag;
} 

JXF_Int  
jxf_3tAPCTLGetLastRelNrm( void *solver, JXF_Real *rel_resid_norm ) 
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 
   *rel_resid_norm = jxf_3tAPCTLDataLastRelNrm(pre_3tapctl_data);

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNpR( void *solver, JXF_Int np )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNpR(pre_3tapctl_data) = np;

   return jxf_error_flag;
}


JXF_Int 
jxf_3tAPCTLSetNpE( void *solver, JXF_Int np )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNpE(pre_3tapctl_data) = np;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNpI( void *solver, JXF_Int np )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNpI(pre_3tapctl_data) = np;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetARRInterpMaxIt( void *solver, JXF_Int max_iter )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data) = max_iter;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAIIInterpMaxIt( void *solver, JXF_Int max_iter )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data) = max_iter;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetARRRelaxMaxIt( void *solver, JXF_Int max_iter )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data) = max_iter;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAEERelaxMaxIt( void *solver, JXF_Int max_iter )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data) = max_iter;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAIIRelaxMaxIt( void *solver, JXF_Int max_iter )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data) = max_iter;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetACCRelaxMaxIt( void *solver, JXF_Int max_iter )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data) = max_iter;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetMaxItDefault( void *solver, JXF_Int maxit_default )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataMaxItDefault(pre_3tapctl_data) = maxit_default;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetARRInterpTol( void *solver, JXF_Real tol )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataARRInterpTol(pre_3tapctl_data) = tol;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAIIInterpTol( void *solver, JXF_Real tol )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAIIInterpTol(pre_3tapctl_data) = tol;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetARRRelaxTol( void *solver, JXF_Real tol )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataARRRelaxTol(pre_3tapctl_data) = tol;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAEERelaxTol( void *solver, JXF_Real tol )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAEERelaxTol(pre_3tapctl_data) = tol;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAIIRelaxTol( void *solver, JXF_Real tol )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data) = tol;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetACCRelaxTol( void *solver, JXF_Real tol )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataACCRelaxTol(pre_3tapctl_data) = tol;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetARRRelaxType( void *solver, JXF_Int ARR_relax_type )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAEERelaxType( void *solver, JXF_Int AEE_relax_type )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAIIRelaxType( void *solver, JXF_Int AII_relax_type )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetTolDefault( void *solver, JXF_Real tol_default )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataTolDefault(pre_3tapctl_data) = tol_default;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThetaWCR( void *solver, JXF_Real theta )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThetaWCR(pre_3tapctl_data) = theta;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThetaWCE( void *solver, JXF_Real theta )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThetaWCE(pre_3tapctl_data) = theta;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThetaWCI( void *solver, JXF_Real theta )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThetaWCI(pre_3tapctl_data) = theta;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThresholdWCR( void *solver, JXF_Real threshold )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThresholdWCR(pre_3tapctl_data) = threshold;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThresholdWCE( void *solver, JXF_Real threshold )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThresholdWCE(pre_3tapctl_data) = threshold;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThresholdWCI( void *solver, JXF_Real threshold )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThresholdWCI(pre_3tapctl_data) = threshold;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetISWCR( void *solver, JXF_Int flag )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataISWCR(pre_3tapctl_data) = flag;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetISWCE( void *solver, JXF_Int flag )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataISWCE(pre_3tapctl_data) = flag;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetISWCI( void *solver, JXF_Int flag )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataISWCI(pre_3tapctl_data) = flag;

   return jxf_error_flag;
}
 
JXF_Int 
jxf_3tAPCTLSetThetaDDR( void *solver, JXF_Real theta )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThetaDDR(pre_3tapctl_data) = theta;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThetaDDE( void *solver, JXF_Real theta )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThetaDDE(pre_3tapctl_data) = theta;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThetaDDI( void *solver, JXF_Real theta )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThetaDDI(pre_3tapctl_data) = theta;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThresholdDDR( void *solver, JXF_Real threshold )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThresholdDDR(pre_3tapctl_data) = threshold;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThresholdDDE( void *solver, JXF_Real threshold )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThresholdDDE(pre_3tapctl_data) = threshold;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetThresholdDDI( void *solver, JXF_Real threshold )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataThresholdDDI(pre_3tapctl_data) = threshold;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetISDDR( void *solver, JXF_Int flag )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataISDDR(pre_3tapctl_data) = flag;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetISDDE( void *solver, JXF_Int flag )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataISDDE(pre_3tapctl_data) = flag;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetISDDI( void *solver, JXF_Int flag )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataISDDI(pre_3tapctl_data) = flag;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNeedCC( void *solver, JXF_Int flag )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNeedCC(pre_3tapctl_data) = flag;

   return jxf_error_flag;
} 


JXF_Int 
jxf_3tAPCTLSetBlockSmoothType( void *solver, JXF_Int blocksmooth_type )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataBlockSmoothType(pre_3tapctl_data) = blocksmooth_type;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetFixItPCTLR( void *solver, JXF_Int fixit )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataFixItPCTLR(pre_3tapctl_data) = fixit;

   return jxf_error_flag;
} 

JXF_Int 
jxf_3tAPCTLSetFixItPCTLE( void *solver, JXF_Int fixit )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataFixItPCTLE(pre_3tapctl_data) = fixit;

   return jxf_error_flag;
} 

JXF_Int 
jxf_3tAPCTLSetFixItPCTLI( void *solver, JXF_Int fixit )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataFixItPCTLI(pre_3tapctl_data) = fixit;

   return jxf_error_flag;
} 

JXF_Int 
jxf_3tAPCTLSetFixItBRLXR( void *solver, JXF_Int fixit )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataFixItBRLXR(pre_3tapctl_data) = fixit;

   return jxf_error_flag;
} 

JXF_Int 
jxf_3tAPCTLSetFixItBRLXE( void *solver, JXF_Int fixit )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataFixItBRLXE(pre_3tapctl_data) = fixit;

   return jxf_error_flag;
} 

JXF_Int 
jxf_3tAPCTLSetFixItBRLXI( void *solver, JXF_Int fixit )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataFixItBRLXI(pre_3tapctl_data) = fixit;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetUseFixedModeR( void *solver, JXF_Int usefixedmode )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataUseFixedModeR(pre_3tapctl_data) = usefixedmode;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetUseFixedModeE( void *solver, JXF_Int usefixedmode )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataUseFixedModeE(pre_3tapctl_data) = usefixedmode;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetUseFixedModeI( void *solver, JXF_Int usefixedmode )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataUseFixedModeI(pre_3tapctl_data) = usefixedmode;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetUsePPCTL( void *solver, JXF_Int use_ppctl )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataUsePPCTL(pre_3tapctl_data) = use_ppctl;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetARRSolverID( void *solver, JXF_Int solverid )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataARRSolverID(pre_3tapctl_data) = solverid;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetACCSolverID( void *solver, JXF_Int solverid )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data) = solverid;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetARRKDim( void *solver, JXF_Int kdim )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataARRKDim(pre_3tapctl_data) = kdim;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetACCKDim( void *solver, JXF_Int kdim )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataACCKDim(pre_3tapctl_data) = kdim;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetTestSubLSIter( void *solver, JXF_Int test_subls_iter )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data) = test_subls_iter;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetDebugFlag( void *solver, JXF_Int debug_flag )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataDebugFlag(pre_3tapctl_data) = debug_flag;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetResetZero( void *solver, JXF_Int reset_zero )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataResetZero(pre_3tapctl_data) = reset_zero;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetInterpType( void *solver, JXF_Int interp_type )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataInterpType(pre_3tapctl_data) = interp_type;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetCoarsenType( void *solver, JXF_Int coarsen_type )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataCoarsenType(pre_3tapctl_data) = coarsen_type;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAggNumLevels( void *solver, JXF_Int agg_num_levels )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAggNumLevels(pre_3tapctl_data) = agg_num_levels;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetCoarseThreshold( void *solver, JXF_Int coarse_threshold )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataCoarseThreshold(pre_3tapctl_data) = coarse_threshold;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetPrintLevelAMG( void *solver, JXF_Int print_level_amg )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data) = print_level_amg;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetStrongThreshold( void *solver, JXF_Real strong_threshold )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataStrongThreshold(pre_3tapctl_data) = strong_threshold;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNumIterAiSetup( void *solver, JXF_Int num_iter_Ai_pctl_setup )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data) = num_iter_Ai_pctl_setup;

   return jxf_error_flag;
} 

JXF_Int 
jxf_3tAPCTLSetNumIterArSetup( void *solver, JXF_Int num_iter_Ar_pctl_setup )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = num_iter_Ar_pctl_setup;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNumIterAePrecond( void *solver, JXF_Int num_iter_Ae_pctl_precond )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) = num_iter_Ae_pctl_precond;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNumIterAiPrecond( void *solver, JXF_Int num_iter_Ai_pctl_precond )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) = num_iter_Ai_pctl_precond;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetNumIterArPrecond( void *solver, JXF_Int num_iter_Ar_pctl_precond )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) = num_iter_Ar_pctl_precond;

   return jxf_error_flag;
}                

JXF_Int 
jxf_3tAPCTLSetNumIterAcPrecond( void *solver, JXF_Int num_iter_Ac_pctl_precond )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) = num_iter_Ac_pctl_precond;

   return jxf_error_flag;
}  

JXF_Int 
jxf_3tAPCTLSetA( void *solver, jxf_ParCSRMatrix *A )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataA(pre_3tapctl_data) = A;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetSubBlocks( void *solver )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;
   jxf_ParCSRMatrix *A = jxf_3tAPCTLDataA(pre_3tapctl_data);
   JXF_Int ng = jxf_3tAPCTLDataNumGroup(pre_3tapctl_data);
   MPI_Comm comm = jxf_ParCSRMatrixComm(A);
   MPI_Comm comm_x, comm_y;
   JXF_Int myid, nprocs, np_R;

   JXF_Int groupid_x = MPI_UNDEFINED;
   JXF_Int groupid_y = MPI_UNDEFINED;

   jxf_ParCSRMatrix  *ARR = NULL;
   jxf_ParCSRMatrix  *AEE = NULL;
   jxf_ParCSRMatrix  *AII = NULL;
   jxf_ParVector     *VRE = NULL;
   jxf_ParVector    **VER = NULL;
   jxf_ParVector     *VEI = NULL;
   jxf_ParVector     *VIE = NULL;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (!jxf_3tAPCTLDataIsDiagElmFirst(pre_3tapctl_data))
   {
      jxf_CSRMatrixReorder(jxf_ParCSRMatrixDiag(A));
   }
   np_R = nprocs / (ng + 2);
   groupid_x = myid / np_R;
   jxf_MPI_Comm_split(comm, groupid_x, myid, &comm_x);
   groupid_y = myid % np_R;
   jxf_MPI_Comm_split(comm, groupid_y, myid, &comm_y);
   jxf_mgGenerateSubBlocks(comm, comm_x, groupid_x, ng, A, &ARR, &AEE, &AII, &VRE, &VER, &VEI, &VIE);
   jxf_3tAPCTLSetARR(solver, ARR);
   jxf_3tAPCTLSetAEE(solver, AEE);
   jxf_3tAPCTLSetAII(solver, AII);
   jxf_3tAPCTLSetVRE(solver, VRE);
   jxf_3tAPCTLSetVER2(solver, VER);
   jxf_3tAPCTLSetVEI(solver, VEI);
   jxf_3tAPCTLSetVIE(solver, VIE);
   jxf_3tAPCTLSetNpR(solver, np_R);
   jxf_3tAPCTLSetComm(solver, comm);
   jxf_3tAPCTLSetCommX(solver, comm_x);
   jxf_3tAPCTLSetCommY(solver, comm_y);
   jxf_3tAPCTLSetGroupIdX(solver, groupid_x);

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetARR( void *solver, jxf_ParCSRMatrix *ARR )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataARR(pre_3tapctl_data) = ARR;

   return jxf_error_flag;
} 

JXF_Int 
jxf_3tAPCTLSetAEE( void *solver, jxf_ParCSRMatrix *AEE )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAEE(pre_3tapctl_data) = AEE;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAII( void *solver, jxf_ParCSRMatrix *AII )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAII(pre_3tapctl_data) = AII;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVRE( void *solver, jxf_ParVector *VRE )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataVRE(pre_3tapctl_data) = VRE;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVER( void *solver, jxf_ParVector *VER )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataVER(pre_3tapctl_data) = VER;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVER2( void *solver, jxf_ParVector **VER )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   }

   jxf_3tAPCTLDataVER2(pre_3tapctl_data) = VER;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVEI( void *solver, jxf_ParVector *VEI )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataVEI(pre_3tapctl_data) = VEI;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVIE( void *solver, jxf_ParVector *VIE )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataVIE(pre_3tapctl_data) = VIE;

   return jxf_error_flag;
}                

JXF_Int 
jxf_3tAPCTLSetARRAll( void *solver, jxf_ParCSRMatrix *ARR )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataARRAll(pre_3tapctl_data) = ARR;

   return jxf_error_flag;
} 

JXF_Int 
jxf_3tAPCTLSetARRAll2( void *solver, jxf_ParCSRMatrix **ARR )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   }

   jxf_3tAPCTLDataARRAll2(pre_3tapctl_data) = ARR;

   return jxf_error_flag;
} 

JXF_Int 
jxf_3tAPCTLSetAEEAll( void *solver, jxf_ParCSRMatrix *AEE )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAEEAll(pre_3tapctl_data) = AEE;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetAIIAll( void *solver, jxf_ParCSRMatrix *AII )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataAIIAll(pre_3tapctl_data) = AII;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVREAll( void *solver, jxf_ParVector *VRE )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataVREAll(pre_3tapctl_data) = VRE;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVREAll2( void *solver, jxf_ParVector **VRE )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataVREAll2(pre_3tapctl_data) = VRE;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVERAll( void *solver, jxf_ParVector *VER )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataVERAll(pre_3tapctl_data) = VER;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVERAll2( void *solver, jxf_ParVector **VER )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   }

   jxf_3tAPCTLDataVERAll2(pre_3tapctl_data) = VER;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVEIAll( void *solver, jxf_ParVector *VEI )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataVEIAll(pre_3tapctl_data) = VEI;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetVIEAll( void *solver, jxf_ParVector *VIE )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataVIEAll(pre_3tapctl_data) = VIE;

   return jxf_error_flag;
}                

JXF_Int
jxf_3tAPCTLIterCount( void *solver )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;
   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);

   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jxf_3tAPCTLIterCount_sp(pre_3tapctl_data);
   }
   else if (nprocs > 1)
   {
      jxf_3tAPCTLIterCount_mp(pre_3tapctl_data);
   }
   
   return 0;
}

JXF_Int 
jxf_3tAPCTLSetComm( void *solver, MPI_Comm comm )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataComm(pre_3tapctl_data) = comm;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetCommX( void *solver, MPI_Comm comm )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataCommX(pre_3tapctl_data) = comm;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetCommY( void *solver, MPI_Comm comm )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataCommY(pre_3tapctl_data) = comm;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetGroupIdX( void *solver, JXF_Int groupid )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data) = groupid;

   return jxf_error_flag;
}

JXF_Int 
jxf_3tAPCTLSetGroupIdY( void *solver, JXF_Int groupid )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jxf_printf("Warning! 3tAPCTL object empty!\n");
      jxf_error_in_arg(1);
      return jxf_error_flag;
   } 

   jxf_3tAPCTLDataGroupIdY(pre_3tapctl_data) = groupid;

   return jxf_error_flag;
}   
         
/*!
 * \fn void *jxf_3tAPCTLDataInitialize
 * \brief Create a jxf_3tAPCTLData object.
 * \author peghoty 
 * \date 2011/09/17
 */
void *
jxf_3tAPCTLDataInitialize()
{
   jxf_3tAPCTLData *pre_3tapctl_data = NULL;  
   
   JXF_Real   tol;               // tolerance    
   JXF_Int      max_iter;          // maximal number of iteration
   JXF_Int      num_relax_pre;     // number of pre-relaxations 
   JXF_Int      num_relax_post;    // number of post-relaxations     
   JXF_Int      print_level;       // 0: switch off; 1: print time; 2: print number of iter; 3: both 1 and 2 

   JXF_Int      num_group;

   JXF_Int      reset_zero;

   JXF_Int      is_diagelm_first;

   /* parameters to describe the weaking coupling */
   JXF_Real theta_wc_R;
   JXF_Real theta_wc_E;
   JXF_Real theta_wc_I;
   JXF_Real threshold_wc_R;
   JXF_Real threshold_wc_E;
   JXF_Real threshold_wc_I;

   /* parameters to describe the diagonal dominance */
   JXF_Real theta_dd_R;
   JXF_Real theta_dd_E;
   JXF_Real theta_dd_I;
   JXF_Real threshold_dd_R;
   JXF_Real threshold_dd_E;
   JXF_Real threshold_dd_I;
   
   /* type of the block smoothing */
   JXF_Int    blocksmooth_type; 
   
   /* maximal number of iterations and tolerance for submatrices */ 
   JXF_Int    maxit_default;
   JXF_Real tol_default;
   
   /* fixed number of iterations and tolerance for submatrices */ 
   JXF_Int    fixit_pctl_R;
   JXF_Int    fixit_pctl_E;
   JXF_Int    fixit_pctl_I;
   JXF_Int    fixit_brlx_R;
   JXF_Int    fixit_brlx_E;
   JXF_Int    fixit_brlx_I;

   /* whether employ the fixed-number-of-iterations mode? peghoty, 2012/02/15 */
   JXF_Int    use_fixedmode_R;
   JXF_Int    use_fixedmode_E;
   JXF_Int    use_fixedmode_I;
   
   /* Whether use the pure PCTL? */
   JXF_Int    use_ppctl;    
   
   /* solver type for interpolation-building of PRR */
   JXF_Int    ARR_solver_id; // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;

   /* solver type for PCTL iteration */
   JXF_Int    ACC_solver_id; // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   
   /* restart parameters for GMRES solver */
   JXF_Int    ARR_kdim;
   JXF_Int    ACC_kdim;
   
   /* Whether test the number of iterations for each sub-linearsystem solution? */
   JXF_Int    test_subls_iter;

   /* variables to count the number of iteration */
   JXF_Int    num_iter_Ai_pctl_setup;
   JXF_Int    num_iter_Ar_pctl_setup;
   JXF_Int    num_iter_Ae_pctl_precond;
   JXF_Int    num_iter_Ai_pctl_precond;
   JXF_Int    num_iter_Ar_pctl_precond;
   JXF_Int    num_iter_Ac_pctl_precond;                 

   JXF_Int    debug_flag;

   /* set some parameters by default */
   tol               = 1.0e-6;            
   max_iter          = 100; 
   num_relax_pre     = 1;
   num_relax_post    = 0;                   
   print_level       = 0;    

   num_group         = 1;

   theta_wc_R        = 1.0e-4;
   theta_wc_E        = 1.0e-4;
   theta_wc_I        = 1.0e-4;
   threshold_wc_R    = 0.5;
   threshold_wc_E    = 0.5;
   threshold_wc_I    = 0.5;
   
   theta_dd_R        = 0.9;
   theta_dd_E        = 0.9;
   theta_dd_I        = 0.9;
   threshold_dd_R    = 1.0;
   threshold_dd_E    = 1.0;
   threshold_dd_I    = 1.0; 
   
   blocksmooth_type  = BLOCKSMOOTH_BD; 
   
   maxit_default     = 200;
   tol_default       = 1.0e-6;
   
   fixit_pctl_R      = 1;
   fixit_pctl_E      = 1;
   fixit_pctl_I      = 1;
   fixit_brlx_R      = 3;
   fixit_brlx_E      = 1;
   fixit_brlx_I      = 1;
   
   use_fixedmode_R   = 1;
   use_fixedmode_E   = 1;
   use_fixedmode_I   = 1; 
   
   use_ppctl         = 0;     
   
   ARR_solver_id     = SOLVER_AMGGMRES;  // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   ACC_solver_id     = SOLVER_AMGGMRES;  // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   ARR_kdim          = 20;
   ACC_kdim          = 20;
   
   test_subls_iter   = 0;

   num_iter_Ai_pctl_setup   = 0;
   num_iter_Ar_pctl_setup   = 0;
   num_iter_Ae_pctl_precond = 0;
   num_iter_Ai_pctl_precond = 0;
   num_iter_Ar_pctl_precond = 0;
   num_iter_Ac_pctl_precond = 0;   

   debug_flag = 0;

   reset_zero = 1;

   is_diagelm_first = 0;

   /* create a 3tAPCTLData object */
   pre_3tapctl_data = jxf_CTAlloc(jxf_3tAPCTLData, 1);
   
   /* set some parameters by default */
   jxf_3tAPCTLSetTol(pre_3tapctl_data, tol);
   jxf_3tAPCTLSetMaxiter(pre_3tapctl_data, max_iter);       
   jxf_3tAPCTLSetNumRlxPre(pre_3tapctl_data, num_relax_pre);
   jxf_3tAPCTLSetNumRlxPost(pre_3tapctl_data, num_relax_post);
   jxf_3tAPCTLSetPrintLevel(pre_3tapctl_data, print_level);
   
   jxf_3tAPCTLSetNumGroup(pre_3tapctl_data, num_group);

   /* max_iter for each subblock */
   jxf_3tAPCTLSetARRInterpMaxIt(pre_3tapctl_data, maxit_default);
   jxf_3tAPCTLSetAIIInterpMaxIt(pre_3tapctl_data, maxit_default);
   jxf_3tAPCTLSetACCRelaxMaxIt(pre_3tapctl_data, maxit_default);
   jxf_3tAPCTLSetMaxItDefault(pre_3tapctl_data, maxit_default);

   /* tolerance for each subblock */
   jxf_3tAPCTLSetARRInterpTol(pre_3tapctl_data, tol_default);
   jxf_3tAPCTLSetAIIInterpTol(pre_3tapctl_data, tol_default);
   jxf_3tAPCTLSetARRRelaxTol(pre_3tapctl_data, tol_default);
   jxf_3tAPCTLSetAEERelaxTol(pre_3tapctl_data, tol_default);
   jxf_3tAPCTLSetAIIRelaxTol(pre_3tapctl_data, tol_default);
   jxf_3tAPCTLSetACCRelaxTol(pre_3tapctl_data, tol_default);
   jxf_3tAPCTLSetTolDefault(pre_3tapctl_data, tol_default);
   
   jxf_3tAPCTLSetThetaWCR(pre_3tapctl_data, theta_wc_R);
   jxf_3tAPCTLSetThetaWCE(pre_3tapctl_data, theta_wc_E);
   jxf_3tAPCTLSetThetaWCI(pre_3tapctl_data, theta_wc_I);
   jxf_3tAPCTLSetThresholdWCR(pre_3tapctl_data, threshold_wc_R);
   jxf_3tAPCTLSetThresholdWCE(pre_3tapctl_data, threshold_wc_E);
   jxf_3tAPCTLSetThresholdWCI(pre_3tapctl_data, threshold_wc_I);
   jxf_3tAPCTLSetThetaDDR(pre_3tapctl_data, theta_dd_R);
   jxf_3tAPCTLSetThetaDDE(pre_3tapctl_data, theta_dd_E);
   jxf_3tAPCTLSetThetaDDI(pre_3tapctl_data, theta_dd_I);
   jxf_3tAPCTLSetThresholdDDR(pre_3tapctl_data, threshold_dd_R);
   jxf_3tAPCTLSetThresholdDDE(pre_3tapctl_data, threshold_dd_E);
   jxf_3tAPCTLSetThresholdDDI(pre_3tapctl_data, threshold_dd_I);
   
   jxf_3tAPCTLSetBlockSmoothType(pre_3tapctl_data, blocksmooth_type);

   jxf_3tAPCTLSetFixItPCTLR(pre_3tapctl_data, fixit_pctl_R);
   jxf_3tAPCTLSetFixItPCTLE(pre_3tapctl_data, fixit_pctl_E);
   jxf_3tAPCTLSetFixItPCTLI(pre_3tapctl_data, fixit_pctl_I);
   jxf_3tAPCTLSetFixItBRLXR(pre_3tapctl_data, fixit_brlx_R);
   jxf_3tAPCTLSetFixItBRLXE(pre_3tapctl_data, fixit_brlx_E);
   jxf_3tAPCTLSetFixItBRLXI(pre_3tapctl_data, fixit_brlx_I);
   
   jxf_3tAPCTLSetUseFixedModeR(pre_3tapctl_data, use_fixedmode_R);
   jxf_3tAPCTLSetUseFixedModeE(pre_3tapctl_data, use_fixedmode_E);
   jxf_3tAPCTLSetUseFixedModeI(pre_3tapctl_data, use_fixedmode_I);
   
   jxf_3tAPCTLSetUsePPCTL(pre_3tapctl_data, use_ppctl); // peghoty, 2012/03/06
   
   jxf_3tAPCTLSetARRSolverID(pre_3tapctl_data, ARR_solver_id);
   jxf_3tAPCTLSetACCSolverID(pre_3tapctl_data, ACC_solver_id);
   jxf_3tAPCTLSetARRKDim(pre_3tapctl_data, ARR_kdim);
   jxf_3tAPCTLSetACCKDim(pre_3tapctl_data, ACC_kdim);
   
   jxf_3tAPCTLSetTestSubLSIter(pre_3tapctl_data, test_subls_iter);  // peghoty, 2012/02/23           
   jxf_3tAPCTLSetNumIterAiSetup(pre_3tapctl_data, num_iter_Ai_pctl_setup); // peghoty, 2012/02/23
   jxf_3tAPCTLSetNumIterArSetup(pre_3tapctl_data, num_iter_Ar_pctl_setup); // peghoty, 2012/02/23
   jxf_3tAPCTLSetNumIterAePrecond(pre_3tapctl_data, num_iter_Ae_pctl_precond); // peghoty, 2012/02/23
   jxf_3tAPCTLSetNumIterAiPrecond(pre_3tapctl_data, num_iter_Ai_pctl_precond); // peghoty, 2012/02/23
   jxf_3tAPCTLSetNumIterArPrecond(pre_3tapctl_data, num_iter_Ar_pctl_precond); // peghoty, 2012/02/23
   jxf_3tAPCTLSetNumIterAcPrecond(pre_3tapctl_data, num_iter_Ac_pctl_precond); // peghoty, 2012/02/23

   jxf_3tAPCTLSetDebugFlag(pre_3tapctl_data, debug_flag);
   jxf_3tAPCTLSetResetZero(pre_3tapctl_data, reset_zero);
   jxf_3tAPCTLSetIsDiagElmFirst(pre_3tapctl_data, is_diagelm_first);

   /* initialize some members */
   jxf_3tAPCTLDataA(pre_3tapctl_data)              = NULL;
   jxf_3tAPCTLDataARR(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataAEE(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataAII(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataVRE(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataVER(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataVER2(pre_3tapctl_data)           = NULL;
   jxf_3tAPCTLDataVEI(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataVIE(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataARRAll(pre_3tapctl_data)         = NULL;
   jxf_3tAPCTLDataARRAll2(pre_3tapctl_data)        = NULL;
   jxf_3tAPCTLDataAEEAll(pre_3tapctl_data)         = NULL;
   jxf_3tAPCTLDataAIIAll(pre_3tapctl_data)         = NULL;
   jxf_3tAPCTLDataVREAll(pre_3tapctl_data)         = NULL;
   jxf_3tAPCTLDataVREAll2(pre_3tapctl_data)        = NULL;
   jxf_3tAPCTLDataVERAll(pre_3tapctl_data)         = NULL;
   jxf_3tAPCTLDataVERAll2(pre_3tapctl_data)        = NULL;
   jxf_3tAPCTLDataVEIAll(pre_3tapctl_data)         = NULL;
   jxf_3tAPCTLDataVIEAll(pre_3tapctl_data)         = NULL;
   jxf_3tAPCTLDataP(pre_3tapctl_data)              = NULL;
   jxf_3tAPCTLDataPRR(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataPII(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataACC(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataARRAMGSolver(pre_3tapctl_data)   = NULL;
   jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data)   = NULL;
   jxf_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data)   = NULL;
   jxf_3tAPCTLDataACCAMGSolver(pre_3tapctl_data)   = NULL;
   jxf_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = NULL;
   jxf_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = NULL;
   jxf_3tAPCTLDataWRR(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataWEE(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataWII(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataWCC(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataGCC(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataRES(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataRHS(pre_3tapctl_data)            = NULL;
   jxf_3tAPCTLDataJAC(pre_3tapctl_data)            = NULL;

   return pre_3tapctl_data;
} 

/*!
 * \fn JXF_Int jxf_3tAPCTLDestroy
 * \brief Destroy the jxf_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLDestroy( void *vdata )
{
   jxf_3tAPCTLData  *pre_3tapctl_data = vdata;
   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);

   JXF_Int nprocs;
   jxf_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jxf_3tAPCTLDestroy_sp(pre_3tapctl_data);
   }
   else if (nprocs > 1)
   {
      jxf_3tAPCTLDestroy_mp(pre_3tapctl_data);
   }
   
   return 0;
}

/*!
 * \fn JXF_Int jxf_3tAPCTLDestroy_sp
 * \brief Destroy the jxf_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2011/09/27
 */
JXF_Int
jxf_3tAPCTLDestroy_sp( void *vdata )
{
   jxf_3tAPCTLData  *pre_3tapctl_data = vdata;
   
   if (pre_3tapctl_data)
   {  
 
      // ARR
      if ( jxf_3tAPCTLDataARR(pre_3tapctl_data) )
      {
         jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataARR(pre_3tapctl_data) );
      }
      
      // VRE
      if ( jxf_3tAPCTLDataVRE(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataVRE(pre_3tapctl_data) );
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
         
      // AEE
      if ( jxf_3tAPCTLDataAEE(pre_3tapctl_data) )
      {
         jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataAEE(pre_3tapctl_data) );
      }
      
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
      
      // AEE_amg_solver
      if ( jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) )
      {
         jxf_PAMGDestroy( jxf_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) );
      }        
   
      // AII
      if ( jxf_3tAPCTLDataAII(pre_3tapctl_data) )
      {
         jxf_ParCSRMatrixDestroy( jxf_3tAPCTLDataAII(pre_3tapctl_data) );
      }
      
      // VIE
      if ( jxf_3tAPCTLDataVIE(pre_3tapctl_data) )
      {
         jxf_ParVectorDestroy( jxf_3tAPCTLDataVIE(pre_3tapctl_data) );
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
 * \fn JXF_Int jxf_3tAPCTLDestroy_mp
 * \brief Destroy the jxf_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLDestroy_mp( void *vdata )
{
   jxf_3tAPCTLData  *pre_3tapctl_data = vdata;
   JXF_Int groupid_x;
   JXF_Int blocksmooth_type;
   JXF_Int  Need_CC;    
   
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
         
         // VRE
         if ( jxf_3tAPCTLDataVRE(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataVRE(pre_3tapctl_data) );
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
         
         // VIE
         if ( jxf_3tAPCTLDataVIE(pre_3tapctl_data) )
         {
            jxf_ParVectorDestroy( jxf_3tAPCTLDataVIE(pre_3tapctl_data) );
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

/*!
 * \fn JXF_Int jxf_3tAPCTLIterCount_mp
 * \brief Count the number of iterations for each sub-matrices (multi-processor case).
 * \author peghoty 
 * \date 2011/09/17
 */
JXF_Int
jxf_3tAPCTLIterCount_mp( void *solver )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   MPI_Comm comm;
   MPI_Comm comm_x;
   JXF_Int groupid_x;
   JXF_Int ARR_relax_type;
   JXF_Int AEE_relax_type;
   JXF_Int AII_relax_type;
   JXF_Int ARR_solver_id;
   JXF_Int ACC_solver_id;
   JXF_Int Need_CC;
   JXF_Int myid, myid_x;   
   
   if (test_subls_iter)
   { 
      comm   = jxf_3tAPCTLDataComm(pre_3tapctl_data);
      comm_x = jxf_3tAPCTLDataCommX(pre_3tapctl_data);
      groupid_x = jxf_3tAPCTLDataGroupIdX(pre_3tapctl_data);
      ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
      AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
      AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);
      ARR_solver_id = jxf_3tAPCTLDataARRSolverID(pre_3tapctl_data);
      ACC_solver_id = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);
      Need_CC = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);
   
      jxf_MPI_Comm_rank(comm, &myid);
      jxf_MPI_Comm_rank(comm_x, &myid_x);
   
      jxf_MPI_Barrier(comm);
   
      if (Need_CC && groupid_x == 0 && myid_x == 0)
      {
         if (ARR_solver_id == SOLVER_AMG)
         {
            jxf_printf(" === num_iter_Ar_pctl_setup   (AMG)      = %d\n", 
                     jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data));
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jxf_printf(" === num_iter_Ar_pctl_setup   (AMGGMRES) = %d\n", 
                     jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data));
         }
      }
   
      if (Need_CC && groupid_x == 2 && myid_x == 0)
      {
         jxf_printf(" === num_iter_Ai_pctl_setup   (AMG)      = %d\n", 
                  jxf_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data));
      } 
   
      jxf_MPI_Barrier(comm);
   
      if (groupid_x == 0 && myid_x == 0)
      {
         if (ARR_relax_type == RELAX_AMG)
         {
            jxf_printf(" === num_iter_Ar_pctl_precond (AMG)      = %d\n", 
                     jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data));
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jxf_printf(" === num_iter_Ar_pctl_precond (JAC)      = %d\n", 
                     jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data));
         }
      }     

      if (groupid_x == 1 && myid_x == 0)
      {
         if (AEE_relax_type == RELAX_AMG)
         {
            jxf_printf(" === num_iter_Ae_pctl_precond (AMG)      = %d\n", 
                     jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data));
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            jxf_printf(" === num_iter_Ae_pctl_precond (JAC)      = %d\n", 
                     jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data));
         }
      }  

      if (groupid_x == 2 && myid_x == 0)
      {
         if (AII_relax_type == RELAX_AMG)
         {
            jxf_printf(" === num_iter_Ai_pctl_precond (AMG)      = %d\n", 
                     jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data));
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jxf_printf(" === num_iter_Ai_pctl_precond (JAC)      = %d\n", 
                     jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data));
         }
      } 
   
      if (Need_CC && myid == 0)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jxf_printf(" === num_iter_Ac_pctl_precond (AMG)      = %d\n", 
                     jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data));
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jxf_printf(" === num_iter_Ac_pctl_precond (AMGGMRES) = %d\n", 
                     jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data));
         }
      }   
   
      jxf_MPI_Barrier(comm);
      jxf_printf("\n"); 
   }

   return (0);
}

/*!
 * \fn JXF_Int jxf_3tAPCTLIterCount_sp
 * \brief Count the number of iterations for each sub-matrices (single-processor case).
 * \author peghoty 
 * \date 2011/09/26
 */
JXF_Int
jxf_3tAPCTLIterCount_sp( void *solver )
{
   jxf_3tAPCTLData *pre_3tapctl_data = solver;
   JXF_Int test_subls_iter = jxf_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   JXF_Int ARR_solver_id;
   JXF_Int ACC_solver_id;
   JXF_Int ARR_relax_type;
   JXF_Int AEE_relax_type;
   JXF_Int AII_relax_type;
   JXF_Int Need_CC;
      
   if (test_subls_iter)
   {
      ARR_solver_id = jxf_3tAPCTLDataARRSolverID(pre_3tapctl_data);
      ACC_solver_id = jxf_3tAPCTLDataACCSolverID(pre_3tapctl_data);
      ARR_relax_type = jxf_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
      AEE_relax_type = jxf_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
      AII_relax_type = jxf_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);
      Need_CC = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);

      if (Need_CC)
      {
         if (ARR_solver_id == SOLVER_AMG)
         {
            jxf_printf(" === num_iter_Ar_pctl_setup   (AMG)      = %d\n", 
                     jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data));
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jxf_printf(" === num_iter_Ar_pctl_setup   (AMGGMRES) = %d\n", 
                     jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data));
         }      
         jxf_printf(" === num_iter_Ai_pctl_setup   (AMG)      = %d\n", 
                  jxf_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data));
      } 
      
      if (ARR_relax_type == RELAX_AMG)
      {
         jxf_printf(" === num_iter_Ar_pctl_precond (AMG)      = %d\n", 
                  jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data));
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         jxf_printf(" === num_iter_Ar_pctl_precond (JAC)      = %d\n", 
                  jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data));
      }

      if (AEE_relax_type == RELAX_AMG)
      {
         jxf_printf(" === num_iter_Ae_pctl_precond (AMG)      = %d\n", 
                  jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data));
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         jxf_printf(" === num_iter_Ae_pctl_precond (JAC)      = %d\n", 
                  jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data));
      }

      if (AII_relax_type == RELAX_AMG)
      {
         jxf_printf(" === num_iter_Ai_pctl_precond (AMG)      = %d\n", 
                  jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data));
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         jxf_printf(" === num_iter_Ai_pctl_precond (JAC)      = %d\n", 
                  jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data));
      }
   
      if (Need_CC)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jxf_printf(" === num_iter_Ac_pctl_precond (AMG)      = %d\n", 
                     jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data));
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jxf_printf(" === num_iter_Ac_pctl_precond (AMGGMRES) = %d\n", 
                     jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data));
         }
      }   
      jxf_printf("\n"); 
   }

   return (0);
}

/*!
 * \fn JXF_Int jxf_GetAPCTLNumIterOfSubLS
 * \brief Get the number of iterations of the sub-linear systems solution.
 * \author peghoty 
 * \date 2012/03/23
 */
JXF_Int 
jxf_GetAPCTLNumIterOfSubLS( JXF_Solver precond, jxf_APCTLKrylovParam *apctlkrylov_param )
{
   jxf_3tAPCTLData *pre_3tapctl_data = (jxf_3tAPCTLData *)precond;
   
   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   JXF_Int np_R = jxf_3tAPCTLDataNpR(pre_3tapctl_data);
   JXF_Int np_E = jxf_3tAPCTLDataNpE(pre_3tapctl_data);
   JXF_Int rootR_id = 0;
   JXF_Int rootE_id = np_R;
   JXF_Int rootI_id = np_R + np_E; 

   JXF_Int num_iter_Ai_pctl_setup = jxf_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data);
   JXF_Int num_iter_Ar_pctl_setup = jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data);
   JXF_Int num_iter_Ae_pctl_precond = jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data);
   JXF_Int num_iter_Ai_pctl_precond = jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data);
   JXF_Int num_iter_Ar_pctl_precond = jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data);
   JXF_Int num_iter_Ac_pctl_precond = jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data); 
  
   // 从各进程组的根进程广播给所有进程
   jxf_MPI_Bcast(&num_iter_Ar_pctl_setup,   1, JXF_MPI_INT, rootR_id, comm);
   jxf_MPI_Bcast(&num_iter_Ar_pctl_precond, 1, JXF_MPI_INT, rootR_id, comm);
   jxf_MPI_Bcast(&num_iter_Ae_pctl_precond, 1, JXF_MPI_INT, rootE_id, comm);   
   jxf_MPI_Bcast(&num_iter_Ai_pctl_setup,   1, JXF_MPI_INT, rootI_id, comm);
   jxf_MPI_Bcast(&num_iter_Ai_pctl_precond, 1, JXF_MPI_INT, rootI_id, comm);   
   
   jxf_APCTLKrylovParamNumIterAiSetup(apctlkrylov_param)   = num_iter_Ai_pctl_setup;
   jxf_APCTLKrylovParamNumIterArSetup(apctlkrylov_param)   = num_iter_Ar_pctl_setup;
   jxf_APCTLKrylovParamNumIterAePrecond(apctlkrylov_param) = num_iter_Ae_pctl_precond;
   jxf_APCTLKrylovParamNumIterAiPrecond(apctlkrylov_param) = num_iter_Ai_pctl_precond;
   jxf_APCTLKrylovParamNumIterArPrecond(apctlkrylov_param) = num_iter_Ar_pctl_precond;
   jxf_APCTLKrylovParamNumIterAcPrecond(apctlkrylov_param) = num_iter_Ac_pctl_precond;
   jxf_APCTLKrylovParamNeedCC(apctlkrylov_param) = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);
   
   return (0);
}

JXF_Int
jxf_GetAPCTLmgNumIterOfSubLS( JXF_Solver precond, JXF_Int groupid_x, JXF_Int ng, jxf_APCTLKrylovParam *apctlkrylov_param )
{
   jxf_3tAPCTLData *pre_3tapctl_data = (jxf_3tAPCTLData *)precond;

   MPI_Comm comm = jxf_3tAPCTLDataComm(pre_3tapctl_data);
   JXF_Int np_R = jxf_3tAPCTLDataNpR(pre_3tapctl_data);
   JXF_Int num_iter_Ai_pctl_setup = jxf_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data);
   JXF_Int num_iter_Ar_pctl_setup = jxf_3tAPCTLDataNumIterArSetup(pre_3tapctl_data);
   JXF_Int num_iter_Ae_pctl_precond = jxf_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data);
   JXF_Int num_iter_Ai_pctl_precond = jxf_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data);
   JXF_Int num_iter_Ar_pctl_precond = jxf_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data);
   JXF_Int num_iter_Ac_pctl_precond = jxf_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data);
   JXF_Int rootE_id = ng * np_R;
   JXF_Int rootI_id = rootE_id + np_R;
   JXF_Int myid, nprocs, gidx, temp_sum;

   JXF_Int *gather_buff = NULL;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   gather_buff = jxf_CTAlloc(JXF_Int, nprocs);

   jxf_MPI_Gather(&num_iter_Ar_pctl_setup, 1, JXF_MPI_INT, gather_buff, 1, JXF_MPI_INT, 0, comm);
   if (myid == 0)
   {
      temp_sum = 0;
      for (gidx = 0; gidx < ng; gidx ++)
      {
         temp_sum += gather_buff[gidx*np_R];
      }
   }
   jxf_MPI_Bcast(&temp_sum, 1, JXF_MPI_INT, 0, comm);
   jxf_APCTLKrylovParamNumIterArSetup(apctlkrylov_param) = temp_sum;

   jxf_MPI_Gather(&num_iter_Ar_pctl_precond, 1, JXF_MPI_INT, gather_buff, 1, JXF_MPI_INT, 0, comm);
   if (myid == 0)
   {
      temp_sum = 0;
      for (gidx = 0; gidx < ng; gidx ++)
      {
         temp_sum += gather_buff[gidx*np_R];
      }
   }
   jxf_MPI_Bcast(&temp_sum, 1, JXF_MPI_INT, 0, comm);
   jxf_APCTLKrylovParamNumIterArPrecond(apctlkrylov_param) = temp_sum;

   // 从各进程组的根进程广播给所有进程
   jxf_MPI_Bcast(&num_iter_Ae_pctl_precond, 1, JXF_MPI_INT, rootE_id, comm);
   jxf_MPI_Bcast(&num_iter_Ai_pctl_setup,   1, JXF_MPI_INT, rootI_id, comm);
   jxf_MPI_Bcast(&num_iter_Ai_pctl_precond, 1, JXF_MPI_INT, rootI_id, comm);

   jxf_APCTLKrylovParamNumIterAiSetup(apctlkrylov_param)   = num_iter_Ai_pctl_setup;
   jxf_APCTLKrylovParamNumIterAePrecond(apctlkrylov_param) = num_iter_Ae_pctl_precond;
   jxf_APCTLKrylovParamNumIterAiPrecond(apctlkrylov_param) = num_iter_Ai_pctl_precond;
   jxf_APCTLKrylovParamNumIterAcPrecond(apctlkrylov_param) = num_iter_Ac_pctl_precond;
   jxf_APCTLKrylovParamNeedCC(apctlkrylov_param) = jxf_3tAPCTLDataNeedCC(pre_3tapctl_data);

   jxf_TFree(gather_buff);

   return (0);
}
