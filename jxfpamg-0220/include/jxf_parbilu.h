#ifndef JXF_UTIL_HEADER 
#include "jxf_util.h"
#endif

#ifndef JXF_MV_HEADER 
#include "jxf_mv.h"
#endif

#include "jxf_krylov.h"

#define MAX(a,b) (((a)>(b))?(a):(b))  
#define JXF_USING_CUDA 1
#define JXF_USING_GPU 1
/*--------------------------------------------------------------------------
 * jxf_ParILUData
 *--------------------------------------------------------------------------*/

// typedef struct
// {
//    JXF_PtrToSolverFcn   setup;
//    JXF_PtrToSolverFcn   solve;
//    JXF_PtrToDestroyFcn  destroy;

// } jxf_Solver;

/*--------------------------------------------------------------------------
 * Accessor functions for the jxf_Solver structure
 *--------------------------------------------------------------------------*/

// #define jxf_SolverSetup(data)       ((data) -> setup)
// #define jxf_SolverSolve(data)       ((data) -> solve)
// #define jxf_SolverDestroy(data)     ((data) -> destroy)

typedef struct jxf_ParBILUData_struct
{
   /* Base solver data structure */
   //jxf_Solver          base;

   /* General data */
   JXF_Int             global_solver;
   jxf_ParBSRMatrix   *matA;
   jxf_ParBSRMatrix   *matL;
   JXF_Real           *matD;
   jxf_ParVector      *D;
   jxf_ParBSRMatrix   *matU;
   jxf_ParBSRMatrix   *matmL;
   JXF_Real           *matmD;
   jxf_ParBSRMatrix   *matmU;
   jxf_ParBSRMatrix   *matS;
   JXF_Real           *droptol; /* Array of 3 elements, for B, (E and F), S respectively */
   JXF_Int             lfil;
   JXF_Int             maxRowNnz;
   JXF_Int            *CF_marker_array;
   JXF_Int            *perm;
   JXF_Int            *qperm;
   JXF_Real            tol_ddPQ;
   jxf_ParVector      *F;
   jxf_ParVector      *U;
   jxf_ParVector      *residual;
   JXF_Real           *rel_res_norms;
   JXF_Int             num_iterations;
   JXF_Real           *l1_norms;
   JXF_Real            final_rel_residual_norm;
   JXF_Real            tol;
   JXF_Real            operator_complexity;
   JXF_Int             logging;
   JXF_Int             print_level;
   JXF_Int             max_iter;
   JXF_Int             tri_solve;
   JXF_Int             lower_jacobi_iters;
   JXF_Int             IR_iters;
   JXF_Int             upper_jacobi_iters;
   JXF_Int             ilu_type;
   JXF_Int             nLU;
   JXF_Int             nI;
   JXF_Int            *u_end; /* used when schur block is formed */
   JXF_Int             sweep;

   /* Iterative ILU parameters */
   JXF_Int             iter_setup_type;
   JXF_Int             iter_setup_option;
   JXF_Int             setup_max_iter;
   JXF_Int             setup_num_iter;
   JXF_Real            setup_tolerance;
   JXF_Real        *setup_history;

   /* temp vectors for solve phase */
   jxf_ParVector      *Utemp;
   jxf_ParVector      *Ftemp;
   jxf_ParVector      *Xtemp;
   jxf_ParVector      *Ytemp;
   jxf_ParVector      *Ztemp;
   JXF_Real           *uext;
   JXF_Real           *fext;

   /* On GPU, we have to form E and F explicitly, since we don't have much control to it */
#if defined(JXF_USING_GPU)
   jxf_BSRMatrix      *matALU_d; /* Matrix holding ILU of A (for A-smoothing) */
   jxf_BSRMatrix      *matBLU_d; /* Matrix holding ILU of B */
   jxf_BSRMatrix      *matSLU_d; /* Matrix holding ILU of S */
   jxf_BSRMatrix      *matE_d;
   jxf_BSRMatrix      *matF_d;
   jxf_ParBSRMatrix   *Aperm;
   jxf_ParBSRMatrix   *R;
   jxf_ParBSRMatrix   *P;
   jxf_Vector         *Ftemp_upper;
   jxf_Vector         *Utemp_lower;
   jxf_Vector         *Adiag_diag;
   jxf_Vector         *Sdiag_diag;
#endif

   /* data structure sor solving Schur System */
//    JXF_Solver          schur_solver;
//    JXF_Solver          schur_precond;
   jxf_ParVector      *rhs;
   jxf_ParVector      *x;

   /* local reordering */
   JXF_Int             reordering_type;

   /* 层次调度 */
   JXF_Int **L_levels, **U_levels;
   JXF_Int *L_level_sizes, *U_level_sizes;
   JXF_Int L_num_levels, U_num_levels;

   JXF_Int nlevL;      // 下三角最大层级数
   JXF_Int *ilevL;     // 下三角层级起始索引
   JXF_Int *jlevL;     // 下三角按层级排序的行号
   JXF_Int nlevU;      // 上三角最大层级数
   JXF_Int *ilevU;     // 上三角层级起始索引
   JXF_Int *jlevU;     // 上三角按层级排序的行号

   JXF_Int *L_perm;
   JXF_Int *L_iperm;
   JXF_Int *U_perm;
   JXF_Int *U_iperm;

} jxf_ParBILUData;

#define jxf_ParILUDataTestOption(ilu_data)                   ((ilu_data) -> test_opt)

#if defined(JXF_USING_GPU)
#define jxf_ParILUDataMatAILUDevice(ilu_data)                ((ilu_data) -> matALU_d)
#define jxf_ParILUDataMatBILUDevice(ilu_data)                ((ilu_data) -> matBLU_d)
#define jxf_ParILUDataMatSILUDevice(ilu_data)                ((ilu_data) -> matSLU_d)
#define jxf_ParILUDataMatEDevice(ilu_data)                   ((ilu_data) -> matE_d)
#define jxf_ParILUDataMatFDevice(ilu_data)                   ((ilu_data) -> matF_d)
#define jxf_ParILUDataAperm(ilu_data)                        ((ilu_data) -> Aperm)
#define jxf_ParILUDataR(ilu_data)                            ((ilu_data) -> R)
#define jxf_ParILUDataP(ilu_data)                            ((ilu_data) -> P)
#define jxf_ParILUDataFTempUpper(ilu_data)                   ((ilu_data) -> Ftemp_upper)
#define jxf_ParILUDataUTempLower(ilu_data)                   ((ilu_data) -> Utemp_lower)
#define jxf_ParILUDataADiagDiag(ilu_data)                    ((ilu_data) -> Adiag_diag)
#define jxf_ParILUDataSDiagDiag(ilu_data)                    ((ilu_data) -> Sdiag_diag)
#endif

#define jxf_ParILUDatasweep(ilu_data)                         ((ilu_data) -> sweep)
#define jxf_ParILUDataGlobalSolver(ilu_data)                 ((ilu_data) -> global_solver)
#define jxf_ParILUDataMatA(ilu_data)                         ((ilu_data) -> matA)
#define jxf_ParILUDataMatL(ilu_data)                         ((ilu_data) -> matL)
#define jxf_ParILUDataMatD(ilu_data)                         ((ilu_data) -> matD)
#define jxf_ParILUDataMatU(ilu_data)                         ((ilu_data) -> matU)
#define jxf_ParILUDataMatLModified(ilu_data)                 ((ilu_data) -> matmL)
#define jxf_ParILUDataMatDModified(ilu_data)                 ((ilu_data) -> matmD)
#define jxf_ParILUDataMatUModified(ilu_data)                 ((ilu_data) -> matmU)
#define jxf_ParILUDataMatS(ilu_data)                         ((ilu_data) -> matS)
#define jxf_ParILUDataDroptol(ilu_data)                      ((ilu_data) -> droptol)
#define jxf_ParILUDataLfil(ilu_data)                         ((ilu_data) -> lfil)
#define jxf_ParILUDataMaxRowNnz(ilu_data)                    ((ilu_data) -> maxRowNnz)
#define jxf_ParILUDataCFMarkerArray(ilu_data)                ((ilu_data) -> CF_marker_array)
#define jxf_ParILUDataPerm(ilu_data)                         ((ilu_data) -> perm)
#define jxf_ParILUDataQPerm(ilu_data)                        ((ilu_data) -> qperm)
#define jxf_ParILUDataTolDDPQ(ilu_data)                      ((ilu_data) -> tol_ddPQ)
#define jxf_ParILUDataD(ilu_data)                            ((ilu_data) -> D)
#define jxf_ParILUDataF(ilu_data)                            ((ilu_data) -> F)
#define jxf_ParILUDataU(ilu_data)                            ((ilu_data) -> U)
#define jxf_ParILUDataResidual(ilu_data)                     ((ilu_data) -> residual)
#define jxf_ParILUDataRelResNorms(ilu_data)                  ((ilu_data) -> rel_res_norms)
#define jxf_ParILUDataNumIterations(ilu_data)                ((ilu_data) -> num_iterations)
#define jxf_ParILUDataL1Norms(ilu_data)                      ((ilu_data) -> l1_norms)
#define jxf_ParILUDataFinalRelResidualNorm(ilu_data)         ((ilu_data) -> final_rel_residual_norm)
#define jxf_ParILUDataTol(ilu_data)                          ((ilu_data) -> tol)
#define jxf_ParILUDataOperatorComplexity(ilu_data)           ((ilu_data) -> operator_complexity)
#define jxf_ParILUDataLogging(ilu_data)                      ((ilu_data) -> logging)
#define jxf_ParILUDataPrintLevel(ilu_data)                   ((ilu_data) -> print_level)
#define jxf_ParILUDataMaxIter(ilu_data)                      ((ilu_data) -> max_iter)
#define jxf_ParILUDataTriSolve(ilu_data)                     ((ilu_data) -> tri_solve)
#define jxf_ParILUDataLowerJacobiIters(ilu_data)             ((ilu_data) -> lower_jacobi_iters)
#define jxf_ParILUDataUpperJacobiIters(ilu_data)             ((ilu_data) -> upper_jacobi_iters)
#define jxf_ParILUDataIRIters(ilu_data)                      ((ilu_data) -> IR_iters)
#define jxf_ParILUDataIluType(ilu_data)                      ((ilu_data) -> ilu_type)
#define jxf_ParILUDataNLU(ilu_data)                          ((ilu_data) -> nLU)
#define jxf_ParILUDataNI(ilu_data)                           ((ilu_data) -> nI)
#define jxf_ParILUDataUEnd(ilu_data)                         ((ilu_data) -> u_end)
#define jxf_ParILUDataUTemp(ilu_data)                        ((ilu_data) -> Utemp)
#define jxf_ParILUDataFTemp(ilu_data)                        ((ilu_data) -> Ftemp)
#define jxf_ParILUDataXTemp(ilu_data)                        ((ilu_data) -> Xtemp)
#define jxf_ParILUDataYTemp(ilu_data)                        ((ilu_data) -> Ytemp)
#define jxf_ParILUDataZTemp(ilu_data)                        ((ilu_data) -> Ztemp)
#define jxf_ParILUDataUExt(ilu_data)                         ((ilu_data) -> uext)
#define jxf_ParILUDataFExt(ilu_data)                         ((ilu_data) -> fext)
// #define jxf_ParILUDataSchurSolver(ilu_data)                  ((ilu_data) -> schur_solver)
// #define jxf_ParILUDataSchurPrecond(ilu_data)                 ((ilu_data) -> schur_precond)
#define jxf_ParILUDataRhs(ilu_data)                          ((ilu_data) -> rhs)
#define jxf_ParILUDataX(ilu_data)                            ((ilu_data) -> x)
#define jxf_ParILUDataReorderingType(ilu_data)               ((ilu_data) -> reordering_type)

/* Iterative ILU setup */
#define jxf_ParILUDataIterativeSetupType(ilu_data)           ((ilu_data) -> iter_setup_type)
#define jxf_ParILUDataIterativeSetupOption(ilu_data)         ((ilu_data) -> iter_setup_option)
#define jxf_ParILUDataIterativeSetupMaxIter(ilu_data)        ((ilu_data) -> setup_max_iter)
#define jxf_ParILUDataIterativeSetupNumIter(ilu_data)        ((ilu_data) -> setup_num_iter)
#define jxf_ParILUDataIterativeSetupTolerance(ilu_data)      ((ilu_data) -> setup_tolerance)
#define jxf_ParILUDataIterativeSetupHistory(ilu_data)        ((ilu_data) -> setup_history)
#define jxf_ParILUDataIterSetupCorrectionNorm(ilu_data,i)    ((ilu_data) -> setup_history[i])
#define jxf_ParILUDataIterSetupResidualNorm(ilu_data,i)      (((ilu_data) -> setup_history + \
                                                                 (ilu_data) -> setup_num_iter)[i])
/* 层次调度 */
#define jxf_ParILUDataL_levels(ilu_data)                   ((ilu_data) -> L_levels)
#define jxf_ParILUDataU_levels(ilu_data)                   ((ilu_data) -> U_levels)
#define jxf_ParILUDataL_level_sizes(ilu_data)              ((ilu_data) -> L_level_sizes)
#define jxf_ParILUDataU_level_sizes(ilu_data)              ((ilu_data) -> U_level_sizes)
#define jxf_ParILUDataL_num_levels(ilu_data)               ((ilu_data) -> L_num_levels)
#define jxf_ParILUDataU_num_levels(ilu_data)               ((ilu_data) -> U_num_levels)

#define jxf_ParILUDataL_perm(ilu_data)                         ((ilu_data) -> L_perm)
#define jxf_ParILUDataL_iperm(ilu_data)                        ((ilu_data) -> L_iperm)

#define jxf_ParILUDataU_perm(ilu_data)                         ((ilu_data) -> U_perm)
#define jxf_ParILUDataU_iperm(ilu_data)                        ((ilu_data) -> U_iperm)


#define FMRK   -1
#define CMRK    1
#define UMRK    0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define MAT_TOL     1e-14
#define EXPAND_FACT 1.3

/* par_ilu.c */
void *jxf_ILUCreate ( void );
JXF_Int JXF_ILUCreate( JXF_Solver *solver );
JXF_Int jxf_ILUDestroy ( void *ilu_vdata );
JXF_Int JXF_ILUDestroy( JXF_Solver solver );

/* parbsr_bilu.c - BILU-specific functions */
void *jxf_BILUCreate( void );
JXF_Int JXF_BILUCreate( JXF_Solver *solver );
JXF_Int jxf_BILUDestroy( void *ilu_vdata );
JXF_Int JXF_BILUDestroy( JXF_Solver solver );
JXF_Int jxf_ILUSetLevelOfFill( void *ilu_vdata, JXF_Int lfil );
JXF_Int JXF_ILUSetLevelOfFill( JXF_Solver solver, JXF_Int lfil );
JXF_Int jxf_ILUSetMaxNnzPerRow( void *ilu_vdata, JXF_Int nzmax );
JXF_Int JXF_ILUSetMaxNnzPerRow( JXF_Solver solver, JXF_Int nzmax );

JXF_Int jxf_ILUSetDropThreshold( void *ilu_vdata, JXF_Real threshold );
JXF_Int JXF_ILUSetDropThreshold( JXF_Solver solver, JXF_Real threshold );

JXF_Int jxf_ILUSetDropThresholdArray( void *ilu_vdata, JXF_Real *threshold );
JXF_Int JXF_ILUSetDropThresholdArray( JXF_Solver solver, JXF_Real *threshold );

JXF_Int jxf_ILUSetType( void *ilu_vdata, JXF_Int ilu_type );
JXF_Int JXF_ILUSetType( JXF_Solver solver, JXF_Int ilu_type );

JXF_Int jxf_ILUSetsweep( void *ilu_vdata, JXF_Real sweep );
JXF_Int JXF_ILUSetsweep( JXF_Solver solver, JXF_Real sweep );

JXF_Int jxf_ILUSetMaxIter( void *ilu_vdata, JXF_Int max_iter );
JXF_Int JXF_ILUSetMaxIter( JXF_Solver solver, JXF_Int max_iter );
JXF_Int jxf_ILUSetTol( void *ilu_vdata, JXF_Real tol );
JXF_Int JXF_ILUSetTol( JXF_Solver solver, JXF_Real tol );
JXF_Int jxf_ILUSetIterativeSetupType( void *ilu_vdata, JXF_Int iter_setup_type );
JXF_Int JXF_ILUSetIterativeSetupType( JXF_Solver solver, JXF_Int iter_setup_type );
// JXF_Int jxf_ILUSetIterativeSetupOption( void *ilu_vdata, JXF_Int iter_setup_option );
JXF_Int jxf_ILUSetIterativeSetupMaxIter( void *ilu_vdata, JXF_Int iter_setup_max_iter );
JXF_Int JXF_ILUSetIterativeSetupMaxIter( JXF_Solver solver, JXF_Int iter_setup_max_iter );

JXF_Int jxf_ILUSetIterativeSetupTolerance( void *ilu_vdata, JXF_Real iter_setup_tolerance );
JXF_Int JXF_ILUSetIterativeSetupTolerance( JXF_Solver solver, JXF_Real iter_setup_tolerance );

//JXF_Int jxf_ILUGetIterativeSetupHistory( void *ilu_vdata,JXF_Real **iter_setup_history );
JXF_Int jxf_ILUSetTriSolve( void *ilu_vdata, JXF_Int tri_solve );
JXF_Int JXF_ILUSetTriSolve( JXF_Solver solver, JXF_Int tri_solve );
JXF_Int jxf_ILUSetLowerJacobiIters( void *ilu_vdata, JXF_Int lower_jacobi_iters );
JXF_Int JXF_ILUSetLowerJacobiIters( JXF_Solver solver, JXF_Int lower_jacobi_iterations );
JXF_Int jxf_ILUSetUpperJacobiIters( void *ilu_vdata, JXF_Int upper_jacobi_iters );
JXF_Int JXF_ILUSetUpperJacobiIters( JXF_Solver solver, JXF_Int upper_jacobi_iterations );
JXF_Int jxf_ILUSetPrintLevel( void *ilu_vdata, JXF_Int print_level );
JXF_Int JXF_ILUSetPrintLevel( JXF_Solver solver, JXF_Int print_level );
JXF_Int jxf_ILUSetIRIters( void *ilu_vdata, JXF_Int IR_iters );
JXF_Int JXF_ILUSetIRIters( JXF_Solver solver, JXF_Int IR_iters );

JXF_Int jxf_ILUSetLogging( void *ilu_vdata, JXF_Int logging );
JXF_Int JXF_ILUSetLogging( JXF_Solver solver, JXF_Int logging );
JXF_Int jxf_ILUSetLocalReordering( void *ilu_vdata, JXF_Int ordering_type );
JXF_Int JXF_ILUSetLocalReordering( JXF_Solver solver, JXF_Int reordering_type );
// JXF_Int jxf_ILUSetSchurSolverMaxIter( void *ilu_vdata, JXF_Int ss_max_iter );
// JXF_Int jxf_ILUSetSchurSolverTol( void *ilu_vdata, JXF_Real ss_tol );
// JXF_Int jxf_ILUSetSchurSolverAbsoluteTol( void *ilu_vdata, JXF_Real ss_absolute_tol );
// JXF_Int jxf_ILUSetSchurSolverLogging( void *ilu_vdata, JXF_Int ss_logging );
// JXF_Int jxf_ILUSetSchurSolverPrintLevel( void *ilu_vdata, JXF_Int ss_print_level );
// JXF_Int jxf_ILUSetSchurSolverRelChange( void *ilu_vdata, JXF_Int ss_rel_change );
// JXF_Int jxf_ILUSetSchurPrecondILUType( void *ilu_vdata, JXF_Int sp_ilu_type );
// JXF_Int jxf_ILUSetSchurPrecondILULevelOfFill( void *ilu_vdata, JXF_Int sp_ilu_lfil );
// JXF_Int jxf_ILUSetSchurPrecondILUMaxNnzPerRow( void *ilu_vdata,
//                                                    JXF_Int sp_ilu_max_row_nnz );
// JXF_Int jxf_ILUSetSchurPrecondILUDropThreshold( void *ilu_vdata, JXF_Real sp_ilu_droptol );
// JXF_Int jxf_ILUSetSchurPrecondILUDropThresholdArray( void *ilu_vdata,
//                                                          JXF_Real *sp_ilu_droptol );
// JXF_Int jxf_ILUSetSchurPrecondPrintLevel( void *ilu_vdata, JXF_Int sp_print_level );
// JXF_Int jxf_ILUSetSchurPrecondMaxIter( void *ilu_vdata, JXF_Int sp_max_iter );
// JXF_Int jxf_ILUSetSchurPrecondTriSolve( void *ilu_vdata, JXF_Int sp_tri_solve );
// JXF_Int jxf_ILUSetSchurPrecondLowerJacobiIters( void *ilu_vdata,
//                                                     JXF_Int sp_lower_jacobi_iters );
// JXF_Int jxf_ILUSetSchurPrecondUpperJacobiIters( void *ilu_vdata,
//                                                     JXF_Int sp_upper_jacobi_iters );
// JXF_Int jxf_ILUSetSchurPrecondTol( void *ilu_vdata, JXF_Int sp_tol );
// JXF_Int jxf_ILUSetSchurNSHDropThreshold( void *ilu_vdata, JXF_Real threshold );
// JXF_Int jxf_ILUSetSchurNSHDropThresholdArray( void *ilu_vdata, JXF_Real *threshold );
JXF_Int jxf_ILUGetNumIterations( void *ilu_vdata, JXF_Int *num_iterations );
JXF_Int JXF_ILUGetNumIterations( JXF_Solver solver, JXF_Int *num_iterations );
JXF_Int jxf_ILUGetFinalRelativeResidualNorm( void *ilu_vdata, JXF_Real *res_norm );
JXF_Int JXF_ILUGetFinalRelativeResidualNorm(  JXF_Solver solver, JXF_Real *res_norm );




JXF_Int jxf_GreedyColoring(jxf_ParCSRMatrix *A,JXF_Int nLU,JXF_Int **row_by_color_out,
                     JXF_Int  **color_starts_out,JXF_Int *num_colors_out);
JXF_Int jxf_GreedyColoring_L(jxf_ParCSRMatrix *A,JXF_Int nLU,JXF_Int **row_by_color_out,
                     JXF_Int  **color_starts_out,JXF_Int *num_colors_out);
JXF_Int jxf_GreedyColoring_U(jxf_ParCSRMatrix *A,JXF_Int nLU,JXF_Int **row_by_color_out,
                     JXF_Int  **color_starts_out,JXF_Int *num_colors_out);

JXF_Int jxf_GreedyColoring_L1(jxf_ParCSRMatrix *A,JXF_Int nLU,JXF_Int **row_by_color_out,
                     JXF_Int  **color_starts_out,JXF_Int *num_colors_out,JXF_Int **perm_out,
                  JXF_Int **inv_perm_out);
JXF_Int jxf_GreedyColoring_U1(jxf_ParCSRMatrix *A,JXF_Int nLU,JXF_Int **row_by_color_out,
                     JXF_Int  **color_starts_out,JXF_Int *num_colors_out,JXF_Int **perm_out,
                  JXF_Int **inv_perm_out);
JXF_Int jxf_GreedyColoring_8GS(jxf_ParCSRMatrix *A,JXF_Int nLU,JXF_Int **row_by_color_out,
                     JXF_Int  **color_starts_out,JXF_Int *num_colors_out,
                     JXF_Int **row_by_color_out1, JXF_Int  **color_starts_out1,JXF_Int *num_colors_out1);
//JXF_Int jxf_ILUWriteSolverParams( void *ilu_vdata );

// JXF_Int jxf_ILUMinHeapAddI( JXF_Int *heap, JXF_Int len );
// JXF_Int jxf_ILUMinHeapAddIIIi( JXF_Int *heap, JXF_Int *I1,
//                                    JXF_Int *Ii1, JXF_Int len );
// JXF_Int jxf_ILUMinHeapAddIRIi( JXF_Int *heap, JXF_Real *I1,
//                                    JXF_Int *Ii1, JXF_Int len );
// JXF_Int jxf_ILUMaxrHeapAddRabsI( JXF_Real *heap, JXF_Int *I1, JXF_Int len );
// JXF_Int jxf_ILUMinHeapRemoveI( JXF_Int *heap, JXF_Int len );
// JXF_Int jxf_ILUMinHeapRemoveIIIi( JXF_Int *heap, JXF_Int *I1,
//                                       JXF_Int *Ii1, JXF_Int len );
// JXF_Int jxf_ILUMinHeapRemoveIRIi( JXF_Int *heap, JXF_Real *I1,
//                                       JXF_Int *Ii1, JXF_Int len );
// JXF_Int jxf_ILUMaxrHeapRemoveRabsI( JXF_Real *heap, JXF_Int *I1, JXF_Int len );
// JXF_Int jxf_ILUMaxQSplitRabsI( JXF_Real *arrayR, JXF_Int *arrayI, JXF_Int left,
//                                    JXF_Int bound, JXF_Int right );
// JXF_Int jxf_ILUMaxRabs( JXF_Real *array_data, JXF_Int *array_j, JXF_Int start,
//                             JXF_Int end, JXF_Int nLU, JXF_Int *rperm, JXF_Real *value,
//                             JXF_Int *index, JXF_Real *l1_norm, JXF_Int *nnz );

// JXF_Int jxf_ILUGetPermddPQPre( JXF_Int n, JXF_Int nLU, JXF_Int *A_diag_i,
//                                    JXF_Int *A_diag_j, JXF_Real *A_diag_data,
//                                    JXF_Real tol, JXF_Int *perm, JXF_Int *rperm,
//                                    JXF_Int *pperm_pre, JXF_Int *qperm_pre, JXF_Int *nB );
// JXF_Int jxf_ILUGetPermddPQ( jxf_ParCSRMatrix *A, JXF_Int **io_pperm, JXF_Int **io_qperm,
//                                 JXF_Real tol, JXF_Int *nB, JXF_Int *nI,
//                                 JXF_Int reordering_type );
// JXF_Int jxf_ILUGetInteriorExteriorPerm( jxf_ParCSRMatrix *A,
//                                             JXF_MemoryLocation memory_location,
//                                             JXF_Int **perm, JXF_Int *nLU,
//                                             JXF_Int reordering_type );
// JXF_Int jxf_ILUGetLocalPerm( jxf_ParCSRMatrix *A, JXF_Int **perm_ptr,
//                                  JXF_Int *nLU, JXF_Int reordering_type );
// JXF_Int jxf_ILUBuildRASExternalMatrix( jxf_ParCSRMatrix *A, JXF_Int *rperm,
//                                            JXF_Int **E_i, JXF_Int **E_j, JXF_Real **E_data );
// JXF_Int jxf_ILUSortOffdColmap( jxf_ParCSRMatrix *A );
// JXF_Int jxf_ILULocalRCMBuildFinalPerm( JXF_Int start, JXF_Int end,
//                                            JXF_Int * G_perm, JXF_Int *perm, JXF_Int *qperm,
//                                            JXF_Int **permp, JXF_Int **qpermp );
//JXF_Int jxf_ILULocalRCM( jxf_CSRMatrix *A, JXF_Int start, JXF_Int end,
//                              JXF_Int **permp, JXF_Int **qpermp, JXF_Int sym );
// JXF_Int jxf_ILULocalRCMMindegree( JXF_Int n, JXF_Int *degree,
//                                       JXF_Int *marker, JXF_Int *rootp );
// JXF_Int jxf_ILULocalRCMOrder( jxf_CSRMatrix *A, JXF_Int *perm );
// JXF_Int jxf_ILULocalRCMFindPPNode( jxf_CSRMatrix *A, JXF_Int *rootp, JXF_Int *marker );
// JXF_Int jxf_ILULocalRCMBuildLevel( jxf_CSRMatrix *A, JXF_Int root, JXF_Int *marker,
//                                        JXF_Int *level_i, JXF_Int *level_j, JXF_Int *nlevp );
// JXF_Int jxf_ILULocalRCMNumbering( jxf_CSRMatrix *A, JXF_Int root, JXF_Int *marker,
//                                       JXF_Int *perm, JXF_Int *current_nump );
// JXF_Int jxf_ILULocalRCMQsort( JXF_Int *perm, JXF_Int start, JXF_Int end,
//                                   JXF_Int *degree );
// JXF_Int jxf_ILULocalRCMReverse( JXF_Int *perm, JXF_Int start, JXF_Int end );


/* par_ilu_setup.c */
JXF_Int jxf_ILUSetup(void *ilu_vdata, jxf_ParCSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u);
JXF_Int JXF_ILUSetup( JXF_Solver solver,
                JXF_ParCSRMatrix A,
                JXF_ParVector b,
                JXF_ParVector x);
/* par_ilu_setup_device.c */
JXF_Int jxf_ILUSetupILUDevice( JXF_Int ilu_type, jxf_ParCSRMatrix *A,
                                   JXF_Int lfil, JXF_Real *tol, JXF_Int *perm_data,
                                   JXF_Int *qperm_data, JXF_Int n, JXF_Int nLU,
                                   jxf_CSRMatrix **BLUptr, jxf_ParCSRMatrix **matSptr,
                                   jxf_CSRMatrix **Eptr, jxf_CSRMatrix **Fptr,
                                   JXF_Int tri_solve );
// JXF_Int jxf_ParILUExtractEBFC( jxf_CSRMatrix *A_diag, JXF_Int nLU,
//                                    jxf_CSRMatrix **Bp, jxf_CSRMatrix **Cp,
//                                    jxf_CSRMatrix **Ep, jxf_CSRMatrix **Fp );
// JXF_Int jxf_ParILURAPReorder( jxf_ParCSRMatrix *A, JXF_Int *perm,
//                                   JXF_Int *rqperm, jxf_ParCSRMatrix **A_pq );
// JXF_Int jxf_ILUSetupLDUtoCusparse( jxf_ParCSRMatrix *L, JXF_Real *D,
//                                        jxf_ParCSRMatrix  *U, jxf_ParCSRMatrix **LDUp );
// JXF_Int jxf_ILUSetupRAPMILU0( jxf_ParCSRMatrix *A, jxf_ParCSRMatrix **ALUp,
//                                   JXF_Int modified );
// JXF_Int jxf_ILUSetupRAPILU0Device( jxf_ParCSRMatrix *A, JXF_Int *perm, JXF_Int n,
//                                        JXF_Int nLU, jxf_ParCSRMatrix **Apermptr,
//                                        jxf_ParCSRMatrix **matSptr, jxf_CSRMatrix **ALUptr,
//                                        jxf_CSRMatrix **BLUptr, jxf_CSRMatrix **CLUptr,
//                                        jxf_CSRMatrix **Eptr, jxf_CSRMatrix **Fptr,
//                                        JXF_Int test_opt );
// JXF_Int jxf_ILUSetupRAPILU0( jxf_ParCSRMatrix *A, JXF_Int *perm, JXF_Int n,
//                                  JXF_Int nLU, jxf_ParCSRMatrix **Lptr, JXF_Real **Dptr,
//                                  jxf_ParCSRMatrix **Uptr, jxf_ParCSRMatrix **mLptr,
//                                  JXF_Real **mDptr, jxf_ParCSRMatrix **mUptr,
//                                  JXF_Int **u_end );
JXF_Int jxf_ILUSetupILU0( jxf_ParCSRMatrix  *A, JXF_Int *perm, JXF_Int *qperm,
                              JXF_Int nLU, JXF_Int nI, jxf_ParCSRMatrix **Lptr,
                              JXF_Real **Dptr, jxf_ParCSRMatrix **Uptr,
                              jxf_ParVector **D_array, JXF_Int **u_end );
JXF_Int jxf_ILUSetupMILU0( jxf_ParCSRMatrix *A, JXF_Int *permp,
                               JXF_Int *qpermp, JXF_Int nLU, JXF_Int nI,
                               jxf_ParCSRMatrix **Lptr, JXF_Real **Dptr,
                               jxf_ParCSRMatrix **Uptr, jxf_ParVector **D_array,
                               JXF_Int **u_end, JXF_Int modified );
JXF_Int jxf_ILUSetupFGPILU_v1(jxf_ParCSRMatrix  *A, JXF_Int  *permp,JXF_Int  *qpermp,
                        JXF_Int  sweep,jxf_ParCSRMatrix **Lptr,JXF_Real **Dptr,
                        jxf_ParCSRMatrix **Uptr, jxf_ParCSRMatrix **Sptr,JXF_Int  **u_end);

JXF_Int jxf_ILUSetupFGPILU_v2(jxf_ParCSRMatrix  *A, JXF_Int  *permp,JXF_Int  *qpermp,
                        JXF_Int  sweep,jxf_ParCSRMatrix **Lptr,JXF_Real **Dptr,
                        jxf_ParCSRMatrix **Uptr, jxf_ParVector **D_array, JXF_Int  **u_end);

JXF_Int jxf_ILUSetupFGPILU_v3(jxf_ParCSRMatrix *A,JXF_Int sweep,
                           jxf_ParCSRMatrix **Lptr, jxf_ParVector **D,
                           jxf_ParCSRMatrix **Uptr, jxf_ParCSRMatrix **Sptr,JXF_Int **u_end);

JXF_Int jxf_ILUSetupFGPBILU_v3(jxf_ParBSRMatrix *A,JXF_Int sweep,
                           jxf_ParBSRMatrix **Lptr, jxf_ParVector **D,
                           jxf_ParBSRMatrix **Uptr, jxf_ParBSRMatrix **Sptr,JXF_Int **u_end);


JXF_Int jxf_ParCSRMatrixM(  jxf_ParCSRMatrix *A,jxf_ParVector    *x );

void jxf_ParCSRMatrixMatmul_Apattern(jxf_ParCSRMatrix *A, jxf_ParCSRMatrix *L, jxf_ParCSRMatrix *B, JXF_Real *C_diag_data, JXF_Real *C_offd_data);

JXF_Int jxf_ILUSetupFGPILU(jxf_ParCSRMatrix  *A, JXF_Int  *permp,JXF_Int  *qpermp,
                        JXF_Int  sweep,jxf_ParCSRMatrix **Lptr,JXF_Real **Dptr,
                        jxf_ParCSRMatrix **Uptr, jxf_ParCSRMatrix **Sptr,JXF_Int  **u_end,                      
                        JXF_Int ***L_levels, JXF_Int **L_level_sizes,JXF_Int *L_num_levels, 
                        JXF_Int ***U_levels, JXF_Int **U_level_sizes,JXF_Int *U_num_levels);

void jxf_CSRMatrixBuildLevelSchedule(jxf_CSRMatrix *A, JXF_Int ***levels,JXF_Int **level_sizes,
                                    JXF_Int *num_levels,  JXF_Int is_upper);
void jxf_CSRMatrixTopologicSortILU(jxf_CSRMatrix *L, jxf_CSRMatrix *U,
                                  JXF_Int *nlevL, JXF_Int **jlevL, JXF_Int **ilevL,
                                  JXF_Int *nlevU, JXF_Int **jlevU, JXF_Int **ilevU);

void jxf_CSRMatrixTopologicSortILU1(jxf_CSRMatrix *L, jxf_CSRMatrix *U,
                                 JXF_Int *nlevL, JXF_Int **jlevL, JXF_Int **ilevL, JXF_Int **permL, JXF_Int **inv_permL, 
                                 JXF_Int *nlevU, JXF_Int **jlevU, JXF_Int **ilevU, JXF_Int **permU, JXF_Int **inv_permU);

// JXF_Int jxf_ILUSetupILUKSymbolic( JXF_Int n, JXF_Int *A_diag_i, JXF_Int *A_diag_j,
//                                       JXF_Int lfil, JXF_Int *perm, JXF_Int *rperm,
//                                       JXF_Int *iw, JXF_Int nLU, JXF_Int *L_diag_i,
//                                       JXF_Int *U_diag_i, JXF_Int *S_diag_i,
//                                       JXF_Int **L_diag_j, JXF_Int **U_diag_j,
//                                       JXF_Int **S_diag_j, JXF_Int **u_end );
JXF_Int jxf_ILUSetupILUK( jxf_ParCSRMatrix *A, JXF_Int lfil, JXF_Int *permp,
                              JXF_Int *qpermp, JXF_Int nLU, JXF_Int nI,
                              jxf_ParCSRMatrix **Lptr, JXF_Real **Dptr,
                              jxf_ParCSRMatrix **Uptr, jxf_ParVector    **D_array,
                              JXF_Int **u_end );

JXF_Int jxf_ILUSetupILUK_S( jxf_ParCSRMatrix *A, JXF_Int lfil, JXF_Int *permp,
                              JXF_Int *qpermp, JXF_Int nLU, JXF_Int nI,
                              jxf_ParCSRMatrix **Lptr, JXF_Real **Dptr,
                              jxf_ParCSRMatrix **Uptr, jxf_ParCSRMatrix **Sptr,
                              JXF_Int **u_end,JXF_Int **L_levels, JXF_Int *L_level_sizes,JXF_Int L_num_levels, 
                              JXF_Int **U_levels, JXF_Int *U_level_sizes,JXF_Int U_num_levels);                                       
// JXF_Int jxf_ILUSetupILUT( jxf_ParCSRMatrix *A, JXF_Int lfil, JXF_Real *tol,
//                               JXF_Int *permp, JXF_Int *qpermp, JXF_Int nLU,
//                               JXF_Int nI, jxf_ParCSRMatrix **Lptr, JXF_Real **Dptr,
//                               jxf_ParCSRMatrix **Uptr, jxf_ParCSRMatrix **Sptr,
//                               JXF_Int **u_end );
// JXF_Int jxf_NSHSetup( void *nsh_vdata, jxf_ParCSRMatrix *A,
//                           jxf_ParVector *f, jxf_ParVector *u );
// JXF_Int jxf_ILUSetupILU0RAS( jxf_ParCSRMatrix *A, JXF_Int *perm,
//                                  JXF_Int nLU, jxf_ParCSRMatrix **Lptr,
//                                  JXF_Real **Dptr, jxf_ParCSRMatrix **Uptr );
// JXF_Int jxf_ILUSetupILUKRASSymbolic( JXF_Int n, JXF_Int *A_diag_i, JXF_Int *A_diag_j,
//                                          JXF_Int *A_offd_i, JXF_Int *A_offd_j,
//                                          JXF_Int *E_i, JXF_Int *E_j, JXF_Int ext,
//                                          JXF_Int lfil, JXF_Int *perm, JXF_Int *rperm,
//                                          JXF_Int *iw, JXF_Int nLU, JXF_Int *L_diag_i,
//                                          JXF_Int *U_diag_i, JXF_Int **L_diag_j,
//                                          JXF_Int **U_diag_j );
// JXF_Int jxf_ILUSetupILUKRAS( jxf_ParCSRMatrix *A, JXF_Int lfil, JXF_Int *perm,
//                                  JXF_Int nLU, jxf_ParCSRMatrix **Lptr,
//                                  JXF_Real **Dptr, jxf_ParCSRMatrix **Uptr );
// JXF_Int jxf_ILUSetupILUTRAS( jxf_ParCSRMatrix *A, JXF_Int lfil,
//                                  JXF_Real *tol, JXF_Int *perm, JXF_Int nLU,
//                                  jxf_ParCSRMatrix **Lptr, JXF_Real **Dptr,
//                                  jxf_ParCSRMatrix **Uptr );

JXF_Int jxf_ILUSolveLU( jxf_ParCSRMatrix *A, jxf_ParVector *f,
                            jxf_ParVector *u, JXF_Int *perm, JXF_Int nLU,
                            jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
                            jxf_ParVector *ftemp, jxf_ParVector *utemp ); 

JXF_Int jxf_TriangularSolveLevel_v1( jxf_ParCSRMatrix *A, jxf_ParVector *f,jxf_ParVector *u,
                               JXF_Int nLU,jxf_ParCSRMatrix *L, JXF_Real *D, 
                               jxf_ParCSRMatrix *U,jxf_ParVector *ftemp, jxf_ParVector *utemp,
                              JXF_Int **L_levels, JXF_Int *L_level_sizes,JXF_Int L_num_levels, 
                              JXF_Int **U_levels, JXF_Int *U_level_sizes,JXF_Int U_num_levels);

JXF_Int jxf_TriangularSolveLevel(jxf_ParCSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u, JXF_Int nLU, 
                              jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U, jxf_ParVector *ftemp, jxf_ParVector *utemp, JXF_Int nlevL, 
                              JXF_Int *jlevL, JXF_Int *ilevL, JXF_Int nlevU, JXF_Int *jlevU, JXF_Int *ilevU);
  
JXF_Int jxf_MultiColorTriangularSolve(jxf_ParCSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u, JXF_Int nLU, jxf_ParCSRMatrix *L, JXF_Real *D, 
                                    jxf_ParCSRMatrix *U, jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int nlevL, JXF_Int *jlevL, JXF_Int *ilevL, JXF_Int nlevU, 
                                    JXF_Int *jlevU, JXF_Int *ilevU, JXF_Int maxIter, JXF_Int lower_jacobi_iters,
                                    JXF_Int upper_jacobi_iters);
JXF_Int jxf_MultiColorTriangularSolve1(jxf_ParCSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u, JXF_Int nLU, jxf_ParCSRMatrix *L, JXF_Real *D, 
                                    jxf_ParCSRMatrix *U, jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int nlevL, JXF_Int *jlevL, JXF_Int *ilevL, JXF_Int nlevU, 
                                    JXF_Int *jlevU, JXF_Int *ilevU, JXF_Int maxIter, JXF_Int lower_jacobi_iters,
                                    JXF_Int upper_jacobi_iters,JXF_Int *L_perm, JXF_Int *L_iperm,JXF_Int *U_perm, JXF_Int *U_iperm);

JXF_Int jxf_MultiL_TriangularSolve(jxf_ParCSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u, JXF_Int nLU, jxf_ParCSRMatrix *L, JXF_Real *D, 
                                    jxf_ParCSRMatrix *U, jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int nlevL, JXF_Int *jlevL, JXF_Int *ilevL, JXF_Int nlevU, 
                                    JXF_Int *jlevU, JXF_Int *ilevU, JXF_Int maxIter);

JXF_Int jxf_MultiL_TriangularSolve1(jxf_ParCSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u, JXF_Int nLU, jxf_ParCSRMatrix *L, JXF_Real *D, 
                                    jxf_ParCSRMatrix *U, jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int nlevL, JXF_Int *jlevL, JXF_Int *ilevL, JXF_Int nlevU, 
                                    JXF_Int *jlevU, JXF_Int *ilevU, JXF_Int maxIter,JXF_Int *L_perm, JXF_Int *L_iperm,JXF_Int *U_perm, JXF_Int *U_iperm);
                                    
JXF_Int jxf_MultiGS_TriangularSolve(jxf_ParCSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u, JXF_Int nLU, jxf_ParCSRMatrix *L, JXF_Real *D, 
                                    jxf_ParCSRMatrix *U, jxf_ParVector *ftemp, jxf_ParVector *utemp, JXF_Int maxIter);
// JXF_Int jxf_JacobiTriangularSolve(jxf_ParCSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u,
//                                 JXF_Int nLU, jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
//                                 jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int maxIter);

JXF_Int jxf_JacobiTriangularSolve( jxf_ParCSRMatrix *A, jxf_ParVector *f,
                                jxf_ParVector *u, JXF_Int *perm, JXF_Int nLU,
                                jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
                                jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int maxIter,
                                JXF_Int lower_jacobi_iters, JXF_Int upper_jacobi_iters );

JXF_Int jxf_JacobiTriangularSolve22( jxf_ParCSRMatrix *A, jxf_ParVector *f,
                              jxf_ParVector *u, JXF_Int *perm, JXF_Int nLU,
                              jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
                              jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int maxIter,
                              JXF_Int lower_jacobi_iters, JXF_Int upper_jacobi_iters );

JXF_Int jxf_JacobiTriangularSolve33( jxf_ParCSRMatrix *A, jxf_ParVector *f,
                                jxf_ParVector *u, JXF_Int *perm, JXF_Int nLU,
                                jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
                                jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int maxIter,
                                JXF_Int lower_jacobi_iters, JXF_Int upper_jacobi_iters );

JXF_Int  jxf_JacobiTriangularSolve_spmv( jxf_ParCSRMatrix *A, jxf_ParVector *f,
                                jxf_ParVector *u, JXF_Int *perm, JXF_Int nLU,
                                jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
                                jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int maxIter,
                                JXF_Int lower_jacobi_iters, JXF_Int upper_jacobi_iters );

JXF_Int  jxf_JacobiTriangularSolve_spmv1( jxf_ParCSRMatrix *A, jxf_ParVector *f,
                                jxf_ParVector *u, JXF_Int *perm, JXF_Int nLU,
                                jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParVector *D_arry, jxf_ParCSRMatrix *U,
                                jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int maxIter,
                                JXF_Int lower_jacobi_iters, JXF_Int upper_jacobi_iters );



JXF_Int   jxf_parJacobi_TriangularSolve(jxf_ParCSRMatrix *A,jxf_ParVector *f,jxf_ParVector  *u,
                                       JXF_Int  nLU, jxf_ParCSRMatrix *L, jxf_ParVector *D_vector,
                                       jxf_ParCSRMatrix *U,jxf_ParVector *ftemp,jxf_ParVector *utemp,jxf_ParVector *unew,jxf_ParVector *ytemp,
                                       JXF_Int maxIter,JXF_Int lower_jacobi_iters,JXF_Int upper_jacobi_iters);

JXF_Int jxf_par_Block_Jacobi_TriangularSolve(jxf_ParBSRMatrix *A,
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
                     JXF_Int           upper_jacobi_iters);
                                
JXF_Int  jxf_JGS_TriangularSolve( jxf_ParCSRMatrix *A, jxf_ParVector *f,
                                jxf_ParVector *u, JXF_Int *perm, JXF_Int nLU,
                                jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
                                jxf_ParVector *ftemp, jxf_ParVector *utemp,JXF_Int maxIter,
                                JXF_Int lower_jacobi_iters, JXF_Int upper_jacobi_iters );

JXF_Int jxf_ILUSolve( void *ilu_vdata, jxf_ParCSRMatrix *A,
                          jxf_ParVector *f, jxf_ParVector *u );
JXF_Int JXF_ILUSolve( JXF_Solver solver,
                JXF_ParCSRMatrix A,
                JXF_ParVector b,
                JXF_ParVector x      );  
JXF_Int jxf_BILUSetup( void               *ilu_vdata,
                jxf_ParBSRMatrix *A,
                jxf_ParVector    *f,
                jxf_ParVector    *u );
JXF_Int JXF_BILUSetup( JXF_Solver      solver,
                jxf_ParBSRMatrix *A,
                JXF_ParVector     b,
                JXF_ParVector     x );
JXF_Int jxf_BILUSolve( void               *ilu_vdata,
                jxf_ParBSRMatrix *A,
                jxf_ParVector    *f,
                jxf_ParVector    *u );
JXF_Int JXF_BILUSolve( JXF_Solver      solver,
                jxf_ParBSRMatrix *A,
                JXF_ParVector     b,
                JXF_ParVector     x );
JXF_Int jxf_ILUSolveLUIter( jxf_ParCSRMatrix *A, jxf_ParVector *f,
                                jxf_ParVector *u, JXF_Int *perm, JXF_Int nLU,
                                jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
                                jxf_ParVector *ftemp, jxf_ParVector *utemp,
                                JXF_Int lower_jacobi_iters, JXF_Int upper_jacobi_iters );

JXF_Int jxf_ILUSolveLUIter_v2( jxf_ParCSRMatrix *A, jxf_ParVector *f,
                                jxf_ParVector *u, JXF_Int *perm, JXF_Int nLU,
                                jxf_ParCSRMatrix *L, JXF_Real *D, jxf_ParCSRMatrix *U,
                                jxf_ParVector *ftemp, jxf_ParVector *utemp,
                                JXF_Int lower_jacobi_iters, JXF_Int upper_jacobi_iters );