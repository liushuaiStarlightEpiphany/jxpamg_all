#ifndef JX_UTIL_HEADER 
#include "jx_util.h"
#endif

#ifndef JX_MV_HEADER 
#include "jx_mv.h"
#endif

#include "jx_krylov.h"

#define MAX(a,b) (((a)>(b))?(a):(b))  
#define JX_USING_CUDA 1
#define JX_USING_GPU 1
/*--------------------------------------------------------------------------
 * jx_ParILUData
 *--------------------------------------------------------------------------*/

// typedef struct
// {
//    JX_PtrToSolverFcn   setup;
//    JX_PtrToSolverFcn   solve;
//    JX_PtrToDestroyFcn  destroy;

// } jx_Solver;

/*--------------------------------------------------------------------------
 * Accessor functions for the jx_Solver structure
 *--------------------------------------------------------------------------*/

// #define jx_SolverSetup(data)       ((data) -> setup)
// #define jx_SolverSolve(data)       ((data) -> solve)
// #define jx_SolverDestroy(data)     ((data) -> destroy)

typedef struct jx_ParILUData_struct
{
   /* Base solver data structure */
   //jx_Solver          base;

   /* General data */
   JX_Int             global_solver;
   jx_ParCSRMatrix   *matA;
   jx_ParCSRMatrix   *matL;
   JX_Real           *matD;
   jx_ParVector      *D;
   jx_ParCSRMatrix   *matU;
   jx_ParCSRMatrix   *matmL;
   JX_Real           *matmD;
   jx_ParCSRMatrix   *matmU;
   jx_ParCSRMatrix   *matS;
   JX_Real           *droptol; /* Array of 3 elements, for B, (E and F), S respectively */
   JX_Int             lfil;
   JX_Int             maxRowNnz;
   JX_Int            *CF_marker_array;
   JX_Int            *perm;
   JX_Int            *qperm;
   JX_Real            tol_ddPQ;
   jx_ParVector      *F;
   jx_ParVector      *U;
   jx_ParVector      *residual;
   JX_Real           *rel_res_norms;
   JX_Int             num_iterations;
   JX_Real           *l1_norms;
   JX_Real            final_rel_residual_norm;
   JX_Real            tol;
   JX_Real            operator_complexity;
   JX_Int             logging;
   JX_Int             print_level;
   JX_Int             max_iter;
   JX_Int             tri_solve;
   JX_Int             lower_jacobi_iters;
   JX_Int             IR_iters;
   JX_Int             upper_jacobi_iters;
   JX_Int             ilu_type;
   JX_Int             nLU;
   JX_Int             nI;
   JX_Int            *u_end; /* used when schur block is formed */
   JX_Int             sweep;

   /* Iterative ILU parameters */
   JX_Int             iter_setup_type;
   JX_Int             iter_setup_option;
   JX_Int             setup_max_iter;
   JX_Int             setup_num_iter;
   JX_Real            setup_tolerance;
   JX_Real        *setup_history;

   /* temp vectors for solve phase */
   jx_ParVector      *Utemp;
   jx_ParVector      *Ftemp;
   jx_ParVector      *Xtemp;
   jx_ParVector      *Ytemp;
   jx_ParVector      *Ztemp;
   JX_Real           *uext;
   JX_Real           *fext;

   /* On GPU, we have to form E and F explicitly, since we don't have much control to it */
#if defined(JX_USING_GPU)
   jx_CSRMatrix      *matALU_d; /* Matrix holding ILU of A (for A-smoothing) */
   jx_CSRMatrix      *matBLU_d; /* Matrix holding ILU of B */
   jx_CSRMatrix      *matSLU_d; /* Matrix holding ILU of S */
   jx_CSRMatrix      *matE_d;
   jx_CSRMatrix      *matF_d;
   jx_ParCSRMatrix   *Aperm;
   jx_ParCSRMatrix   *R;
   jx_ParCSRMatrix   *P;
   jx_Vector         *Ftemp_upper;
   jx_Vector         *Utemp_lower;
   jx_Vector         *Adiag_diag;
   jx_Vector         *Sdiag_diag;
#endif

   /* data structure sor solving Schur System */
//    JX_Solver          schur_solver;
//    JX_Solver          schur_precond;
   jx_ParVector      *rhs;
   jx_ParVector      *x;

   /* local reordering */
   JX_Int             reordering_type;

   /* 层次调度 */
   JX_Int **L_levels, **U_levels;
   JX_Int *L_level_sizes, *U_level_sizes;
   JX_Int L_num_levels, U_num_levels;

   JX_Int nlevL;      // 下三角最大层级数
   JX_Int *ilevL;     // 下三角层级起始索引
   JX_Int *jlevL;     // 下三角按层级排序的行号
   JX_Int nlevU;      // 上三角最大层级数
   JX_Int *ilevU;     // 上三角层级起始索引
   JX_Int *jlevU;     // 上三角按层级排序的行号

   JX_Int *L_perm;
   JX_Int *L_iperm;
   JX_Int *U_perm;
   JX_Int *U_iperm;

} jx_ParILUData;

#define jx_ParILUDataTestOption(ilu_data)                   ((ilu_data) -> test_opt)

#if defined(JX_USING_GPU)
#define jx_ParILUDataMatAILUDevice(ilu_data)                ((ilu_data) -> matALU_d)
#define jx_ParILUDataMatBILUDevice(ilu_data)                ((ilu_data) -> matBLU_d)
#define jx_ParILUDataMatSILUDevice(ilu_data)                ((ilu_data) -> matSLU_d)
#define jx_ParILUDataMatEDevice(ilu_data)                   ((ilu_data) -> matE_d)
#define jx_ParILUDataMatFDevice(ilu_data)                   ((ilu_data) -> matF_d)
#define jx_ParILUDataAperm(ilu_data)                        ((ilu_data) -> Aperm)
#define jx_ParILUDataR(ilu_data)                            ((ilu_data) -> R)
#define jx_ParILUDataP(ilu_data)                            ((ilu_data) -> P)
#define jx_ParILUDataFTempUpper(ilu_data)                   ((ilu_data) -> Ftemp_upper)
#define jx_ParILUDataUTempLower(ilu_data)                   ((ilu_data) -> Utemp_lower)
#define jx_ParILUDataADiagDiag(ilu_data)                    ((ilu_data) -> Adiag_diag)
#define jx_ParILUDataSDiagDiag(ilu_data)                    ((ilu_data) -> Sdiag_diag)
#endif

#define jx_ParILUDatasweep(ilu_data)                         ((ilu_data) -> sweep)
#define jx_ParILUDataGlobalSolver(ilu_data)                 ((ilu_data) -> global_solver)
#define jx_ParILUDataMatA(ilu_data)                         ((ilu_data) -> matA)
#define jx_ParILUDataMatL(ilu_data)                         ((ilu_data) -> matL)
#define jx_ParILUDataMatD(ilu_data)                         ((ilu_data) -> matD)
#define jx_ParILUDataMatU(ilu_data)                         ((ilu_data) -> matU)
#define jx_ParILUDataMatLModified(ilu_data)                 ((ilu_data) -> matmL)
#define jx_ParILUDataMatDModified(ilu_data)                 ((ilu_data) -> matmD)
#define jx_ParILUDataMatUModified(ilu_data)                 ((ilu_data) -> matmU)
#define jx_ParILUDataMatS(ilu_data)                         ((ilu_data) -> matS)
#define jx_ParILUDataDroptol(ilu_data)                      ((ilu_data) -> droptol)
#define jx_ParILUDataLfil(ilu_data)                         ((ilu_data) -> lfil)
#define jx_ParILUDataMaxRowNnz(ilu_data)                    ((ilu_data) -> maxRowNnz)
#define jx_ParILUDataCFMarkerArray(ilu_data)                ((ilu_data) -> CF_marker_array)
#define jx_ParILUDataPerm(ilu_data)                         ((ilu_data) -> perm)
#define jx_ParILUDataQPerm(ilu_data)                        ((ilu_data) -> qperm)
#define jx_ParILUDataTolDDPQ(ilu_data)                      ((ilu_data) -> tol_ddPQ)
#define jx_ParILUDataD(ilu_data)                            ((ilu_data) -> D)
#define jx_ParILUDataF(ilu_data)                            ((ilu_data) -> F)
#define jx_ParILUDataU(ilu_data)                            ((ilu_data) -> U)
#define jx_ParILUDataResidual(ilu_data)                     ((ilu_data) -> residual)
#define jx_ParILUDataRelResNorms(ilu_data)                  ((ilu_data) -> rel_res_norms)
#define jx_ParILUDataNumIterations(ilu_data)                ((ilu_data) -> num_iterations)
#define jx_ParILUDataL1Norms(ilu_data)                      ((ilu_data) -> l1_norms)
#define jx_ParILUDataFinalRelResidualNorm(ilu_data)         ((ilu_data) -> final_rel_residual_norm)
#define jx_ParILUDataTol(ilu_data)                          ((ilu_data) -> tol)
#define jx_ParILUDataOperatorComplexity(ilu_data)           ((ilu_data) -> operator_complexity)
#define jx_ParILUDataLogging(ilu_data)                      ((ilu_data) -> logging)
#define jx_ParILUDataPrintLevel(ilu_data)                   ((ilu_data) -> print_level)
#define jx_ParILUDataMaxIter(ilu_data)                      ((ilu_data) -> max_iter)
#define jx_ParILUDataTriSolve(ilu_data)                     ((ilu_data) -> tri_solve)
#define jx_ParILUDataLowerJacobiIters(ilu_data)             ((ilu_data) -> lower_jacobi_iters)
#define jx_ParILUDataUpperJacobiIters(ilu_data)             ((ilu_data) -> upper_jacobi_iters)
#define jx_ParILUDataIRIters(ilu_data)                      ((ilu_data) -> IR_iters)
#define jx_ParILUDataIluType(ilu_data)                      ((ilu_data) -> ilu_type)
#define jx_ParILUDataNLU(ilu_data)                          ((ilu_data) -> nLU)
#define jx_ParILUDataNI(ilu_data)                           ((ilu_data) -> nI)
#define jx_ParILUDataUEnd(ilu_data)                         ((ilu_data) -> u_end)
#define jx_ParILUDataUTemp(ilu_data)                        ((ilu_data) -> Utemp)
#define jx_ParILUDataFTemp(ilu_data)                        ((ilu_data) -> Ftemp)
#define jx_ParILUDataXTemp(ilu_data)                        ((ilu_data) -> Xtemp)
#define jx_ParILUDataYTemp(ilu_data)                        ((ilu_data) -> Ytemp)
#define jx_ParILUDataZTemp(ilu_data)                        ((ilu_data) -> Ztemp)
#define jx_ParILUDataUExt(ilu_data)                         ((ilu_data) -> uext)
#define jx_ParILUDataFExt(ilu_data)                         ((ilu_data) -> fext)
// #define jx_ParILUDataSchurSolver(ilu_data)                  ((ilu_data) -> schur_solver)
// #define jx_ParILUDataSchurPrecond(ilu_data)                 ((ilu_data) -> schur_precond)
#define jx_ParILUDataRhs(ilu_data)                          ((ilu_data) -> rhs)
#define jx_ParILUDataX(ilu_data)                            ((ilu_data) -> x)
#define jx_ParILUDataReorderingType(ilu_data)               ((ilu_data) -> reordering_type)

/* Iterative ILU setup */
#define jx_ParILUDataIterativeSetupType(ilu_data)           ((ilu_data) -> iter_setup_type)
#define jx_ParILUDataIterativeSetupOption(ilu_data)         ((ilu_data) -> iter_setup_option)
#define jx_ParILUDataIterativeSetupMaxIter(ilu_data)        ((ilu_data) -> setup_max_iter)
#define jx_ParILUDataIterativeSetupNumIter(ilu_data)        ((ilu_data) -> setup_num_iter)
#define jx_ParILUDataIterativeSetupTolerance(ilu_data)      ((ilu_data) -> setup_tolerance)
#define jx_ParILUDataIterativeSetupHistory(ilu_data)        ((ilu_data) -> setup_history)
#define jx_ParILUDataIterSetupCorrectionNorm(ilu_data,i)    ((ilu_data) -> setup_history[i])
#define jx_ParILUDataIterSetupResidualNorm(ilu_data,i)      (((ilu_data) -> setup_history + \
                                                                 (ilu_data) -> setup_num_iter)[i])
/* 层次调度 */
#define jx_ParILUDataL_levels(ilu_data)                   ((ilu_data) -> L_levels)
#define jx_ParILUDataU_levels(ilu_data)                   ((ilu_data) -> U_levels)
#define jx_ParILUDataL_level_sizes(ilu_data)              ((ilu_data) -> L_level_sizes)
#define jx_ParILUDataU_level_sizes(ilu_data)              ((ilu_data) -> U_level_sizes)
#define jx_ParILUDataL_num_levels(ilu_data)               ((ilu_data) -> L_num_levels)
#define jx_ParILUDataU_num_levels(ilu_data)               ((ilu_data) -> U_num_levels)

#define jx_ParILUDataL_perm(ilu_data)                         ((ilu_data) -> L_perm)
#define jx_ParILUDataL_iperm(ilu_data)                        ((ilu_data) -> L_iperm)

#define jx_ParILUDataU_perm(ilu_data)                         ((ilu_data) -> U_perm)
#define jx_ParILUDataU_iperm(ilu_data)                        ((ilu_data) -> U_iperm)


#define FMRK   -1
#define CMRK    1
#define UMRK    0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define MAT_TOL     1e-14
#define EXPAND_FACT 1.3

/* par_ilu.c */
void *jx_ILUCreate ( void );
JX_Int JX_ILUCreate( JX_Solver *solver );
JX_Int jx_ILUDestroy ( void *ilu_vdata );
JX_Int JX_ILUDestroy( JX_Solver solver );
JX_Int jx_ILUSetLevelOfFill( void *ilu_vdata, JX_Int lfil );
JX_Int JX_ILUSetLevelOfFill( JX_Solver solver, JX_Int lfil );
JX_Int jx_ILUSetMaxNnzPerRow( void *ilu_vdata, JX_Int nzmax );
JX_Int JX_ILUSetMaxNnzPerRow( JX_Solver solver, JX_Int nzmax );

JX_Int jx_ILUSetDropThreshold( void *ilu_vdata, JX_Real threshold );
JX_Int JX_ILUSetDropThreshold( JX_Solver solver, JX_Real threshold );

JX_Int jx_ILUSetDropThresholdArray( void *ilu_vdata, JX_Real *threshold );
JX_Int JX_ILUSetDropThresholdArray( JX_Solver solver, JX_Real *threshold );

JX_Int jx_ILUSetType( void *ilu_vdata, JX_Int ilu_type );
JX_Int JX_ILUSetType( JX_Solver solver, JX_Int ilu_type );

JX_Int jx_ILUSetsweep( void *ilu_vdata, JX_Real sweep );
JX_Int JX_ILUSetsweep( JX_Solver solver, JX_Real sweep );

JX_Int jx_ILUSetMaxIter( void *ilu_vdata, JX_Int max_iter );
JX_Int JX_ILUSetMaxIter( JX_Solver solver, JX_Int max_iter );
JX_Int jx_ILUSetTol( void *ilu_vdata, JX_Real tol );
JX_Int JX_ILUSetTol( JX_Solver solver, JX_Real tol );
JX_Int jx_ILUSetIterativeSetupType( void *ilu_vdata, JX_Int iter_setup_type );
JX_Int JX_ILUSetIterativeSetupType( JX_Solver solver, JX_Int iter_setup_type );
// JX_Int jx_ILUSetIterativeSetupOption( void *ilu_vdata, JX_Int iter_setup_option );
JX_Int jx_ILUSetIterativeSetupMaxIter( void *ilu_vdata, JX_Int iter_setup_max_iter );
JX_Int JX_ILUSetIterativeSetupMaxIter( JX_Solver solver, JX_Int iter_setup_max_iter );

JX_Int jx_ILUSetIterativeSetupTolerance( void *ilu_vdata, JX_Real iter_setup_tolerance );
JX_Int JX_ILUSetIterativeSetupTolerance( JX_Solver solver, JX_Real iter_setup_tolerance );

//JX_Int jx_ILUGetIterativeSetupHistory( void *ilu_vdata,JX_Real **iter_setup_history );
JX_Int jx_ILUSetTriSolve( void *ilu_vdata, JX_Int tri_solve );
JX_Int JX_ILUSetTriSolve( JX_Solver solver, JX_Int tri_solve );
JX_Int jx_ILUSetLowerJacobiIters( void *ilu_vdata, JX_Int lower_jacobi_iters );
JX_Int JX_ILUSetLowerJacobiIters( JX_Solver solver, JX_Int lower_jacobi_iterations );
JX_Int jx_ILUSetUpperJacobiIters( void *ilu_vdata, JX_Int upper_jacobi_iters );
JX_Int JX_ILUSetUpperJacobiIters( JX_Solver solver, JX_Int upper_jacobi_iterations );
JX_Int jx_ILUSetPrintLevel( void *ilu_vdata, JX_Int print_level );
JX_Int JX_ILUSetPrintLevel( JX_Solver solver, JX_Int print_level );
JX_Int jx_ILUSetIRIters( void *ilu_vdata, JX_Int IR_iters );
JX_Int JX_ILUSetIRIters( JX_Solver solver, JX_Int IR_iters );

JX_Int jx_ILUSetLogging( void *ilu_vdata, JX_Int logging );
JX_Int JX_ILUSetLogging( JX_Solver solver, JX_Int logging );
JX_Int jx_ILUSetLocalReordering( void *ilu_vdata, JX_Int ordering_type );
JX_Int JX_ILUSetLocalReordering( JX_Solver solver, JX_Int reordering_type );
// JX_Int jx_ILUSetSchurSolverMaxIter( void *ilu_vdata, JX_Int ss_max_iter );
// JX_Int jx_ILUSetSchurSolverTol( void *ilu_vdata, JX_Real ss_tol );
// JX_Int jx_ILUSetSchurSolverAbsoluteTol( void *ilu_vdata, JX_Real ss_absolute_tol );
// JX_Int jx_ILUSetSchurSolverLogging( void *ilu_vdata, JX_Int ss_logging );
// JX_Int jx_ILUSetSchurSolverPrintLevel( void *ilu_vdata, JX_Int ss_print_level );
// JX_Int jx_ILUSetSchurSolverRelChange( void *ilu_vdata, JX_Int ss_rel_change );
// JX_Int jx_ILUSetSchurPrecondILUType( void *ilu_vdata, JX_Int sp_ilu_type );
// JX_Int jx_ILUSetSchurPrecondILULevelOfFill( void *ilu_vdata, JX_Int sp_ilu_lfil );
// JX_Int jx_ILUSetSchurPrecondILUMaxNnzPerRow( void *ilu_vdata,
//                                                    JX_Int sp_ilu_max_row_nnz );
// JX_Int jx_ILUSetSchurPrecondILUDropThreshold( void *ilu_vdata, JX_Real sp_ilu_droptol );
// JX_Int jx_ILUSetSchurPrecondILUDropThresholdArray( void *ilu_vdata,
//                                                          JX_Real *sp_ilu_droptol );
// JX_Int jx_ILUSetSchurPrecondPrintLevel( void *ilu_vdata, JX_Int sp_print_level );
// JX_Int jx_ILUSetSchurPrecondMaxIter( void *ilu_vdata, JX_Int sp_max_iter );
// JX_Int jx_ILUSetSchurPrecondTriSolve( void *ilu_vdata, JX_Int sp_tri_solve );
// JX_Int jx_ILUSetSchurPrecondLowerJacobiIters( void *ilu_vdata,
//                                                     JX_Int sp_lower_jacobi_iters );
// JX_Int jx_ILUSetSchurPrecondUpperJacobiIters( void *ilu_vdata,
//                                                     JX_Int sp_upper_jacobi_iters );
// JX_Int jx_ILUSetSchurPrecondTol( void *ilu_vdata, JX_Int sp_tol );
// JX_Int jx_ILUSetSchurNSHDropThreshold( void *ilu_vdata, JX_Real threshold );
// JX_Int jx_ILUSetSchurNSHDropThresholdArray( void *ilu_vdata, JX_Real *threshold );
JX_Int jx_ILUGetNumIterations( void *ilu_vdata, JX_Int *num_iterations );
JX_Int JX_ILUGetNumIterations( JX_Solver solver, JX_Int *num_iterations );
JX_Int jx_ILUGetFinalRelativeResidualNorm( void *ilu_vdata, JX_Real *res_norm );
JX_Int JX_ILUGetFinalRelativeResidualNorm(  JX_Solver solver, JX_Real *res_norm );




JX_Int jx_GreedyColoring(jx_ParCSRMatrix *A,JX_Int nLU,JX_Int **row_by_color_out,
                     JX_Int  **color_starts_out,JX_Int *num_colors_out);
JX_Int jx_GreedyColoring_L(jx_ParCSRMatrix *A,JX_Int nLU,JX_Int **row_by_color_out,
                     JX_Int  **color_starts_out,JX_Int *num_colors_out);
JX_Int jx_GreedyColoring_U(jx_ParCSRMatrix *A,JX_Int nLU,JX_Int **row_by_color_out,
                     JX_Int  **color_starts_out,JX_Int *num_colors_out);

JX_Int jx_GreedyColoring_L1(jx_ParCSRMatrix *A,JX_Int nLU,JX_Int **row_by_color_out,
                     JX_Int  **color_starts_out,JX_Int *num_colors_out,JX_Int **perm_out,
                  JX_Int **inv_perm_out);
JX_Int jx_GreedyColoring_U1(jx_ParCSRMatrix *A,JX_Int nLU,JX_Int **row_by_color_out,
                     JX_Int  **color_starts_out,JX_Int *num_colors_out,JX_Int **perm_out,
                  JX_Int **inv_perm_out);
JX_Int jx_GreedyColoring_8GS(jx_ParCSRMatrix *A,JX_Int nLU,JX_Int **row_by_color_out,
                     JX_Int  **color_starts_out,JX_Int *num_colors_out,
                     JX_Int **row_by_color_out1, JX_Int  **color_starts_out1,JX_Int *num_colors_out1);
//JX_Int jx_ILUWriteSolverParams( void *ilu_vdata );

// JX_Int jx_ILUMinHeapAddI( JX_Int *heap, JX_Int len );
// JX_Int jx_ILUMinHeapAddIIIi( JX_Int *heap, JX_Int *I1,
//                                    JX_Int *Ii1, JX_Int len );
// JX_Int jx_ILUMinHeapAddIRIi( JX_Int *heap, JX_Real *I1,
//                                    JX_Int *Ii1, JX_Int len );
// JX_Int jx_ILUMaxrHeapAddRabsI( JX_Real *heap, JX_Int *I1, JX_Int len );
// JX_Int jx_ILUMinHeapRemoveI( JX_Int *heap, JX_Int len );
// JX_Int jx_ILUMinHeapRemoveIIIi( JX_Int *heap, JX_Int *I1,
//                                       JX_Int *Ii1, JX_Int len );
// JX_Int jx_ILUMinHeapRemoveIRIi( JX_Int *heap, JX_Real *I1,
//                                       JX_Int *Ii1, JX_Int len );
// JX_Int jx_ILUMaxrHeapRemoveRabsI( JX_Real *heap, JX_Int *I1, JX_Int len );
// JX_Int jx_ILUMaxQSplitRabsI( JX_Real *arrayR, JX_Int *arrayI, JX_Int left,
//                                    JX_Int bound, JX_Int right );
// JX_Int jx_ILUMaxRabs( JX_Real *array_data, JX_Int *array_j, JX_Int start,
//                             JX_Int end, JX_Int nLU, JX_Int *rperm, JX_Real *value,
//                             JX_Int *index, JX_Real *l1_norm, JX_Int *nnz );

// JX_Int jx_ILUGetPermddPQPre( JX_Int n, JX_Int nLU, JX_Int *A_diag_i,
//                                    JX_Int *A_diag_j, JX_Real *A_diag_data,
//                                    JX_Real tol, JX_Int *perm, JX_Int *rperm,
//                                    JX_Int *pperm_pre, JX_Int *qperm_pre, JX_Int *nB );
// JX_Int jx_ILUGetPermddPQ( jx_ParCSRMatrix *A, JX_Int **io_pperm, JX_Int **io_qperm,
//                                 JX_Real tol, JX_Int *nB, JX_Int *nI,
//                                 JX_Int reordering_type );
// JX_Int jx_ILUGetInteriorExteriorPerm( jx_ParCSRMatrix *A,
//                                             JX_MemoryLocation memory_location,
//                                             JX_Int **perm, JX_Int *nLU,
//                                             JX_Int reordering_type );
// JX_Int jx_ILUGetLocalPerm( jx_ParCSRMatrix *A, JX_Int **perm_ptr,
//                                  JX_Int *nLU, JX_Int reordering_type );
// JX_Int jx_ILUBuildRASExternalMatrix( jx_ParCSRMatrix *A, JX_Int *rperm,
//                                            JX_Int **E_i, JX_Int **E_j, JX_Real **E_data );
// JX_Int jx_ILUSortOffdColmap( jx_ParCSRMatrix *A );
// JX_Int jx_ILULocalRCMBuildFinalPerm( JX_Int start, JX_Int end,
//                                            JX_Int * G_perm, JX_Int *perm, JX_Int *qperm,
//                                            JX_Int **permp, JX_Int **qpermp );
//JX_Int jx_ILULocalRCM( jx_CSRMatrix *A, JX_Int start, JX_Int end,
//                              JX_Int **permp, JX_Int **qpermp, JX_Int sym );
// JX_Int jx_ILULocalRCMMindegree( JX_Int n, JX_Int *degree,
//                                       JX_Int *marker, JX_Int *rootp );
// JX_Int jx_ILULocalRCMOrder( jx_CSRMatrix *A, JX_Int *perm );
// JX_Int jx_ILULocalRCMFindPPNode( jx_CSRMatrix *A, JX_Int *rootp, JX_Int *marker );
// JX_Int jx_ILULocalRCMBuildLevel( jx_CSRMatrix *A, JX_Int root, JX_Int *marker,
//                                        JX_Int *level_i, JX_Int *level_j, JX_Int *nlevp );
// JX_Int jx_ILULocalRCMNumbering( jx_CSRMatrix *A, JX_Int root, JX_Int *marker,
//                                       JX_Int *perm, JX_Int *current_nump );
// JX_Int jx_ILULocalRCMQsort( JX_Int *perm, JX_Int start, JX_Int end,
//                                   JX_Int *degree );
// JX_Int jx_ILULocalRCMReverse( JX_Int *perm, JX_Int start, JX_Int end );


/* par_ilu_setup.c */
JX_Int jx_ILUSetup(void *ilu_vdata, jx_ParCSRMatrix *A, jx_ParVector *f, jx_ParVector *u);
JX_Int JX_ILUSetup( JX_Solver solver,
                JX_ParCSRMatrix A,
                JX_ParVector b,
                JX_ParVector x);
/* par_ilu_setup_device.c */
JX_Int jx_ILUSetupILUDevice( JX_Int ilu_type, jx_ParCSRMatrix *A,
                                   JX_Int lfil, JX_Real *tol, JX_Int *perm_data,
                                   JX_Int *qperm_data, JX_Int n, JX_Int nLU,
                                   jx_CSRMatrix **BLUptr, jx_ParCSRMatrix **matSptr,
                                   jx_CSRMatrix **Eptr, jx_CSRMatrix **Fptr,
                                   JX_Int tri_solve );
// JX_Int jx_ParILUExtractEBFC( jx_CSRMatrix *A_diag, JX_Int nLU,
//                                    jx_CSRMatrix **Bp, jx_CSRMatrix **Cp,
//                                    jx_CSRMatrix **Ep, jx_CSRMatrix **Fp );
// JX_Int jx_ParILURAPReorder( jx_ParCSRMatrix *A, JX_Int *perm,
//                                   JX_Int *rqperm, jx_ParCSRMatrix **A_pq );
// JX_Int jx_ILUSetupLDUtoCusparse( jx_ParCSRMatrix *L, JX_Real *D,
//                                        jx_ParCSRMatrix  *U, jx_ParCSRMatrix **LDUp );
// JX_Int jx_ILUSetupRAPMILU0( jx_ParCSRMatrix *A, jx_ParCSRMatrix **ALUp,
//                                   JX_Int modified );
// JX_Int jx_ILUSetupRAPILU0Device( jx_ParCSRMatrix *A, JX_Int *perm, JX_Int n,
//                                        JX_Int nLU, jx_ParCSRMatrix **Apermptr,
//                                        jx_ParCSRMatrix **matSptr, jx_CSRMatrix **ALUptr,
//                                        jx_CSRMatrix **BLUptr, jx_CSRMatrix **CLUptr,
//                                        jx_CSRMatrix **Eptr, jx_CSRMatrix **Fptr,
//                                        JX_Int test_opt );
// JX_Int jx_ILUSetupRAPILU0( jx_ParCSRMatrix *A, JX_Int *perm, JX_Int n,
//                                  JX_Int nLU, jx_ParCSRMatrix **Lptr, JX_Real **Dptr,
//                                  jx_ParCSRMatrix **Uptr, jx_ParCSRMatrix **mLptr,
//                                  JX_Real **mDptr, jx_ParCSRMatrix **mUptr,
//                                  JX_Int **u_end );
JX_Int jx_ILUSetupILU0( jx_ParCSRMatrix  *A, JX_Int *perm, JX_Int *qperm,
                              JX_Int nLU, JX_Int nI, jx_ParCSRMatrix **Lptr,
                              JX_Real **Dptr, jx_ParCSRMatrix **Uptr,
                              jx_ParVector **D_array, JX_Int **u_end );
JX_Int jx_ILUSetupMILU0( jx_ParCSRMatrix *A, JX_Int *permp,
                               JX_Int *qpermp, JX_Int nLU, JX_Int nI,
                               jx_ParCSRMatrix **Lptr, JX_Real **Dptr,
                               jx_ParCSRMatrix **Uptr, jx_ParVector **D_array,
                               JX_Int **u_end, JX_Int modified );
JX_Int jx_ILUSetupFGPILU_v1(jx_ParCSRMatrix  *A, JX_Int  *permp,JX_Int  *qpermp,
                        JX_Int  sweep,jx_ParCSRMatrix **Lptr,JX_Real **Dptr,
                        jx_ParCSRMatrix **Uptr, jx_ParCSRMatrix **Sptr,JX_Int  **u_end);

JX_Int jx_ILUSetupFGPILU_v2(jx_ParCSRMatrix  *A, JX_Int  *permp,JX_Int  *qpermp,
                        JX_Int  sweep,jx_ParCSRMatrix **Lptr,JX_Real **Dptr,
                        jx_ParCSRMatrix **Uptr, jx_ParVector **D_array, JX_Int  **u_end);

JX_Int jx_ILUSetupFGPILU_v3(jx_ParCSRMatrix *A,JX_Int sweep,
                           jx_ParCSRMatrix **Lptr, jx_ParVector **D,
                           jx_ParCSRMatrix **Uptr, jx_ParCSRMatrix **Sptr,JX_Int **u_end);

JX_Int jx_ParCSRMatrixM(  jx_ParCSRMatrix *A,jx_ParVector    *x );

void jx_ParCSRMatrixMatmul_Apattern(jx_ParCSRMatrix *A, jx_ParCSRMatrix *L, jx_ParCSRMatrix *B, JX_Real *C_diag_data, JX_Real *C_offd_data);

JX_Int jx_ILUSetupFGPILU(jx_ParCSRMatrix  *A, JX_Int  *permp,JX_Int  *qpermp,
                        JX_Int  sweep,jx_ParCSRMatrix **Lptr,JX_Real **Dptr,
                        jx_ParCSRMatrix **Uptr, jx_ParCSRMatrix **Sptr,JX_Int  **u_end,                      
                        JX_Int ***L_levels, JX_Int **L_level_sizes,JX_Int *L_num_levels, 
                        JX_Int ***U_levels, JX_Int **U_level_sizes,JX_Int *U_num_levels);

void jx_CSRMatrixBuildLevelSchedule(jx_CSRMatrix *A, JX_Int ***levels,JX_Int **level_sizes,
                                    JX_Int *num_levels,  JX_Int is_upper);
void jx_CSRMatrixTopologicSortILU(jx_CSRMatrix *L, jx_CSRMatrix *U,
                                  JX_Int *nlevL, JX_Int **jlevL, JX_Int **ilevL,
                                  JX_Int *nlevU, JX_Int **jlevU, JX_Int **ilevU);

void jx_CSRMatrixTopologicSortILU1(jx_CSRMatrix *L, jx_CSRMatrix *U,
                                 JX_Int *nlevL, JX_Int **jlevL, JX_Int **ilevL, JX_Int **permL, JX_Int **inv_permL, 
                                 JX_Int *nlevU, JX_Int **jlevU, JX_Int **ilevU, JX_Int **permU, JX_Int **inv_permU);

// JX_Int jx_ILUSetupILUKSymbolic( JX_Int n, JX_Int *A_diag_i, JX_Int *A_diag_j,
//                                       JX_Int lfil, JX_Int *perm, JX_Int *rperm,
//                                       JX_Int *iw, JX_Int nLU, JX_Int *L_diag_i,
//                                       JX_Int *U_diag_i, JX_Int *S_diag_i,
//                                       JX_Int **L_diag_j, JX_Int **U_diag_j,
//                                       JX_Int **S_diag_j, JX_Int **u_end );
JX_Int jx_ILUSetupILUK( jx_ParCSRMatrix *A, JX_Int lfil, JX_Int *permp,
                              JX_Int *qpermp, JX_Int nLU, JX_Int nI,
                              jx_ParCSRMatrix **Lptr, JX_Real **Dptr,
                              jx_ParCSRMatrix **Uptr, jx_ParVector    **D_array,
                              JX_Int **u_end );

JX_Int jx_ILUSetupILUK_S( jx_ParCSRMatrix *A, JX_Int lfil, JX_Int *permp,
                              JX_Int *qpermp, JX_Int nLU, JX_Int nI,
                              jx_ParCSRMatrix **Lptr, JX_Real **Dptr,
                              jx_ParCSRMatrix **Uptr, jx_ParCSRMatrix **Sptr,
                              JX_Int **u_end,JX_Int **L_levels, JX_Int *L_level_sizes,JX_Int L_num_levels, 
                              JX_Int **U_levels, JX_Int *U_level_sizes,JX_Int U_num_levels);                                       
// JX_Int jx_ILUSetupILUT( jx_ParCSRMatrix *A, JX_Int lfil, JX_Real *tol,
//                               JX_Int *permp, JX_Int *qpermp, JX_Int nLU,
//                               JX_Int nI, jx_ParCSRMatrix **Lptr, JX_Real **Dptr,
//                               jx_ParCSRMatrix **Uptr, jx_ParCSRMatrix **Sptr,
//                               JX_Int **u_end );
// JX_Int jx_NSHSetup( void *nsh_vdata, jx_ParCSRMatrix *A,
//                           jx_ParVector *f, jx_ParVector *u );
// JX_Int jx_ILUSetupILU0RAS( jx_ParCSRMatrix *A, JX_Int *perm,
//                                  JX_Int nLU, jx_ParCSRMatrix **Lptr,
//                                  JX_Real **Dptr, jx_ParCSRMatrix **Uptr );
// JX_Int jx_ILUSetupILUKRASSymbolic( JX_Int n, JX_Int *A_diag_i, JX_Int *A_diag_j,
//                                          JX_Int *A_offd_i, JX_Int *A_offd_j,
//                                          JX_Int *E_i, JX_Int *E_j, JX_Int ext,
//                                          JX_Int lfil, JX_Int *perm, JX_Int *rperm,
//                                          JX_Int *iw, JX_Int nLU, JX_Int *L_diag_i,
//                                          JX_Int *U_diag_i, JX_Int **L_diag_j,
//                                          JX_Int **U_diag_j );
// JX_Int jx_ILUSetupILUKRAS( jx_ParCSRMatrix *A, JX_Int lfil, JX_Int *perm,
//                                  JX_Int nLU, jx_ParCSRMatrix **Lptr,
//                                  JX_Real **Dptr, jx_ParCSRMatrix **Uptr );
// JX_Int jx_ILUSetupILUTRAS( jx_ParCSRMatrix *A, JX_Int lfil,
//                                  JX_Real *tol, JX_Int *perm, JX_Int nLU,
//                                  jx_ParCSRMatrix **Lptr, JX_Real **Dptr,
//                                  jx_ParCSRMatrix **Uptr );

JX_Int jx_ILUSolveLU( jx_ParCSRMatrix *A, jx_ParVector *f,
                            jx_ParVector *u, JX_Int *perm, JX_Int nLU,
                            jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U,
                            jx_ParVector *ftemp, jx_ParVector *utemp ); 

JX_Int jx_TriangularSolveLevel_v1( jx_ParCSRMatrix *A, jx_ParVector *f,jx_ParVector *u,
                               JX_Int nLU,jx_ParCSRMatrix *L, JX_Real *D, 
                               jx_ParCSRMatrix *U,jx_ParVector *ftemp, jx_ParVector *utemp,
                              JX_Int **L_levels, JX_Int *L_level_sizes,JX_Int L_num_levels, 
                              JX_Int **U_levels, JX_Int *U_level_sizes,JX_Int U_num_levels);

JX_Int jx_TriangularSolveLevel(jx_ParCSRMatrix *A, jx_ParVector *f, jx_ParVector *u, JX_Int nLU, 
                              jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U, jx_ParVector *ftemp, jx_ParVector *utemp, JX_Int nlevL, 
                              JX_Int *jlevL, JX_Int *ilevL, JX_Int nlevU, JX_Int *jlevU, JX_Int *ilevU);
  
JX_Int jx_MultiColorTriangularSolve(jx_ParCSRMatrix *A, jx_ParVector *f, jx_ParVector *u, JX_Int nLU, jx_ParCSRMatrix *L, JX_Real *D, 
                                    jx_ParCSRMatrix *U, jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int nlevL, JX_Int *jlevL, JX_Int *ilevL, JX_Int nlevU, 
                                    JX_Int *jlevU, JX_Int *ilevU, JX_Int maxIter, JX_Int lower_jacobi_iters,
                                    JX_Int upper_jacobi_iters);
JX_Int jx_MultiColorTriangularSolve1(jx_ParCSRMatrix *A, jx_ParVector *f, jx_ParVector *u, JX_Int nLU, jx_ParCSRMatrix *L, JX_Real *D, 
                                    jx_ParCSRMatrix *U, jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int nlevL, JX_Int *jlevL, JX_Int *ilevL, JX_Int nlevU, 
                                    JX_Int *jlevU, JX_Int *ilevU, JX_Int maxIter, JX_Int lower_jacobi_iters,
                                    JX_Int upper_jacobi_iters,JX_Int *L_perm, JX_Int *L_iperm,JX_Int *U_perm, JX_Int *U_iperm);

JX_Int jx_MultiL_TriangularSolve(jx_ParCSRMatrix *A, jx_ParVector *f, jx_ParVector *u, JX_Int nLU, jx_ParCSRMatrix *L, JX_Real *D, 
                                    jx_ParCSRMatrix *U, jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int nlevL, JX_Int *jlevL, JX_Int *ilevL, JX_Int nlevU, 
                                    JX_Int *jlevU, JX_Int *ilevU, JX_Int maxIter);

JX_Int jx_MultiL_TriangularSolve1(jx_ParCSRMatrix *A, jx_ParVector *f, jx_ParVector *u, JX_Int nLU, jx_ParCSRMatrix *L, JX_Real *D, 
                                    jx_ParCSRMatrix *U, jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int nlevL, JX_Int *jlevL, JX_Int *ilevL, JX_Int nlevU, 
                                    JX_Int *jlevU, JX_Int *ilevU, JX_Int maxIter,JX_Int *L_perm, JX_Int *L_iperm,JX_Int *U_perm, JX_Int *U_iperm);
                                    
JX_Int jx_MultiGS_TriangularSolve(jx_ParCSRMatrix *A, jx_ParVector *f, jx_ParVector *u, JX_Int nLU, jx_ParCSRMatrix *L, JX_Real *D, 
                                    jx_ParCSRMatrix *U, jx_ParVector *ftemp, jx_ParVector *utemp, JX_Int maxIter);
// JX_Int jx_JacobiTriangularSolve(jx_ParCSRMatrix *A, jx_ParVector *f, jx_ParVector *u,
//                                 JX_Int nLU, jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U,
//                                 jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int maxIter);

JX_Int jx_JacobiTriangularSolve( jx_ParCSRMatrix *A, jx_ParVector *f,
                                jx_ParVector *u, JX_Int *perm, JX_Int nLU,
                                jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U,
                                jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int maxIter,
                                JX_Int lower_jacobi_iters, JX_Int upper_jacobi_iters );

JX_Int jx_JacobiTriangularSolve22( jx_ParCSRMatrix *A, jx_ParVector *f,
                              jx_ParVector *u, JX_Int *perm, JX_Int nLU,
                              jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U,
                              jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int maxIter,
                              JX_Int lower_jacobi_iters, JX_Int upper_jacobi_iters );

JX_Int jx_JacobiTriangularSolve33( jx_ParCSRMatrix *A, jx_ParVector *f,
                                jx_ParVector *u, JX_Int *perm, JX_Int nLU,
                                jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U,
                                jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int maxIter,
                                JX_Int lower_jacobi_iters, JX_Int upper_jacobi_iters );

JX_Int  jx_JacobiTriangularSolve_spmv( jx_ParCSRMatrix *A, jx_ParVector *f,
                                jx_ParVector *u, JX_Int *perm, JX_Int nLU,
                                jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U,
                                jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int maxIter,
                                JX_Int lower_jacobi_iters, JX_Int upper_jacobi_iters );

JX_Int  jx_JacobiTriangularSolve_spmv1( jx_ParCSRMatrix *A, jx_ParVector *f,
                                jx_ParVector *u, JX_Int *perm, JX_Int nLU,
                                jx_ParCSRMatrix *L, JX_Real *D, jx_ParVector *D_arry, jx_ParCSRMatrix *U,
                                jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int maxIter,
                                JX_Int lower_jacobi_iters, JX_Int upper_jacobi_iters );



JX_Int   jx_parJacobi_TriangularSolve(jx_ParCSRMatrix *A,jx_ParVector *f,jx_ParVector  *u,
                                       JX_Int  nLU, jx_ParCSRMatrix *L, jx_ParVector *D_vector,
                                       jx_ParCSRMatrix *U,jx_ParVector *ftemp,jx_ParVector *utemp,jx_ParVector *unew,jx_ParVector *ytemp,
                                       JX_Int maxIter,JX_Int lower_jacobi_iters,JX_Int upper_jacobi_iters);
                                
JX_Int  jx_JGS_TriangularSolve( jx_ParCSRMatrix *A, jx_ParVector *f,
                                jx_ParVector *u, JX_Int *perm, JX_Int nLU,
                                jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U,
                                jx_ParVector *ftemp, jx_ParVector *utemp,JX_Int maxIter,
                                JX_Int lower_jacobi_iters, JX_Int upper_jacobi_iters );

JX_Int jx_ILUSolve( void *ilu_vdata, jx_ParCSRMatrix *A,
                          jx_ParVector *f, jx_ParVector *u );
JX_Int JX_ILUSolve( JX_Solver solver,
                JX_ParCSRMatrix A,
                JX_ParVector b,
                JX_ParVector x      );  
JX_Int jx_ILUSolveLUIter( jx_ParCSRMatrix *A, jx_ParVector *f,
                                jx_ParVector *u, JX_Int *perm, JX_Int nLU,
                                jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U,
                                jx_ParVector *ftemp, jx_ParVector *utemp,
                                JX_Int lower_jacobi_iters, JX_Int upper_jacobi_iters );

JX_Int jx_ILUSolveLUIter_v2( jx_ParCSRMatrix *A, jx_ParVector *f,
                                jx_ParVector *u, JX_Int *perm, JX_Int nLU,
                                jx_ParCSRMatrix *L, JX_Real *D, jx_ParCSRMatrix *U,
                                jx_ParVector *ftemp, jx_ParVector *utemp,
                                JX_Int lower_jacobi_iters, JX_Int upper_jacobi_iters );