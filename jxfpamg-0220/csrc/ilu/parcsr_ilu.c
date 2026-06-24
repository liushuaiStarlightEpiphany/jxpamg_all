#include "jxf_parilu.h"
#include "jxf_krylov.h"


/*--------------------------------------------------------------------------
 * JXF_ILUCreate
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUCreate( JXF_Solver *solver )
{
   if (!solver)
   {
      jxf_error_in_arg(2);
      return jxf_error_flag;
   }
   *solver = ( (JXF_Solver) jxf_ILUCreate( ) );
   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * JXF_ILUDestroy
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUDestroy( JXF_Solver solver )
{
   return ( jxf_ILUDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetup
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetup( JXF_Solver solver,
                JXF_ParCSRMatrix A,
                JXF_ParVector b,
                JXF_ParVector x      )
{
   return ( jxf_ILUSetup( (void *) solver,
                            (jxf_ParCSRMatrix *) A,
                            (jxf_ParVector *) b,
                            (jxf_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSolve
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSolve( JXF_Solver solver,
                JXF_ParCSRMatrix A,
                JXF_ParVector b,
                JXF_ParVector x      )
{
   return ( jxf_ILUSolve( (void *) solver,
                            (jxf_ParCSRMatrix *) A,
                            (jxf_ParVector *) b,
                            (jxf_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetPrintLevel
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetPrintLevel( JXF_Solver solver, JXF_Int print_level )
{
   return jxf_ILUSetPrintLevel( solver, print_level );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetLogging
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetLogging( JXF_Solver solver, JXF_Int logging )
{
   return jxf_ILUSetLogging(solver, logging );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetMaxIter
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetMaxIter( JXF_Solver solver, JXF_Int max_iter )
{
   return jxf_ILUSetMaxIter( solver, max_iter );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetIterativeSetupType
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetIterativeSetupType( JXF_Solver solver, JXF_Int iter_setup_type )
{
   return jxf_ILUSetIterativeSetupType( solver, iter_setup_type );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetIterativeSetupMaxIter
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetIterativeSetupMaxIter( JXF_Solver solver, JXF_Int iter_setup_max_iter )
{
   return jxf_ILUSetIterativeSetupMaxIter( solver, iter_setup_max_iter );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetIterativeSetupTolerance
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetIterativeSetupTolerance( JXF_Solver solver, JXF_Real iter_setup_tolerance )
{
   return jxf_ILUSetIterativeSetupTolerance( solver, iter_setup_tolerance );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetTriSolve
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetTriSolve( JXF_Solver solver, JXF_Int tri_solve )
{
   return jxf_ILUSetTriSolve( solver, tri_solve );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetLowerJacobiIters
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetLowerJacobiIters( JXF_Solver solver, JXF_Int lower_jacobi_iters )
{
   return jxf_ILUSetLowerJacobiIters( solver, lower_jacobi_iters );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetUpperJacobiIters
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetUpperJacobiIters( JXF_Solver solver, JXF_Int upper_jacobi_iters )
{
   return jxf_ILUSetUpperJacobiIters( solver, upper_jacobi_iters );
}

JXF_Int
JXF_ILUSetIRIters( JXF_Solver solver, JXF_Int IR_iters )
{
   return jxf_ILUSetIRIters( solver, IR_iters );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetTol
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetTol( JXF_Solver solver, JXF_Real tol )
{
   return jxf_ILUSetTol( solver, tol );
}

JXF_Int
JXF_ILUSetsweep( JXF_Solver solver, JXF_Real sweep)
{
   return jxf_ILUSetsweep( solver, sweep );
}
/*--------------------------------------------------------------------------
 * JXF_ILUSetDropThreshold
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetDropThreshold( JXF_Solver solver, JXF_Real threshold )
{
   return jxf_ILUSetDropThreshold( solver, threshold );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetDropThresholdArray
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetDropThresholdArray( JXF_Solver solver, JXF_Real *threshold )
{
   return jxf_ILUSetDropThresholdArray( solver, threshold );
}


/*--------------------------------------------------------------------------
 * JXF_ILUSetMaxNnzPerRow
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetMaxNnzPerRow( JXF_Solver solver, JXF_Int nzmax )
{
   return jxf_ILUSetMaxNnzPerRow( solver, nzmax );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetLevelOfFill
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetLevelOfFill( JXF_Solver solver, JXF_Int lfil )
{
   return jxf_ILUSetLevelOfFill( solver, lfil );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetType
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetType( JXF_Solver solver, JXF_Int ilu_type )
{
   return jxf_ILUSetType( solver, ilu_type );
}

/*--------------------------------------------------------------------------
 * JXF_ILUSetLocalReordering
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUSetLocalReordering(  JXF_Solver solver, JXF_Int ordering_type )
{
   return jxf_ILUSetLocalReordering(solver, ordering_type);
}

/*--------------------------------------------------------------------------
 * JXF_ILUGetNumIterations
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUGetNumIterations( JXF_Solver solver, JXF_Int *num_iterations )
{
   return jxf_ILUGetNumIterations( solver, num_iterations );
}

/*--------------------------------------------------------------------------
 * JXF_ILUGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

JXF_Int
JXF_ILUGetFinalRelativeResidualNorm(  JXF_Solver solver, JXF_Real *res_norm )
{
   return jxf_ILUGetFinalRelativeResidualNorm(solver, res_norm);
}

void *
jxf_ILUCreate( void )
{
   jxf_ParILUData  *ilu_data;

   ilu_data = jxf_CTAlloc(jxf_ParILUData, 1);

   /* general data */
   jxf_ParILUDataGlobalSolver(ilu_data)                 = 0;
   jxf_ParILUDataMatA(ilu_data)                         = NULL;
   jxf_ParILUDataMatL(ilu_data)                         = NULL;
   jxf_ParILUDataMatD(ilu_data)                         = NULL;
   jxf_ParILUDataMatU(ilu_data)                         = NULL;
   jxf_ParILUDataMatS(ilu_data)                         = NULL;
   // jxf_ParILUDataSchurSolver(ilu_data)                  = NULL;
   // jxf_ParILUDataSchurPrecond(ilu_data)                 = NULL;
   jxf_ParILUDataRhs(ilu_data)                          = NULL;
   jxf_ParILUDataX(ilu_data)                            = NULL;

   /* TODO (VPM): Transform this into a stack array */
   jxf_ParILUDataDroptol(ilu_data) = jxf_TAlloc(JXF_Real, 3);
   jxf_ParILUDataDroptol(ilu_data)[0]                   = 1.0e-02; /* droptol for B */
   jxf_ParILUDataDroptol(ilu_data)[1]                   = 1.0e-02; /* droptol for E and F */
   jxf_ParILUDataDroptol(ilu_data)[2]                   = 1.0e-02; /* droptol for S */
   jxf_ParILUDataLfil(ilu_data)                         = 0;
   jxf_ParILUDataMaxRowNnz(ilu_data)                    = 1000;
   jxf_ParILUDataCFMarkerArray(ilu_data)                = NULL;
   jxf_ParILUDataPerm(ilu_data)                         = NULL;
   jxf_ParILUDataQPerm(ilu_data)                        = NULL;
   jxf_ParILUDataTolDDPQ(ilu_data)                      = 1.0e-01;
   jxf_ParILUDataF(ilu_data)                            = NULL;
   jxf_ParILUDataU(ilu_data)                            = NULL;
   jxf_ParILUDataFTemp(ilu_data)                        = NULL;
   jxf_ParILUDataUTemp(ilu_data)                        = NULL;
   jxf_ParILUDataXTemp(ilu_data)                        = NULL;
   jxf_ParILUDataYTemp(ilu_data)                        = NULL;
   jxf_ParILUDataZTemp(ilu_data)                        = NULL;
   jxf_ParILUDataUExt(ilu_data)                         = NULL;
   jxf_ParILUDataFExt(ilu_data)                         = NULL;
   jxf_ParILUDataResidual(ilu_data)                     = NULL;
   jxf_ParILUDataRelResNorms(ilu_data)                  = NULL;
   jxf_ParILUDataNumIterations(ilu_data)                = 0;
   jxf_ParILUDataMaxIter(ilu_data)                      = 20;
   jxf_ParILUDataTriSolve(ilu_data)                     = 1;
   jxf_ParILUDataLowerJacobiIters(ilu_data)             = 5;
   jxf_ParILUDataUpperJacobiIters(ilu_data)             = 5;
   jxf_ParILUDataIRIters(ilu_data)                      = 5;
   jxf_ParILUDataTol(ilu_data)                          = 1.0e-7;
   jxf_ParILUDataLogging(ilu_data)                      = 0;
   jxf_ParILUDataPrintLevel(ilu_data)                   = 0;
   jxf_ParILUDataL1Norms(ilu_data)                      = NULL;
   jxf_ParILUDataOperatorComplexity(ilu_data)           = 0.;
   jxf_ParILUDataIluType(ilu_data)                      = 0;
   jxf_ParILUDataNLU(ilu_data)                          = 0;
   jxf_ParILUDataNI(ilu_data)                           = 0;
   jxf_ParILUDatasweep(ilu_data)                        = 5;
   jxf_ParILUDataUEnd(ilu_data)                         = NULL;
   jxf_ParILUDataL_perm(ilu_data)                         = NULL;
   jxf_ParILUDataL_iperm(ilu_data)                        = NULL;

   jxf_ParILUDataU_perm(ilu_data)                         = NULL;
   jxf_ParILUDataU_iperm(ilu_data)                        = NULL;


   /* Iterative setup variables */
   jxf_ParILUDataIterativeSetupType(ilu_data)           = 0;
   //jxf_ParILUDataIterativeSetupOption(ilu_data)         = 0;
   jxf_ParILUDataIterativeSetupMaxIter(ilu_data)        = 100;
   jxf_ParILUDataIterativeSetupNumIter(ilu_data)        = 0;
   jxf_ParILUDataIterativeSetupTolerance(ilu_data)      = 1.e-6;
   jxf_ParILUDataIterativeSetupHistory(ilu_data)        = NULL;

   /* reordering_type default to use local RCM */
   jxf_ParILUDataReorderingType(ilu_data)               = 0;

   jxf_ParILUDataL_levels(ilu_data)                     = NULL;
   jxf_ParILUDataL_level_sizes(ilu_data)                = NULL;

   jxf_ParILUDataU_levels(ilu_data)                     = NULL;
   jxf_ParILUDataU_level_sizes(ilu_data)                = NULL;

   return ilu_data;
}

JXF_Int
jxf_ILUDestroy(void *data)
{
   jxf_ParILUData *ilu_data = (jxf_ParILUData *)data;

   if (ilu_data)
   {
      /* General data */
      if (jxf_ParILUDataMatA(ilu_data))
      {
         jxf_ParCSRMatrixDestroy(jxf_ParILUDataMatA(ilu_data));
         jxf_ParILUDataMatA(ilu_data) = NULL;
      }
      if (jxf_ParILUDataMatL(ilu_data))
      {
         jxf_ParCSRMatrixDestroy(jxf_ParILUDataMatL(ilu_data));
         jxf_ParILUDataMatL(ilu_data) = NULL;
      }
      if (jxf_ParILUDataMatU(ilu_data))
      {
         jxf_ParCSRMatrixDestroy(jxf_ParILUDataMatU(ilu_data));
         jxf_ParILUDataMatU(ilu_data) = NULL;
      }
      if (jxf_ParILUDataMatD(ilu_data))
      {
         jxf_TFree(jxf_ParILUDataMatD(ilu_data));
         jxf_ParILUDataMatD(ilu_data) = NULL;
      }
      if (jxf_ParILUDataPerm(ilu_data))
      {
         jxf_TFree(jxf_ParILUDataPerm(ilu_data));
         jxf_ParILUDataPerm(ilu_data) = NULL;
      }
      if (jxf_ParILUDataQPerm(ilu_data))
      {
         jxf_TFree(jxf_ParILUDataQPerm(ilu_data));
         jxf_ParILUDataQPerm(ilu_data) = NULL;
      }
      // if ( jxf_ParILUDataL_perm(ilu_data))
      // {
      //    jxf_TFree(jxf_ParILUDataL_perm(ilu_data));
      //    jxf_ParILUDataL_perm(ilu_data) = NULL;
      // }
      // if ( jxf_ParILUDataU_perm(ilu_data))
      // {
      //    jxf_TFree(jxf_ParILUDataU_perm(ilu_data));
      //    jxf_ParILUDataU_perm(ilu_data) = NULL;
      // }
      // if ( jxf_ParILUDataL_iperm(ilu_data))
      // {
      //    jxf_TFree(jxf_ParILUDataL_iperm(ilu_data));
      //    jxf_ParILUDataL_iperm(ilu_data)(ilu_data) = NULL;
      // }
      // if ( jxf_ParILUDataL_iperm(ilu_data))
      // {
      //    jxf_TFree(jxf_ParILUDataU_iperm(ilu_data));
      //    jxf_ParILUDataU_iperm(ilu_data)(ilu_data) = NULL;
      // }
      /* Free the main ILU data structure */
      // jxf_TFree(ilu_data);
   }

   return 0; /* Return success */
}



/*--------------------------------------------------------------------------
 * jxf_ILUSetLevelOfFill
 *
 * Set fill level for ILUK
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetLevelOfFill( void      *ilu_vdata,
                         JXF_Int  lfil )
{
   jxf_ParILUData *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataLfil(ilu_data) = lfil;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetMaxNnzPerRow
 *
 * Set max non-zeros per row in factors for ILUT
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetMaxNnzPerRow( void      *ilu_vdata,
                          JXF_Int  nzmax )
{
   jxf_ParILUData *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataMaxRowNnz(ilu_data) = nzmax;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetDropThreshold
 *
 * Set threshold for dropping in LU factors for ILUT
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetDropThreshold( void       *ilu_vdata,
                           JXF_Real  threshold )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   if (!(jxf_ParILUDataDroptol(ilu_data)))
   {
      jxf_ParILUDataDroptol(ilu_data) = jxf_TAlloc(JXF_Real, 3);
   }
   jxf_ParILUDataDroptol(ilu_data)[0] = threshold;
   jxf_ParILUDataDroptol(ilu_data)[1] = threshold;
   jxf_ParILUDataDroptol(ilu_data)[2] = threshold;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetDropThresholdArray
 *
 * Set array of threshold for dropping in LU factors for ILUT
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetDropThresholdArray( void       *ilu_vdata,
                                JXF_Real *threshold )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   if (!(jxf_ParILUDataDroptol(ilu_data)))
   {
      jxf_ParILUDataDroptol(ilu_data) = jxf_TAlloc(JXF_Real, 3);
   }

   jxf_ParILUDataDroptol(ilu_data)[0] = threshold[0];
   jxf_ParILUDataDroptol(ilu_data)[1] = threshold[1];
   jxf_ParILUDataDroptol(ilu_data)[2] = threshold[2];

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetType
 *
 * Set ILU factorization type
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetType( void      *ilu_vdata,
                  JXF_Int  ilu_type )
{
   jxf_ParILUData *ilu_data = (jxf_ParILUData*) ilu_vdata;
   jxf_ParILUDataIluType(ilu_data) = ilu_type;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetMaxIter
 *
 * Set max number of iterations for ILU solver
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetMaxIter( void      *ilu_vdata,
                     JXF_Int  max_iter )
{
   jxf_ParILUData *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataMaxIter(ilu_data) = max_iter;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetIterativeSetupType
 *
 * Set iterative ILU setup algorithm
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetIterativeSetupType( void      *ilu_vdata,
                                JXF_Int  iter_setup_type)
{
   jxf_ParILUData *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataIterativeSetupType(ilu_data) = iter_setup_type;

   return jxf_error_flag;
}
/*--------------------------------------------------------------------------
 * jxf_ILUSetIterativeSetupMaxIter
 *
 * Set maximum number of iterations for iterative ILU setup
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetIterativeSetupMaxIter( void      *ilu_vdata,
                                   JXF_Int  iter_setup_max_iter)
{
   jxf_ParILUData *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataIterativeSetupMaxIter(ilu_data) = iter_setup_max_iter;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetIterativeSetupTolerance
 *
 * Set dropping tolerance for iterative ILU setup
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetIterativeSetupTolerance( void       *ilu_vdata,
                                     JXF_Real  iter_setup_tolerance)
{
   jxf_ParILUData *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataIterativeSetupTolerance(ilu_data) = iter_setup_tolerance;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUGetIterativeSetupHistory
 *
 * Get array of corrections and/or residual norms computed during ILU's
 * iterative setup algorithm.
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUGetIterativeSetupHistory( void           *ilu_vdata,
                                   JXF_Real **iter_setup_history)
{
   jxf_ParILUData *ilu_data = (jxf_ParILUData*) ilu_vdata;

   *iter_setup_history = jxf_ParILUDataIterativeSetupHistory(ilu_data);

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetTriSolve
 *
 * Set ILU triangular solver type
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetTriSolve( void      *ilu_vdata,
                      JXF_Int  tri_solve )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataTriSolve(ilu_data) = tri_solve;

   // printf("ilu_tri_solver2: %d\n",tri_solve);

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetLowerJacobiIters
 *
 * Set Lower Jacobi iterations for iterative triangular solver
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetLowerJacobiIters( void     *ilu_vdata,
                              JXF_Int lower_jacobi_iters )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataLowerJacobiIters(ilu_data) = lower_jacobi_iters;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetUpperJacobiIters
 *
 * Set Upper Jacobi iterations for iterative triangular solver
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetUpperJacobiIters( void      *ilu_vdata,
                              JXF_Int  upper_jacobi_iters )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataUpperJacobiIters(ilu_data) = upper_jacobi_iters;

   return jxf_error_flag;
}

JXF_Int
jxf_ILUSetIRIters( void      *ilu_vdata,
                              JXF_Int  IR_iters )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataIRIters(ilu_data) = IR_iters;

   return jxf_error_flag;
}
/*--------------------------------------------------------------------------
 * jxf_ILUSetTol
 *
 * Set convergence tolerance for ILU solver
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetTol( void       *ilu_vdata,
                 JXF_Real  tol )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataTol(ilu_data) = tol;

   return jxf_error_flag;
}

JXF_Int
jxf_ILUSetsweep( void       *ilu_vdata,
                 JXF_Real  sweep )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDatasweep(ilu_data) = sweep;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetPrintLevel
 *
 * Set print level for ILU solver
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetPrintLevel( void      *ilu_vdata,
                        JXF_Int  print_level )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataPrintLevel(ilu_data) = print_level;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetLogging
 *
 * Set print level for ilu solver
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetLogging( void      *ilu_vdata,
                     JXF_Int  logging )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataLogging(ilu_data) = logging;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetLocalReordering
 *
 * Set type of reordering for local matrix
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUSetLocalReordering( void      *ilu_vdata,
                             JXF_Int  ordering_type )
{
   jxf_ParILUData   *ilu_data = (jxf_ParILUData*) ilu_vdata;

   jxf_ParILUDataReorderingType(ilu_data) = ordering_type;

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUGetNumIterations
 *
 * Get number of iterations for ILU solver
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUGetNumIterations( void      *ilu_vdata,
                           JXF_Int *num_iterations )
{
   jxf_ParILUData  *ilu_data = (jxf_ParILUData*) ilu_vdata;

   if (!ilu_data)
   {
      jxf_error_in_arg(1);
      return jxf_error_flag;
   }

   *num_iterations = jxf_ParILUDataNumIterations(ilu_data);

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUGetFinalRelativeResidualNorm
 *
 * Get residual norms for ILU solver
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_ILUGetFinalRelativeResidualNorm( void       *ilu_vdata,
                                       JXF_Real *res_norm )
{
   jxf_ParILUData  *ilu_data = (jxf_ParILUData*) ilu_vdata;

   if (!ilu_data)
   {
      jxf_error_in_arg(1);
      return jxf_error_flag;
   }

   *res_norm = jxf_ParILUDataFinalRelResidualNorm(ilu_data);

   return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUWriteSolverParams
 *
 * Print solver params
 *
 * TODO (VPM): check runtime switch to decide whether running on host or device
 *--------------------------------------------------------------------------*/

// JXF_Int
// jxf_ILUWriteSolverParams(void *ilu_vdata)
// {
//    jxf_ParILUData  *ilu_data = (jxf_ParILUData*) ilu_vdata;

//    jxf_printf("ILU Setup parameters: \n");
//    jxf_printf("ILU factorization type: %d : ", jxf_ParILUDataIluType(ilu_data));
//    switch (jxf_ParILUDataIluType(ilu_data))
//    {
//       case 0:
// // #if defined(JXF_USING_GPU)
// //          if ( jxf_ParILUDataLfil(ilu_data) == 0 )
// //          {
// //             jxf_printf("Block Jacobi with GPU-accelerated ILU0 \n");
// //             jxf_printf("Operator Complexity (Fill factor) = %f \n",
// //                          jxf_ParILUDataOperatorComplexity(ilu_data));
// //          }
// //          else
// // #endif
//          jxf_printf("Block Jacobi with ILU(%d) \n", jxf_ParILUDataLfil(ilu_data));
//          jxf_printf("Operator Complexity (Fill factor) = %f \n",
//                         jxf_ParILUDataOperatorComplexity(ilu_data));
//          break;

//       // case 1:
//       //    jxf_printf("Block Jacobi with ILUT \n");
//       //    jxf_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
//       //                 jxf_ParILUDataDroptol(ilu_data)[0],
//       //                 jxf_ParILUDataDroptol(ilu_data)[1],
//       //                 jxf_ParILUDataDroptol(ilu_data)[2]);
//       //    jxf_printf("Max nnz per row = %d \n", jxf_ParILUDataMaxRowNnz(ilu_data));
//       //    jxf_printf("Operator Complexity (Fill factor) = %f \n",
//       //                 jxf_ParILUDataOperatorComplexity(ilu_data));
//       //    break;



//       default:
//          jxf_printf("Unknown type \n");
//          break;
//    }

//    jxf_printf("\n ILU Solver Parameters: \n");
//    jxf_printf("Max number of iterations: %d\n", jxf_ParILUDataMaxIter(ilu_data));
//    if (jxf_ParILUDataTriSolve(ilu_data))
//    {
//       jxf_printf("  Triangular solver type: exact (1)\n");
//    }
//    else
//    {
//       jxf_printf("  Triangular solver type: iterative (0)\n");
//       jxf_printf(" Lower Jacobi Iterations: %d\n", jxf_ParILUDataLowerJacobiIters(ilu_data));
//       jxf_printf(" Upper Jacobi Iterations: %d\n", jxf_ParILUDataUpperJacobiIters(ilu_data));
//    }
//    jxf_printf("      Stopping tolerance: %e\n", jxf_ParILUDataTol(ilu_data));

//    return jxf_error_flag;
// }

// JXF_Int
// jxf_ILUGetPermddPQPre(JXF_Int   n,
//                         JXF_Int   nLU,
//                         JXF_Int  *A_diag_i,
//                         JXF_Int  *A_diag_j,
//                         JXF_Real *A_diag_data,
//                         JXF_Real  tol,
//                         JXF_Int  *perm,
//                         JXF_Int  *rperm,
//                         JXF_Int  *pperm_pre,
//                         JXF_Int  *qperm_pre,
//                         JXF_Int  *nB)
// {
//    JXF_UNUSED_VAR(n);

//    JXF_Int   i, ii, nB_pre, k1, k2;
//    JXF_Real  gtol, max_value, norm;

//    JXF_Int   *jcol, *jnnz;
//    JXF_Real  *weight;

//    weight = jxf_TAlloc(JXF_Real, nLU + 1, JXF_MEMORY_HOST);
//    jcol   = jxf_TAlloc(JXF_Int, nLU + 1, JXF_MEMORY_HOST);
//    jnnz   = jxf_TAlloc(JXF_Int, nLU + 1, JXF_MEMORY_HOST);

//    max_value = -1.0;

//    /* first need to build gtol */
//    for (ii = 0; ii < nLU; ii++)
//    {
//       /* find real row */
//       i = perm[ii];
//       k1 = A_diag_i[i];
//       k2 = A_diag_i[i + 1];

//       /* find max|a| of that row and its index */
//       jxf_ILUMaxRabs(A_diag_data, A_diag_j, k1, k2, nLU, rperm,
//                        weight + ii, jcol + ii, &norm, jnnz + ii);
//       weight[ii] /= norm;
//       if (weight[ii] > max_value)
//       {
//          max_value = weight[ii];
//       }
//    }

//    gtol = tol * max_value;

//    /* second loop to pre select B */
//    nB_pre = 0;
//    for ( ii = 0 ; ii < nLU ; ii ++)
//    {
//       /* keep this row */
//       if (weight[ii] > gtol)
//       {
//          weight[nB_pre] /= (JXF_Real)(jnnz[ii]);
//          pperm_pre[nB_pre] = perm[ii];
//          qperm_pre[nB_pre++] = A_diag_j[jcol[ii]];
//       }
//    }

//    *nB = nB_pre;

//    /* sort from small to large */
//    jxf_qsort3(weight, pperm_pre, qperm_pre, 0, nB_pre - 1);

//    jxf_TFree(weight);
//    jxf_TFree(jcol);
//    jxf_TFree(jnnz);

//    return jxf_error_flag;
// }


// JXF_Int
// jxf_ILUMaxRabs(JXF_Real  *array_data,
//                  JXF_Int   *array_j,
//                  JXF_Int    start,
//                  JXF_Int    end,
//                  JXF_Int    nLU,
//                  JXF_Int   *rperm,
//                  JXF_Real  *value,
//                  JXF_Int   *index,
//                  JXF_Real  *l1_norm,
//                  JXF_Int   *nnz)
// {
//    JXF_Int i, idx, col, nz;
//    JXF_Real val, max_value, norm;

//    nz = 0;
//    norm = 0.0;
//    max_value = -1.0;
//    idx = -1;
//    if (rperm)
//    {
//       /* apply rperm and nLU */
//       for (i = start ; i < end ; i ++)
//       {
//          col = rperm[array_j[i]];
//          if (col > nLU)
//          {
//             /* this old column is in new external part */
//             continue;
//          }
//          nz ++;
//          val = abs(array_data[i]);
//          norm += val;
//          if (max_value < val)
//          {
//             max_value = val;
//             idx = i;
//          }
//       }
//    }
//    else
//    {
//       /* basic search */
//       for (i = start ; i < end ; i ++)
//       {
//          val = abs(array_data[i]);
//          norm += val;
//          if (max_value < val)
//          {
//             max_value = val;
//             idx = i;
//          }
//       }
//       nz = end - start;
//    }

//    *value = max_value;
//    if (index)
//    {
//       *index = idx;
//    }
//    if (l1_norm)
//    {
//       *l1_norm = norm;
//    }
//    if (nnz)
//    {
//       *nnz = nz;
//    }

//    return jxf_error_flag;
// }



// /*--------------------------------------------------------------------------
//  * jxf_ILULocalRCMBuildFinalPerm
//  *--------------------------------------------------------------------------*/

// JXF_Int
// jxf_ILULocalRCMBuildFinalPerm(JXF_Int   start,
//                                 JXF_Int   end,
//                                 JXF_Int  *G_perm,
//                                 JXF_Int  *perm,
//                                 JXF_Int  *qperm,
//                                 JXF_Int **permp,
//                                 JXF_Int **qpermp)
// {
//    /* update to new index */
//    JXF_Int i = 0;
//    JXF_Int num_nodes = end - start;
//    JXF_Int *perm_temp = jxf_TAlloc(JXF_Int, num_nodes, JXF_MEMORY_HOST);

//    for ( i = 0 ; i < num_nodes ; i ++)
//    {
//       perm_temp[i] = perm[i + start];
//    }
//    for ( i = 0 ; i < num_nodes ; i ++)
//    {
//       perm[i + start] = perm_temp[G_perm[i]];
//    }
//    if (perm != qperm)
//    {
//       for ( i = 0 ; i < num_nodes ; i ++)
//       {
//          perm_temp[i] = qperm[i + start];
//       }
//       for ( i = 0 ; i < num_nodes ; i ++)
//       {
//          qperm[i + start] = perm_temp[G_perm[i]];
//       }
//    }

//    *permp   = perm;
//    *qpermp  = qperm;

//    jxf_TFree(perm_temp, JXF_MEMORY_HOST);

//    return jxf_error_flag;
// }

/*--------------------------------------------------------------------------
 * jxf_ILULocalRCM
 *
 * This function computes the RCM ordering of a sub matrix of
 * sparse matrix B = A(perm,perm)
 * For nonsymmetrix problem, is the RCM ordering of B + B'
 * A: The input CSR matrix
 * start:      the start position of the submatrix in B
 * end:        the end position of the submatrix in B ( exclude end, [start,end) )
 * permp:      pointer to the row permutation array such that B = A(perm, perm)
 *             point to NULL if you want to work directly on A
 *             on return, permp will point to the new permutation where
 *             in [start, end) the matrix will reordered. if *permp is not NULL,
 *             we assume that it lives on the host memory at input. At output,
 *             it lives in the same memory location as A.
 * qpermp:     pointer to the col permutation array such that B = A(perm, perm)
 *             point to NULL or equal to permp if you want symmetric order
 *             on return, qpermp will point to the new permutation where
 *             in [start, end) the matrix will reordered. if *qpermp is not NULL,
 *             we assume that it lives on the host memory at input. At output,
 *             it lives in the same memory location as A.
 * sym:        set to nonzero to work on A only(symmetric), otherwise A + A'.
 *             WARNING: if you use non-symmetric reordering, that is,
 *             different row and col reordering, the resulting A might be non-symmetric.
 *             Be careful if you are using non-symmetric reordering
 *
 * TODO (VPM): Implement RCM computation on the device.
 *             Use IntArray for perm.
 *             Move this function and internal RCM calls to parcsr_mv.
 *--------------------------------------------------------------------------*/

// JXF_Int
// jxf_ILULocalRCM(jxf_CSRMatrix *A,
//                   JXF_Int        start,
//                   JXF_Int        end,
//                   JXF_Int      **permp,
//                   JXF_Int      **qpermp,
//                   JXF_Int        sym)
// {
//    /* Input variables */
//    JXF_Int               num_nodes       = end - start;
//    JXF_Int               n               = jxf_CSRMatrixNumRows(A);
//    JXF_Int               ncol            = jxf_CSRMatrixNumCols(A);
//    JXF_MemoryLocation    memory_location = jxf_CSRMatrixMemoryLocation(A);
//    JXF_Int               A_nnz           = jxf_CSRMatrixNumNonzeros(A);
//    JXF_Int              *A_i;
//    JXF_Int              *A_j;

//    /* Local variables */
//    jxf_CSRMatrix         *GT        = NULL;
//    jxf_CSRMatrix         *GGT       = NULL;
//    jxf_CSRMatrix         *G         = NULL;
//    JXF_Int               *G_i       = NULL;
//    JXF_Int               *G_j       = NULL;
//    JXF_Int               *G_perm    = NULL;
//    JXF_Int               *perm_temp = NULL;
//    JXF_Int               *rqperm    = NULL;
//    JXF_Int               *d_perm    = NULL;
//    JXF_Int               *d_qperm   = NULL;
//    JXF_Int               *perm      = *permp;
//    JXF_Int               *qperm     = *qpermp;

//    JXF_Int                perm_is_qperm;
//    JXF_Int                i, j, row, col, r1, r2;
//    JXF_Int                G_nnz, G_capacity;

//    /* Set flag for computing row and column permutations (true) or only row permutation (false) */
//    perm_is_qperm = (perm == qperm) ? 1 : 0;

//    /* 1: Preprosessing
//     * Check error in input, set some parameters
//     */
//    if (num_nodes <= 0)
//    {
//       /* don't do this if we are too small */
//       return jxf_error_flag;
//    }

//    if (n != ncol || end > n || start < 0)
//    {
//       /* don't do this if the input has error */
//       jxf_printf("Error input, abort RCM\n");
//       return jxf_error_flag;
//    }

//    JXF_ANNOTATE_FUNC_BEGIN;
//    jxf_GpuProfilingPushRange("ILULocalRCM");

//    /* create permutation array if we don't have one yet */
//    if (!perm)
//    {
//       perm = jxf_TAlloc(JXF_Int, n, JXF_MEMORY_HOST);
//       for (i = 0; i < n; i++)
//       {
//          perm[i] = i;
//       }
//    }

//    /* Check for symmetric reordering, then point qperm to row reordering */
//    if (!qperm)
//    {
//       qperm = perm;
//    }

//    /* Compute reverse qperm ordering */
//    rqperm = jxf_TAlloc(JXF_Int, n, JXF_MEMORY_HOST);
//    for (i = 0; i < n; i++)
//    {
//       rqperm[qperm[i]] = i;
//    }

//    /* Set/Move A_i and A_j to host */
//    if (jxf_GetActualMemLocation(memory_location) == jxf_MEMORY_DEVICE)
//    {
//       A_i = jxf_TAlloc(JXF_Int, n + 1, JXF_MEMORY_HOST);
//       A_j = jxf_TAlloc(JXF_Int, A_nnz, JXF_MEMORY_HOST);

//       jxf_TMemcpy(A_i, jxf_CSRMatrixI(A), JXF_Int, n + 1,
//                     JXF_MEMORY_HOST, JXF_MEMORY_DEVICE);
//       jxf_TMemcpy(A_j, jxf_CSRMatrixJ(A), JXF_Int, A_nnz,
//                     JXF_MEMORY_HOST, JXF_MEMORY_DEVICE);
//    }
//    else
//    {
//       A_i = jxf_CSRMatrixI(A);
//       A_j = jxf_CSRMatrixJ(A);
//    }

//    /* 2: Build Graph
//     * Build Graph for RCM ordering
//     */
//    G_nnz = 0;
//    G_capacity = jxf_max((A_nnz * n * n / num_nodes / num_nodes) - num_nodes, 1);
//    G_i = jxf_TAlloc(JXF_Int, num_nodes + 1, JXF_MEMORY_HOST);
//    G_j = jxf_TAlloc(JXF_Int, G_capacity, JXF_MEMORY_HOST);

//    /* TODO (VPM): Extend jxf_CSRMatrixPermute to replace the block below */
//    for (i = 0; i < num_nodes; i++)
//    {
//       G_i[i] = G_nnz;
//       row = perm[i + start];
//       r1 = A_i[row];
//       r2 = A_i[row + 1];
//       for (j = r1; j < r2; j ++)
//       {
//          col = rqperm[A_j[j]];
//          if (col != row && col >= start && col < end)
//          {
//             /* this is an entry in G */
//             G_j[G_nnz++] = col - start;
//             if (G_nnz >= G_capacity)
//             {
//                JXF_Int tmp = G_capacity;
//                G_capacity = (JXF_Int) (G_capacity * EXPAND_FACT + 1);
//                G_j = jxf_TReAlloc_v2(G_j, JXF_Int, tmp, JXF_Int,
//                                        G_capacity, JXF_MEMORY_HOST);
//             }
//          }
//       }
//    }
//    G_i[num_nodes] = G_nnz;

//    /* Free memory */
//    if (A_i != jxf_CSRMatrixI(A))
//    {
//       jxf_TFree(A_i, JXF_MEMORY_HOST);
//    }
//    if (A_j != jxf_CSRMatrixJ(A))
//    {
//       jxf_TFree(A_j, JXF_MEMORY_HOST);
//    }

//    /* Create matrix G on the host */
//    G = jxf_CSRMatrixCreate(num_nodes, num_nodes, G_nnz);
//    jxf_CSRMatrixMemoryLocation(G) = JXF_MEMORY_HOST;
//    jxf_CSRMatrixI(G) = G_i;
//    jxf_CSRMatrixJ(G) = G_j;

//    /* Check if G is not empty (no need to do any kind of RCM) */
//    if (G_nnz > 0)
//    {
//       /* Sum G with G' if G is nonsymmetric */
//       if (!sym)
//       {
//          jxf_CSRMatrixData(G) = jxf_CTAlloc(JXF_Complex, G_nnz);
//          jxf_CSRMatrixTranspose(G, &GT, 1);
//          GGT = jxf_CSRMatrixAdd(1.0, G, 1.0, GT);
//          jxf_CSRMatrixDestroy(G);
//          jxf_CSRMatrixDestroy(GT);
//          G = GGT;
//          GGT = NULL;
//       }

//       /* 3: Build RCM on the host */
//       G_perm = jxf_TAlloc(JXF_Int, num_nodes, JXF_MEMORY_HOST);
//       jxf_ILULocalRCMOrder(G, G_perm);

//       /* 4: Post processing
//        * Free, set value, return
//        */

//       /* update to new index */
//       perm_temp = jxf_TAlloc(JXF_Int, num_nodes, JXF_MEMORY_HOST);
//       jxf_TMemcpy(perm_temp, &perm[start], JXF_Int, num_nodes,
//                     JXF_MEMORY_HOST, JXF_MEMORY_HOST);
//       for (i = 0; i < num_nodes; i++)
//       {
//          perm[i + start] = perm_temp[G_perm[i]];
//       }

//       if (!perm_is_qperm)
//       {
//          jxf_TMemcpy(perm_temp, &qperm[start], JXF_Int, num_nodes,
//                        JXF_MEMORY_HOST, JXF_MEMORY_HOST);
//          for (i = 0; i < num_nodes; i++)
//          {
//             qperm[i + start] = perm_temp[G_perm[i]];
//          }
//       }
//    }

//    /* Move to device memory if needed */
//    if (memory_location == JXF_MEMORY_DEVICE)
//    {
//       d_perm = jxf_TAlloc(JXF_Int, n, JXF_MEMORY_DEVICE);
//       jxf_TMemcpy(d_perm, perm, JXF_Int, n,
//                     JXF_MEMORY_DEVICE, JXF_MEMORY_HOST);
//       jxf_TFree(perm, JXF_MEMORY_HOST);

//       perm = d_perm;
//       if (perm_is_qperm)
//       {
//          qperm = d_perm;
//       }
//       else
//       {
//          d_qperm = jxf_TAlloc(JXF_Int, n, JXF_MEMORY_DEVICE);
//          jxf_TMemcpy(d_qperm, qperm, JXF_Int, n,
//                        JXF_MEMORY_DEVICE, JXF_MEMORY_HOST);
//          jxf_TFree(qperm, JXF_MEMORY_HOST);

//          qperm = d_qperm;
//       }
//    }

//    /* Set output pointers */
//    *permp  = perm;
//    *qpermp = qperm;

//    /* Free memory */
//    jxf_CSRMatrixDestroy(G);
//    jxf_TFree(G_perm, JXF_MEMORY_HOST);
//    jxf_TFree(perm_temp, JXF_MEMORY_HOST);
//    jxf_TFree(rqperm, JXF_MEMORY_HOST);

//    jxf_GpuProfilingPopRange();
//    JXF_ANNOTATE_FUNC_END;

//    return jxf_error_flag;
// }



// JXF_Int
// jxf_ILUGetLocalPerm(jxf_ParCSRMatrix  *A,
//                       JXF_Int          **perm_ptr,
//                       JXF_Int           *nLU,
//                       JXF_Int            reordering_type)
// {
//    /* get basic information of A */
//    JXF_Int             num_rows = jxf_ParCSRMatrixNumRows(A);
//    jxf_CSRMatrix      *A_diag = jxf_ParCSRMatrixDiag(A);

//    /* Local variables */
//    JXF_Int            *perm = NULL;

//    /* Compute local RCM ordering on the host */
//    if (reordering_type != 0)
//    {
//       jxf_ILULocalRCM(A_diag, 0, num_rows, &perm, &perm, 1);
//    }

//    /* Set output pointers */
//    *nLU = num_rows;
//    *perm_ptr = perm;

//    return jxf_error_flag;
// }