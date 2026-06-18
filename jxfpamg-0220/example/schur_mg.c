//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  apctl_mg.c -- 适合于JASMIN 应用程序接口的 APCTL-Krylov 解法器调用示例程序.
 *
 * Created by Yue Xiaoqiang 2019/10/03
 *
 * Xiangtan University
 * yuexq1111@163.com
 *
 */

#include "jxf_pamg.h"
#include "jxf_krylov.h"
#include "jxf_apctl.h"

jxf_CSRMatrix *
jxf_CSRMatrixTDMGReorderByVariablesT( jxf_CSRMatrix *A, JXF_Int num_groups );
jxf_Vector *
jxf_SeqVectorTDMGReorderByVariablesT( jxf_Vector *x, JXF_Int num_groups );
JXF_Int
jxf_ParVectorSetRandomValues( jxf_ParVector *v, JXF_Int seed );
JXF_Int
jxf_SeqVectorSetRandomValues( jxf_Vector *x, JXF_Int seed );

//===========================================================================================//
//====================================== 主  函  数 ==========================================//
//===========================================================================================//

int
main( int argc, char *argv[] )
{
   MPI_Comm comm = MPI_COMM_WORLD;
   JXF_Int   myid, nprocs;

   JXF_Int arg_index   = 0;
   JXF_Int print_usage = 0;

   char *MatFile = NULL;
   char *RhsFile = NULL;

   /* local variables */
   jxf_CSRMatrix *A_s = NULL;
   jxf_CSRMatrix *B_s = NULL;

   jxf_ParCSRMatrix *par_mat = NULL;

   jxf_Vector *f_s = NULL;
   jxf_Vector *b_s = NULL;

   jxf_ParVector *par_rhs = NULL;

   jxf_ParVector *par_sol = NULL;

   JXF_Int ng, file_sss, problem_id;

   JXF_Int initguess = 0;
   JXF_Int num_iterations;

   JXF_Int *mypartition = NULL;

   JXF_Real norm, final_res_norm;
   JXF_Real starttime, endtime;
   JXF_Real starttimeT, endtimeT;

   JXF_Int     solver_id;
   JXF_Solver  schur_solver;
   JXF_Real    tol;                   // tolerance
   JXF_Int     max_iter;              // maximal number of iteration
   JXF_Int     k_dim;                 // number of restart
   JXF_Int     is_check_restarted;    // peghoty, 2011/11/08
   JXF_Int     print_level;           // how many info to be output?
   JXF_Int     keepsol;               // whether save the solution?

   JXF_Real    strong_threshold;
   JXF_Int     interp_type;
   JXF_Int     coarsen_type;
   JXF_Int     agg_num_levels;
   JXF_Int     coarse_threshold;
   JXF_Int     print_level_amg;
   JXF_Int     print_level_schur;  // how much info to be output?
                                  // 1: CPU information
                                  // 2: inner iteration information
                                  // 3: both CPU and inner iteration information

   JXF_Real  ARR_relax_tol;
   JXF_Real  AEE_relax_tol;
   JXF_Real  AII_relax_tol;
   JXF_Int   ARR_relax_type;
   JXF_Int   AEE_relax_type;
   JXF_Int   AII_relax_type;
   JXF_Int   fixit_pctl_R;          // fixed number of iterations for ARR in SCHUR
   JXF_Int   fixit_pctl_E;          // fixed number of iterations for AEE in SCHUR
   JXF_Int   fixit_pctl_I;          // fixed number of iterations for AII in SCHUR
   JXF_Int   fixit_brlx_R;          // fixed number of iterations for ARR in Block Relaxation
   JXF_Int   fixit_brlx_E;          // fixed number of iterations for AEE in Block Relaxation
   JXF_Int   fixit_brlx_I;          // fixed number of iterations for AII in Block Relaxation
   JXF_Int   use_fixedmode_R;
   JXF_Int   use_fixedmode_E;
   JXF_Int   use_fixedmode_I;

   /* parameters to describe the weaking coupling between AEE and VER, VEI */
   JXF_Real  theta_wc_E;
   JXF_Real  threshold_wc_E;

   /* parameters to describe the diagonal dominance */
   JXF_Real  theta_dd_R;
   JXF_Real  theta_dd_E;
   JXF_Real  theta_dd_I;
   JXF_Real  threshold_dd_R;
   JXF_Real  threshold_dd_E;
   JXF_Real  threshold_dd_I;

   /* flag to indicate whether diagonal elements of the three DiagonalBlock matrices
      are firstly stored in each row for the CSR format. peghoty, 2012/03/06 */
   JXF_Int   is_diagelm_first;
 
   /* Whether test the number of iterations for each sub-linearsystem solution? */
   JXF_Int    test_subls_iter;
   JXF_Int    debug_flag;
   JXF_Int    reset_zero;

   JXF_Solver solver;

   //--------------------------
   //  启动 MPI
   //--------------------------
   jxf_MPI_Init(&argc, &argv);
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   ng         = 1;
   solver_id  = 22;  // 22: SCHUR1-GMRES, 23: SCHUR2-GMRES
   problem_id = 13;

   //--------------------------------
   // 设置解法器 SCHUR-GMRES 的参数
   //--------------------------------
   tol                = 1.0e-8;
   max_iter           = 200;
   k_dim              = 30;
   is_check_restarted = 0;
   print_level        = 3;
   keepsol            = 0;
   print_level_schur  = 3;
   ARR_relax_type     = RELAX_AMG;
   AEE_relax_type     = RELAX_AMG;
   AII_relax_type     = RELAX_AMG;
   fixit_pctl_R       = 1;
   fixit_pctl_E       = 1;
   fixit_pctl_I       = 1;
   fixit_brlx_R       = 3;
   fixit_brlx_E       = 1;
   fixit_brlx_I       = 1;
   ARR_relax_tol      = 1.0e-6;
   AEE_relax_tol      = 1.0e-6;
   AII_relax_tol      = 1.0e-6;
   use_fixedmode_R    = 1;
   use_fixedmode_E    = 1;
   use_fixedmode_I    = 1;
   theta_wc_E         = 1.0e-4;
   threshold_wc_E     = 0.5;
   theta_dd_R         = 0.9;
   theta_dd_E         = 0.9;
   theta_dd_I         = 0.9;
   threshold_dd_R     = 1.0;
   threshold_dd_E     = 1.0;
   threshold_dd_I     = 1.0;
   strong_threshold   = 0.25;
   interp_type        = 0;
   coarsen_type       = 6;
   agg_num_levels     = 0;
   coarse_threshold   = 100;
   print_level_amg    = 3;
   is_diagelm_first   = 0;
   test_subls_iter    = 1;
   debug_flag         = 0; /* 0:no; 1:setup; 2:solve; 3:setup+solve */
   reset_zero         = 1;

   //-----------------------
   //  命令行修改参数
   //-----------------------
   while (arg_index < argc)
   {
       if ( strcmp(argv[arg_index], "-pid") == 0 )
       {
           arg_index ++;
           problem_id = atoi(argv[arg_index++]);
       }
       else if ( strcmp(argv[arg_index], "-sid") == 0 )
       {
           arg_index ++;
           solver_id = atoi(argv[arg_index++]);
       }
       else if ( strcmp(argv[arg_index], "-ng") == 0 )
       {
           arg_index ++;
           ng = atoi(argv[arg_index++]);
       }
       else if ( strcmp(argv[arg_index], "-help") == 0 )
       {
           print_usage = 1;
           break;
       }
       else
       {
           arg_index ++;
       }
   }

   if (print_usage)
   {
      jxf_printf("\n");
      jxf_printf("  Usage: %s [<options>]\n", argv[0]);
      jxf_printf("\n");
      jxf_printf("  -pid     <val> : problem id\n");
      jxf_printf("  -sid     <val> : solver id\n");
      jxf_printf("  -fb      <val> : file base\n");
      jxf_printf("  -ng      <val> : number of groups\n");
      jxf_printf("  -help          : using help message\n\n");
      exit(1);
   }

   //-------------------------------------------------------------
   //  提供线性系统文件
   //-------------------------------------------------------------

   file_sss = 0;
   switch (problem_id)
   {
      case 13:
      MatFile = "/home/yuexq/data/3d3t/128/A.000061.01";
      RhsFile = "/home/yuexq/data/3d3t/128/b.000061.01";
      break;
      case 14:
      MatFile = "/home/yuexq/data/3d3t/128/A.000095.01";
      RhsFile = "/home/yuexq/data/3d3t/128/b.000095.01";
      break;
      case 15:
      MatFile = "/home/yuexq/data/3d3t/128/A.000106.01";
      RhsFile = "/home/yuexq/data/3d3t/128/b.000106.01";
      break;
   }

   if (myid == 0)
   {
      jxf_printf("\n\n ++++++++++++++ problem %d ng = %d np = %d ++++++++++++\n\n", problem_id, ng, nprocs);
   }

   //----------------------------------------------------------------------------------------
   // 在 0 号进程，利用数据文件生成串行矩阵 A_s, 串行右端 f_s,
   //----------------------------------------------------------------------------------------

   starttime = jxf_MPI_Wtime();

   if (myid == 0)
   {
      if (file_sss)
      {
         A_s = jxf_CSRMatrixRead(MatFile, 1);
      }
      else
      {
         B_s = jxf_CSRMatrixRead(MatFile, 1);
         A_s = jxf_CSRMatrixTDMGReorderByVariablesT(B_s, ng);
         jxf_CSRMatrixDestroy(B_s);
      }
      if (file_sss)
      {
         f_s = jxf_SeqVectorRead(RhsFile);
      }
      else
      {
         b_s = jxf_SeqVectorRead(RhsFile);
         f_s = jxf_SeqVectorTDMGReorderByVariablesT(b_s, ng);
         jxf_SeqVectorDestroy(b_s);
      }
   }

   //-------------------------------------------------------------------------
   //  生成并行矩阵和并行向量
   //-------------------------------------------------------------------------

   par_mat = jxf_CSRMatrixToParCSRMatrix(comm, A_s, NULL, NULL);
   jxf_ParCSRMatrixGetRowPartitioning(par_mat, &mypartition);
   par_rhs = jxf_VectorToParVector(comm, f_s, mypartition);
   par_sol = jxf_ParVectorCreate(comm, jxf_ParVectorGlobalSize(par_rhs), mypartition);
   jxf_ParVectorSetPartitioningOwner(par_sol, 0);
   jxf_ParVectorInitialize(par_sol);
   if (initguess == 0)
   {
      jxf_ParVectorSetConstantValues(par_sol, 0.0);
   }
   else
   {
      jxf_ParVectorSetRandomValues(par_sol, 22775);
      norm = jxf_ParVectorInnerProd(par_sol, par_sol);
      norm = 1.0 / sqrt(norm);
      jxf_ParVectorScale(norm, par_sol);
   }

   endtime = jxf_MPI_Wtime();
   jxf_GetWallTime(comm, "BuildParLinearSystem", starttime, endtime, 0, 2);

   //----------------------------------------------------------------------------------------
   // 求解线性代数系统
   //----------------------------------------------------------------------------------------

   starttimeT = jxf_MPI_Wtime();

   switch (solver_id)
   {
      case 22:  /* SCHUR1-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: SCHUR1-GMRES(%d) \n\n", k_dim);

         starttime = jxf_MPI_Wtime();

         JXF_3tAPCTLDataInitialize(&schur_solver);
         JXF_3tAPCTLSetNumGroup(schur_solver, ng);
         JXF_3tAPCTLSetA(schur_solver, par_mat);
         JXF_3tAPCTLSetIsDiagElmFirst(schur_solver, is_diagelm_first);
         JXF_3tAPCTLSetSubBlocks(schur_solver);
         JXF_3tAPCTLSetMaxiter(schur_solver, 1);
         JXF_3tAPCTLSetPrintLevel(schur_solver, print_level_schur);
         JXF_3tAPCTLSetARRRelaxTol(schur_solver, ARR_relax_tol);
         JXF_3tAPCTLSetAEERelaxTol(schur_solver, AEE_relax_tol);
         JXF_3tAPCTLSetAIIRelaxTol(schur_solver, AII_relax_tol);
         JXF_3tAPCTLSetARRRelaxType(schur_solver, ARR_relax_type);
         JXF_3tAPCTLSetAEERelaxType(schur_solver, AEE_relax_type);
         JXF_3tAPCTLSetAIIRelaxType(schur_solver, AII_relax_type);
         JXF_3tAPCTLSetThetaWCE(schur_solver, theta_wc_E);
         JXF_3tAPCTLSetThresholdWCE(schur_solver, threshold_wc_E);
         JXF_3tAPCTLSetThetaDDR(schur_solver, theta_dd_R);
         JXF_3tAPCTLSetThetaDDE(schur_solver, theta_dd_E);
         JXF_3tAPCTLSetThetaDDI(schur_solver, theta_dd_I);
         JXF_3tAPCTLSetThresholdDDR(schur_solver, threshold_dd_R);
         JXF_3tAPCTLSetThresholdDDE(schur_solver, threshold_dd_E);
         JXF_3tAPCTLSetThresholdDDI(schur_solver, threshold_dd_I);
         JXF_3tAPCTLSetFixItPCTLR(schur_solver, fixit_pctl_R);
         JXF_3tAPCTLSetFixItPCTLE(schur_solver, fixit_pctl_E);
         JXF_3tAPCTLSetFixItPCTLI(schur_solver, fixit_pctl_I);
         JXF_3tAPCTLSetFixItBRLXR(schur_solver, fixit_brlx_R);
         JXF_3tAPCTLSetFixItBRLXE(schur_solver, fixit_brlx_E);
         JXF_3tAPCTLSetFixItBRLXI(schur_solver, fixit_brlx_I);
         JXF_3tAPCTLSetUseFixedModeR(schur_solver, use_fixedmode_R);
         JXF_3tAPCTLSetUseFixedModeE(schur_solver, use_fixedmode_E);
         JXF_3tAPCTLSetUseFixedModeI(schur_solver, use_fixedmode_I);
         JXF_3tAPCTLSetTestSubLSIter(schur_solver, test_subls_iter);
         JXF_3tAPCTLSetDebugFlag(schur_solver, debug_flag);
         JXF_3tAPCTLSetResetZero(schur_solver, reset_zero);
         JXF_3tAPCTLSetStrongThreshold(schur_solver, strong_threshold);
         JXF_3tAPCTLSetInterpType(schur_solver, interp_type);
         JXF_3tAPCTLSetCoarsenType(schur_solver, coarsen_type);
         JXF_3tAPCTLSetAggNumLevels(schur_solver, agg_num_levels);
         JXF_3tAPCTLSetCoarseThreshold(schur_solver, coarse_threshold);
         JXF_3tAPCTLSetPrintLevelAMG(schur_solver, print_level_amg);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted);
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level);

         JXF_GMRESSetPrecond( solver, (JXF_PtrToSolverFcn) JXF_3tABSC1mgPrecond, (JXF_PtrToSolverFcn) NULL, schur_solver );

         JXF_3tABSC1Setup4mgJasmin(schur_solver, (JXF_ParCSRMatrix)par_mat);

         JXF_GMRESSetup( solver, (JXF_Matrix)par_mat, (JXF_Vector)par_rhs, (JXF_Vector)par_sol );

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "SCHUR1 Setup", starttime, endtime, 0, 2);

         starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve( solver,
                        (JXF_Matrix)par_mat, // preOperator
                        (JXF_Matrix)par_mat,
                        (JXF_Vector)par_rhs,
                        (JXF_Vector)par_sol );

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "SCHUR1 Solve", starttime, endtime, 0, 2);

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
 
         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
  
         if (print_level_schur)
         {
            JXF_3tAPCTLIterCount(schur_solver);
         }

         JXF_3tAPCTLDestroy4mgJasmin(schur_solver);
         JXF_ParCSRGMRESDestroy(solver);

      }
      break;

      case 23:  /* SCHUR2-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: SCHUR2-GMRES(%d) \n\n", k_dim);

         starttime = jxf_MPI_Wtime();

         JXF_3tAPCTLDataInitialize(&schur_solver);
         JXF_3tAPCTLSetNumGroup(schur_solver, ng);
         JXF_3tAPCTLSetA(schur_solver, par_mat);
         JXF_3tAPCTLSetIsDiagElmFirst(schur_solver, is_diagelm_first);
         JXF_3tAPCTLSetSubBlocks(schur_solver);
         JXF_3tAPCTLSetMaxiter(schur_solver, 1);
         JXF_3tAPCTLSetPrintLevel(schur_solver, print_level_schur);
         JXF_3tAPCTLSetARRRelaxTol(schur_solver, ARR_relax_tol);
         JXF_3tAPCTLSetAEERelaxTol(schur_solver, AEE_relax_tol);
         JXF_3tAPCTLSetAIIRelaxTol(schur_solver, AII_relax_tol);
         JXF_3tAPCTLSetARRRelaxType(schur_solver, ARR_relax_type);
         JXF_3tAPCTLSetAEERelaxType(schur_solver, AEE_relax_type);
         JXF_3tAPCTLSetAIIRelaxType(schur_solver, AII_relax_type);
         JXF_3tAPCTLSetThetaWCE(schur_solver, theta_wc_E);
         JXF_3tAPCTLSetThresholdWCE(schur_solver, threshold_wc_E);
         JXF_3tAPCTLSetThetaDDR(schur_solver, theta_dd_R);
         JXF_3tAPCTLSetThetaDDE(schur_solver, theta_dd_E);
         JXF_3tAPCTLSetThetaDDI(schur_solver, theta_dd_I);
         JXF_3tAPCTLSetThresholdDDR(schur_solver, threshold_dd_R);
         JXF_3tAPCTLSetThresholdDDE(schur_solver, threshold_dd_E);
         JXF_3tAPCTLSetThresholdDDI(schur_solver, threshold_dd_I);
         JXF_3tAPCTLSetFixItPCTLR(schur_solver, fixit_pctl_R);
         JXF_3tAPCTLSetFixItPCTLE(schur_solver, fixit_pctl_E);
         JXF_3tAPCTLSetFixItPCTLI(schur_solver, fixit_pctl_I);
         JXF_3tAPCTLSetFixItBRLXR(schur_solver, fixit_brlx_R);
         JXF_3tAPCTLSetFixItBRLXE(schur_solver, fixit_brlx_E);
         JXF_3tAPCTLSetFixItBRLXI(schur_solver, fixit_brlx_I);
         JXF_3tAPCTLSetUseFixedModeR(schur_solver, use_fixedmode_R);
         JXF_3tAPCTLSetUseFixedModeE(schur_solver, use_fixedmode_E);
         JXF_3tAPCTLSetUseFixedModeI(schur_solver, use_fixedmode_I);
         JXF_3tAPCTLSetTestSubLSIter(schur_solver, test_subls_iter);
         JXF_3tAPCTLSetDebugFlag(schur_solver, debug_flag);
         JXF_3tAPCTLSetResetZero(schur_solver, reset_zero);
         JXF_3tAPCTLSetStrongThreshold(schur_solver, strong_threshold);
         JXF_3tAPCTLSetInterpType(schur_solver, interp_type);
         JXF_3tAPCTLSetCoarsenType(schur_solver, coarsen_type);
         JXF_3tAPCTLSetAggNumLevels(schur_solver, agg_num_levels);
         JXF_3tAPCTLSetCoarseThreshold(schur_solver, coarse_threshold);
         JXF_3tAPCTLSetPrintLevelAMG(schur_solver, print_level_amg);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted);
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level);

         JXF_GMRESSetPrecond( solver, (JXF_PtrToSolverFcn) JXF_3tABSC2mgPrecond, (JXF_PtrToSolverFcn) NULL, schur_solver );

         JXF_3tABSC2Setup4mgJasmin(schur_solver, (JXF_ParCSRMatrix)par_mat);

         JXF_GMRESSetup( solver, (JXF_Matrix)par_mat, (JXF_Vector)par_rhs, (JXF_Vector)par_sol );

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "SCHUR2 Setup", starttime, endtime, 0, 2);

         starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve( solver,
                        (JXF_Matrix)par_mat, // preOperator
                        (JXF_Matrix)par_mat,
                        (JXF_Vector)par_rhs,
                        (JXF_Vector)par_sol );

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "SCHUR2 Solve", starttime, endtime, 0, 2);

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
 
         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
  
         if (print_level_schur)
         {
            JXF_3tAPCTLIterCount(schur_solver);
         }

         JXF_3tAPCTLDestroy4mgJasmin(schur_solver);
         JXF_ParCSRGMRESDestroy(solver);

      }
      break;
   }

   endtimeT = jxf_MPI_Wtime();
   jxf_GetWallTime(comm, "Total Sove Time", starttimeT, endtimeT, 0, 2);

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

   //-----------------------------------------------------------
   //  释放内存
   //-----------------------------------------------------------

   if (myid == 0)
   {
      jxf_CSRMatrixDestroy(A_s);
      jxf_SeqVectorDestroy(f_s);
   }
   jxf_ParCSRMatrixDestroy(par_mat);
   jxf_ParVectorDestroy(par_rhs);
   jxf_ParVectorDestroy(par_sol);

   //-----------------------------------------------------------
   //  终止 MPI
   //-----------------------------------------------------------
   MPI_Finalize();

   return 0;
}

jxf_CSRMatrix *
jxf_CSRMatrixTDMGReorderByVariablesT( jxf_CSRMatrix *A, JXF_Int num_groups )
{
   jxf_CSRMatrix *C = NULL;
   JXF_Int num_rows = jxf_CSRMatrixNumRows(A);
   JXF_Int *IA = jxf_CSRMatrixI(A);
   JXF_Int *JA = jxf_CSRMatrixJ(A);
   JXF_Real *AA = jxf_CSRMatrixData(A);
   JXF_Int *IC = NULL;
   JXF_Int *JC = NULL;
   JXF_Real *CC = NULL;
   JXF_Int RowA = 0, RowC = 0;
   JXF_Int ng_p_two = num_groups + 2;
   JXF_Int sub_num_rows = num_rows / ng_p_two;
   JXF_Int num_nonzerosC, j, mdo, col, row_end;

   C = jxf_CSRMatrixCreate(num_rows, num_rows, jxf_CSRMatrixNumNonzeros(A));
   jxf_CSRMatrixInitialize(C);
   IC = jxf_CSRMatrixI(C);
   JC = jxf_CSRMatrixJ(C);
   CC = jxf_CSRMatrixData(C);
   IC[0] = 0;
   num_nonzerosC = 0;
   for (RowC = 0; RowC < num_rows; RowC ++)
   {
      mdo = RowC / sub_num_rows;
      RowA = ng_p_two * (RowC - mdo * sub_num_rows) + mdo;
      row_end = IA[RowA+1];
      for (j = IA[RowA]; j < row_end; j ++)
      {
         col = JA[j];
         mdo = col % ng_p_two;
         JC[num_nonzerosC] = (col - mdo) / ng_p_two + mdo * sub_num_rows;
         CC[num_nonzerosC] = AA[j];
         num_nonzerosC ++;
      }
      IC[RowC+1] = num_nonzerosC;
   }

   return C;
}

jxf_Vector *
jxf_SeqVectorTDMGReorderByVariablesT( jxf_Vector *x, JXF_Int num_groups )
{
   JXF_Int size = jxf_VectorSize(x);
   JXF_Real *x_data = jxf_VectorData(x);
   jxf_Vector *y = jxf_SeqVectorCreate(size);
   JXF_Real *y_data = NULL;
   JXF_Int ng_p_two = num_groups + 2;
   JXF_Int sub_size = size / ng_p_two;
   JXF_Int Rowx = 0, Rowy, mdo;

   jxf_SeqVectorInitialize(y);
   y_data = jxf_VectorData(y);
   for (Rowy = 0; Rowy < size; Rowy ++)
   {
      mdo = Rowy / sub_size;
      Rowx = ng_p_two * (Rowy - mdo * sub_size) + mdo;
      y_data[Rowy] = x_data[Rowx];
   }

   return y;
}

JXF_Int
jxf_ParVectorSetRandomValues( jxf_ParVector *v, JXF_Int seed )
{
   JXF_Int my_id;
   jxf_Vector *v_local = jxf_ParVectorLocalVector(v);

   MPI_Comm 	comm = jxf_ParVectorComm(v);
   jxf_MPI_Comm_rank(comm, &my_id); 

   seed *= (my_id + 1);
           
   return jxf_SeqVectorSetRandomValues(v_local,seed);
}

JXF_Int
jxf_SeqVectorSetRandomValues( jxf_Vector *x, JXF_Int seed )
{
   JXF_Real  *vector_data = jxf_VectorData(x);
   JXF_Int      size        = jxf_VectorSize(x);
   JXF_Int      i, ierr = 0;

   jxf_SeedRand(seed);

   size *= jxf_VectorNumVectors(x);

   for (i = 0; i < size; i ++)
   {
      vector_data[i] = 2.0 * jxf_Rand() - 1.0;
   }
   
   return ierr;
}
