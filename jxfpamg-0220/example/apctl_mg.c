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

//===========================================================================================//
//====================================== 主  函  数 ==========================================//
//===========================================================================================//

int
main( int argc, char *argv[] )
{
   MPI_Comm  comm = MPI_COMM_WORLD;
   JXF_Int       myid, nprocs;

   JXF_Int arg_index   = 0;
   JXF_Int print_usage = 0;

   char *MatFile = NULL;
   char *RhsFile = NULL;

   /* local variables */
   jxf_CSRMatrix    **ARR_s = NULL;
   jxf_CSRMatrix     *AEE_s = NULL;
   jxf_CSRMatrix     *AII_s = NULL;
   jxf_Vector       **VRE_s = NULL;
   jxf_Vector       **VER_s = NULL;
   jxf_Vector        *VEI_s = NULL;
   jxf_Vector        *VIE_s = NULL;

   jxf_ParCSRMatrix  **ARR_p = NULL;
   jxf_ParCSRMatrix   *AEE_p = NULL;
   jxf_ParCSRMatrix   *AII_p = NULL;
   jxf_ParVector     **VRE_p = NULL;
   jxf_ParVector     **VER_p = NULL;
   jxf_ParVector      *VEI_p = NULL;
   jxf_ParVector      *VIE_p = NULL;

   jxf_CSRMatrix     *A_s  = NULL;
   jxf_CSRMatrix     *B_s  = NULL;

   jxf_Vector        *f_s  = NULL;
   jxf_Vector        *b_s  = NULL;
   jxf_Vector        *u_s  = NULL;

   jxf_Vector       **fR_s = NULL;
   jxf_Vector        *fE_s = NULL;
   jxf_Vector        *fI_s = NULL;

   jxf_ParVector    **fR_p = NULL;
   jxf_ParVector     *fE_p = NULL;
   jxf_ParVector     *fI_p = NULL;

   jxf_Vector       **uR_s = NULL;
   jxf_Vector        *uE_s = NULL;
   jxf_Vector        *uI_s = NULL;

   jxf_ParVector    **uR_p = NULL;
   jxf_ParVector     *uE_p = NULL;
   jxf_ParVector     *uI_p = NULL;

   JXF_Int ng, idx, file_sss, problem_id, file_base;

   JXF_Int *mypartition = NULL;

   jxf_APCTLKrylovParam *apctlkrylov_param = NULL; // apctl-gmres

   JXF_Int     solver_id;             // 1: PCG; 2: PGMRES; 3: PBiCGSTAB
   JXF_Int     precond_id;            // 1: APCTL; 2: Schur1; 3: Schur2
   JXF_Real  tol;                   // tolerance of the APCTL-Krylov method
   JXF_Int     max_iter;              // maximal number of iteration
   JXF_Int     k_dim;                 // number of restart
   JXF_Int     is_check_restarted;    // peghoty, 2011/11/08
   JXF_Int     print_level;           // how many info to be output?
   JXF_Int     TTest;                 // whether timing the program?
   JXF_Int     keepsol;               // whether save the solution?

   JXF_Real    strong_threshold;

   JXF_Int     interp_type;
   JXF_Int     coarsen_type;
   JXF_Int     agg_num_levels;
   JXF_Int     coarse_threshold;

   JXF_Int     print_level_amg;
   JXF_Int     print_level_apctl;     // how much info to be output in apctl?
                                  // 1: CPU information
                                  // 2: inner iteration information
                                  // 3: both CPU and inner iteration information
   JXF_Int     blocksmooth_type;      // BD or GS type preconditioner when Coarse Correction is not needed?
                                  // BLOCKSMOOTH_BD:
                                  // BLOCKSMOOTH_GS:

   /* solver type and restart number for interpolation-building of PRR */
   JXF_Int     interp_solver_ARR;     // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   JXF_Int     interp_kdim_ARR;       // restart parameters for GMRES solver

   JXF_Int     interp_maxit_ARR;      // maximal number of iteration for ARR to build PRR
   JXF_Int     interp_maxit_AII;      // maximal number of iteration for AII to build PII
   JXF_Real  interp_tol_ARR;        // tolerance for ARR to build PRR
   JXF_Real  interp_tol_AII;        // tolerance for AII to build PII

   JXF_Real  ARR_relax_tol;
   JXF_Real  AEE_relax_tol;
   JXF_Real  AII_relax_tol;
   JXF_Real  ACC_relax_tol;

   JXF_Int     ARR_relax_type;
   JXF_Int     AEE_relax_type;
   JXF_Int     AII_relax_type;

   JXF_Int     fixit_pctl_R;          // fixed number of iterations for ARR in PCTL
   JXF_Int     fixit_pctl_E;          // fixed number of iterations for AEE in PCTL
   JXF_Int     fixit_pctl_I;          // fixed number of iterations for AII in PCTL
   JXF_Int     fixit_brlx_R;          // fixed number of iterations for ARR in Block Relaxation
   JXF_Int     fixit_brlx_E;          // fixed number of iterations for AEE in Block Relaxation
   JXF_Int     fixit_brlx_I;          // fixed number of iterations for AII in Block Relaxation

   /* whether employ the fixed-number-of-iterations mode? peghoty, 2012/02/15 */
   JXF_Int     use_fixedmode_R;
   JXF_Int     use_fixedmode_E;
   JXF_Int     use_fixedmode_I;

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
   JXF_Int    is_diagelm_first;
 
   /* Whether use the pure PCTL? */
   JXF_Int    use_ppctl; 

   /* Whether test the number of iterations for each sub-linearsystem solution? */
   JXF_Int    test_subls_iter;

   JXF_Int    debug_flag;

   JXF_Int    reset_zero;

   //--------------------------
   //  启动 MPI
   //--------------------------
   jxf_MPI_Init(&argc, &argv);
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   ng = 1;
   file_sss = 0;
   file_base = 1;
   precond_id = 1;      // 1: APCTL; 2: Schur1; 3: Schur2
   problem_id = 13;

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
       else if ( strcmp(argv[arg_index], "-pcd") == 0 )
       {
           arg_index ++;
           precond_id = atoi(argv[arg_index++]);
       } 
       else if ( strcmp(argv[arg_index], "-fb") == 0 )
       {
           arg_index ++;
           file_base = atoi(argv[arg_index++]);
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
      jxf_printf("  -pcd     <val> : preconditioner id\n");
      jxf_printf("  -fb      <val> : file base\n");
      jxf_printf("  -ng      <val> : number of groups\n");
      jxf_printf("  -help          : using help message\n\n");
      exit(1);
   }

   //-------------------------------------------------------------
   //  提供线性系统文件
   //-------------------------------------------------------------
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

   ARR_s = jxf_CTAlloc(jxf_CSRMatrix *, ng);
   VRE_s = jxf_CTAlloc(jxf_Vector *, ng);
   VER_s = jxf_CTAlloc(jxf_Vector *, ng);
   ARR_p = jxf_CTAlloc(jxf_ParCSRMatrix *, ng);
   VRE_p = jxf_CTAlloc(jxf_ParVector *, ng);
   VER_p = jxf_CTAlloc(jxf_ParVector *, ng);
   fR_s  = jxf_CTAlloc(jxf_Vector *, ng);
   fR_p = jxf_CTAlloc(jxf_ParVector *, ng);
   uR_s = jxf_CTAlloc(jxf_Vector *, ng);
   uR_p = jxf_CTAlloc(jxf_ParVector *, ng);

   //----------------------------------------------------------------------------------------
   //  在 0 号进程，利用数据文件生成串行矩阵 A_s, 串行右端 f_s,
   //  串行解向量 u_s, 并抽取子矩阵和子向量
   //----------------------------------------------------------------------------------------
   if (myid == 0)
   {
      if (file_sss)
      {
         A_s = jxf_CSRMatrixRead(MatFile, file_base);
      }
      else
      {
         B_s = jxf_CSRMatrixRead(MatFile, file_base);
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
      u_s = jxf_SeqVectorCreate(jxf_VectorSize(f_s));
      jxf_SeqVectorInitialize(u_s);
      jxf_SeqVectorSetConstantValues(u_s, 0.0);
      jxf_mgGetSubBlocks_REIV(A_s, ARR_s, &AEE_s, &AII_s, VRE_s, VER_s, &VEI_s, &VIE_s, ng);
      jxf_mgGetSubVecs(f_s, fR_s, &fE_s, &fI_s, ng);
      jxf_mgGetSubVecs(u_s, uR_s, &uE_s, &uI_s, ng);
   }

   //-------------------------------------------------------------------------
   //  生成并行矩阵和并行向量
   //-------------------------------------------------------------------------

   for (idx = 0; idx < ng; idx ++) ARR_p[idx] = jxf_CSRMatrixToParCSRMatrix(comm, ARR_s[idx], NULL, NULL);
   AEE_p = jxf_CSRMatrixToParCSRMatrix(comm, AEE_s, NULL, NULL);
   AII_p = jxf_CSRMatrixToParCSRMatrix(comm, AII_s, NULL, NULL);
   for (idx = 0; idx < ng; idx ++) VRE_p[idx] = jxf_VectorToParVector(comm, VRE_s[idx], NULL);
   for (idx = 0; idx < ng; idx ++) VER_p[idx] = jxf_VectorToParVector(comm, VER_s[idx], NULL);
   VEI_p = jxf_VectorToParVector(comm, VEI_s, NULL);
   VIE_p = jxf_VectorToParVector(comm, VIE_s, NULL);

   for (idx = 0; idx < ng; idx ++) fR_p[idx] = jxf_VectorToParVector(comm, fR_s[idx], NULL);
   fE_p = jxf_VectorToParVector(comm, fE_s, NULL);
   fI_p = jxf_VectorToParVector(comm, fI_s, NULL);
   for (idx = 0; idx < ng; idx ++) uR_p[idx] = jxf_VectorToParVector(comm, uR_s[idx], NULL);
   uE_p = jxf_VectorToParVector(comm, uE_s, NULL);
   uI_p = jxf_VectorToParVector(comm, uI_s, NULL);

   //------------------------------------------------------------------------------
   //  设置解法器 APCTL-GMRES 的参数
   //------------------------------------------------------------------------------
   solver_id          = 2;      // 2: PGMRES
   tol                = 1.0e-8;
   max_iter           = 200;
   k_dim              = 30;
   is_check_restarted = 0;
   print_level        = 3;
   TTest              = 1;
   keepsol            = 0;

   print_level_apctl  = 0;
   blocksmooth_type   = BLOCKSMOOTH_BD; // BlockDiag

   interp_solver_ARR  = SOLVER_AMG; // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   interp_kdim_ARR    = 20;

   ARR_relax_type     = RELAX_AMG;
   AEE_relax_type     = RELAX_AMG;
   AII_relax_type     = RELAX_AMG;

   interp_maxit_ARR   = 200;
   interp_maxit_AII   = 200;
   interp_tol_ARR     = 1.0e-6;
   interp_tol_AII     = 1.0e-6;

   fixit_pctl_R       = 1;
   fixit_pctl_E       = 1;
   fixit_pctl_I       = 1;
   fixit_brlx_R       = 3;
   fixit_brlx_E       = 1;
   fixit_brlx_I       = 1;

   ARR_relax_tol      = 1.0e-6;
   AEE_relax_tol      = 1.0e-6;
   AII_relax_tol      = 1.0e-6;
   ACC_relax_tol      = 1.0e-6;

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

   strong_threshold = 0.25;

   interp_type = 0;
   coarsen_type = 6;
   agg_num_levels = 0;
   coarse_threshold = 100;
   print_level_amg = 3;

   is_diagelm_first   = 0;
   use_ppctl          = 0;  /* Whether use the pure PCTL? */

   test_subls_iter    = 1;

   debug_flag         = 0; /* 0:no; 1:setup; 2:solve; 3:setup+solve */

   reset_zero         = 1;

   apctlkrylov_param = jxf_APCTLKrylovParamInitialize();
   jxf_APCTLKrylovParamSetNumGroup(apctlkrylov_param, ng);
   jxf_APCTLKrylovParamSetSolverID(apctlkrylov_param, solver_id);
   jxf_APCTLKrylovParamSetPrecondID(apctlkrylov_param, precond_id);
   jxf_APCTLKrylovParamSetTol(apctlkrylov_param, tol);
   jxf_APCTLKrylovParamSetMaxIter(apctlkrylov_param, max_iter);
   jxf_APCTLKrylovParamSetKDim(apctlkrylov_param, k_dim);
   jxf_APCTLKrylovParamSetISCheckRestarted(apctlkrylov_param, is_check_restarted);
   jxf_APCTLKrylovParamSetPrintLevel(apctlkrylov_param, print_level);
   jxf_APCTLKrylovParamSetTTest(apctlkrylov_param, TTest);
   jxf_APCTLKrylovParamSetKeepSol(apctlkrylov_param, keepsol);

   jxf_APCTLKrylovParamSetPrintLevelAPCTL(apctlkrylov_param, print_level_apctl);
   jxf_APCTLKrylovParamSetBlockSmoothType(apctlkrylov_param, blocksmooth_type);
   jxf_APCTLKrylovParamSetInterpSolverARR(apctlkrylov_param, interp_solver_ARR);
   jxf_APCTLKrylovParamSetInterpKdimARR(apctlkrylov_param, interp_kdim_ARR);
   jxf_APCTLKrylovParamSetInterpMaxitARR(apctlkrylov_param, interp_maxit_ARR);
   jxf_APCTLKrylovParamSetInterpMaxitAII(apctlkrylov_param, interp_maxit_AII);
   jxf_APCTLKrylovParamSetInterpTolARR(apctlkrylov_param, interp_tol_ARR);
   jxf_APCTLKrylovParamSetInterpTolAII(apctlkrylov_param, interp_tol_AII);
   jxf_APCTLKrylovParamSetARRRelaxTol(apctlkrylov_param, ARR_relax_tol);
   jxf_APCTLKrylovParamSetAEERelaxTol(apctlkrylov_param, AEE_relax_tol);
   jxf_APCTLKrylovParamSetAIIRelaxTol(apctlkrylov_param, AII_relax_tol);
   jxf_APCTLKrylovParamSetACCRelaxTol(apctlkrylov_param, ACC_relax_tol);
   jxf_APCTLKrylovParamSetFixitPCTLR(apctlkrylov_param, fixit_pctl_R);
   jxf_APCTLKrylovParamSetFixitPCTLE(apctlkrylov_param, fixit_pctl_E);
   jxf_APCTLKrylovParamSetFixitPCTLI(apctlkrylov_param, fixit_pctl_I);
   jxf_APCTLKrylovParamSetFixitBrlxR(apctlkrylov_param, fixit_brlx_R);
   jxf_APCTLKrylovParamSetFixitBrlxE(apctlkrylov_param, fixit_brlx_E);
   jxf_APCTLKrylovParamSetFixitBrlxI(apctlkrylov_param, fixit_brlx_I);
   jxf_APCTLKrylovParamSetARRRelaxType(apctlkrylov_param, ARR_relax_type);
   jxf_APCTLKrylovParamSetAEERelaxType(apctlkrylov_param, AEE_relax_type);
   jxf_APCTLKrylovParamSetAIIRelaxType(apctlkrylov_param, AII_relax_type);
   jxf_APCTLKrylovParamSetUseFixedModeR(apctlkrylov_param, use_fixedmode_R);
   jxf_APCTLKrylovParamSetUseFixedModeE(apctlkrylov_param, use_fixedmode_E);
   jxf_APCTLKrylovParamSetUseFixedModeI(apctlkrylov_param, use_fixedmode_I);
   jxf_APCTLKrylovParamSetThetaWCE(apctlkrylov_param, theta_wc_E);
   jxf_APCTLKrylovParamSetParamThresholdWCE(apctlkrylov_param, threshold_wc_E);
   jxf_APCTLKrylovParamSetThetaDDR(apctlkrylov_param, theta_dd_R);
   jxf_APCTLKrylovParamSetThetaDDE(apctlkrylov_param, theta_dd_E);
   jxf_APCTLKrylovParamSetThetaDDI(apctlkrylov_param, theta_dd_I);
   jxf_APCTLKrylovParamSetThresholdDDR(apctlkrylov_param, threshold_dd_R);
   jxf_APCTLKrylovParamSetThresholdDDE(apctlkrylov_param, threshold_dd_E);
   jxf_APCTLKrylovParamSetThresholdDDI(apctlkrylov_param, threshold_dd_I);
   jxf_APCTLKrylovParamSetISDiagElmFirst(apctlkrylov_param, is_diagelm_first);
   jxf_APCTLKrylovParamSetUsePPCTL(apctlkrylov_param, use_ppctl);
   jxf_APCTLKrylovParamSetTestSubLSIter(apctlkrylov_param, test_subls_iter);
   jxf_APCTLKrylovParamSetDebugFlag(apctlkrylov_param, debug_flag);
   jxf_APCTLKrylovParamSetResetZero(apctlkrylov_param, reset_zero);
   jxf_APCTLKrylovParamSetStrongThreshold(apctlkrylov_param, strong_threshold);
   jxf_APCTLKrylovParamSetInterpType(apctlkrylov_param, interp_type);
   jxf_APCTLKrylovParamSetCoarsenType(apctlkrylov_param, coarsen_type);
   jxf_APCTLKrylovParamSetAggNumLevels(apctlkrylov_param, agg_num_levels);
   jxf_APCTLKrylovParamSetCoarseThreshold(apctlkrylov_param, coarse_threshold);
   jxf_APCTLKrylovParamSetPrintLevelAMG(apctlkrylov_param, print_level_amg);

   //-----------------------------------------------------------
   //  调用 APCTL-GMRES 法进行求解
   //-----------------------------------------------------------
   jxf_ApctlKrylov_mgJASMIN(ARR_p, AEE_p, AII_p, VRE_p, VER_p, VEI_p, VIE_p,
                      fR_p, fE_p, fI_p, uR_p, uE_p, uI_p, apctlkrylov_param);

   if (myid == 0)
   {
      // 打印迭代次数和最终相对残量范数
      jxf_printf(" >> Number of Iterations  : %d\n", apctlkrylov_param->num_iterations);
      jxf_printf(" >> Last Relative Res Norm: %le\n\n", apctlkrylov_param->last_rel_nrm);

      // 打印 apctl 预条件子中各子系统的迭代次数
      if (test_subls_iter)
      {
         if (jxf_APCTLKrylovParamNeedCC(apctlkrylov_param))
         {
            jxf_printf(" >> num_iter_Ai_pctl_setup: %d\n", apctlkrylov_param->num_iter_Ai_pctl_setup);
            jxf_printf(" >> num_iter_Ar_pctl_setup: %d\n", apctlkrylov_param->num_iter_Ar_pctl_setup);
            jxf_printf(" >> num_iter_Ar_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ar_pctl_precond);
            jxf_printf(" >> num_iter_Ae_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ae_pctl_precond);
            jxf_printf(" >> num_iter_Ai_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ai_pctl_precond);
            jxf_printf(" >> num_iter_Ac_pctl_precond: %d\n\n", apctlkrylov_param->num_iter_Ac_pctl_precond);
         }
         else
         {
            jxf_printf(" >> num_iter_Ar_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ar_pctl_precond);
            jxf_printf(" >> num_iter_Ae_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ae_pctl_precond);
            jxf_printf(" >> num_iter_Ai_pctl_precond: %d\n\n", apctlkrylov_param->num_iter_Ai_pctl_precond);
         }
      }

      // 打印各个阶段的 CPU 时间信息
      if (TTest)
      {
         jxf_printf(" >> cpu_time_datatransf: %.3f seconds\n", apctlkrylov_param->cpu_trans);
         jxf_printf(" >> cpu_time_setupphase: %.3f seconds\n", apctlkrylov_param->cpu_setup);
         jxf_printf(" >> cpu_time_solvephase: %.3f seconds\n\n", apctlkrylov_param->cpu_solve);
         jxf_printf(" >> cpu_time_setupsolve: %.3f seconds\n\n", apctlkrylov_param->cpu_total);
      }
   }

   //-----------------------------------------------------------
   //  释放内存
   //-----------------------------------------------------------

   if (myid == 0)
   {
      jxf_CSRMatrixDestroy(A_s);
      jxf_SeqVectorDestroy(f_s);
      jxf_SeqVectorDestroy(u_s);

      for (idx = 0; idx < ng; idx ++) jxf_CSRMatrixDestroy(ARR_s[idx]);
      jxf_TFree(ARR_s);
      jxf_CSRMatrixDestroy(AEE_s);
      jxf_CSRMatrixDestroy(AII_s);

      for (idx = 0; idx < ng; idx ++) jxf_SeqVectorDestroy(VRE_s[idx]);
      jxf_TFree(VRE_s);
      for (idx = 0; idx < ng; idx ++) jxf_SeqVectorDestroy(VER_s[idx]);
      jxf_TFree(VER_s);
      jxf_SeqVectorDestroy(VEI_s);
      jxf_SeqVectorDestroy(VIE_s);

      for (idx = 0; idx < ng; idx ++) jxf_SeqVectorDestroy(uR_s[idx]);
      jxf_TFree(uR_s);
      jxf_SeqVectorDestroy(uE_s);
      jxf_SeqVectorDestroy(uI_s);

      for (idx = 0; idx < ng; idx ++) jxf_SeqVectorDestroy(fR_s[idx]);
      jxf_TFree(fR_s);
      jxf_SeqVectorDestroy(fE_s);
      jxf_SeqVectorDestroy(fI_s);
   }

   for (idx = 0; idx < ng; idx ++) jxf_ParCSRMatrixDestroy(ARR_p[idx]);
   jxf_TFree(ARR_p);
   jxf_ParCSRMatrixDestroy(AEE_p);
   jxf_ParCSRMatrixDestroy(AII_p);
   for (idx = 0; idx < ng; idx ++) jxf_ParVectorDestroy(VRE_p[idx]);
   jxf_TFree(VRE_p);
   for (idx = 0; idx < ng; idx ++) jxf_ParVectorDestroy(VER_p[idx]);
   jxf_TFree(VER_p);
   jxf_ParVectorDestroy(VEI_p);
   jxf_ParVectorDestroy(VIE_p);
   for (idx = 0; idx < ng; idx ++) jxf_ParVectorDestroy(fR_p[idx]);
   jxf_TFree(fR_p);
   jxf_ParVectorDestroy(fE_p);
   jxf_ParVectorDestroy(fI_p);
   for (idx = 0; idx < ng; idx ++) jxf_ParVectorDestroy(uR_p[idx]);
   jxf_TFree(uR_p);
   jxf_ParVectorDestroy(uE_p);
   jxf_ParVectorDestroy(uI_p);
   jxf_TFree(mypartition);
   jxf_TFree(apctlkrylov_param);

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
