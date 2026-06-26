#include <time.h>
#include "jx_mv.h"
#include "jx_pamg.h"
#include "jx_ilu.h"
#include "jx_krylov.h"
#include "jx_diagscale.h"
#include "jx_euclid.h"
#include "jx_combined.h"


JX_Int JXPAMG_Solver_Interface(MPI_Comm comm, JX_Int ilower, JX_Int iupper, 
                               JX_Int* row_ptr, JX_Int* col_idx, JX_Int* val,
                               JX_Int num_rows, JX_Real* rhs, 
                               JX_Int MaxIter, JX_Real mtol, JX_Int* iternum, JX_Real* residual, JX_Real* costTime, JX_Real* x)
{
   JX_Int **grid_relax_points = NULL;
   JX_Int *num_grid_sweeps = NULL;
   
   /* DiagScale Precond */
   JX_Solver ds_solver;
   
   /* ILU Precond */
   JX_Solver ilu_solver;
   JX_Real drop_tol;
   
   /* Euclid solver */
   JX_Solver euclid_solver;
   JX_Int euclid_level;
   JX_Int euclid_bj;
   
   /* Combined */
   JX_Solver combined_solver;
   JX_Int theta_psi;
   JX_Int theta_rho;
   JX_Int theta_phi;
   JX_Real theta_dis;
   
   /* JXPAMG solver */
   JX_Solver    amg_solver;
   JX_Int       max_levels;
   JX_Int       cycle_type;
   JX_Int       relax_type;
   JX_Int       measure_type;
   JX_Int       rap2;
   // JX_Int       num_functions;
   JX_Int       ns_down;
   JX_Int       ns_up;
   JX_Int       ns_coarse;
   JX_Int       restri_type;
   JX_Int       keepTranspose;
   JX_Int       coarsen_type;
   JX_Int       coarse_solver;
   JX_Int       interp_type;
   JX_Int       P_max_elmts;
   JX_Int       agg_num_levels;
   JX_Int       ai_measure_type;
   JX_Int       ai_relax_type;
   JX_Real      strong_threshold;
   JX_Real      max_row_sum;
   
   JX_Real      relax_wt;
   JX_Real      outer_wt;
   
   JX_Real      S_commpkg_switch;
   JX_Real      AIR_strong_th;

   JX_Int       coarse_threshold;
   JX_Real      coarse_ratio;
   JX_Int       coarsestsolverid;
   JX_Int       conv_criteria;
   JX_Int       amg_print_level;
   JX_Int       CF;
   
   /* iterative method */
   JX_Solver    solver;
   JX_Real      resdown_0_threshold;
   JX_Real      convfac_threshold_2;
   JX_Int       max_iter;
   JX_Int       k_dim;
   JX_Int       is_check_restarted;  /* peghoty, 2011/11/08 */
   JX_Int       twonorm;
   JX_Int       problem_id;
   JX_Int       file_base;
   JX_Int       print_level;
   JX_Int       keepsol;
   JX_Int       TTest;
   JX_Int       cgs;
   JX_Int       unroll;
   
   /* other variables */
   JX_Int       lu_length;
   JX_Int       glosize, i;
   JX_Int       last_precond_type;
   JX_Int       initguess = 0;
   JX_Int       num_iterations;
   JX_Real      final_res_norm;
   JX_Real      norm;
   JX_Real      eps = 1e-8;
   JX_Real      trunc_factor;

   JX_Int       solver_id;
   JX_Int       myid, nprocs;
   JX_Real      starttime, endtime;
   //--------------------------
   // 获取进程数和进程编号
   //--------------------------
   jx_MPI_comm_rank(comm, &myid);
   jx_MPI_comm_size(comm, &nprocs);
   
   //-----------------------
   //  参数设置
   //-----------------------
   max_levels       = 25;       /* 最大网格层数 */
   cycle_type       = 1;        /* Cycle 类型  1: V_Cycle; 2：W_Cycle */
   relax_type       = 6;        /* Relax 类型  3: hGS; 6：hSGS */
   coarse_solver    = 9;        /* coarse_solver 粗空间解法器类型  9: GE; 10: Pardiso */
   measure_type     = 0;        /* 影响值的计算方式 0：局部；1：全局 */
   rap2             = 0;        /* RAP计算方式  0：RAP；1：先算Q=AP，再算RQ */
   // num_functions    = 1;     /* 一个节点自由度的个数 */
   ns_down          = 1;
   ns_up            = 1;
   ns_coarse        = 1;
   restri_type      = 0;        /* 限制算子类型 0: P^T, 1: AIR, 2: AIR-2 */
   keepTranspose    = 0;        /* 存放限制算子  0：no；1：yes */
   CF               = 1;
                           // zhaoli,2021.06.29, 
                           // CF=-1: 缺省值（ CF = 1），
                           // CF=0: 自然序，
                           // CF=1: down cycle 先 C 后 F, up cycle 先 F 后 C
                           // CF=2: down cycle 先 F 后 C, up cycle 先 C 后 F

   coarsen_type     = 10;        /* 粗化策略 */
   // coarsen_type = 0: CLJP
   // coarsen_type = 1: Ruge
   // coarsen_type = 11: Ruge 1st pass only
   // coarsen_type = 2: Ruge2B
   // coarsen_type = 3: Ruge3
   // coarsen_type = 4: Ruge3c
   // coarsen_type = 5: Ruge relax special points
   // coarsen_type = 6: Falgout
   // coarsen_type = 8: PMIS
   // coarsen_type = 10: HMIS
   // coarsen_type = 90: RCLJP
   // coarsen_type = 91: RRS0
   // coarsen_type = 990: CLJP_AI
   // coarsen_type = 991: Ruge_AI
   // coarsen_type = 993: Ruge3_AI
   // coarsen_type = 96: Falgout_AI
   // coarsen_type = 98: PMIS_AI
   // coarsen_type = 910: HMIS_AI
   // coarsen_type = 908, 918, 928, 938, 968: AI-TYPE.  
   
   interp_type      = 0;        /* 插值策略 */
   // interp_type = 0: modified classical interpolation
   // interp_type = 3: direct interpolation (with separation of weights)
   // interp_type = 4: multipass interpolation
   // interp_type = 5: multipass interpolation (with separation of weights)
   // interp_type = 6: extended classical modified interpolation
   // interp_type = 7: extended (if no common C neighbor) classical modified interpolation
   // interp_type = 8: standard interpolation
   // interp_type = 9: standard interpolation (with separation of weights)
   trunc_factor     = 0;        /* 插值矩阵截断因子 */
   P_max_elmts      = 0;        /* 插值算子每行最大非零元个数 */
   agg_num_levels   = 0;        /* Aggressive粗化的层数 */
   ai_measure_type  = 0;        /* AI-策略  0: no; 1: yes */
   ai_relax_type    = 0;        /* AI-磨光  0: no; 1: yes */
   amg_print_level  = 3;        /* work only when AMG as preconditioner */
   //strong_threshold = 0.1;     /* 强弱连通参数, 0.25 for 2D, 0.5 for 3D is recommended */
   strong_threshold = 0.1;
   max_row_sum      = 0.9;      /* 行和参数 */
   relax_wt         = 1.0;
   outer_wt         = 1.0;
   S_commpkg_switch = 1.0;
   AIR_strong_th    = 0.25;
   
   coarse_threshold = 100;      /* 最粗网格层上网格节点个数的最大值 */
   coarse_ratio     = 0.75;     /* 相邻两个网格层的粗点个数超过细点个数的 coarse_ratio, 则换成 CLJP 粗化 */
   coarsestsolverid = 9;        /* 最粗网格层解法器 */
   conv_criteria    = 0;        /* 收敛准则类型 */
   
   theta_psi = 4;               /* 控制多尺度强度的度量 */
   theta_rho = 3;               /* 控制多尺度分布的第一个度量 */
   theta_phi = 3;               /* 控制多尺度分布的第二个度量 */
   theta_dis = 1.0e-3;          /* 控制多尺度分布的个数 */
   
   drop_tol         = 0.0;      /* Drop-tolerance for ILU(0) factorization */
   
   euclid_level     = 1;        /* level of fill-in */
   euclid_bj        = 0;        /* Select PILU (0) or Block Jacobi ILU (1) */
   
   resdown_0_threshold = 1.0e-4; /* 第一次残量相对初始残差的下降阈值 */
   convfac_threshold_2 = 0.1;    /* 相继两次残量下降阈值 */
   max_iter            = 1000;   /* 迭代法最大迭代次数 */
   k_dim               = 30;     /* 回头数 */
   is_check_restarted  = 1;      /* peghoty, 2011/11/08 */
   twonorm             = 1;      /* PCG 法中的范数控制类型，0: B 范数; 1: l2 范数 */
   print_level         = 0;      /* 0: 关闭；1：Setup参数；2：Solve参数；3：Setup+Solve参数 */
   keepsol             = 0;      /* 是否保存解向量 */
   TTest               = 1;      /* 是否测试时间 */
   cgs                 = 1;      /* COGMRES: if 2 performs reorthogonalization */
   unroll              = 0;      /* COGMRES: Set number of unrolling in mass funcyions, can be 4 or 8. Default: no unrolling */
   solver_id           = 32;
   problem_id          = 1;
   file_base           = 0;

   JX_IJMatrix jx_A;
   JX_IJVector jx_x;
   JX_IJVector jx_b;

   JX_Int nbelements;

   nbelements = mshPtr->elementsNum;

   clock_t start, end;

   start = clock();

   //矩阵初始化
   JX_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &jx_A);
   JX_IJMatrixSetObjectType(jx_A, JX_PARCSR);
   JX_IJMatrixInitialize(jx_A);

   JX_IJVectorCreate(comm, ilower, iupper, &jx_b);
   JX_IJVectorSetObjectType(jx_b, JX_PARCSR);
   JX_IJVectorInitialize(jx_b);

   JX_IJVectorCreate(comm, ilower, iupper, &jx_x);
   JX_IJVectorSetObjectType(jx_x, JX_PARCSR);
   JX_IJVectorInitialize(jx_x);

   //矩阵装配
   JX_Int num;
   JX_Int row;
   JX_Int col;
   JX_Real value;
   JX_Int  i;

   for (i = 0; i < num_rows; i++)
   {
      row = i+ilower;
      num = row_ptr[i+1] - row_ptr[i];
      for (JX_Int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
      {
         col = col_idx[j];
         value = val[j];
         JX_IJMatrixSetValues(jx_A, 1, 1, row, col, value);
      }
      JX_IJVectorSetValues(jx_b, 1, row, rhs[i]);
      JX_IJVectorSetValues(jx_x, 1, row, x[i]);
   }

   JX_IJMatrixAssemble(jx_A);
   JX_IJVectorAssemble(jx_b);
   JX_IJVectorAssemble(jx_x);

   //矩阵求解
   
   JX_Real mid_value;

   JX_ParCSRMatrix parcsr_AMGp;
   JX_Solver par_solver;
   JX_Solver par_precond;
   JX_Int **grid_relax_points = NULL;
   JX_Int *num_grid_sweeps = NULL;


   JX_IJMatrixGetObject(jx_A, (void**)&parcsr_AMGp);
   jx_ParVector *par_x1 = jx_IJVectorObject(jx_x);
   jx_ParVector *par_b1 = jx_IJVectorObject(jx_b);
   
   // commpute residual R0 = b - A*x1
   jx_ParVector  *par_R0 = NULL;
   jx_ParVectorCopy(par_b1, par_R0);
   jx_ParCSRMatrixMatvec(-1.0, parcsr_AMGp, par_x1, 1.0, par_R0);

//-------------------------------------------//
    JX_Real b_norm2 = jx_ParVectorNorm2(par_b1);
    JX_Real R0_norm2 = jx_ParVectorNorm2(par_R0);
    jx_printf("2-norm of par_b1: %e\n", b_norm2);    
    jx_printf("2-norm of par_R0: %e\n", R0_norm2);
//-------------------------------------------//

   //Reset the tolerance to the initial relative residual norm with R0_norm2 intsead of b_norm2

   mtol *= R0_norm2 / b_norm2;



    if (restri_type) /* Set Restriction to be AIR */
    {
        interp_type = 100; /* 1-pt Interp */
        relax_type = 3;
        ns_down = 3;
        ns_up = 3;
        grid_relax_points = jx_CTAlloc(JX_Int *, 4);
        grid_relax_points[0] = NULL;
        grid_relax_points[1] = jx_CTAlloc(JX_Int, ns_down);
        grid_relax_points[2] = jx_CTAlloc(JX_Int, ns_up);
        grid_relax_points[3] = jx_CTAlloc(JX_Int, ns_coarse);
        for (i = 0; i < ns_down; i ++) grid_relax_points[1][i] = 0; /* down cycle */
        if (ns_up == 3) /* up cycle */
        {
            grid_relax_points[2][0] = -1; // F
            grid_relax_points[2][1] = -1; // F
            grid_relax_points[2][2] = 1;  // C
        }
        else if (ns_up == 2)
        {
            grid_relax_points[2][0] = -1;
            grid_relax_points[2][1] = -1;
        }
        for (i = 0; i < ns_coarse; i ++) grid_relax_points[3][i] = 0; /* coarse: all */
        coarse_threshold = 100;
        agg_num_levels = 0; /* does not support aggressive coarsening */
    }


   // zhaoli,2021.06.29, 
   // CF=0: 自然序，
   // CF=1: down cycle 先 C 后 F, up cycle 先 F 后 C
   // CF=2: down cycle 先 F 后 C, up cycle 先 C 后 F
   if (CF==0) 
   {
      grid_relax_points = jx_CTAlloc(JX_Int *, 4);
      grid_relax_points[0] = jx_CTAlloc(JX_Int, ns_down);   // 最细网格层
      grid_relax_points[1] = jx_CTAlloc(JX_Int, ns_down);   // 前磨光
      grid_relax_points[2] = jx_CTAlloc(JX_Int, ns_up);     // 后磨光
      grid_relax_points[3] = jx_CTAlloc(JX_Int, ns_coarse); // 粗空间

      num_grid_sweeps   = jx_CTAlloc(JX_Int,4);
      num_grid_sweeps[0] = ns_down;
      num_grid_sweeps[1] = ns_down;
      num_grid_sweeps[2] = ns_up;
      num_grid_sweeps[3] = ns_coarse;

      /* fine grid  所有点 */
      for (i = 0; i < ns_down; i++){
         grid_relax_points[0][i]   = 0; 
      }
      /* down cycle  所有点 */
      for (i = 0; i < ns_down; i++){
         grid_relax_points[1][i]   = 0; 
      }
      /* up cycle  所有点 */
      for (i = 0; i < ns_up; i++){
         grid_relax_points[2][i]   = 0; 
      }
      for (i = 0; i < ns_coarse; i ++) grid_relax_points[3][i] = 0; /* coarse: all */
   }
   else if( CF==1 ){
      ns_down *= 2;
      ns_up   *= 2;

      grid_relax_points = jx_CTAlloc(JX_Int *, 4);
      grid_relax_points[0] = jx_CTAlloc(JX_Int, ns_down);   // 最细网格层
      grid_relax_points[1] = jx_CTAlloc(JX_Int, ns_down);   // 前磨光
      grid_relax_points[2] = jx_CTAlloc(JX_Int, ns_up);     // 后磨光
      grid_relax_points[3] = jx_CTAlloc(JX_Int, ns_coarse); // 粗空间

      num_grid_sweeps   = jx_CTAlloc(JX_Int,4);
      num_grid_sweeps[0] = ns_down;
      num_grid_sweeps[1] = ns_down;
      num_grid_sweeps[2] = ns_up;
      num_grid_sweeps[3] = ns_coarse;


      /* fine grid  先 C 后 F*/
      for (i = 0; i < ns_down; i+=2){
         grid_relax_points[0][i]   = 1; 
         grid_relax_points[0][i+1] = -1; 
      }
      /* down cycle 先 C 后 F*/
      for (i = 0; i < ns_down; i+=2){
         grid_relax_points[1][i]   = 1; 
         grid_relax_points[1][i+1] = -1; 
      }
      /* up cycle 先 F 后 C*/
      for (i = 0; i < ns_up; i+=2){
         grid_relax_points[2][i]   = -1; 
         grid_relax_points[2][i+1] = 1; 
      }
      for (i = 0; i < ns_coarse; i ++) grid_relax_points[3][i] = 0; /* coarse: all */
   }
   else if( CF==2 ){
      ns_down *= 2;
      ns_up   *= 2;

      grid_relax_points = jx_CTAlloc(JX_Int *, 4);
      grid_relax_points[0] = jx_CTAlloc(JX_Int, ns_down);   // 最细网格层
      grid_relax_points[1] = jx_CTAlloc(JX_Int, ns_down);   // 前磨光
      grid_relax_points[2] = jx_CTAlloc(JX_Int, ns_up);     // 后磨光
      grid_relax_points[3] = jx_CTAlloc(JX_Int, ns_coarse); // 粗空间

      num_grid_sweeps   = jx_CTAlloc(JX_Int,4);
      num_grid_sweeps[0] = ns_down;
      num_grid_sweeps[1] = ns_down;
      num_grid_sweeps[2] = ns_up;
      num_grid_sweeps[3] = ns_coarse;


      /* fine grid  先 F 后 C*/
      for (i = 0; i < ns_down; i+=2){
         grid_relax_points[0][i]   = -1; 
         grid_relax_points[0][i+1] = 1; 
      }
      /* down cycle 先 F 后 C*/
      for (i = 0; i < ns_down; i+=2){
         grid_relax_points[1][i]   = -1; 
         grid_relax_points[1][i+1] = 1; 
      }
      /* up cycle 先 C 后 F*/
      for (i = 0; i < ns_up; i+=2){
         grid_relax_points[2][i]   = 1; 
         grid_relax_points[2][i+1] = -1; 
      }
      for (i = 0; i < ns_coarse; i ++) grid_relax_points[3][i] = 0; /* coarse: all */
   }

    switch (solver_id)
   {
        case 22:   /* PAMG-GMRES */
	   {
            JX_PAMGCreate(&par_precond);
            JX_PAMGSetRestriction(par_precond, restri_type);
            JX_PAMGSetMaxLevels(par_precond, max_levels);
            JX_PAMGSetMaxIter(par_precond, ns_up);
            JX_PAMGSetCycleType(par_precond, cycle_type);
            JX_PAMGSetMeasureType(par_precond, measure_type);
            JX_PAMGSetRAP2(par_precond, rap2);
            JX_PAMGSetKeepTranspose(par_precond, keepTranspose);
            JX_PAMGSetCoarsenType(par_precond, coarsen_type);
            JX_PAMGSetInterpType(par_precond, interp_type);
            JX_PAMGSetPMaxElmts(par_precond, P_max_elmts);
            JX_PAMGSetAggNumLevels(par_precond, agg_num_levels);
            JX_PAMGSetAIMeasureType(par_precond, ai_measure_type);
            JX_PAMGSetAIRelaxType(par_precond, ai_relax_type);
            JX_PAMGSetStrongThreshold(par_precond, strong_threshold);
            JX_PAMGSetMaxRowSum(par_precond, max_row_sum);
            JX_PAMGSetPrintLevel(par_precond, amg_print_level);
            JX_PAMGSetCoarseThreshold(par_precond, coarse_threshold);
            JX_PAMGSetRelaxWt(par_precond, relax_wt);
            JX_PAMGSetOuterWt(par_precond, outer_wt);
            JX_PAMGSetScommPkgSwitch(par_precond, S_commpkg_switch);
            JX_PAMGSetAIRStrongTh(par_precond, AIR_strong_th);
            if (ns_down > -1) JX_PAMGSetCycleNumSweeps(par_precond, ns_down, 1);           /* sweep for "down" */
            if (ns_up > -1) JX_PAMGSetCycleNumSweeps(par_precond, ns_up, 2);           /* sweep for "up" */
            JX_PAMGSetCycleNumSweeps(par_precond, ns_coarse, 3);           /* sweep for "coarsest" */
            JX_PAMGSetCycleRelaxType(par_precond, relax_type, 1);  /* relax_type for "down" */
            JX_PAMGSetCycleRelaxType(par_precond, relax_type, 2);  /* relax_type for "up" */
            JX_PAMGSetCycleRelaxType(par_precond, coarse_solver, 3);  /* relax_type for "coarsest" */
            
            JX_ParCSRGMRESCreate(comm, &par_solver);
            JX_GMRESSetKDim(par_solver, k_dim);
            JX_GMRESSetIsCheckRestarted(par_solver, is_check_restarted); 
            JX_GMRESSetMaxIter(par_solver, max_iter);
            JX_GMRESSetTol(par_solver, mtol);
            JX_GMRESSetLogging(par_solver, 1);
            JX_GMRESSetPrintLevel(par_solver, print_level); /* 是否在屏幕上打印残量等信息 */
            
            JX_GMRESSetPrecond(par_solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                        (JX_PtrToSolverFcn)JX_PAMGSetup, par_precond);
            
            JX_PAMGSetup(par_precond, (JX_ParCSRMatrix)parcsr_AMGp);
            
            JX_GMRESSetup(par_solver, (JX_Matrix)parcsr_AMGp, (JX_Vector)par_b1, (JX_Vector)par_x1);
            
            JX_GMRESSolve(par_solver, (JX_Matrix)parcsr_AMGp, // preOperater
                                (JX_Matrix)parcsr_AMGp, (JX_Vector)par_b1, (JX_Vector)par_x1);
        
            JX_GMRESGetNumIterations(par_solver, iternum);
            JX_GMRESGetFinalRelativeResidualNorm(par_solver, residual);

            JX_PAMGDestroy(par_precond);
            JX_ParCSRGMRESDestroy(par_solver);
        }
        break;

        case 32:   /* PAMG-BiCGSTAB */
        {
            if (myid == 0) jx_printf("\n >>> Solver: PAMG-BiCGSTAB \n\n");
            
            if (TTest) starttime = jx_MPI_Wtime();
            
            JX_PAMGCreate(&amg_solver);
            JX_PAMGSetMaxLevels(amg_solver, max_levels);
            JX_PAMGSetMaxIter(amg_solver, 1);
            JX_PAMGSetCycleType(amg_solver, cycle_type);
            JX_PAMGSetMeasureType(amg_solver, measure_type);
            JX_PAMGSetRAP2(amg_solver, rap2);
            JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
            JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
            JX_PAMGSetInterpType(amg_solver, interp_type);
            JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
            JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
            JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
            JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
            JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
            JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
            JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
            JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
            JX_PAMGSetRelaxWt(amg_solver, relax_wt);
            JX_PAMGSetOuterWt(amg_solver, outer_wt);
            if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
            if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
            JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
            JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
            JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
            JX_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */
            
            JX_ParCSRBiCGSTABCreate(comm, &solver);
            JX_BiCGSTABSetMaxIter(solver, max_iter);
            JX_BiCGSTABSetTol(solver, mtol);
            JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
            JX_BiCGSTABSetConvCriteria(solver, 0);
            JX_BiCGSTABSetLogging(solver, 1);
            JX_BiCGSTABSetPrintLevel(solver, print_level);
            
            JX_BiCGSTABSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                        (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);
            
            JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)parcsr_AMGp);
            
            JX_BiCGSTABSetup(solver, (JX_Matrix)parcsr_AMGp, (JX_Vector)par_b1, (JX_Vector)par_x1);
            
            if (TTest)
            {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Setup", starttime, endtime, 0, 4);
            }
            
            if (TTest) starttime = jx_MPI_Wtime();
            
            JX_BiCGSTABSolve(solver, (JX_Matrix)parcsr_AMGp, // preOperater
                                    (JX_Matrix)parcsr_AMGp, (JX_Vector)par_b1, (JX_Vector)par_x1);
            
            if (TTest)
            {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Solve", starttime, endtime, 0, 4);
            }
            
            JX_BiCGSTABGetNumIterations(solver, &num_iterations);
            JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
            
            if (print_level == 0 && myid == 0)
            {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> ||B_AX||/||B|| = %.4e\n", final_res_norm);
            jx_printf(" >>> ||B_AX||/||B-AX0|| = %.4e\n", final_res_norm*b_norm2/R0_norm2);
            }
            JX_PAMGDestroy(amg_solver);
            JX_ParCSRBiCGSTABDestroy(solver);
        }
        break;
    }
   
    for (JX_Int i = ilower; i <= iupper; i++)
    {
        JX_IJVectorGetValues(jx_x, 1, &i, &mid_value);
        x[i - ilower] = mid_value;
    }

    //内存释放
    JX_IJMatrixDestroy(jx_A);
    JX_IJVectorDestroy(jx_x);
    JX_IJVectorDestroy(jx_b);

    end = clock();

    (*costTime) = ((JX_Real)end - (JX_Real)start) / CLOCKS_PER_SEC;
}