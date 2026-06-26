#include "naviixLogical.h"
#include <time.h>
#include "naviixMsh.h"
#include "naviixParallel.h"
#include "naviixMatrix.h"
#include "jx_mv.h"
#include "jx_pamg.h"
#include "jx_ilu.h"
#include "jx_krylov.h"
#include "jx_diagscale.h"
#include "jx_euclid.h"
#include "jx_combined.h"

JX_Int JXPAMG_Solver_Interface(Matrix* mat, JX_Int MaxIter, JX_Real mtol, JX_Int* iternum, JX_Real* residual, JX_Real* costTime, JX_Real* x)
{

    /* JXPAMG solver */
    JX_Int       max_levels;
    JX_Int       cycle_type;
    JX_Int       relax_type;
    JX_Int       measure_type;
    JX_Int       rap2;
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
    JX_Real     strong_threshold;
    JX_Real     max_row_sum;
    JX_Real     relax_wt;
    JX_Real     outer_wt;
    JX_Real     S_commpkg_switch;
    JX_Real     AIR_strong_th;
    JX_Int       coarse_threshold;
    JX_Int       amg_print_level;
    JX_Int       CF;
    JX_Int       TTest;
    JX_Real starttime, endtime;

    
    /* iterative method */
    JX_Int      solver_id           = 32;
    JX_Int       max_iter;
    JX_Int       k_dim;
    JX_Int       is_check_restarted;  /* peghoty, 2011/11/08 */
    JX_Int       print_level;
    
    //-----------------------
    //  参数设置
    //-----------------------
    max_levels       = 25;       /* 最大网格层数 */
    cycle_type       = 1;        /* Cycle 类型  1: V_Cycle; 2：W_Cycle */
    relax_type       = 6;        /* Relax 类型  3: hGS; 6：hSGS */
    coarse_solver    = 9;        /* coarse_solver 粗空间解法器类型  9: GE; 10: Pardiso */
    measure_type     = 0;        /* 影响值的计算方式 0：局部；1：全局 */
    rap2             = 0;        /* RAP计算方式  0：RAP；1：先算Q=AP，再算RQ */
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
    P_max_elmts      = 0;        /* 插值算子每行最大非零元个数 */
    agg_num_levels   = 0;        /* Aggressive粗化的层数 */
    ai_measure_type  = 0;        /* AI-策略  0: no; 1: yes */
    ai_relax_type    = 0;        /* AI-磨光  0: no; 1: yes */
    amg_print_level  = 0;        /* work only when AMG as preconditioner */
    //strong_threshold = 0.1;     /* 强弱连通参数, 0.25 for 2D, 0.5 for 3D is recommended */
    strong_threshold = 0.25;
    max_row_sum      = 0.9;      /* 行和参数 */
    relax_wt         = 1.0;
    outer_wt         = 1.0;
    S_commpkg_switch = 1.0;
    AIR_strong_th    = 0.25;
    
    coarse_threshold = 100;      /* 最粗网格层上网格节点个数的最大值 */
    
    max_iter            = 1000;   /* 迭代法最大迭代次数 */
    k_dim               = 30;     /* 回头数 */
    is_check_restarted  = 1;      /* peghoty, 2011/11/08 */
    print_level         = 0;      /* 0: 关闭；1：Setup参数；2：Solve参数；3：Setup+Solve参数 */
    TTest         = 1;              /* 0: 关闭；1：打印求解时间 */


    JX_IJMatrix jx_A;
    JX_IJVector jx_x;
    JX_IJVector jx_b;

    JX_Int ilower, iupper, local_size;
    JX_Int nbelements;

    nbelements = mshPtr->elementsNum;

    clock_t start, end;

    start = clock();

    ilower = mshPtr->elements[0].gId;
    iupper = mshPtr->elements[nbelements - 1].gId;
    local_size = iupper - ilower + 1;

    //矩阵初始化
    JX_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &jx_A);
    JX_IJMatrixSetObjectType(jx_A, JX_PARCSR);
    JX_IJMatrixInitialize(jx_A);

    JX_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &jx_b);
    JX_IJVectorSetObjectType(jx_b, JX_PARCSR);
    JX_IJVectorInitialize(jx_b);

    JX_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &jx_x);
    JX_IJVectorSetObjectType(jx_x, JX_PARCSR);
    JX_IJVectorInitialize(jx_x);

    //矩阵装配
    JX_Int num;
    JX_Int row;
    JX_Int ncols;
    JX_Int col[MAX_FACES + 1];
    JX_Int nvals;
    double val[MAX_FACES + 1];

    for (JX_Int elem = 0; elem < mshPtr->elementsNum; elem++)
    {
        ncols = 0;
        nvals = 0;

        row = mshPtr->elements[elem].gId;

        col[ncols] = mshPtr->elements[elem].gId;
        ncols++;

        val[nvals] = mat->diag[elem];
        nvals++;

        for (JX_Int j = 0; j < mshPtr->elements[elem].adjElemsNum; j++)
        {
            col[ncols] = mshPtr->elements[elem].adjElems[j];
            ncols++;

            val[nvals] = mat->off_diag[elem][j];
            nvals++;
        }

        num = mshPtr->elements[elem].adjElemsNum + 1;

        JX_IJMatrixSetValues(jx_A, 1, &num, &mshPtr->elements[elem].gId, col, val);
        JX_IJVectorSetValues(jx_b, 1, &mshPtr->elements[elem].gId, &mat->rhs[elem]);
    }

    JX_IJMatrixAssemble(jx_A);
    JX_IJVectorAssemble(jx_b);
    JX_IJVectorAssemble(jx_x);

    //矩阵求解
    JX_Int  i;

    JX_Real mid_value;

    JX_ParCSRMatrix parcsr_AMGp;
    JX_Solver par_solver;
    JX_Solver par_precond;
    JX_Int **grid_relax_points = NULL;
    JX_Int *num_grid_sweeps = NULL;


    JX_IJMatrixGetObject(jx_A, (void**)&parcsr_AMGp);
    jx_ParVector *par_x1 = jx_IJVectorObject(jx_x);
    jx_ParVector *par_b1 = jx_IJVectorObject(jx_b);

//-------------------------------------------//
    JX_Real b_norm2 = jx_ParVectorNorm2(par_b1);
    jx_printf("2-norm of par_b1: %e\n", b_norm2);
//-------------------------------------------//

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
            JX_PAMGSetSCommPkgSwitch(par_precond, S_commpkg_switch);
            JX_PAMGSetAIRStrongTh(par_precond, AIR_strong_th);
            if (ns_down > -1) JX_PAMGSetCycleNumSweeps(par_precond, ns_down, 1);           /* sweep for "down" */
            if (ns_up > -1) JX_PAMGSetCycleNumSweeps(par_precond, ns_up, 2);           /* sweep for "up" */
            JX_PAMGSetCycleNumSweeps(par_precond, ns_coarse, 3);           /* sweep for "coarsest" */
            JX_PAMGSetCycleRelaxType(par_precond, relax_type, 1);  /* relax_type for "down" */
            JX_PAMGSetCycleRelaxType(par_precond, relax_type, 2);  /* relax_type for "up" */
            JX_PAMGSetCycleRelaxType(par_precond, coarse_solver, 3);  /* relax_type for "coarsest" */
            
            JX_ParCSRGMRESCreate(MPI_COMM_WORLD, &par_solver);
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
            if (CF != -1) {
            JX_PAMGSetNumGridSweeps(par_precond, num_grid_sweeps);       // zhaoli
            JX_PAMGSetGridRelaxPoints(par_precond, grid_relax_points);   // zhaoli
		    }
            JX_PAMGSetSCommPkgSwitch(par_precond, S_commpkg_switch);
            JX_PAMGSetAIRStrongTh(par_precond, AIR_strong_th);
            if (ns_down > -1) JX_PAMGSetCycleNumSweeps(par_precond, ns_down, 1);           /* sweep for "down" */
            if (ns_up > -1) JX_PAMGSetCycleNumSweeps(par_precond, ns_up, 2);           /* sweep for "up" */
            JX_PAMGSetCycleNumSweeps(par_precond, ns_coarse, 3);           /* sweep for "coarsest" */
            JX_PAMGSetCycleRelaxType(par_precond, relax_type, 1);  /* relax_type for "down" */
            JX_PAMGSetCycleRelaxType(par_precond, relax_type, 2);  /* relax_type for "up" */
            JX_PAMGSetCycleRelaxType(par_precond, coarse_solver, 3);  /* relax_type for "coarsest" */
            
            JX_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &par_solver);
            JX_BiCGSTABSetMaxIter(par_solver, max_iter);
            JX_BiCGSTABSetTol(par_solver, mtol);
            JX_BiCGSTABSetAbsoluteTol(par_solver, 0.0);
            JX_BiCGSTABSetConvCriteria(par_solver, 0);
            JX_BiCGSTABSetLogging(par_solver, 1);
            JX_BiCGSTABSetPrintLevel(par_solver, print_level); /* 是否在屏幕上打印残量等信息 */

            JX_BiCGSTABSetPrecond(par_solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                        (JX_PtrToSolverFcn)JX_PAMGSetup, par_precond);
            
            JX_PAMGSetup(par_precond, (JX_ParCSRMatrix)parcsr_AMGp);
            
            JX_BiCGSTABSetup(par_solver, (JX_Matrix)parcsr_AMGp, (JX_Vector)par_b1, (JX_Vector)par_x1);
            
            if (TTest)
            {
                endtime = jx_MPI_Wtime();
                jx_GetWallTime(MPI_COMM_WORLD, "PAMG-BiCGSTAB Setup", starttime, endtime, 0, 4);
            }
            
            if (TTest) starttime = jx_MPI_Wtime();
         
            JX_BiCGSTABSolve(par_solver, (JX_Matrix)parcsr_AMGp, // preOperater
                                (JX_Matrix)parcsr_AMGp, (JX_Vector)par_b1, (JX_Vector)par_x1);

            if (TTest)
            {
                endtime = jx_MPI_Wtime();
                jx_GetWallTime(MPI_COMM_WORLD, "PAMG-BiCGSTAB Solve", starttime, endtime, 0, 4);
            }
        
            JX_BiCGSTABGetNumIterations(par_solver, iternum);
            JX_BiCGSTABGetFinalRelativeResidualNorm(par_solver, residual);

            if (print_level == 0 && myid == 0)
            {
                jx_printf(" >>> num_iterations = %d\n", iternum);
                jx_printf(" >>> final_res_norm = %.4le\n", residual);
            }

            JX_PAMGDestroy(par_precond);
            JX_ParCSRBiCGSTABDestroy(par_solver);
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