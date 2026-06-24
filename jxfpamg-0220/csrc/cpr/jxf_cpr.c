//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jxf_cpr.c --  JXFPAMG CPR预条件器实现文件 
 *  Date: 2025/10/08
 */ 

#include "jxf_cpr.h"
#include "jxf_pamg.h"     // AMG求解器
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*---------------------------------------------------------------*
 *             内部辅助函数                                       *
 *----------------------------------------------------------------*/

/**
 * \brief 获取BSR矩阵的块大小
 */
static inline JXF_Int jxf_GetBSRBlockSize(jxf_ParBSRMatrix *A_bsr)
{
    if (A_bsr == NULL || A_bsr->diag == NULL) return 1;
    return jxf_BSRMatrixBlockSize(A_bsr->diag);
}

/**
 * \brief 创建与压力矩阵维度匹配的向量   
 */
static jxf_ParVector* jxf_CreatePressureVector(jxf_ParCSRMatrix *A_pressure)
{
    if (A_pressure == NULL) return NULL;
    
    MPI_Comm      comm     = jxf_ParCSRMatrixComm(A_pressure);
    JXF_BigInt     global_size = jxf_ParCSRMatrixGlobalNumRows(A_pressure);
    JXF_BigInt*    partition = jxf_ParCSRMatrixRowStarts(A_pressure);
    jxf_ParVector* x_par    = jxf_ParVectorCreate(comm, global_size, partition);
    jxf_ParVectorInitialize(x_par);
    
    return x_par;
}

/**
 * \brief 创建与BSR矩阵维度匹配的全局向量
 */
jxf_ParVector* jxf_CreateGlobalVector(jxf_ParBSRMatrix *A_bsr)
{
    if (A_bsr == NULL) return NULL;
    
    MPI_Comm      comm     = A_bsr->comm;
    // 获取BSR矩阵的块信息
    jxf_BSRMatrix* A_diag = jxf_ParBSRMatrixDiag(A_bsr);
    if (!A_diag) return NULL;
    
    JXF_Int block_size = jxf_BSRMatrixBlockSize(A_diag);
    JXF_BigInt     global_size = jxf_ParBSRMatrixGlobalNumRows(A_bsr);
    // 计算标量维度 = 块行数 × 块大小
    JXF_BigInt global_scalar_rows = global_size * block_size;
    JXF_BigInt*    partition ;
    jxf_ParBSRMatrixGetRowPartitioning(A_bsr, &partition);
    jxf_ParVector* x_par    = jxf_ParVectorCreate(comm, global_scalar_rows, partition);
    jxf_ParVectorInitialize(x_par);
    
    return x_par;
}

/**
 * \brief 限制算子：从全局向量提取压力分量
 */
JXF_Int jxf_RestrictPressureVector(jxf_ParVector *global_vec,
                                       jxf_ParVector *pressure_vec,
                                       JXF_Int block_size,
                                       JXF_Int pressure_index)
{
    if (global_vec == NULL || pressure_vec == NULL) return jxf_error_flag;

    jxf_Vector* x_local = jxf_ParVectorLocalVector(global_vec);
    jxf_Vector* y_local = jxf_ParVectorLocalVector(pressure_vec);
    JXF_Real*   x_data  = jxf_VectorData(x_local);
    JXF_Real*   y_data  = jxf_VectorData(y_local);
    JXF_Int     x_row   = jxf_VectorSize(x_local);
    JXF_Int     y_row   = jxf_VectorSize(y_local);
    JXF_Int     i;

    if (y_row * block_size != x_row) {
        ERROR_PRINTF("%s, y_row %d, nb %d, y_row * nb: %d !=  x_row: %d\n", 
                     __FUNCTION__, y_row, block_size, y_row * block_size, x_row);
        return jxf_error_flag;
    }

    for (i = 0; i < y_row; i++) {
        y_data[i] = x_data[i * block_size + pressure_index];
    }

    return JXF_SUCCESS;
}

/**
 * \brief 延拓算子：将压力分量延拓到全局向量
 */
JXF_Int jxf_ProlongatePressureVector(jxf_ParVector *pressure_vec,
                                         jxf_ParVector *global_vec,
                                         JXF_Int block_size,
                                         JXF_Int pressure_index,
                                         JXF_Int type)
{
    if (pressure_vec == NULL || global_vec == NULL) return jxf_error_flag;
    
    jxf_Vector* x_local = jxf_ParVectorLocalVector(pressure_vec);
    jxf_Vector* y_local = jxf_ParVectorLocalVector(global_vec);
    JXF_Real*   x_data  = jxf_VectorData(x_local);
    JXF_Real*   y_data  = jxf_VectorData(y_local);
    JXF_Int     x_row   = jxf_VectorSize(x_local);
    JXF_Int     y_row   = jxf_VectorSize(y_local);
    JXF_Int     i, j;

    if (y_row != x_row * block_size) {
        ERROR_PRINTF("%s, x_row %d, y_row %d, nb %d, y_row: %d !=  x_row * nb: %d\n", 
                     __FUNCTION__, x_row, y_row, block_size, y_row, x_row * block_size);
        return jxf_error_flag;
    }

    if (type == 0) {
        // 仅延拓：将压力分量放入全局向量，其他分量清零
        for (i = 0; i < x_row; i++) {
            for (j = 0; j < block_size; j++) {
                y_data[i * block_size + j] = (j == pressure_index) ? x_data[i] : 0.0;
            }
        }
    } else if (type == 1) {
        // 延拓+修正：将压力分量加到全局向量的对应位置
        for (i = 0; i < x_row; i++) {
            y_data[i * block_size + pressure_index] += x_data[i];
        }
    } else {
        WARN_PRINTF("%s, type %d don't implenmented!\n", __FUNCTION__, type);
        return jxf_error_flag;
    }

    return JXF_SUCCESS;
}

/*----------------------------------------------------------------*
 *             CPR预条件器创建和销毁                              *
 *----------------------------------------------------------------*/

jxf_CPRPrecond* JXF_CPRCreate(MPI_Comm comm)
{
    // printf(" [%s:%d] enter JXF_CPRCreate\n",__FUNCTION__, __LINE__);
    // fflush(stdout);

    jxf_CPRPrecond *cpr = (jxf_CPRPrecond*)malloc(sizeof(jxf_CPRPrecond));
    if (cpr == NULL) return NULL;
    
    // printf(" [%s:%d] JXF_CPRCreate\n",__FUNCTION__, __LINE__);
    // fflush(stdout);

    memset(cpr, 0, sizeof(jxf_CPRPrecond));
    cpr->comm = comm;

    // printf(" [%s:%d] JXF_CPRCreate\n",__FUNCTION__, __LINE__);
    // fflush(stdout);    

    // 设置默认参数
    cpr->pressure_index = 0;
    cpr->block_size = 1;
    cpr->threshold = 1e-15;
    cpr->stage1_maxit = 1;
    cpr->stage2_maxit = 1;
    cpr->stage1_solver_type = 1;  // AMG求解器
    cpr->stage2_solver_type = 2;  // BGS求解器
    cpr->is_initialized = 0;
    
    // printf(" [%s:%d] JXF_CPRCreate\n",__FUNCTION__, __LINE__);
    // fflush(stdout);

    return cpr;
}

JXF_Int JXF_CPRDestroy(jxf_CPRPrecond **cpr_ptr)
{
    if (cpr_ptr == NULL || *cpr_ptr == NULL) return jxf_error_flag;
    
    jxf_CPRPrecond *cpr = *cpr_ptr;
    
    // 销毁压力矩阵
    if (cpr->A_pressure != NULL) {
        jxf_ParCSRMatrixDestroy(&cpr->A_pressure);
    }
    
    // 销毁阶段1求解器
    if (cpr->stage1_solver != NULL) {
        JXF_PAMGDestroy(cpr->stage1_solver);
    }
    
    // 销毁阶段2求解器
    if (cpr->stage2_solver != NULL) {
        // 假设是BGS求解器，根据实际类型调用对应的销毁函数
        // JXF_BGSDestroy(cpr->stage2_solver);
        free(cpr->stage2_solver);
    }
    
    // 销毁工作向量
    if (cpr->rp != NULL) {
        jxf_ParVectorDestroy(cpr->rp);
    }
    
    if (cpr->xp != NULL) {
        jxf_ParVectorDestroy(cpr->xp);
    }
    
    // if (cpr->work != NULL) {
    //     jxf_ParVectorDestroy(cpr->work);
    // }
    
    free(cpr);
    *cpr_ptr = NULL;
    
    return JXF_SUCCESS;
}

JXF_Int JXF_CPRSetParameter(jxf_CPRPrecond *cpr, const char *param_name, JXF_Int value)
{
    if (cpr == NULL) return jxf_error_flag;
    
    if (strcmp(param_name, "pressure_index") == 0) {
        cpr->pressure_index = value;
    } else if (strcmp(param_name, "stage1_maxit") == 0) {
        cpr->stage1_maxit = value;
    } else if (strcmp(param_name, "stage2_maxit") == 0) {
        cpr->stage2_maxit = value;
    } else if (strcmp(param_name, "stage1_solver_type") == 0) {
        cpr->stage1_solver_type = value;
    } else if (strcmp(param_name, "stage2_solver_type") == 0) {
        cpr->stage2_solver_type = value;
    } else if (strcmp(param_name, "print_level") == 0) {
        cpr->print_level = value;
    } else if (strcmp(param_name, "block_size") == 0) {
        cpr->block_size = value;
    } else {
        return jxf_error_flag;
    }
    
    return JXF_SUCCESS;
}

JXF_Int JXF_CPRSetRealParameter(jxf_CPRPrecond *cpr, const char *param_name, JXF_Real value)
{
    if (cpr == NULL) return jxf_error_flag;
    
    if (strcmp(param_name, "threshold") == 0) {
        cpr->threshold = value;
    } else {
        return jxf_error_flag;
    }
    
    return JXF_SUCCESS;
}

/*----------------------------------------------------------------*
 *             CPR设置和求解函数实现                              *
 *----------------------------------------------------------------*/

JXF_Int JXF_CPRSetup(jxf_CPRPrecond *cpr, jxf_ParBSRMatrix* parbsr_matrix)
{


    if (cpr == NULL || parbsr_matrix == NULL) return jxf_error_flag;
    
    cpr->A_bsr = parbsr_matrix;
    cpr->block_size = jxf_GetBSRBlockSize(cpr->A_bsr);
    
    JXF_Real t1, t2;
    
    // ------------------------------------------------------------
    // 阶段1：压力子系统设置
    // ------------------------------------------------------------
    t1 = jxf_MPI_Wtime();
    
    // // 0. 先并行解耦

    // // 使用ABF解耦（推荐）
    // JXF_DecoupType decoup_type = JXF_DECOUP_ABF;
    // JXF_Int is_thermal = 0;  // 假设是非热力模型，根据实际情况调整
    
    // JXF_Int decoup_result = jxf_ParBSRMatrixDecouple(cpr->A_bsr, decoup_type, is_thermal);
    // if (decoup_result != JXF_SUCCESS) {
    //     if (myid == 0) printf("Error applying decoupling\n");
    //     return jxf_error_flag;
    // }
    
    // if (myid == 0) {
    //     printf("Decoupling completed (method: %s)\n", 
    //         jxf_DecoupTypeToString(decoup_type));
    // }


    // 1. 提取压力矩阵
    cpr->A_pressure = jxf_ParBSRMatrixGetSubmatrix(cpr->A_bsr,
                                                  cpr->pressure_index,
                                                  cpr->threshold);
    // printf(" [%s:%d] matrix_C = %p \n",__FUNCTION__, __LINE__, (void*)cpr->A_pressure);
    // fflush(stdout);  
    // printf(" [%s:%d] jxf_ParCSRMatrixComm(matrix_C) = %p \n",__FUNCTION__, __LINE__, (void*)jxf_ParCSRMatrixComm(cpr->A_pressure));
    // fflush(stdout);     
    // printf(" [%s:%d] jxf_CSRMatrixI(diag) = %p \n",__FUNCTION__, __LINE__, (void*)jxf_CSRMatrixI(jxf_ParCSRMatrixDiag(cpr->A_pressure)));
    // fflush(stdout);     

    if (cpr->A_pressure == NULL) return jxf_error_flag;
    
    // 2. 创建压力求解器（AMG）
    //-----------------------
    //  参数设置
    //-----------------------
    JXF_Int max_levels       = 25;       /* 最大网格层数 */
    JXF_Int max_its          = 1;       /* 最大iter */
    JXF_Int cycle_type       = 1;        /* Cycle 类型  1: V_Cycle; 2：W_Cycle */
    JXF_Int relax_type       = 6;        /* Relax 类型  3: hGS; 6：hSGS */
    JXF_Int coarse_solver    = 9;        /* coarse_solver 粗空间解法器类型  9: GE; 10: Pardiso */
    JXF_Int measure_type     = 0;        /* 影响值的计算方式 0：局部；1：全局 */
    JXF_Int rap2             = 0;        /* RAP计算方式  0：RAP；1：先算Q=AP，再算RQ */
    JXF_Int ns_down          = 1;
    JXF_Int ns_up            = 1;
    JXF_Int ns_coarse        = 1;
    JXF_Int restri_type      = 0;        /* 限制算子类型 0: P^T, 1: AIR, 2: AIR-2 */
    JXF_Int keepTranspose    = 0;        /* 存放限制算子  0：no；1：yes */
    JXF_Int CF               = -1;
    JXF_Int coarsen_type     = 10;        /* 粗化策略 */
    JXF_Int interp_type      = 6;        /* 插值策略 */
    JXF_Int trunc_factor     = 0;        /* 插值矩阵截断因子 */
    JXF_Int P_max_elmts      = 0;        /* 插值算子每行最大非零元个数 */
    JXF_Int agg_num_levels   = 0;        /* Aggressive粗化的层数 */
    JXF_Int ai_measure_type  = 0;        /* AI-策略  0: no; 1: yes */
    JXF_Int ai_relax_type    = 0;        /* AI-磨光  0: no; 1: yes */
    JXF_Int amg_print_level  = 0;        /* work only when AMG as preconditioner */
    JXF_Real strong_threshold = 0.25;     /* 强弱连通参数, 0.25 for 2D, 0.5 for 3D is recommended */
    JXF_Real max_row_sum      = 0.9;      /* 行和参数 */
    JXF_Real relax_wt         = 1.0;
    JXF_Real outer_wt         = 1.0;
    JXF_Real S_commpkg_switch = 1.0;
    JXF_Real AIR_strong_th    = 0.25;
    JXF_Int coarse_threshold  = 1000;      /* 最粗网格层上网格节点个数的最大值 */
    JXF_Real coarse_ratio     = 0.75;     /* 相邻两个网格层的粗点个数超过细点个数的 coarse_ratio, 则换成 CLJP 粗化 */
    JXF_Int conv_criteria     = 0;        /* 收敛准则类型 */
    JXF_Solver   amg_solver;

    JXF_PAMGCreate(&amg_solver);
    JXF_PAMGSetMaxLevels(amg_solver, max_levels);
    JXF_PAMGSetMaxIter(amg_solver, max_its);
    JXF_PAMGSetCycleType(amg_solver, cycle_type);
    JXF_PAMGSetMeasureType(amg_solver, measure_type);
    JXF_PAMGSetRAP2(amg_solver, rap2);
    JXF_PAMGSetKeepTranspose(amg_solver, keepTranspose);
    JXF_PAMGSetCoarsenType(amg_solver, coarsen_type);
    JXF_PAMGSetInterpType(amg_solver, interp_type);
    JXF_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
    JXF_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
    JXF_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
    JXF_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
    JXF_PAMGSetStrongThreshold(amg_solver, strong_threshold);
    JXF_PAMGSetMaxRowSum(amg_solver, max_row_sum);
    JXF_PAMGSetPrintLevel(amg_solver, amg_print_level);
    JXF_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
    JXF_PAMGSetRelaxWt(amg_solver, relax_wt);
    JXF_PAMGSetOuterWt(amg_solver, outer_wt);
    JXF_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
    JXF_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
    if (ns_down > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
    if (ns_up > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
    JXF_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
    JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
    JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
    JXF_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */
    
    JXF_PAMGSetup(amg_solver, (JXF_ParCSRMatrix)cpr->A_pressure);
    

    
    cpr->stage1_solver = amg_solver;
    // if (cpr->stage1_solver == NULL) return jxf_error_flag;
    
    // 设置AMG求解器
    // JXF_PAMGSetup(cpr->stage1_solver, (JXF_ParCSRMatrix)cpr->A_pressure);
    
    // 3. 创建工作向量
    cpr->rp = jxf_CreatePressureVector(cpr->A_pressure);
    cpr->xp = jxf_CreatePressureVector(cpr->A_pressure);



    // // 创建全局工作向量
    cpr->work = jxf_CreateGlobalVector(cpr->A_bsr);
    if (cpr->work == NULL) return jxf_error_flag;



    t2 = jxf_MPI_Wtime();
    cpr->stage1_setup_time = t2 - t1;
    
    // ------------------------------------------------------------
    // 阶段2：全局求解器设置
    // ------------------------------------------------------------
    t1 = jxf_MPI_Wtime();
    
    // 根据类型创建第二阶段求解器
    // 这里可以添加更多类型的求解器
    if (cpr->stage2_solver_type == 2) {  // BGS求解器
        // 暂时简化处理
        cpr->stage2_solver = NULL;
    }
    
    t2 = jxf_MPI_Wtime();
    cpr->stage2_setup_time = t2 - t1;
    
    cpr->is_initialized = 1;
    
    return JXF_SUCCESS;
}

JXF_Int JXF_CPRPrecond(jxf_CPRPrecond *cpr, jxf_ParBSRMatrix* par_matrix,
                     jxf_ParVector *par_rhs, jxf_ParVector *par_sol)
{
    MPI_Comm      comm     = par_matrix->comm;
    if (cpr == NULL || !cpr->is_initialized) return jxf_error_flag;
    if (par_rhs == NULL || par_sol == NULL) return jxf_error_flag;
    
    JXF_Real t1, t2;
    JXF_Int i;
    
    JXF_Int block_size = cpr->block_size;
    JXF_Int pressure_index = cpr->pressure_index;

    // 初始化解向量为零
    // jxf_ParVectorSetConstantValues(par_sol, 0.0);

    // ------------------------------------------------------------
    // 阶段1：求解压力子系统
    // ------------------------------------------------------------
    
    // 1.1 限制：从全局残差提取压力残差
    jxf_RestrictPressureVector(par_rhs, cpr->rp, block_size, pressure_index);

    // 1.2 求解压力系统：xp = A_p^{-1} * rp
    jxf_ParVectorSetConstantValues(cpr->xp, 0.0);

        // JXF_Int          *partitioning;

    // 计算并打印残差范数（调试用）
    if (cpr->print_level > 1) {
        // jxf_ParCSRMatrixGetRowPartitioning(cpr->A_pressure, &partitioning);
        // // 创建临时残差向量
        // jxf_ParVector *res_before = jxf_ParVectorCreate(comm,jxf_ParVectorGlobalSize(cpr->rp),partitioning);
        jxf_ParVector *res_before = jxf_CreatePressureVector(cpr->A_pressure);
        // 计算残差: r = b - A*x
        jxf_ParVectorCopy(cpr->rp, res_before);
        jxf_ParCSRMatrixMatvec(-1.0, cpr->A_pressure, cpr->xp, 1.0, res_before);

        JXF_Real xp_norm = jxf_ParVectorNorm2(cpr->xp);
        JXF_Real rp_norm = jxf_ParVectorNorm2(cpr->rp);
        JXF_Real res_before_norm = jxf_ParVectorNorm2(res_before);

        printf("CPR: xp_norm = %.6e, rp_norm = %.6e,  res_before_norm = %.6e\n", xp_norm, rp_norm, res_before_norm);
        fflush(stdout);
    }
    t1 = jxf_MPI_Wtime();
    for (i = 0; i < cpr->stage1_maxit; i++) {
        // 使用AMG求解压力系统
        JXF_PAMGSolve(cpr->stage1_solver, 
                     (JXF_ParCSRMatrix)cpr->A_pressure,
                     (JXF_Vector)cpr->rp,
                     (JXF_ParVector)cpr->xp);
    }

    t2 = jxf_MPI_Wtime();
    cpr->stage1_solve_time += t2 - t1;

    // 计算并打印残差范数（调试用）
    if (cpr->print_level > 1) {
        // // 创建临时残差向量
        // jxf_ParVector *res_after = jxf_ParVectorCreate(comm,jxf_ParVectorGlobalSize(cpr->rp),partitioning);
        jxf_ParVector *res_after = jxf_CreatePressureVector(cpr->A_pressure);
        // 计算残差: r = b - A*x
        jxf_ParVectorCopy(cpr->rp, res_after);
        jxf_ParCSRMatrixMatvec(-1.0, cpr->A_pressure, cpr->xp, 1.0, res_after);

        JXF_Real xp_after_norm = jxf_ParVectorNorm2(cpr->xp);
        JXF_Real rp_after_norm = jxf_ParVectorNorm2(cpr->rp);
        JXF_Real res_after_norm = jxf_ParVectorNorm2(res_after);

        printf("CPR: xp_after_norm = %.6e, rp_after_norm = %.6e,  res_after_norm = %.6e\n", xp_after_norm, rp_after_norm, res_after_norm);
        fflush(stdout);
    
    }


        // 计算并打印残差范数（调试用）
    if (cpr->print_level > 1) {
        // // 创建临时残差向量
        jxf_ParVector *global_res_before = jxf_CreateGlobalVector(cpr->A_bsr);
        // 计算残差: r = b - A*x
        jxf_ParVectorCopy(par_rhs, global_res_before);
        jxf_ParBSRMatrixMatvec(-1.0, cpr->A_bsr, par_sol, 1.0, global_res_before);

        JXF_Real global_xp_before_norm = jxf_ParVectorNorm2(par_sol);
        JXF_Real global_rp_before_norm = jxf_ParVectorNorm2(par_rhs);
        JXF_Real global_res_before_norm = jxf_ParVectorNorm2(global_res_before);

        printf("CPR: global_xp_before_norm = %.6e, global_rp_before_norm = %.6e,  global_res_before_norm = %.6e\n", global_xp_before_norm, global_rp_before_norm, global_res_before_norm);
        fflush(stdout);
    }
    // 1.3 延拓：将压力解延拓到全局向量
    jxf_ProlongatePressureVector(cpr->xp, par_sol, block_size, pressure_index, 0);
    // 打印AMG压力求解后的残差
    {
        jxf_ParVector *__r = jxf_CreateGlobalVector(cpr->A_bsr);
        jxf_ParVectorCopy(par_rhs, __r);
        jxf_ParBSRMatrixMatvec(-1.0, cpr->A_bsr, par_sol, 1.0, __r);
        printf("CPR: after AMG ||r||=%.6e\n", jxf_ParVectorNorm2(__r));
        jxf_ParVectorDestroy(__r);
    }

        // 计算并打印残差范数（调试用）
    if (cpr->print_level > 1) {
        // // 创建临时残差向量
        jxf_ParVector *global_res_after = jxf_CreateGlobalVector(cpr->A_bsr);
        // 计算残差: r = b - A*x
        jxf_ParVectorCopy(par_rhs, global_res_after);
        jxf_ParBSRMatrixMatvec(-1.0, cpr->A_bsr, par_sol, 1.0, global_res_after);

        JXF_Real global_xp_after_norm = jxf_ParVectorNorm2(par_sol);
        JXF_Real global_rp_after_norm = jxf_ParVectorNorm2(par_rhs);
        JXF_Real global_res_after_norm = jxf_ParVectorNorm2(global_res_after);

        printf("CPR: global_xp_after_norm = %.6e, global_rp_after_norm = %.6e,  global_res_after_norm = %.6e\n", global_xp_after_norm, global_rp_after_norm, global_res_after_norm);
        fflush(stdout);
    }
    // ------------------------------------------------------------
    // 阶段2：全局修正
    // ------------------------------------------------------------
    t1 = jxf_MPI_Wtime();
    
    // 设置松弛参数（这些可以配置在cpr->config中）
    JXF_Real relax_weight = 0.7;    // 松弛权重
    JXF_Real omega = 0.9;          // 阻尼因子
    JXF_Int forward_or_backward = 1; // 1: 前向, 0: 对称, -1: 后向

        // 创建临时残差向量
        jxf_ParVector *residual_before = jxf_CreateGlobalVector(cpr->A_bsr);
        jxf_ParVector *residual_after = jxf_CreateGlobalVector(cpr->A_bsr);

    if (cpr->print_level > 1) {

        // 计算残差: r = b - A*x
        jxf_ParVectorCopy(par_rhs, residual_before);
        jxf_ParVectorCopy(par_rhs, residual_after);

        jxf_ParBSRMatrixMatvec(-1.0, cpr->A_bsr, par_sol, 1.0, residual_before);

        JXF_Real residual_before_norm = jxf_ParVectorNorm2(residual_before);
        JXF_Real par_sol_before_norm = jxf_ParVectorNorm2(par_sol);

        printf("CPR: residual_before_norm = %.6e, par_sol_before_norm = %.6e \n",residual_before_norm, par_sol_before_norm);
        fflush(stdout);
    }
    for (i = 0; i < cpr->stage2_maxit; i++) {

        // 应用第二阶段预条件器: 使用简单的雅可比松弛
        // 这里可以根据stage2_solver_type调用不同的求解器
        if (cpr->stage2_solver_type == 2) {
            // 简单的雅可比迭代（对角预处理）
            // 计算对角线的逆
            // jxf_ParVectorCopy(residual, cpr->work);

            jxf_ParBSRHGSRelax(cpr->A_bsr, par_sol, par_rhs, cpr->work, relax_weight, omega, forward_or_backward);
            // // 修正解: x = x + delta_x               ! dont need
            // jxf_ParVectorAxpy(1.0, cpr->work, par_sol);
 
        } else if(cpr->stage2_solver_type == 3) {
            // 默认处理：直接使用残差作为修正
            jxf_ParBSRHSGSRelax(cpr->A_bsr, par_sol, par_rhs, cpr->work, relax_weight, omega, forward_or_backward);
        }
    }
    
        if (cpr->print_level > 1) {
            jxf_ParBSRMatrixMatvec(-1.0, cpr->A_bsr, par_sol, 1.0, residual_after);
            JXF_Real residual_after_norm = jxf_ParVectorNorm2(residual_after);
            JXF_Real par_sol_after_norm = jxf_ParVectorNorm2(par_sol);

            printf("CPR: residual_after_norm = %.6e, par_sol_after_norm = %.6e \n",residual_after_norm, par_sol_after_norm);
            fflush(stdout);        
        }
    // jxf_ParVectorDestroy(residual);

    // 打印磨光后的残差
    {
        jxf_ParVector *__r = jxf_CreateGlobalVector(cpr->A_bsr);
        jxf_ParVectorCopy(par_rhs, __r);
        jxf_ParBSRMatrixMatvec(-1.0, cpr->A_bsr, par_sol, 1.0, __r);
        printf("CPR: after smooth ||r||=%.6e\n", jxf_ParVectorNorm2(__r));
        jxf_ParVectorDestroy(__r);
    }
    t2 = jxf_MPI_Wtime();
    cpr->stage2_solve_time += t2 - t1;
    
    return JXF_SUCCESS;
}

/*----------------------------------------------------------------*
 *             工具函数实现                                       *
 *----------------------------------------------------------------*/

JXF_Int JXF_CPRGetStatistics(jxf_CPRPrecond *cpr, JXF_Real *stats)
{
    if (cpr == NULL || stats == NULL) return jxf_error_flag;
    
    stats[0] = cpr->stage1_setup_time;
    stats[1] = cpr->stage1_solve_time;
    stats[2] = cpr->stage2_setup_time;
    stats[3] = cpr->stage2_solve_time;
    
    return JXF_SUCCESS;
}

JXF_Int JXF_CPRPrint(jxf_CPRPrecond *cpr, JXF_Int level)
{
    if (cpr == NULL) return jxf_error_flag;
    
    if (level >= 1) {
        jxf_printf("\n=== CPR Preconditioner ===\n");
        jxf_printf("Pressure index: %d\n", cpr->pressure_index);
        jxf_printf("Block size: %d\n", cpr->block_size);
        jxf_printf("Stage1 maxit: %d, Stage2 maxit: %d\n", 
                  cpr->stage1_maxit, cpr->stage2_maxit);
        
        if (level >= 2) {
            jxf_printf("Stage1 setup time: %.4f s\n", cpr->stage1_setup_time);
            jxf_printf("Stage1 solve time: %.4f s\n", cpr->stage1_solve_time);
            jxf_printf("Stage2 setup time: %.4f s\n", cpr->stage2_setup_time);
            jxf_printf("Stage2 solve time: %.4f s\n", cpr->stage2_solve_time);
        }
    }
    
    return JXF_SUCCESS;
}