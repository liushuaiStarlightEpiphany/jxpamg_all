// test_cpr_gmres.c - CPR-GMRES方法测试程序
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#include "jxf_mv.h"
#include "jxf_bsr_mv.h"
#include "jxf_parbsr_mv.h"
#include "jxf_cpr.h"
#include "jxf_bsr_decoup.h"
#include "jxf_pamg.h"
#include "jxf_krylov.h"

typedef double Real;
typedef int Int;

// 函数声明
jxf_BSRMatrix* read_bsr_matrix(const char* filename, int binary);
jxf_ParBSRMatrix* distribute_bsr_matrix(MPI_Comm comm, jxf_BSRMatrix* A_bsr, 
                                       JXF_BigInt* row_part, JXF_BigInt* col_part);

// 读取右端项文件函数
jxf_Vector* read_rhs_file(const char* filename, int expected_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening RHS file: %s\n", filename);
        return NULL;
    }
    
    // 读取向量维度
    int vector_size;
    if (fscanf(file, "%d", &vector_size) != 1) {
        printf("Error reading vector size from RHS file\n");
        fclose(file);
        return NULL;
    }
    
    // 检查维度是否匹配
    if (expected_size != -1 && vector_size != expected_size) {
        printf("Warning: RHS vector size (%d) does not match expected size (%d)\n", 
               vector_size, expected_size);
    }
    
    // 创建向量
    jxf_Vector* rhs = jxf_SeqVectorCreate(vector_size);
    jxf_SeqVectorInitialize(rhs);
    JXF_Real* rhs_data = jxf_VectorData(rhs);
    
    // 初始化向量为0
    for (int i = 0; i < vector_size; i++) {
        // rhs_data[i] = 0.0;
        fscanf(file, "%lf", &rhs_data[i]);
    }
    
    // 读取索引和数值
    // int index;
    // double value;
    // int count = 0;
    // while (fscanf(file, "%d %lf", &index, &value) == 2) {
    //     if (index >= 0 && index < vector_size) {
    //         rhs_data[index] = value;
    //         count++;
    //     } else {
    //         printf("Warning: Index %d out of range [0, %d)\n", index, vector_size);
    //     }
    // }
    
    // printf("Read %d non-zero entries from RHS file (total size: %d)\n", 
    //        count, vector_size);
    
    fclose(file);
    return rhs;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int myid, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    jxf_Vector       *b_Ser;
    jxf_Vector       *x_Ser;

    jxf_ParVector     *par_rhs;
    jxf_ParVector     *par_sol;
    
    if (argc < 3) {
        if (myid == 0) {
            printf("Usage: %s <matrix_file> <matrix_type> [binary] [rhs_file]\n", argv[0]);
            printf("  matrix_type: 0=CSR, 1=BSR\n");
            printf("  binary: 0=text, 1=binary (default: 0)\n");
            printf("  rhs_file: right-hand side file (optional)\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    char* filename = argv[1];
    int matrix_type = atoi(argv[2]);
    int binary = (argc > 3) ? atoi(argv[3]) : 0;
    char* rhs_file = (argc > 4) ? argv[4] : NULL;
    char* filename_output = (argc > 5) ? argv[5] : NULL;
    
    if (matrix_type == 1) {
        // 测试BSR矩阵的CPR-GMRES
        if (myid == 0) {
            printf("\n=================================================\n");
            printf("Testing CPR-GMRES for BSR Matrix\n");
            printf("Matrix file: %s\n", filename);
            printf("Matrix type: BSR\n");
            printf("Binary format: %s\n", binary ? "yes" : "no");
            if (rhs_file) printf("RHS file: %s\n", rhs_file);
            printf("=================================================\n\n");
        }
        
        // ------------------------------------------------------------
        // 1. 读取并分发BSR矩阵
        // ------------------------------------------------------------
        jxf_BSRMatrix* A_bsr = NULL;
        if (myid == 0) {
            if (binary) {
                A_bsr = jxf_BSRMatrixRead_Binary(filename);
            } else {
                A_bsr = jxf_BSRMatrixRead(filename);
            }
            
            if (!A_bsr) {
                printf("Error reading BSR matrix from %s\n", filename);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            printf("BSR Matrix Info (myid 0):\n");
            printf("  Block size: %d\n", jxf_BSRMatrixBlockSize(A_bsr));
            printf("  Number of block rows: %d\n", jxf_BSRMatrixNumRows(A_bsr));
            printf("  Number of block cols: %d\n", jxf_BSRMatrixNumCols(A_bsr));
            printf("  Number of block nonzeros: %d\n", jxf_BSRMatrixNumNonzeros(A_bsr));
            
            int blk_size = jxf_BSRMatrixBlockSize(A_bsr);
            int nrows = jxf_BSRMatrixNumRows(A_bsr);
            int ncols = jxf_BSRMatrixNumCols(A_bsr);
            int Scalar_nrows = nrows * blk_size;
            printf("  Scalar dimensions: %d x %d\n", nrows * blk_size, ncols * blk_size);

            // 将矩阵保存成二进制格式
            // jxf_BSRMatrixPrint_Binary(A_bsr,filename_output);
            
            if (rhs_file) {
                // 从文件读取右端项
                printf("Reading RHS from file: %s\n", rhs_file);
                b_Ser = read_rhs_file(rhs_file, Scalar_nrows);
                if (!b_Ser) {
                    printf("Error reading RHS file, using default RHS (all ones)\n");
                    // 如果读取失败，使用默认的1向量
                    b_Ser = jxf_SeqVectorCreate(Scalar_nrows);
                    jxf_SeqVectorInitialize(b_Ser);
                    JXF_Real *b_data = jxf_VectorData(b_Ser);
                    for (int i = 0; i < Scalar_nrows; i++) {
                        b_data[i] = 1.0;
                    }
                }
            } else {
                // 使用默认的1向量
                b_Ser = jxf_SeqVectorCreate(Scalar_nrows);
                jxf_SeqVectorInitialize(b_Ser);
                JXF_Real *b_data = jxf_VectorData(b_Ser);
                for (int i = 0; i < Scalar_nrows; i++) {
                    b_data[i] = 1.0;
                }
                printf("Using default RHS (all ones)\n");
            }
            
            // 创建初始解向量（全零）
            x_Ser = jxf_SeqVectorCreate(Scalar_nrows);
            jxf_SeqVectorInitialize(x_Ser);
            JXF_Real *x_data = jxf_VectorData(x_Ser);
            for (int i = 0; i < Scalar_nrows; i++) {
                x_data[i] = 0.0;
            }
        }
        
            // ------------------------------------------------------------
            // 2. 在串行数据上进行解耦
            // ------------------------------------------------------------
        if (myid == 0){

            printf("\nApplying decoupling to serial BSR system...\n");
            
            // 选择解耦方法
            JXF_DecoupType decoup_type = JXF_DECOUP_ABF;  // 推荐使用ABF
            // JXF_DecoupType decoup_type = JXF_DECOUP_ANL;  // 推荐使用ABF
            JXF_Int is_thermal = 0;  // 假设非热力模型
            
            printf("  Decoupling method: %s\n", jxf_DecoupTypeToString(decoup_type));
            printf("  Is thermal model: %s\n", is_thermal ? "yes" : "no");
            
            // 备份原始矩阵（如果需要保留原始数据）
            // jxf_BSRMatrix* A_bsr_orig = jxf_BSRMatrixDuplicate(A_bsr);
            // jxf_Vector* b_Ser_orig = jxf_SeqVectorDuplicate(b_Ser);
            
            // 执行解耦（同时修改矩阵和右端项）
            double decoup_start = MPI_Wtime();
            JXF_Int decoup_result = jxf_BSRMatrixDecouple(A_bsr, b_Ser, decoup_type, is_thermal);
            double decoup_end = MPI_Wtime();
            
            if (decoup_result != JXF_SUCCESS) {
                printf("Error applying decoupling to serial system\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            printf("  Decoupling completed in %.6f seconds\n", decoup_end - decoup_start);
            printf("  Matrix and RHS have been modified by decoupling\n");
            
            // 可选：验证解耦效果
            // 可以检查解耦后的对角块是否接近单位阵
            if (1) {
                int nb = jxf_BSRMatrixBlockSize(A_bsr);
                JXF_Real* data = jxf_BSRMatrixData(A_bsr);
                JXF_Int* rpt = jxf_BSRMatrixI(A_bsr);
                int nblocks = jxf_BSRMatrixNumRows(A_bsr);
                
                // 检查第一个对角块
                if (nblocks > 0) {
                    JXF_Real* first_diag_block = data;
                    printf("  First diagonal block after decoupling:\n");
                    for (int i = 0; i < nb; i++) {
                        for (int j = 0; j < nb; j++) {
                            printf("    %12.6e", first_diag_block[i*nb + j]);
                        }
                        printf("\n");
                    }
                }
            }
        
            // // 广播是否成功读取和解耦的标志
            // int read_success = (myid == 0) ? 1 : 0;
            // MPI_Bcast(&read_success, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            // if (!read_success) {
            //     if (myid == 0) printf("Error during matrix reading/decoupling\n");
            //     MPI_Finalize();
            //     return 1;
            // }
        }

        // 生成分区
        JXF_BigInt* global_row_start = NULL;
        JXF_BigInt* global_col_start = NULL;
        if (myid == 0) {
            jxf_GeneratePartitioning(jxf_BSRMatrixNumRows(A_bsr), size, &global_row_start);
            jxf_GeneratePartitioning(jxf_BSRMatrixNumCols(A_bsr), size, &global_col_start);
        }
        
        // 分发矩阵到各进程
        jxf_ParBSRMatrix* A_parbsr = jxf_BSRMatrixToParBSRMatrix(
            MPI_COMM_WORLD, A_bsr, global_row_start, global_col_start);
        
        // jxf_ParBSRMatrixReorder(A_parbsr);
        // jxf_ParBSRMatrixSetNumNonzeros(A_parbsr);
        
        // 清理串行矩阵内存
        // if (myid == 0) {
        //     jxf_BSRMatrixDestroy(A_bsr);
        // }
        
        if (!A_parbsr) {
            if (myid == 0) printf("Error creating parallel BSR matrix\n");
            MPI_Finalize();
            return 1;
        }
        
        // 获取矩阵信息
        int blk_size = jxf_ParBSRMatrixBlockSize(A_parbsr);
        int local_nrows = jxf_ParBSRMatrixNumRows(A_parbsr);
        int global_nrows = jxf_ParBSRMatrixGlobalNumRows(A_parbsr);
        int local_scalar_size = local_nrows * blk_size;
        int global_scalar_size = global_nrows * blk_size;
        
        // if (myid == 0) {
        //     printf("\nParallel BSR Matrix Info:\n");
        //     printf("  Block size: %d\n", blk_size);
        //     printf("  Global block rows: %d\n", global_nrows);
        //     printf("  Global scalar rows: %d\n", global_scalar_size);
        //     printf("\n");
        // }
        
        // ------------------------------------------------------------
        // 2. 创建右端项和初始解向量
        // ------------------------------------------------------------
        // 创建分区数组（标量行分区）
        JXF_Int  *partitioning;
        jxf_ParBSRMatrixGetRowPartitioning(A_parbsr, &partitioning);
        

        // 创建并行向量
        par_rhs = jxf_VectorToParVector(jxf_ParBSRMatrixComm(A_parbsr), b_Ser, partitioning);
        par_sol = jxf_VectorToParVector(jxf_ParBSRMatrixComm(A_parbsr), x_Ser, partitioning);
        
        if (!par_rhs || !par_sol) {
            printf("Rank %d: ERROR: Failed to create parallel vectors!\n", myid);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // ------------------------------------------------------------
        // 3. 创建CPR预条件器
        // ------------------------------------------------------------
        if (myid == 0) {
            printf("\nCreating CPR preconditioner...\n");
        }
        
        jxf_CPRPrecond* cpr = JXF_CPRCreate(MPI_COMM_WORLD);
        
        // 设置CPR参数
        JXF_CPRSetParameter(cpr, "pressure_index", 0);    // 压力变量索引
        JXF_CPRSetParameter(cpr, "block_size", blk_size);
        JXF_CPRSetParameter(cpr, "stage1_maxit", 1);      // 阶段1迭代次数
        JXF_CPRSetParameter(cpr, "stage2_maxit", 3);      // 阶段2迭代次数
        JXF_CPRSetParameter(cpr, "stage1_solver_type", 1); // AMG求解器
        JXF_CPRSetParameter(cpr, "stage2_solver_type", 2); // BGS求解器
        JXF_CPRSetParameter(cpr, "print_level", 1); // 打印等级
        
        JXF_CPRSetRealParameter(cpr, "threshold", 1e-15); // 提取压力矩阵阈值
        
        // 设置CPR
        double setup_start = MPI_Wtime();
        JXF_Int setup_result = JXF_CPRSetup(cpr, A_parbsr);
        double setup_end = MPI_Wtime();
        
        if (setup_result != JXF_SUCCESS) {
            if (myid == 0) printf("Error setting up CPR preconditioner\n");
            MPI_Finalize();
            return 1;
        }
           
        if (myid == 0) {
            printf("CPR setup completed in %.6f seconds\n", setup_end - setup_start);
        }
           
        // ------------------------------------------------------------
        // 4. 创建GMRES求解器并设置CPR为预条件器
        // ------------------------------------------------------------
        if (myid == 0) {
            printf("\nCreating GMRES solver with CPR preconditioner...\n");
        }
        fflush(stdout); 
        
        // GMRES参数
        int k_dim = 30;           // Krylov子空间维度
        int max_iter = 100;      // 最大迭代次数
        Real tol = 1e-4;          // 收敛容差
        int print_level = 1;      // 打印级别
        int is_check_restarted = 1; // 检查重启

        // 创建GMRES求解器
        JXF_Solver gmres_solver;
        JXF_ParBSRGMRESCreate(MPI_COMM_WORLD, &gmres_solver);

        // 设置GMRES参数
        JXF_GMRESSetKDim(gmres_solver, k_dim);
        JXF_GMRESSetIsCheckRestarted(gmres_solver, is_check_restarted);
        JXF_GMRESSetMaxIter(gmres_solver, max_iter);
        JXF_GMRESSetTol(gmres_solver, tol);
        JXF_GMRESSetLogging(gmres_solver, 1);
        JXF_GMRESSetPrintLevel(gmres_solver, print_level);

        // 设置CPR为预条件器
        JXF_GMRESSetPrecond(gmres_solver, 
                          (JXF_PtrToSolverFcn)JXF_CPRPrecond,
                          (JXF_PtrToSolverFcn)JXF_CPRSetup, 
                          cpr);

        // GMRES设置阶段
        double gmres_setup_start = MPI_Wtime();
        JXF_GMRESSetup(gmres_solver, (JXF_Matrix)A_parbsr, 
                      (JXF_Vector)par_rhs, (JXF_Vector)par_sol);
        double gmres_setup_end = MPI_Wtime();
        
        if (myid == 0) {
            printf("GMRES setup completed in %.6f seconds\n", 
                   gmres_setup_end - gmres_setup_start);
            fflush(stdout);  
        }
        
        // ------------------------------------------------------------
        // 5. 使用CPR-GMRES求解线性系统
        // ------------------------------------------------------------
        if (myid == 0) {
            printf("\nSolving linear system with CPR-GMRES...\n");
            printf("GMRES parameters:\n");
            printf("  Krylov subspace dimension: %d\n", k_dim);
            printf("  Maximum iterations: %d\n", max_iter);
            printf("  Tolerance: %.1e\n", tol);
            printf("\n");
            fflush(stdout);  
        }
        
        // 初始残差
        jxf_ParVector* residual = jxf_ParVectorCreate(
            MPI_COMM_WORLD, global_scalar_size, NULL);
        jxf_ParVectorInitialize(residual);

        jxf_ParVectorCopy(par_rhs, residual);
        jxf_ParBSRMatrixMatvec(-1.0, A_parbsr, par_sol, 1.0, residual);

        Real init_res_norm = jxf_ParVectorNorm2(residual);
        if (myid == 0) {
            printf("Initial residual norm: %.6e\n", init_res_norm);
        }

        // 求解
        double solve_start = MPI_Wtime();
        JXF_GMRESSolve(gmres_solver, (JXF_Matrix)A_parbsr,
                      (JXF_Matrix)A_parbsr, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);
        double solve_end = MPI_Wtime();

        // 最终残差
        jxf_ParVectorCopy(par_rhs, residual);
        jxf_ParBSRMatrixMatvec(-1.0, A_parbsr, par_sol, 1.0, residual);
        Real final_res_norm = jxf_ParVectorNorm2(residual);
        
        // 获取求解器统计信息
        JXF_Int num_iterations = 0;
        JXF_Real final_rel_res_norm = 0.0;
        
        JXF_GMRESGetNumIterations(gmres_solver, &num_iterations);
        JXF_GMRESGetFinalRelativeResidualNorm(gmres_solver, &final_rel_res_norm);
        
        if (myid == 0) {
            printf("\n=================================================\n");
            printf("CPR-GMRES Solution Results:\n");
            printf("=================================================\n");
            printf("  Number of iterations: %d\n", num_iterations);
            printf("  Final absolute residual norm: %.6e\n", final_res_norm);
            printf("  Final relative residual norm: %.6e\n", final_rel_res_norm);
            printf("  Initial residual norm: %.6e\n", init_res_norm);
            printf("  Reduction factor: %.6e\n", final_res_norm / init_res_norm);
            printf("  Solve time: %.6f seconds\n", solve_end - solve_start);
            printf("  Total time (setup + solve): %.6f seconds\n", 
                   (setup_end - setup_start) + (solve_end - solve_start));
            printf("=================================================\n\n");
        }
        
        // 获取CPR统计信息
        Real cpr_stats[4];
        JXF_CPRGetStatistics(cpr, cpr_stats);
        if (myid == 0) {
            printf("CPR Statistics:\n");
            printf("  Stage1 setup time: %.4f s\n", cpr_stats[0]);
            printf("  Stage1 solve time: %.4f s\n", cpr_stats[1]);
            printf("  Stage2 setup time: %.4f s\n", cpr_stats[2]);
            printf("  Stage2 solve time: %.4f s\n", cpr_stats[3]);
            printf("\n");
        }
        
        // 打印更详细的CPR信息
        JXF_CPRPrint(cpr, 2);
        
        // ------------------------------------------------------------
        // 6. 验证解的正确性（可选）
        // ------------------------------------------------------------
        if (myid == 0) {
            printf("\nVerifying solution...\n");
        }
        
        // 计算残差
        Real residual_norm = jxf_ParVectorNorm2(residual);
        Real rhs_norm = jxf_ParVectorNorm2(par_rhs);
        
        if (myid == 0) {
            printf("  Residual norm: %.6e\n", residual_norm);
            printf("  RHS norm:      %.6e\n", rhs_norm);
            printf("  Relative residual: %.6e\n", residual_norm / (rhs_norm + 1e-16));
            
            if (residual_norm / (rhs_norm + 1e-16) < tol) {
                printf("  ✓ Solution converged within tolerance\n");
            } else {
                printf("  ✗ Solution did not converge within tolerance\n");
            }
        }
        
        // ------------------------------------------------------------
        // 7. 清理内存
        // ------------------------------------------------------------
        // jxf_ParVectorDestroy(residual);
        // jxf_GMRESDestroy(gmres_solver);
        // JXF_CPRDestroy(&cpr);
        // jxf_ParVectorDestroy(par_rhs);
        // jxf_ParVectorDestroy(par_sol);
        
        // // // 清理串行向量
        // // if (myid == 0) {
        // //     if (b_Ser) jxf_VectorDestroy(b_Ser);
        // //     if (x_Ser) jxf_VectorDestroy(x_Ser);
        // // }
        
        // jxf_ParBSRMatrixDestroy(A_parbsr);
        
        // if (global_row_start) free(global_row_start);
        // if (global_col_start) free(global_col_start);
        
    } else {
        if (myid == 0) {
            printf("CSR matrix testing (not supported in this test)\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}

// 辅助函数：分发BSR矩阵
jxf_ParBSRMatrix* distribute_bsr_matrix(MPI_Comm comm, jxf_BSRMatrix* A_bsr, 
                                       JXF_BigInt* row_part, JXF_BigInt* col_part) {
    int myid;
    MPI_Comm_rank(comm, &myid);
    
    if (myid == 0 && !A_bsr) {
        return NULL;
    }
    
    return jxf_BSRMatrixToParBSRMatrix(comm, A_bsr, row_part, col_part);
}