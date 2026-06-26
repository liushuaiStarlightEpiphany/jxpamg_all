// test_bsr_mv.c - BSR矩阵向量乘法测试程序
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#include "jx_mv.h"
#include "jx_bsr_mv.h"
#include "jx_parbsr_mv.h"

typedef double Real;
typedef int Int;

// 函数声明
jx_BSRMatrix* read_bsr_matrix(const char* filename, int binary);
jx_ParBSRMatrix* distribute_bsr_matrix(MPI_Comm comm, jx_BSRMatrix* A_bsr);
void verify_matvec(jx_BSRMatrix* A, jx_Vector* x, jx_Vector* y);
void verify_par_matvec(jx_ParBSRMatrix* A, jx_ParVector* x, jx_ParVector* y);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int myid, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 3) {
        if (myid == 0) {
            printf("Usage: %s <matrix_file> <matrix_type> [binary]\n", argv[0]);
            printf("  matrix_type: 0=CSR, 1=BSR\n");
            printf("  binary: 0=text, 1=binary (default: 0)\n");
        }
        MPI_Finalize();
        return 1;
    }

    jx_Vector       *x_Ser;
    jx_Vector       *y_Ser;

    jx_ParVector     *par_x;
    jx_ParVector     *par_y;
    
    char* filename = argv[1];
    int matrix_type = atoi(argv[2]);
    int binary = (argc > 3) ? atoi(argv[3]) : 0;
    
    if (matrix_type == 1) {
        // 测试BSR矩阵
        if (myid == 0) {
            printf("\n========================================\n");
            printf("Testing BSR Matrix-Vector Multiplication\n");
            printf("Matrix file: %s\n", filename);
            printf("Matrix type: BSR\n");
            printf("Binary format: %s\n", binary ? "yes" : "no");
            printf("========================================\n\n");
        }
        
        // 读取BSR矩阵
        jx_BSRMatrix* A_bsr = NULL;
        if (myid == 0) {
            if(binary){
                A_bsr = jx_BSRMatrixRead_Binary(filename);
            }
            else{
                A_bsr = jx_BSRMatrixRead(filename);
            }
            if (!A_bsr) {
                printf("Error reading BSR matrix from %s\n", filename);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // printf("BSR Matrix Info (myid 0):\n");
            // printf("  Block size: %d\n", jx_BSRMatrixBlockSize(A_bsr));
            // printf("  Number of block rows: %d\n", jx_BSRMatrixNumRows(A_bsr));
            // printf("  Number of block cols: %d\n", jx_BSRMatrixNumCols(A_bsr));
            // printf("  Number of block nonzeros: %d\n", jx_BSRMatrixNumNonzeros(A_bsr));
            
            int blk_size = jx_BSRMatrixBlockSize(A_bsr);
            int nrows = jx_BSRMatrixNumRows(A_bsr);
            int ncols = jx_BSRMatrixNumCols(A_bsr);
            // printf("  Scalar dimensions: %d x %d\n", nrows * blk_size, ncols * blk_size);
        }

        if (myid == 0 && !A_bsr) {
            return NULL;
        }
        
        // 分发矩阵到各进程
        JX_BigInt* global_row_start = NULL;
        JX_BigInt* global_col_start = NULL;
        if (myid == 0) {
            jx_GeneratePartitioning(jx_BSRMatrixNumRows(A_bsr), size, &global_row_start);
            jx_GeneratePartitioning(jx_BSRMatrixNumCols(A_bsr), size, &global_col_start);
            // printf("Rank %d: Full scalar global_row_start array (length %d): ", myid, size + 1);
            // for (int i = 0; i <= size; i++) {
            //     printf("%d ", global_row_start[i]);
            // }
        }

        // 分发到各进程
        jx_ParBSRMatrix* A_parbsr = jx_BSRMatrixToParBSRMatrix(MPI_COMM_WORLD, A_bsr, global_row_start, global_col_start);
        // jx_ParBSRMatrix* A_parbsr = jx_BSRMatrixToParBSRMatrix(MPI_COMM_WORLD, A_bsr, NULL, NULL);
        
        // 检查并行矩阵创建是否成功
        if (!A_parbsr) {
            printf("Rank %d: ERROR: Failed to create parallel BSR matrix!\n", myid);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // 获取矩阵的块大小和全局维度
        int blk_size = jx_ParBSRMatrixBlockSize(A_parbsr);
        JX_BigInt global_block_rows = jx_ParBSRMatrixGlobalNumRows(A_parbsr);
        JX_BigInt global_scalar_rows = global_block_rows * blk_size;
        
        // printf("Rank %d: Block size: %d, Global block rows: %lld, Global scalar rows: %lld\n",
        //        myid, blk_size, (long long)global_block_rows, (long long)global_scalar_rows);
        
        // 创建分区数组（标量行分区）
        JX_Int  *partitioning;
        jx_ParBSRMatrixGetRowPartitioning(A_parbsr, &partitioning);
        
        // printf("Rank %d:  partitioning: [%lld, %lld]\n",
        //        myid, (long long)partitioning[0], (long long)partitioning[1]);
                       
        // printf("Rank %d: Scalar partitioning: [%lld, %lld]\n",
        //        myid, (long long)scalar_partitioning[0], (long long)scalar_partitioning[1]);
        
        // printf("Rank %d: Full scalar partitioning array (length %d): ", myid, size + 1);
        //     for (int i = 0; i <= size; i++) {
        //         printf("%d ", partitioning[i]);
        //     }
        if (myid ==0 ){
            x_Ser = jx_SeqVectorCreate(global_scalar_rows);
            jx_SeqVectorInitialize(x_Ser);
            // printf("Rank %d: after jx_SeqVectorCreate\n", myid);
            // fflush(stdout);
            JX_Real *x_data;
            x_data = jx_VectorData(x_Ser);
            for (int i = 0; i < global_scalar_rows; i++) {
                x_data[i] = 1.0;
            }
            // printf("Rank %d: Before jx_SeqVectorInitialize\n", myid);
            // fflush(stdout);
            
            // printf("Rank %d: Before jx_SeqVectorCreate\n", myid);
            // fflush(stdout);
            y_Ser = jx_SeqVectorCreate(global_scalar_rows);
            jx_SeqVectorInitialize(y_Ser);

            JX_Real *y_data;
            y_data = jx_VectorData(y_Ser);
            for (int i = 0; i < global_scalar_rows; i++) {
                y_data[i] = 0.0;
            }

            // printf("Rank %d: Before jx_VectorToParVector\n", myid);
            // fflush(stdout);
        }
        // 创建并行向量
        par_x = jx_VectorToParVector(jx_ParBSRMatrixComm(A_parbsr), x_Ser, partitioning);
        par_y = jx_VectorToParVector(jx_ParBSRMatrixComm(A_parbsr), y_Ser, partitioning);
        
        
        if (!par_x || !par_y) {
            printf("Rank %d: ERROR: Failed to create parallel vectors!\n", myid);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // printf("Rank %d: Parallel vectors created, par_x=%p, par_y=%p\n", 
        //        myid, (void*)par_x, (void*)par_y);
        // fflush(stdout);
        
        // 在Rank 0上创建参考向量用于验证
        jx_Vector* local_y_ref = NULL;
        if (myid == 0 && A_bsr) {
            // 创建参考向量（使用串行矩阵）
            local_y_ref = jx_SeqVectorCreate(jx_BSRMatrixNumRows(A_bsr) * blk_size);
            jx_SeqVectorInitialize(local_y_ref);
        }
        
        // 计算参考值（仅在rank 0上）
        if (myid == 0 && A_bsr) {
            // // 创建临时向量用于参考计算
            // jx_Vector* ref_x = jx_SeqVectorCreate(jx_BSRMatrixNumRows(A_bsr) * blk_size);
            // jx_SeqVectorInitialize(ref_x);
            
            // // 复制Rank 0的数据到参考向量
            // int ref_local_size = jx_VectorSize(local_x);
            // Real* ref_data = jx_VectorData(ref_x);
            // memcpy(ref_data, x_data, ref_local_size * sizeof(Real));
            
            // // 从其他进程收集数据
            // for (int src = 1; src < size; src++) {
            //     // 接收数据大小
            //     int recv_size;
            //     MPI_Recv(&recv_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
            //     // 接收数据并放入正确位置
            //     int offset = src * local_size;  // 简化假设：每个进程数据大小相同
            //     MPI_Recv(ref_data + offset, recv_size, MPI_DOUBLE, src, 1, 
            //             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // }
            
            // 计算参考值
            jx_BSRMatrixMatvec(1.0, A_bsr, x_Ser, 0.0, local_y_ref);
            printf("\nReference matvec completed on myid 0\n");
            
            // // 清理临时向量
            // jx_SeqVectorDestroy(ref_x);
        }
        // } else if (myid > 0) {
        //     // 发送数据给Rank 0用于参考计算
        //     MPI_Send(&local_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        //     MPI_Send(x_data, local_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        // }
        
        // 执行并行矩阵向量乘法
        // printf("Rank %d: Before matvec\n", myid);
        // fflush(stdout);
        
        double start_time = MPI_Wtime();
        jx_ParBSRMatrixMatvec(1.0, A_parbsr, par_x, 0.0, par_y);
        double end_time = MPI_Wtime();
        
        // printf("Rank %d: After matvec, time = %.6f seconds\n", myid, end_time - start_time);
        // fflush(stdout);
        
        if (myid == 0) {
            printf("Parallel BSR matvec completed in %.6f seconds\n", end_time - start_time);
        }
        
        // 验证结果
        if (myid == 0 && A_bsr) {
            // 收集并行计算结果
            Real* all_y = malloc(global_scalar_rows * sizeof(Real));
            
            // 复制Rank 0的结果
            jx_Vector* local_y = jx_ParVectorLocalVector(par_y);
            int local_scalar_size = jx_VectorSize(local_y);
            memcpy(all_y, jx_VectorData(local_y), local_scalar_size * sizeof(Real));
            
            // 接收其他进程的结果
            for (int src = 1; src < size; src++) {
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                int offset = src * local_scalar_size;  // 简化假设
                MPI_Recv(all_y + offset, recv_size, MPI_DOUBLE, src, 3, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            // 计算误差
            Real* ref_data = jx_VectorData(local_y_ref);
            Real error = 0.0;
            Real norm_ref = 0.0;
            for (JX_BigInt i = 0; i < global_scalar_rows; i++) {
                Real diff = all_y[i] - ref_data[i];
                error += diff * diff;
                norm_ref += ref_data[i] * ref_data[i];
            }
            error = sqrt(error);
            norm_ref = sqrt(norm_ref);
            
            printf("\nVerification Results:\n");
            printf("  Relative error: %.6e\n", error / (norm_ref + 1e-16));
            printf("  Absolute error: %.6e\n", error);
            printf("  Reference norm: %.6e\n", norm_ref);
            
            // free(all_y);
        } else if (myid > 0) {
            // 发送结果给Rank 0
            jx_Vector* local_y = jx_ParVectorLocalVector(par_y);
            int local_scalar_size = jx_VectorSize(local_y);
            MPI_Send(&local_scalar_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Send(jx_VectorData(local_y), local_scalar_size, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        }
        
        // // 清理内存
        // jx_ParVectorDestroy(par_x);
        // jx_ParVectorDestroy(par_y);
        // if (myid == 0 && A_bsr) {
        //     jx_BSRMatrixDestroy(A_bsr);
        //     // jx_SeqVectorDestroy(local_y_ref);
        // }
        // if (myid == 0) {
        //     free(global_row_start);
        //     free(global_col_start);
        // }
        
        // jx_ParBSRMatrixDestroy(A_parbsr);
        
    } else {
        if (myid == 0) {
            printf("CSR matrix testing (existing functionality)\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}

// 读取BSR矩阵
jx_BSRMatrix* read_bsr_matrix(const char* filename, int binary) {
    if (binary) {
        return jx_BSRMatrixRead_Binary(filename);
    } else {
        return jx_BSRMatrixRead(filename);
    }
}

// 分发BSR矩阵到各进程
jx_ParBSRMatrix* distribute_bsr_matrix(MPI_Comm comm, jx_BSRMatrix* A_bsr) {
    int myid;
    MPI_Comm_rank(comm, &myid);
    
    if (myid == 0 && !A_bsr) {
        return NULL;
    }
    
    // 使用jx_BSRMatrixToParBSRMatrix函数分发矩阵
    return jx_BSRMatrixToParBSRMatrix(comm, A_bsr, NULL, NULL);
}