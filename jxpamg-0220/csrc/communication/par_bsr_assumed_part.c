//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_par_bsr_assumed_part.c -- assumed partition operations for parallel BSR matrices.
 *  Date: 2025/10/08
 */ 

#include "jx_parbsr_mv.h"

/*!
 * \fn JX_Int jx_ParBSRMatrixCreateAssumedPartition
 * \brief Create assumed partition for BSR matrix.
 * \date 2025/10/08
 */
// JX_Int
// jx_ParBSRMatrixCreateAssumedPartition( jx_ParBSRMatrix *matrix )
// {
//     JX_Int        global_num_cols;
//     JX_Int        myid;
//     JX_Int        col_start = 0, col_end = 0;

//     MPI_Comm      comm;
//     jx_IJAssumedPart *apart;

//     global_num_cols = jx_ParBSRMatrixGlobalNumCols(matrix);
//     comm            = jx_ParBSRMatrixComm(matrix);

//     /* find out my actual range of rows and columns */
//     col_start = jx_ParBSRMatrixFirstColDiag(matrix);
//     col_end   = jx_ParBSRMatrixLastColDiag(matrix);

//     jx_MPI_Comm_rank(comm, &myid);

//     /* allocate space */
//     apart = jx_CTAlloc(jx_IJAssumedPart, 1);

//     /* get my assumed partitioning - we want partitioning of the vector that
//        the matrix multiplies - so we use the col start and end */
//     jx_GetAssumedPartitionRowRange(comm, myid, 0, global_num_cols, &(apart->row_start), &(apart->row_end));

//     /* allocate some space for the partition of the assumed partition */
//     apart->length = 0;
//     /* room for 10 owners of the assumed partition */
//     apart->storage_length = 10; /* need to be >=1 */
//     apart->proc_list      = jx_CTAlloc(JX_Int, apart->storage_length);
//     apart->row_start_list = jx_CTAlloc(JX_Int, apart->storage_length);
//     apart->row_end_list   = jx_CTAlloc(JX_Int, apart->storage_length);

//     /* now we want to reconcile our actual partition with the assumed partition */
//     jx_LocateAssumedPartition(comm, col_start, col_end, 0, global_num_cols, apart, myid);

//     /* this partition will be saved in the matrix data structure until the
//        matrix is destroyed */
//     jx_ParBSRMatrixAssumedPartition(matrix) = apart;

//     return jx_error_flag;
// }

JX_Int
jx_ParBSRMatrixCreateAssumedPartition( jx_ParBSRMatrix *matrix )
{
    JX_Int        global_num_cols;
    JX_Int        myid;
    JX_Int        col_start = 0, col_end = 0;

    MPI_Comm      comm;
    jx_IJAssumedPart *apart;

    // 获取MPI信息
    comm = jx_ParBSRMatrixComm(matrix);
    jx_MPI_Comm_rank(comm, &myid);
    
    // fflush(stdout);
    
    if (matrix == NULL) {
        printf("Rank %d: [ERROR] NULL matrix pointer!\n", myid);
        return jx_error_flag;
    }

    global_num_cols = jx_ParBSRMatrixGlobalNumCols(matrix);
    // fflush(stdout);

    /* find out my actual range of rows and columns */
    col_start = jx_ParBSRMatrixFirstColDiag(matrix);
    col_end   = jx_ParBSRMatrixLastColDiag(matrix);

    // fflush(stdout);

    /* allocate space */
    apart = jx_CTAlloc(jx_IJAssumedPart, 1);

    // // 分配内存后添加初始化代码
    // apart->proc_list = jx_CTAlloc(JX_Int, apart->storage_length);
    // apart->row_start_list = jx_CTAlloc(JX_Int, apart->storage_length);
    // apart->row_end_list = jx_CTAlloc(JX_Int, apart->storage_length);
    // apart->sort_index = NULL;

    // // 确保内存分配成功
    // if (apart->proc_list == NULL || apart->row_start_list == NULL || apart->row_end_list == NULL) {
    //     printf("Rank %d: [ERROR] Failed to allocate memory for assumed partition lists!\n", myid);
    //     jx_TFree(apart->proc_list);
    //     jx_TFree(apart->row_start_list);
    //     jx_TFree(apart->row_end_list);
    //     jx_TFree(apart);
    //     return jx_error_flag;
    // }

    // // 初始化分配的内存
    // for (int i = 0; i < apart->storage_length; i++) {
    //     apart->proc_list[i] = -1;
    //     apart->row_start_list[i] = 0;
    //     apart->row_end_list[i] = 0;
    // }

    // if (apart == NULL) {
    //     printf("Rank %d: [ERROR] Failed to allocate memory for assumed partition!\n", myid);
    //     return jx_error_flag;
    // }
    // fflush(stdout);

    /* get my assumed partitioning - we want partitioning of the vector that
       the matrix multiplies - so we use the col start and end */
    fflush(stdout);
    
    jx_GetAssumedPartitionRowRange(comm, myid, 0, global_num_cols, &(apart->row_start), &(apart->row_end));
    
    fflush(stdout);

    /* allocate some space for the partition of the assumed partition */
    apart->length = 0;
    /* room for 10 owners of the assumed partition */
    apart->storage_length = 10; /* need to be >=1 */
    apart->proc_list      = jx_CTAlloc(JX_Int, apart->storage_length);
    apart->row_start_list = jx_CTAlloc(JX_Int, apart->storage_length);
    apart->row_end_list   = jx_CTAlloc(JX_Int, apart->storage_length);
    apart->sort_index     = NULL;

    //        myid, (void*)apart->proc_list, (void*)apart->row_start_list, (void*)apart->row_end_list);
    // fflush(stdout);
    
    // 检查分配是否成功
    if (apart->proc_list == NULL || apart->row_start_list == NULL || apart->row_end_list == NULL) {
        printf("Rank %d: [ERROR] Failed to allocate memory for assumed partition lists!\n", myid);
        jx_TFree(apart->proc_list);
        jx_TFree(apart->row_start_list);
        jx_TFree(apart->row_end_list);
        jx_TFree(apart);
        return jx_error_flag;
    }

    /* now we want to reconcile our actual partition with the assumed partition */
    // fflush(stdout);
    
    jx_LocateAssumedPartition(comm, col_start, col_end, 0, global_num_cols, apart, myid);
    
    // fflush(stdout);

    /* this partition will be saved in the matrix data structure until the
       matrix is destroyed */
    jx_ParBSRMatrixAssumedPartition(matrix) = apart;
    
    // fflush(stdout);

    return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParBSRMatrixDestroyAssumedPartition
 * \brief Destroy assumed partition for BSR matrix.
 * \date 2025/10/08
 */
JX_Int
jx_ParBSRMatrixDestroyAssumedPartition( jx_ParBSRMatrix *matrix )
{
    jx_IJAssumedPart *apart;

    apart = jx_ParBSRMatrixAssumedPartition(matrix);

    if (apart->storage_length > 0)
    {
        jx_TFree(apart->proc_list);
        jx_TFree(apart->row_start_list);
        jx_TFree(apart->row_end_list);
        if (apart->sort_index)
        {
            jx_TFree(apart->sort_index);
        }
    }

    jx_TFree(apart);

    return jx_error_flag;
}