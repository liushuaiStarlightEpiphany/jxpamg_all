//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jxf_par_bsr_assumed_part.c -- assumed partition operations for parallel BSR matrices.
 *  Date: 2025/10/08
 */ 

#include "jxf_parbsr_mv.h"

/*!
 * \fn JXF_Int jxf_ParBSRMatrixCreateAssumedPartition
 * \brief Create assumed partition for BSR matrix.
 * \date 2025/10/08
 */
// JXF_Int
// jxf_ParBSRMatrixCreateAssumedPartition( jxf_ParBSRMatrix *matrix )
// {
//     JXF_Int        global_num_cols;
//     JXF_Int        myid;
//     JXF_Int        col_start = 0, col_end = 0;

//     MPI_Comm      comm;
//     jxf_IJAssumedPart *apart;

//     global_num_cols = jxf_ParBSRMatrixGlobalNumCols(matrix);
//     comm            = jxf_ParBSRMatrixComm(matrix);

//     /* find out my actual range of rows and columns */
//     col_start = jxf_ParBSRMatrixFirstColDiag(matrix);
//     col_end   = jxf_ParBSRMatrixLastColDiag(matrix);

//     jxf_MPI_Comm_rank(comm, &myid);

//     /* allocate space */
//     apart = jxf_CTAlloc(jxf_IJAssumedPart, 1);

//     /* get my assumed partitioning - we want partitioning of the vector that
//        the matrix multiplies - so we use the col start and end */
//     jxf_GetAssumedPartitionRowRange(comm, myid, 0, global_num_cols, &(apart->row_start), &(apart->row_end));

//     /* allocate some space for the partition of the assumed partition */
//     apart->length = 0;
//     /* room for 10 owners of the assumed partition */
//     apart->storage_length = 10; /* need to be >=1 */
//     apart->proc_list      = jxf_CTAlloc(JXF_Int, apart->storage_length);
//     apart->row_start_list = jxf_CTAlloc(JXF_Int, apart->storage_length);
//     apart->row_end_list   = jxf_CTAlloc(JXF_Int, apart->storage_length);

//     /* now we want to reconcile our actual partition with the assumed partition */
//     jxf_LocateAssumedPartition(comm, col_start, col_end, 0, global_num_cols, apart, myid);

//     /* this partition will be saved in the matrix data structure until the
//        matrix is destroyed */
//     jxf_ParBSRMatrixAssumedPartition(matrix) = apart;

//     return jxf_error_flag;
// }

JXF_Int
jxf_ParBSRMatrixCreateAssumedPartition( jxf_ParBSRMatrix *matrix )
{
    JXF_Int        global_num_cols;
    JXF_Int        myid;
    JXF_Int        col_start = 0, col_end = 0;

    MPI_Comm      comm;
    jxf_IJAssumedPart *apart;

    // 获取MPI信息
    comm = jxf_ParBSRMatrixComm(matrix);
    jxf_MPI_Comm_rank(comm, &myid);
    
    // printf("Rank %d: [DEBUG] Starting jxf_ParBSRMatrixCreateAssumedPartition\n", myid);
    // printf("Rank %d: [DEBUG] matrix = %p\n", myid, (void*)matrix);
    // fflush(stdout);
    
    if (matrix == NULL) {
        printf("Rank %d: [ERROR] NULL matrix pointer!\n", myid);
        return jxf_error_flag;
    }

    global_num_cols = jxf_ParBSRMatrixGlobalNumCols(matrix);
    // printf("Rank %d: [DEBUG] global_num_cols = %lld\n", myid, (long long)global_num_cols);
    // fflush(stdout);

    /* find out my actual range of rows and columns */
    col_start = jxf_ParBSRMatrixFirstColDiag(matrix);
    col_end   = jxf_ParBSRMatrixLastColDiag(matrix);

    // printf("Rank %d: [DEBUG] col_start = %d, col_end = %d\n", myid, col_start, col_end);
    // fflush(stdout);

    /* allocate space */
    apart = jxf_CTAlloc(jxf_IJAssumedPart, 1);

    // // 分配内存后添加初始化代码
    // apart->proc_list = jxf_CTAlloc(JXF_Int, apart->storage_length);
    // apart->row_start_list = jxf_CTAlloc(JXF_Int, apart->storage_length);
    // apart->row_end_list = jxf_CTAlloc(JXF_Int, apart->storage_length);
    // apart->sort_index = NULL;

    // // 确保内存分配成功
    // if (apart->proc_list == NULL || apart->row_start_list == NULL || apart->row_end_list == NULL) {
    //     printf("Rank %d: [ERROR] Failed to allocate memory for assumed partition lists!\n", myid);
    //     jxf_TFree(apart->proc_list);
    //     jxf_TFree(apart->row_start_list);
    //     jxf_TFree(apart->row_end_list);
    //     jxf_TFree(apart);
    //     return jxf_error_flag;
    // }

    // // 初始化分配的内存
    // for (int i = 0; i < apart->storage_length; i++) {
    //     apart->proc_list[i] = -1;
    //     apart->row_start_list[i] = 0;
    //     apart->row_end_list[i] = 0;
    // }

    // if (apart == NULL) {
    //     printf("Rank %d: [ERROR] Failed to allocate memory for assumed partition!\n", myid);
    //     return jxf_error_flag;
    // }
    // printf("Rank %d: [DEBUG] Allocated apart at %p\n", myid, (void*)apart);
    // fflush(stdout);

    /* get my assumed partitioning - we want partitioning of the vector that
       the matrix multiplies - so we use the col start and end */
    printf("Rank %d: [DEBUG] Calling jxf_GetAssumedPartitionRowRange\n", myid);
    fflush(stdout);
    
    jxf_GetAssumedPartitionRowRange(comm, myid, 0, global_num_cols, &(apart->row_start), &(apart->row_end));
    
    printf("Rank %d: [DEBUG] After jxf_GetAssumedPartitionRowRange: row_start=%d, row_end=%d\n", 
           myid, apart->row_start, apart->row_end);
    fflush(stdout);

    /* allocate some space for the partition of the assumed partition */
    apart->length = 0;
    /* room for 10 owners of the assumed partition */
    apart->storage_length = 10; /* need to be >=1 */
    apart->proc_list      = jxf_CTAlloc(JXF_Int, apart->storage_length);
    apart->row_start_list = jxf_CTAlloc(JXF_Int, apart->storage_length);
    apart->row_end_list   = jxf_CTAlloc(JXF_Int, apart->storage_length);
    apart->sort_index     = NULL;

    // printf("Rank %d: [DEBUG] Allocated lists: proc_list=%p, row_start_list=%p, row_end_list=%p\n",
    //        myid, (void*)apart->proc_list, (void*)apart->row_start_list, (void*)apart->row_end_list);
    // fflush(stdout);
    
    // 检查分配是否成功
    if (apart->proc_list == NULL || apart->row_start_list == NULL || apart->row_end_list == NULL) {
        printf("Rank %d: [ERROR] Failed to allocate memory for assumed partition lists!\n", myid);
        jxf_TFree(apart->proc_list);
        jxf_TFree(apart->row_start_list);
        jxf_TFree(apart->row_end_list);
        jxf_TFree(apart);
        return jxf_error_flag;
    }

    /* now we want to reconcile our actual partition with the assumed partition */
    // printf("Rank %d: [DEBUG] Calling jxf_LocateAssumedPartition\n", myid);
    // fflush(stdout);
    
    jxf_LocateAssumedPartition(comm, col_start, col_end, 0, global_num_cols, apart, myid);
    
    // printf("Rank %d: [DEBUG] After jxf_LocateAssumedPartition\n", myid);
    // fflush(stdout);

    /* this partition will be saved in the matrix data structure until the
       matrix is destroyed */
    jxf_ParBSRMatrixAssumedPartition(matrix) = apart;
    
    // printf("Rank %d: [DEBUG] Assumed partition set to %p\n", myid, (void*)apart);
    // printf("Rank %d: [DEBUG] Assumed partition set to %p\n", myid, (void*)jxf_ParBSRMatrixAssumedPartition(matrix));
    // printf("Rank %d: [DEBUG] Finished jxf_ParBSRMatrixCreateAssumedPartition\n", myid);
    // fflush(stdout);

    return jxf_error_flag;
}

/*!
 * \fn JXF_Int jxf_ParBSRMatrixDestroyAssumedPartition
 * \brief Destroy assumed partition for BSR matrix.
 * \date 2025/10/08
 */
JXF_Int
jxf_ParBSRMatrixDestroyAssumedPartition( jxf_ParBSRMatrix *matrix )
{
    jxf_IJAssumedPart *apart;

    apart = jxf_ParBSRMatrixAssumedPartition(matrix);

    if (apart->storage_length > 0)
    {
        jxf_TFree(apart->proc_list);
        jxf_TFree(apart->row_start_list);
        jxf_TFree(apart->row_end_list);
        if (apart->sort_index)
        {
            jxf_TFree(apart->sort_index);
        }
    }

    jxf_TFree(apart);

    return jxf_error_flag;
}