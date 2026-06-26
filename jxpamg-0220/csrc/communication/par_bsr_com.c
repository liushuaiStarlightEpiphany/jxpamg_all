//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_par_bsr_com.c -- communication operations for parallel BSR matrices and vectors.
 *  Date: 2025/10/08
 */ 

#include "jx_mv.h"
#include <stddef.h>

/*!
 * \fn jx_ParCSRCommHandle *jx_ParBSRCommHandleCreate
 * \brief Sets up a communication handle for BSR matrices, posts receives and initiates sends.
 * \note There are different options for job:
 * job = 1 : is used to initialize communication exchange for the parts
 *           of vector needed to perform a Matvec, requires send_data 
 *           and recv_data to be doubles, recv_vec_starts and 
 *           send_map_starts need to be set in comm_pkg.
 * job = 2 : is used to initialize communication exchange for the parts
 *           of vector needed to perform a MatvecT, requires send_data 
 *           and recv_data to be doubles, recv_vec_starts and 
 *           send_map_starts need to be set in comm_pkg.
 * \date 2025/10/08
 */
jx_ParCSRCommHandle *
jx_ParBSRCommHandleCreate( JX_Int                job,
                           JX_Int                bnnz,
                           jx_ParCSRCommPkg *comm_pkg,
                           void               *send_data, 
                           void               *recv_data )
{
    JX_Int                  num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
    JX_Int                  num_recvs = jx_ParCSRCommPkgNumRecvs(comm_pkg);
    MPI_Comm                comm      = jx_ParCSRCommPkgComm(comm_pkg);

    jx_ParCSRCommHandle *comm_handle;
    JX_Int                  num_requests;
    MPI_Request         *requests;
    JX_Int                  i, j, my_id, num_procs, ip, vec_start, vec_len;
    JX_Real            *d_send_data = (JX_Real *)send_data;
    JX_Real            *d_recv_data = (JX_Real *)recv_data;

    num_requests = num_sends + num_recvs;
    requests = jx_CTAlloc(MPI_Request, num_requests);

    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);

    j = 0;

    switch (job)
    {
        case 1:
        {
            for (i = 0; i < num_recvs; i ++)
            {
                ip        = jx_ParCSRCommPkgRecvProc(comm_pkg, i); 
                vec_start = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i);
                vec_len   = (jx_ParCSRCommPkgRecvVecStart(comm_pkg,i+1) - vec_start) * bnnz;
                jx_MPI_Irecv(&d_recv_data[vec_start * bnnz], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
            }
            for (i = 0; i < num_sends; i ++)
            {
                vec_start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
                vec_len   = (jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - vec_start) * bnnz;
                ip        = jx_ParCSRCommPkgSendProc(comm_pkg, i);
                jx_MPI_Isend(&d_send_data[vec_start * bnnz], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
            }
            break;
        }
        case 2:
        {
            for (i = 0; i < num_sends; i ++)
            {
                vec_start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
                vec_len   = (jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - vec_start) * bnnz;
                ip        = jx_ParCSRCommPkgSendProc(comm_pkg, i);
                jx_MPI_Irecv(&d_recv_data[vec_start * bnnz], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
            }
            for (i = 0; i < num_recvs; i ++)
            {
                ip        = jx_ParCSRCommPkgRecvProc(comm_pkg, i); 
                vec_start = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i);
                vec_len   = (jx_ParCSRCommPkgRecvVecStart(comm_pkg,i+1) - vec_start) * bnnz;
                jx_MPI_Isend(&d_send_data[vec_start * bnnz], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
            }
            break;
        }
    }

    /* set up comm_handle and return */
    comm_handle = jx_CTAlloc(jx_ParCSRCommHandle, 1);

    // printf("Rank %d: Allocated comm_handle at %p\n", my_id, (void*)comm_handle);
    // printf("Rank %d: Size of jx_ParCSRCommHandle: %zu bytes\n", my_id, sizeof(jx_ParCSRCommHandle));
    // printf("Rank %d: Field offsets:\n", my_id);
    // printf("  comm_pkg offset: %zu\n", offsetof(jx_ParCSRCommHandle, comm_pkg));
    // printf("  send_data offset: %zu\n", offsetof(jx_ParCSRCommHandle, send_data));
    // printf("  recv_data offset: %zu\n", offsetof(jx_ParCSRCommHandle, recv_data));
    // printf("  num_requests offset: %zu\n", offsetof(jx_ParCSRCommHandle, num_requests));
    // printf("  requests offset: %zu\n", offsetof(jx_ParCSRCommHandle, requests));
    // fflush(stdout);
    
    // 使用 memset 清零，确保所有字段初始化为 0
    memset(comm_handle, 0, sizeof(jx_ParCSRCommHandle));

    jx_ParCSRCommHandleCommPkg(comm_handle)     = comm_pkg;
    jx_ParCSRCommHandleSendData(comm_handle)    = send_data;
    jx_ParCSRCommHandleRecvData(comm_handle)    = recv_data;
    jx_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
    jx_ParCSRCommHandleRequests(comm_handle)    = requests;

    // // 验证设置
    // printf("Rank %d: Set comm_handle fields:\n", my_id);
    // printf("  comm_handle = %p\n", (void*)comm_handle);
    // printf("  comm_pkg = %p\n", (void*)comm_handle->comm_pkg);
    // printf("  send_data = %p\n", (void*)comm_handle->send_data);
    // printf("  recv_data = %p\n", (void*)comm_handle->recv_data);
    // printf("  num_requests = %d\n", comm_handle->num_requests);
    // printf("  requests = %p\n", (void*)comm_handle->requests);
    // fflush(stdout);

    return (comm_handle);
}

/*!
 * \fn JX_Int jx_ParBSRCommHandleDestroy 
 * \date 2025/10/08
 */
JX_Int
jx_ParBSRCommHandleDestroy( jx_ParCSRCommHandle *comm_handle )
{
    int my_id; MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    fflush(stderr);

    if (comm_handle == NULL) 
    {
        fflush(stderr);
        return jx_error_flag;
    }
    
    fflush(stderr);
    
    MPI_Status   *status   = NULL;
    MPI_Request  *requests = jx_ParCSRCommHandleRequests(comm_handle);
    JX_Int num_requests = jx_ParCSRCommHandleNumRequests(comm_handle);
    
    if (num_requests)
    {
        fflush(stderr);
        status = jx_CTAlloc(MPI_Status, num_requests);
        jx_MPI_Waitall(num_requests, requests, status);
        jx_TFree(status);
        fflush(stderr);
    }

    jx_TFree(requests);
    jx_TFree(comm_handle);

    return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GetCommPkgBlockRTFromCommPkgBlockA
 * \brief Get comm package for block RT matrix from comm package of block A matrix.
 * \date 2025/10/08
 */
JX_Int
jx_GetCommPkgBlockRTFromCommPkgBlockA( jx_ParBSRMatrix *RT,
                                       jx_ParBSRMatrix *A,
                                       JX_Int          *tmp_map_offd,
                                       JX_Int          *fine_to_coarse_offd )
{
    MPI_Comm              comm              = jx_ParBSRMatrixComm(RT);
    jx_ParCSRCommPkg      *comm_pkg_A       = jx_ParBSRMatrixCommPkg(A);
    JX_Int                num_recvs_A       = jx_ParCSRCommPkgNumRecvs(comm_pkg_A);
    JX_Int                *recv_procs_A     = jx_ParCSRCommPkgRecvProcs(comm_pkg_A);
    JX_Int                *recv_vec_starts_A = jx_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
    JX_Int                num_sends_A       = jx_ParCSRCommPkgNumSends(comm_pkg_A);
    JX_Int                *send_procs_A     = jx_ParCSRCommPkgSendProcs(comm_pkg_A);

    jx_ParCSRCommPkg      *comm_pkg = NULL;
    JX_Int                num_recvs_RT;
    JX_Int                *recv_procs_RT;
    JX_Int                *recv_vec_starts_RT;
    JX_Int                num_sends_RT;
    JX_Int                *send_procs_RT;
    JX_Int                *send_map_starts_RT;
    JX_Int                *send_map_elmts_RT;
    JX_Int                *send_elmts = NULL;

    JX_Int                *col_map_offd_RT  = jx_ParBSRMatrixColMapOffd(RT);
    JX_Int                num_cols_offd_RT  = jx_BSRMatrixNumCols(jx_ParBSRMatrixOffd(RT));
    JX_Int                first_col_diag    = jx_ParBSRMatrixFirstColDiag(RT);

    JX_Int                i, j;
    JX_Int                vec_len, vec_start;
    JX_Int                num_procs, my_id;
    JX_Int                num_requests;
    JX_Int                offd_col, proc_num;

    JX_Int                *proc_mark;
    JX_Int                *change_array;

    MPI_Request           *requests;
    MPI_Status            *status;

    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);

    /* determine num_recvs, recv_procs and recv_vec_starts for RT */
    proc_mark = jx_CTAlloc(JX_Int, num_recvs_A);

    for (i = 0; i < num_recvs_A; i++)
    {
        proc_mark[i] = 0;
    }

    proc_num     = 0;
    num_recvs_RT = 0;
    if (num_cols_offd_RT)
    {
        for (i = 0; i < num_recvs_A; i++)
        {
            for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i + 1]; j++)
            {
                offd_col = tmp_map_offd[proc_num];
                if (offd_col == j)
                {
                    proc_mark[i]++;
                    proc_num++;
                    if (proc_num == num_cols_offd_RT)
                    {
                        break;
                    }
                }
            }
            if (proc_mark[i])
            {
                num_recvs_RT++;
            }
            if (proc_num == num_cols_offd_RT)
            {
                break;
            }
        }
    }

    for (i = 0; i < num_cols_offd_RT; i++)
    {
        col_map_offd_RT[i] = fine_to_coarse_offd[tmp_map_offd[i]];
    }

    recv_procs_RT      = jx_CTAlloc(JX_Int, num_recvs_RT);
    recv_vec_starts_RT = jx_CTAlloc(JX_Int, num_recvs_RT + 1);

    j                     = 0;
    recv_vec_starts_RT[0] = 0;
    for (i = 0; i < num_recvs_A; i++)
        if (proc_mark[i])
        {
            recv_procs_RT[j]          = recv_procs_A[i];
            recv_vec_starts_RT[j + 1] = recv_vec_starts_RT[j] + proc_mark[i];
            j++;
        }

    /* send num_changes to recv_procs_A and receive change_array from send_procs_A */
    num_requests = num_recvs_A + num_sends_A;
    requests     = jx_CTAlloc(MPI_Request, num_requests);
    status       = jx_CTAlloc(MPI_Status, num_requests);

    change_array = jx_CTAlloc(JX_Int, num_sends_A);

    j = 0;
    for (i = 0; i < num_sends_A; i++)
        jx_MPI_Irecv(&change_array[i], 1, JX_MPI_INT, send_procs_A[i], 0, comm, &requests[j++]);

    for (i = 0; i < num_recvs_A; i++)
        jx_MPI_Isend(&proc_mark[i], 1, JX_MPI_INT, recv_procs_A[i], 0, comm, &requests[j++]);

    jx_MPI_Waitall(num_requests, requests, status);

    jx_TFree(proc_mark);

    /* if change_array[i] is 0, omit send_procs_A[i] in send_procs_RT */
    num_sends_RT = 0;
    for (i = 0; i < num_sends_A; i++)
        if (change_array[i])
        {
            num_sends_RT++;
        }

    send_procs_RT      = jx_CTAlloc(JX_Int, num_sends_RT);
    send_map_starts_RT = jx_CTAlloc(JX_Int, num_sends_RT + 1);

    j                     = 0;
    send_map_starts_RT[0] = 0;
    for (i = 0; i < num_sends_A; i++)
        if (change_array[i])
        {
            send_procs_RT[j]          = send_procs_A[i];
            send_map_starts_RT[j + 1] = send_map_starts_RT[j] + change_array[i];
            j++;
        }

    /* generate send_map_elmts */
    send_elmts         = jx_CTAlloc(JX_Int, send_map_starts_RT[num_sends_RT]);
    send_map_elmts_RT  = jx_CTAlloc(JX_Int, send_map_starts_RT[num_sends_RT]);

    j = 0;
    for (i = 0; i < num_sends_RT; i++)
    {
        vec_start = send_map_starts_RT[i];
        vec_len   = send_map_starts_RT[i + 1] - vec_start;
        jx_MPI_Irecv(&send_elmts[vec_start], vec_len, JX_MPI_INT, send_procs_RT[i], 0, comm, &requests[j++]);
    }

    for (i = 0; i < num_recvs_RT; i++)
    {
        vec_start = recv_vec_starts_RT[i];
        vec_len   = recv_vec_starts_RT[i + 1] - vec_start;
        jx_MPI_Isend(&col_map_offd_RT[vec_start], vec_len, JX_MPI_INT, recv_procs_RT[i], 0, comm, &requests[j++]);
    }

    jx_MPI_Waitall(j, requests, status);

    for (i = 0; i < send_map_starts_RT[num_sends_RT]; i++)
    {
        send_map_elmts_RT[i] = send_elmts[i] - first_col_diag;
    }

    /* Create communication package */
    comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);
    jx_ParCSRCommPkgComm(comm_pkg) = comm;
    jx_ParCSRCommPkgNumRecvs(comm_pkg)      = num_recvs_RT;
    jx_ParCSRCommPkgRecvProcs(comm_pkg)     = recv_procs_RT;
    jx_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts_RT;
    jx_ParCSRCommPkgNumSends(comm_pkg)      = num_sends_RT;
    jx_ParCSRCommPkgSendProcs(comm_pkg)     = send_procs_RT;
    jx_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts_RT;
    jx_ParCSRCommPkgSendMapElmts(comm_pkg)  = send_map_elmts_RT;

    jx_ParBSRMatrixCommPkg(RT) = comm_pkg;

    /* Free memory */
    jx_TFree(status);
    jx_TFree(requests);
    jx_TFree(send_elmts);
    jx_TFree(change_array);

    return jx_error_flag;
}


/*!
 * \fn JX_Int jx_BlockMatvecCommPkgCreate
 * \brief Generates the comm_pkg for block matrix A.
 *        If no row and / or column partitioning is given, 
 *        the routine determines them with MPE_Decomp1d.
 * \date 2025/10/09
 */
JX_Int
jx_BlockMatvecCommPkgCreate(jx_ParBSRMatrix *par_A)
{
    // 获取MPI信息
    int myid;
    MPI_Comm comm = jx_ParBSRMatrixComm(par_A);
    jx_MPI_Comm_rank(comm, &myid);
    
    // printf("Rank %d: ===== ENTERING jx_BlockMatvecCommPkgCreate =====\n", myid);
    // printf("Rank %d: Matrix pointer: %p\n", myid, (void*)par_A);
    // fflush(stdout);
    
    JX_Int num_sends;
    JX_Int *send_procs;
    JX_Int *send_map_starts;
    JX_Int *send_map_elmts;
 
    JX_Int num_recvs;
    JX_Int *recv_procs;
    JX_Int *recv_vec_starts;
   
    jx_ParCSRCommPkg *comm_pkg;

    JX_Int first_col_diag = jx_ParBSRMatrixFirstColDiag(par_A);
    JX_Int *col_map_offd = jx_ParBSRMatrixColMapOffd(par_A);
    JX_Int num_cols_offd = jx_BSRMatrixNumCols(jx_ParBSRMatrixOffd(par_A));
    
    // printf("Rank %d: Matrix information:\n", myid);
    // printf("Rank %d:   first_col_diag = %d\n", myid, first_col_diag);
    // printf("Rank %d:   col_map_offd = %p\n", myid, (void*)col_map_offd);
    // printf("Rank %d:   num_cols_offd = %d\n", myid, num_cols_offd);
    // if (col_map_offd && num_cols_offd > 0) {
    //     printf("Rank %d:   First 5 col_map_offd values:\n", myid);
    //     for (int i = 0; i < 5 && i < num_cols_offd; i++) {
    //         printf("Rank %d:     col_map_offd[%d] = %d\n", myid, i, col_map_offd[i]);
    //     }
    // }
    // fflush(stdout);

#if JX_NO_GLOBAL_PARTITION
    printf("Rank %d: Using NO_GLOBAL_PARTITION mode (assumed partition)\n", myid);
    
    JX_Int row_start = 0, row_end = 0;
    JX_Int col_start = 0, col_end = 0;
    JX_Int global_num_cols;
    
    jx_IJAssumedPart *apart;
    
    /* get parbsr_A information */
    // 使用矩阵已有的本地范围信息
    col_start = jx_ParBSRMatrixFirstColDiag(par_A);
    col_end = jx_ParBSRMatrixLastColDiag(par_A);
    
    global_num_cols = jx_ParBSRMatrixGlobalNumCols(par_A);
    
    printf("Rank %d: Local column range: [%d, %d]\n", myid, col_start, col_end);
    printf("Rank %d: Global number of columns: %d\n", myid, global_num_cols);
    fflush(stdout);

    /* Create the assumed partition */
    if (jx_ParBSRMatrixAssumedPartition(par_A) == NULL)
    {
        printf("Rank %d: Creating assumed partition\n", myid);
        fflush(stdout);
        
        // 直接调用创建假设分区函数
        jx_ParBSRMatrixCreateAssumedPartition(par_A);
        
        // 检查创建是否成功
        printf("Rank %d: After creation: matrix->assumed_partition = %p\n", 
               myid, (void*)par_A->assumed_partition);
        fflush(stdout);
    }
    
    apart = jx_ParBSRMatrixAssumedPartition(par_A);
    printf("Rank %d: Assumed partition pointer: %p\n", myid, (void*)apart);
    
    // 关键检查：如果 apart 是 NULL，尝试使用全局分区方法作为后备
    if (apart == NULL) {
        printf("Rank %d: WARNING: Assumed partition is NULL!\n", myid);
        printf("Rank %d: Trying to use global partition as fallback...\n", myid);
        
        JX_Int *col_starts = jx_ParBSRMatrixColStarts(par_A);
        JX_Int num_cols_diag = jx_BSRMatrixNumCols(jx_ParBSRMatrixDiag(par_A));
        
        if (col_starts != NULL) {
            printf("Rank %d: Using global partition fallback\n", myid);
            jx_MatvecCommPkgCreate_core(comm, col_map_offd, first_col_diag, col_starts,
                                      num_cols_diag, num_cols_offd,
                                      first_col_diag, col_map_offd, 1,
                                      &num_recvs, &recv_procs, &recv_vec_starts,
                                      &num_sends, &send_procs, &send_map_starts, &send_map_elmts);
        } else {
            printf("Rank %d: ERROR: Cannot create communication package - no partition available!\n", myid);
            return jx_error_flag;
        }
    } else {
        printf("Rank %d: Using assumed partition method\n", myid);
        fflush(stdout);
        
        /* get commpkg info information using assumed partition */
        jx_NewCommPkgCreate_core(comm, col_map_offd, first_col_diag, 
                                col_start, col_end, 
                                num_cols_offd, global_num_cols,
                                &num_recvs, &recv_procs, &recv_vec_starts,
                                &num_sends, &send_procs, &send_map_starts, 
                                &send_map_elmts, apart);
    }
   
#else
    // printf("Rank %d: Using GLOBAL_PARTITION mode\n", myid);
   
    JX_Int *col_starts = jx_ParBSRMatrixColStarts(par_A);
    JX_Int num_cols_diag = jx_BSRMatrixNumCols(jx_ParBSRMatrixDiag(par_A));
    
    // printf("Rank %d: Global partition information:\n", myid);
    // printf("Rank %d:   col_starts = %p\n", myid, (void*)col_starts);
    // printf("Rank %d:   num_cols_diag = %d\n", myid, num_cols_diag);
    
    // if (col_starts) {
    //     // 打印列划分信息（假设最多有3个进程的信息）
    //     int num_procs;
    //     jx_MPI_Comm_size(comm, &num_procs);
    //     printf("Rank %d:   col_starts values (first %d): ", myid, jx_min(3, num_procs+1));
    //     for (int i = 0; i <= jx_min(3, num_procs); i++) {
    //         printf("%d ", col_starts[i]);
    //     }
    //     printf("\n");
    // }
    // fflush(stdout);

    // if (col_starts == NULL) {
    //     printf("Rank %d: ERROR: col_starts is NULL in GLOBAL_PARTITION mode!\n", myid);
    //     return jx_error_flag;
    // }
    
    // if (num_cols_diag == 0) {
    //     printf("Rank %d: WARNING: num_cols_diag is 0\n", myid);
    // }
    
    // 直接重用 CSR 的通信包创建核心函数
    jx_MatvecCommPkgCreate_core(comm, col_map_offd, first_col_diag, col_starts,
                               num_cols_diag, num_cols_offd,
                               first_col_diag, col_map_offd, 1,
                               &num_recvs, &recv_procs, &recv_vec_starts,
                               &num_sends, &send_procs, &send_map_starts, &send_map_elmts);
#endif

    // printf("Rank %d: Communication package created: num_recvs=%d, num_sends=%d\n", 
    //        myid, num_recvs, num_sends);
    // fflush(stdout);

    /*-----------------------------------------------------------
     *  setup commpkg
     *----------------------------------------------------------*/

    comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);

    jx_ParCSRCommPkgComm(comm_pkg) = comm;

    jx_ParCSRCommPkgNumRecvs(comm_pkg)      = num_recvs;
    jx_ParCSRCommPkgRecvProcs(comm_pkg)     = recv_procs;
    jx_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
    jx_ParCSRCommPkgNumSends(comm_pkg)      = num_sends;
    jx_ParCSRCommPkgSendProcs(comm_pkg)     = send_procs;
    jx_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
    jx_ParCSRCommPkgSendMapElmts(comm_pkg)  = send_map_elmts;

    jx_ParBSRMatrixCommPkg(par_A) = comm_pkg;

    // printf("Rank %d: ===== EXITING jx_BlockMatvecCommPkgCreate =====\n", myid);
    // fflush(stdout);

    return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BlockMatvecCommPkgCreate
 * \brief Generates the comm_pkg for block matrix A.
 * \date 2025/10/08
 */
// JX_Int
// jx_BlockMatvecCommPkgCreate( jx_ParBSRMatrix *A )
// {
//     JX_Int                num_recvs, *recv_procs, *recv_vec_starts;
//     JX_Int                num_sends, *send_procs, *send_map_starts;
//     JX_Int                *send_map_elmts;

//     JX_Int                num_cols_off_d;
//     JX_Int                *col_map_off_d;

//     JX_Int                first_col_diag;
//     JX_Int                global_num_cols;

//     MPI_Comm              comm;
//     jx_ParCSRCommPkg      *comm_pkg = NULL;
//     jx_IJAssumedPart      *apart;

//     /* get parbsr_A information */
//     col_map_off_d  = jx_ParBSRMatrixColMapOffd(A);
//     num_cols_off_d = jx_BSRMatrixNumCols(jx_ParBSRMatrixOffd(A));

//     global_num_cols = jx_ParBSRMatrixGlobalNumCols(A);
//     comm = jx_ParBSRMatrixComm(A);
//     first_col_diag = jx_ParBSRMatrixFirstColDiag(A);

//     /* Create the assumed partition */
//     if (jx_ParBSRMatrixAssumedPartition(A) == NULL)
//     {
//         jx_ParBSRMatrixCreateAssumedPartition(A);
//     }

//     apart = jx_ParBSRMatrixAssumedPartition(A);

//     /* get commpkg info information */
//     jx_NewCommPkgCreate_core(comm, col_map_off_d, first_col_diag, 
//                              first_col_diag, jx_ParBSRMatrixLastColDiag(A), 
//                              num_cols_off_d, global_num_cols,
//                              &num_recvs, &recv_procs, &recv_vec_starts,
//                              &num_sends, &send_procs, &send_map_starts, 
//                              &send_map_elmts, apart);

//     if (!num_recvs)
//     {
//         jx_TFree(recv_procs);
//         recv_procs = NULL;
//     }
//     if (!num_sends)
//     {
//         jx_TFree(send_procs);
//         jx_TFree(send_map_elmts);
//         send_procs     = NULL;
//         send_map_elmts = NULL;
//     }

//     /* setup commpkg */
//     comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);
//     jx_ParCSRCommPkgComm(comm_pkg) = comm;
//     jx_ParCSRCommPkgNumRecvs(comm_pkg)      = num_recvs;
//     jx_ParCSRCommPkgRecvProcs(comm_pkg)     = recv_procs;
//     jx_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
//     jx_ParCSRCommPkgNumSends(comm_pkg)      = num_sends;
//     jx_ParCSRCommPkgSendProcs(comm_pkg)     = send_procs;
//     jx_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
//     jx_ParCSRCommPkgSendMapElmts(comm_pkg)  = send_map_elmts;

//     jx_ParBSRMatrixCommPkg(A) = comm_pkg;

//     return jx_error_flag;
// }


// JX_Int
// jx_BlockMatvecCommPkgCreate( jx_ParBSRMatrix *A )
// {
//     // 获取MPI信息
//     int myid;
//     MPI_Comm comm = jx_ParBSRMatrixComm(A);
//     jx_MPI_Comm_rank(comm, &myid);
    
//     printf("Rank %d: ===== ENTERING jx_BlockMatvecCommPkgCreate =====\n", myid);
//     printf("Rank %d: Matrix pointer: %p\n", myid, (void*)A);
//     fflush(stdout);
    
//     JX_Int                num_recvs, *recv_procs, *recv_vec_starts;
//     JX_Int                num_sends, *send_procs, *send_map_starts;
//     JX_Int                *send_map_elmts;

//     JX_Int                num_cols_off_d;
//     JX_Int                *col_map_off_d;

//     JX_Int                first_col_diag;
//     JX_Int                global_num_cols;

//     // MPI_Comm              comm;
//     jx_ParCSRCommPkg      *comm_pkg = NULL;
//     jx_IJAssumedPart      *apart;

//     /* get parbsr_A information */
//     col_map_off_d  = jx_ParBSRMatrixColMapOffd(A);
//     num_cols_off_d = jx_BSRMatrixNumCols(jx_ParBSRMatrixOffd(A));
    
//     printf("Rank %d: col_map_off_d = %p\n", myid, (void*)col_map_off_d);
//     printf("Rank %d: num_cols_off_d = %d\n", myid, num_cols_off_d);
//     fflush(stdout);
    
//     // 检查col_map_off_d数组内容
//     if (col_map_off_d && num_cols_off_d > 0) {
//         printf("Rank %d: First 10 elements of col_map_off_d:\n", myid);
//         for (int i = 0; i < 10 && i < num_cols_off_d; i++) {
//             printf("  col_map_off_d[%d] = %d\n", i, col_map_off_d[i]);
//         }
//         fflush(stdout);
//     }

//     global_num_cols = jx_ParBSRMatrixGlobalNumCols(A);
//     comm = jx_ParBSRMatrixComm(A);
//     first_col_diag = jx_ParBSRMatrixFirstColDiag(A);
    
//     printf("Rank %d: global_num_cols = %lld\n", myid, (long long)global_num_cols);
//     printf("Rank %d: first_col_diag = %d\n", myid, first_col_diag);
//     printf("Rank %d: last_col_diag = %d\n", myid, jx_ParBSRMatrixLastColDiag(A));
//     fflush(stdout);

//     /* Create the assumed partition */
//     if (jx_ParBSRMatrixAssumedPartition(A) == NULL)
//     {
//         printf("Rank %d: Creating assumed partition\n", myid);
//         printf("Rank %d: Matrix pointer before creation: %p\n", myid, (void*)A);
//         printf("Rank %d: matrix->assumed_partition before creation: %p\n", 
//                myid, (void*)A->assumed_partition);
//         fflush(stdout);

//         jx_ParBSRMatrixCreateAssumedPartition(A);

//                // 添加详细检查
//         printf("Rank %d: Matrix pointer after creation: %p\n", myid, (void*)A);
//         printf("Rank %d: Using macro: jx_ParBSRMatrixAssumedPartition(A) = %p\n", 
//                myid, (void*)jx_ParBSRMatrixAssumedPartition(A));
//         printf("Rank %d: Direct access: A->assumed_partition = %p\n", 
//                myid, (void*)A->assumed_partition);
//         fflush(stdout);
//     }

//     apart = jx_ParBSRMatrixAssumedPartition(A);
//     printf("Rank %d: Assumed partition created: %p\n", myid, (void*)apart);
//     printf("Rank %d: apart == A->assumed_partition? %s\n", 
//            myid, apart == A->assumed_partition ? "YES" : "NO");
//     fflush(stdout);

//     /* get commpkg info information */
//     printf("Rank %d: Calling jx_NewCommPkgCreate_core\n", myid);
//     fflush(stdout);
    
//     jx_NewCommPkgCreate_core(comm, col_map_off_d, first_col_diag, 
//                              first_col_diag, jx_ParBSRMatrixLastColDiag(A), 
//                              num_cols_off_d, global_num_cols,
//                              &num_recvs, &recv_procs, &recv_vec_starts,
//                              &num_sends, &send_procs, &send_map_starts, 
//                              &send_map_elmts, apart);
    
//     printf("Rank %d: jx_NewCommPkgCreate_core returned\n", myid);
//     printf("Rank %d: num_recvs = %d, num_sends = %d\n", myid, num_recvs, num_sends);
//     fflush(stdout);

//     if (!num_recvs)
//     {
//         printf("Rank %d: No receives, freeing recv_procs\n", myid);
//         jx_TFree(recv_procs);
//         recv_procs = NULL;
//     }
//     if (!num_sends)
//     {
//         printf("Rank %d: No sends, freeing send_procs and send_map_elmts\n", myid);
//         jx_TFree(send_procs);
//         jx_TFree(send_map_elmts);
//         send_procs     = NULL;
//         send_map_elmts = NULL;
//     }

//     /* setup commpkg */
//     printf("Rank %d: Setting up comm_pkg\n", myid);
//     fflush(stdout);
    
//     comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);
//     jx_ParCSRCommPkgComm(comm_pkg) = comm;
//     jx_ParCSRCommPkgNumRecvs(comm_pkg)      = num_recvs;
//     jx_ParCSRCommPkgRecvProcs(comm_pkg)     = recv_procs;
//     jx_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
//     jx_ParCSRCommPkgNumSends(comm_pkg)      = num_sends;
//     jx_ParCSRCommPkgSendProcs(comm_pkg)     = send_procs;
//     jx_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
//     jx_ParCSRCommPkgSendMapElmts(comm_pkg)  = send_map_elmts;

//     jx_ParBSRMatrixCommPkg(A) = comm_pkg;
    
//     printf("Rank %d: ===== EXITING jx_BlockMatvecCommPkgCreate =====\n", myid);
//     fflush(stdout);

//     return jx_error_flag;
// }

/*!
 * \fn JX_Int jx_BuildBSRMatrixMPIDataType
 * \brief Build MPI datatype for BSR matrix.
 * \date 2025/10/08
 */
JX_Int
jx_BuildBSRMatrixMPIDataType( JX_Int          block_size,
                              JX_Int          num_nonzeros,
                              JX_Int          num_rows,
                              JX_Real         *a_data,
                              JX_Int          *a_i,
                              JX_Int          *a_j,
                              MPI_Datatype    *bsr_matrix_datatype )
{
    JX_Int          block_lens[3];
    MPI_Aint        displ[3];
    MPI_Datatype    types[3];

    block_lens[0] = num_nonzeros * block_size * block_size;
    block_lens[1] = num_rows + 1;
    block_lens[2] = num_nonzeros;

    types[0] = JX_MPI_REAL;
    types[1] = JX_MPI_INT;
    types[2] = JX_MPI_INT;

    jx_MPI_Address(a_data, &displ[0]);
    jx_MPI_Address(a_i, &displ[1]);
    jx_MPI_Address(a_j, &displ[2]);
    jx_MPI_Type_struct(3, block_lens, displ, types, bsr_matrix_datatype);
    jx_MPI_Type_commit(bsr_matrix_datatype);

    return jx_error_flag;
}