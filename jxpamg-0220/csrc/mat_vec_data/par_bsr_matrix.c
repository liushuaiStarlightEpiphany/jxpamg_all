//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_par_bsr_matrix.c -- member functions for jx_ParBSRMatrix class.
 *  Date: 2025/10/25
 */ 

#include "jx_parbsr_mv.h"


/*!
 * \fn jx_ParBSRMatrix* jx_ParBSRMatrixCreate
 * \brief Create a parallel block sparse row matrix.
 * \date 2025/10/25
 */

jx_ParBSRMatrix* 
jx_ParBSRMatrixCreate( MPI_Comm   comm,
                       JX_Int        block_size,  
                       JX_BigInt        global_num_rows,
                       JX_BigInt        global_num_cols,
                       JX_BigInt       *row_starts,
                       JX_BigInt       *col_starts,
                       JX_Int        num_cols_offd,
                       JX_Int        num_nonzeros_diag,
                       JX_Int        num_nonzeros_offd )
{
   jx_ParBSRMatrix  *matrix;
   
   JX_Int num_procs, my_id;
   JX_Int local_num_rows, local_num_cols;
   JX_BigInt first_row_index, first_col_diag;
   
   matrix = jx_CTAlloc(jx_ParBSRMatrix, 1);

   jx_MPI_Comm_rank(comm, &my_id);
   jx_MPI_Comm_size(comm, &num_procs);

   if (!row_starts)
   {
      jx_GeneratePartitioning(global_num_rows, num_procs, &row_starts);
   }
   
   if (!col_starts)
   {
      if (global_num_rows == global_num_cols)
      {
         col_starts = row_starts;
      }
      else
      {
         jx_GeneratePartitioning(global_num_cols, num_procs, &col_starts);
      }
   }

   first_row_index = row_starts[my_id];
   local_num_rows  = row_starts[my_id+1] - first_row_index;
   first_col_diag  = col_starts[my_id];
   local_num_cols  = col_starts[my_id+1] - first_col_diag;

   jx_ParBSRMatrixComm(matrix) = comm;
   jx_ParBSRMatrixDiag(matrix) = jx_BSRMatrixCreate(block_size, local_num_rows, local_num_cols, num_nonzeros_diag);
   jx_ParBSRMatrixOffd(matrix) = jx_BSRMatrixCreate(block_size, local_num_rows, num_cols_offd, num_nonzeros_offd);
   jx_ParBSRMatrixGlobalNumRows(matrix) = global_num_rows;
   jx_ParBSRMatrixGlobalNumCols(matrix) = global_num_cols;
   jx_ParBSRMatrixFirstRowIndex(matrix) = first_row_index;
   jx_ParBSRMatrixFirstColDiag(matrix)  = first_col_diag;
 
   jx_ParBSRMatrixLastRowIndex(matrix) = first_row_index + (JX_BigInt)local_num_rows - 1;
   jx_ParBSRMatrixLastColDiag(matrix)  = first_col_diag  + (JX_BigInt)local_num_cols - 1;

   jx_ParBSRMatrixColMapOffd(matrix)       = NULL;
   jx_ParBSRMatrixAssumedPartition(matrix) = NULL;

   jx_ParBSRMatrixRowStarts(matrix) = row_starts;
   jx_ParBSRMatrixColStarts(matrix) = col_starts;

//    jx_ParBSRMatrixDiagT(matrix) = NULL;
//    jx_ParBSRMatrixOffdT(matrix) = NULL;

   jx_ParBSRMatrixCommPkg(matrix)  = NULL;
   jx_ParBSRMatrixCommPkgT(matrix) = NULL;

   /* set defaults */
   jx_ParBSRMatrixOwnsData(matrix) = 1;
   jx_ParBSRMatrixOwnsRowStarts(matrix) = 1;
   jx_ParBSRMatrixOwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
   {
      jx_ParBSRMatrixOwnsColStarts(matrix) = 0;
   }
   jx_ParBSRMatrixRowindices(matrix)   = NULL;
   jx_ParBSRMatrixRowvalues(matrix)    = NULL;
   jx_ParBSRMatrixGetrowactive(matrix) = 0;

   return matrix;
}


// jx_ParBSRMatrix* jx_ParBSRMatrixCreate(MPI_Comm comm, JX_Int block_size, JX_BigInt global_num_rows, JX_BigInt global_num_cols,
//                                        JX_BigInt* row_starts_in, JX_BigInt* col_starts_in, JX_Int num_cols_offd,
//                                        JX_Int num_nonzeros_diag, JX_Int num_nonzeros_offd)
// {
//     jx_ParBSRMatrix* matrix;
//     JX_Int           num_procs, my_id;
//     JX_Int           local_num_rows;
//     JX_Int           local_num_cols;
//     JX_BigInt        first_row_index, first_col_diag;
//     JX_BigInt        row_starts[2];
//     JX_BigInt        col_starts[2];

//     matrix = jx_CTAlloc(jx_ParBSRMatrix, 1);

//     jx_MPI_Comm_rank(comm, &my_id);
//     jx_MPI_Comm_size(comm, &num_procs);

//     if (!row_starts_in) {
//         jx_GenerateLocalPartitioning(global_num_rows, num_procs, my_id, row_starts);
//     } else {
//         row_starts[0] = row_starts_in[0];
//         row_starts[1] = row_starts_in[1];
//     }

//     if (!col_starts_in) {
//         jx_GenerateLocalPartitioning(global_num_cols, num_procs, my_id, col_starts);
//     } else {
//         col_starts[0] = col_starts_in[0];
//         col_starts[1] = col_starts_in[1];
//     }

//     /* row_starts[0] is start of local rows.
//        row_starts[1] is start of next processor's rows */
//     first_row_index                 = row_starts[0];
//     local_num_rows                  = (JX_Int)(row_starts[1] - first_row_index);
//     first_col_diag                  = col_starts[0];
//     local_num_cols                  = (JX_Int)(col_starts[1] - first_col_diag);
//     jx_ParBSRMatrixComm(matrix) = comm;
//     jx_ParBSRMatrixDiag(matrix) = jx_BSRMatrixCreate(block_size, local_num_rows, local_num_cols, num_nonzeros_diag);
//     jx_ParBSRMatrixOffd(matrix) = jx_BSRMatrixCreate(block_size, local_num_rows, num_cols_offd, num_nonzeros_offd);

//     jx_ParBSRMatrixBlockSize(matrix)        = block_size;
//     jx_ParBSRMatrixGlobalNumRows(matrix)    = global_num_rows;
//     jx_ParBSRMatrixGlobalNumCols(matrix)    = global_num_cols;
//     jx_ParBSRMatrixFirstRowIndex(matrix)    = first_row_index;
//     jx_ParBSRMatrixFirstColDiag(matrix)     = first_col_diag;
//     jx_ParBSRMatrixLastRowIndex(matrix)     = first_row_index + (JX_BigInt)local_num_rows - 1;
//     jx_ParBSRMatrixLastColDiag(matrix)      = first_col_diag + (JX_BigInt)local_num_cols - 1;
//     jx_ParBSRMatrixRowStarts(matrix)[0]     = row_starts[0];
//     jx_ParBSRMatrixRowStarts(matrix)[1]     = row_starts[1];
//     jx_ParBSRMatrixColStarts(matrix)[0]     = col_starts[0];
//     jx_ParBSRMatrixColStarts(matrix)[1]     = col_starts[1];
//     jx_ParBSRMatrixColMapOffd(matrix)       = NULL;
//     jx_ParBSRMatrixCommPkg(matrix)          = NULL;
//     jx_ParBSRMatrixCommPkgT(matrix)         = NULL;
//     jx_ParBSRMatrixAssumedPartition(matrix) = NULL;
//    //  jx_ParBSRMatrixDiagT(matrix)            = NULL;
//    //  jx_ParBSRMatrixOffdT(matrix)            = NULL;

//     /* set defaults */
//     jx_ParBSRMatrixOwnsData(matrix) = 1;

//     return matrix;
// }

/*!
 * \fn JX_Int jx_ParBSRMatrixDestroy
 * \brief Destroy a parallel block sparse row matrix.
 * \date 2025/10/25
 */
JX_Int jx_ParBSRMatrixDestroy(jx_ParBSRMatrix* matrix)
{
    if (matrix) {
        if (jx_ParBSRMatrixOwnsData(matrix)) {
            jx_BSRMatrixDestroy(jx_ParBSRMatrixDiag(matrix));
            jx_BSRMatrixDestroy(jx_ParBSRMatrixOffd(matrix));
            if (jx_ParBSRMatrixColMapOffd(matrix)) {
                jx_TFree(jx_ParBSRMatrixColMapOffd(matrix));
            }
            if (jx_ParBSRMatrixCommPkg(matrix)) {
                jx_MatvecCommPkgDestroy((jx_ParCSRCommPkg*)jx_ParBSRMatrixCommPkg(matrix));
            }
            if (jx_ParBSRMatrixCommPkgT(matrix)) {
                jx_MatvecCommPkgDestroy((jx_ParCSRCommPkg*)jx_ParBSRMatrixCommPkgT(matrix));
            }
        }

        // if (jx_ParBSRMatrixDiagT(matrix)) {
        //     jx_BSRMatrixDestroy(jx_ParBSRMatrixDiagT(matrix));
        // }
        // if (jx_ParBSRMatrixOffdT(matrix)) {
        //     jx_BSRMatrixDestroy(jx_ParBSRMatrixOffdT(matrix));
        // }

        if (jx_ParBSRMatrixAssumedPartition(matrix)) {
            jx_ParBSRMatrixDestroyAssumedPartition(matrix);
        }

        jx_TFree(matrix);
    }

    return 0;
}

/*!
 * \fn JX_Int jx_ParBSRMatrixInitialize
 * \brief Initialize a parallel block sparse row matrix.
 * \date 2025/10/25
 */
JX_Int jx_ParBSRMatrixInitialize(jx_ParBSRMatrix* matrix)
{
    JX_Int ierr = 0;

    jx_BSRMatrixInitialize(jx_ParBSRMatrixDiag(matrix));
    jx_BSRMatrixInitialize(jx_ParBSRMatrixOffd(matrix));
    jx_ParBSRMatrixColMapOffd(matrix) = jx_CTAlloc(JX_BigInt, jx_BSRMatrixNumCols(jx_ParBSRMatrixOffd(matrix)));

    return ierr;
}

/*!
 * \fn JX_Int jx_ParBSRMatrixSetNumNonzeros
 * \brief Set the total number of nonzeros in parallel BSR matrix.
 * \date 2025/10/25
 */
JX_Int jx_ParBSRMatrixSetNumNonzeros(jx_ParBSRMatrix* matrix)
{
    MPI_Comm          comm           = jx_ParBSRMatrixComm(matrix);
    jx_BSRMatrix*     diag           = jx_ParBSRMatrixDiag(matrix);
    JX_Int*           diag_i         = jx_BSRMatrixI(diag);
    jx_BSRMatrix*     offd           = jx_ParBSRMatrixOffd(matrix);
    JX_Int*           offd_i         = jx_BSRMatrixI(offd);
    JX_Int            local_num_rows = jx_BSRMatrixNumRows(diag);
    JX_BigInt         total_num_nonzeros;
    JX_BigInt         local_num_nonzeros;
    JX_Int            ierr = 0;

    local_num_nonzeros = (JX_BigInt)(diag_i[local_num_rows] + offd_i[local_num_rows]);
    jx_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, JX_MPI_INT, MPI_SUM, comm);
    jx_ParBSRMatrixNumNonzeros(matrix) = total_num_nonzeros;

    return ierr;
}

/*!
 * \fn JX_Int jx_ParBSRMatrixSetDNumNonzeros
 * \brief Set the total number of nonzeros (as double) in parallel BSR matrix.
 * \date 2025/10/25
 */
JX_Int jx_ParBSRMatrixSetDNumNonzeros(jx_ParBSRMatrix* matrix)
{
    MPI_Comm          comm           = jx_ParBSRMatrixComm(matrix);
    jx_BSRMatrix*     diag           = jx_ParBSRMatrixDiag(matrix);
    JX_Int*           diag_i         = jx_BSRMatrixI(diag);
    jx_BSRMatrix*     offd           = jx_ParBSRMatrixOffd(matrix);
    JX_Int*           offd_i         = jx_BSRMatrixI(offd);
    JX_Int            local_num_rows = jx_BSRMatrixNumRows(diag);
    JX_Real           total_num_nonzeros;
    JX_Real           local_num_nonzeros;
    JX_Int            ierr = 0;

    local_num_nonzeros = (JX_Real)diag_i[local_num_rows] + (JX_Real)offd_i[local_num_rows];
    jx_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, JX_MPI_REAL, MPI_SUM, comm);
    jx_ParBSRMatrixDNumNonzeros(matrix) = total_num_nonzeros;

    return ierr;
}

/*!
 * \fn JX_Int jx_ParBSRMatrixSetDataOwner
 * \brief Set data ownership flag for parallel BSR matrix.
 * \date 2025/10/25
 */
JX_Int jx_ParBSRMatrixSetDataOwner(jx_ParBSRMatrix* matrix, JX_Int owns_data)
{
    JX_Int ierr = 0;

    jx_ParBSRMatrixOwnsData(matrix) = owns_data;

    return ierr;
}

/*!
 * \fn jx_ParCSRMatrix* jx_ParBSRMatrixCompress
 * \brief Compress BSR matrix to CSR by taking F-norm for each sub-block.
 * \date 2025/10/25
 */
jx_ParCSRMatrix* jx_ParBSRMatrixCompress(jx_ParBSRMatrix* matrix)
{
    MPI_Comm          comm              = jx_ParBSRMatrixComm(matrix);
    jx_BSRMatrix*     diag              = jx_ParBSRMatrixDiag(matrix);
    jx_BSRMatrix*     offd              = jx_ParBSRMatrixOffd(matrix);
    JX_BigInt         global_num_rows   = jx_ParBSRMatrixGlobalNumRows(matrix);
    JX_BigInt         global_num_cols   = jx_ParBSRMatrixGlobalNumCols(matrix);
    JX_BigInt*        row_starts        = jx_ParBSRMatrixRowStarts(matrix);
    JX_BigInt*        col_starts        = jx_ParBSRMatrixColStarts(matrix);
    JX_Int            num_cols_offd     = jx_BSRMatrixNumCols(offd);
    JX_Int            num_nonzeros_diag = jx_BSRMatrixNumNonzeros(diag);
    JX_Int            num_nonzeros_offd = jx_BSRMatrixNumNonzeros(offd);

    jx_ParCSRMatrix* matrix_C;
    JX_Int i;

    matrix_C = jx_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols, row_starts, col_starts, 
                                     num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
    jx_ParCSRMatrixInitialize(matrix_C);

    jx_CSRMatrixDestroy(jx_ParCSRMatrixDiag(matrix_C));
    jx_ParCSRMatrixDiag(matrix_C) = jx_BSRMatrixCompress(diag);
    jx_CSRMatrixDestroy(jx_ParCSRMatrixOffd(matrix_C));
    jx_ParCSRMatrixOffd(matrix_C) = jx_BSRMatrixCompress(offd);

    for (i = 0; i < num_cols_offd; i++) 
        jx_ParCSRMatrixColMapOffd(matrix_C)[i] = jx_ParBSRMatrixColMapOffd(matrix)[i];
    
    return matrix_C;
}

/*!
 * \fn jx_ParCSRMatrix* jx_ParBSRMatrixGetSubmatrix
 * \brief Get submatrix from parallel BSR matrix.
 * \date 2025/10/25
 */
jx_ParCSRMatrix* jx_ParBSRMatrixGetSubmatrix(jx_ParBSRMatrix* matrix, JX_Int subposition, JX_Real threshold)
{


    MPI_Comm          comm              = jx_ParBSRMatrixComm(matrix);
    jx_BSRMatrix*     diag              = jx_ParBSRMatrixDiag(matrix);
    jx_BSRMatrix*     offd              = jx_ParBSRMatrixOffd(matrix);
    JX_BigInt         global_num_rows   = jx_ParBSRMatrixGlobalNumRows(matrix);
    JX_BigInt         global_num_cols   = jx_ParBSRMatrixGlobalNumCols(matrix);
    JX_BigInt*        row_starts        = jx_ParBSRMatrixRowStarts(matrix);
    JX_BigInt*        col_starts        = jx_ParBSRMatrixColStarts(matrix);
    JX_Int            num_cols_offd     = jx_BSRMatrixNumCols(offd);
    JX_Int            num_nonzeros_diag = jx_BSRMatrixNumNonzeros(diag);
    JX_Int            num_nonzeros_offd = jx_BSRMatrixNumNonzeros(offd);

    jx_ParCSRMatrix* matrix_C;
    JX_Int i;


    matrix_C = jx_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols, row_starts, col_starts,
                                     num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
    jx_ParCSRMatrixInitialize(matrix_C);



    jx_CSRMatrixDestroy(jx_ParCSRMatrixDiag(matrix_C));
    jx_ParCSRMatrixDiag(matrix_C) = jx_BSRMatrixGetSubmatrix(diag, subposition);



    jx_CSRMatrixDestroy(jx_ParCSRMatrixOffd(matrix_C));
    jx_ParCSRMatrixOffd(matrix_C) = jx_BSRMatrixGetSubmatrix(offd, subposition);



    for (i = 0; i < num_cols_offd; i++) 
        jx_ParCSRMatrixColMapOffd(matrix_C)[i] = jx_ParBSRMatrixColMapOffd(matrix)[i];



    // Drop Small Entries
    JX_Int type = 0;
    /*
     *    type 0: threshold
     *    type 1: threshold*(1-norm of row)
     *    type 2: threshold*(2-norm of row)
     *    type -1: threshold*(infinity norm of row)
     */
    // jx_ParCSRMatrixDropSmallEntries(matrix_C, threshold, type);

    // printf(" [%s:%d] matrix_C = %p \n",__FUNCTION__, __LINE__, (void*)matrix_C);
    // fflush(stdout);  
    // printf(" [%s:%d] jx_ParCSRMatrixComm(matrix_C) = %p \n",__FUNCTION__, __LINE__, (void*)jx_ParCSRMatrixComm(matrix_C));
    // fflush(stdout);  
    // printf(" [%s:%d] jx_ParCSRMatrixDiag(ParCSR) = %p \n",__FUNCTION__, __LINE__, (void*)jx_ParCSRMatrixDiag(matrix_C));
    // fflush(stdout);  
    // printf(" [%s:%d] jx_CSRMatrixI(ParCSR) = %p \n",__FUNCTION__, __LINE__, (void*)jx_CSRMatrixI(jx_ParCSRMatrixDiag(matrix_C)));
    // fflush(stdout);  

    return matrix_C;
}

/*!
 * \fn jx_ParCSRMatrix* jx_ParBSRMatrixConvertToParCSRMatrix
 * \brief Convert parallel BSR matrix to parallel CSR matrix.
 * \date 2025/10/25
 */
jx_ParCSRMatrix* jx_ParBSRMatrixConvertToParCSRMatrix(jx_ParBSRMatrix* matrix)
{
    MPI_Comm          comm              = jx_ParBSRMatrixComm(matrix);
    jx_BSRMatrix*     diag              = jx_ParBSRMatrixDiag(matrix);
    jx_BSRMatrix*     offd              = jx_ParBSRMatrixOffd(matrix);
    JX_Int            block_size        = jx_ParBSRMatrixBlockSize(matrix);
    JX_BigInt         global_num_rows   = jx_ParBSRMatrixGlobalNumRows(matrix);
    JX_BigInt         global_num_cols   = jx_ParBSRMatrixGlobalNumCols(matrix);
    JX_BigInt*        row_starts        = jx_ParBSRMatrixRowStarts(matrix);
    JX_BigInt*        col_starts        = jx_ParBSRMatrixColStarts(matrix);
    JX_Int            num_cols_offd     = jx_BSRMatrixNumCols(offd);
    JX_Int            num_nonzeros_diag = jx_BSRMatrixNumNonzeros(diag);
    JX_Int            num_nonzeros_offd = jx_BSRMatrixNumNonzeros(offd);

    jx_ParCSRMatrix* matrix_C;
    JX_BigInt        matrix_C_row_starts[2];
    JX_BigInt        matrix_C_col_starts[2];

    JX_Int *   counter, *new_j_map;
    JX_Int     size_j, size_map, index, new_num_cols, removed = 0;
    JX_Int*    offd_j;
    JX_BigInt *col_map_offd, *new_col_map_offd;

    JX_Int num_procs, i, j;

    jx_CSRMatrix *diag_nozeros, *offd_nozeros;

    jx_MPI_Comm_size(comm, &num_procs);

    for (i = 0; i < 2; i++) {
        matrix_C_row_starts[i] = row_starts[i] * (JX_BigInt)block_size;
        matrix_C_col_starts[i] = col_starts[i] * (JX_BigInt)block_size;
    }

    matrix_C = jx_ParCSRMatrixCreate(comm, global_num_rows * (JX_BigInt)block_size, 
                                     global_num_cols * (JX_BigInt)block_size, 
                                     matrix_C_row_starts, matrix_C_col_starts,
                                     num_cols_offd * block_size, 
                                     num_nonzeros_diag * block_size * block_size,
                                     num_nonzeros_offd * block_size * block_size);
    jx_ParCSRMatrixInitialize(matrix_C);

    /* DIAG */
    jx_CSRMatrixDestroy(jx_ParCSRMatrixDiag(matrix_C));
    jx_ParCSRMatrixDiag(matrix_C) = jx_BSRMatrixConvertToCSRMatrix(diag);

    /* Delete zeros */
#if 1
    diag_nozeros = jx_CSRMatrixDeleteZeros(jx_ParCSRMatrixDiag(matrix_C), 1e-14);
    if (diag_nozeros) {
        jx_CSRMatrixDestroy(jx_ParCSRMatrixDiag(matrix_C));
        jx_ParCSRMatrixDiag(matrix_C) = diag_nozeros;
    }
#endif

    /* OFF-DIAG */
    jx_CSRMatrixDestroy(jx_ParCSRMatrixOffd(matrix_C));
    jx_ParCSRMatrixOffd(matrix_C) = jx_BSRMatrixConvertToCSRMatrix(offd);

#if 1
    /* Delete zeros */
    offd_nozeros = jx_CSRMatrixDeleteZeros(jx_ParCSRMatrixOffd(matrix_C), 1e-14);
    if (offd_nozeros) {
        jx_CSRMatrixDestroy(jx_ParCSRMatrixOffd(matrix_C));
        jx_ParCSRMatrixOffd(matrix_C) = offd_nozeros;
        removed = 1;
    }
#endif

    /* Convert the col_map_offd */
    for (i = 0; i < num_cols_offd; i++)
        for (j = 0; j < block_size; j++)
            jx_ParCSRMatrixColMapOffd(matrix_C)[i * block_size + j] = 
                jx_ParBSRMatrixColMapOffd(matrix)[i] * (JX_BigInt)block_size + (JX_BigInt)j;

    /* If we deleted zeros, compress col_map_offd */
    if (removed) {
        size_map  = num_cols_offd * block_size;
        counter   = jx_CTAlloc(JX_Int, size_map);
        new_j_map = jx_CTAlloc(JX_Int, size_map);

        offd_j       = jx_CSRMatrixJ(jx_ParCSRMatrixOffd(matrix_C));
        col_map_offd = jx_ParCSRMatrixColMapOffd(matrix_C);

        size_j = jx_CSRMatrixNumNonzeros(jx_ParCSRMatrixOffd(matrix_C));
        /* Mark which off_d entries are found in j */
        for (i = 0; i < size_j; i++) {
            counter[offd_j[i]] = 1;
        }
        /* Find new numbering for columns */
        index = 0;
        for (i = 0; i < size_map; i++) {
            if (counter[i]) {
                new_j_map[i] = index++;
            }
        }
        new_num_cols = index;
        
        /* If there are some col entries to remove */
        if (!(index == size_map)) {
            /* Adjust j entries */
            for (i = 0; i < size_j; i++) {
                offd_j[i] = new_j_map[offd_j[i]];
            }
            /* Compress col map */
            new_col_map_offd = jx_CTAlloc(JX_BigInt, new_num_cols);
            index = 0;
            for (i = 0; i < size_map; i++) {
                if (counter[i]) {
                    new_col_map_offd[index++] = col_map_offd[i];
                }
            }
            /* Set the new col map */
            jx_TFree(col_map_offd);
            jx_ParCSRMatrixColMapOffd(matrix_C) = new_col_map_offd;
            /* Modify the number of cols */
            jx_CSRMatrixNumCols(jx_ParCSRMatrixOffd(matrix_C)) = new_num_cols;
        }
        jx_TFree(new_j_map);
        jx_TFree(counter);
    }

    jx_ParCSRMatrixSetNumNonzeros(matrix_C);
    jx_ParCSRMatrixSetDNumNonzeros(matrix_C);

    /* We will not copy the comm package */
    jx_ParCSRMatrixCommPkg(matrix_C) = NULL;

    return matrix_C;
}

/*!
 * \fn jx_ParBSRMatrix* jx_ParBSRMatrixConvertFromParCSRMatrix
 * \brief Convert parallel CSR matrix to parallel BSR matrix.
 * \date 2025/10/25
 */
jx_ParBSRMatrix* jx_ParBSRMatrixConvertFromParCSRMatrix(jx_ParCSRMatrix* matrix, JX_Int matrix_C_block_size)
{
    MPI_Comm          comm            = jx_ParCSRMatrixComm(matrix);
    jx_CSRMatrix*     diag            = jx_ParCSRMatrixDiag(matrix);
    jx_CSRMatrix*     offd            = jx_ParCSRMatrixOffd(matrix);
    JX_BigInt         global_num_rows = jx_ParCSRMatrixGlobalNumRows(matrix);
    JX_BigInt         global_num_cols = jx_ParCSRMatrixGlobalNumCols(matrix);
    JX_BigInt*        row_starts      = jx_ParCSRMatrixRowStarts(matrix);
    JX_BigInt*        col_starts      = jx_ParCSRMatrixColStarts(matrix);
    JX_Int            num_cols_offd   = jx_CSRMatrixNumCols(offd);
    JX_BigInt*        col_map_offd    = jx_ParCSRMatrixColMapOffd(matrix);
    JX_BigInt*        map_to_node     = NULL;
    JX_Int *          counter = NULL, *col_in_j_map = NULL;
    JX_BigInt*        matrix_C_col_map_offd = NULL;

    JX_Int matrix_C_num_cols_offd;
    JX_Int matrix_C_num_nonzeros_offd;
    JX_Int num_rows, num_nodes, num_error;

    JX_Int*     offd_i    = jx_CSRMatrixI(offd);
    JX_Int*     offd_j    = jx_CSRMatrixJ(offd);
    JX_Real*    offd_data = jx_CSRMatrixData(offd);

    jx_ParBSRMatrix* matrix_C;
    JX_BigInt        matrix_C_row_starts[2];
    JX_BigInt        matrix_C_col_starts[2];
    jx_BSRMatrix*    matrix_C_diag;
    jx_BSRMatrix*    matrix_C_offd;

    JX_Int *    matrix_C_offd_i = NULL, *matrix_C_offd_j = NULL;
    JX_Real*    matrix_C_offd_data = NULL;

    JX_Int rank, num_procs, i, j, k, k_map, count, index, start_index, pos, row;

    jx_MPI_Comm_rank(comm, &rank);
    jx_MPI_Comm_size(comm, &num_procs);

    for (i = 0; i < 2; i++) {
        matrix_C_row_starts[i] = row_starts[i] / (JX_BigInt)matrix_C_block_size;
        matrix_C_col_starts[i] = col_starts[i] / (JX_BigInt)matrix_C_block_size;
    }

    /* Figure out the new number of offd columns */
    num_cols_offd = offd->num_cols;
    num_rows      = diag->num_rows;
    num_nodes     = num_rows / matrix_C_block_size;
    num_error     = num_rows % matrix_C_block_size;

    if (num_error != 0) {
        jx_printf("%s, rank: %d, num_rows: %d cannot be divided by block_size: %d\n", 
                  __FUNCTION__, rank, num_rows, matrix_C_block_size);
        return NULL;
    }

    /************* Create the diagonal part ************/
    matrix_C_diag = jx_BSRMatrixConvertFromCSRMatrix(diag, matrix_C_block_size);

    /******* The offd part *******************/

    /* Can't use the same function for the offd part - because this isn't square
       and the offd j entries aren't global numbering (have to consider the offd map) */
    matrix_C_offd_i = jx_CTAlloc(JX_Int, num_nodes + 1);

    matrix_C_num_cols_offd     = 0;
    matrix_C_offd_i[0]         = 0;
    matrix_C_num_nonzeros_offd = 0;

    if (num_cols_offd) {
        map_to_node            = jx_CTAlloc(JX_BigInt, num_cols_offd);
        matrix_C_num_cols_offd = 1;
        map_to_node[0]         = col_map_offd[0] / (JX_BigInt)matrix_C_block_size;
        for (i = 1; i < num_cols_offd; i++) {
            map_to_node[i] = col_map_offd[i] / (JX_BigInt)matrix_C_block_size;
            if (map_to_node[i] > map_to_node[i - 1]) {
                matrix_C_num_cols_offd++;
            }
        }

        matrix_C_col_map_offd = jx_CTAlloc(JX_BigInt, matrix_C_num_cols_offd);
        col_in_j_map          = jx_CTAlloc(JX_Int, num_cols_offd);

        matrix_C_col_map_offd[0] = map_to_node[0];
        col_in_j_map[0]          = 0;
        count                    = 1;
        j                        = 1;

        /* Fill in the col_map_offd - these are global numbers */
        for (i = 1; i < num_cols_offd; i++) {
            if (map_to_node[i] > map_to_node[i - 1]) {
                matrix_C_col_map_offd[count++] = map_to_node[i];
            }
            col_in_j_map[j++] = count - 1;
        }

        /* Now figure the nonzeros */
        matrix_C_num_nonzeros_offd = 0;
        counter                    = jx_CTAlloc(JX_Int, matrix_C_num_cols_offd);
        for (i = 0; i < matrix_C_num_cols_offd; i++) {
            counter[i] = -1;
        }

        for (i = 0; i < num_nodes; i++) /* For each block row */
        {
            matrix_C_offd_i[i] = matrix_C_num_nonzeros_offd;
            for (j = 0; j < matrix_C_block_size; j++) {
                row = i * matrix_C_block_size + j;
                for (k = offd_i[row]; k < offd_i[row + 1]; k++) /* Go through single row */
                {
                    k_map = col_in_j_map[offd_j[k]]; /* Nodal col - see if this has been in this block row already */

                    if (counter[k_map] < i) /* Not yet counted for this nodal row */
                    {
                        counter[k_map] = i;
                        matrix_C_num_nonzeros_offd++;
                    }
                }
            }
        }
        /* Fill in final i entry */
        matrix_C_offd_i[num_nodes] = matrix_C_num_nonzeros_offd;
    }

    /* Create offd matrix */
    matrix_C_offd = jx_BSRMatrixCreate(matrix_C_block_size, num_nodes, matrix_C_num_cols_offd, matrix_C_num_nonzeros_offd);

    /* Assign i */
    jx_BSRMatrixI(matrix_C_offd) = matrix_C_offd_i;

    /* Create (and allocate j and data) */
    if (matrix_C_num_nonzeros_offd) {
        matrix_C_offd_j    = jx_CTAlloc(JX_Int, matrix_C_num_nonzeros_offd);
        matrix_C_offd_data = jx_CTAlloc(JX_Real, matrix_C_num_nonzeros_offd * matrix_C_block_size * matrix_C_block_size);
        jx_BSRMatrixJ(matrix_C_offd)    = matrix_C_offd_j;
        jx_BSRMatrixData(matrix_C_offd) = matrix_C_offd_data;

        for (i = 0; i < matrix_C_num_cols_offd; i++) {
            counter[i] = -1;
        }

        index       = 0; /* Keep track of entry in matrix_C_offd_j */
        start_index = 0;
        for (i = 0; i < num_nodes; i++) /* For each block row */
        {
            for (j = 0; j < matrix_C_block_size; j++) /* For each row in block */
            {
                row = i * matrix_C_block_size + j;
                for (k = offd_i[row]; k < offd_i[row + 1]; k++) /* Go through single row's cols */
                {
                    k_map = col_in_j_map[offd_j[k]];  /* Nodal col for off_d */
                    if (counter[k_map] < start_index) /* Not yet counted for this nodal row */
                    {
                        counter[k_map]         = index;
                        matrix_C_offd_j[index] = k_map;
                        /* Copy the data: which position + which row + which col */
                        pos = (index * matrix_C_block_size * matrix_C_block_size) + (j * matrix_C_block_size) +
                              (JX_Int)(col_map_offd[offd_j[k]] % (JX_BigInt)matrix_C_block_size);
                        matrix_C_offd_data[pos] = offd_data[k];
                        index++;
                    } else /* This col has already been listed for this row */
                    {
                        /* Copy the data: which position + which row + which col */
                        pos = (counter[k_map] * matrix_C_block_size * matrix_C_block_size) + (j * matrix_C_block_size) +
                              (JX_Int)(col_map_offd[offd_j[k]] % (JX_BigInt)(matrix_C_block_size));
                        matrix_C_offd_data[pos] = offd_data[k];
                    }
                }
            }
            start_index = index; /* First index for current nodal row */
        }
    }

    /* ********* Create the new matrix  *************/
    matrix_C = jx_ParBSRMatrixCreate(comm, matrix_C_block_size, 
                                     global_num_rows / (JX_BigInt)matrix_C_block_size,
                                     global_num_cols / (JX_BigInt)matrix_C_block_size, 
                                     matrix_C_row_starts, matrix_C_col_starts, 
                                     matrix_C_num_cols_offd,
                                     jx_BSRMatrixNumNonzeros(matrix_C_diag), 
                                     matrix_C_num_nonzeros_offd);

    /* Use the diag and off diag matrices we have already created */
    jx_BSRMatrixDestroy(jx_ParBSRMatrixDiag(matrix_C));
    jx_ParBSRMatrixDiag(matrix_C) = matrix_C_diag;
    jx_BSRMatrixDestroy(jx_ParBSRMatrixOffd(matrix_C));
    jx_ParBSRMatrixOffd(matrix_C) = matrix_C_offd;

    jx_ParBSRMatrixColMapOffd(matrix_C) = matrix_C_col_map_offd;

    /* ********* Don't bother to copy the comm_pkg *************/
    jx_ParBSRMatrixCommPkg(matrix_C) = NULL;

    /* CLEAN UP !!!! */
    jx_TFree(map_to_node);
    jx_TFree(col_in_j_map);
    jx_TFree(counter);

    return matrix_C;
}

/*!
 * \fn jx_ParVector* jx_ParVectorCreateFromBlock
 * \brief Create a parallel vector from block information.
 * \date 2025/10/25
 */
jx_ParVector* jx_ParVectorCreateFromBlock(MPI_Comm comm, JX_BigInt p_global_size, JX_BigInt* p_partitioning, JX_Int block_size)
{
    jx_ParVector* vector;
    JX_Int        num_procs, my_id;
    JX_BigInt     global_size;
    JX_BigInt     new_partitioning[2]; /* Need to create a new partitioning */

    global_size = p_global_size * (JX_BigInt)block_size;

    vector = jx_CTAlloc(jx_ParVector, 1);
    jx_MPI_Comm_rank(comm, &my_id);
    jx_MPI_Comm_size(comm, &num_procs);

    if (!p_partitioning) {
        jx_GenerateLocalPartitioning(global_size, num_procs, my_id, new_partitioning);
    } else /* Adjust for block_size */
    {
        new_partitioning[0] = p_partitioning[0] * (JX_BigInt)block_size;
        new_partitioning[1] = p_partitioning[1] * (JX_BigInt)block_size;
    }

    jx_ParVectorComm(vector) = comm;
    jx_ParVectorGlobalSize(vector) = global_size;
    jx_ParVectorFirstIndex(vector) = new_partitioning[0];
    jx_ParVectorLastIndex(vector) = new_partitioning[1] - 1;
    jx_ParVectorPartitioning(vector)[0] = new_partitioning[0];
    jx_ParVectorPartitioning(vector)[1] = new_partitioning[1];
    jx_ParVectorLocalVector(vector) = jx_SeqVectorCreate(new_partitioning[1] - new_partitioning[0]);

    /* set defaults */
    jx_ParVectorOwnsData(vector) = 1;

    return vector;
}

/*!
 * \fn jx_ParBSRMatrix* jx_BSRMatrixToParBSRMatrix
 * \brief Generate a ParBSRMatrix distributed across processors from a BSRMatrix on proc 0.
 * \date 2025/10/25
 */

/*!
 * \fn jx_ParBSRMatrix* jx_BSRMatrixToParBSRMatrix
 * \brief Generates a ParBSRMatrix distributed across the 
 *        processors in comm from a BSRMatrix on proc 0.
 * \note This shouldn't be used with the JX_NO_GLOBAL_PARTITON option!  
 * \date 2025/10/25
 */
jx_ParBSRMatrix* 
jx_BSRMatrixToParBSRMatrix( MPI_Comm       comm,
                            jx_BSRMatrix  *A,
                            JX_BigInt        *row_starts,
                            JX_BigInt        *col_starts )
{
    JX_BigInt         *global_data = NULL;
    JX_BigInt          global_size;
    JX_BigInt          global_num_rows;
    JX_BigInt          global_num_cols;
    JX_Int         *local_num_rows = NULL;

    JX_Int          num_procs, my_id;
    JX_Int         *local_num_nonzeros = NULL;
    JX_Int          num_nonzeros;
   
    JX_Real       *a_data = NULL;
    JX_Int        *a_i    = NULL;
    JX_Int        *a_j    = NULL;
   
    jx_BSRMatrix *local_A;
    MPI_Request  *requests;
    MPI_Status   *status, status0;
    MPI_Datatype *bsr_matrix_datatypes;
    jx_ParBSRMatrix *par_matrix;
    JX_BigInt first_col_diag;
    JX_BigInt last_col_diag;
    JX_Int i, j, ind;
    JX_Int block_size;

    jx_MPI_Comm_rank(comm, &my_id);
    jx_MPI_Comm_size(comm, &num_procs);

    global_data = jx_CTAlloc(JX_BigInt, 2*num_procs + 6);
    if (my_id == 0) 
    {
        global_size = 3;
        if (row_starts) 
        {
            if (col_starts)
            {
                if (col_starts != row_starts)
                {
                    global_data[3] = 2;
                    global_size = 2*num_procs+6;
                    for (i = 0; i < num_procs + 1; i ++)
                    {
                        global_data[i+4] = row_starts[i];
                    }
                    for (i = 0; i < num_procs + 1; i ++)
                    {
                        global_data[i+num_procs+5] = col_starts[i];
                    }
                }
                else
                {
                    global_data[3] = 0;
                    global_size = num_procs+5;
                    for (i = 0; i < num_procs + 1; i ++)
                    {
                        global_data[i+4] = row_starts[i];
                    }
                }
            }
            else
            {
                global_data[3] = 1;
                global_size = num_procs+5;
                for (i = 0; i < num_procs + 1; i ++)
                {
                    global_data[i+4] = row_starts[i];
                }
            }
        }
        else 
        {
            if (col_starts)
            {
                global_data[3] = 3;
                global_size = num_procs+5;
                for (i = 0; i < num_procs + 1; i ++)
                {
                    global_data[i+4] = col_starts[i];
                }
            }
        }
        
        global_data[0] = jx_BSRMatrixNumRows(A);
        global_data[1] = jx_BSRMatrixNumCols(A);
        global_data[2] = global_size;
        block_size = jx_BSRMatrixBlockSize(A);
        a_data = jx_BSRMatrixData(A);
        a_i = jx_BSRMatrixI(A);
        a_j = jx_BSRMatrixJ(A);
    }
    
    /* Broadcast block size first */
    jx_MPI_Bcast(&block_size, 1, JX_MPI_INT, 0, comm);
    
    /* Broadcast global data */
    jx_MPI_Bcast(global_data, 3, JX_MPI_INT, 0, comm);
    global_num_rows = global_data[0];
    global_num_cols = global_data[1];
    global_size = global_data[2];
    
    if (global_size > 3)
    {
        jx_MPI_Bcast(&global_data[3], global_size - 3, JX_MPI_INT, 0, comm);
        if (my_id > 0)
        {
            if (global_data[3] < 3)
            {
                row_starts = jx_CTAlloc(JX_Int, num_procs + 1);
                for (i = 0; i < num_procs + 1; i ++)
                {
                    row_starts[i] = global_data[i+4];
                }
                if (global_data[3] == 0)
                {
                    col_starts = row_starts;
                }
                if (global_data[3] == 2)
                {
                    col_starts = jx_CTAlloc(JX_Int, num_procs+1);
                    for (i = 0; i < num_procs + 1; i ++)
                    {
                        col_starts[i] = global_data[i+num_procs+5];
                    }
                }
            }
            else
            {
                col_starts = jx_CTAlloc(JX_Int, num_procs + 1);
                for (i = 0; i < num_procs + 1; i ++)
                {
                    col_starts[i] = global_data[i+4];
                }
            }
        }
    }
    jx_TFree(global_data);

    local_num_rows = jx_CTAlloc(JX_Int, num_procs);
    bsr_matrix_datatypes = jx_CTAlloc(MPI_Datatype, num_procs);

    /* Create parallel BSR matrix with global partitioning */
    par_matrix = jx_ParBSRMatrixCreate(comm, block_size, global_num_rows, global_num_cols, 
                                       row_starts, col_starts, 0, 0, 0);

    row_starts = jx_ParBSRMatrixRowStarts(par_matrix);
    col_starts = jx_ParBSRMatrixColStarts(par_matrix);

    for (i = 0; i < num_procs; i ++)
    {
        local_num_rows[i] = row_starts[i+1] - row_starts[i];
    }

    if (my_id == 0)
    {
        local_num_nonzeros = jx_CTAlloc(JX_Int, num_procs);
        for (i = 0; i < num_procs - 1; i ++)
        {
            local_num_nonzeros[i] = a_i[row_starts[i+1]] - a_i[row_starts[i]];
        }
        local_num_nonzeros[num_procs-1] = a_i[global_num_rows] - a_i[row_starts[num_procs-1]];
    }
    jx_MPI_Scatter(local_num_nonzeros, 1, JX_MPI_INT, &num_nonzeros, 1, JX_MPI_INT, 0, comm);

    if (my_id == 0) 
    {
        num_nonzeros = local_num_nonzeros[0];
    }

    local_A = jx_BSRMatrixCreate(block_size, local_num_rows[my_id], global_num_cols, num_nonzeros);

    if (my_id == 0)
    {
        requests = jx_CTAlloc(MPI_Request, num_procs - 1);
        status = jx_CTAlloc(MPI_Status, num_procs - 1);
        j = 0;
        for (i = 1; i < num_procs; i ++)
        {
            ind = a_i[row_starts[i]];
            jx_BuildBSRMatrixMPIDataType(block_size, local_num_nonzeros[i], 
                                          local_num_rows[i],
                                          &a_data[ind * block_size * block_size],
                                          &a_i[row_starts[i]],
                                          &a_j[ind],
                                          &bsr_matrix_datatypes[i]);
            jx_MPI_Isend(MPI_BOTTOM, 1, bsr_matrix_datatypes[i], i, 0, comm, &requests[j++]);
            jx_MPI_Type_free(&bsr_matrix_datatypes[i]);
        }
        jx_BSRMatrixData(local_A) = a_data;
        jx_BSRMatrixI(local_A) = a_i;
        jx_BSRMatrixJ(local_A) = a_j;
        jx_BSRMatrixOwnsData(local_A) = 0;
        jx_MPI_Waitall(num_procs-1, requests, status);
        jx_TFree(requests);
        jx_TFree(status);
        jx_TFree(local_num_nonzeros);
    }
    else
    {
        jx_BSRMatrixInitialize(local_A);
        jx_BuildBSRMatrixMPIDataType(block_size, num_nonzeros, 
                                      local_num_rows[my_id],
                                      jx_BSRMatrixData(local_A),
                                      jx_BSRMatrixI(local_A),
                                      jx_BSRMatrixJ(local_A),
                                      bsr_matrix_datatypes);
        jx_MPI_Recv(MPI_BOTTOM, 1, bsr_matrix_datatypes[0], 0, 0, comm, &status0);
        jx_MPI_Type_free(bsr_matrix_datatypes);
    }

    first_col_diag = col_starts[my_id];
    last_col_diag  = col_starts[my_id+1] - 1;

    jx_GenerateDiagAndOffdBSR(local_A, par_matrix, first_col_diag, last_col_diag);

    /* set pointers back to NULL before destroying */
    if (my_id == 0)
    {      
        jx_BSRMatrixData(local_A) = NULL;
        jx_BSRMatrixI(local_A) = NULL;
        jx_BSRMatrixJ(local_A) = NULL; 
    }      
    jx_BSRMatrixDestroy(local_A);
    jx_TFree(local_num_rows);
    jx_TFree(bsr_matrix_datatypes);

    return par_matrix;
}

// jx_ParBSRMatrix* jx_BSRMatrixToParBSRMatrix(MPI_Comm comm, jx_BSRMatrix* A, JX_BigInt* global_row_starts, JX_BigInt* global_col_starts)
// {


//     jx_ParBSRMatrix* parbsr_A;

//     JX_BigInt* global_data;
//     JX_BigInt  global_size;
//     JX_BigInt  global_num_rows;
//     JX_BigInt  global_num_cols;

//     JX_Int    num_procs, my_id;
//     JX_Int*   num_rows_proc;
//     JX_Int*   num_nonzeros_proc;
//     JX_BigInt row_starts[2];
//     JX_BigInt col_starts[2];

//     jx_BSRMatrix* local_A;
//     JX_Real*      A_data;
//     JX_Int*       A_i;
//     JX_Int*       A_j;
//     JX_Int        block_size = 0;

//     MPI_Request*  requests;
//     MPI_Status *  status, status0;
//     MPI_Datatype* bsr_matrix_datatypes;

//     JX_Int free_global_row_starts = 0;
//     JX_Int free_global_col_starts = 0;

//     JX_Int    total_size;
//     JX_BigInt first_col_diag;
//     JX_BigInt last_col_diag;
//     JX_Int    num_rows;
//     JX_Int    num_nonzeros;
//     JX_Int    i, ind;

//     jx_MPI_Comm_rank(comm, &my_id);
//     jx_MPI_Comm_size(comm, &num_procs);

//     // 在函数开头添加
//     printf("Rank %d: Entering jx_BSRMatrixToParBSRMatrix\n", my_id);
//     fflush(stdout);

//     total_size = 4;
//     if (my_id == 0) {
//         total_size += 2 * (num_procs + 1);
//     }

//     global_data = jx_CTAlloc(JX_BigInt, total_size);
//     if (my_id == 0) {
//         global_size = 3;
//         if (global_row_starts) {
//             if (global_col_starts) {
//                 if (global_col_starts != global_row_starts) {
//                     global_data[3] = 2;
//                     global_size += (JX_BigInt)(2 * (num_procs + 1) + 1);
//                     for (i = 0; i < (num_procs + 1); i++) {
//                         global_data[i + 4] = global_row_starts[i];
//                     }
//                     for (i = 0; i < (num_procs + 1); i++) {
//                         global_data[i + num_procs + 5] = global_col_starts[i];
//                     }
//                 } else {
//                     global_data[3] = 0;
//                     global_size += (JX_BigInt)((num_procs + 1) + 1);
//                     for (i = 0; i < (num_procs + 1); i++) {
//                         global_data[i + 4] = global_row_starts[i];
//                     }
//                 }
//             } else {
//                 global_data[3] = 1;
//                 global_size += (JX_BigInt)((num_procs + 1) + 1);
//                 for (i = 0; i < (num_procs + 1); i++) {
//                     global_data[i + 4] = global_row_starts[i];
//                 }
//             }
//         } else {
//             if (global_col_starts) {
//                 global_data[3] = 3;
//                 global_size += (JX_BigInt)((num_procs + 1) + 1);
//                 for (i = 0; i < (num_procs + 1); i++) {
//                     global_data[i + 4] = global_col_starts[i];
//                 }
//             }
//         }

//         global_data[0] = (JX_BigInt)jx_BSRMatrixNumRows(A);
//         global_data[1] = (JX_BigInt)jx_BSRMatrixNumCols(A);
//         global_data[2] = global_size;
//         A_data     = jx_BSRMatrixData(A);
//         A_i        = jx_BSRMatrixI(A);
//         A_j        = jx_BSRMatrixJ(A);
//         block_size = jx_BSRMatrixBlockSize(A);
//     }
//     jx_MPI_Bcast(&block_size, 1, JX_MPI_INT, 0, comm);
//     jx_MPI_Bcast(global_data, 4, JX_MPI_BIG_INT, 0, comm);
//     global_num_rows = global_data[0];
//     global_num_cols = global_data[1];
//     global_size     = global_data[2];

//     if (global_size > 3) {
//         JX_Int send_start;

//         if (global_data[3] == 2) {
//             send_start = 4;
//             jx_MPI_Scatter(&global_data[send_start], 1, JX_MPI_BIG_INT, &row_starts[0], 1, JX_MPI_BIG_INT, 0, comm);

//             send_start = 5;
//             jx_MPI_Scatter(&global_data[send_start], 1, JX_MPI_BIG_INT, &row_starts[1], 1, JX_MPI_BIG_INT, 0, comm);

//             send_start = 4 + (num_procs + 1);
//             jx_MPI_Scatter(&global_data[send_start], 1, JX_MPI_BIG_INT, &col_starts[0], 1, JX_MPI_BIG_INT, 0, comm);

//             send_start = 5 + (num_procs + 1);
//             jx_MPI_Scatter(&global_data[send_start], 1, JX_MPI_BIG_INT, &col_starts[1], 1, JX_MPI_BIG_INT, 0, comm);
//         } else if ((global_data[3] == 0) || (global_data[3] == 1)) {
//             send_start = 4;
//             jx_MPI_Scatter(&global_data[send_start], 1, JX_MPI_BIG_INT, &row_starts[0], 1, JX_MPI_BIG_INT, 0, comm);

//             send_start = 5;
//             jx_MPI_Scatter(&global_data[send_start], 1, JX_MPI_BIG_INT, &row_starts[1], 1, JX_MPI_BIG_INT, 0, comm);

//             if (global_data[3] == 0) {
//                 col_starts[0] = row_starts[0];
//                 col_starts[1] = row_starts[1];
//             }
//         } else {
//             send_start = 4;
//             jx_MPI_Scatter(&global_data[send_start], 1, JX_MPI_BIG_INT, &col_starts[0], 1, JX_MPI_BIG_INT, 0, comm);

//             send_start = 5;
//             jx_MPI_Scatter(&global_data[send_start], 1, JX_MPI_BIG_INT, &col_starts[1], 1, JX_MPI_BIG_INT, 0, comm);
//         }
//     }
//     jx_TFree(global_data);

//     // 在创建parbsr_A前添加
//     printf("Rank %d: Creating ParBSRMatrix, block_size=%d, global_num_rows=%lld\n", 
//         my_id, block_size, (long long)global_num_rows);
//     printf("Rank %d: row_starts[0]=%lld, row_starts[1]=%lld\n", 
//         my_id, (long long)row_starts[0], (long long)row_starts[1]);
//     printf("Rank %d: col_starts[0]=%lld, col_starts[1]=%lld\n", 
//         my_id, (long long)col_starts[0], (long long)col_starts[1]);
//     fflush(stdout);

//     // Create ParBSR matrix
//     parbsr_A = jx_ParBSRMatrixCreate(comm, block_size, global_num_rows, global_num_cols, 
//                                      row_starts, col_starts, 0, 0, 0);

//     printf("Rank %d: Created parbsr_A at %p\n", my_id, (void*)parbsr_A);
//     if (parbsr_A) {
//         printf("Rank %d: parbsr_A->diag = %p\n", my_id, (void*)jx_ParBSRMatrixDiag(parbsr_A));
//         printf("Rank %d: parbsr_A->offd = %p\n", my_id, (void*)jx_ParBSRMatrixOffd(parbsr_A));
//         printf("Rank %d: parbsr_A->block_size = %d\n", my_id, jx_ParBSRMatrixBlockSize(parbsr_A));
//     }
//     fflush(stdout);

//     // Allocate memory for building ParBSR matrix
//     num_rows_proc     = NULL;
//     num_nonzeros_proc = NULL;
//     if (my_id == 0) {
//         num_rows_proc     = jx_CTAlloc(JX_Int, num_procs);
//         num_nonzeros_proc = jx_CTAlloc(JX_Int, num_procs);
//     }

//     if (my_id == 0) {
//         if (!global_row_starts) {
//             jx_GeneratePartitioning(global_num_rows, num_procs, &global_row_starts);
//             free_global_row_starts = 1;
//         }
//         if (!global_col_starts) {
//             jx_GeneratePartitioning(global_num_rows, num_procs, &global_col_starts);
//             free_global_col_starts = 1;
//         }

//         for (i = 0; i < num_procs; i++) {
//             num_rows_proc[i]     = (JX_Int)(global_row_starts[i + 1] - global_row_starts[i]);
//             num_nonzeros_proc[i] = A_i[(JX_Int)global_row_starts[i + 1]] - A_i[(JX_Int)global_row_starts[i]];
//         }
//     }
//     jx_MPI_Scatter(num_rows_proc, 1, JX_MPI_INT, &num_rows, 1, JX_MPI_INT, 0, comm);
//     jx_MPI_Scatter(num_nonzeros_proc, 1, JX_MPI_INT, &num_nonzeros, 1, JX_MPI_INT, 0, comm);

//     local_A = jx_BSRMatrixCreate(block_size, num_rows, (JX_Int)global_num_cols, num_nonzeros);

//     bsr_matrix_datatypes = jx_CTAlloc(MPI_Datatype, num_procs);
//     if (my_id == 0) {
//         requests = jx_CTAlloc(MPI_Request, num_procs - 1);
//         status   = jx_CTAlloc(MPI_Status, num_procs - 1);
//         for (i = 1; i < num_procs; i++) {
//             ind = A_i[(JX_Int)global_row_starts[i]];

//             jx_BuildBSRMatrixMPIDataType(block_size, num_nonzeros_proc[i], num_rows_proc[i], 
//                                          &A_data[ind * block_size * block_size],
//                                          &A_i[(JX_Int)global_row_starts[i]], &A_j[ind], 
//                                          &bsr_matrix_datatypes[i]);
//             jx_MPI_Isend(MPI_BOTTOM, 1, bsr_matrix_datatypes[i], i, 0, comm, &requests[i - 1]);
//             jx_MPI_Type_free(&bsr_matrix_datatypes[i]);
//         }
//         jx_BSRMatrixData(local_A)     = A_data;
//         jx_BSRMatrixI(local_A)        = A_i;
//         jx_BSRMatrixJ(local_A)        = A_j;
//         jx_BSRMatrixOwnsData(local_A) = 0;

//         jx_MPI_Waitall(num_procs - 1, requests, status);

//         jx_TFree(requests);
//         jx_TFree(status);
//         jx_TFree(num_rows_proc);
//         jx_TFree(num_nonzeros_proc);

//         if (free_global_row_starts) {
//             jx_TFree(global_row_starts);
//         }
//         if (free_global_col_starts) {
//             jx_TFree(global_col_starts);
//         }
//     } else {
//         jx_BSRMatrixInitialize(local_A);
//         jx_BuildBSRMatrixMPIDataType(block_size, num_nonzeros, num_rows, jx_BSRMatrixData(local_A), 
//                                      jx_BSRMatrixI(local_A), jx_BSRMatrixJ(local_A), &bsr_matrix_datatypes[0]);
//         jx_MPI_Recv(MPI_BOTTOM, 1, bsr_matrix_datatypes[0], 0, 0, comm, &status0);
//         jx_MPI_Type_free(bsr_matrix_datatypes);
//     }

//     first_col_diag = jx_ParBSRMatrixFirstColDiag(parbsr_A);
//     last_col_diag  = jx_ParBSRMatrixLastColDiag(parbsr_A);

//     jx_GenerateDiagAndOffdBSR(local_A, parbsr_A, first_col_diag, last_col_diag);

//     /* Set pointers back to NULL before destroying */
//     if (my_id == 0) {
//         jx_BSRMatrixData(local_A) = NULL;
//         jx_BSRMatrixI(local_A)    = NULL;
//         jx_BSRMatrixJ(local_A)    = NULL;
//     }
//     jx_BSRMatrixDestroy(local_A);
//     jx_TFree(bsr_matrix_datatypes);

//     return parbsr_A;
// }

/*!
 * \fn JX_Int jx_GenerateDiagAndOffdBSR
 * \brief Generate diagonal and off-diagonal parts for parallel BSR matrix.
 * \date 2025/10/25
 */
JX_Int jx_GenerateDiagAndOffdBSR(jx_BSRMatrix* A, jx_ParBSRMatrix* matrix, JX_BigInt first_col_diag, JX_BigInt last_col_diag)
{

    // printf("DEBUG GenerateDiagAndOffdBSR: A=%p, matrix=%p\n", (void*)A, (void*)matrix);
    // printf("DEBUG first_col_diag=%lld, last_col_diag=%lld\n", 
    //        (long long)first_col_diag, (long long)last_col_diag);
    // fflush(stdout);

    JX_Int      i, j, k;
    JX_Int      jo, jd;
    JX_Int      num_rows = jx_BSRMatrixNumRows(A);
    JX_Int      num_cols = jx_BSRMatrixNumCols(A); // global_column
    JX_Real*    a_data   = jx_BSRMatrixData(A);
    JX_Int*     a_i      = jx_BSRMatrixI(A);
    JX_Int*     a_j      = jx_BSRMatrixJ(A);

    jx_BSRMatrix* diag              = jx_ParBSRMatrixDiag(matrix);
    jx_BSRMatrix* offd              = jx_ParBSRMatrixOffd(matrix);
    JX_Int        diag_num_rows     = jx_BSRMatrixNumRows(diag);
    JX_Int        diag_num_nonzeros = jx_BSRMatrixNumNonzeros(diag);
    JX_Int        offd_num_rows     = jx_BSRMatrixNumRows(offd);
    JX_Int        offd_num_nonzeros = jx_BSRMatrixNumNonzeros(offd);

    JX_BigInt* col_map_offd;

    JX_Real *diag_data, *offd_data;
    JX_Int * diag_i, *offd_i;
    JX_Int * diag_j, *offd_j;
    JX_Int*  marker;
    JX_Int   num_cols_diag, num_cols_offd;
    JX_Int   first_elmt   = a_i[0];
    JX_Int   num_nonzeros = a_i[num_rows] - first_elmt;
    JX_Int   counter;
    JX_Int   block_size = jx_BSRMatrixBlockSize(diag);
    JX_Int   bnnz       = block_size * block_size;

    num_cols_diag = (JX_Int)(last_col_diag - first_col_diag + 1);
    num_cols_offd = 0;

    if (num_cols - num_cols_diag) {
        // printf("DEBUG: Creating separate diag/offd matrices\n");
        // fflush(stdout);

        jx_BSRMatrixI(diag) = jx_CTAlloc(JX_Int, diag_num_rows + 1);
        diag_i  = jx_BSRMatrixI(diag);

        jx_BSRMatrixI(offd) = jx_CTAlloc(JX_Int, offd_num_rows + 1);
        offd_i  = jx_BSRMatrixI(offd);
        marker  = jx_CTAlloc(JX_Int, num_cols);

        for (i = 0; i < num_cols; i++) {
            marker[i] = 0;
        }

        jo = 0;
        jd = 0;
        for (i = 0; i < num_rows; i++) {
            offd_i[i] = jo;
            diag_i[i] = jd;

            for (j = a_i[i] - first_elmt; j < a_i[i + 1] - first_elmt; j++) {
                if (a_j[j] < (JX_Int)first_col_diag || a_j[j] > (JX_Int)last_col_diag) {
                    if (!marker[a_j[j]]) {
                        marker[a_j[j]] = 1;
                        num_cols_offd++;
                    }
                    jo++;
                } else {
                    jd++;
                }
            }
        }
        offd_i[num_rows] = jo;
        diag_i[num_rows] = jd;

        jx_ParBSRMatrixColMapOffd(matrix) = jx_CTAlloc(JX_BigInt, num_cols_offd);
        col_map_offd         = jx_ParBSRMatrixColMapOffd(matrix);

        counter = 0;
        for (i = 0; i < num_cols; i++) {
            if (marker[i]) {
                col_map_offd[counter] = (JX_BigInt)i;
                marker[i]             = counter;
                counter++;
            }
        }

        jx_BSRMatrixNumNonzeros(diag) = jd;
        jx_BSRMatrixJ(diag)            = jx_CTAlloc(JX_Int, jd);
        jx_BSRMatrixData(diag)         = jx_CTAlloc(JX_Real, jd * bnnz);
        diag_data          = jx_BSRMatrixData(diag);
        diag_j             = jx_BSRMatrixJ(diag);

        jx_BSRMatrixNumNonzeros(offd) = jo;
        jx_BSRMatrixNumCols(offd)     = num_cols_offd;
        jx_BSRMatrixJ(offd)            = jx_CTAlloc(JX_Int, jo);
        jx_BSRMatrixData(offd)         = jx_CTAlloc(JX_Real, jo * bnnz);
        offd_data          = jx_BSRMatrixData(offd);
        offd_j             = jx_BSRMatrixJ(offd);

        jo = 0;
        jd = 0;
        for (i = 0; i < num_rows; i++) {
            for (j = a_i[i] - first_elmt; j < a_i[i + 1] - first_elmt; j++) {
                if (a_j[j] < (JX_Int)first_col_diag || a_j[j] > (JX_Int)last_col_diag) {
                    for (k = 0; k < bnnz; k++) 
                        offd_data[jo * bnnz + k] = a_data[j * bnnz + k];
                    offd_j[jo++] = marker[a_j[j]];
                } else {
                    for (k = 0; k < bnnz; k++) 
                        diag_data[jd * bnnz + k] = a_data[j * bnnz + k];
                    diag_j[jd++] = (JX_Int)(a_j[j] - first_col_diag);
                }
            }
        }
        jx_TFree(marker);
    } else {
        // printf("DEBUG: Using original matrix data directly\n");
        // printf("DEBUG: num_cols=%d, num_cols_diag=%d\n", num_cols, num_cols_diag);
        // fflush(stdout);

        jx_BSRMatrixNumNonzeros(diag) = num_nonzeros;
        jx_BSRMatrixInitialize(diag);
        diag_data = jx_BSRMatrixData(diag);
        diag_i    = jx_BSRMatrixI(diag);
        diag_j    = jx_BSRMatrixJ(diag);

        for (i = 0; i < num_nonzeros; i++) {
            for (k = 0; k < bnnz; k++) 
                diag_data[i * bnnz + k] = a_data[i * bnnz + k];
            diag_j[i] = a_j[i];
        }
        offd_i = jx_CTAlloc(JX_Int, num_rows + 1);

        for (i = 0; i < num_rows + 1; i++) {
            diag_i[i] = a_i[i];
            offd_i[i] = 0;
        }

        jx_BSRMatrixNumCols(offd) = 0;
        jx_BSRMatrixI(offd)        = offd_i;
    }

    return 0;
}


/*!
 * \fn JX_Int jx_ParBSRMatrixGetRowPartitioning
 * \brief Get the row partitioning for a parallel BSR matrix.
 * \note For BSR matrices, row partitioning should be in scalar rows, not block rows.
 * \author peghoty
 * \date 2009/07/10
 * \modified 2025/10/08 (updated for BSR format)
 */ 
JX_Int jx_ParBSRMatrixGetRowPartitioning(jx_ParBSRMatrix *matrix, JX_Int **row_partitioning_ptr)
{  
    JX_Int *row_partitioning, *row_starts;
    JX_Int  num_procs,myid, i;
    JX_Int  blk_size;

    if (!matrix) 
    {
        jx_error_in_arg(1);
        return jx_error_flag;
    }

    jx_MPI_Comm_size(jx_ParBSRMatrixComm(matrix), &num_procs);
    jx_MPI_Comm_rank(jx_ParBSRMatrixComm(matrix), &myid);
    row_starts = jx_ParBSRMatrixRowStarts(matrix);
    blk_size = jx_ParBSRMatrixBlockSize(matrix);
    
    if (!row_starts) 
    {
        *row_partitioning_ptr = NULL;
        return -1;
    }
    
    // Allocate memory for scalar row partitioning
    // Length is num_procs + 1, same as block row partitioning
    row_partitioning = jx_CTAlloc(JX_Int, num_procs + 1);
    
    // Convert block row partitioning to scalar row partitioning
    // Each block row contains blk_size scalar rows
    for (i = 0; i < num_procs + 1; i++)
    {
        row_partitioning[i] = row_starts[i] * blk_size;
        // printf("Rank %d: row_partitioning = %d\n", myid, row_partitioning[i]);
    }

    *row_partitioning_ptr = row_partitioning;

    return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParBSRMatrixGetColPartitioning
 * \brief Get the column partitioning for a parallel BSR matrix.
 * \note For BSR matrices, column partitioning should be in scalar columns, not block columns.
 * \date 2025/10/08
 */ 
JX_Int jx_ParBSRMatrixGetColPartitioning(jx_ParBSRMatrix *matrix, JX_Int **col_partitioning_ptr)
{  
    JX_Int *col_partitioning, *col_starts;
    JX_Int  num_procs, i;
    JX_Int  blk_size;

    if (!matrix) 
    {
        jx_error_in_arg(1);
        return jx_error_flag;
    }

    jx_MPI_Comm_size(jx_ParBSRMatrixComm(matrix), &num_procs);
    col_starts = jx_ParBSRMatrixColStarts(matrix);
    blk_size = jx_ParBSRMatrixBlockSize(matrix);
    
    if (!col_starts) 
    {
        *col_partitioning_ptr = NULL;
        return -1;
    }
    
    // Allocate memory for scalar column partitioning
    col_partitioning = jx_CTAlloc(JX_Int, num_procs + 1);
    
    // Convert block column partitioning to scalar column partitioning
    for (i = 0; i < num_procs + 1; i++)
    {
        col_partitioning[i] = col_starts[i] * blk_size;
    }

    *col_partitioning_ptr = col_partitioning;

    return jx_error_flag;
}