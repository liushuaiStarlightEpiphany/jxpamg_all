//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jxf_par_bsr_matrix.c -- member functions for jxf_ParBSRMatrix class.
 *  Date: 2025/10/25
 */ 

#include "jxf_parbsr_mv.h"


/*!
 * \fn jxf_ParBSRMatrix* jxf_ParBSRMatrixCreate
 * \brief Create a parallel block sparse row matrix.
 * \date 2025/10/25
 */

jxf_ParBSRMatrix* 
jxf_ParBSRMatrixCreate( MPI_Comm   comm,
                       JXF_Int        block_size,  
                       JXF_BigInt        global_num_rows,
                       JXF_BigInt        global_num_cols,
                       JXF_BigInt       *row_starts,
                       JXF_BigInt       *col_starts,
                       JXF_Int        num_cols_offd,
                       JXF_Int        num_nonzeros_diag,
                       JXF_Int        num_nonzeros_offd )
{
   jxf_ParBSRMatrix  *matrix;
   
   JXF_Int num_procs, my_id;
   JXF_Int local_num_rows, local_num_cols;
   JXF_BigInt first_row_index, first_col_diag;
   
   matrix = jxf_CTAlloc(jxf_ParBSRMatrix, 1);

   jxf_MPI_Comm_rank(comm, &my_id);
   jxf_MPI_Comm_size(comm, &num_procs);

   if (!row_starts)
   {
      jxf_GeneratePartitioning(global_num_rows, num_procs, &row_starts);
   }
   
   if (!col_starts)
   {
      if (global_num_rows == global_num_cols)
      {
         col_starts = row_starts;
      }
      else
      {
         jxf_GeneratePartitioning(global_num_cols, num_procs, &col_starts);
      }
   }

   first_row_index = row_starts[my_id];
   local_num_rows  = row_starts[my_id+1] - first_row_index;
   first_col_diag  = col_starts[my_id];
   local_num_cols  = col_starts[my_id+1] - first_col_diag;

   jxf_ParBSRMatrixComm(matrix) = comm;
   jxf_ParBSRMatrixDiag(matrix) = jxf_BSRMatrixCreate(block_size, local_num_rows, local_num_cols, num_nonzeros_diag);
   jxf_ParBSRMatrixOffd(matrix) = jxf_BSRMatrixCreate(block_size, local_num_rows, num_cols_offd, num_nonzeros_offd);
   jxf_ParBSRMatrixGlobalNumRows(matrix) = global_num_rows;
   jxf_ParBSRMatrixGlobalNumCols(matrix) = global_num_cols;
   jxf_ParBSRMatrixFirstRowIndex(matrix) = first_row_index;
   jxf_ParBSRMatrixFirstColDiag(matrix)  = first_col_diag;
 
   jxf_ParBSRMatrixLastRowIndex(matrix) = first_row_index + (JXF_BigInt)local_num_rows - 1;
   jxf_ParBSRMatrixLastColDiag(matrix)  = first_col_diag  + (JXF_BigInt)local_num_cols - 1;

   jxf_ParBSRMatrixColMapOffd(matrix)       = NULL;
   jxf_ParBSRMatrixAssumedPartition(matrix) = NULL;

   jxf_ParBSRMatrixRowStarts(matrix) = row_starts;
   jxf_ParBSRMatrixColStarts(matrix) = col_starts;

//    jxf_ParBSRMatrixDiagT(matrix) = NULL;
//    jxf_ParBSRMatrixOffdT(matrix) = NULL;

   jxf_ParBSRMatrixCommPkg(matrix)  = NULL;
   jxf_ParBSRMatrixCommPkgT(matrix) = NULL;

   /* set defaults */
   jxf_ParBSRMatrixOwnsData(matrix) = 1;
   jxf_ParBSRMatrixOwnsRowStarts(matrix) = 1;
   jxf_ParBSRMatrixOwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
   {
      jxf_ParBSRMatrixOwnsColStarts(matrix) = 0;
   }
   jxf_ParBSRMatrixRowindices(matrix)   = NULL;
   jxf_ParBSRMatrixRowvalues(matrix)    = NULL;
   jxf_ParBSRMatrixGetrowactive(matrix) = 0;

   return matrix;
}


// jxf_ParBSRMatrix* jxf_ParBSRMatrixCreate(MPI_Comm comm, JXF_Int block_size, JXF_BigInt global_num_rows, JXF_BigInt global_num_cols,
//                                        JXF_BigInt* row_starts_in, JXF_BigInt* col_starts_in, JXF_Int num_cols_offd,
//                                        JXF_Int num_nonzeros_diag, JXF_Int num_nonzeros_offd)
// {
//     jxf_ParBSRMatrix* matrix;
//     JXF_Int           num_procs, my_id;
//     JXF_Int           local_num_rows;
//     JXF_Int           local_num_cols;
//     JXF_BigInt        first_row_index, first_col_diag;
//     JXF_BigInt        row_starts[2];
//     JXF_BigInt        col_starts[2];

//     matrix = jxf_CTAlloc(jxf_ParBSRMatrix, 1);

//     jxf_MPI_Comm_rank(comm, &my_id);
//     jxf_MPI_Comm_size(comm, &num_procs);

//     if (!row_starts_in) {
//         jxf_GenerateLocalPartitioning(global_num_rows, num_procs, my_id, row_starts);
//     } else {
//         row_starts[0] = row_starts_in[0];
//         row_starts[1] = row_starts_in[1];
//     }

//     if (!col_starts_in) {
//         jxf_GenerateLocalPartitioning(global_num_cols, num_procs, my_id, col_starts);
//     } else {
//         col_starts[0] = col_starts_in[0];
//         col_starts[1] = col_starts_in[1];
//     }

//     /* row_starts[0] is start of local rows.
//        row_starts[1] is start of next processor's rows */
//     first_row_index                 = row_starts[0];
//     local_num_rows                  = (JXF_Int)(row_starts[1] - first_row_index);
//     first_col_diag                  = col_starts[0];
//     local_num_cols                  = (JXF_Int)(col_starts[1] - first_col_diag);
//     jxf_ParBSRMatrixComm(matrix) = comm;
//     jxf_ParBSRMatrixDiag(matrix) = jxf_BSRMatrixCreate(block_size, local_num_rows, local_num_cols, num_nonzeros_diag);
//     jxf_ParBSRMatrixOffd(matrix) = jxf_BSRMatrixCreate(block_size, local_num_rows, num_cols_offd, num_nonzeros_offd);

//     jxf_ParBSRMatrixBlockSize(matrix)        = block_size;
//     jxf_ParBSRMatrixGlobalNumRows(matrix)    = global_num_rows;
//     jxf_ParBSRMatrixGlobalNumCols(matrix)    = global_num_cols;
//     jxf_ParBSRMatrixFirstRowIndex(matrix)    = first_row_index;
//     jxf_ParBSRMatrixFirstColDiag(matrix)     = first_col_diag;
//     jxf_ParBSRMatrixLastRowIndex(matrix)     = first_row_index + (JXF_BigInt)local_num_rows - 1;
//     jxf_ParBSRMatrixLastColDiag(matrix)      = first_col_diag + (JXF_BigInt)local_num_cols - 1;
//     jxf_ParBSRMatrixRowStarts(matrix)[0]     = row_starts[0];
//     jxf_ParBSRMatrixRowStarts(matrix)[1]     = row_starts[1];
//     jxf_ParBSRMatrixColStarts(matrix)[0]     = col_starts[0];
//     jxf_ParBSRMatrixColStarts(matrix)[1]     = col_starts[1];
//     jxf_ParBSRMatrixColMapOffd(matrix)       = NULL;
//     jxf_ParBSRMatrixCommPkg(matrix)          = NULL;
//     jxf_ParBSRMatrixCommPkgT(matrix)         = NULL;
//     jxf_ParBSRMatrixAssumedPartition(matrix) = NULL;
//    //  jxf_ParBSRMatrixDiagT(matrix)            = NULL;
//    //  jxf_ParBSRMatrixOffdT(matrix)            = NULL;

//     /* set defaults */
//     jxf_ParBSRMatrixOwnsData(matrix) = 1;

//     return matrix;
// }

/*!
 * \fn JXF_Int jxf_ParBSRMatrixDestroy
 * \brief Destroy a parallel block sparse row matrix.
 * \date 2025/10/25
 */
JXF_Int jxf_ParBSRMatrixDestroy(jxf_ParBSRMatrix* matrix)
{
    if (matrix) {
        if (jxf_ParBSRMatrixOwnsData(matrix)) {
            jxf_BSRMatrixDestroy(jxf_ParBSRMatrixDiag(matrix));
            jxf_BSRMatrixDestroy(jxf_ParBSRMatrixOffd(matrix));
            if (jxf_ParBSRMatrixColMapOffd(matrix)) {
                jxf_TFree(jxf_ParBSRMatrixColMapOffd(matrix));
            }
            if (jxf_ParBSRMatrixCommPkg(matrix)) {
                jxf_MatvecCommPkgDestroy((jxf_ParCSRCommPkg*)jxf_ParBSRMatrixCommPkg(matrix));
            }
            if (jxf_ParBSRMatrixCommPkgT(matrix)) {
                jxf_MatvecCommPkgDestroy((jxf_ParCSRCommPkg*)jxf_ParBSRMatrixCommPkgT(matrix));
            }
        }

        // if (jxf_ParBSRMatrixDiagT(matrix)) {
        //     jxf_BSRMatrixDestroy(jxf_ParBSRMatrixDiagT(matrix));
        // }
        // if (jxf_ParBSRMatrixOffdT(matrix)) {
        //     jxf_BSRMatrixDestroy(jxf_ParBSRMatrixOffdT(matrix));
        // }

        if (jxf_ParBSRMatrixAssumedPartition(matrix)) {
            jxf_ParBSRMatrixDestroyAssumedPartition(matrix);
        }

        jxf_TFree(matrix);
    }

    return 0;
}

/*!
 * \fn JXF_Int jxf_ParBSRMatrixInitialize
 * \brief Initialize a parallel block sparse row matrix.
 * \date 2025/10/25
 */
JXF_Int jxf_ParBSRMatrixInitialize(jxf_ParBSRMatrix* matrix)
{
    JXF_Int ierr = 0;

    jxf_BSRMatrixInitialize(jxf_ParBSRMatrixDiag(matrix));
    jxf_BSRMatrixInitialize(jxf_ParBSRMatrixOffd(matrix));
    jxf_ParBSRMatrixColMapOffd(matrix) = jxf_CTAlloc(JXF_BigInt, jxf_BSRMatrixNumCols(jxf_ParBSRMatrixOffd(matrix)));

    return ierr;
}

/*!
 * \fn JXF_Int jxf_ParBSRMatrixSetNumNonzeros
 * \brief Set the total number of nonzeros in parallel BSR matrix.
 * \date 2025/10/25
 */
JXF_Int jxf_ParBSRMatrixSetNumNonzeros(jxf_ParBSRMatrix* matrix)
{
    MPI_Comm          comm           = jxf_ParBSRMatrixComm(matrix);
    jxf_BSRMatrix*     diag           = jxf_ParBSRMatrixDiag(matrix);
    JXF_Int*           diag_i         = jxf_BSRMatrixI(diag);
    jxf_BSRMatrix*     offd           = jxf_ParBSRMatrixOffd(matrix);
    JXF_Int*           offd_i         = jxf_BSRMatrixI(offd);
    JXF_Int            local_num_rows = jxf_BSRMatrixNumRows(diag);
    JXF_BigInt         total_num_nonzeros;
    JXF_BigInt         local_num_nonzeros;
    JXF_Int            ierr = 0;

    local_num_nonzeros = (JXF_BigInt)(diag_i[local_num_rows] + offd_i[local_num_rows]);
    jxf_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, JXF_MPI_INT, MPI_SUM, comm);
    jxf_ParBSRMatrixNumNonzeros(matrix) = total_num_nonzeros;

    return ierr;
}

/*!
 * \fn JXF_Int jxf_ParBSRMatrixSetDNumNonzeros
 * \brief Set the total number of nonzeros (as double) in parallel BSR matrix.
 * \date 2025/10/25
 */
JXF_Int jxf_ParBSRMatrixSetDNumNonzeros(jxf_ParBSRMatrix* matrix)
{
    MPI_Comm          comm           = jxf_ParBSRMatrixComm(matrix);
    jxf_BSRMatrix*     diag           = jxf_ParBSRMatrixDiag(matrix);
    JXF_Int*           diag_i         = jxf_BSRMatrixI(diag);
    jxf_BSRMatrix*     offd           = jxf_ParBSRMatrixOffd(matrix);
    JXF_Int*           offd_i         = jxf_BSRMatrixI(offd);
    JXF_Int            local_num_rows = jxf_BSRMatrixNumRows(diag);
    JXF_Real           total_num_nonzeros;
    JXF_Real           local_num_nonzeros;
    JXF_Int            ierr = 0;

    local_num_nonzeros = (JXF_Real)diag_i[local_num_rows] + (JXF_Real)offd_i[local_num_rows];
    jxf_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, JXF_MPI_REAL, MPI_SUM, comm);
    jxf_ParBSRMatrixDNumNonzeros(matrix) = total_num_nonzeros;

    return ierr;
}

/*!
 * \fn JXF_Int jxf_ParBSRMatrixSetDataOwner
 * \brief Set data ownership flag for parallel BSR matrix.
 * \date 2025/10/25
 */
JXF_Int jxf_ParBSRMatrixSetDataOwner(jxf_ParBSRMatrix* matrix, JXF_Int owns_data)
{
    JXF_Int ierr = 0;

    jxf_ParBSRMatrixOwnsData(matrix) = owns_data;

    return ierr;
}

/*!
 * \fn jxf_ParCSRMatrix* jxf_ParBSRMatrixCompress
 * \brief Compress BSR matrix to CSR by taking F-norm for each sub-block.
 * \date 2025/10/25
 */
jxf_ParCSRMatrix* jxf_ParBSRMatrixCompress(jxf_ParBSRMatrix* matrix)
{
    MPI_Comm          comm              = jxf_ParBSRMatrixComm(matrix);
    jxf_BSRMatrix*     diag              = jxf_ParBSRMatrixDiag(matrix);
    jxf_BSRMatrix*     offd              = jxf_ParBSRMatrixOffd(matrix);
    JXF_BigInt         global_num_rows   = jxf_ParBSRMatrixGlobalNumRows(matrix);
    JXF_BigInt         global_num_cols   = jxf_ParBSRMatrixGlobalNumCols(matrix);
    JXF_BigInt*        row_starts        = jxf_ParBSRMatrixRowStarts(matrix);
    JXF_BigInt*        col_starts        = jxf_ParBSRMatrixColStarts(matrix);
    JXF_Int            num_cols_offd     = jxf_BSRMatrixNumCols(offd);
    JXF_Int            num_nonzeros_diag = jxf_BSRMatrixNumNonzeros(diag);
    JXF_Int            num_nonzeros_offd = jxf_BSRMatrixNumNonzeros(offd);

    jxf_ParCSRMatrix* matrix_C;
    JXF_Int i;

    matrix_C = jxf_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols, row_starts, col_starts, 
                                     num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
    jxf_ParCSRMatrixInitialize(matrix_C);

    jxf_CSRMatrixDestroy(jxf_ParCSRMatrixDiag(matrix_C));
    jxf_ParCSRMatrixDiag(matrix_C) = jxf_BSRMatrixCompress(diag);
    jxf_CSRMatrixDestroy(jxf_ParCSRMatrixOffd(matrix_C));
    jxf_ParCSRMatrixOffd(matrix_C) = jxf_BSRMatrixCompress(offd);

    for (i = 0; i < num_cols_offd; i++) 
        jxf_ParCSRMatrixColMapOffd(matrix_C)[i] = jxf_ParBSRMatrixColMapOffd(matrix)[i];
    
    return matrix_C;
}

/*!
 * \fn jxf_ParCSRMatrix* jxf_ParBSRMatrixGetSubmatrix
 * \brief Get submatrix from parallel BSR matrix.
 * \date 2025/10/25
 */
jxf_ParCSRMatrix* jxf_ParBSRMatrixGetSubmatrix(jxf_ParBSRMatrix* matrix, JXF_Int subposition, JXF_Real threshold)
{


    MPI_Comm          comm              = jxf_ParBSRMatrixComm(matrix);
    jxf_BSRMatrix*     diag              = jxf_ParBSRMatrixDiag(matrix);
    jxf_BSRMatrix*     offd              = jxf_ParBSRMatrixOffd(matrix);
    JXF_BigInt         global_num_rows   = jxf_ParBSRMatrixGlobalNumRows(matrix);
    JXF_BigInt         global_num_cols   = jxf_ParBSRMatrixGlobalNumCols(matrix);
    JXF_BigInt*        row_starts        = jxf_ParBSRMatrixRowStarts(matrix);
    JXF_BigInt*        col_starts        = jxf_ParBSRMatrixColStarts(matrix);
    JXF_Int            num_cols_offd     = jxf_BSRMatrixNumCols(offd);
    JXF_Int            num_nonzeros_diag = jxf_BSRMatrixNumNonzeros(diag);
    JXF_Int            num_nonzeros_offd = jxf_BSRMatrixNumNonzeros(offd);

    jxf_ParCSRMatrix* matrix_C;
    JXF_Int i;


    matrix_C = jxf_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols, row_starts, col_starts,
                                     num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
    jxf_ParCSRMatrixInitialize(matrix_C);



    jxf_CSRMatrixDestroy(jxf_ParCSRMatrixDiag(matrix_C));
    jxf_ParCSRMatrixDiag(matrix_C) = jxf_BSRMatrixGetSubmatrix(diag, subposition);



    jxf_CSRMatrixDestroy(jxf_ParCSRMatrixOffd(matrix_C));
    jxf_ParCSRMatrixOffd(matrix_C) = jxf_BSRMatrixGetSubmatrix(offd, subposition);



    for (i = 0; i < num_cols_offd; i++) 
        jxf_ParCSRMatrixColMapOffd(matrix_C)[i] = jxf_ParBSRMatrixColMapOffd(matrix)[i];



    // Drop Small Entries
    JXF_Int type = 0;
    /*
     *    type 0: threshold
     *    type 1: threshold*(1-norm of row)
     *    type 2: threshold*(2-norm of row)
     *    type -1: threshold*(infinity norm of row)
     */
    // jxf_ParCSRMatrixDropSmallEntries(matrix_C, threshold, type);

    // printf(" [%s:%d] matrix_C = %p \n",__FUNCTION__, __LINE__, (void*)matrix_C);
    // fflush(stdout);  
    // printf(" [%s:%d] jxf_ParCSRMatrixComm(matrix_C) = %p \n",__FUNCTION__, __LINE__, (void*)jxf_ParCSRMatrixComm(matrix_C));
    // fflush(stdout);  
    // printf(" [%s:%d] jxf_ParCSRMatrixDiag(ParCSR) = %p \n",__FUNCTION__, __LINE__, (void*)jxf_ParCSRMatrixDiag(matrix_C));
    // fflush(stdout);  
    // printf(" [%s:%d] jxf_CSRMatrixI(ParCSR) = %p \n",__FUNCTION__, __LINE__, (void*)jxf_CSRMatrixI(jxf_ParCSRMatrixDiag(matrix_C)));
    // fflush(stdout);  

    return matrix_C;
}

/*!
 * \fn jxf_ParCSRMatrix* jxf_ParBSRMatrixConvertToParCSRMatrix
 * \brief Convert parallel BSR matrix to parallel CSR matrix.
 * \date 2025/10/25
 */
jxf_ParCSRMatrix* jxf_ParBSRMatrixConvertToParCSRMatrix(jxf_ParBSRMatrix* matrix)
{
    MPI_Comm          comm              = jxf_ParBSRMatrixComm(matrix);
    jxf_BSRMatrix*     diag              = jxf_ParBSRMatrixDiag(matrix);
    jxf_BSRMatrix*     offd              = jxf_ParBSRMatrixOffd(matrix);
    JXF_Int            block_size        = jxf_ParBSRMatrixBlockSize(matrix);
    JXF_BigInt         global_num_rows   = jxf_ParBSRMatrixGlobalNumRows(matrix);
    JXF_BigInt         global_num_cols   = jxf_ParBSRMatrixGlobalNumCols(matrix);
    JXF_BigInt*        row_starts        = jxf_ParBSRMatrixRowStarts(matrix);
    JXF_BigInt*        col_starts        = jxf_ParBSRMatrixColStarts(matrix);
    JXF_Int            num_cols_offd     = jxf_BSRMatrixNumCols(offd);
    JXF_Int            num_nonzeros_diag = jxf_BSRMatrixNumNonzeros(diag);
    JXF_Int            num_nonzeros_offd = jxf_BSRMatrixNumNonzeros(offd);

    jxf_ParCSRMatrix* matrix_C;
    JXF_BigInt        matrix_C_row_starts[2];
    JXF_BigInt        matrix_C_col_starts[2];

    JXF_Int *   counter, *new_j_map;
    JXF_Int     size_j, size_map, index, new_num_cols, removed = 0;
    JXF_Int*    offd_j;
    JXF_BigInt *col_map_offd, *new_col_map_offd;

    JXF_Int num_procs, i, j;

    jxf_CSRMatrix *diag_nozeros, *offd_nozeros;

    jxf_MPI_Comm_size(comm, &num_procs);

    for (i = 0; i < 2; i++) {
        matrix_C_row_starts[i] = row_starts[i] * (JXF_BigInt)block_size;
        matrix_C_col_starts[i] = col_starts[i] * (JXF_BigInt)block_size;
    }

    matrix_C = jxf_ParCSRMatrixCreate(comm, global_num_rows * (JXF_BigInt)block_size, 
                                     global_num_cols * (JXF_BigInt)block_size, 
                                     matrix_C_row_starts, matrix_C_col_starts,
                                     num_cols_offd * block_size, 
                                     num_nonzeros_diag * block_size * block_size,
                                     num_nonzeros_offd * block_size * block_size);
    jxf_ParCSRMatrixInitialize(matrix_C);

    /* DIAG */
    jxf_CSRMatrixDestroy(jxf_ParCSRMatrixDiag(matrix_C));
    jxf_ParCSRMatrixDiag(matrix_C) = jxf_BSRMatrixConvertToCSRMatrix(diag);

    /* Delete zeros */
#if 1
    diag_nozeros = jxf_CSRMatrixDeleteZeros(jxf_ParCSRMatrixDiag(matrix_C), 1e-14);
    if (diag_nozeros) {
        jxf_CSRMatrixDestroy(jxf_ParCSRMatrixDiag(matrix_C));
        jxf_ParCSRMatrixDiag(matrix_C) = diag_nozeros;
    }
#endif

    /* OFF-DIAG */
    jxf_CSRMatrixDestroy(jxf_ParCSRMatrixOffd(matrix_C));
    jxf_ParCSRMatrixOffd(matrix_C) = jxf_BSRMatrixConvertToCSRMatrix(offd);

#if 1
    /* Delete zeros */
    offd_nozeros = jxf_CSRMatrixDeleteZeros(jxf_ParCSRMatrixOffd(matrix_C), 1e-14);
    if (offd_nozeros) {
        jxf_CSRMatrixDestroy(jxf_ParCSRMatrixOffd(matrix_C));
        jxf_ParCSRMatrixOffd(matrix_C) = offd_nozeros;
        removed = 1;
    }
#endif

    /* Convert the col_map_offd */
    for (i = 0; i < num_cols_offd; i++)
        for (j = 0; j < block_size; j++)
            jxf_ParCSRMatrixColMapOffd(matrix_C)[i * block_size + j] = 
                jxf_ParBSRMatrixColMapOffd(matrix)[i] * (JXF_BigInt)block_size + (JXF_BigInt)j;

    /* If we deleted zeros, compress col_map_offd */
    if (removed) {
        size_map  = num_cols_offd * block_size;
        counter   = jxf_CTAlloc(JXF_Int, size_map);
        new_j_map = jxf_CTAlloc(JXF_Int, size_map);

        offd_j       = jxf_CSRMatrixJ(jxf_ParCSRMatrixOffd(matrix_C));
        col_map_offd = jxf_ParCSRMatrixColMapOffd(matrix_C);

        size_j = jxf_CSRMatrixNumNonzeros(jxf_ParCSRMatrixOffd(matrix_C));
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
            new_col_map_offd = jxf_CTAlloc(JXF_BigInt, new_num_cols);
            index = 0;
            for (i = 0; i < size_map; i++) {
                if (counter[i]) {
                    new_col_map_offd[index++] = col_map_offd[i];
                }
            }
            /* Set the new col map */
            jxf_TFree(col_map_offd);
            jxf_ParCSRMatrixColMapOffd(matrix_C) = new_col_map_offd;
            /* Modify the number of cols */
            jxf_CSRMatrixNumCols(jxf_ParCSRMatrixOffd(matrix_C)) = new_num_cols;
        }
        jxf_TFree(new_j_map);
        jxf_TFree(counter);
    }

    jxf_ParCSRMatrixSetNumNonzeros(matrix_C);
    jxf_ParCSRMatrixSetDNumNonzeros(matrix_C);

    /* We will not copy the comm package */
    jxf_ParCSRMatrixCommPkg(matrix_C) = NULL;

    return matrix_C;
}

/*!
 * \fn jxf_ParBSRMatrix* jxf_ParBSRMatrixConvertFromParCSRMatrix
 * \brief Convert parallel CSR matrix to parallel BSR matrix.
 * \date 2025/10/25
 */
jxf_ParBSRMatrix* jxf_ParBSRMatrixConvertFromParCSRMatrix(jxf_ParCSRMatrix* matrix, JXF_Int matrix_C_block_size)
{
    MPI_Comm          comm            = jxf_ParCSRMatrixComm(matrix);
    jxf_CSRMatrix*     diag            = jxf_ParCSRMatrixDiag(matrix);
    jxf_CSRMatrix*     offd            = jxf_ParCSRMatrixOffd(matrix);
    JXF_BigInt         global_num_rows = jxf_ParCSRMatrixGlobalNumRows(matrix);
    JXF_BigInt         global_num_cols = jxf_ParCSRMatrixGlobalNumCols(matrix);
    JXF_BigInt*        row_starts      = jxf_ParCSRMatrixRowStarts(matrix);
    JXF_BigInt*        col_starts      = jxf_ParCSRMatrixColStarts(matrix);
    JXF_Int            num_cols_offd   = jxf_CSRMatrixNumCols(offd);
    JXF_BigInt*        col_map_offd    = jxf_ParCSRMatrixColMapOffd(matrix);
    JXF_BigInt*        map_to_node     = NULL;
    JXF_Int *          counter = NULL, *col_in_j_map = NULL;
    JXF_BigInt*        matrix_C_col_map_offd = NULL;

    JXF_Int matrix_C_num_cols_offd;
    JXF_Int matrix_C_num_nonzeros_offd;
    JXF_Int num_rows, num_nodes, num_error;

    JXF_Int*     offd_i    = jxf_CSRMatrixI(offd);
    JXF_Int*     offd_j    = jxf_CSRMatrixJ(offd);
    JXF_Real*    offd_data = jxf_CSRMatrixData(offd);

    jxf_ParBSRMatrix* matrix_C;
    JXF_BigInt        matrix_C_row_starts[2];
    JXF_BigInt        matrix_C_col_starts[2];
    jxf_BSRMatrix*    matrix_C_diag;
    jxf_BSRMatrix*    matrix_C_offd;

    JXF_Int *    matrix_C_offd_i = NULL, *matrix_C_offd_j = NULL;
    JXF_Real*    matrix_C_offd_data = NULL;

    JXF_Int rank, num_procs, i, j, k, k_map, count, index, start_index, pos, row;

    jxf_MPI_Comm_rank(comm, &rank);
    jxf_MPI_Comm_size(comm, &num_procs);

    for (i = 0; i < 2; i++) {
        matrix_C_row_starts[i] = row_starts[i] / (JXF_BigInt)matrix_C_block_size;
        matrix_C_col_starts[i] = col_starts[i] / (JXF_BigInt)matrix_C_block_size;
    }

    /* Figure out the new number of offd columns */
    num_cols_offd = offd->num_cols;
    num_rows      = diag->num_rows;
    num_nodes     = num_rows / matrix_C_block_size;
    num_error     = num_rows % matrix_C_block_size;

    if (num_error != 0) {
        jxf_printf("%s, rank: %d, num_rows: %d cannot be divided by block_size: %d\n", 
                  __FUNCTION__, rank, num_rows, matrix_C_block_size);
        return NULL;
    }

    /************* Create the diagonal part ************/
    matrix_C_diag = jxf_BSRMatrixConvertFromCSRMatrix(diag, matrix_C_block_size);

    /******* The offd part *******************/

    /* Can't use the same function for the offd part - because this isn't square
       and the offd j entries aren't global numbering (have to consider the offd map) */
    matrix_C_offd_i = jxf_CTAlloc(JXF_Int, num_nodes + 1);

    matrix_C_num_cols_offd     = 0;
    matrix_C_offd_i[0]         = 0;
    matrix_C_num_nonzeros_offd = 0;

    if (num_cols_offd) {
        map_to_node            = jxf_CTAlloc(JXF_BigInt, num_cols_offd);
        matrix_C_num_cols_offd = 1;
        map_to_node[0]         = col_map_offd[0] / (JXF_BigInt)matrix_C_block_size;
        for (i = 1; i < num_cols_offd; i++) {
            map_to_node[i] = col_map_offd[i] / (JXF_BigInt)matrix_C_block_size;
            if (map_to_node[i] > map_to_node[i - 1]) {
                matrix_C_num_cols_offd++;
            }
        }

        matrix_C_col_map_offd = jxf_CTAlloc(JXF_BigInt, matrix_C_num_cols_offd);
        col_in_j_map          = jxf_CTAlloc(JXF_Int, num_cols_offd);

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
        counter                    = jxf_CTAlloc(JXF_Int, matrix_C_num_cols_offd);
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
    matrix_C_offd = jxf_BSRMatrixCreate(matrix_C_block_size, num_nodes, matrix_C_num_cols_offd, matrix_C_num_nonzeros_offd);

    /* Assign i */
    jxf_BSRMatrixI(matrix_C_offd) = matrix_C_offd_i;

    /* Create (and allocate j and data) */
    if (matrix_C_num_nonzeros_offd) {
        matrix_C_offd_j    = jxf_CTAlloc(JXF_Int, matrix_C_num_nonzeros_offd);
        matrix_C_offd_data = jxf_CTAlloc(JXF_Real, matrix_C_num_nonzeros_offd * matrix_C_block_size * matrix_C_block_size);
        jxf_BSRMatrixJ(matrix_C_offd)    = matrix_C_offd_j;
        jxf_BSRMatrixData(matrix_C_offd) = matrix_C_offd_data;

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
                              (JXF_Int)(col_map_offd[offd_j[k]] % (JXF_BigInt)matrix_C_block_size);
                        matrix_C_offd_data[pos] = offd_data[k];
                        index++;
                    } else /* This col has already been listed for this row */
                    {
                        /* Copy the data: which position + which row + which col */
                        pos = (counter[k_map] * matrix_C_block_size * matrix_C_block_size) + (j * matrix_C_block_size) +
                              (JXF_Int)(col_map_offd[offd_j[k]] % (JXF_BigInt)(matrix_C_block_size));
                        matrix_C_offd_data[pos] = offd_data[k];
                    }
                }
            }
            start_index = index; /* First index for current nodal row */
        }
    }

    /* ********* Create the new matrix  *************/
    matrix_C = jxf_ParBSRMatrixCreate(comm, matrix_C_block_size, 
                                     global_num_rows / (JXF_BigInt)matrix_C_block_size,
                                     global_num_cols / (JXF_BigInt)matrix_C_block_size, 
                                     matrix_C_row_starts, matrix_C_col_starts, 
                                     matrix_C_num_cols_offd,
                                     jxf_BSRMatrixNumNonzeros(matrix_C_diag), 
                                     matrix_C_num_nonzeros_offd);

    /* Use the diag and off diag matrices we have already created */
    jxf_BSRMatrixDestroy(jxf_ParBSRMatrixDiag(matrix_C));
    jxf_ParBSRMatrixDiag(matrix_C) = matrix_C_diag;
    jxf_BSRMatrixDestroy(jxf_ParBSRMatrixOffd(matrix_C));
    jxf_ParBSRMatrixOffd(matrix_C) = matrix_C_offd;

    jxf_ParBSRMatrixColMapOffd(matrix_C) = matrix_C_col_map_offd;

    /* ********* Don't bother to copy the comm_pkg *************/
    jxf_ParBSRMatrixCommPkg(matrix_C) = NULL;

    /* CLEAN UP !!!! */
    jxf_TFree(map_to_node);
    jxf_TFree(col_in_j_map);
    jxf_TFree(counter);

    return matrix_C;
}

/*!
 * \fn jxf_ParVector* jxf_ParVectorCreateFromBlock
 * \brief Create a parallel vector from block information.
 * \date 2025/10/25
 */
jxf_ParVector* jxf_ParVectorCreateFromBlock(MPI_Comm comm, JXF_BigInt p_global_size, JXF_BigInt* p_partitioning, JXF_Int block_size)
{
    jxf_ParVector* vector;
    JXF_Int        num_procs, my_id;
    JXF_BigInt     global_size;
    JXF_BigInt     new_partitioning[2]; /* Need to create a new partitioning */

    global_size = p_global_size * (JXF_BigInt)block_size;

    vector = jxf_CTAlloc(jxf_ParVector, 1);
    jxf_MPI_Comm_rank(comm, &my_id);
    jxf_MPI_Comm_size(comm, &num_procs);

    if (!p_partitioning) {
        jxf_GenerateLocalPartitioning(global_size, num_procs, my_id, new_partitioning);
    } else /* Adjust for block_size */
    {
        new_partitioning[0] = p_partitioning[0] * (JXF_BigInt)block_size;
        new_partitioning[1] = p_partitioning[1] * (JXF_BigInt)block_size;
    }

    jxf_ParVectorComm(vector) = comm;
    jxf_ParVectorGlobalSize(vector) = global_size;
    jxf_ParVectorFirstIndex(vector) = new_partitioning[0];
    jxf_ParVectorLastIndex(vector) = new_partitioning[1] - 1;
    jxf_ParVectorPartitioning(vector)[0] = new_partitioning[0];
    jxf_ParVectorPartitioning(vector)[1] = new_partitioning[1];
    jxf_ParVectorLocalVector(vector) = jxf_SeqVectorCreate(new_partitioning[1] - new_partitioning[0]);

    /* set defaults */
    jxf_ParVectorOwnsData(vector) = 1;

    return vector;
}

/*!
 * \fn jxf_ParBSRMatrix* jxf_BSRMatrixToParBSRMatrix
 * \brief Generate a ParBSRMatrix distributed across processors from a BSRMatrix on proc 0.
 * \date 2025/10/25
 */

/*!
 * \fn jxf_ParBSRMatrix* jxf_BSRMatrixToParBSRMatrix
 * \brief Generates a ParBSRMatrix distributed across the 
 *        processors in comm from a BSRMatrix on proc 0.
 * \note This shouldn't be used with the JXF_NO_GLOBAL_PARTITON option!  
 * \date 2025/10/25
 */
jxf_ParBSRMatrix* 
jxf_BSRMatrixToParBSRMatrix( MPI_Comm       comm,
                            jxf_BSRMatrix  *A,
                            JXF_BigInt        *row_starts,
                            JXF_BigInt        *col_starts )
{
    JXF_BigInt         *global_data = NULL;
    JXF_BigInt          global_size;
    JXF_BigInt          global_num_rows;
    JXF_BigInt          global_num_cols;
    JXF_Int         *local_num_rows = NULL;

    JXF_Int          num_procs, my_id;
    JXF_Int         *local_num_nonzeros = NULL;
    JXF_Int          num_nonzeros;
   
    JXF_Real       *a_data = NULL;
    JXF_Int        *a_i    = NULL;
    JXF_Int        *a_j    = NULL;
   
    jxf_BSRMatrix *local_A;
    MPI_Request  *requests;
    MPI_Status   *status, status0;
    MPI_Datatype *bsr_matrix_datatypes;
    jxf_ParBSRMatrix *par_matrix;
    JXF_BigInt first_col_diag;
    JXF_BigInt last_col_diag;
    JXF_Int i, j, ind;
    JXF_Int block_size;

    jxf_MPI_Comm_rank(comm, &my_id);
    jxf_MPI_Comm_size(comm, &num_procs);

    global_data = jxf_CTAlloc(JXF_BigInt, 2*num_procs + 6);
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
        
        global_data[0] = jxf_BSRMatrixNumRows(A);
        global_data[1] = jxf_BSRMatrixNumCols(A);
        global_data[2] = global_size;
        block_size = jxf_BSRMatrixBlockSize(A);
        a_data = jxf_BSRMatrixData(A);
        a_i = jxf_BSRMatrixI(A);
        a_j = jxf_BSRMatrixJ(A);
    }
    
    /* Broadcast block size first */
    jxf_MPI_Bcast(&block_size, 1, JXF_MPI_INT, 0, comm);
    
    /* Broadcast global data */
    jxf_MPI_Bcast(global_data, 3, JXF_MPI_INT, 0, comm);
    global_num_rows = global_data[0];
    global_num_cols = global_data[1];
    global_size = global_data[2];
    
    if (global_size > 3)
    {
        jxf_MPI_Bcast(&global_data[3], global_size - 3, JXF_MPI_INT, 0, comm);
        if (my_id > 0)
        {
            if (global_data[3] < 3)
            {
                row_starts = jxf_CTAlloc(JXF_Int, num_procs + 1);
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
                    col_starts = jxf_CTAlloc(JXF_Int, num_procs+1);
                    for (i = 0; i < num_procs + 1; i ++)
                    {
                        col_starts[i] = global_data[i+num_procs+5];
                    }
                }
            }
            else
            {
                col_starts = jxf_CTAlloc(JXF_Int, num_procs + 1);
                for (i = 0; i < num_procs + 1; i ++)
                {
                    col_starts[i] = global_data[i+4];
                }
            }
        }
    }
    jxf_TFree(global_data);

    local_num_rows = jxf_CTAlloc(JXF_Int, num_procs);
    bsr_matrix_datatypes = jxf_CTAlloc(MPI_Datatype, num_procs);

    /* Create parallel BSR matrix with global partitioning */
    par_matrix = jxf_ParBSRMatrixCreate(comm, block_size, global_num_rows, global_num_cols, 
                                       row_starts, col_starts, 0, 0, 0);

    row_starts = jxf_ParBSRMatrixRowStarts(par_matrix);
    col_starts = jxf_ParBSRMatrixColStarts(par_matrix);

    for (i = 0; i < num_procs; i ++)
    {
        local_num_rows[i] = row_starts[i+1] - row_starts[i];
    }

    if (my_id == 0)
    {
        local_num_nonzeros = jxf_CTAlloc(JXF_Int, num_procs);
        for (i = 0; i < num_procs - 1; i ++)
        {
            local_num_nonzeros[i] = a_i[row_starts[i+1]] - a_i[row_starts[i]];
        }
        local_num_nonzeros[num_procs-1] = a_i[global_num_rows] - a_i[row_starts[num_procs-1]];
    }
    jxf_MPI_Scatter(local_num_nonzeros, 1, JXF_MPI_INT, &num_nonzeros, 1, JXF_MPI_INT, 0, comm);

    if (my_id == 0) 
    {
        num_nonzeros = local_num_nonzeros[0];
    }

    local_A = jxf_BSRMatrixCreate(block_size, local_num_rows[my_id], global_num_cols, num_nonzeros);

    if (my_id == 0)
    {
        requests = jxf_CTAlloc(MPI_Request, num_procs - 1);
        status = jxf_CTAlloc(MPI_Status, num_procs - 1);
        j = 0;
        for (i = 1; i < num_procs; i ++)
        {
            ind = a_i[row_starts[i]];
            jxf_BuildBSRMatrixMPIDataType(block_size, local_num_nonzeros[i], 
                                          local_num_rows[i],
                                          &a_data[ind * block_size * block_size],
                                          &a_i[row_starts[i]],
                                          &a_j[ind],
                                          &bsr_matrix_datatypes[i]);
            jxf_MPI_Isend(MPI_BOTTOM, 1, bsr_matrix_datatypes[i], i, 0, comm, &requests[j++]);
            jxf_MPI_Type_free(&bsr_matrix_datatypes[i]);
        }
        jxf_BSRMatrixData(local_A) = a_data;
        jxf_BSRMatrixI(local_A) = a_i;
        jxf_BSRMatrixJ(local_A) = a_j;
        jxf_BSRMatrixOwnsData(local_A) = 0;
        jxf_MPI_Waitall(num_procs-1, requests, status);
        jxf_TFree(requests);
        jxf_TFree(status);
        jxf_TFree(local_num_nonzeros);
    }
    else
    {
        jxf_BSRMatrixInitialize(local_A);
        jxf_BuildBSRMatrixMPIDataType(block_size, num_nonzeros, 
                                      local_num_rows[my_id],
                                      jxf_BSRMatrixData(local_A),
                                      jxf_BSRMatrixI(local_A),
                                      jxf_BSRMatrixJ(local_A),
                                      bsr_matrix_datatypes);
        jxf_MPI_Recv(MPI_BOTTOM, 1, bsr_matrix_datatypes[0], 0, 0, comm, &status0);
        jxf_MPI_Type_free(bsr_matrix_datatypes);
    }

    first_col_diag = col_starts[my_id];
    last_col_diag  = col_starts[my_id+1] - 1;

    jxf_GenerateDiagAndOffdBSR(local_A, par_matrix, first_col_diag, last_col_diag);

    /* set pointers back to NULL before destroying */
    if (my_id == 0)
    {      
        jxf_BSRMatrixData(local_A) = NULL;
        jxf_BSRMatrixI(local_A) = NULL;
        jxf_BSRMatrixJ(local_A) = NULL; 
    }      
    jxf_BSRMatrixDestroy(local_A);
    jxf_TFree(local_num_rows);
    jxf_TFree(bsr_matrix_datatypes);

    return par_matrix;
}

// jxf_ParBSRMatrix* jxf_BSRMatrixToParBSRMatrix(MPI_Comm comm, jxf_BSRMatrix* A, JXF_BigInt* global_row_starts, JXF_BigInt* global_col_starts)
// {


//     jxf_ParBSRMatrix* parbsr_A;

//     JXF_BigInt* global_data;
//     JXF_BigInt  global_size;
//     JXF_BigInt  global_num_rows;
//     JXF_BigInt  global_num_cols;

//     JXF_Int    num_procs, my_id;
//     JXF_Int*   num_rows_proc;
//     JXF_Int*   num_nonzeros_proc;
//     JXF_BigInt row_starts[2];
//     JXF_BigInt col_starts[2];

//     jxf_BSRMatrix* local_A;
//     JXF_Real*      A_data;
//     JXF_Int*       A_i;
//     JXF_Int*       A_j;
//     JXF_Int        block_size = 0;

//     MPI_Request*  requests;
//     MPI_Status *  status, status0;
//     MPI_Datatype* bsr_matrix_datatypes;

//     JXF_Int free_global_row_starts = 0;
//     JXF_Int free_global_col_starts = 0;

//     JXF_Int    total_size;
//     JXF_BigInt first_col_diag;
//     JXF_BigInt last_col_diag;
//     JXF_Int    num_rows;
//     JXF_Int    num_nonzeros;
//     JXF_Int    i, ind;

//     jxf_MPI_Comm_rank(comm, &my_id);
//     jxf_MPI_Comm_size(comm, &num_procs);

//     // 在函数开头添加
//     printf("Rank %d: Entering jxf_BSRMatrixToParBSRMatrix\n", my_id);
//     fflush(stdout);

//     total_size = 4;
//     if (my_id == 0) {
//         total_size += 2 * (num_procs + 1);
//     }

//     global_data = jxf_CTAlloc(JXF_BigInt, total_size);
//     if (my_id == 0) {
//         global_size = 3;
//         if (global_row_starts) {
//             if (global_col_starts) {
//                 if (global_col_starts != global_row_starts) {
//                     global_data[3] = 2;
//                     global_size += (JXF_BigInt)(2 * (num_procs + 1) + 1);
//                     for (i = 0; i < (num_procs + 1); i++) {
//                         global_data[i + 4] = global_row_starts[i];
//                     }
//                     for (i = 0; i < (num_procs + 1); i++) {
//                         global_data[i + num_procs + 5] = global_col_starts[i];
//                     }
//                 } else {
//                     global_data[3] = 0;
//                     global_size += (JXF_BigInt)((num_procs + 1) + 1);
//                     for (i = 0; i < (num_procs + 1); i++) {
//                         global_data[i + 4] = global_row_starts[i];
//                     }
//                 }
//             } else {
//                 global_data[3] = 1;
//                 global_size += (JXF_BigInt)((num_procs + 1) + 1);
//                 for (i = 0; i < (num_procs + 1); i++) {
//                     global_data[i + 4] = global_row_starts[i];
//                 }
//             }
//         } else {
//             if (global_col_starts) {
//                 global_data[3] = 3;
//                 global_size += (JXF_BigInt)((num_procs + 1) + 1);
//                 for (i = 0; i < (num_procs + 1); i++) {
//                     global_data[i + 4] = global_col_starts[i];
//                 }
//             }
//         }

//         global_data[0] = (JXF_BigInt)jxf_BSRMatrixNumRows(A);
//         global_data[1] = (JXF_BigInt)jxf_BSRMatrixNumCols(A);
//         global_data[2] = global_size;
//         A_data     = jxf_BSRMatrixData(A);
//         A_i        = jxf_BSRMatrixI(A);
//         A_j        = jxf_BSRMatrixJ(A);
//         block_size = jxf_BSRMatrixBlockSize(A);
//     }
//     jxf_MPI_Bcast(&block_size, 1, JXF_MPI_INT, 0, comm);
//     jxf_MPI_Bcast(global_data, 4, JXF_MPI_BIG_INT, 0, comm);
//     global_num_rows = global_data[0];
//     global_num_cols = global_data[1];
//     global_size     = global_data[2];

//     if (global_size > 3) {
//         JXF_Int send_start;

//         if (global_data[3] == 2) {
//             send_start = 4;
//             jxf_MPI_Scatter(&global_data[send_start], 1, JXF_MPI_BIG_INT, &row_starts[0], 1, JXF_MPI_BIG_INT, 0, comm);

//             send_start = 5;
//             jxf_MPI_Scatter(&global_data[send_start], 1, JXF_MPI_BIG_INT, &row_starts[1], 1, JXF_MPI_BIG_INT, 0, comm);

//             send_start = 4 + (num_procs + 1);
//             jxf_MPI_Scatter(&global_data[send_start], 1, JXF_MPI_BIG_INT, &col_starts[0], 1, JXF_MPI_BIG_INT, 0, comm);

//             send_start = 5 + (num_procs + 1);
//             jxf_MPI_Scatter(&global_data[send_start], 1, JXF_MPI_BIG_INT, &col_starts[1], 1, JXF_MPI_BIG_INT, 0, comm);
//         } else if ((global_data[3] == 0) || (global_data[3] == 1)) {
//             send_start = 4;
//             jxf_MPI_Scatter(&global_data[send_start], 1, JXF_MPI_BIG_INT, &row_starts[0], 1, JXF_MPI_BIG_INT, 0, comm);

//             send_start = 5;
//             jxf_MPI_Scatter(&global_data[send_start], 1, JXF_MPI_BIG_INT, &row_starts[1], 1, JXF_MPI_BIG_INT, 0, comm);

//             if (global_data[3] == 0) {
//                 col_starts[0] = row_starts[0];
//                 col_starts[1] = row_starts[1];
//             }
//         } else {
//             send_start = 4;
//             jxf_MPI_Scatter(&global_data[send_start], 1, JXF_MPI_BIG_INT, &col_starts[0], 1, JXF_MPI_BIG_INT, 0, comm);

//             send_start = 5;
//             jxf_MPI_Scatter(&global_data[send_start], 1, JXF_MPI_BIG_INT, &col_starts[1], 1, JXF_MPI_BIG_INT, 0, comm);
//         }
//     }
//     jxf_TFree(global_data);

//     // 在创建parbsr_A前添加
//     printf("Rank %d: Creating ParBSRMatrix, block_size=%d, global_num_rows=%lld\n", 
//         my_id, block_size, (long long)global_num_rows);
//     printf("Rank %d: row_starts[0]=%lld, row_starts[1]=%lld\n", 
//         my_id, (long long)row_starts[0], (long long)row_starts[1]);
//     printf("Rank %d: col_starts[0]=%lld, col_starts[1]=%lld\n", 
//         my_id, (long long)col_starts[0], (long long)col_starts[1]);
//     fflush(stdout);

//     // Create ParBSR matrix
//     parbsr_A = jxf_ParBSRMatrixCreate(comm, block_size, global_num_rows, global_num_cols, 
//                                      row_starts, col_starts, 0, 0, 0);

//     printf("Rank %d: Created parbsr_A at %p\n", my_id, (void*)parbsr_A);
//     if (parbsr_A) {
//         printf("Rank %d: parbsr_A->diag = %p\n", my_id, (void*)jxf_ParBSRMatrixDiag(parbsr_A));
//         printf("Rank %d: parbsr_A->offd = %p\n", my_id, (void*)jxf_ParBSRMatrixOffd(parbsr_A));
//         printf("Rank %d: parbsr_A->block_size = %d\n", my_id, jxf_ParBSRMatrixBlockSize(parbsr_A));
//     }
//     fflush(stdout);

//     // Allocate memory for building ParBSR matrix
//     num_rows_proc     = NULL;
//     num_nonzeros_proc = NULL;
//     if (my_id == 0) {
//         num_rows_proc     = jxf_CTAlloc(JXF_Int, num_procs);
//         num_nonzeros_proc = jxf_CTAlloc(JXF_Int, num_procs);
//     }

//     if (my_id == 0) {
//         if (!global_row_starts) {
//             jxf_GeneratePartitioning(global_num_rows, num_procs, &global_row_starts);
//             free_global_row_starts = 1;
//         }
//         if (!global_col_starts) {
//             jxf_GeneratePartitioning(global_num_rows, num_procs, &global_col_starts);
//             free_global_col_starts = 1;
//         }

//         for (i = 0; i < num_procs; i++) {
//             num_rows_proc[i]     = (JXF_Int)(global_row_starts[i + 1] - global_row_starts[i]);
//             num_nonzeros_proc[i] = A_i[(JXF_Int)global_row_starts[i + 1]] - A_i[(JXF_Int)global_row_starts[i]];
//         }
//     }
//     jxf_MPI_Scatter(num_rows_proc, 1, JXF_MPI_INT, &num_rows, 1, JXF_MPI_INT, 0, comm);
//     jxf_MPI_Scatter(num_nonzeros_proc, 1, JXF_MPI_INT, &num_nonzeros, 1, JXF_MPI_INT, 0, comm);

//     local_A = jxf_BSRMatrixCreate(block_size, num_rows, (JXF_Int)global_num_cols, num_nonzeros);

//     bsr_matrix_datatypes = jxf_CTAlloc(MPI_Datatype, num_procs);
//     if (my_id == 0) {
//         requests = jxf_CTAlloc(MPI_Request, num_procs - 1);
//         status   = jxf_CTAlloc(MPI_Status, num_procs - 1);
//         for (i = 1; i < num_procs; i++) {
//             ind = A_i[(JXF_Int)global_row_starts[i]];

//             jxf_BuildBSRMatrixMPIDataType(block_size, num_nonzeros_proc[i], num_rows_proc[i], 
//                                          &A_data[ind * block_size * block_size],
//                                          &A_i[(JXF_Int)global_row_starts[i]], &A_j[ind], 
//                                          &bsr_matrix_datatypes[i]);
//             jxf_MPI_Isend(MPI_BOTTOM, 1, bsr_matrix_datatypes[i], i, 0, comm, &requests[i - 1]);
//             jxf_MPI_Type_free(&bsr_matrix_datatypes[i]);
//         }
//         jxf_BSRMatrixData(local_A)     = A_data;
//         jxf_BSRMatrixI(local_A)        = A_i;
//         jxf_BSRMatrixJ(local_A)        = A_j;
//         jxf_BSRMatrixOwnsData(local_A) = 0;

//         jxf_MPI_Waitall(num_procs - 1, requests, status);

//         jxf_TFree(requests);
//         jxf_TFree(status);
//         jxf_TFree(num_rows_proc);
//         jxf_TFree(num_nonzeros_proc);

//         if (free_global_row_starts) {
//             jxf_TFree(global_row_starts);
//         }
//         if (free_global_col_starts) {
//             jxf_TFree(global_col_starts);
//         }
//     } else {
//         jxf_BSRMatrixInitialize(local_A);
//         jxf_BuildBSRMatrixMPIDataType(block_size, num_nonzeros, num_rows, jxf_BSRMatrixData(local_A), 
//                                      jxf_BSRMatrixI(local_A), jxf_BSRMatrixJ(local_A), &bsr_matrix_datatypes[0]);
//         jxf_MPI_Recv(MPI_BOTTOM, 1, bsr_matrix_datatypes[0], 0, 0, comm, &status0);
//         jxf_MPI_Type_free(bsr_matrix_datatypes);
//     }

//     first_col_diag = jxf_ParBSRMatrixFirstColDiag(parbsr_A);
//     last_col_diag  = jxf_ParBSRMatrixLastColDiag(parbsr_A);

//     jxf_GenerateDiagAndOffdBSR(local_A, parbsr_A, first_col_diag, last_col_diag);

//     /* Set pointers back to NULL before destroying */
//     if (my_id == 0) {
//         jxf_BSRMatrixData(local_A) = NULL;
//         jxf_BSRMatrixI(local_A)    = NULL;
//         jxf_BSRMatrixJ(local_A)    = NULL;
//     }
//     jxf_BSRMatrixDestroy(local_A);
//     jxf_TFree(bsr_matrix_datatypes);

//     return parbsr_A;
// }

/*!
 * \fn JXF_Int jxf_GenerateDiagAndOffdBSR
 * \brief Generate diagonal and off-diagonal parts for parallel BSR matrix.
 * \date 2025/10/25
 */
JXF_Int jxf_GenerateDiagAndOffdBSR(jxf_BSRMatrix* A, jxf_ParBSRMatrix* matrix, JXF_BigInt first_col_diag, JXF_BigInt last_col_diag)
{

    // printf("DEBUG GenerateDiagAndOffdBSR: A=%p, matrix=%p\n", (void*)A, (void*)matrix);
    // printf("DEBUG first_col_diag=%lld, last_col_diag=%lld\n", 
    //        (long long)first_col_diag, (long long)last_col_diag);
    // fflush(stdout);

    JXF_Int      i, j, k;
    JXF_Int      jo, jd;
    JXF_Int      num_rows = jxf_BSRMatrixNumRows(A);
    JXF_Int      num_cols = jxf_BSRMatrixNumCols(A); // global_column
    JXF_Real*    a_data   = jxf_BSRMatrixData(A);
    JXF_Int*     a_i      = jxf_BSRMatrixI(A);
    JXF_Int*     a_j      = jxf_BSRMatrixJ(A);

    jxf_BSRMatrix* diag              = jxf_ParBSRMatrixDiag(matrix);
    jxf_BSRMatrix* offd              = jxf_ParBSRMatrixOffd(matrix);
    JXF_Int        diag_num_rows     = jxf_BSRMatrixNumRows(diag);
    JXF_Int        diag_num_nonzeros = jxf_BSRMatrixNumNonzeros(diag);
    JXF_Int        offd_num_rows     = jxf_BSRMatrixNumRows(offd);
    JXF_Int        offd_num_nonzeros = jxf_BSRMatrixNumNonzeros(offd);

    JXF_BigInt* col_map_offd;

    JXF_Real *diag_data, *offd_data;
    JXF_Int * diag_i, *offd_i;
    JXF_Int * diag_j, *offd_j;
    JXF_Int*  marker;
    JXF_Int   num_cols_diag, num_cols_offd;
    JXF_Int   first_elmt   = a_i[0];
    JXF_Int   num_nonzeros = a_i[num_rows] - first_elmt;
    JXF_Int   counter;
    JXF_Int   block_size = jxf_BSRMatrixBlockSize(diag);
    JXF_Int   bnnz       = block_size * block_size;

    num_cols_diag = (JXF_Int)(last_col_diag - first_col_diag + 1);
    num_cols_offd = 0;

    if (num_cols - num_cols_diag) {
        // printf("DEBUG: Creating separate diag/offd matrices\n");
        // fflush(stdout);

        jxf_BSRMatrixI(diag) = jxf_CTAlloc(JXF_Int, diag_num_rows + 1);
        diag_i  = jxf_BSRMatrixI(diag);

        jxf_BSRMatrixI(offd) = jxf_CTAlloc(JXF_Int, offd_num_rows + 1);
        offd_i  = jxf_BSRMatrixI(offd);
        marker  = jxf_CTAlloc(JXF_Int, num_cols);

        for (i = 0; i < num_cols; i++) {
            marker[i] = 0;
        }

        jo = 0;
        jd = 0;
        for (i = 0; i < num_rows; i++) {
            offd_i[i] = jo;
            diag_i[i] = jd;

            for (j = a_i[i] - first_elmt; j < a_i[i + 1] - first_elmt; j++) {
                if (a_j[j] < (JXF_Int)first_col_diag || a_j[j] > (JXF_Int)last_col_diag) {
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

        jxf_ParBSRMatrixColMapOffd(matrix) = jxf_CTAlloc(JXF_BigInt, num_cols_offd);
        col_map_offd         = jxf_ParBSRMatrixColMapOffd(matrix);

        counter = 0;
        for (i = 0; i < num_cols; i++) {
            if (marker[i]) {
                col_map_offd[counter] = (JXF_BigInt)i;
                marker[i]             = counter;
                counter++;
            }
        }

        jxf_BSRMatrixNumNonzeros(diag) = jd;
        jxf_BSRMatrixJ(diag)            = jxf_CTAlloc(JXF_Int, jd);
        jxf_BSRMatrixData(diag)         = jxf_CTAlloc(JXF_Real, jd * bnnz);
        diag_data          = jxf_BSRMatrixData(diag);
        diag_j             = jxf_BSRMatrixJ(diag);

        jxf_BSRMatrixNumNonzeros(offd) = jo;
        jxf_BSRMatrixNumCols(offd)     = num_cols_offd;
        jxf_BSRMatrixJ(offd)            = jxf_CTAlloc(JXF_Int, jo);
        jxf_BSRMatrixData(offd)         = jxf_CTAlloc(JXF_Real, jo * bnnz);
        offd_data          = jxf_BSRMatrixData(offd);
        offd_j             = jxf_BSRMatrixJ(offd);

        jo = 0;
        jd = 0;
        for (i = 0; i < num_rows; i++) {
            for (j = a_i[i] - first_elmt; j < a_i[i + 1] - first_elmt; j++) {
                if (a_j[j] < (JXF_Int)first_col_diag || a_j[j] > (JXF_Int)last_col_diag) {
                    for (k = 0; k < bnnz; k++) 
                        offd_data[jo * bnnz + k] = a_data[j * bnnz + k];
                    offd_j[jo++] = marker[a_j[j]];
                } else {
                    for (k = 0; k < bnnz; k++) 
                        diag_data[jd * bnnz + k] = a_data[j * bnnz + k];
                    diag_j[jd++] = (JXF_Int)(a_j[j] - first_col_diag);
                }
            }
        }
        jxf_TFree(marker);
    } else {
        // printf("DEBUG: Using original matrix data directly\n");
        // printf("DEBUG: num_cols=%d, num_cols_diag=%d\n", num_cols, num_cols_diag);
        // fflush(stdout);

        jxf_BSRMatrixNumNonzeros(diag) = num_nonzeros;
        jxf_BSRMatrixInitialize(diag);
        diag_data = jxf_BSRMatrixData(diag);
        diag_i    = jxf_BSRMatrixI(diag);
        diag_j    = jxf_BSRMatrixJ(diag);

        for (i = 0; i < num_nonzeros; i++) {
            for (k = 0; k < bnnz; k++) 
                diag_data[i * bnnz + k] = a_data[i * bnnz + k];
            diag_j[i] = a_j[i];
        }
        offd_i = jxf_CTAlloc(JXF_Int, num_rows + 1);

        for (i = 0; i < num_rows + 1; i++) {
            diag_i[i] = a_i[i];
            offd_i[i] = 0;
        }

        jxf_BSRMatrixNumCols(offd) = 0;
        jxf_BSRMatrixI(offd)        = offd_i;
    }

    return 0;
}


/*!
 * \fn JXF_Int jxf_ParBSRMatrixGetRowPartitioning
 * \brief Get the row partitioning for a parallel BSR matrix.
 * \note For BSR matrices, row partitioning should be in scalar rows, not block rows.
 * \author peghoty
 * \date 2009/07/10
 * \modified 2025/10/08 (updated for BSR format)
 */ 
JXF_Int jxf_ParBSRMatrixGetRowPartitioning(jxf_ParBSRMatrix *matrix, JXF_Int **row_partitioning_ptr)
{  
    JXF_Int *row_partitioning, *row_starts;
    JXF_Int  num_procs,myid, i;
    JXF_Int  blk_size;

    if (!matrix) 
    {
        jxf_error_in_arg(1);
        return jxf_error_flag;
    }

    jxf_MPI_Comm_size(jxf_ParBSRMatrixComm(matrix), &num_procs);
    jxf_MPI_Comm_rank(jxf_ParBSRMatrixComm(matrix), &myid);
    row_starts = jxf_ParBSRMatrixRowStarts(matrix);
    blk_size = jxf_ParBSRMatrixBlockSize(matrix);
    
    if (!row_starts) 
    {
        *row_partitioning_ptr = NULL;
        return -1;
    }
    
    // Allocate memory for scalar row partitioning
    // Length is num_procs + 1, same as block row partitioning
    row_partitioning = jxf_CTAlloc(JXF_Int, num_procs + 1);
    
    // Convert block row partitioning to scalar row partitioning
    // Each block row contains blk_size scalar rows
    for (i = 0; i < num_procs + 1; i++)
    {
        row_partitioning[i] = row_starts[i] * blk_size;
        // printf("Rank %d: row_partitioning = %d\n", myid, row_partitioning[i]);
    }

    *row_partitioning_ptr = row_partitioning;

    return jxf_error_flag;
}

/*!
 * \fn JXF_Int jxf_ParBSRMatrixGetColPartitioning
 * \brief Get the column partitioning for a parallel BSR matrix.
 * \note For BSR matrices, column partitioning should be in scalar columns, not block columns.
 * \date 2025/10/08
 */ 
JXF_Int jxf_ParBSRMatrixGetColPartitioning(jxf_ParBSRMatrix *matrix, JXF_Int **col_partitioning_ptr)
{  
    JXF_Int *col_partitioning, *col_starts;
    JXF_Int  num_procs, i;
    JXF_Int  blk_size;

    if (!matrix) 
    {
        jxf_error_in_arg(1);
        return jxf_error_flag;
    }

    jxf_MPI_Comm_size(jxf_ParBSRMatrixComm(matrix), &num_procs);
    col_starts = jxf_ParBSRMatrixColStarts(matrix);
    blk_size = jxf_ParBSRMatrixBlockSize(matrix);
    
    if (!col_starts) 
    {
        *col_partitioning_ptr = NULL;
        return -1;
    }
    
    // Allocate memory for scalar column partitioning
    col_partitioning = jxf_CTAlloc(JXF_Int, num_procs + 1);
    
    // Convert block column partitioning to scalar column partitioning
    for (i = 0; i < num_procs + 1; i++)
    {
        col_partitioning[i] = col_starts[i] * blk_size;
    }

    *col_partitioning_ptr = col_partitioning;

    return jxf_error_flag;
}