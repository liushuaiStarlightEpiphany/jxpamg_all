#include "jxf_parbilu.h"
#include "jxf_krylov.h"
#include "jxf_util.h"
#include "jxf_parbsr_mv.h"

#include "bsr_block_ops.h"

#include <float.h>

#define JXF_USING_CUDA 1
#define JXF_USING_GPU 1
/*--------------------------------------------------------------------------
 * jxf_BILUSetup
 *--------------------------------------------------------------------------*/

JXF_Int
jxf_BILUSetup(void *ilu_vdata, jxf_ParBSRMatrix *A, jxf_ParVector *f, jxf_ParVector *u)
{
    MPI_Comm comm = jxf_ParBSRMatrixComm(A);
    // JXF_MemoryLocation  memory_location     = jxf_ParCSRMatrixMemoryLocation(A);
    jxf_ParBILUData *ilu_data = (jxf_ParBILUData *)ilu_vdata;
    //    jxf_ParILUData     *schur_precond_ilu;
    //    jxf_ParNSHData     *schur_solver_nsh;

    /* Pointers to ilu data */
    JXF_Int logging = jxf_ParILUDataLogging(ilu_data);
    JXF_Int print_level = jxf_ParILUDataPrintLevel(ilu_data);
    JXF_Int ilu_type = jxf_ParILUDataIluType(ilu_data);
    JXF_Int nLU = jxf_ParILUDataNLU(ilu_data);
    JXF_Int nI = jxf_ParILUDataNI(ilu_data);
    JXF_Int fill_level = jxf_ParILUDataLfil(ilu_data);
    JXF_Int sweep = jxf_ParILUDatasweep(ilu_data);
    JXF_Int max_row_elmts = jxf_ParILUDataMaxRowNnz(ilu_data);
    JXF_Real *droptol = jxf_ParILUDataDroptol(ilu_data);
    JXF_Int *CF_marker_array = jxf_ParILUDataCFMarkerArray(ilu_data);
    JXF_Int *perm = jxf_ParILUDataPerm(ilu_data);
    JXF_Int *qperm = jxf_ParILUDataQPerm(ilu_data);
    JXF_Real tol_ddPQ = jxf_ParILUDataTolDDPQ(ilu_data);

    jxf_ParBSRMatrix *matA = jxf_ParILUDataMatA(ilu_data);
    jxf_ParBSRMatrix *matL = jxf_ParILUDataMatL(ilu_data);
    JXF_Real *matD = jxf_ParILUDataMatD(ilu_data);
    jxf_ParBSRMatrix *matU = jxf_ParILUDataMatU(ilu_data);
    jxf_ParBSRMatrix *matmL = jxf_ParILUDataMatLModified(ilu_data);
    JXF_Real *matmD = jxf_ParILUDataMatDModified(ilu_data);
    jxf_ParBSRMatrix *matmU = jxf_ParILUDataMatUModified(ilu_data);
    jxf_ParBSRMatrix *matS = jxf_ParILUDataMatS(ilu_data);
    JXF_Int n = jxf_BSRMatrixNumRows(jxf_ParBSRMatrixDiag(A));
    JXF_Int reordering_type = jxf_ParILUDataReorderingType(ilu_data);
    JXF_Real nnzS; /* Total nnz in S */
    JXF_Real nnzS_offd_local;
    JXF_Real nnzS_offd;
    JXF_Int size_C /* Total size of coarse grid */;

    jxf_ParVector *Utemp = NULL;
    jxf_ParVector *Ftemp = NULL;
    jxf_ParVector *Xtemp = NULL;
    jxf_ParVector *Ytemp = NULL;
    jxf_ParVector *Ztemp = NULL;
    JXF_Real *uext = NULL;
    JXF_Real *fext = NULL;
    jxf_ParVector *rhs = NULL;
    jxf_ParVector *x = NULL;

    /* TODO (VPM): Change F_array and U_array variable names */
    jxf_ParVector *D_array = jxf_ParILUDataD(ilu_data);
    jxf_ParVector *F_array = jxf_ParILUDataF(ilu_data);
    jxf_ParVector *U_array = jxf_ParILUDataU(ilu_data);
    jxf_ParVector *residual = jxf_ParILUDataResidual(ilu_data);
    JXF_Real *rel_res_norms = jxf_ParILUDataRelResNorms(ilu_data);

    /* might need for Schur Complement */
    JXF_Int *u_end = NULL;
    //    JXF_Solver          schur_solver         = NULL;
    //    JXF_Solver          schur_precond        = NULL;
    //    JXF_Solver          schur_precond_gotten = NULL;

    /* Whether or not to use exact (direct) triangular solves */
    JXF_Int tri_solve = jxf_ParILUDataTriSolve(ilu_data);

    /* help to build external */
    jxf_ParCSRCommPkg *comm_pkg;
    JXF_Int buffer_size;
    JXF_Int num_sends;
    JXF_Int send_size;
    JXF_Int recv_size;
    JXF_Int num_procs, my_id;

    jxf_MPI_Comm_size(comm, &num_procs);
    jxf_MPI_Comm_rank(comm, &my_id);

    /* Free previously allocated data, if any not destroyed */
    jxf_ParBSRMatrixDestroy(matL);
    matL = NULL;
    jxf_ParBSRMatrixDestroy(matU);
    matU = NULL;
    jxf_ParBSRMatrixDestroy(matmL);
    matmL = NULL;
    jxf_ParBSRMatrixDestroy(matmU);
    matmU = NULL;
    jxf_ParBSRMatrixDestroy(matS);
    matS = NULL;

    // jxf_ParVectorDestroy(D_array); D_array = NULL;

    for (JXF_Int i = 0; i < jxf_ParILUDataL_num_levels(ilu_data); i++)
    {
        jxf_TFree(jxf_ParILUDataL_levels(ilu_data)[i]);
    }
    jxf_TFree(jxf_ParILUDataL_levels(ilu_data));
    jxf_TFree(jxf_ParILUDataL_level_sizes(ilu_data));
    jxf_ParILUDataL_num_levels(ilu_data) = 0;

    for (JXF_Int i = 0; i < jxf_ParILUDataU_num_levels(ilu_data); i++)
    {
        jxf_TFree(jxf_ParILUDataU_levels(ilu_data)[i]);
    }
    jxf_TFree(jxf_ParILUDataU_levels(ilu_data));
    jxf_TFree(jxf_ParILUDataU_level_sizes(ilu_data));
    jxf_ParILUDataU_num_levels(ilu_data) = 0;

    jxf_TFree(matD);
    jxf_TFree(matmD);
    jxf_TFree(CF_marker_array);

    /* clear old l1_norm data, if created */
    jxf_TFree(jxf_ParILUDataL1Norms(ilu_data));

    /* setup temporary storage
     * first check is they've already here
     */
    jxf_ParVectorDestroy(jxf_ParILUDataUTemp(ilu_data));
    jxf_ParVectorDestroy(jxf_ParILUDataFTemp(ilu_data));
    jxf_ParVectorDestroy(jxf_ParILUDataRhs(ilu_data));
    jxf_ParVectorDestroy(jxf_ParILUDataX(ilu_data));
    jxf_ParVectorDestroy(jxf_ParILUDataResidual(ilu_data));
    jxf_TFree(jxf_ParILUDataUExt(ilu_data));
    jxf_TFree(jxf_ParILUDataFExt(ilu_data));
    jxf_TFree(jxf_ParILUDataUEnd(ilu_data));
    jxf_TFree(jxf_ParILUDataRelResNorms(ilu_data));

    jxf_ParILUDataUTemp(ilu_data) = NULL;
    jxf_ParILUDataFTemp(ilu_data) = NULL;
    jxf_ParILUDataRhs(ilu_data) = NULL;
    jxf_ParILUDataX(ilu_data) = NULL;
    jxf_ParILUDataResidual(ilu_data) = NULL;

    /* Create work vectors */
    /* Utemp/Ftemp: length = block_rows * block_size (vector length, not block count) */
    JXF_Int vec_global_size = jxf_ParBSRMatrixGlobalNumRows(A) * jxf_ParBSRMatrixBlockSize(A);
    JXF_Int *scalar_part = NULL;
    jxf_ParBSRMatrixGetRowPartitioning(A, &scalar_part);
    Utemp = jxf_ParVectorCreate(jxf_ParBSRMatrixComm(A),
                               vec_global_size,
                               scalar_part);
    jxf_ParVectorInitialize(Utemp);
    jxf_ParILUDataUTemp(ilu_data) = Utemp;

    Ftemp = jxf_ParVectorCreate(jxf_ParBSRMatrixComm(A),
                               vec_global_size,
                               scalar_part);
    jxf_ParVectorInitialize(Ftemp);
    jxf_ParILUDataFTemp(ilu_data) = Ftemp;
    jxf_TFree(scalar_part);

    Xtemp = jxf_ParVectorCreate(comm, jxf_ParVectorGlobalSize(u), jxf_ParVectorPartitioning(u));
    jxf_ParVectorInitialize(Xtemp);
    jxf_ParILUDataXTemp(ilu_data) = Xtemp;


    Ytemp = jxf_ParVectorCreate(comm, jxf_ParVectorGlobalSize(u), jxf_ParVectorPartitioning(u));
    jxf_ParVectorInitialize(Ytemp);
    jxf_ParILUDataYTemp(ilu_data) = Ytemp;

    /* set matrix, solution and rhs pointers */
    matA = A;
    F_array = f;
    U_array = u;

    /*层次调度*/
    // JXF_Int **L_levels, **U_levels;
    // JXF_Int *L_level_sizes, *U_level_sizes;
    // JXF_Int L_num_levels, U_num_levels;

    // /* Create perm array if necessary */
    // if (!perm)
    // {
    //    switch (ilu_type)
    //    {
    //       case 0: case 1:
    //       default:
    //          /* RCM or none */
    //          jxf_ILUGetLocalPerm(matA, &perm, &nLU, reordering_type);
    //          break;
    //    }
    // }

    /* Factorization */
    switch (ilu_type)
    {
        // case 0: /* BJ + jxf_iluk() */
        // {
        //     jxf_ILUSetupILUK(matA, fill_level, perm, perm, n, n,
        //                     &matL, &matD, &matU, &D_array, &u_end);
        //     if (tri_solve == 2)
        //     {
        //         jxf_CSRMatrixTopologicSortILU(jxf_ParCSRMatrixDiag(matL), jxf_ParCSRMatrixDiag(matU),
        //                                     &(ilu_data->nlevL), &(ilu_data->jlevL), &(ilu_data->ilevL),
        //                                     &(ilu_data->nlevU), &(ilu_data->jlevU), &(ilu_data->ilevU));
        //     }
        //     else if (tri_solve == 3)
        //     {
        //         jxf_GreedyColoring_L(matL, n, &(ilu_data->jlevL), &(ilu_data->ilevL), &(ilu_data->nlevL));
        //         jxf_GreedyColoring_U(matU, n, &(ilu_data->jlevU), &(ilu_data->ilevU), &(ilu_data->nlevU));
        //     }
        //     else if (tri_solve == 12)
        //     {
        //         jxf_GreedyColoring_L1(matL, n, &(ilu_data->jlevL), &(ilu_data->ilevL), &(ilu_data->nlevL), &(ilu_data->L_perm), &(ilu_data->L_iperm));
        //         jxf_GreedyColoring_U1(matU, n, &(ilu_data->jlevU), &(ilu_data->ilevU), &(ilu_data->nlevU), &(ilu_data->U_perm), &(ilu_data->U_iperm));
        //     }
        //     if (tri_solve == 7)
        //     {
        //         jxf_CSRMatrixTopologicSortILU(jxf_ParCSRMatrixDiag(matL), jxf_ParCSRMatrixDiag(matU),
        //                                     &(ilu_data->nlevL), &(ilu_data->jlevL), &(ilu_data->ilevL),
        //                                     &(ilu_data->nlevU), &(ilu_data->jlevU), &(ilu_data->ilevU));
        //     }
        //     else if (tri_solve == 8)
        //     {
        //         jxf_GreedyColoring_8GS(matA, n, &(ilu_data->jlevL), &(ilu_data->ilevL), &(ilu_data->nlevL), &(ilu_data->jlevU), &(ilu_data->ilevU), &(ilu_data->nlevU));
        //     }
        //     else if (tri_solve == 13)
        //     {
        //         jxf_CSRMatrixTopologicSortILU1(jxf_ParCSRMatrixDiag(matL), jxf_ParCSRMatrixDiag(matU),
        //                                     &(ilu_data->nlevL), &(ilu_data->jlevL), &(ilu_data->ilevL), &(ilu_data->L_perm), &(ilu_data->L_iperm),
        //                                     &(ilu_data->nlevU), &(ilu_data->jlevU), &(ilu_data->ilevU), &(ilu_data->U_perm), &(ilu_data->U_iperm));
        //     }
        // }
        // break;

        // case 2: /* BJ + jxf_iluk() */
        // {
        //     jxf_ILUSetupFGPILU_v2(matA, perm, perm, sweep,
        //                         &matL, &matD, &matU, &D_array, &u_end);
        //     if (tri_solve == 2)
        //     {
        //         jxf_CSRMatrixTopologicSortILU(jxf_ParCSRMatrixDiag(matL), jxf_ParCSRMatrixDiag(matU),
        //                                     &(ilu_data->nlevL), &(ilu_data->jlevL), &(ilu_data->ilevL),
        //                                     &(ilu_data->nlevU), &(ilu_data->jlevU), &(ilu_data->ilevU));
        //     }
        //     else if (tri_solve == 3)
        //     {
        //         jxf_GreedyColoring_L(matL, n, &(ilu_data->jlevL), &(ilu_data->ilevL), &(ilu_data->nlevL));
        //         jxf_GreedyColoring_U(matU, n, &(ilu_data->jlevU), &(ilu_data->ilevU), &(ilu_data->nlevU));
        //     }
        //     else if (tri_solve == 12)
        //     {
        //         jxf_GreedyColoring_L1(matL, n, &(ilu_data->jlevL), &(ilu_data->ilevL), &(ilu_data->nlevL), &(ilu_data->L_perm), &(ilu_data->L_iperm));
        //         jxf_GreedyColoring_U1(matU, n, &(ilu_data->jlevU), &(ilu_data->ilevU), &(ilu_data->nlevU), &(ilu_data->U_perm), &(ilu_data->U_iperm));
        //     }
        //     if (tri_solve == 7)
        //     {
        //         jxf_CSRMatrixTopologicSortILU(jxf_ParCSRMatrixDiag(matL), jxf_ParCSRMatrixDiag(matU),
        //                                     &(ilu_data->nlevL), &(ilu_data->jlevL), &(ilu_data->ilevL),
        //                                     &(ilu_data->nlevU), &(ilu_data->jlevU), &(ilu_data->ilevU));
        //     }
        //     else if (tri_solve == 8)
        //     {
        //         jxf_GreedyColoring_8GS(matA, n, &(ilu_data->jlevL), &(ilu_data->ilevL), &(ilu_data->nlevL), &(ilu_data->jlevU), &(ilu_data->ilevU), &(ilu_data->nlevU));
        //     }
        //     else if (tri_solve == 13)
        //     {
        //         jxf_CSRMatrixTopologicSortILU1(jxf_ParCSRMatrixDiag(matL), jxf_ParCSRMatrixDiag(matU),
        //                                     &(ilu_data->nlevL), &(ilu_data->jlevL), &(ilu_data->ilevL), &(ilu_data->L_perm), &(ilu_data->L_iperm),
        //                                     &(ilu_data->nlevU), &(ilu_data->jlevU), &(ilu_data->ilevU), &(ilu_data->U_perm), &(ilu_data->U_iperm));
        //     }
        // }
        // break;

        case 3:
        {
            jxf_ILUSetupFGPBILU_v3(matA, sweep, &matL, &D_array, &matU, &matS, &u_end);
        }
        break;
    }

    // D_array = jxf_ParVectorCreate(jxf_ParCSRMatrixComm(A),
    //                             jxf_ParCSRMatrixGlobalNumRows(A),
    //                             jxf_ParCSRMatrixRowStarts(A) );
    // jxf_ParVectorInitialize(D_array);
    // jxf_ParILUDataD(ilu_data) =  D_array;

    // jxf_Vector *D_local = jxf_ParVectorLocalVector(D_array);
    // JXF_Real  *D_data   = jxf_VectorData(D_local);
    // for (JXF_Int i = 0; i < n; i++)
    // {
    //    D_data[i] = matD[i];
    // }

    // memcpy(jxf_VectorData(D_local), D_data, n * sizeof(JXF_Real));

    /* Set pointers to ilu data */
    jxf_ParILUDataMatA(ilu_data) = matA;
    jxf_ParILUDataXTemp(ilu_data) = Xtemp;
    jxf_ParILUDataYTemp(ilu_data) = Ytemp;
    jxf_ParILUDataZTemp(ilu_data) = Ztemp;
    jxf_ParILUDataF(ilu_data) = F_array;
    jxf_ParILUDataU(ilu_data) = U_array;
    jxf_ParILUDataMatL(ilu_data) = matL;
    jxf_ParILUDataMatD(ilu_data) = matD;
    jxf_ParILUDataMatU(ilu_data) = matU;
    jxf_ParILUDataMatLModified(ilu_data) = matmL;
    jxf_ParILUDataMatDModified(ilu_data) = matmD;
    jxf_ParILUDataMatUModified(ilu_data) = matmU;
    jxf_ParILUDataMatS(ilu_data) = matS;
    jxf_ParILUDataCFMarkerArray(ilu_data) = CF_marker_array;
    jxf_ParILUDataPerm(ilu_data) = perm;
    jxf_ParILUDataQPerm(ilu_data) = qperm;
    jxf_ParILUDataNLU(ilu_data) = nLU;
    jxf_ParILUDataNI(ilu_data) = nI;
    jxf_ParILUDataUEnd(ilu_data) = u_end;
    jxf_ParILUDataUExt(ilu_data) = uext;
    jxf_ParILUDataFExt(ilu_data) = fext;
    jxf_ParILUDataD(ilu_data) = D_array;

    /* compute operator complexity */
    {
        MPI_Comm m_comm = jxf_ParBSRMatrixComm(matA);
        jxf_BSRMatrix *m_diag = jxf_ParBSRMatrixDiag(matA);
        JXF_Int *m_diag_i = jxf_BSRMatrixI(m_diag);
        jxf_BSRMatrix *m_offd = jxf_ParBSRMatrixOffd(matA);
        JXF_Int *m_offd_i = jxf_BSRMatrixI(m_offd);
        JXF_Int m_nrows = jxf_BSRMatrixNumRows(m_diag);
        JXF_Real m_local_nnz = (JXF_Real)m_diag_i[m_nrows] + (JXF_Real)m_offd_i[m_nrows];
        JXF_Real m_total_nnz;
        jxf_MPI_Allreduce(&m_local_nnz, &m_total_nnz, 1, JXF_MPI_REAL, MPI_SUM, m_comm);
        jxf_ParBSRMatrixDNumNonzeros(matA) = m_total_nnz;
    }
    nnzS = 0.0;

    /* size_C is the size of global coarse grid, upper left part */
    size_C = jxf_ParBSRMatrixGlobalNumRows(matA);

    /* TODO (VPM): Move ILU statistics printout to its own function */
    if ((my_id == 0) && (print_level > 0))
    {
        jxf_printf("ILU SETUP: operator complexity = %f  \n",
                  jxf_ParILUDataOperatorComplexity(ilu_data));
        if (jxf_ParILUDataTriSolve(ilu_data) == 1)
        {
            jxf_printf("ILU SOLVE: using direct triangular solves\n",
                      jxf_ParILUDataOperatorComplexity(ilu_data));
        }
        if (jxf_ParILUDataTriSolve(ilu_data) == 2)
        {
            jxf_printf("ILU SOLVE: using s triangular solves\n",
                      jxf_ParILUDataOperatorComplexity(ilu_data));
        }
        if (jxf_ParILUDataTriSolve(ilu_data) == 3)
        {
            jxf_printf("ILU SOLVE: using GS iterative triangular solves\n",
                      jxf_ParILUDataOperatorComplexity(ilu_data));
        }
    }

    if (logging > 1)
    {
        residual =
            jxf_ParVectorCreate(jxf_ParCSRMatrixComm(matA),
                               jxf_ParCSRMatrixGlobalNumRows(matA),
                               jxf_ParCSRMatrixRowStarts(matA));
        jxf_ParVectorInitialize(residual);
        jxf_ParILUDataResidual(ilu_data) = residual;
    }
    else
    {
        jxf_ParILUDataResidual(ilu_data) = NULL;
    }
    rel_res_norms = jxf_CTAlloc(JXF_Real, jxf_ParILUDataMaxIter(ilu_data));
    jxf_ParILUDataRelResNorms(ilu_data) = rel_res_norms;

    // jxf_GpuProfilingPopRange();

    return jxf_error_flag;
}

/*--------------------------------------------------------------------------
 * jxf_ILUSetupILUK
 *
 * Setup ILU(k) numeric factorization
 *
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * permp: permutation array indicating ordering of factorization.
 *        Perm could come from a CF_marker array or a reordering routine.
 * qpermp: column permutation array.
 * nLU: size of computed LDU factorization.
 * nI: number of interial unknowns, nI should obey nI >= nLU
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sprt: Schur Complement, if no Schur Complement, it will be set to NULL
 *--------------------------------------------------------------------------*/

// JXF_Int
// jxf_ILUSetupILUK(jxf_ParCSRMatrix *A,
//                 JXF_Int lfil,
//                 JXF_Int *permp,
//                 JXF_Int *qpermp,
//                 JXF_Int nLU,
//                 JXF_Int nI,
//                 jxf_ParCSRMatrix **Lptr,
//                 JXF_Real **Dptr,
//                 jxf_ParCSRMatrix **Uptr,
//                 jxf_ParVector **D_array,
//                 JXF_Int **u_end)
// {
//     /*
//      * 1: Setup and create buffers
//      * matL/U: the ParCSR matrix for L and U
//      * L/U_diag: the diagonal csr matrix of matL/U
//      * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
//      * ii = outer loop from 0 to nLU - 1
//      * i = the real col number in diag inside the outer loop
//      * iw =  working array store the reverse of active col number
//      * iL = working array store the active col number
//      */

//     /* call ILU0 if lfil is 0 */
//     if (lfil == 0)
//     {
//         // printf("enter jxf_ILUSetupILU0 \n");
//         return jxf_ILUSetupILU0(A, permp, qpermp, nLU, nI, Lptr, Dptr, Uptr, D_array, u_end);
//     }

//     return jxf_error_flag;
// }

/*--------------------------------------------------------------------------
 * jxf_ILUSetupILU0
 *
 * Setup ILU(0)
 *
 * A = input matrix
 * perm = permutation array indicating ordering of rows.
 *        Perm could come from a CF_marker array or a reordering routine.
 *         When set to NULL, identity permutation is used.
 * qperm = permutation array indicating ordering of columns.
 *         When set to NULL, identity permutation is used.
 * nI = number of interial unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *       Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

// JXF_Int
// jxf_ILUSetupILU0(jxf_ParCSRMatrix *A,
//                 JXF_Int *perm,
//                 JXF_Int *qperm,
//                 JXF_Int nLU,
//                 JXF_Int nI,
//                 jxf_ParCSRMatrix **Lptr,
//                 JXF_Real **Dptr,
//                 jxf_ParCSRMatrix **Uptr,
//                 jxf_ParVector **D_array,
//                 JXF_Int **u_end)
// {
//     return jxf_ILUSetupMILU0(A, perm, qperm, nLU, nI, Lptr, Dptr, Uptr, D_array, u_end, 0);
// }

// JXF_Int
// jxf_ILUSetupMILU0(jxf_ParCSRMatrix *A,
//                  JXF_Int *permp,
//                  JXF_Int *qpermp,
//                  JXF_Int nLU,
//                  JXF_Int nI,
//                  jxf_ParCSRMatrix **Lptr,
//                  JXF_Real **Dptr,
//                  jxf_ParCSRMatrix **Uptr,
//                  jxf_ParVector **D_array,
//                  JXF_Int **u_end,
//                  JXF_Int modified)
// {
//     JXF_Int i, ii, j, k, k1, k2, k3, ctrU, ctrL, ctrS;
//     JXF_Int lenl, lenu, jpiv, col, jpos;
//     JXF_Int *iw, *iL, *iU;
//     JXF_Real dd, t, dpiv, lxu, *wU, *wL;
//     JXF_Real drop;

//     /* communication stuffs for S */
//     MPI_Comm comm = jxf_ParCSRMatrixComm(A);
//     JXF_Int S_offd_nnz, S_offd_ncols;
//     jxf_ParCSRCommPkg *comm_pkg;
//     jxf_ParCSRCommHandle *comm_handle;
//     JXF_Int num_sends, begin, end;
//     JXF_Int *send_buf = NULL;
//     JXF_Int num_procs, my_id;

//     /* data objects for A */
//     jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//     jxf_CSRMatrix *A_offd = jxf_ParCSRMatrixOffd(A);
//     JXF_Real *A_diag_data = jxf_CSRMatrixData(A_diag);
//     JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//     JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);
//     // JXF_Real               *A_offd_data     = jxf_CSRMatrixData(A_offd);
//     // JXF_Int                *a        = jxf_CSRMatrixI(A_offd);
//     // JXF_Int                *A_offd_j        = jxf_CSRMatrixJ(A_offd);
//     // JXF_MemoryLocation      memory_location = jxf_ParCSRMatrixMemoryLocation(A);

//     /* size of problem and schur system */
//     JXF_Int n = jxf_CSRMatrixNumRows(A_diag);
//     JXF_Int m = n - nLU;
//     JXF_Int e = nI - nLU;
//     JXF_Int m_e = n - nI;
//     JXF_Real local_nnz, total_nnz;
//     JXF_Int *u_end_array;

//     /* data objects for L, D, U */
//     jxf_ParCSRMatrix *matL;
//     jxf_ParCSRMatrix *matU;
//     jxf_CSRMatrix *L_diag;
//     jxf_CSRMatrix *U_diag;
//     jxf_ParVector *vec_D;
//     jxf_Vector *D_local;
//     JXF_Real *D_data, *D1_data;
//     JXF_Real *L_diag_data;
//     JXF_Int *L_diag_i;
//     JXF_Int *L_diag_j;
//     JXF_Real *U_diag_data;
//     JXF_Int *U_diag_i;
//     JXF_Int *U_diag_j;

//     /* memory management */
//     JXF_Int initial_alloc = 0;
//     JXF_Int capacity_L;
//     JXF_Int capacity_U;
//     JXF_Int nnz_A = A_diag_i[n];

//     /* reverse permutation array */
//     JXF_Int *rperm;
//     JXF_Int *perm, *qperm;

//     /* start setup
//      * get communication stuffs first
//      */
//     jxf_MPI_Comm_size(comm, &num_procs);
//     jxf_MPI_Comm_rank(comm, &my_id);
//     comm_pkg = jxf_ParCSRMatrixCommPkg(A);

//     /* setup if not yet built */
//     if (!comm_pkg)
//     {
//         jxf_MatvecCommPkgCreate(A);
//         comm_pkg = jxf_ParCSRMatrixCommPkg(A);
//     }

//     /* check for correctness */
//     if (nLU < 0 || nLU > n)
//     {
//         jxf_error_w_msg(JXF_ERROR_ARG, "WARNING: nLU out of range.\n");
//     }
//     if (e < 0)
//     {
//         jxf_error_w_msg(JXF_ERROR_ARG, "WARNING: nLU should not exceed nI.\n");
//     }

//     /* Allocate memory for u_end array */
//     u_end_array = jxf_TAlloc(JXF_Int, nLU);

//     /* Allocate memory for L,D,U,S factors */
//     if (n > 0)
//     {
//         initial_alloc = (JXF_Int)(nLU + ceil((nnz_A / 2.0) * nLU / n));
//     }
//     capacity_L = initial_alloc;
//     capacity_U = initial_alloc;

//     D_data = jxf_TAlloc(JXF_Real, n);
//     D1_data = jxf_TAlloc(JXF_Real, n);
//     L_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     L_diag_j = jxf_TAlloc(JXF_Int, capacity_L);
//     L_diag_data = jxf_TAlloc(JXF_Real, capacity_L);
//     U_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     U_diag_j = jxf_TAlloc(JXF_Int, capacity_U);
//     U_diag_data = jxf_TAlloc(JXF_Real, capacity_U);
//     /* allocate working arrays */
//     iw = jxf_TAlloc(JXF_Int, 3 * n);
//     iL = iw + n;
//     rperm = iw + 2 * n;
//     wL = jxf_TAlloc(JXF_Real, n);

//     ctrU = ctrL = 0;
//     L_diag_i[0] = U_diag_i[0] = 0;
//     /* set marker array iw to -1 */
//     for (i = 0; i < n; i++)
//     {
//         iw[i] = -1;
//     }

//     /* get reverse permutation (rperm).
//      * create permutation if they are null
//      * rperm holds the reordered indexes.
//      * rperm only used for column
//      */

//     if (!permp)
//     {
//         perm = jxf_TAlloc(JXF_Int, n);
//         for (i = 0; i < n; i++)
//         {
//             perm[i] = i;
//         }
//     }
//     else
//     {
//         perm = permp;
//     }

//     if (!qpermp)
//     {
//         qperm = jxf_TAlloc(JXF_Int, n);
//         for (i = 0; i < n; i++)
//         {
//             qperm[i] = i;
//         }
//     }
//     else
//     {
//         qperm = qpermp;
//     }

//     for (i = 0; i < n; i++)
//     {
//         rperm[qperm[i]] = i;
//     }

//     /*---------  Begin Factorization. Work in permuted space  ----*/
//     for (ii = 0; ii < nLU; ii++)
//     {
//         // get row i
//         i = perm[ii];
//         // get extents of row i
//         k1 = A_diag_i[i];
//         k2 = A_diag_i[i + 1];
//         // track the drop
//         drop = 0.0;

//         /*-------------------- unpack L & U-parts of row of A in arrays w */
//         iU = iL + ii;
//         wU = wL + ii;
//         /*--------------------  diagonal entry */
//         dd = 0.0;
//         lenl = lenu = 0;
//         iw[ii] = ii;
//         /*-------------------- scan & unwrap column */
//         for (j = k1; j < k2; j++)
//         {
//             col = rperm[A_diag_j[j]];
//             t = A_diag_data[j];
//             if (col < ii)
//             {
//                 iw[col] = lenl;
//                 iL[lenl] = col;
//                 wL[lenl++] = t;
//             }
//             else if (col > ii)
//             {
//                 iw[col] = lenu;
//                 iU[lenu] = col;
//                 wU[lenu++] = t;
//             }
//             else
//             {
//                 dd = t;
//             }
//         }

//         /* eliminate row */
//         /*-------------------------------------------------------------------------
//          *  In order to do the elimination in the correct order we must select the
//          *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
//          *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
//          *  entering the elimination loop.
//          *-----------------------------------------------------------------------*/
//         //      jxf_quickSortIR(iL, wL, iw, 0, (lenl-1));
//         jxf_qsort3ir(iL, wL, iw, 0, (lenl - 1));
//         for (j = 0; j < lenl; j++)
//         {
//             jpiv = iL[j];
//             /* get factor/ pivot element */
//             dpiv = wL[j] * D_data[jpiv];
//             /* store entry in L */
//             wL[j] = dpiv;

//             /* zero out element - reset pivot */
//             iw[jpiv] = -1;
//             /* combine current row and pivot row */
//             for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
//             {
//                 col = U_diag_j[k];
//                 jpos = iw[col];

//                 /* Only fill-in nonzero pattern (jpos != 0) */
//                 if (jpos < 0)
//                 {
//                     drop = drop - U_diag_data[k] * dpiv;
//                     continue;
//                 }

//                 lxu = -U_diag_data[k] * dpiv;
//                 if (col < ii)
//                 {
//                     /* dealing with L part */
//                     wL[jpos] += lxu;
//                 }
//                 else if (col > ii)
//                 {
//                     /* dealing with U part */
//                     wU[jpos] += lxu;
//                     // printf("wU\n",wU[jpos]);
//                 }
//                 else
//                 {
//                     /* diagonal update */
//                     dd += lxu;
//                 }
//             }
//         }
//         /* modify when necessary */
//         if (modified)
//         {
//             dd = dd + drop;
//         }

//         /* restore iw (only need to restore diagonal and U part */
//         iw[ii] = -1;
//         for (j = 0; j < lenu; j++)
//         {
//             iw[iU[j]] = -1;
//         }

//         /* Update LDU factors */
//         /* L part */
//         /* Check that memory is sufficient */
//         if (lenl > 0)
//         {
//             while ((ctrL + lenl) > capacity_L)
//             {
//                 JXF_Int tmp = capacity_L;
//                 capacity_L = (JXF_Int)(capacity_L * EXPAND_FACT + 1);
//                 L_diag_j = jxf_TReAlloc(L_diag_j, JXF_Int, capacity_L);
//                 L_diag_data = jxf_TReAlloc(L_diag_data, JXF_Real, capacity_L);
//             }
//             memcpy(&L_diag_j[ctrL], iL, lenl * sizeof(JXF_Int));
//             memcpy(&L_diag_data[ctrL], wL, lenl * sizeof(JXF_Real));
//         }
//         L_diag_i[ii + 1] = (ctrL += lenl);

//         /* diagonal part (we store the inverse) */
//         if (abs(dd) < MAT_TOL)
//         {
//             dd = 1.0e-6;
//         }
//         D_data[ii] = 1. / dd;

//         /* U part */
//         /* Check that memory is sufficient */
//         if (lenu > 0)
//         {
//             while ((ctrU + lenu) > capacity_U)
//             {
//                 JXF_Int tmp = capacity_U;
//                 capacity_U = (JXF_Int)(capacity_U * EXPAND_FACT + 1);
//                 U_diag_j = jxf_TReAlloc(U_diag_j, JXF_Int, capacity_U);
//                 U_diag_data = jxf_TReAlloc(U_diag_data, JXF_Real, capacity_U);
//             }
//             memcpy(&U_diag_j[ctrU], iU, lenu * sizeof(JXF_Int));
//             memcpy(&U_diag_data[ctrU], wU, lenu * sizeof(JXF_Real));
//         }
//         U_diag_i[ii + 1] = (ctrU += lenu);

//         // if (lenu > 0) {
//         // printf("Row ii=%d, U部分非零元（共%d个）：\n", ii, lenu);
//         // for (j = 0; j < lenu; j++) {
//         //    printf("  col=%d, val=%e\n", iU[j], wU[j]);
//         // }
//         // } else {
//         //    printf("Row ii=%d, U部分无非零元\n", ii);
//         // }
//     }

//     matL = jxf_ParCSRMatrixCreate(comm,
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A),
//                                  0,
//                                  ctrL,
//                                  0);

//     L_diag = jxf_ParCSRMatrixDiag(matL);
//     jxf_CSRMatrixI(L_diag) = L_diag_i;
//     if (ctrL)
//     {
//         jxf_CSRMatrixData(L_diag) = L_diag_data;
//         jxf_CSRMatrixJ(L_diag) = L_diag_j;
//     }
//     else
//     {
//         /* we've allocated some memory, so free if not used */
//         jxf_TFree(L_diag_j);
//         jxf_TFree(L_diag_data);
//     }
//     /* store (global) total number of nonzeros */
//     local_nnz = (JXF_Real)ctrL;
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

//     matU = jxf_ParCSRMatrixCreate(comm,
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A),
//                                  0,
//                                  ctrU,
//                                  0);

//     U_diag = jxf_ParCSRMatrixDiag(matU);
//     jxf_CSRMatrixI(U_diag) = U_diag_i;
//     if (ctrU)
//     {
//         jxf_CSRMatrixData(U_diag) = U_diag_data;
//         jxf_CSRMatrixJ(U_diag) = U_diag_j;
//     }
//     else
//     {
//         /* we've allocated some memory, so free if not used */
//         jxf_TFree(U_diag_j);
//         jxf_TFree(U_diag_data);
//     }
//     /* store (global) total number of nonzeros */
//     local_nnz = (JXF_Real)ctrU;
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
//     /* free memory */
//     jxf_TFree(wL);
//     jxf_TFree(iw);
//     if (!permp)
//     {
//         jxf_TFree(perm);
//     }
//     if (!qpermp)
//     {
//         jxf_TFree(qperm);
//     }

//     memcpy(D1_data, D_data, n * sizeof(JXF_Real));

//     vec_D = jxf_ParVectorCreate(jxf_ParCSRMatrixComm(A),
//                                 jxf_ParCSRMatrixGlobalNumRows(A),
//                                 jxf_ParCSRMatrixRowStarts(A) );
//     jxf_ParVectorInitialize(vec_D);
//     D_local = jxf_ParVectorLocalVector(vec_D);
//     jxf_VectorData(D_local) = D1_data;

//     /* set matrix pointers */
//     *Lptr = matL;
//     *Dptr = D_data;
//     *D_array = vec_D;
//     *Uptr = matU;
//     // *Sptr = matS;
//     *u_end = u_end_array;

//     // char FileNameCoaMat[256];
//     // jxf_sprintf(FileNameCoaMat, "A_CSR_%d", 1);
//     // jxf_ParCSRMatrixPrint(A, FileNameCoaMat);
//     // jxf_sprintf(FileNameCoaMat, "L_CSR_%d", 1);
//     // jxf_ParCSRMatrixPrint(matL, FileNameCoaMat);
//     // jxf_sprintf(FileNameCoaMat, "U_CSR_%d", 1);
//     // jxf_ParCSRMatrixPrint(matU, FileNameCoaMat);

//     // FILE    *fp;
//     // jxf_sprintf(FileNameCoaMat, "D_%d", 1);
//     // fp = fopen(FileNameCoaMat, "w");
//     // for(j = 0; j < n; j++){
//     // jxf_fprintf(fp, "%.14e\n", D_data[j]);
//     // }
//     // fclose(fp);

//     return jxf_error_flag;
// }

// JXF_Int
// jxf_ILUSetupFGPILU_v1(jxf_ParCSRMatrix *A,
//                      JXF_Int *permp,
//                      JXF_Int *qpermp,
//                      JXF_Int sweep,
//                      jxf_ParCSRMatrix **Lptr,
//                      JXF_Real **Dptr,
//                      jxf_ParCSRMatrix **Uptr,
//                      jxf_ParCSRMatrix **Sptr,
//                      JXF_Int **u_end)
// {
//     /*
//      * 1: Setup and create buffers
//      * matL/U: the ParCSR matrix for L and U
//      * L/U_diag: the diagonal csr matrix of matL/U
//      * A_diag_*: temporary pointer for the diagonal matrix of A and its '*' slot
//      * ii = outer loop from 0 to nLU - 1
//      * i = the real col number in diag inside the outer loop
//      * iw = working array store the reverse of active col number
//      * iL = working array store the active col number
//      */
//     MPI_Comm comm = jxf_ParCSRMatrixComm(A);
//     jxf_ParCSRCommPkg *comm_pkg;
//     JXF_Int num_procs, my_id;
//     JXF_Real *wU, *wL, t;

//     JXF_Int ii, i, s, j, k, l, jj, ROW_OMP;
//     JXF_Int tid;
//     JXF_Int i_start, i_end, countu, countl, col;
//     JXF_Int L_nnz = 0, U_nnz = 0;
//     /* data objects for A */
//     jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//     jxf_CSRMatrix *A_offd = jxf_ParCSRMatrixOffd(A);
//     JXF_Real *A_diag_data = jxf_CSRMatrixData(A_diag);
//     JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//     JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);
//     JXF_Real *A_offd_data = jxf_CSRMatrixData(A_offd);
//     JXF_Int *a = jxf_CSRMatrixI(A_offd);
//     JXF_Int *A_offd_j = jxf_CSRMatrixJ(A_offd);
//     JXF_Int n = jxf_CSRMatrixNumRows(A_diag);
//     JXF_Real local_nnz, total_nnz;
//     JXF_Int *u_end_array;

//     /* data objects for L, D, U */
//     JXF_Int *workL, *workU;
//     jxf_ParCSRMatrix *matL;
//     jxf_ParCSRMatrix *matU;
//     jxf_CSRMatrix *L_diag;
//     jxf_CSRMatrix *U_diag;
//     JXF_Real *D_data;
//     JXF_Real *L_diag_data;
//     JXF_Int *L_diag_i;
//     JXF_Int *L_diag_j;
//     JXF_Real *U_diag_data;
//     JXF_Int *U_diag_i;
//     JXF_Int *U_diag_j;

//     JXF_Real *D1_data, *d_data, *d1_data;
//     JXF_Real *L1_diag_data;
//     JXF_Real *U1_diag_data;

//     JXF_Int initial_alloc = 0;
//     JXF_Int capacity_L;
//     JXF_Int capacity_U;
//     JXF_Int nnz_A = A_diag_i[n];
//     /* reverse permutation array */
//     JXF_Int *rperm;
//     JXF_Int *perm, *qperm;
//     JXF_Int nt = jxf_NumThreads();

//     // MPI 初始化
//     jxf_MPI_Comm_size(comm, &num_procs);
//     jxf_MPI_Comm_rank(comm, &my_id);
//     comm_pkg = jxf_ParCSRMatrixCommPkg(A);

//     /* setup if not yet built */
//     if (!comm_pkg)
//     {
//         jxf_MatvecCommPkgCreate(A);
//         comm_pkg = jxf_ParCSRMatrixCommPkg(A);
//     }

//     // 动态调整线程数
//     if (n <= 1000)
//     {
//         nt = 1; // n ≤ e^3, use 1 thread
//     }
//     else if (n <= 10000)
//     {
//         nt = 2; // e^3 < n ≤ 10^4, use 2 threads
//     }
//     else if (n <= 100000)
//     {
//         nt = 4; // e^4 < n ≤ 10^5, use 4 threads
//     }
//     else if (n <= 1000000)
//     {
//         nt = 8; // e^5 < n ≤ 10^6, use 8 threads
//     }
//     else if (n <= 10000000)
//     {
//         nt = 16; // e^6 < n ≤ 10^7, use 16 threads
//     }
//     else if (n <= 100000000)
//     {
//         nt = 32; // e^7 < n ≤ 10^8, use 32 threads
//     }
//     else
//     {
//         nt = 64; // n > e^8, use 64 threads
//     }

//     // 计算每个线程的行数，向上取整
//     ROW_OMP = (n + nt - 1) / nt;

//     /* 分配 workL 和 workU 记录每个线程的非零元素前缀和 */
//     workL = jxf_TAlloc(JXF_Int, nt + 1);
//     workU = jxf_TAlloc(JXF_Int, nt + 1);
//     for (i = 0; i <= nt; i++)
//     {
//         workL[i] = 0;
//         workU[i] = 0;
//     }

// // 并行计算 L 和 U 的非零元素数量，避免数据竞争
// #pragma omp parallel private(i, j, col, i_start, i_end) num_threads(nt)
//     {
//         JXF_Int tid = omp_get_thread_num();
//         JXF_Int local_workL = 0, local_workU = 0;
//         i_start = tid * ROW_OMP;
//         i_end = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//         for (i = i_start; i < i_end; i++)
//         {
//             for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
//             {
//                 col = A_diag_j[j];
//                 if (col < i)
//                     local_workL++;
//                 else if (col > i)
//                     local_workU++;
//             }
//         }
//         workL[tid + 1] = local_workL;
//         workU[tid + 1] = local_workU;
// #pragma omp barrier
// #pragma omp single
//         {
//             for (i = 1; i <= nt; i++)
//             {
//                 workL[i] += workL[i - 1];
//                 workU[i] += workU[i - 1];
//             }
//         }
//     }
//     L_nnz = workL[nt];
//     U_nnz = workU[nt];

//     // 错误检查
//     if (L_nnz < 0 || U_nnz < 0)
//     {
//         printf("Error: Invalid nnz values: L_nnz=%d, U_nnz=%d\n", L_nnz, U_nnz);
//         jxf_TFree(workL);
//         jxf_TFree(workU);
//         return -1;
//     }

//     // 分配矩阵数据
//     D_data = jxf_TAlloc(JXF_Real, n);
//     d1_data = jxf_TAlloc(JXF_Real, n);
//     d_data = jxf_TAlloc(JXF_Real, n);
//     L_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     L_diag_j = jxf_TAlloc(JXF_Int, L_nnz);
//     L_diag_data = jxf_TAlloc(JXF_Real, L_nnz);
//     L1_diag_data = jxf_TAlloc(JXF_Real, L_nnz);
//     U_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     U_diag_j = jxf_TAlloc(JXF_Int, U_nnz);
//     U_diag_data = jxf_TAlloc(JXF_Real, U_nnz);
//     U1_diag_data = jxf_TAlloc(JXF_Real, U_nnz);

// // 初始化 L 和 U 的索引和数据
// #pragma omp parallel for private(tid, i, j, col, i_start, i_end, countu, countl, t) num_threads(nt) schedule(static)
//     for (tid = 0; tid < nt; tid++)
//     {
//         countu = 0;
//         countl = 0;
//         i_start = tid * ROW_OMP;
//         i_end = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//         for (i = i_start; i < i_end; i++)
//         {
//             L_diag_i[i] = workL[tid] + countl;
//             U_diag_i[i] = workU[tid] + countu;

//             for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
//             {
//                 col = A_diag_j[j];
//                 t = A_diag_data[j];

//                 if (col > i)
//                 {
//                     U_diag_j[workU[tid] + countu] = col;
//                     U_diag_data[workU[tid] + countu] = t;
//                     countu++;
//                 }
//                 else if (col < i)
//                 {
//                     L_diag_j[workL[tid] + countl] = col;
//                     L_diag_data[workL[tid] + countl] = t;
//                     countl++;
//                 }
//                 else
//                 {
//                     d_data[i] = t;
//                     if (fabs(t) < MAT_TOL)
//                     {
//                         t = 1.0e-6;
//                     }
//                     D_data[i] = 1. / t;
//                 }
//             }
//         }
//     }
//     L_diag_i[n] = L_nnz;
//     U_diag_i[n] = U_nnz;

//     // 复制初始数据
//     memcpy(d1_data, d_data, n * sizeof(JXF_Real));
//     memcpy(L1_diag_data, L_diag_data, L_nnz * sizeof(JXF_Real));
//     memcpy(U1_diag_data, U_diag_data, U_nnz * sizeof(JXF_Real));

//     // 调试输出
//     if (my_id == 0)
//     {
//         printf("sweep = %d \n", sweep);
//     }

//     // ILU 因子分解的多次扫描
//     for (s = 0; s < sweep; s++)
//     {
// #pragma omp parallel num_threads(nt)
//         {
// // 更新 U 矩阵
// #pragma omp for private(ii, j, k, l, jj, col) schedule(static)
//             for (ii = 0; ii < n; ii++)
//             {
//                 for (j = U_diag_i[ii]; j < U_diag_i[ii + 1]; j++)
//                 {
//                     U_diag_data[j] = U1_diag_data[j]; // 恢复原始 U 值
//                     for (k = L_diag_i[ii]; k < L_diag_i[ii + 1]; k++)
//                     {
//                         jj = L_diag_j[k];
//                         for (l = U_diag_i[jj]; l < U_diag_i[jj + 1]; l++)
//                         {
//                             if (U_diag_j[l] == U_diag_j[j])
//                             {
//                                 U_diag_data[j] -= L_diag_data[k] * U_diag_data[l];
//                             }
//                         }
//                     }
//                 }
//             }
// #pragma omp barrier

// // 更新 L 矩阵
// #pragma omp for private(ii, j, k, l, jj, col) schedule(static)
//             for (ii = 0; ii < n; ii++)
//             {
//                 for (j = L_diag_i[ii]; j < L_diag_i[ii + 1]; j++)
//                 {
//                     L_diag_data[j] = L1_diag_data[j]; // 恢复原始 L 值
//                     col = L_diag_j[j];
//                     for (k = L_diag_i[ii]; k < L_diag_i[ii + 1]; k++)
//                     {
//                         if (L_diag_j[k] < col)
//                         {
//                             jj = L_diag_j[k];
//                             for (l = U_diag_i[jj]; l < U_diag_i[jj + 1]; l++)
//                             {
//                                 if (U_diag_j[l] == col)
//                                 {
//                                     L_diag_data[j] -= L_diag_data[k] * U_diag_data[l];
//                                 }
//                             }
//                         }
//                     }
//                     L_diag_data[j] *= D_data[col]; // 按对角线逆缩放
//                 }
//             }
// #pragma omp barrier

// // 更新对角线 d_data
// #pragma omp for private(ii, j, k, l, jj) schedule(static)
//             for (ii = 0; ii < n; ii++)
//             {
//                 d_data[ii] = d1_data[ii]; // 初始化为原始对角值
//                 for (k = L_diag_i[ii]; k < L_diag_i[ii + 1]; k++)
//                 {
//                     jj = L_diag_j[k];
//                     for (l = U_diag_i[jj]; l < U_diag_i[jj + 1]; l++)
//                     {
//                         if (U_diag_j[l] == ii)
//                         {
//                             d_data[ii] -= L_diag_data[k] * U_diag_data[l];
//                             break;
//                         }
//                     }
//                 }
//             }
// #pragma omp barrier

// // 更新 D_data
// #pragma omp for private(ii, t) schedule(static)
//             for (ii = 0; ii < n; ii++)
//             {
//                 t = d_data[ii];
//                 if (fabs(t) < MAT_TOL)
//                 {
//                     t = 1.0e-6;
//                 }
//                 D_data[ii] = 1. / t;
//             }
//         }
//     }

//     // 创建 L 矩阵
//     matL = jxf_ParCSRMatrixCreate(comm,
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A),
//                                  0,
//                                  L_nnz,
//                                  0);

//     L_diag = jxf_ParCSRMatrixDiag(matL);
//     jxf_CSRMatrixI(L_diag) = L_diag_i;
//     if (L_nnz)
//     {
//         jxf_CSRMatrixData(L_diag) = L_diag_data;
//         jxf_CSRMatrixJ(L_diag) = L_diag_j;
//     }
//     else
//     {
//         jxf_TFree(L_diag_j);
//         jxf_TFree(L_diag_data);
//     }
//     local_nnz = (JXF_Real)L_nnz;
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

//     // 创建 U 矩阵
//     matU = jxf_ParCSRMatrixCreate(comm,
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A),
//                                  0,
//                                  U_nnz,
//                                  0);

//     U_diag = jxf_ParCSRMatrixDiag(matU);
//     jxf_CSRMatrixI(U_diag) = U_diag_i;
//     if (U_nnz)
//     {
//         jxf_CSRMatrixData(U_diag) = U_diag_data;
//         jxf_CSRMatrixJ(U_diag) = U_diag_j;
//     }
//     else
//     {
//         jxf_TFree(U_diag_j);
//         jxf_TFree(U_diag_data);
//     }
//     local_nnz = (JXF_Real)U_nnz;
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

//     // 设置输出参数
//     *Lptr = matL;
//     *Dptr = D_data;
//     *Uptr = matU;

//     // 释放临时内存
//     jxf_TFree(workL);
//     jxf_TFree(workU);
//     jxf_TFree(L1_diag_data);
//     jxf_TFree(U1_diag_data);
//     jxf_TFree(d_data);
//     jxf_TFree(d1_data);

//     return jxf_error_flag;
// }
// // JXF_Int jxf_ILUSetupFGPILU_v2(jxf_ParCSRMatrix  *A, JXF_Int  *permp,JXF_Int  *qpermp,
// //                         JXF_Int  sweep,jxf_ParCSRMatrix **Lptr,JXF_Real **Dptr,
// //                         jxf_ParCSRMatrix **Uptr, jxf_ParVector **D_array, JXF_Int  **u_end)

// JXF_Int jxf_ILUSetupFGPILU_v2(jxf_ParCSRMatrix *A,
//                          JXF_Int *permp,
//                          JXF_Int *qpermp,
//                          JXF_Int sweep,
//                          jxf_ParCSRMatrix **Lptr,
//                          JXF_Real **Dptr,
//                          jxf_ParCSRMatrix **Uptr,
//                          jxf_ParVector **D_array,
//                          JXF_Int **u_end)
// {
//     MPI_Comm comm = jxf_ParCSRMatrixComm(A);
//     JXF_Int num_procs, my_id;
//     jxf_ParCSRCommPkg *comm_pkg;
//     JXF_Int i, s, j, k, l, jj, col, countu, countl, ROW_OMP,temp_idx;
//     JXF_Int tid, i_start, i_end;
//     JXF_Real t, local_nnz, total_nnz,temp_data;

//     /* Data objects for A */
//     jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//     jxf_CSRMatrix *A_offd = jxf_ParCSRMatrixOffd(A);
//     JXF_Real *A_diag_data = jxf_CSRMatrixData(A_diag);
//     JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//     JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);
//     JXF_Int n = jxf_CSRMatrixNumRows(A_diag);
//     JXF_Int nnz_A = A_diag_i[n];

//     /* Data objects for L, D, U */
//     jxf_ParCSRMatrix *matL, *matU;
//     jxf_CSRMatrix *L_diag, *U_diag;
//     JXF_Real *D_data, *d_data, *d1_data;
//     JXF_Real *L_diag_data, *L1_diag_data;
//     JXF_Int *L_diag_i, *L_diag_j;
//     JXF_Real *U_diag_data, *U1_diag_data;
//     JXF_Int *U_diag_i, *U_diag_j;
//     JXF_Int *workL, *workU;
//     JXF_Int L_nnz = 0, U_nnz = 0;
//     JXF_Int nt = jxf_NumThreads();

//     /* MPI initialization */
//     jxf_MPI_Comm_size(comm, &num_procs);
//     jxf_MPI_Comm_rank(comm, &my_id);
//     comm_pkg = jxf_ParCSRMatrixCommPkg(A);
//     if (!comm_pkg) {
//         jxf_MatvecCommPkgCreate(A);
//         comm_pkg = jxf_ParCSRMatrixCommPkg(A);
//     }

//     /* Dynamic thread adjustment */
//    //  if (n <= 1000) nt = 1;
//    //  else if (n <= 10000) nt = 2;
//    //  else if (n <= 100000) nt = 4;
//    //  else if (n <= 1000000) nt = 8;
//    //  else if (n <= 10000000) nt = 16;
//    //  else if (n <= 100000000) nt = 32;
//    //  else nt = 64;

//     ROW_OMP = (n + nt - 1) / nt;

//     /* Allocate workL and workU for non-zero prefix sums */
//     workL = jxf_TAlloc(JXF_Int, nt + 1);
//     workU = jxf_TAlloc(JXF_Int, nt + 1);
//     if (!workL || !workU) {
//         printf("Error: Memory allocation failed for workL or workU\n");
//         jxf_TFree(workL);
//         jxf_TFree(workU);
//         return -1;
//     }
//     for (i = 0; i <= nt; i++) {
//         workL[i] = 0;
//         workU[i] = 0;
//     }

//     /* Parallel computation of L and U non-zero counts */
//     #pragma omp parallel private(i, j, col, i_start, i_end) num_threads(nt)
//     {
//         int tid = omp_get_thread_num();
//         int local_workL = 0, local_workU = 0;
//         i_start = tid * ROW_OMP;
//         i_end = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//         for (i = i_start; i < i_end; i++) {
//             for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++) {
//                 col = A_diag_j[j];
//                 if (col < i)
//                     local_workL++;
//                 else if (col > i)
//                     local_workU++;
//             }
//         }
//         workL[tid + 1] = local_workL;
//         workU[tid + 1] = local_workU;
//         #pragma omp barrier
//         #pragma omp single
//         {
//             for (i = 1; i <= nt; i++) {
//                 workL[i] += workL[i - 1];
//                 workU[i] += workU[i - 1];
//             }
//         }
//     }
//     L_nnz = workL[nt];
//     U_nnz = workU[nt];

//     /* Error checking */
//     if (L_nnz < 0 || U_nnz < 0) {
//         printf("Error: Invalid nnz values: L_nnz=%d, U_nnz=%d\n", L_nnz, U_nnz);
//         jxf_TFree(workL);
//         jxf_TFree(workU);
//         return -1;
//     }

//     /* Allocate matrices */
//     D_data = jxf_TAlloc(JXF_Real, n);
//     d_data = jxf_TAlloc(JXF_Real, n);
//     d1_data = jxf_TAlloc(JXF_Real, n);
//     L_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     U_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     L_diag_j = jxf_TAlloc(JXF_Int, L_nnz);
//     L_diag_data = jxf_TAlloc(JXF_Real, L_nnz);
//     U_diag_j = jxf_TAlloc(JXF_Int, U_nnz);
//     U_diag_data = jxf_TAlloc(JXF_Real, U_nnz);
//     L1_diag_data = jxf_TAlloc(JXF_Real, L_nnz);
//     U1_diag_data = jxf_TAlloc(JXF_Real, U_nnz);

//     if (!D_data || !d_data || !d1_data || !L_diag_i || !U_diag_i ||
//         !L_diag_j || !L_diag_data || !U_diag_j || !U_diag_data ||
//         !L1_diag_data || !U1_diag_data) {
//         printf("Error: Memory allocation failed for matrices\n");
//         jxf_TFree(D_data); jxf_TFree(d_data); jxf_TFree(d1_data);
//         jxf_TFree(L_diag_i); jxf_TFree(U_diag_i);
//         jxf_TFree(L_diag_j); jxf_TFree(L_diag_data);
//         jxf_TFree(U_diag_j); jxf_TFree(U_diag_data);
//         jxf_TFree(L1_diag_data); jxf_TFree(U1_diag_data);
//         jxf_TFree(workL); jxf_TFree(workU);
//         return -1;
//     }

//     /* Initialize L and U indices and data */
//     #pragma omp parallel for private(tid, i, j, col, i_start, i_end, countu, countl, t) num_threads(nt) schedule(static)
//     for (tid = 0; tid < nt; tid++) {
//         countu = 0;
//         countl = 0;
//         i_start = tid * ROW_OMP;
//         i_end = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//         for (i = i_start; i < i_end; i++) {
//             L_diag_i[i] = workL[tid] + countl;
//             U_diag_i[i] = workU[tid] + countu;

//             for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++) {
//                 col = A_diag_j[j];
//                 t = A_diag_data[j];

//                 if (col > i) {
//                     U_diag_j[workU[tid] + countu] = col;
//                     U_diag_data[workU[tid] + countu] = t;
//                     countu++;
//                 } else if (col < i) {
//                     L_diag_j[workL[tid] + countl] = col;
//                     L_diag_data[workL[tid] + countl] = t;
//                     countl++;
//                 } else {
//                     d_data[i] = t;
//                     if (fabs(t) < MAT_TOL) {
//                         t = 1.0e-6;
//                     }
//                     D_data[i] = 1. / t;
//                 }
//             }
//         }
//     }
//     L_diag_i[n] = L_nnz;
//     U_diag_i[n] = U_nnz;

//     /* Sort L_diag_j and U_diag_j using insertion sort */
//     #pragma omp parallel for private(i, j, k, temp_idx, temp_data) num_threads(nt) schedule(static)
//     for (i = 0; i < n; i++) {
//         /* Sort L_diag_j and L_diag_data */
//         if (L_diag_i[i + 1] > L_diag_i[i]) {
//             JXF_Int len = L_diag_i[i + 1] - L_diag_i[i];
//             for (j = L_diag_i[i] + 1; j < L_diag_i[i + 1]; j++) {
//                 temp_idx = L_diag_j[j];
//                 temp_data = L_diag_data[j];
//                 k = j - 1;
//                 while (k >= L_diag_i[i] && L_diag_j[k] > temp_idx) {
//                     L_diag_j[k + 1] = L_diag_j[k];
//                     L_diag_data[k + 1] = L_diag_data[k];
//                     k--;
//                 }
//                 L_diag_j[k + 1] = temp_idx;
//                 L_diag_data[k + 1] = temp_data;
//             }
//         }
//         /* Sort U_diag_j and U_diag_data */
//         if (U_diag_i[i + 1] > U_diag_i[i]) {
//             JXF_Int len = U_diag_i[i + 1] - U_diag_i[i];
//             for (j = U_diag_i[i] + 1; j < U_diag_i[i + 1]; j++) {
//                 temp_idx = U_diag_j[j];
//                 temp_data = U_diag_data[j];
//                 k = j - 1;
//                 while (k >= U_diag_i[i] && U_diag_j[k] > temp_idx) {
//                     U_diag_j[k + 1] = U_diag_j[k];
//                     U_diag_data[k + 1] = U_diag_data[k];
//                     k--;
//                 }
//                 U_diag_j[k + 1] = temp_idx;
//                 U_diag_data[k + 1] = temp_data;
//             }
//         }
//     }

//     memcpy(d1_data, d_data, n * sizeof(JXF_Real));
//     memcpy(L1_diag_data, L_diag_data, L_nnz * sizeof(JXF_Real));
//     memcpy(U1_diag_data, U_diag_data, U_nnz * sizeof(JXF_Real));

//     /* Debug output */
//     if (my_id == 0) {
//         printf("sweep = %d \n", sweep);
//     }

//     /* ILU sweeps */
//     for (s = 0; s < sweep; s++) {
//         #pragma omp parallel num_threads(nt)
//         {
//             /* Update U matrix */
//             #pragma omp for private(i, j, k, l, jj, col) schedule(static)
//             for (i = 0; i < n; i++) {
//                 for (j = U_diag_i[i]; j < U_diag_i[i + 1]; j++) {
//                     U_diag_data[j] = U1_diag_data[j];
//                     for (k = L_diag_i[i]; k < L_diag_i[i + 1]; k++) {
//                         jj = L_diag_j[k];
//                         for (l = U_diag_i[jj]; l < U_diag_i[jj + 1]; l++) {
//                             if (U_diag_j[l] == U_diag_j[j]) {
//                                 U_diag_data[j] -= L_diag_data[k] * U_diag_data[l];
//                             }
//                         }
//                     }
//                 }
//             }
//             #pragma omp barrier

//             /* Update L matrix */
//             #pragma omp for private(i, j, k, l, jj, col) schedule(static)
//             for (i = 0; i < n; i++) {
//                 for (j = L_diag_i[i]; j < L_diag_i[i + 1]; j++) {
//                     L_diag_data[j] = L1_diag_data[j];
//                     col = L_diag_j[j];
//                     for (k = L_diag_i[i]; k < L_diag_i[i + 1]; k++) {
//                         if (L_diag_j[k] < col) {
//                             jj = L_diag_j[k];
//                             for (l = U_diag_i[jj]; l < U_diag_i[jj + 1]; l++) {
//                                 if (U_diag_j[l] == col) {
//                                     L_diag_data[j] -= L_diag_data[k] * U_diag_data[l];
//                                 }
//                             }
//                         }
//                     }
//                     L_diag_data[j] *= D_data[col];
//                 }
//             }
//             #pragma omp barrier

//             /* Update diagonal d_data */
//             #pragma omp for private(i, j, k, l, jj) schedule(static)
//             for (i = 0; i < n; i++) {
//                 d_data[i] = d1_data[i];
//                 for (k = L_diag_i[i]; k < L_diag_i[i + 1]; k++) {
//                     jj = L_diag_j[k];
//                     for (l = U_diag_i[jj]; l < U_diag_i[jj + 1]; l++) {
//                         if (U_diag_j[l] == i) {
//                             d_data[i] -= L_diag_data[k] * U_diag_data[l];
//                             break;
//                         }
//                     }
//                 }
//             }
//             #pragma omp barrier

//             /* Update D_data */
//             #pragma omp for private(i, t) schedule(static)
//             for (i = 0; i < n; i++) {
//                 t = d_data[i];
//                 if (fabs(t) < MAT_TOL) t = 1.0e-6;
//                 D_data[i] = 1. / t;
//             }
//         }
//     }

//     /* Create L matrix */
//     matL = jxf_ParCSRMatrixCreate(comm, jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A), 0, L_nnz, 0);
//     L_diag = jxf_ParCSRMatrixDiag(matL);
//     jxf_CSRMatrixI(L_diag) = L_diag_i;
//     if (L_nnz) {
//         jxf_CSRMatrixData(L_diag) = L_diag_data;
//         jxf_CSRMatrixJ(L_diag) = L_diag_j;
//     } else {
//         jxf_TFree(L_diag_j);
//         jxf_TFree(L_diag_data);
//     }
//     local_nnz = (JXF_Real)L_nnz;
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

//     /* Create U matrix */
//     matU = jxf_ParCSRMatrixCreate(comm, jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A), 0, U_nnz, 0);
//     U_diag = jxf_ParCSRMatrixDiag(matU);
//     jxf_CSRMatrixI(U_diag) = U_diag_i;
//     if (U_nnz) {
//         jxf_CSRMatrixData(U_diag) = U_diag_data;
//         jxf_CSRMatrixJ(U_diag) = U_diag_j;
//     } else {
//         jxf_TFree(U_diag_j);
//         jxf_TFree(U_diag_data);
//     }
//     local_nnz = (JXF_Real)U_nnz;
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

//     /* Set output parameters */
//     *Lptr = matL;
//     *Dptr = D_data;
//     *Uptr = matU;
//     // *Sptr = NULL;
//     *u_end = NULL;

//     /* Free temporary memory */
//     jxf_TFree(workL);
//     jxf_TFree(workU);
//     jxf_TFree(d_data);
//     jxf_TFree(d1_data);
//     jxf_TFree(L1_diag_data);
//     jxf_TFree(U1_diag_data);

//     return jxf_error_flag;
// }


// JXF_Int jxf_ILUSetupFGPILU_v2(jxf_ParCSRMatrix  *A, JXF_Int  *permp,JXF_Int  *qpermp,
//                         JXF_Int  sweep,jxf_ParCSRMatrix **Lptr,JXF_Real **Dptr,
//                         jxf_ParCSRMatrix **Uptr, jxf_ParVector **D_array, JXF_Int  **u_end)
// {
//     MPI_Comm comm = jxf_ParCSRMatrixComm(A);
//     JXF_Int num_procs, my_id;
//     jxf_ParCSRCommPkg *comm_pkg;
//     JXF_Int i, s, j, k, l, jj, col, countu, countl, ROW_OMP, temp_idx;
//     JXF_Int tid, i_start, i_end;
//     JXF_Real t, local_nnz, total_nnz, temp_data;

//     /* Data objects for A */
//     jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//     jxf_CSRMatrix *A_offd = jxf_ParCSRMatrixOffd(A);
//     JXF_Real *A_diag_data = jxf_CSRMatrixData(A_diag);
//     JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//     JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);
//     JXF_Int n = jxf_CSRMatrixNumRows(A_diag);
//     JXF_Int nnz_A = A_diag_i[n];

//     /* Data objects for L, D, U */
//     jxf_ParCSRMatrix *matL, *matU;
//     jxf_CSRMatrix *L_diag, *U_diag;
//     JXF_Real *D_data, *D1_data, *d_data, *d1_data;
//     JXF_Real *L_diag_data, *L1_diag_data,*L2_diag_data;
//     JXF_Int *L_diag_i, *L_diag_j;
//     JXF_Real *U_diag_data, *U1_diag_data,*U2_diag_data;
//     JXF_Int *U_diag_i, *U_diag_j;
//     JXF_Int *workL, *workU;
//     JXF_Int L_nnz = 0, U_nnz = 0;
//     JXF_Int nt = jxf_NumThreads();

//     jxf_ParVector *vec_D;
//     jxf_Vector *D_local;

//     /* MPI initialization */
//     jxf_MPI_Comm_size(comm, &num_procs);
//     jxf_MPI_Comm_rank(comm, &my_id);
//     comm_pkg = jxf_ParCSRMatrixCommPkg(A);
//     if (!comm_pkg)
//     {
//         jxf_MatvecCommPkgCreate(A);
//         comm_pkg = jxf_ParCSRMatrixCommPkg(A);
//     }

//     ROW_OMP = (n + nt - 1) / nt;

//     /* Allocate workL and workU for non-zero prefix sums */
//     workL = jxf_TAlloc(JXF_Int, nt + 1);
//     workU = jxf_TAlloc(JXF_Int, nt + 1);
//     if (!workL || !workU)
//     {
//         printf("Error: Memory allocation failed for workL or workU\n");
//         jxf_TFree(workL);
//         jxf_TFree(workU);
//         return -1;
//     }
//     for (i = 0; i <= nt; i++)
//     {
//         workL[i] = 0;
//         workU[i] = 0;
//     }

// /* Parallel computation of L and U non-zero counts */
// #pragma omp parallel private(i, j, col, i_start, i_end) num_threads(nt)
//     {
//         JXF_Int tid = omp_get_thread_num();
//         JXF_Int local_workL = 0, local_workU = 0;
//         i_start = tid * ROW_OMP;
//         i_end = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//         for (i = i_start; i < i_end; i++)
//         {
//             for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
//             {
//                 col = A_diag_j[j];
//                 if (col < i)
//                     local_workL++;
//                 else if (col > i)
//                     local_workU++;
//             }
//         }
//         workL[tid + 1] = local_workL;
//         workU[tid + 1] = local_workU;
// #pragma omp barrier
// #pragma omp single
//         {
//             for (i = 1; i <= nt; i++)
//             {
//                 workL[i] += workL[i - 1];
//                 workU[i] += workU[i - 1];
//             }
//         }
//     }
//     L_nnz = workL[nt];
//     U_nnz = workU[nt];

//     /* Error checking */
//     if (L_nnz < 0 || U_nnz < 0)
//     {
//         printf("Error: Invalid nnz values: L_nnz=%d, U_nnz=%d\n", L_nnz, U_nnz);
//         jxf_TFree(workL);
//         jxf_TFree(workU);
//         return -1;
//     }

//     /* Allocate matrices */
//     D_data = jxf_TAlloc(JXF_Real, n);
//     D1_data = jxf_TAlloc(JXF_Real, n);
//     d_data = jxf_TAlloc(JXF_Real, n);
//     d1_data = jxf_TAlloc(JXF_Real, n);
//     L_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     U_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     L_diag_j = jxf_TAlloc(JXF_Int, L_nnz);
//     L_diag_data = jxf_TAlloc(JXF_Real, L_nnz);
//     U_diag_j = jxf_TAlloc(JXF_Int, U_nnz);
//     U_diag_data = jxf_TAlloc(JXF_Real, U_nnz);
//     L1_diag_data = jxf_TAlloc(JXF_Real, L_nnz);
//     U1_diag_data = jxf_TAlloc(JXF_Real, U_nnz);

//     // //中间变量
//     L2_diag_data = jxf_TAlloc(JXF_Real, L_nnz);
//     U2_diag_data = jxf_TAlloc(JXF_Real, U_nnz);

//     if (!D_data || !d_data || !d1_data || !L_diag_i || !U_diag_i ||
//         !L_diag_j || !L_diag_data || !U_diag_j || !U_diag_data ||
//         !L1_diag_data || !U1_diag_data)
//     {
//         printf("Error: Memory allocation failed for matrices\n");
//         jxf_TFree(D_data);
//         jxf_TFree(d_data);
//         jxf_TFree(d1_data);
//         jxf_TFree(L_diag_i);
//         jxf_TFree(U_diag_i);
//         jxf_TFree(L_diag_j);
//         jxf_TFree(L_diag_data);
//         jxf_TFree(U_diag_j);
//         jxf_TFree(U_diag_data);
//         jxf_TFree(L1_diag_data);
//         jxf_TFree(U1_diag_data);
//         jxf_TFree(workL);
//         jxf_TFree(workU);
//         return -1;
//     }

// /* Initialize L and U indices and data */
// #pragma omp parallel for private(tid, i, j, col, i_start, i_end, countu, countl, t) num_threads(nt) schedule(static)
//     for (tid = 0; tid < nt; tid++)
//     {
//         countu = 0;
//         countl = 0;
//         i_start = tid * ROW_OMP;
//         i_end = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//         for (i = i_start; i < i_end; i++)
//         {
//             L_diag_i[i] = workL[tid] + countl;
//             U_diag_i[i] = workU[tid] + countu;

//             for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
//             {
//                 col = A_diag_j[j];
//                 t = A_diag_data[j];

//                 if (col > i)
//                 {
//                     U_diag_j[workU[tid] + countu] = col;
//                     U_diag_data[workU[tid] + countu] = t;
//                     countu++;
//                 }
//                 else if (col < i)
//                 {
//                     L_diag_j[workL[tid] + countl] = col;
//                     L_diag_data[workL[tid] + countl] = t;
//                     countl++;
//                 }
//                 else
//                 {
//                     d_data[i] = t;
//                     if (fabs(t) < MAT_TOL)
//                     {
//                         t = 1.0e-6;
//                     }
//                     D_data[i] = 1. / t;
//                 }
//             }
//         }
//     }
//     L_diag_i[n] = L_nnz;
//     U_diag_i[n] = U_nnz;

// /* Sort L_diag_j and U_diag_j using insertion sort */
// #pragma omp parallel for private(i, j, k, temp_idx, temp_data) num_threads(nt) schedule(static)
//     for (i = 0; i < n; i++)
//     {
//         /* Sort L_diag_j and L_diag_data */
//         if (L_diag_i[i + 1] > L_diag_i[i])
//         {
//             JXF_Int len = L_diag_i[i + 1] - L_diag_i[i];
//             for (j = L_diag_i[i] + 1; j < L_diag_i[i + 1]; j++)
//             {
//                 temp_idx = L_diag_j[j];
//                 temp_data = L_diag_data[j];
//                 k = j - 1;
//                 while (k >= L_diag_i[i] && L_diag_j[k] > temp_idx)
//                 {
//                     L_diag_j[k + 1] = L_diag_j[k];
//                     L_diag_data[k + 1] = L_diag_data[k];
//                     k--;
//                 }
//                 L_diag_j[k + 1] = temp_idx;
//                 L_diag_data[k + 1] = temp_data;
//             }
//         }
//         /* Sort U_diag_j and U_diag_data */
//         if (U_diag_i[i + 1] > U_diag_i[i])
//         {
//             JXF_Int len = U_diag_i[i + 1] - U_diag_i[i];
//             for (j = U_diag_i[i] + 1; j < U_diag_i[i + 1]; j++)
//             {
//                 temp_idx = U_diag_j[j];
//                 temp_data = U_diag_data[j];
//                 k = j - 1;
//                 while (k >= U_diag_i[i] && U_diag_j[k] > temp_idx)
//                 {
//                     U_diag_j[k + 1] = U_diag_j[k];
//                     U_diag_data[k + 1] = U_diag_data[k];
//                     k--;
//                 }
//                 U_diag_j[k + 1] = temp_idx;
//                 U_diag_data[k + 1] = temp_data;
//             }
//         }
//     }

//     memcpy(d1_data, d_data, n * sizeof(JXF_Real));
//     // memcpy(L1_diag_data, L_diag_data, L_nnz * sizeof(JXF_Real));
//     // memcpy(U1_diag_data, U_diag_data, U_nnz * sizeof(JXF_Real));


// for (i = 0; i < n; i++)
//     {
//         for (j = L_diag_i[i]; j < L_diag_i[i + 1]; j++)
//         {
//             temp_data = L_diag_data[j];
//             L_diag_data[j] = D_data[i] * temp_data;
//         }
//     }


//     /* ILU sweeps */
//     for (s = 0; s < sweep; s++)
//     {
// #pragma omp parallel num_threads(nt)
//         {

//             /* Update diagonal d_data */
// #pragma omp for private(i, j, k, l, jj) schedule(static)
//             for (i = 0; i < n; i++)
//             {
//                 d_data[i] = d1_data[i];
//                 for (k = L_diag_i[i]; k < L_diag_i[i + 1]; k++)
//                 {
//                     jj = L_diag_j[k];
//                     for (l = U_diag_i[jj]; l < U_diag_i[jj + 1]; l++)
//                     {
//                         if (U_diag_j[l] == i)
//                         {
//                             d_data[i] -= L_diag_data[k] * U_diag_data[l];
//                         }
//                     }
//                 }
//             }
// #pragma omp barrier

// /* Update D_data */
// #pragma omp for private(i, t) schedule(static)
//             for (i = 0; i < n; i++)
//             {
//                 t = d_data[i];
//                 if (fabs(t) < MAT_TOL)
//                     t = 1.0e-6;
//                 D_data[i] = 1. / t;
//             }


// /* Update U matrix */
// #pragma omp for private(i, j, k, l, jj, col) schedule(static)
//             for (i = 0; i < n; i++)
//             {
//                 for (j = U_diag_i[i]; j < U_diag_i[i + 1]; j++)
//                 {
//                     // U2_diag_data[j] = U1_diag_data[j];
//                     for (k = L_diag_i[i]; k < L_diag_i[i + 1]; k++)
//                     {
//                         jj = L_diag_j[k];
//                         for (l = U_diag_i[jj]; l < U_diag_i[jj + 1]; l++)
//                         {
//                             if (U_diag_j[l] == U_diag_j[j])
//                             {
//                                 U_diag_data[j] -= L_diag_data[k] * U_diag_data[l];
//                             }
//                         }
//                     }
//                 }
//             }
// #pragma omp barrier

// /* Update L matrix */
// #pragma omp for private(i, j, k, l, jj, col) schedule(static)
//             for (i = 0; i < n; i++)
//             {
//                 for (j = L_diag_i[i]; j < L_diag_i[i + 1]; j++)
//                 {
//                     // L2_diag_data[j] = L1_diag_data[j];
//                     col = L_diag_j[j];
//                     for (k = L_diag_i[i]; k < L_diag_i[i + 1]; k++)
//                     {
//                         if (L_diag_j[k] < col)
//                         {
//                             jj = L_diag_j[k];
//                             for (l = U_diag_i[jj]; l < U_diag_i[jj + 1]; l++)
//                             {
//                                 if (U_diag_j[l] == col)
//                                 {
//                                     L_diag_data[j] -= L_diag_data[k] * U_diag_data[l];
//                                 }
//                             }
//                         }
//                     }
//                     L_diag_data[j] *= D_data[col];
//                 }
//             }
//             // memcpy(L_diag_data, L2_diag_data, L_nnz * sizeof(JXF_Real));
//             // memcpy(U_diag_data, U2_diag_data, U_nnz * sizeof(JXF_Real));

// #pragma omp barrier
//         }
//     }

//     /* Create L matrix */
//     matL = jxf_ParCSRMatrixCreate(comm, jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A), 0, L_nnz, 0);
//     L_diag = jxf_ParCSRMatrixDiag(matL);
//     jxf_CSRMatrixI(L_diag) = L_diag_i;
//     if (L_nnz)
//     {
//         jxf_CSRMatrixData(L_diag) = L_diag_data;
//         jxf_CSRMatrixJ(L_diag) = L_diag_j;
//     }
//     else
//     {
//         jxf_TFree(L_diag_j);
//         jxf_TFree(L_diag_data);
//     }
//     local_nnz = (JXF_Real)L_nnz;
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

//     /* Create U matrix */
//     matU = jxf_ParCSRMatrixCreate(comm, jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A), 0, U_nnz, 0);
//     U_diag = jxf_ParCSRMatrixDiag(matU);
//     jxf_CSRMatrixI(U_diag) = U_diag_i;
//     if (U_nnz)
//     {
//         jxf_CSRMatrixData(U_diag) = U_diag_data;
//         jxf_CSRMatrixJ(U_diag) = U_diag_j;
//     }
//     else
//     {
//         jxf_TFree(U_diag_j);
//         jxf_TFree(U_diag_data);
//     }
//     local_nnz = (JXF_Real)U_nnz;
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matU) = total_nnz;


//     vec_D = jxf_ParVectorCreate(jxf_ParCSRMatrixComm(A),
//                                 jxf_ParCSRMatrixGlobalNumRows(A),
//                                 jxf_ParCSRMatrixRowStarts(A) );
//     jxf_ParVectorInitialize(vec_D);
//     D_local = jxf_ParVectorLocalVector(vec_D);
//     D1_data = jxf_VectorData(D_local);

//     memcpy(D1_data, D_data, n * sizeof(JXF_Real));

//     /* Set output parameters */
//     *Lptr = matL;
//     *Dptr = D_data;
//     *Uptr = matU;
//     *u_end = NULL;
//     *D_array = vec_D;

// //     char FileNameCoaMat[256];
// //     jxf_sprintf(FileNameCoaMat, "L_CSR_%d", 2);
// //     jxf_ParCSRMatrixPrint(matL, FileNameCoaMat);
// //     jxf_sprintf(FileNameCoaMat, "U_CSR_%d", 2);
// //     jxf_ParCSRMatrixPrint(matU, FileNameCoaMat);

// //    FILE *fp;
// //    jxf_sprintf(FileNameCoaMat, "D_%d", 2);
// //    fp = fopen(FileNameCoaMat, "w");
// //    for(j = 0; j < n; j++){
// //    jxf_fprintf(fp, "%.14e\n", D_data[j]);
// //    }
// //    fclose(fp);


//     /* Free temporary memory */
//     jxf_TFree(workL);
//     jxf_TFree(workU);
//     jxf_TFree(d_data);
//     jxf_TFree(d1_data);
//     jxf_TFree(L1_diag_data);
//     jxf_TFree(U1_diag_data);

//     return jxf_error_flag;
// }




// void jxf_CSRMatrixTopologicSortILU(jxf_CSRMatrix *L, jxf_CSRMatrix *U,
//                                   JXF_Int *nlevL, JXF_Int **jlevL, JXF_Int **ilevL,
//                                   JXF_Int *nlevU, JXF_Int **jlevU, JXF_Int **ilevU)
// {
//     JXF_Int *L_I = jxf_CSRMatrixI(L);
//     JXF_Int *L_J = jxf_CSRMatrixJ(L);
//     JXF_Int *U_I = jxf_CSRMatrixI(U);
//     JXF_Int *U_J = jxf_CSRMatrixJ(U);
//     JXF_Int n = jxf_CSRMatrixNumRows(L);

//     // 检查矩阵行数一致性
//     if (jxf_CSRMatrixNumRows(L) != jxf_CSRMatrixNumRows(U)) {
//         *jlevL = *ilevL = *jlevU = *ilevU = NULL;
//         *nlevL = *nlevU = 0;
//         return;
//     }
//     // 分配内存
//     JXF_Int *level = (JXF_Int *)calloc(n, sizeof(JXF_Int));
//     *jlevL = (JXF_Int *)calloc(n, sizeof(JXF_Int));
//     *ilevL = (JXF_Int *)calloc(n + 1, sizeof(JXF_Int));
//     *jlevU = (JXF_Int *)calloc(n, sizeof(JXF_Int));
//     *ilevU = (JXF_Int *)calloc(n + 1, sizeof(JXF_Int));

//     // 检查内存分配
//     if (!level || !(*jlevL) || !(*ilevL) || !(*jlevU) || !(*ilevU)) {
//         free(level);
//         free(*jlevL); free(*ilevL); free(*jlevU); free(*ilevU);
//         *jlevL = *ilevL = *jlevU = *ilevU = NULL;
//         *nlevL = *nlevU = 0;
//         return;
//     }

//     JXF_Int i, j, k, l;
//     JXF_Int local_nlevL = 0, local_nlevU = 0;

//     // for (i = 0; i <= n; i++) printf("%d ", L_I[i]);
//     // printf("\n");
//     // for (i = 0; i < L_I[n]; i++) printf("%d ", L_J[i]);
//     // printf("\n");

//     // for (i = 0; i <= n; i++) printf("%d ", U_I[i]);
//     // printf("\n");
//     // for (i = 0; i < U_I[n]; i++) printf("%d ", U_J[i]);
//     // printf("\n");

//     for(i=0; i<n; i++) level[i] = 0;

//     // 初始化 ilevL
//     for (i = 0; i <= n; i++) (*ilevL)[i] = 0;
//     for (i = 0; i <= n; i++) (*ilevU)[i] = 0;
//     for (i = 0; i < n; i++) (*jlevU)[i] = 0;
//     for (i = 0; i < n; i++) (*jlevL)[i] = 0;

//     // ---------- Step 1: 构造下三角矩阵 L 的层次 ----------
//     for (i = 0; i < n; i++) {
//         l = 0;
//         for (j = L_I[i]; j < L_I[i + 1]; j++) {
//             if (L_J[j] < i) {
//                 l = (l > level[L_J[j]]) ? l : level[L_J[j]];
//             }
//         }
//         level[i] = l + 1;
//         (*ilevL)[l + 1]++;
//         //printf("%d",l + 1);
//         local_nlevL = (local_nlevL > (l + 1)) ? local_nlevL : (l + 1);
//     }
//     //printf("\n");

//     for (i = 1; i <= local_nlevL; i++) (*ilevL)[i] += (*ilevL)[i - 1];

//     for (i = 0; i < n; i++) {
//         k = (*ilevL)[level[i] - 1];
//         (*jlevL)[k] = i;
//         (*ilevL)[level[i] - 1]++;
//     }

//     for (i = local_nlevL - 1; i > 0; i--) (*ilevL)[i] = (*ilevL)[i - 1];
//     (*ilevL)[0] = 0;

//     *nlevL = local_nlevL + 1; // 保持原始代码的 +1 逻辑

//     // printf("Lower Triangular Matrix L:\n");
//     // printf("Number of levels (nlevL): %d\n", *nlevL);
//     // printf("jlevL (row indices per level): ");
//     // for (i = 0; i < n; i++) printf("%d ", (*jlevL)[i]);
//     // printf("\n");
//     // printf("ilevL (level offsets): ");
//     // for (i = 0; i <= local_nlevL; i++) printf("%d ", (*ilevL)[i]);
//     // printf("\n\n");

//     // ---------- Step 2: 构造上三角矩阵 U 的层次 ----------
//     for (i = 0; i < n; i++) level[i] = 0; // 重新清零 level
//     for (i = 0; i <= n; i++) (*ilevU)[i] = 0; // 初始化 ilevU

//     for (i = n - 1; i >= 0; i--) {
//         l = 0;
//         for (j = U_I[i]; j < U_I[i + 1]; j++) {
//             if (U_J[j] > i) {
//                 l = (l > level[U_J[j]]) ? l : level[U_J[j]];
//             }
//         }
//         level[i] = l + 1;
//         (*ilevU)[l + 1]++;
//         //printf("%d",l + 1);
//         local_nlevU = (local_nlevU > (l + 1)) ? local_nlevU : (l + 1);
//     }
//     //printf("\n");
//     for (i = 1; i <= local_nlevU; i++) (*ilevU)[i] += (*ilevU)[i - 1];

//     for (i = n - 1; i >= 0; i--) {
//         k = (*ilevU)[level[i] - 1];
//         (*jlevU)[k] = i;
//         (*ilevU)[level[i] - 1]++;
//     }

//     for (i = local_nlevU - 1; i > 0; i--) (*ilevU)[i] = (*ilevU)[i - 1];
//     (*ilevU)[0] = 0;

//     *nlevU = local_nlevU + 1;

//    //  printf("Upper Triangular Matrix U:\n");
//    //  printf("Number of levels (nlevU): %d\n", *nlevU);
//    //  printf("jlevU (row indices per level): ");
//    //  for (i = 0; i < n; i++) printf("%d ", (*jlevU)[i]);
//    //  printf("\n");
//    //  printf("ilevU (level offsets): ");
//    //  for (i = 0; i <= local_nlevU; i++) printf("%d ", (*ilevU)[i]);
//    //  printf("\n");

//     free(level);
// }

// void jxf_CSRMatrixTopologicSortILU1(jxf_CSRMatrix *L, jxf_CSRMatrix *U,
//                                    JXF_Int *nlevL, JXF_Int **jlevL, JXF_Int **ilevL, JXF_Int **permL, JXF_Int **inv_permL,
//                                    JXF_Int *nlevU, JXF_Int **jlevU, JXF_Int **ilevU, JXF_Int **permU, JXF_Int **inv_permU)
// {
//     JXF_Int *L_I = jxf_CSRMatrixI(L);
//     JXF_Int *L_J = jxf_CSRMatrixJ(L);
//     JXF_Real *L_data = jxf_CSRMatrixData(L);
//     JXF_Int *U_I = jxf_CSRMatrixI(U);
//     JXF_Int *U_J = jxf_CSRMatrixJ(U);
//     JXF_Real *U_data = jxf_CSRMatrixData(U);
//     JXF_Int n = jxf_CSRMatrixNumRows(L);

//     // 检查矩阵行数一致性
//     if (jxf_CSRMatrixNumRows(L) != jxf_CSRMatrixNumRows(U))
//     {
//         *jlevL = *ilevL = *jlevU = *ilevU = NULL;
//         *nlevL = *nlevU = 0;
//         return;
//     }
//     // 分配内存
//     JXF_Int *level = (JXF_Int *)calloc(n, sizeof(JXF_Int));
//     *jlevL = (JXF_Int *)calloc(n, sizeof(JXF_Int));
//     *ilevL = (JXF_Int *)calloc(n + 1, sizeof(JXF_Int));
//     *jlevU = (JXF_Int *)calloc(n, sizeof(JXF_Int));
//     *ilevU = (JXF_Int *)calloc(n + 1, sizeof(JXF_Int));

//     JXF_Int i, j, k, l;
//     JXF_Int local_nlevL = 0, local_nlevU = 0;

//     // for (i = 0; i <= n; i++) printf("%d ", L_I[i]);
//     // printf("\n");
//     // for (i = 0; i < L_I[n]; i++) printf("%d ", L_J[i]);
//     // printf("\n");

//     // for (i = 0; i <= n; i++) printf("%d ", U_I[i]);
//     // printf("\n");
//     // for (i = 0; i < U_I[n]; i++) printf("%d ", U_J[i]);
//     // printf("\n");

//     for (i = 0; i < n; i++)
//         level[i] = 0;

//     // 初始化 ilevL
//     for (i = 0; i <= n; i++)
//         (*ilevL)[i] = 0;
//     for (i = 0; i <= n; i++)
//         (*ilevU)[i] = 0;
//     for (i = 0; i < n; i++)
//         (*jlevU)[i] = 0;
//     for (i = 0; i < n; i++)
//         (*jlevL)[i] = 0;

//     // ---------- Step 1: 构造下三角矩阵 L 的层次 ----------
//     for (i = 0; i < n; i++)
//     {
//         l = 0;
//         for (j = L_I[i]; j < L_I[i + 1]; j++)
//         {
//             if (L_J[j] < i)
//             {
//                 l = (l > level[L_J[j]]) ? l : level[L_J[j]];
//             }
//         }
//         level[i] = l + 1;
//         (*ilevL)[l + 1]++;
//         // printf("%d",l + 1);
//         local_nlevL = (local_nlevL > (l + 1)) ? local_nlevL : (l + 1);
//     }
//     // printf("\n");

//     for (i = 1; i <= local_nlevL; i++)
//         (*ilevL)[i] += (*ilevL)[i - 1];

//     for (i = 0; i < n; i++)
//     {
//         k = (*ilevL)[level[i] - 1];
//         (*jlevL)[k] = i;
//         (*ilevL)[level[i] - 1]++;
//     }

//     for (i = local_nlevL - 1; i > 0; i--)
//         (*ilevL)[i] = (*ilevL)[i - 1];
//     (*ilevL)[0] = 0;

//     *nlevL = local_nlevL + 1; // 保持原始代码的 +1 逻辑

//     // printf("Lower Triangular Matrix L:\n");
//     // printf("Number of levels (nlevL): %d\n", *nlevL);
//     // printf("jlevL (row indices per level): ");
//     // for (i = 0; i < n; i++) printf("%d ", (*jlevL)[i]);
//     // printf("\n");
//     // printf("ilevL (level offsets): ");
//     // for (i = 0; i <= local_nlevL; i++) printf("%d ", (*ilevL)[i]);
//     // printf("\n\n");

//     // ---------- Step 2: 构造上三角矩阵 U 的层次 ----------
//     for (i = 0; i < n; i++)
//         level[i] = 0; // 重新清零 level
//     for (i = 0; i <= n; i++)
//         (*ilevU)[i] = 0; // 初始化 ilevU

//     for (i = n - 1; i >= 0; i--)
//     {
//         l = 0;
//         for (j = U_I[i]; j < U_I[i + 1]; j++)
//         {
//             if (U_J[j] > i)
//             {
//                 l = (l > level[U_J[j]]) ? l : level[U_J[j]];
//             }
//         }
//         level[i] = l + 1; // 最大层+1
//         (*ilevU)[l + 1]++;
//         // printf("%d",l + 1);
//         local_nlevU = (local_nlevU > (l + 1)) ? local_nlevU : (l + 1);
//     }
//     // printf("\n");
//     for (i = 1; i <= local_nlevU; i++)
//         (*ilevU)[i] += (*ilevU)[i - 1];

//     for (i = n - 1; i >= 0; i--)
//     {
//         k = (*ilevU)[level[i] - 1];
//         (*jlevU)[k] = i;
//         (*ilevU)[level[i] - 1]++;
//     }

//     for (i = local_nlevU - 1; i > 0; i--)
//         (*ilevU)[i] = (*ilevU)[i - 1];
//     (*ilevU)[0] = 0;

//     *nlevU = local_nlevU + 1;

//     //  printf("Upper Triangular Matrix U:\n");
//     //  printf("Number of levels (nlevU): %d\n", *nlevU);
//     //  printf("jlevU (row indices per level): ");
//     //  for (i = 0; i < n; i++) printf("%d ", (*jlevU)[i]);
//     //  printf("\n");
//     //  printf("ilevU (level offsets): ");
//     //  for (i = 0; i <= local_nlevU; i++) printf("%d ", (*ilevU)[i]);
//     //  printf("\n");

//     // ---------- Step 3: 根据 jlevL 构造 permL / inv_permL ----------
//     JXF_Int *permL_local = (JXF_Int *)malloc(n * sizeof(JXF_Int));
//     JXF_Int *inv_permL_local = (JXF_Int *)malloc(n * sizeof(JXF_Int));

//     for (JXF_Int new_i = 0; new_i < n; new_i++)
//     {
//         JXF_Int old_i = (*jlevL)[new_i]; // jlevL already holds old row indices in level order
//         permL_local[old_i] = new_i;     // old -> new
//         inv_permL_local[new_i] = old_i; // new -> old
//     }

//     // ---------- Step 4: 重排 L（只重排行） ----------
//     JXF_Int *row_nnz_L = (JXF_Int *)calloc(n, sizeof(JXF_Int));

//     for (JXF_Int ii = 0; ii < n; ii++)
//     {
//         JXF_Int old_i = inv_permL_local[ii];
//         row_nnz_L[ii] = L_I[old_i + 1] - L_I[old_i];
//     }

//     JXF_Int *new_L_I = (JXF_Int *)malloc((n + 1) * sizeof(JXF_Int));

//     new_L_I[0] = 0;
//     for (JXF_Int ii = 1; ii <= n; ii++)
//         new_L_I[ii] = new_L_I[ii - 1] + row_nnz_L[ii - 1];

//     JXF_Int nnzL = new_L_I[n];
//     JXF_Int *new_L_J = (JXF_Int *)malloc(nnzL * sizeof(JXF_Int));
//     JXF_Real *new_L_data = (JXF_Real *)malloc(nnzL * sizeof(JXF_Real));
//     JXF_Int *posL = (JXF_Int *)calloc(n, sizeof(JXF_Int));

//     for (JXF_Int ii = 0; ii < n; ii++)
//     {
//         JXF_Int old_i = inv_permL_local[ii];
//         for (JXF_Int jj = L_I[old_i]; jj < L_I[old_i + 1]; jj++)
//         {
//             JXF_Int insert_pos = new_L_I[ii] + posL[ii];
//             new_L_J[insert_pos] = L_J[jj]; // 列保持旧编号
//             new_L_data[insert_pos] = L_data[jj];
//             posL[ii]++;
//         }
//     }

//     // 替换 L 的 CSR 指针（注意：请确认你的 API 是否允许直接赋值）
//     free(L_I);
//     free(L_J);
//     free(L_data);
//     jxf_CSRMatrixI(L) = new_L_I;
//     jxf_CSRMatrixJ(L) = new_L_J;
//     jxf_CSRMatrixData(L) = new_L_data;

//     // ---------- 输出 permL / inv_permL ----------
//     *permL = permL_local;
//     *inv_permL = inv_permL_local;

//     // 清理 L 临时内存
//     free(row_nnz_L);
//     free(posL);

//     // ---------- 对 U 做同样的处理：构造 permU/inv_permU 并重排 U 行 ----------
//     JXF_Int *permU_local = (JXF_Int *)malloc(n * sizeof(JXF_Int));
//     JXF_Int *inv_permU_local = (JXF_Int *)malloc(n * sizeof(JXF_Int));

//     for (JXF_Int new_i = 0; new_i < n; new_i++)
//     {
//         JXF_Int old_i = (*jlevU)[new_i];
//         permU_local[old_i] = new_i;
//         inv_permU_local[new_i] = old_i;
//     }

//     JXF_Int *row_nnz_U = (JXF_Int *)calloc(n, sizeof(JXF_Int));
//     for (JXF_Int ii = 0; ii < n; ii++)
//     {
//         JXF_Int old_i = inv_permU_local[ii];
//         row_nnz_U[ii] = U_I[old_i + 1] - U_I[old_i];
//     }

//     JXF_Int *new_U_I = (JXF_Int *)malloc((n + 1) * sizeof(JXF_Int));

//     new_U_I[0] = 0;
//     for (JXF_Int ii = 1; ii <= n; ii++)
//         new_U_I[ii] = new_U_I[ii - 1] + row_nnz_U[ii - 1];

//     JXF_Int nnzU = new_U_I[n];
//     JXF_Int *new_U_J = (JXF_Int *)malloc(nnzU * sizeof(JXF_Int));
//     JXF_Real *new_U_data = (JXF_Real *)malloc(nnzU * sizeof(JXF_Real));
//     JXF_Int *posU = (JXF_Int *)calloc(n, sizeof(JXF_Int));

//     for (JXF_Int ii = 0; ii < n; ii++)
//     {
//         JXF_Int old_i = inv_permU_local[ii];
//         for (JXF_Int jj = U_I[old_i]; jj < U_I[old_i + 1]; jj++)
//         {
//             JXF_Int insert_pos = new_U_I[ii] + posU[ii];
//             new_U_J[insert_pos] = U_J[jj]; // 列保持旧编号
//             new_U_data[insert_pos] = U_data[jj];
//             posU[ii]++;
//         }
//     }

//     // 替换 U 的 CSR 指针
//     free(U_I);
//     free(U_J);
//     free(U_data);
//     jxf_CSRMatrixI(U) = new_U_I;
//     jxf_CSRMatrixJ(U) = new_U_J;
//     jxf_CSRMatrixData(U) = new_U_data;

//     // 输出 permU / inv_permU
//     *permU = permU_local;
//     *inv_permU = inv_permU_local;

//     // cleanup
//     free(row_nnz_U);
//     free(posU);

//     free(level);
// }

// JXF_Int
// jxf_GreedyColoring(jxf_ParCSRMatrix *A,
//                   JXF_Int          nLU,
//                   JXF_Int         **row_by_color_out,
//                   JXF_Int         **color_starts_out,
//                   JXF_Int          *num_colors_out)
// {
//     jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//     JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//     JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);

//     JXF_Int i, jj, c, neighbor;

//     JXF_Int *color = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));
//     for (i = 0; i < nLU; i++) color[i] = -1;

//     JXF_Int used_colors_size = 16;
//     char *used_colors = (char *)calloc(used_colors_size, sizeof(char));

//     JXF_Int *color_counts = (JXF_Int *)calloc(nLU + 1, sizeof(JXF_Int));

//     JXF_Int num_colors = 0;
//     for (i = 0; i < nLU; i++) {
//         if (num_colors > 0) memset(used_colors, 0, num_colors * sizeof(char));

//         for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++) {
//             neighbor = A_diag_j[jj];
//             if (color[neighbor] >= 0)
//                 used_colors[color[neighbor]] = 1;
//         }

//         c = 0;
//         while (c < num_colors && used_colors[c] == 1) c++;

//         if (c >= num_colors) {
//             if (num_colors >= used_colors_size) {
//                 used_colors_size *= 2;
//                 char *new_used_colors = (char *)realloc(used_colors, used_colors_size * sizeof(char));
//                 if (!new_used_colors) {
//                     fprintf(stderr, "Memory reallocation failed!\n");
//                     exit(1);
//                 }
//                 used_colors = new_used_colors;
//                 memset(used_colors + num_colors, 0, (used_colors_size - num_colors) * sizeof(char));
//             }
//             num_colors++;
//         }

//         color[i] = c;
//         color_counts[c]++;
//     }

//     JXF_Int *row_by_color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *color_starts = (JXF_Int *)malloc((num_colors + 1) * sizeof(JXF_Int));

//     color_starts[0] = 0;
//     for (c = 0; c < num_colors; c++) {
//         color_starts[c + 1] = color_starts[c] + color_counts[c];
//     }

//     // 用 color_pos 作为写位置指针，避免破坏 color_counts 和 color_starts
//     JXF_Int *color_pos = (JXF_Int *)malloc(num_colors * sizeof(JXF_Int));
//     memcpy(color_pos, color_starts, num_colors * sizeof(JXF_Int));

//     for (i = 0; i < nLU; i++) {
//         row_by_color[color_pos[color[i]]++] = i;
//     }

//     *row_by_color_out = row_by_color;
//     *color_starts_out = color_starts;
//     *num_colors_out = num_colors;

//     free(color);
//     free(used_colors);
//     free(color_counts);
//     free(color_pos);

//     printf("Number of levels : %d\n", *num_colors_out);

//     return 0;
// }

// 7.28
// JXF_Int
// jxf_GreedyColoring_L(jxf_ParCSRMatrix *L,
//                     JXF_Int nLU,
//                     JXF_Int **row_by_color_out,
//                     JXF_Int **color_starts_out,
//                     JXF_Int *num_colors_out)
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Int *L_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_j = jxf_CSRMatrixJ(L_diag);

//     JXF_Int *color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int used_colors_size = 16;
//     char *used = (char *)calloc(used_colors_size, sizeof(char));

//     for (JXF_Int i = 0; i < nLU; i++)
//         color[i] = -1;

//     JXF_Int max_color = -1;

//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         // 标记当前行依赖的颜色
//         if (max_color >= 0)
//             memset(used, 0, (max_color + 1) * sizeof(char));

//         for (JXF_Int jj = L_i[i]; jj < L_i[i + 1]; jj++)
//         {
//             JXF_Int col = L_j[jj];
//             if (col < i && color[col] >= 0)
//             {
//                 used[color[col]] = 1;
//             }
//         }

//         // 选一个未使用的最小颜色
//         JXF_Int c = 0;
//         while (c < used_colors_size && used[c])
//             c++;
//         if (c >= used_colors_size)
//         {
//             used_colors_size *= 2;
//             char *new_used = (char *)realloc(used, used_colors_size * sizeof(char));
//             if (!new_used)
//             {
//                 fprintf(stderr, "Memory reallocation failed!\n");
//                 free(color);
//                 free(used);
//                 exit(1);
//             }
//             used = new_used;
//             memset(used + c, 0, (used_colors_size - c) * sizeof(char));
//         }
//         color[i] = c;
//         if (c > max_color)
//             max_color = c;

//         // 清除标记
//         for (JXF_Int jj = L_i[i]; jj < L_i[i + 1]; jj++)
//         {
//             JXF_Int col = L_j[jj];
//             if (col < i && color[col] >= 0)
//             {
//                 used[color[col]] = 0;
//             }
//         }
//     }

//     free(used);
//     *num_colors_out = max_color + 1;

//     // 统计每个颜色的数量
//     JXF_Int *color_count = (JXF_Int *)calloc(*num_colors_out, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//         color_count[color[i]]++;

//     // 构造 color_starts 数组
//     JXF_Int *color_starts = (JXF_Int *)malloc((*num_colors_out + 1) * sizeof(JXF_Int));
//     color_starts[0] = 0;
//     for (JXF_Int c = 1; c <= *num_colors_out; c++)
//         color_starts[c] = color_starts[c - 1] + color_count[c - 1];

//     // 构造 row_by_color 数组
//     JXF_Int *row_by_color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *temp_index = (JXF_Int *)malloc(*num_colors_out * sizeof(JXF_Int));
//     memcpy(temp_index, color_starts, *num_colors_out * sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         JXF_Int c = color[i];
//         row_by_color[temp_index[c]++] = i;
//     }

//     // 输出结果
//     *row_by_color_out = row_by_color;
//     *color_starts_out = color_starts;

//     //  printf("Number of levels : %d\n", *num_colors_out);
//     //  printf("jlevU (row indices per level): ");
//     //  for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*row_by_color_out)[i]);
//     //  printf("\n");
//     //  printf("ilevU (level offsets): ");
//     //  for (JXF_Int i = 0; i <= *num_colors_out; i++) printf("%d ", (*color_starts_out)[i]);
//     //  printf("\n");
//     // 清理
//     free(color_count);
//     free(temp_index);
//     free(color);

//     return 0;
// }

// JXF_Int jxf_GreedyColoring_U(jxf_ParCSRMatrix *U,
//                            JXF_Int nLU,
//                            JXF_Int **row_by_color_out,
//                            JXF_Int **color_starts_out,
//                            JXF_Int *num_colors_out)
// {
//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Int *U_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_j = jxf_CSRMatrixJ(U_diag);

//     JXF_Int *color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int used_colors_size = 16;
//     char *used = (char *)calloc(used_colors_size, sizeof(char));

//     for (JXF_Int i = 0; i < nLU; i++)
//         color[i] = -1;

//     JXF_Int max_color = -1;

//     // 从最后一行开始染色，让最后一行最先染、染成颜色0
//     for (JXF_Int i = nLU - 1; i >= 0; i--)
//     {
//         if (max_color >= 0)
//             memset(used, 0, (max_color + 1) * sizeof(char));

//         // 当前行依赖于行号更小的行（下三角）
//         for (JXF_Int jj = U_i[i]; jj < U_i[i + 1]; jj++)
//         {
//             JXF_Int col = U_j[jj];
//             if (col > i && color[col] >= 0)
//             {
//                 used[color[col]] = 1;
//             }
//         }

//         // 选择未用的最小颜色
//         JXF_Int c = 0;
//         while (c < used_colors_size && used[c])
//             c++;
//         if (c >= used_colors_size)
//         {
//             used_colors_size *= 2;
//             char *new_used = (char *)realloc(used, used_colors_size * sizeof(char));
//             if (!new_used)
//             {
//                 fprintf(stderr, "Memory reallocation failed!\n");
//                 free(color);
//                 free(used);
//                 exit(1);
//             }
//             used = new_used;
//             memset(used + c, 0, (used_colors_size - c) * sizeof(char));
//         }

//         color[i] = c;
//         if (c > max_color)
//             max_color = c;

//         // 清除标记
//         for (JXF_Int jj = U_i[i]; jj < U_i[i + 1]; jj++)
//         {
//             JXF_Int col = U_j[jj];
//             if (col < i && color[col] >= 0)
//             {
//                 used[color[col]] = 0;
//             }
//         }
//     }

//     free(used);
//     *num_colors_out = max_color + 1;

//     // 统计每个颜色的数量
//     JXF_Int *color_count = (JXF_Int *)calloc(*num_colors_out, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//         color_count[color[i]]++;

//     // 构造 color_starts 数组
//     JXF_Int *color_starts = (JXF_Int *)malloc((*num_colors_out + 1) * sizeof(JXF_Int));
//     color_starts[0] = 0;
//     for (JXF_Int c = 1; c <= *num_colors_out; c++)
//         color_starts[c] = color_starts[c - 1] + color_count[c - 1];

//     // 构造 row_by_color 数组，并保证每个颜色的行是按从大到小顺序排列（倒着写入）
//     JXF_Int *row_by_color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *temp_index = (JXF_Int *)malloc(*num_colors_out * sizeof(JXF_Int));
//     memcpy(temp_index, color_starts, *num_colors_out * sizeof(JXF_Int));

//     for (JXF_Int i = nLU - 1; i >= 0; i--)
//     {
//         JXF_Int c = color[i];
//         row_by_color[temp_index[c]++] = i;
//     }

//     // 输出结果
//     *row_by_color_out = row_by_color;
//     *color_starts_out = color_starts;

//     // 打印调试信息
//     //  printf("Number of levels : %d\n", *num_colors_out);
//     //  printf("jlevU (row indices per level): ");
//     //  for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*row_by_color_out)[i]);
//     //  printf("\n");
//     //  printf("ilevU (level offsets): ");
//     //  for (JXF_Int i = 0; i <= *num_colors_out; i++) printf("%d ", (*color_starts_out)[i]);
//     //  printf("\n");

//     // 清理
//     free(color_count);
//     free(temp_index);
//     free(color);

//     return 0;
// }

// JXF_Int
// jxf_GreedyColoring_L1(jxf_ParCSRMatrix *L,
//                   JXF_Int           nLU,
//                   JXF_Int         **row_by_color_out,
//                   JXF_Int         **color_starts_out,
//                   JXF_Int          *num_colors_out,
//                   JXF_Int         **perm_out,
//                   JXF_Int         **inv_perm_out)
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Int *L_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_j = jxf_CSRMatrixJ(L_diag);
//     JXF_Real *L_data = jxf_CSRMatrixData(L_diag);

//     JXF_Int *color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int used_colors_size = 16;
//     char *used = (char *)calloc(used_colors_size, sizeof(char));

//     for (JXF_Int i = 0; i < nLU; i++) color[i] = -1;

//     JXF_Int max_color = -1;

//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         // 标记当前行依赖的颜色
//         if (max_color >= 0) memset(used, 0, (max_color + 1) * sizeof(char));

//         for (JXF_Int jj = L_i[i]; jj < L_i[i+1]; jj++)
//         {
//             JXF_Int col = L_j[jj];
//             if (col < i && color[col] >= 0)
//             {
//                 used[color[col]] = 1;
//             }
//         }

//         // 选一个未使用的最小颜色
//         JXF_Int c = 0;
//         while (c < used_colors_size && used[c]) c++;
//         if (c >= used_colors_size) {
//             used_colors_size *= 2;
//             char *new_used = (char *)realloc(used, used_colors_size * sizeof(char));
//             if (!new_used) {
//                 fprintf(stderr, "Memory reallocation failed!\n");
//                 free(color);
//                 free(used);
//                 exit(1);
//             }
//             used = new_used;
//             memset(used + c, 0, (used_colors_size - c) * sizeof(char));
//         }
//         color[i] = c;
//         if (c > max_color) max_color = c;

//         // 清除标记
//         for (JXF_Int jj = L_i[i]; jj < L_i[i+1]; jj++)
//         {
//             JXF_Int col = L_j[jj];
//             if (col < i && color[col] >= 0)
//             {
//                 used[color[col]] = 0;
//             }
//         }
//     }

//     free(used);
//     *num_colors_out = max_color + 1;

//     // 统计每个颜色的数量
//     JXF_Int *color_count = (JXF_Int *)calloc(*num_colors_out, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//         color_count[color[i]]++;

//     // 构造 color_starts 数组
//     JXF_Int *color_starts = (JXF_Int *)malloc((*num_colors_out + 1) * sizeof(JXF_Int));
//     color_starts[0] = 0;
//     for (JXF_Int c = 1; c <= *num_colors_out; c++)
//         color_starts[c] = color_starts[c - 1] + color_count[c - 1];

//     // 构造 row_by_color 数组
//     JXF_Int *row_by_color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *temp_index = (JXF_Int *)malloc(*num_colors_out * sizeof(JXF_Int));
//     memcpy(temp_index, color_starts, *num_colors_out * sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         JXF_Int c = color[i];
//         row_by_color[temp_index[c]++] = i;
//     }

//     // === 构造 perm / inv_perm ===
//     JXF_Int *perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *inv_perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     for (JXF_Int new_i = 0; new_i < nLU; new_i++) {
//         JXF_Int old_i = row_by_color[new_i];
//         perm[old_i] = new_i;
//         inv_perm[new_i] = old_i;
//     }

//     // === 重排 L_diag 到新顺序 ===
//     JXF_Int *new_i = (JXF_Int *)malloc((nLU + 1) * sizeof(JXF_Int));
//     JXF_Int *row_nnz = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++) {
//         JXF_Int old_i = inv_perm[i];
//         row_nnz[i] = L_i[old_i + 1] - L_i[old_i];
//     }

//     new_i[0] = 0;
//     for (JXF_Int i = 1; i <= nLU; i++)
//         new_i[i] = new_i[i - 1] + row_nnz[i - 1];

//     JXF_Int nnz = new_i[nLU];
//     JXF_Int *new_j = (JXF_Int *)malloc(nnz * sizeof(JXF_Int));
//     JXF_Real *new_data = (JXF_Real *)malloc(nnz * sizeof(JXF_Real));
//     JXF_Int *pos = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++) {
//         JXF_Int old_i = inv_perm[i];
//         JXF_Int row_start = L_i[old_i];
//         for (JXF_Int jj = L_i[old_i]; jj < L_i[old_i + 1]; jj++) {
//             JXF_Int insert_pos = new_i[i] + pos[i];
//             new_j[insert_pos] = L_j[jj];
//             new_data[insert_pos] = L_data[jj];
//             pos[i]++;
//         }
//     }

//     // 替换原有 L_diag 内容
//     free(L_i);
//     free(L_j);
//     free(L_data);

//    jxf_CSRMatrixData(L_diag ) = new_data;
//    jxf_CSRMatrixI(L_diag ) = new_i;
//    jxf_CSRMatrixJ(L_diag ) = new_j;

//    // === 生成 jlevL_perm（即 perm[row_by_color]） ===
//     JXF_Int *row_by_color_perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         row_by_color_perm[i] = perm[row_by_color[i]];
//     }

//     // 输出结果
//     *row_by_color_out = row_by_color_perm;
//     *color_starts_out = color_starts;
//     *perm_out = perm;
//     *inv_perm_out = inv_perm;

//      for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*perm_out)[i]);
//     printf("\n");
//     printf("inv_perm_out: ");
//     for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*inv_perm_out)[i]);
//     printf("\n");

//     printf("Number of levels : %d\n", *num_colors_out);
//     printf("jlevU (row indices per level): ");
//     for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*row_by_color_out)[i]);
//     printf("\n");
//     printf("ilevU (level offsets): ");
//     for (JXF_Int i = 0; i <= *num_colors_out; i++) printf("%d ", (*color_starts_out)[i]);
//     printf("\n");

//     // 清理
//     free(color_count);
//     free(temp_index);
//     free(color);
//    free(row_nnz);
//     free(pos);

//     return 0;
// }

// JXF_Int
// jxf_GreedyColoring_L1(jxf_ParCSRMatrix *L,
//                      JXF_Int nLU,
//                      JXF_Int **row_by_color_out, // 按颜色分组的行号（旧编号）
//                      JXF_Int **color_starts_out, // 每个颜色块的起始位置
//                      JXF_Int *num_colors_out,    // 颜色数
//                      JXF_Int **perm_out,         // old → new 行编号
//                      JXF_Int **inv_perm_out)     // new → old 行编号
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Int *L_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_j = jxf_CSRMatrixJ(L_diag);
//     JXF_Real *L_data = jxf_CSRMatrixData(L_diag);

//     // === Step 1: greedy coloring ===
//     JXF_Int *color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int used_colors_size = 16;
//     char *used = (char *)calloc(used_colors_size, sizeof(char));

//     for (JXF_Int i = 0; i < nLU; i++)
//         color[i] = -1;
//     JXF_Int max_color = -1;

//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         // reset used[] for current row
//         if (max_color >= 0)
//             memset(used, 0, (max_color + 1) * sizeof(char));

//         // 标记依赖的已用颜色 (只看下三角 col<i)
//         for (JXF_Int jj = L_i[i]; jj < L_i[i + 1]; jj++)
//         {
//             JXF_Int col = L_j[jj];
//             if (col < i && color[col] >= 0)
//                 used[color[col]] = 1;
//         }

//         // 选最小未使用颜色
//         JXF_Int c = 0;
//         while (c < used_colors_size && used[c])
//             c++;
//         if (c >= used_colors_size)
//         {
//             used_colors_size *= 2;
//             used = (char *)realloc(used, used_colors_size * sizeof(char));
//             memset(used + c, 0, (used_colors_size - c) * sizeof(char));
//         }
//         color[i] = c;
//         if (c > max_color)
//             max_color = c;
//     }

//     free(used);
//     *num_colors_out = max_color + 1;

//     // === Step 2: 构造 row_by_color / color_starts ===
//     JXF_Int *color_count = (JXF_Int *)calloc(*num_colors_out, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//         color_count[color[i]]++;

//     JXF_Int *color_starts = (JXF_Int *)malloc((*num_colors_out + 1) * sizeof(JXF_Int));
//     color_starts[0] = 0;
//     for (JXF_Int c = 1; c <= *num_colors_out; c++)
//         color_starts[c] = color_starts[c - 1] + color_count[c - 1];

//     JXF_Int *row_by_color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *temp_index = (JXF_Int *)malloc(*num_colors_out * sizeof(JXF_Int));
//     memcpy(temp_index, color_starts, *num_colors_out * sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         JXF_Int c = color[i];
//         row_by_color[temp_index[c]++] = i; // 保存的是旧行号
//     }

//     // === Step 3: 构造 perm / inv_perm ===
//     JXF_Int *perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *inv_perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     for (JXF_Int new_i = 0; new_i < nLU; new_i++)
//     {
//         JXF_Int old_i = row_by_color[new_i];
//         perm[old_i] = new_i;
//         inv_perm[new_i] = old_i;
//     }

//     // === Step 4: 按 new 行顺序重排 L (只重排行，不重排列) ===
//     JXF_Int *row_nnz = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         JXF_Int old_i = inv_perm[i];
//         row_nnz[i] = L_i[old_i + 1] - L_i[old_i];
//     }

//     JXF_Int *new_i = (JXF_Int *)malloc((nLU + 1) * sizeof(JXF_Int));
//     new_i[0] = 0;
//     for (JXF_Int i = 1; i <= nLU; i++)
//         new_i[i] = new_i[i - 1] + row_nnz[i - 1];

//     JXF_Int nnz = new_i[nLU];
//     JXF_Int *new_j = (JXF_Int *)malloc(nnz * sizeof(JXF_Int));
//     JXF_Real *new_data = (JXF_Real *)malloc(nnz * sizeof(JXF_Real));
//     JXF_Int *pos = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         JXF_Int old_i = inv_perm[i];
//         for (JXF_Int jj = L_i[old_i]; jj < L_i[old_i + 1]; jj++)
//         {
//             JXF_Int insert_pos = new_i[i] + pos[i];
//             new_j[insert_pos] = L_j[jj]; // 注意：列号保持旧编号！
//             new_data[insert_pos] = L_data[jj];
//             pos[i]++;
//         }
//     }

//     // 替换 L_diag 内容（这里假设 L_diag 拥有数据内存）
//     free(L_i);
//     free(L_j);
//     free(L_data);
//     jxf_CSRMatrixData(L_diag) = new_data;
//     jxf_CSRMatrixI(L_diag) = new_i;
//     jxf_CSRMatrixJ(L_diag) = new_j;

//     // === 输出结果 ===
//     *row_by_color_out = row_by_color; // 旧行号，按颜色分组顺序
//     *color_starts_out = color_starts; // 每个颜色块的起始偏移
//     *perm_out = perm;                 // old → new
//     *inv_perm_out = inv_perm;         // new → old

//     // // === Debug 打印 ===
//     // printf("perm (old->new): ");
//     // for (JXF_Int i = 0; i < nLU; i++) printf("%d ", perm[i]);
//     // printf("\n");

//     // printf("inv_perm (new->old): ");
//     // for (JXF_Int i = 0; i < nLU; i++) printf("%d ", inv_perm[i]);
//     // printf("\n");

//     // printf("Number of colors: %d\n", *num_colors_out);
//     // printf("row_by_color (old row indices per color): ");
//     // for (JXF_Int i = 0; i < nLU; i++) printf("%d ", row_by_color[i]);
//     // printf("\n");

//     // printf("color_starts (offsets): ");
//     // for (JXF_Int i = 0; i <= *num_colors_out; i++) printf("%d ", color_starts[i]);
//     // printf("\n");

//     // 清理
//     free(color_count);
//     free(temp_index);
//     free(color);
//     free(row_nnz);
//     free(pos);

//     return 0;
// }

// JXF_Int jxf_GreedyColoring_U1(jxf_ParCSRMatrix *U,
//                             JXF_Int nLU,
//                             JXF_Int **row_by_color_out,
//                             JXF_Int **color_starts_out,
//                             JXF_Int *num_colors_out,
//                             JXF_Int **perm_out,
//                             JXF_Int **inv_perm_out)
// {
//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Int *U_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_j = jxf_CSRMatrixJ(U_diag);
//     JXF_Real *U_data = jxf_CSRMatrixData(U_diag);

//     JXF_Int *color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int used_colors_size = 16;
//     char *used = (char *)calloc(used_colors_size, sizeof(char));

//     for (JXF_Int i = 0; i < nLU; i++)
//         color[i] = -1;

//     JXF_Int max_color = -1;

//     // 从最后一行开始染色，让最后一行最先染、染成颜色0
//     for (JXF_Int i = nLU - 1; i >= 0; i--)
//     {
//         if (max_color >= 0)
//             memset(used, 0, (max_color + 1) * sizeof(char));

//         // 当前行依赖于行号更小的行（下三角）
//         for (JXF_Int jj = U_i[i]; jj < U_i[i + 1]; jj++)
//         {
//             JXF_Int col = U_j[jj];
//             if (col > i && color[col] >= 0)
//             {
//                 used[color[col]] = 1;
//             }
//         }

//         // 选择未用的最小颜色
//         JXF_Int c = 0;
//         while (c < used_colors_size && used[c])
//             c++;
//         if (c >= used_colors_size)
//         {
//             used_colors_size *= 2;
//             char *new_used = (char *)realloc(used, used_colors_size * sizeof(char));
//             if (!new_used)
//             {
//                 fprintf(stderr, "Memory reallocation failed!\n");
//                 free(color);
//                 free(used);
//                 exit(1);
//             }
//             used = new_used;
//             memset(used + c, 0, (used_colors_size - c) * sizeof(char));
//         }

//         color[i] = c;
//         if (c > max_color)
//             max_color = c;

//         // 清除标记
//         for (JXF_Int jj = U_i[i]; jj < U_i[i + 1]; jj++)
//         {
//             JXF_Int col = U_j[jj];
//             if (col < i && color[col] >= 0)
//             {
//                 used[color[col]] = 0;
//             }
//         }
//     }

//     free(used);

//     *num_colors_out = max_color + 1;

//     // 统计每个颜色的数量
//     JXF_Int *color_count = (JXF_Int *)calloc(*num_colors_out, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//         color_count[color[i]]++;

//     // 构造 color_starts 数组
//     JXF_Int *color_starts = (JXF_Int *)malloc((*num_colors_out + 1) * sizeof(JXF_Int));
//     color_starts[0] = 0;
//     for (JXF_Int c = 1; c <= *num_colors_out; c++)
//         color_starts[c] = color_starts[c - 1] + color_count[c - 1];

//     // 构造 row_by_color 数组，并保证每个颜色的行是按从大到小顺序排列（倒着写入）
//     JXF_Int *row_by_color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *temp_index = (JXF_Int *)malloc(*num_colors_out * sizeof(JXF_Int));
//     memcpy(temp_index, color_starts, *num_colors_out * sizeof(JXF_Int));

//     for (JXF_Int i = nLU - 1; i >= 0; i--)
//     {
//         JXF_Int c = color[i];
//         row_by_color[temp_index[c]++] = i;
//     }

//     // === 构造 perm / inv_perm ===
//     JXF_Int *perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *inv_perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));

//     for (JXF_Int new_i = 0; new_i < nLU; new_i++)
//     {
//         JXF_Int old_i = row_by_color[new_i];
//         perm[old_i] = new_i;
//         inv_perm[new_i] = old_i;
//     }

//     // === 重排 U_diag 的行指针与数据 ===
//     // 根据新的行顺序，重新排列 U 的行，但列索引保持不变。
//     JXF_Int *new_i = (JXF_Int *)malloc((nLU + 1) * sizeof(JXF_Int));
//     JXF_Int *row_nnz = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));

//     // 计算每一行的非零元数量（按旧顺序）
//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         JXF_Int old_i = inv_perm[i];
//         row_nnz[i] = U_i[old_i + 1] - U_i[old_i];
//     }

//     // 构造新的 row pointer 数组 new_i
//     new_i[0] = 0;
//     for (JXF_Int i = 1; i <= nLU; i++)
//     {
//         new_i[i] = new_i[i - 1] + row_nnz[i - 1];
//     }

//     // 分配新存储空间
//     JXF_Int nnz = new_i[nLU];
//     JXF_Int *new_j = (JXF_Int *)malloc(nnz * sizeof(JXF_Int));
//     JXF_Real *new_data = (JXF_Real *)malloc(nnz * sizeof(JXF_Real));
//     JXF_Int *pos = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));

//     // 按新行顺序搬运旧数据
//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         JXF_Int old_i = inv_perm[i];
//         for (JXF_Int jj = U_i[old_i]; jj < U_i[old_i + 1]; jj++)
//         {
//             JXF_Int insert_pos = new_i[i] + pos[i];
//             new_j[insert_pos] = U_j[jj];       // 列索引保持不变
//             new_data[insert_pos] = U_data[jj]; // 数值保持不变
//             pos[i]++;
//         }
//     }

//     // 释放旧存储
//     free(U_i);
//     free(U_j);
//     free(U_data);

//     // 更新 U_diag 到新存储
//     jxf_CSRMatrixI(U_diag) = new_i;
//     jxf_CSRMatrixJ(U_diag) = new_j;
//     jxf_CSRMatrixData(U_diag) = new_data;

//     // 输出结果
//     *row_by_color_out = row_by_color;
//     *color_starts_out = color_starts;
//     *perm_out = perm;
//     *inv_perm_out = inv_perm;

//     // for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*perm_out)[i]);
//     // printf("\n");
//     // printf("inv_perm_out: ");
//     // for (JXF_Int i = 0; i <= nLU; i++) printf("%d ", (*inv_perm_out)[i]);
//     // printf("\n");

//     // // 打印调试信息
//     // printf("Number of levels : %d\n", *num_colors_out);
//     // printf("jlevU (row indices per level): ");
//     // for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*row_by_color_out)[i]);
//     // printf("\n");
//     // printf("ilevU (level offsets): ");
//     // for (JXF_Int i = 0; i <= *num_colors_out; i++) printf("%d ", (*color_starts_out)[i]);
//     // printf("\n");

//     // 清理
//     free(color_count);
//     free(temp_index);
//     free(color);
//     free(row_nnz);
//     free(pos);

//     return 0;
// }

// JXF_Int
// jxf_GreedyColoring_L1(jxf_ParCSRMatrix *L,
//                   JXF_Int           nLU,
//                   JXF_Int         **row_by_color_out,
//                   JXF_Int         **color_starts_out,
//                   JXF_Int          *num_colors_out,
//                   JXF_Int         **perm_out,
//                   JXF_Int         **inv_perm_out)
// {
//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     JXF_Int *L_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_j = jxf_CSRMatrixJ(L_diag);
//     JXF_Real *L_data = jxf_CSRMatrixData(L_diag);

//     JXF_Int *color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int used_colors_size = 16;
//     char *used = (char *)calloc(used_colors_size, sizeof(char));

//     for (JXF_Int i = 0; i < nLU; i++) color[i] = -1;

//     JXF_Int max_color = -1;

//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         // 标记当前行依赖的颜色
//         if (max_color >= 0) memset(used, 0, (max_color + 1) * sizeof(char));

//         for (JXF_Int jj = L_i[i]; jj < L_i[i+1]; jj++)
//         {
//             JXF_Int col = L_j[jj];
//             if (col < i && color[col] >= 0)
//             {
//                 used[color[col]] = 1;
//             }
//         }

//         // 选一个未使用的最小颜色
//         JXF_Int c = 0;
//         while (c < used_colors_size && used[c]) c++;
//         if (c >= used_colors_size) {
//             used_colors_size *= 2;
//             char *new_used = (char *)realloc(used, used_colors_size * sizeof(char));
//             if (!new_used) {
//                 fprintf(stderr, "Memory reallocation failed!\n");
//                 free(color);
//                 free(used);
//                 exit(1);
//             }
//             used = new_used;
//             memset(used + c, 0, (used_colors_size - c) * sizeof(char));
//         }
//         color[i] = c;
//         if (c > max_color) max_color = c;

//         // 清除标记
//         for (JXF_Int jj = L_i[i]; jj < L_i[i+1]; jj++)
//         {
//             JXF_Int col = L_j[jj];
//             if (col < i && color[col] >= 0)
//             {
//                 used[color[col]] = 0;
//             }
//         }
//     }

//     free(used);
//     *num_colors_out = max_color + 1;

//     // 统计每个颜色的数量
//     JXF_Int *color_count = (JXF_Int *)calloc(*num_colors_out, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//         color_count[color[i]]++;

//     // 构造 color_starts 数组
//     JXF_Int *color_starts = (JXF_Int *)malloc((*num_colors_out + 1) * sizeof(JXF_Int));
//     color_starts[0] = 0;
//     for (JXF_Int c = 1; c <= *num_colors_out; c++)
//         color_starts[c] = color_starts[c - 1] + color_count[c - 1];

//     // 构造 row_by_color 数组
//     JXF_Int *row_by_color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *temp_index = (JXF_Int *)malloc(*num_colors_out * sizeof(JXF_Int));
//     memcpy(temp_index, color_starts, *num_colors_out * sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         JXF_Int c = color[i];
//         row_by_color[temp_index[c]++] = i;
//     }

//     // === 构造 perm / inv_perm ===
//     JXF_Int *perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *inv_perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     for (JXF_Int new_i = 0; new_i < nLU; new_i++) {
//         JXF_Int old_i = row_by_color[new_i];
//         perm[old_i] = new_i;
//         inv_perm[new_i] = old_i;
//     }

//     // === 重排 L_diag 到新顺序 ===
//     JXF_Int *new_i = (JXF_Int *)malloc((nLU + 1) * sizeof(JXF_Int));
//     JXF_Int *row_nnz = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++) {
//         JXF_Int old_i = inv_perm[i];
//         row_nnz[i] = L_i[old_i + 1] - L_i[old_i];
//     }

//     new_i[0] = 0;
//     for (JXF_Int i = 1; i <= nLU; i++)
//         new_i[i] = new_i[i - 1] + row_nnz[i - 1];

//     JXF_Int nnz = new_i[nLU];
//     JXF_Int *new_j = (JXF_Int *)malloc(nnz * sizeof(JXF_Int));
//     JXF_Real *new_data = (JXF_Real *)malloc(nnz * sizeof(JXF_Real));
//     JXF_Int *pos = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++) {
//         JXF_Int old_i = inv_perm[i];
//         JXF_Int row_start = L_i[old_i];
//         for (JXF_Int jj = L_i[old_i]; jj < L_i[old_i + 1]; jj++) {
//             JXF_Int old_j = L_j[jj];
//             JXF_Int new_j_idx = perm[old_j];

//             JXF_Int insert_pos = new_i[i] + pos[i];
//             new_j[insert_pos] = new_j_idx;
//             new_data[insert_pos] = L_data[jj];
//             pos[i]++;
//         }
//     }

//     // 替换原有 L_diag 内容
//     free(L_i);
//     free(L_j);
//     free(L_data);

//    jxf_CSRMatrixData(L_diag ) = new_data;
//    jxf_CSRMatrixI(L_diag ) = new_i;
//    jxf_CSRMatrixJ(L_diag ) = new_j;

//    // === 生成 jlevL_perm（即 perm[row_by_color]） ===
//     JXF_Int *row_by_color_perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         row_by_color_perm[i] = perm[row_by_color[i]];
//     }

//     // 输出结果
//     *row_by_color_out = row_by_color_perm;
//     *color_starts_out = color_starts;
//     *perm_out = perm;
//     *inv_perm_out = inv_perm;

//     //  for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*perm_out)[i]);
//     // printf("\n");
//     // printf("inv_perm_out: ");
//     // for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*inv_perm_out)[i]);
//     // printf("\n");

//     // printf("Number of levels : %d\n", *num_colors_out);
//     // printf("jlevU (row indices per level): ");
//     // for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*row_by_color_out)[i]);
//     // printf("\n");
//     // printf("ilevU (level offsets): ");
//     // for (JXF_Int i = 0; i <= *num_colors_out; i++) printf("%d ", (*color_starts_out)[i]);
//     // printf("\n");

//     // 清理
//     free(color_count);
//     free(temp_index);
//     free(color);
//    free(row_nnz);
//     free(pos);

//     return 0;
// }

// JXF_Int jxf_GreedyColoring_U1(jxf_ParCSRMatrix *U,
//                                  JXF_Int           nLU,
//                                  JXF_Int         **row_by_color_out,
//                                  JXF_Int         **color_starts_out,
//                                  JXF_Int          *num_colors_out,
//                                  JXF_Int         **perm_out,
//                                  JXF_Int         **inv_perm_out)
// {
//     jxf_CSRMatrix *U_diag = jxf_ParCSRMatrixDiag(U);
//     JXF_Int *U_i = jxf_CSRMatrixI(U_diag);
//     JXF_Int *U_j = jxf_CSRMatrixJ(U_diag);
//     JXF_Real *U_data = jxf_CSRMatrixData(U_diag);

//     JXF_Int *color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int used_colors_size = 16;
//     char *used = (char *)calloc(used_colors_size, sizeof(char));

//     for (JXF_Int i = 0; i < nLU; i++) color[i] = -1;

//     JXF_Int max_color = -1;

//     // 从最后一行开始染色，让最后一行最先染、染成颜色0
//     for (JXF_Int i = nLU - 1; i >= 0; i--)
//     {
//         if (max_color >= 0)
//             memset(used, 0, (max_color + 1) * sizeof(char));

//         // 当前行依赖于行号更小的行（下三角）
//         for (JXF_Int jj = U_i[i]; jj < U_i[i+1]; jj++)
//         {
//             JXF_Int col = U_j[jj];
//             if (col > i && color[col] >= 0)
//             {
//                 used[color[col]] = 1;
//             }
//         }

//         // 选择未用的最小颜色
//         JXF_Int c = 0;
//         while (c < used_colors_size && used[c]) c++;
//         if (c >= used_colors_size)
//         {
//             used_colors_size *= 2;
//             char *new_used = (char *)realloc(used, used_colors_size * sizeof(char));
//             if (!new_used)
//             {
//                 fprintf(stderr, "Memory reallocation failed!\n");
//                 free(color);
//                 free(used);
//                 exit(1);
//             }
//             used = new_used;
//             memset(used + c, 0, (used_colors_size - c) * sizeof(char));
//         }

//         color[i] = c;
//         if (c > max_color) max_color = c;

//         // 清除标记
//         for (JXF_Int jj = U_i[i]; jj < U_i[i+1]; jj++)
//         {
//             JXF_Int col = U_j[jj];
//             if (col < i && color[col] >= 0)
//             {
//                 used[color[col]] = 0;
//             }
//         }
//     }

//     free(used);

//     *num_colors_out = max_color + 1;

//     // 统计每个颜色的数量
//     JXF_Int *color_count = (JXF_Int *)calloc(*num_colors_out, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//         color_count[color[i]]++;

//     // 构造 color_starts 数组
//     JXF_Int *color_starts = (JXF_Int *)malloc((*num_colors_out + 1) * sizeof(JXF_Int));
//     color_starts[0] = 0;
//     for (JXF_Int c = 1; c <= *num_colors_out; c++)
//         color_starts[c] = color_starts[c - 1] + color_count[c - 1];

//     // 构造 row_by_color 数组，并保证每个颜色的行是按从大到小顺序排列（倒着写入）
//     JXF_Int *row_by_color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *temp_index = (JXF_Int *)malloc(*num_colors_out * sizeof(JXF_Int));
//     memcpy(temp_index, color_starts, *num_colors_out * sizeof(JXF_Int));

//     for (JXF_Int i = nLU - 1; i >= 0; i--)
//     {
//         JXF_Int c = color[i];
//         row_by_color[temp_index[c]++] = i;
//     }
//     //     // === 构造 perm / inv_perm ===
//     // JXF_Int *perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     // JXF_Int *inv_perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     // for (JXF_Int new_i = 0; new_i < nLU; new_i++) {
//     //     perm[new_i] = new_i;
//     //     inv_perm[new_i] = new_i;
//     // }

//     // === 构造 perm / inv_perm ===
//     JXF_Int *perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *inv_perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     for (JXF_Int new_i = 0; new_i < nLU; new_i++) {
//         JXF_Int old_i = row_by_color[nLU - 1 - new_i];
//         perm[old_i] = new_i;
//         inv_perm[new_i] = old_i;
//     }

//     // === 重排 U_diag 到新顺序 ===
//     JXF_Int *new_i = (JXF_Int *)malloc((nLU + 1) * sizeof(JXF_Int));
//     JXF_Int *row_nnz = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++) {
//         JXF_Int old_i = inv_perm[i];
//         row_nnz[i] = U_i[old_i + 1] - U_i[old_i];
//     }

//     new_i[0] = 0;
//     for (JXF_Int i = 1; i <= nLU; i++)
//         new_i[i] = new_i[i - 1] + row_nnz[i - 1];

//     JXF_Int nnz = new_i[nLU];
//     JXF_Int *new_j = (JXF_Int *)malloc(nnz * sizeof(JXF_Int));
//     JXF_Real *new_data = (JXF_Real *)malloc(nnz * sizeof(JXF_Real));
//     JXF_Int *pos = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++) {
//         JXF_Int old_i = inv_perm[i];
//         for (JXF_Int jj = U_i[old_i]; jj < U_i[old_i + 1]; jj++) {
//             JXF_Int old_j = U_j[jj];
//             JXF_Int new_j_idx = perm[old_j];

//             JXF_Int insert_pos = new_i[i] + pos[i];
//             new_j[insert_pos] = new_j_idx;
//             new_data[insert_pos] = U_data[jj];
//             pos[i]++;
//         }
//     }

//     // 释放旧指针（确保无共享）
//     free(U_i);
//     free(U_j);
//     free(U_data);

//     // 更新 U_diag 指针
//     jxf_CSRMatrixI(U_diag) = new_i;
//     jxf_CSRMatrixJ(U_diag) = new_j;
//     jxf_CSRMatrixData(U_diag) = new_data;

//     // === 生成 jlevU_perm（即 perm[row_by_color]） ===
//     JXF_Int *row_by_color_perm = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++)
//     {
//         row_by_color_perm[i] = perm[row_by_color[i]];
//     }

//     // 输出结果
//     *row_by_color_out = row_by_color_perm;
//     *color_starts_out = color_starts;
//     *perm_out = perm;
//     *inv_perm_out = inv_perm;

//     // for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*perm_out)[i]);
//     // printf("\n");
//     // printf("inv_perm_out: ");
//     // for (JXF_Int i = 0; i <= nLU; i++) printf("%d ", (*inv_perm_out)[i]);
//     // printf("\n");

//     //打印调试信息
//     // printf("Number of levels : %d\n", *num_colors_out);
//     // printf("jlevU (row indices per level): ");
//     // for (JXF_Int i = 0; i < nLU; i++) printf("%d ", (*row_by_color_out)[i]);
//     // printf("\n");
//     // printf("ilevU (level offsets): ");
//     // for (JXF_Int i = 0; i <= *num_colors_out; i++) printf("%d ", (*color_starts_out)[i]);
//     // printf("\n");

//     // 清理
//     free(color_count);
//     free(temp_index);
//     free(color);
//    free(row_nnz);
//     free(pos);

//     return 0;
// }

// 对称矩阵A染色
// JXF_Int jxf_GreedyColoring(jxf_ParCSRMatrix *A,
//                                JXF_Int          nLU,
//                                JXF_Int         **row_by_color_out,
//                                JXF_Int         **color_starts_out,
//                                JXF_Int          *num_colors_out)
//  {
//      jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//      JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//      JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);

//     // 分配并初始化颜色数组为-1（未着色）
//     JXF_Int *color = (JXF_Int *)calloc(nLU, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++) color[i] = -1;

//     // 用于标记节点是否已着色
//     char *colored = (char *)calloc(nLU, sizeof(char));
//     JXF_Int *level = (JXF_Int *)calloc(nLU, sizeof(JXF_Int)); // 节点着色等级

//     // 初始化所有节点为0级
//     for (JXF_Int i = 0; i < nLU; i++) level[i] = 0;

//     // 多级着色
//     JXF_Int current_color = 0;
//     JXF_Int remaining = nLU;

//     while (remaining > 0) {
//         JXF_Int colored_this_round = 0;

//         // 遍历所有未着色节点
//         for (JXF_Int i = 0; i < nLU; i++) {
//             if (colored[i]) continue; // 已着色节点跳过

//             // 检查是否所有依赖节点都已完成（颜色值小于当前颜色）
//             JXF_Int can_color = 1;

//             // 检查所有邻居（包括前向和后向依赖）
//             for (JXF_Int jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++) {
//                 JXF_Int neighbor = A_diag_j[jj];

//                 // 如果邻居已着色且颜色相同，则冲突
//                 if (colored[neighbor] && color[neighbor] == current_color) {
//                     can_color = 0;
//                     break;
//                 }

//                 // 如果邻居未着色但依赖关系更强，则等待
//                 if (!colored[neighbor] && level[neighbor] < level[i]) {
//                     can_color = 0;
//                     break;
//                 }
//             }

//             if (can_color) {
//                 color[i] = current_color;
//                 colored[i] = 1;
//                 colored_this_round++;
//             } else {
//                 // 无法着色，提升等级以便下轮考虑
//                 level[i]++;
//             }
//         }

//         if (colored_this_round == 0) {
//             // 处理可能出现的死锁：找出最小等级节点强制着色
//             JXF_Int min_level = nLU + 1;
//             JXF_Int min_index = -1;

//             for (JXF_Int i = 0; i < nLU; i++) {
//                 if (!colored[i] && level[i] < min_level) {
//                     min_level = level[i];
//                     min_index = i;
//                 }
//             }

//             if (min_index >= 0) {
//                 color[min_index] = current_color;
//                 colored[min_index] = 1;
//                 colored_this_round = 1;
//             }
//         }

//         remaining -= colored_this_round;
//         current_color++;
//     }

//     // 统计颜色分布
//     JXF_Int *color_counts = (JXF_Int *)calloc(current_color, sizeof(JXF_Int));
//     for (JXF_Int i = 0; i < nLU; i++) {
//         color_counts[color[i]]++;
//     }

//     // 创建颜色起始索引
//     JXF_Int *color_starts = (JXF_Int *)malloc((current_color + 1) * sizeof(JXF_Int));
//     color_starts[0] = 0;
//     for (JXF_Int c = 0; c < current_color; c++) {
//         color_starts[c + 1] = color_starts[c] + color_counts[c];
//     }

//     // 按颜色排序节点
//     JXF_Int *row_by_color = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *temp_counts = (JXF_Int *)malloc(current_color * sizeof(JXF_Int));
//     memcpy(temp_counts, color_starts, current_color * sizeof(JXF_Int));

//     for (JXF_Int i = 0; i < nLU; i++) {
//         JXF_Int c = color[i];
//         row_by_color[temp_counts[c]++] = i;
//     }

//     // 设置输出
//     *row_by_color_out = row_by_color;
//     *color_starts_out = color_starts;
//     *num_colors_out = current_color;

//     printf("GS iteration coloring completed: total colors = %d\n",  current_color);

//     // 清理
//     free(color);
//     free(colored);
//     free(level);
//     free(color_counts);
//     free(temp_counts);

//     return 0; // 成功
// }

// JXF_Int jxf_GreedyColoring_8GS(jxf_ParCSRMatrix *A,
//                              JXF_Int nLU,
//                              JXF_Int **row_by_color_out,
//                              JXF_Int **color_starts_out,
//                              JXF_Int *num_colors_out,
//                              JXF_Int **row_by_color_out1,
//                              JXF_Int **color_starts_out1,
//                              JXF_Int *num_colors_out1)
// {
//    JXF_Int nx = 127; //(JXF_Int)(round(pow((float)nLU, 1.0 / 3.0)));
//    JXF_Int ny = 127;
//    JXF_Int nz = 127;

//     if (nx * ny * nz != nLU) {
//         printf("Error: Grid size %d x %d x %d does not match global rows %d\n", nx, ny, nz, nLU);
//         return -1;
//     }

//     /*======================= 正向着色 ========================*/
//     JXF_Int *color_count = (JXF_Int *)calloc(8, sizeof(JXF_Int));
//     for (JXF_Int local_row = 0; local_row < nLU; local_row++) {
//         JXF_Int i = local_row % nx;
//         JXF_Int j = (local_row / nx) % ny;
//         JXF_Int k = local_row / (nx * ny);
//         JXF_Int color = (i % 2) + (j % 2) * 2 + (k % 2) * 4;
//         color_count[color]++;
//     }

//     *row_by_color_out = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     *color_starts_out = (JXF_Int *)malloc(9 * sizeof(JXF_Int));
//     (*color_starts_out)[0] = 0;
//     for (JXF_Int c = 0; c < 8; c++) {
//         (*color_starts_out)[c + 1] = (*color_starts_out)[c] + color_count[c];
//     }
//     *num_colors_out = 8;

//     JXF_Int *temp_count = (JXF_Int *)calloc(8, sizeof(JXF_Int));
//     for (JXF_Int c = 0; c < 8; c++) {
//         temp_count[c] = (*color_starts_out)[c];
//     }
//     for (JXF_Int local_row = 0; local_row < nLU; local_row++) {
//         JXF_Int i = local_row % nx;
//         JXF_Int j = (local_row / nx) % ny;
//         JXF_Int k = local_row / (nx * ny);
//         JXF_Int color = (i % 2) + (j % 2) * 2 + (k % 2) * 4;
//         (*row_by_color_out)[temp_count[color]++] = local_row;
//     }

//     /*======================= 反向着色 ========================*/
//     // 计算最后一个点的原始颜色
//     JXF_Int last_r = nLU - 1;
//     JXF_Int i_last = last_r % nx;
//     JXF_Int j_last = (last_r / nx) % ny;
//     JXF_Int k_last = last_r / (nx * ny);
//     JXF_Int base_color = (i_last % 2) + (j_last % 2) * 2 + (k_last % 2) * 4;

//     JXF_Int *colors = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     JXF_Int *color_count1 = (JXF_Int *)calloc(8, sizeof(JXF_Int));
//     for (JXF_Int r = 0; r < nLU; r++) {
//         JXF_Int i = r % nx;
//         JXF_Int j = (r / nx) % ny;
//         JXF_Int k = r / (nx * ny);
//         JXF_Int orig_color = (i % 2) + (j % 2) * 2 + (k % 2) * 4;
//         colors[r] = (orig_color - base_color + 8) % 8;
//         color_count1[colors[r]]++;
//     }

//     *row_by_color_out1 = (JXF_Int *)malloc(nLU * sizeof(JXF_Int));
//     *color_starts_out1 = (JXF_Int *)malloc(9 * sizeof(JXF_Int));
//     (*color_starts_out1)[0] = 0;
//     for (JXF_Int c = 0; c < 8; c++) {
//         (*color_starts_out1)[c + 1] = (*color_starts_out1)[c] + color_count1[c];
//     }
//     *num_colors_out1 = 8;

//     JXF_Int *color_pos = (JXF_Int *)calloc(8, sizeof(JXF_Int));
//     for (JXF_Int c = 0; c < 8; c++) {
//         color_pos[c] = (*color_starts_out1)[c];
//     }

//     for (JXF_Int r = nLU - 1; r >= 0; r--) {
//         JXF_Int c = colors[r];
//         (*row_by_color_out1)[color_pos[c]++] = r;
//     }

//     free(colors);
//     free(color_count);
//     free(color_count1);
//     free(temp_count);
//     free(color_pos);

//     return 0;
// }

// JXF_Int jxf_ILUSetupFGPILU_v3(jxf_ParCSRMatrix *A,
//                             JXF_Int sweep,
//                             jxf_ParCSRMatrix **Lptr,
//                             jxf_ParVector **D,
//                             jxf_ParCSRMatrix **Uptr,
//                             jxf_ParCSRMatrix **Sptr,
//                             JXF_Int **u_end)
// {
//     MPI_Comm comm = jxf_ParCSRMatrixComm(A);
//     JXF_Int num_procs, my_id;
//     jxf_ParCSRCommPkg *comm_pkg = NULL;

//     /* MPI initialization */
//     jxf_MPI_Comm_size(comm, &num_procs);
//     jxf_MPI_Comm_rank(comm, &my_id);

//     comm_pkg = jxf_ParCSRMatrixCommPkg(A);
//     if (!comm_pkg)
//     {
//         jxf_MatvecCommPkgCreate(A);
//         comm_pkg = jxf_ParCSRMatrixCommPkg(A);
//     }

//     JXF_Int i, j, k;
//     JXF_Int ROW_OMP, temp_idx;
//     JXF_Int nthreads = 1;
//     JXF_Real local_nnz = 0.0, total_nnz = 0.0;

//     /* Data objects for A */
//     jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//     JXF_Real *A_diag_data = jxf_CSRMatrixData(A_diag);
//     JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//     JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);
//     JXF_Int n = jxf_CSRMatrixNumRows(A_diag);
//     JXF_Int nnz_A = A_diag_i[n];

//     JXF_Int first_col_diag_A = jxf_ParCSRMatrixFirstColDiag(A);
//     JXF_Int num_cols_diag_A = jxf_CSRMatrixNumCols(A_diag);
//     JXF_Int last_col_diag_A = first_col_diag_A + num_cols_diag_A - 1;
//     JXF_Int *col_map_offd_A = jxf_ParCSRMatrixColMapOffd(A);

//     /* Data objects for L, D, U */
//     jxf_ParCSRMatrix *matL = NULL, *matU = NULL;
//     jxf_CSRMatrix *L_diag = NULL, *U_diag = NULL;
//     JXF_Real *D_data = NULL;
//     JXF_Real *d_data = NULL;
//     JXF_Real *L_diag_data = NULL, *U_diag_data = NULL;
//     JXF_Int *L_diag_i = NULL, *L_diag_j = NULL;
//     JXF_Int *U_diag_i = NULL, *U_diag_j = NULL;
//     JXF_Int *workL = NULL, *workU = NULL;
//     JXF_Int *workLO = NULL, *workUO = NULL;
//     JXF_Int LD_nnz = 0, UD_nnz = 0;

//     /* temporaries for C = A - L*U */
//     JXF_Real *C_diag_data = NULL;

//     JXF_Int nt = 1;
//     JXF_Int global_row, global_col;

//     /* Data objects for A */
//     jxf_CSRMatrix *A_offd = jxf_ParCSRMatrixOffd(A);
//     JXF_Real *A_offd_data = jxf_CSRMatrixData(A_offd);
//     JXF_Int *A_offd_i = jxf_CSRMatrixI(A_offd);
//     JXF_Int *A_offd_j = jxf_CSRMatrixJ(A_offd);
//     JXF_Int nnz_AO = A_offd_i[n];

//     JXF_Int num_cols_offd_A =  jxf_CSRMatrixNumCols(A_offd);
//     JXF_Int L_offd_cols =  jxf_CSRMatrixNumCols(A_offd);
//     JXF_Int U_offd_cols =  jxf_CSRMatrixNumCols(A_offd);

//     jxf_CSRMatrix *L_offd = NULL, *U_offd = NULL;
//     JXF_Real *L_offd_data = NULL, *U_offd_data = NULL;
//     JXF_Int *L_offd_i = NULL, *L_offd_j = NULL;
//     JXF_Int *U_offd_i = NULL, *U_offd_j = NULL;
//     JXF_Int LO_nnz = 0, UO_nnz = 0;

//     JXF_Real *C_offd_data = NULL;

//     C_diag_data = jxf_TAlloc(JXF_Real, nnz_A);
//     C_offd_data = jxf_TAlloc(JXF_Real, nnz_AO);

//     /* Thread info */
//     nt = jxf_NumActiveThreads();
//     if (nt <= 0)
//         nt = 1;
//     nthreads = nt;

//     ROW_OMP = (n + nt - 1) / nt;

//     /* allocate prefix arrays (per-thread) */
//     workL = jxf_TAlloc(JXF_Int, nt + 1);
//     workU = jxf_TAlloc(JXF_Int, nt + 1);

//     workLO = jxf_TAlloc(JXF_Int, nt + 1);
//     workUO = jxf_TAlloc(JXF_Int, nt + 1);
//     if (!workL || !workU)
//     {
//         fprintf(stderr, "Memory alloc failure workL/workU\n");
//         jxf_TFree(workL);
//         jxf_TFree(workU);
//         return -1;
//     }
//     for (i = 0; i <= nt; i++)
//     {
//         workL[i] = 0;
//         workU[i] = 0;
//     }
     
// /* 1) Count diag (local) lower/upper nnz per-thread */
// #pragma omp parallel num_threads(nt)
//     {
//         JXF_Int tid = jxf_GetThreadNum();
//         JXF_Int istart = tid * ROW_OMP;
//         JXF_Int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//         JXF_Int localL = 0, localU = 0;
//         for (JXF_Int r = istart; r < iend && r < n; r++)
//         {
//             for (JXF_Int jj = A_diag_i[r]; jj < A_diag_i[r + 1]; jj++)
//             {
//                 JXF_Int col = A_diag_j[jj];
//                 if (col < r)
//                     localL++;
//                 else if (col > r)
//                     localU++;
//             }
//         }
//         workL[tid + 1] = localL;
//         workU[tid + 1] = localU;
// #pragma omp barrier
// #pragma omp single
//         {
//             for (JXF_Int t = 1; t <= nt; t++)
//             {
//                 workL[t] += workL[t - 1];
//                 workU[t] += workU[t - 1];
//             }
//         }
//     }

//     LD_nnz = workL[nt];
//     UD_nnz = workU[nt];

//     /* allocate diag arrays */
//     D_data = jxf_TAlloc(JXF_Real, n);
//     d_data = jxf_TAlloc(JXF_Real, n);

//     L_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     U_diag_i = jxf_TAlloc(JXF_Int, n + 1);
//     L_diag_j = jxf_TAlloc(JXF_Int, LD_nnz);
//     U_diag_j = jxf_TAlloc(JXF_Int, UD_nnz);
//     L_diag_data = jxf_TAlloc(JXF_Real, LD_nnz);
//     U_diag_data = jxf_TAlloc(JXF_Real, UD_nnz);

// /* fill diag index/data from A (split into L/U and diagonal) */
// #pragma omp parallel num_threads(nt)
//     {
//         JXF_Int tid = jxf_GetThreadNum();
//         JXF_Int istart = tid * ROW_OMP;
//         JXF_Int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//         JXF_Int cntL = 0, cntU = 0;
//         for (JXF_Int r = istart; r < iend && r < n; r++)
//         {
//             L_diag_i[r] = workL[tid] + cntL;  //构建偏移指针
//             U_diag_i[r] = workU[tid] + cntU;
//             for (JXF_Int jj = A_diag_i[r]; jj < A_diag_i[r + 1]; jj++)
//             {
//                 JXF_Int col = A_diag_j[jj];
//                 JXF_Real aval = A_diag_data[jj];
//                 if (col < r)
//                 {
//                     L_diag_j[workL[tid] + cntL] = col;
//                     L_diag_data[workL[tid] + cntL] = D_data[r] * aval;
//                     cntL++;
//                 }
//                 else if (col > r)
//                 {
//                     U_diag_j[workU[tid] + cntU] = col;
//                     U_diag_data[workU[tid] + cntU] = aval;
//                     cntU++;
//                 }
//                 else
//                 {
//                     d_data[r] = aval;
//                     /* guard diagonal */
//                     if (fabs(d_data[r]) < MAT_TOL)
//                     {
//                         d_data[r] = (JXF_Real)1e-6;
//                     }
//                     D_data[r] = (JXF_Real)1.0 / d_data[r];
//                 }
//             }
//         }
//     }
//     L_diag_i[n] = LD_nnz;
//     U_diag_i[n] = UD_nnz;

//     if (num_procs > 1)
//     {
//         /* reset work arrays for offd counting */
//         for (i = 0; i <= nt; i++)
//         {
//             workLO[i] = 0;
//             workUO[i] = 0;
//         }

// /* 2) Count offd lower/upper nnz per-thread (use global compare) */
// #pragma omp parallel num_threads(nt)
//         {
//             JXF_Int tid = jxf_GetThreadNum();
//             JXF_Int istart = tid * ROW_OMP;
//             JXF_Int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//             JXF_Int localL = 0, localU = 0;
//             for (JXF_Int r = istart; r < iend && r < n; r++)
//             {
//                 JXF_Int global_row_idx = r + first_col_diag_A;
//                 for (JXF_Int jj = A_offd_i[r]; jj < A_offd_i[r + 1]; jj++)
//                 {
//                     JXF_Int offd_col_idx = A_offd_j[jj]; /* index into col_map_offd_A */
//                     JXF_Int global_col_idx = col_map_offd_A[offd_col_idx];
//                     if (global_col_idx < global_row_idx)
//                         localL++;
//                     else if (global_col_idx > global_row_idx)
//                         localU++;
//                 }
//             }
//             workLO[tid + 1] = localL;
//             workUO[tid + 1] = localU;
// #pragma omp barrier
// #pragma omp single
//             {
//                 for (JXF_Int t = 1; t <= nt; t++)
//                 {
//                     workLO[t] += workLO[t - 1];
//                     workUO[t] += workUO[t - 1];
//                 }
//             }
//         }

//         LO_nnz = workLO[nt];
//         UO_nnz = workUO[nt];

//         /* allocate offd arrays */
//         L_offd_i = jxf_TAlloc(JXF_Int, n + 1);
//         U_offd_i = jxf_TAlloc(JXF_Int, n + 1);
//         L_offd_j = jxf_TAlloc(JXF_Int, LO_nnz);
//         U_offd_j = jxf_TAlloc(JXF_Int, UO_nnz);
//         L_offd_data = jxf_TAlloc(JXF_Real, LO_nnz);
//         U_offd_data = jxf_TAlloc(JXF_Real, UO_nnz);

// /* fill offd structures from A */
// #pragma omp parallel num_threads(nt)
//         {
//             JXF_Int tid = jxf_GetThreadNum();
//             JXF_Int istart = tid * ROW_OMP;
//             JXF_Int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//             JXF_Int cntL = 0, cntU = 0;
//             for (JXF_Int r = istart; r < iend && r < n; r++)
//             {
//                 L_offd_i[r] = workLO[tid] + cntL;
//                 U_offd_i[r] = workUO[tid] + cntU;
//                 JXF_Int global_row_idx = r + first_col_diag_A;
//                 for (JXF_Int jj = A_offd_i[r]; jj < A_offd_i[r + 1]; jj++)
//                 {
//                     JXF_Int offd_idx = A_offd_j[jj];
//                     JXF_Int global_col_idx = col_map_offd_A[offd_idx];
//                     JXF_Real aval = A_offd_data[jj];
//                     if (global_col_idx < global_row_idx)
//                     {
//                         L_offd_j[workLO[tid] + cntL] = offd_idx;
//                         L_offd_data[workLO[tid] + cntL] = D_data[r] * aval;
//                         cntL++;
//                     }
//                     else   //(global_col_idx > global_row_idx)
//                     {
//                         U_offd_j[workUO[tid] + cntU] = offd_idx;
//                         U_offd_data[workUO[tid] + cntU] = aval;
//                         cntU++;
//                     }
//                 }
//             }
//         }
//         L_offd_i[n] = LO_nnz;
//         U_offd_i[n] = UO_nnz;
//     }

//     /* create ParCSR matrices matL and matU (structure only) */
//     matL = jxf_ParCSRMatrixCreate(comm,
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumCols(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A),
//                                  L_offd_cols,
//                                  LD_nnz,
//                                  LO_nnz );
//     /* attach diag/offd pointers for L */
//     L_diag = jxf_ParCSRMatrixDiag(matL);
//     jxf_CSRMatrixI(L_diag) = L_diag_i;
//     jxf_CSRMatrixJ(L_diag) = L_diag_j;
//     jxf_CSRMatrixData(L_diag) = L_diag_data;
//     // jxf_CSRMatrixNumRows(L_diag); /* just ensure object exists */
//     // /* offd */
//     // jxf_ParCSRMatrixOffd(matL);       /* ensure offd exists if function provided */
//     // jxf_ParCSRMatrixColMapOffd(matL); /* ensure colmap exists if needed */

//     /* For U */
//     matU = jxf_ParCSRMatrixCreate(comm,
//                                  jxf_ParCSRMatrixGlobalNumRows(A),
//                                  jxf_ParCSRMatrixGlobalNumCols(A),
//                                  jxf_ParCSRMatrixRowStarts(A),
//                                  jxf_ParCSRMatrixColStarts(A),
//                                  U_offd_cols,
//                                  UD_nnz,
//                                  UO_nnz);
//     U_diag = jxf_ParCSRMatrixDiag(matU);
//     jxf_CSRMatrixI(U_diag) = U_diag_i;
//     jxf_CSRMatrixJ(U_diag) = U_diag_j;
//     jxf_CSRMatrixData(U_diag) = U_diag_data;

//     if (num_procs > 1)
//     {
//         // JXF_Int num_cols_offd_A = jxf_ParCSRMatrixNumColsOffd(A);
//         // JXF_Int *col_map_offd_A = jxf_ParCSRMatrixColMapOffd(A);

//         /* offd 部分指针绑定 */
//         jxf_CSRMatrix *L_offd = jxf_ParCSRMatrixOffd(matL);
//         jxf_CSRMatrix *U_offd = jxf_ParCSRMatrixOffd(matU);

//         jxf_CSRMatrixI(L_offd) = L_offd_i;
//         jxf_CSRMatrixJ(L_offd) = L_offd_j;
//         jxf_CSRMatrixData(L_offd) = L_offd_data;
//         jxf_CSRMatrixNumCols(L_offd) = L_offd_cols;

//         jxf_CSRMatrixI(U_offd) = U_offd_i;
//         jxf_CSRMatrixJ(U_offd) = U_offd_j;
//         jxf_CSRMatrixData(U_offd) = U_offd_data;
//         jxf_CSRMatrixNumCols(U_offd) = U_offd_cols;

//         /* col_map_offd 复制 */
//         JXF_Int *col_map_offd_L = jxf_CTAlloc(JXF_Int, L_offd_cols);
//         JXF_Int *col_map_offd_U = jxf_CTAlloc(JXF_Int, U_offd_cols);

//         if (num_cols_offd_A > 0 && col_map_offd_A)
//         {
//             for (i = 0; i < num_cols_offd_A; i++)
//             {
//                 col_map_offd_L[i] = col_map_offd_A[i];
//                 col_map_offd_U[i] = col_map_offd_A[i];
//             }
//         }
//         jxf_ParCSRMatrixColMapOffd(matL) = col_map_offd_L;
//         jxf_ParCSRMatrixColMapOffd(matU) = col_map_offd_U;

//         /* 通信包创建 */
//         jxf_MatvecCommPkgCreate(matL);
//         jxf_MatvecCommPkgCreate(matU);
//     }

//     /* MAIN SWEEPS: ILU iterations */
//     for (JXF_Int sweep_it = 1; sweep_it < sweep; sweep_it++)
//     {
//         /* compute C = A - L * U (uses provided function to write to C_diag_data/C_offd_data) */
//         jxf_ParCSRMatrixMatmul_Apattern(A, matL, matU, C_diag_data, C_offd_data);

//         #pragma omp parallel num_threads(nt)
//         {
//             JXF_Int tid = jxf_GetThreadNum();
//             JXF_Int istart = tid * ROW_OMP;
//             JXF_Int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//             for (JXF_Int r = istart; r < iend && r < n; r++)
//             {
//                 j = A_diag_i[r]; 
//                 JXF_Int col = A_diag_j[j];
//                 JXF_Real aval = C_diag_data[j];
                    
//                 d_data[r] = aval;
//                 /* guard diagonal */
//                 if (fabs(d_data[r]) < MAT_TOL)
//                 {
//                     d_data[r] = (JXF_Real)1e-6;
//                 }
//                 D_data[r] = (JXF_Real)1.0 / d_data[r];
//             }
//             for (JXF_Int r = istart; r < iend && r < n; r++)
//             {
//                 JXF_Int localLpos = L_diag_i[r];
//                 JXF_Int localUpos = U_diag_i[r];
//                 for (JXF_Int jj = A_diag_i[r]; jj < A_diag_i[r + 1]; jj++)
//                 {
//                     JXF_Int col = A_diag_j[jj];
//                     JXF_Real cval = C_diag_data[jj];
//                     if (col < r)
//                     {
//                         /* store in L_diag_data at next free position for r */
//                         L_diag_data[localLpos++] = D_data[r] * cval;
//                     }
//                     else if (col > r)
//                     {
//                         U_diag_data[localUpos++] = cval;
//                     }
//                     else
//                     {
//                         /* diagonal: update d_data and D_data */
//                         d_data[r] = cval;
//                         if (fabs(d_data[r]) < MAT_TOL)
//                         {
//                             d_data[r] = (JXF_Real)1e-6;
//                         }
//                         D_data[r] = (JXF_Real)1.0 / d_data[r];  //对角元提前放的原因
//                     }
//                 }
//             }
//         } /* end omp parallel */

//         if (num_procs > 1)
//         {
//         #pragma omp parallel num_threads(nt)
//             {
//                 JXF_Int tid = jxf_GetThreadNum();
//                 JXF_Int istart = tid * ROW_OMP;
//                 JXF_Int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//                 // JXF_Int countL = 0, countU = 0;
//                 for (JXF_Int r = istart; r < iend && r < n; r++)
//                 {
//                     /* iterate A_offd positions; we must know mapping from A_offd positions to L_offd/U_offd positions:
//                     earlier when we built L_offd_j/U_offd_j we used workL/workU offsets; we can reuse the same scheme:
//                     */
//                     JXF_Int countU = L_offd_i[tid];
//                     JXF_Int countL = U_offd_i[tid];
//                     JXF_Int global_row_idx = r + first_col_diag_A;
//                     for (JXF_Int jj = A_offd_i[r]; jj < A_offd_i[r + 1]; jj++)
//                     {
//                         JXF_Int offd_idx = A_offd_j[jj];
//                         JXF_Int gcol = col_map_offd_A[offd_idx];
//                         JXF_Real cval = C_offd_data[jj];
//                         if (gcol > global_row_idx)
//                         {
//                             U_offd_data[countU] = cval;
//                             countU++;
//                         }
//                         else  // (gcol < global_row_idx)
//                         {
//                             L_offd_data[countL] = D_data[r] * cval;
//                             countL++;
//                         }
//                     } /* end A_offd loop */
//                 } /* end for rows */
//             } /* end omp parallel */
//         }

//     } /* end sweeps */

//     /* compute and set global nnz counts for matU and matL (float stored) */
//     local_nnz = (JXF_Real)(UD_nnz + UO_nnz);
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

//     local_nnz = (JXF_Real)(LD_nnz + LO_nnz);
//     jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
//     jxf_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

//     jxf_ParVector *D_array = jxf_ParVectorCreate(jxf_ParCSRMatrixComm(A),
//                                                jxf_ParCSRMatrixGlobalNumRows(A),
//                                                jxf_ParCSRMatrixRowStarts(A));
//     jxf_ParVectorInitialize(D_array);

//     jxf_Vector *D_local = jxf_ParVectorLocalVector(D_array);
//     jxf_VectorData(D_local) = D_data;


//     /* outputs */
//     *Lptr = matL;
//     // *Dptr = D_data;
//     *Uptr = matU;
//     *Sptr = NULL;
//     *u_end = NULL;
//     *D = D_array;

//     /* free temporaries */
//     jxf_TFree(workL);
//     workL = NULL;
//     jxf_TFree(workU);
//     workU = NULL;

//     jxf_TFree(workLO);
//     workLO = NULL;
//     jxf_TFree(workUO);
//     workUO = NULL;

//     jxf_TFree(C_diag_data);
//     C_diag_data = NULL;
//     jxf_TFree(C_offd_data);
//     C_offd_data = NULL;
//     jxf_TFree(d_data);

//     /* don't free L/U arrays: returned via matL/matU ownership */

//     return 0;
// }

// void jxf_ParCSRMatrixMatmul_Apattern(jxf_ParCSRMatrix *A, jxf_ParCSRMatrix *L, jxf_ParCSRMatrix *B,
//                                     JXF_Real *C_diag_data, JXF_Real *C_offd_data)
// {

//     MPI_Comm comm = jxf_ParCSRMatrixComm(A);
//     JXF_Int num_procs;
//     jxf_MPI_Comm_size(comm, &num_procs);

//     /* Basic pointers to A and B blocks */
//     jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//     jxf_CSRMatrix *A_offd = jxf_ParCSRMatrixOffd(A);
//     JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//     JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);
//     JXF_Real *A_diag_data = jxf_CSRMatrixData(A_diag);
//     JXF_Int *A_offd_i = jxf_CSRMatrixI(A_offd);
//     JXF_Int *A_offd_j = jxf_CSRMatrixJ(A_offd);
//     JXF_Real *A_offd_data = jxf_CSRMatrixData(A_offd);

//     JXF_Int *col_map_offd_A = jxf_ParCSRMatrixColMapOffd(A);

//     JXF_Int *my_diag_array;
//     JXF_Int *my_offd_array;
//     JXF_Int max_num_threads = jxf_NumThreads();
//     my_diag_array = jxf_CTAlloc(JXF_Int, max_num_threads);
//     my_offd_array = jxf_CTAlloc(JXF_Int, max_num_threads);

//     jxf_CSRMatrix *B_diag = jxf_ParCSRMatrixDiag(B);
//     jxf_CSRMatrix *B_offd = jxf_ParCSRMatrixOffd(B);
//     JXF_Int *B_diag_i = jxf_CSRMatrixI(B_diag);
//     JXF_Int *B_diag_j = jxf_CSRMatrixJ(B_diag);
//     JXF_Real *B_diag_data = jxf_CSRMatrixData(B_diag);
//     JXF_Int *B_offd_i = jxf_CSRMatrixI(B_offd);
//     JXF_Int *B_offd_j = jxf_CSRMatrixJ(B_offd);
//     JXF_Real *B_offd_data = jxf_CSRMatrixData(B_offd);

//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     jxf_CSRMatrix *L_offd = jxf_ParCSRMatrixOffd(L);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_offd_i = jxf_CSRMatrixI(L_offd);
//     JXF_Int *L_offd_j = jxf_CSRMatrixJ(L_offd);
//     JXF_Real *L_offd_data = jxf_CSRMatrixData(L_offd);

//     /* local sizes / meta */
//     JXF_Int num_rows_diag_A = jxf_CSRMatrixNumRows(A_diag);
//     JXF_Int num_cols_diag_A = jxf_CSRMatrixNumCols(A_diag);
//     JXF_Int num_cols_offd_A = jxf_CSRMatrixNumCols(A_offd);
//     JXF_Int first_col_diag_A = jxf_ParCSRMatrixFirstColDiag(A);
//     JXF_Int last_col_diag_A = first_col_diag_A + num_cols_diag_A - 1;

//     JXF_Int first_col_diag_B = jxf_ParCSRMatrixFirstColDiag(B);
//     JXF_Int num_cols_diag_B = jxf_CSRMatrixNumCols(B_diag);
//     JXF_Int last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

//     jxf_CSRMatrix *Bs_ext = NULL;
//     JXF_Real *Bs_ext_data = NULL;
//     JXF_Int *Bs_ext_i = NULL;
//     JXF_Int *Bs_ext_j = NULL;
//     JXF_Int *B_ext_diag_i = NULL, *B_ext_offd_i = NULL;
//     JXF_Int *B_ext_diag_j = NULL, *B_ext_offd_j = NULL;
//     JXF_Real *B_ext_diag_data = NULL, *B_ext_offd_data = NULL;
//     JXF_Int B_ext_diag_size = 0, B_ext_offd_size = 0;

//     // /* For create C later (not used here, since C arrays provided) */
//     // JXF_Int *row_starts_A = jxf_ParCSRMatrixRowStarts(A);

//     if (num_procs > 1)
//     {
//         /*---------------------------------------------------------------------
//          * If there exists no CommPkg for A, a CommPkg is generated using
//          * equally load balanced partitionings within
//          * jxf_ParCSRMatrixExtractBExt
//          *--------------------------------------------------------------------*/
//         Bs_ext = jxf_ParCSRMatrixExtractBExt(B, A, 1);
//         Bs_ext_data = jxf_CSRMatrixData(Bs_ext);
//         Bs_ext_i = jxf_CSRMatrixI(Bs_ext);
//         Bs_ext_j = jxf_CSRMatrixJ(Bs_ext);
//     }
//     B_ext_diag_i = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
//     B_ext_offd_i = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
//     B_ext_diag_size = 0;
//     B_ext_offd_size = 0;

//     // printf("1\n");

//     #if JXF_USING_OPENMP
//     #pragma omp parallel
//     #endif
//     {
//         JXF_Int size, rest, ii;
//         JXF_Int ns, ne;
//         JXF_Int i1, i, j;
//         JXF_Int my_offd_size, my_diag_size;
//         JXF_Int cnt_offd, cnt_diag;

//         JXF_Int num_threads = jxf_NumActiveThreads();

//         size = num_cols_offd_A / num_threads;
//         rest = num_cols_offd_A - size * num_threads;
//         ii = jxf_GetThreadNum();
//         if (ii < rest)
//         {
//             ns = ii * size + ii;
//             ne = (ii + 1) * size + ii + 1;
//         }
//         else
//         {
//             ns = ii * size + rest;
//             ne = (ii + 1) * size + rest;
//         }

//         my_diag_size = 0;
//         my_offd_size = 0;
//         for (i = ns; i < ne; i++)
//         {
//             B_ext_diag_i[i] = my_diag_size;
//             B_ext_offd_i[i] = my_offd_size;
//             for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
//             {
//                 if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
//                 {
//                     my_offd_size++;
//                 }
//                 else
//                 {
//                     my_diag_size++;
//                 }
//             }
//         }
//         my_diag_array[ii] = my_diag_size;
//         my_offd_array[ii] = my_offd_size;

//         #if JXF_USING_OPENMP
//         #pragma omp barrier
//         #endif

//         if (ii)
//         {
//             my_diag_size = my_diag_array[0];
//             my_offd_size = my_offd_array[0];
//             for (i1 = 1; i1 < ii; i1++)
//             {
//                 my_diag_size += my_diag_array[i1];
//                 my_offd_size += my_offd_array[i1];
//             }

//             for (i1 = ns; i1 < ne; i1++)
//             {
//                 B_ext_diag_i[i1] += my_diag_size;
//                 B_ext_offd_i[i1] += my_offd_size;
//             }
//         }
//         else
//         {
//             B_ext_diag_size = 0;
//             B_ext_offd_size = 0;
//             for (i1 = 0; i1 < num_threads; i1++)
//             {
//                 B_ext_diag_size += my_diag_array[i1];
//                 B_ext_offd_size += my_offd_array[i1];
//             }
//             B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
//             B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

//             if (B_ext_diag_size)
//             {
//                 B_ext_diag_j = jxf_CTAlloc(JXF_Int, B_ext_diag_size);
//                 B_ext_diag_data = jxf_CTAlloc(JXF_Real, B_ext_diag_size);
//             }
//             if (B_ext_offd_size)
//             {
//                 B_ext_offd_j = jxf_CTAlloc(JXF_Int, B_ext_offd_size);
//                 B_ext_offd_data = jxf_CTAlloc(JXF_Real, B_ext_offd_size);
//             }
//         }
        
//         #if JXF_USING_OPENMP
//         #pragma omp barrier
//         #endif

//         cnt_offd = B_ext_offd_i[ns];
//         cnt_diag = B_ext_diag_i[ns];
//         for (i = ns; i < ne; i++)
//         {
//             for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
//                 if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
//                 {
//                     B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
//                     B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
//                 }
//                 else
//                 {
//                     B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
//                     B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
//                 }
//         }

//     #if JXF_USING_OPENMP
//     #pragma omp barrier
//     #endif
//     }

//     JXF_Int ii1, jj1, jj2, jj3, jj4;
//     JXF_Int i2;
//     JXF_Real a_entry;
//     JXF_Int target_col;
//     JXF_Int global_col;

//     /* offd part */
//     if (num_cols_offd_A)
//     {
//         for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
//         {
//             for (jj1 = A_offd_i[ii1]; jj1 < A_offd_i[ii1 + 1]; jj1++)
//             {
//                 C_offd_data[jj1] = A_offd_data[jj1];
//                 target_col = A_offd_j[jj1];
//                 global_col = col_map_offd_A[target_col];

//                 // L_offd * B_ext_offd
//                 if(num_procs > 1)
//                 {
//                     for (jj2 = L_offd_i[ii1]; jj2 < L_offd_i[ii1 + 1]; jj2++)
//                     {
//                         i2 = L_offd_j[jj2];
//                         for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
//                         {
//                             if (B_ext_offd_j[jj3] == global_col)
//                             {
//                                 C_offd_data[jj1] -= L_offd_data[jj2] * B_ext_offd_data[jj3];
//                             }
//                         }
//                     }
//                 }

//                 // L_diag * B_offd
//                 for (jj2 = L_diag_i[ii1]; jj2 < L_diag_i[ii1 + 1]; jj2++)
//                 {
//                     i2 = L_diag_j[jj2];
//                     for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
//                     {
//                         if (B_offd_j[jj3] == target_col)
//                         {
//                             C_offd_data[jj1] -= L_diag_data[jj2] * B_offd_data[jj3];
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     // printf("4\n");


//         for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
//         {
//             for (jj1 = A_diag_i[ii1]; jj1 < A_diag_i[ii1 + 1]; jj1++)
//             {
//                 C_diag_data[jj1] = A_diag_data[jj1];
//                 target_col = A_diag_j[jj1];

//                 if(num_procs > 1)
//                 {
//                     // L_offd * B_ext_diag
//                     for (jj2 = L_offd_i[ii1]; jj2 < L_offd_i[ii1 + 1]; jj2++)
//                     {
//                         i2 = L_offd_j[jj2];
//                         for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
//                         {
//                             if (B_ext_diag_j[jj3] == target_col)
//                             {
//                                 C_diag_data[jj1] -= L_offd_data[jj2] * B_ext_diag_data[jj3];
//                             }
//                         }
//                     }
//                 }

//                 // L_diag * B_diag
//                 for (jj2 = L_diag_i[ii1]; jj2 < L_diag_i[ii1 + 1]; jj2++)
//                 {
//                     i2 = L_diag_j[jj2];
//                     for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
//                     {
//                         if (B_diag_j[jj3] == target_col)
//                         {
//                             C_diag_data[jj1] -= L_diag_data[jj2] * B_diag_data[jj3];
//                         }
//                     }
//                 }
//             }
//         }


//     if (num_procs > 1)
//     {
//     jxf_CSRMatrixDestroy(Bs_ext);
//     Bs_ext = NULL;
//     }
    
//     jxf_TFree(B_ext_diag_i);
//     if (B_ext_diag_size)
//     {
//     jxf_TFree(B_ext_diag_j);
//     jxf_TFree(B_ext_diag_data);
//     }
//     jxf_TFree(B_ext_offd_i);
//     if (B_ext_offd_size)
//     {
//     jxf_TFree(B_ext_offd_j);
//     jxf_TFree(B_ext_offd_data);
//     }

//     jxf_TFree(my_diag_array);
//     jxf_TFree(my_offd_array);
//     return;
// }


// void jxf_ParCSRMatrixMatmul_Apattern(jxf_ParCSRMatrix *A, jxf_ParCSRMatrix *L, jxf_ParCSRMatrix *B,
//                                     JXF_Real *C_diag_data, JXF_Real *C_offd_data)
// {

//     MPI_Comm comm = jxf_ParCSRMatrixComm(A);
//     JXF_Int num_procs;
//     jxf_MPI_Comm_size(comm, &num_procs);

//     /* Basic pointers to A and B blocks */
//     jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//     jxf_CSRMatrix *A_offd = jxf_ParCSRMatrixOffd(A);
//     JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//     JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);
//     JXF_Real *A_diag_data = jxf_CSRMatrixData(A_diag);
//     JXF_Int A_diag_nnz = jxf_CSRMatrixNumNonzeros(A_diag);
//     JXF_Int *A_offd_i = jxf_CSRMatrixI(A_offd);
//     JXF_Int *A_offd_j = jxf_CSRMatrixJ(A_offd);
//     JXF_Real *A_offd_data = jxf_CSRMatrixData(A_offd);
//     JXF_Int A_offd_nnz = jxf_CSRMatrixNumNonzeros(A_offd);


//     JXF_Int *col_map_offd_A = jxf_ParCSRMatrixColMapOffd(A);

//     JXF_Int *my_diag_array;
//     JXF_Int *my_offd_array;
//     JXF_Int max_num_threads = jxf_NumThreads();
//     my_diag_array = jxf_CTAlloc(JXF_Int, max_num_threads);
//     my_offd_array = jxf_CTAlloc(JXF_Int, max_num_threads);

//     jxf_CSRMatrix *B_diag = jxf_ParCSRMatrixDiag(B);
//     jxf_CSRMatrix *B_offd = jxf_ParCSRMatrixOffd(B);
//     JXF_Int *B_diag_i = jxf_CSRMatrixI(B_diag);
//     JXF_Int *B_diag_j = jxf_CSRMatrixJ(B_diag);
//     JXF_Real *B_diag_data = jxf_CSRMatrixData(B_diag);
//     JXF_Int *B_offd_i = jxf_CSRMatrixI(B_offd);
//     JXF_Int *B_offd_j = jxf_CSRMatrixJ(B_offd);
//     JXF_Real *B_offd_data = jxf_CSRMatrixData(B_offd);

//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     jxf_CSRMatrix *L_offd = jxf_ParCSRMatrixOffd(L);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_offd_i = jxf_CSRMatrixI(L_offd);
//     JXF_Int *L_offd_j = jxf_CSRMatrixJ(L_offd);
//     JXF_Real *L_offd_data = jxf_CSRMatrixData(L_offd);

//     /* local sizes / meta */
//     JXF_Int num_rows_diag_A = jxf_CSRMatrixNumRows(A_diag);
//     JXF_Int num_cols_diag_A = jxf_CSRMatrixNumCols(A_diag);
//     JXF_Int num_cols_offd_A = jxf_CSRMatrixNumCols(A_offd);
//     JXF_Int first_col_diag_A = jxf_ParCSRMatrixFirstColDiag(A);
//     JXF_Int last_col_diag_A = first_col_diag_A + num_cols_diag_A - 1;

//     JXF_Int first_col_diag_B = jxf_ParCSRMatrixFirstColDiag(B);
//     JXF_Int num_cols_diag_B = jxf_CSRMatrixNumCols(B_diag);
//     JXF_Int last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

//     jxf_CSRMatrix *Bs_ext = NULL;
//     JXF_Real *Bs_ext_data = NULL;
//     JXF_Int *Bs_ext_i = NULL;
//     JXF_Int *Bs_ext_j = NULL;
//     JXF_Int *B_ext_diag_i = NULL, *B_ext_offd_i = NULL;
//     JXF_Int *B_ext_diag_j = NULL, *B_ext_offd_j = NULL;
//     JXF_Real *B_ext_diag_data = NULL, *B_ext_offd_data = NULL;
//     JXF_Int B_ext_diag_size = 0, B_ext_offd_size = 0;

//     // /* For create C later (not used here, since C arrays provided) */
//     // JXF_Int *row_starts_A = jxf_ParCSRMatrixRowStarts(A);

//     if (num_procs > 1)
//     {
//         /*---------------------------------------------------------------------
//          * If there exists no CommPkg for A, a CommPkg is generated using
//          * equally load balanced partitionings within
//          * jxf_ParCSRMatrixExtractBExt
//          *--------------------------------------------------------------------*/
//         Bs_ext = jxf_ParCSRMatrixExtractBExt(B, A, 1);
//         Bs_ext_data = jxf_CSRMatrixData(Bs_ext);
//         Bs_ext_i = jxf_CSRMatrixI(Bs_ext);
//         Bs_ext_j = jxf_CSRMatrixJ(Bs_ext);
//     }
//     B_ext_diag_i = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
//     B_ext_offd_i = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
//     B_ext_diag_size = 0;
//     B_ext_offd_size = 0;

//     // printf("1\n");

//     #if JXF_USING_OPENMP
//     #pragma omp parallel
//     #endif
//     {
//         JXF_Int size, rest, ii;
//         JXF_Int ns, ne;
//         JXF_Int i1, i, j;
//         JXF_Int my_offd_size, my_diag_size;
//         JXF_Int cnt_offd, cnt_diag;

//         JXF_Int num_threads = jxf_NumActiveThreads();

//         size = num_cols_offd_A / num_threads;
//         rest = num_cols_offd_A - size * num_threads;
//         ii = jxf_GetThreadNum();
//         if (ii < rest)
//         {
//             ns = ii * size + ii;
//             ne = (ii + 1) * size + ii + 1;
//         }
//         else
//         {
//             ns = ii * size + rest;
//             ne = (ii + 1) * size + rest;
//         }

//         my_diag_size = 0;
//         my_offd_size = 0;
//         for (i = ns; i < ne; i++)
//         {
//             B_ext_diag_i[i] = my_diag_size;
//             B_ext_offd_i[i] = my_offd_size;
//             for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
//             {
//                 if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
//                 {
//                     my_offd_size++;
//                 }
//                 else
//                 {
//                     my_diag_size++;
//                 }
//             }
//         }
//         my_diag_array[ii] = my_diag_size;
//         my_offd_array[ii] = my_offd_size;

//         #if JXF_USING_OPENMP
//         #pragma omp barrier
//         #endif

//         if (ii)
//         {
//             my_diag_size = my_diag_array[0];
//             my_offd_size = my_offd_array[0];
//             for (i1 = 1; i1 < ii; i1++)
//             {
//                 my_diag_size += my_diag_array[i1];
//                 my_offd_size += my_offd_array[i1];
//             }

//             for (i1 = ns; i1 < ne; i1++)
//             {
//                 B_ext_diag_i[i1] += my_diag_size;
//                 B_ext_offd_i[i1] += my_offd_size;
//             }
//         }
//         else
//         {
//             B_ext_diag_size = 0;
//             B_ext_offd_size = 0;
//             for (i1 = 0; i1 < num_threads; i1++)
//             {
//                 B_ext_diag_size += my_diag_array[i1];
//                 B_ext_offd_size += my_offd_array[i1];
//             }
//             B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
//             B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

//             if (B_ext_diag_size)
//             {
//                 B_ext_diag_j = jxf_CTAlloc(JXF_Int, B_ext_diag_size);
//                 B_ext_diag_data = jxf_CTAlloc(JXF_Real, B_ext_diag_size);
//             }
//             if (B_ext_offd_size)
//             {
//                 B_ext_offd_j = jxf_CTAlloc(JXF_Int, B_ext_offd_size);
//                 B_ext_offd_data = jxf_CTAlloc(JXF_Real, B_ext_offd_size);
//             }
//         }
        
//         #if JXF_USING_OPENMP
//         #pragma omp barrier
//         #endif

//         cnt_offd = B_ext_offd_i[ns];
//         cnt_diag = B_ext_diag_i[ns];
//         for (i = ns; i < ne; i++)
//         {
//             for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
//                 if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
//                 {
//                     B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
//                     B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
//                 }
//                 else
//                 {
//                     B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
//                     B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
//                 }
//         }

//     #if JXF_USING_OPENMP
//     #pragma omp barrier
//     #endif
//     }

//     JXF_Int ii1, jj1, jj2, jj3, jj4;
//     JXF_Int i2;
//     JXF_Real a_entry;
//     JXF_Int target_col;
//     JXF_Int global_col;

//     memcpy(C_diag_data, A_diag_data, A_diag_nnz * sizeof(JXF_Real));
//     memcpy(C_offd_data, A_offd_data, A_offd_nnz * sizeof(JXF_Real));

//     /* offd part */
//     if (num_cols_offd_A)
//     {
//         for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
//         {
//             for (jj2 = L_offd_i[ii1]; jj2 < L_offd_i[ii1 + 1]; jj2++)
//             {
//                 i2 = L_offd_j[jj2];
//                 for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
//                 {
//                     for (jj1 = A_offd_i[ii1]; jj1 < A_offd_i[ii1 + 1]; jj1++)
//                     {
//                         target_col = A_offd_j[jj1];
//                         global_col = col_map_offd_A[target_col];
//                         if(B_ext_offd_j[jj3] == global_col)
//                         {
//                             C_offd_data[jj1] -= L_offd_data[jj2] * B_ext_offd_data[jj3];
//                         }
//                     }
//                 }
//             }

//         // L_diag * B_offd
//             for (jj2 = L_diag_i[ii1]; jj2 < L_diag_i[ii1 + 1]; jj2++)
//             {
//                 i2 = L_diag_j[jj2];
//                 for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
//                 {
//                     for (jj1 = A_offd_i[ii1]; jj1 < A_offd_i[ii1 + 1]; jj1++)
//                     {
//                         if(B_offd_j[jj3] == A_offd_j[jj1])
//                         {
//                             C_offd_data[jj1] -= L_diag_data[jj2] * B_offd_data[jj3];
//                         }
//                     }
//                 }
//             }
//         }

//         for (jj2 = L_offd_i[ii1]; jj2 < L_offd_i[ii1 + 1]; jj2++)
//             {
//                 i2 = L_offd_j[jj2];
//                 for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
//                 {
//                     target_col = B_ext_diag_j[jj3];
//                     for (jj1 = A_diag_i[ii1]; jj1 < A_diag_i[ii1 + 1]; jj1++)
//                     {
//                         if(A_diag_j[jj1] == target_col)
//                         {
//                             C_diag_data[jj1] -= L_offd_data[jj2] * B_ext_diag_data[jj3];
//                         }
//                     }
//                 }
//             }
//     }

//     // L_offd * B_ext_diag
//         for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
//         {
//             // L_diag * B_diag
//             for (jj2 = L_diag_i[ii1]; jj2 < L_diag_i[ii1 + 1]; jj2++)
//             {
//                 i2 = L_diag_j[jj2];
//                 for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
//                 {
//                     target_col = B_diag_j[jj3];
//                     for (jj1 = A_diag_i[ii1]; jj1 < A_diag_i[ii1 + 1]; jj1++)
//                     {
//                         if(A_diag_j[jj1] == target_col)
//                         {
//                             C_diag_data[jj1] -= L_diag_data[jj2] * B_diag_data[jj3];
//                         }
//                     }
//                 }
//             }
//         }

//     if (num_procs > 1)
//     {
//     jxf_CSRMatrixDestroy(Bs_ext);
//     Bs_ext = NULL;
//     }
    
//     jxf_TFree(B_ext_diag_i);
//     if (B_ext_diag_size)
//     {
//     jxf_TFree(B_ext_diag_j);
//     jxf_TFree(B_ext_diag_data);
//     }
//     jxf_TFree(B_ext_offd_i);
//     if (B_ext_offd_size)
//     {
//     jxf_TFree(B_ext_offd_j);
//     jxf_TFree(B_ext_offd_data);
//     }

//     jxf_TFree(my_diag_array);
//     jxf_TFree(my_offd_array);
//     return;
// }



JXF_Int jxf_ILUSetupFGPBILU_v3(jxf_ParBSRMatrix *A,
                            JXF_Int sweep,
                            jxf_ParBSRMatrix **Lptr,
                            jxf_ParVector **D,
                            jxf_ParBSRMatrix **Uptr,
                            jxf_ParBSRMatrix **Sptr,
                            JXF_Int **u_end)
{
    MPI_Comm comm = jxf_ParBSRMatrixComm(A);
    JXF_Int num_procs, my_id;
    jxf_ParCSRCommPkg *comm_pkg = NULL;

    /* MPI initialization */
    jxf_MPI_Comm_size(comm, &num_procs);
    jxf_MPI_Comm_rank(comm, &my_id);

    comm_pkg = jxf_ParBSRMatrixCommPkg(A);
    if (!comm_pkg)
    {
        jxf_BlockMatvecCommPkgCreate(A);
        comm_pkg = jxf_ParBSRMatrixCommPkg(A);
    }

    JXF_Int i, j, k;
    JXF_Int ROW_OMP, temp_idx;
    JXF_Int nthreads = 1;
    JXF_Real local_nnz = 0.0, total_nnz = 0.0;

    /* Data objects for A */
    jxf_BSRMatrix *A_diag = jxf_ParBSRMatrixDiag(A);
    JXF_Real *A_diag_data = jxf_BSRMatrixData(A_diag);
    JXF_Int *A_diag_i = jxf_BSRMatrixI(A_diag);
    JXF_Int *A_diag_j = jxf_BSRMatrixJ(A_diag);
    JXF_Int n = jxf_BSRMatrixNumRows(A_diag);
    JXF_Int nnz_A = A_diag_i[n];

    JXF_Int first_col_diag_A = jxf_ParBSRMatrixFirstColDiag(A);
    JXF_Int num_cols_diag_A = jxf_BSRMatrixNumCols(A_diag);
    JXF_Int last_col_diag_A = first_col_diag_A + num_cols_diag_A - 1;
    JXF_Int *col_map_offd_A = jxf_ParBSRMatrixColMapOffd(A);
    // JXF_Int L_offd_cols =  0;
    // JXF_Int U_offd_cols =  0;

    /* Data objects for L, D, U */
    jxf_ParBSRMatrix *matL = NULL, *matU = NULL;
    jxf_BSRMatrix *L_diag = NULL, *U_diag = NULL;
    JXF_Real *D_data = NULL;
    JXF_Real *d_data = NULL;
    JXF_Real *L_diag_data = NULL, *U_diag_data = NULL;
    JXF_Int *L_diag_i = NULL, *L_diag_j = NULL;
    JXF_Int *U_diag_i = NULL, *U_diag_j = NULL;
    JXF_Int *workL = NULL, *workU = NULL;
    JXF_Int *workLO = NULL, *workUO = NULL;  //add
    JXF_Int LD_nnz = 0, UD_nnz = 0;

    /* temporaries for C = A - L*U */
    JXF_Real *C_diag_data = NULL;

    JXF_Int nt = 1;
    JXF_Int global_row, global_col;

    /* Data objects for A */
    jxf_BSRMatrix *A_offd = jxf_ParBSRMatrixOffd(A);
    JXF_Real *A_offd_data = jxf_BSRMatrixData(A_offd);
    JXF_Int *A_offd_i = jxf_BSRMatrixI(A_offd);
    JXF_Int *A_offd_j = jxf_BSRMatrixJ(A_offd);
    JXF_Int nnz_AO = A_offd_i[n];

    JXF_Int num_cols_offd_A =  jxf_BSRMatrixNumCols(A_offd);
    JXF_Int L_offd_cols =  jxf_BSRMatrixNumCols(A_offd);
    JXF_Int U_offd_cols =  jxf_BSRMatrixNumCols(A_offd);

    jxf_BSRMatrix *L_offd = NULL, *U_offd = NULL;
    JXF_Real *L_offd_data = NULL, *U_offd_data = NULL;
    JXF_Int *L_offd_i = NULL, *L_offd_j = NULL;
    JXF_Int *U_offd_i = NULL, *U_offd_j = NULL;
    JXF_Int LO_nnz = 0, UO_nnz = 0;

    JXF_Real *C_offd_data = NULL;

    JXF_Int block_size = jxf_BSRMatrixBlockSize(A_diag);
    JXF_Int bnnz = block_size * block_size;

    C_diag_data = jxf_TAlloc(JXF_Real, nnz_A * bnnz);
    C_offd_data = jxf_TAlloc(JXF_Real, nnz_AO * bnnz);

    /* Thread info */
    nt = jxf_NumActiveThreads();
    if (nt <= 0)
        nt = 1;
    nthreads = nt;

    ROW_OMP = (n + nt - 1) / nt;

    /* allocate prefix arrays (per-thread) */
    workL = jxf_TAlloc(JXF_Int, nt + 1);
    workU = jxf_TAlloc(JXF_Int, nt + 1);
    workLO = jxf_TAlloc(JXF_Int, nt + 1);
    workUO = jxf_TAlloc(JXF_Int, nt + 1);
    if (!workL || !workU)
    {
        fprintf(stderr, "Memory alloc failure workL/workU\n");
        jxf_TFree(workL);
        jxf_TFree(workU);
        return -1;
    }
    for (i = 0; i <= nt; i++)
    {
        workL[i] = 0;
        workU[i] = 0;
    }

/* 1) Count diag (local) lower/upper nnz per-thread */
#pragma omp parallel num_threads(nt)
    {
        JXF_Int tid = jxf_GetThreadNum();
        JXF_Int istart = tid * ROW_OMP;
        JXF_Int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
        JXF_Int localL = 0, localU = 0;
        for (JXF_Int r = istart; r < iend && r < n; r++)
        {
            for (JXF_Int jj = A_diag_i[r]; jj < A_diag_i[r + 1]; jj++)
            {
                JXF_Int col = A_diag_j[jj];
                if (col < r)
                    localL++;
                else if (col > r)
                    localU++;
            }
        }
        workL[tid + 1] = localL;
        workU[tid + 1] = localU;
#pragma omp barrier
#pragma omp single
        {
            for (JXF_Int t = 1; t <= nt; t++)
            {
                workL[t] += workL[t - 1];
                workU[t] += workU[t - 1];
            }
        }
    }

    LD_nnz = workL[nt];
    UD_nnz = workU[nt];

    // /* allocate diag arrays */
    
    // D_data = jxf_TAlloc(JXF_Real, n);
    // d_data = jxf_TAlloc(JXF_Real, n);

    // L_diag_i = jxf_TAlloc(JXF_Int, n + 1);
    // U_diag_i = jxf_TAlloc(JXF_Int, n + 1);
    // L_diag_j = jxf_TAlloc(JXF_Int, LD_nnz);
    // U_diag_j = jxf_TAlloc(JXF_Int, UD_nnz);
    // L_diag_data = jxf_TAlloc(JXF_Real, LD_nnz);
    // U_diag_data = jxf_TAlloc(JXF_Real, UD_nnz);


    /* allocate diag arrays (BSR format) */
    /* D_data: one inverse block per block row, each block has bnnz elements */
    D_data = jxf_TAlloc(JXF_Real, n * bnnz);
    d_data = jxf_TAlloc(JXF_Real, n * bnnz);

    /* L_diag_i / U_diag_i: block row pointers, length = block_rows + 1 */
    L_diag_i = jxf_TAlloc(JXF_Int, n + 1);
    U_diag_i = jxf_TAlloc(JXF_Int, n + 1);

    /* L_diag_j / U_diag_j: block column indices, length = number of nonzero blocks */
    L_diag_j = jxf_TAlloc(JXF_Int, LD_nnz);
    U_diag_j = jxf_TAlloc(JXF_Int, UD_nnz);

    /* L_diag_data / U_diag_data: each nonzero block has bnnz elements */
    L_diag_data = jxf_TAlloc(JXF_Real, LD_nnz * bnnz);
    U_diag_data = jxf_TAlloc(JXF_Real, UD_nnz * bnnz);

// C_diag_data = jxf_TAlloc(JXF_Real, nnz_A);

/* fill diag index/data from A (split into L/U and diagonal) */
#pragma omp parallel num_threads(nt)
    {
        int tid = jxf_GetThreadNum();
        int istart = tid * ROW_OMP;
        int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
        int cntL = 0, cntU = 0;
        for (int r = istart; r < iend && r < n; r++)
        {
            L_diag_i[r] = workL[tid] + cntL;
            U_diag_i[r] = workU[tid] + cntU;
            for (int jj = A_diag_i[r]; jj < A_diag_i[r + 1]; jj++)
            {
                JXF_Int col = A_diag_j[jj];
                JXF_Real* A_blk = A_diag_data + jj * bnnz;
                if (col < r)
                {
                    L_diag_j[workL[tid] + cntL] = col;
                    JXF_Real* L_blk = L_diag_data + (workL[tid] + cntL) * bnnz;
                    JXF_Real* D_inv_col = D_data + col * bnnz;
                    jxf_bsr_block_matmul(A_blk, D_inv_col, L_blk, block_size);
                    cntL++;
                }
                else if (col > r)
                {
                    U_diag_j[workU[tid] + cntU] = col;
                    JXF_Real* U_blk = U_diag_data + (workU[tid] + cntU) * bnnz;
                    jxf_bsr_block_copy(A_blk, U_blk, 1.0, block_size);
                    cntU++;
                }
                else
                {
                    JXF_Real* diag_blk = d_data + r * bnnz;
                    jxf_bsr_block_copy(A_blk, diag_blk, 1.0, block_size);
                    /* guard diagonal */
                    JXF_Real det = jxf_bsr_block_det(diag_blk, block_size);
                    if (fabs(det) < MAT_TOL) {
                        memset(diag_blk, 0, bnnz * sizeof(JXF_Real));
                        for (int k = 0; k < block_size; k++) diag_blk[k * block_size + k] = 1.0;
                    }
                    JXF_Real* D_blk = D_data + r * bnnz;
                    jxf_bsr_block_copy(diag_blk, D_blk, 1.0, block_size);
                    jxf_bsr_block_inv(D_blk, block_size);
                }
            }
        }
    }
    
    L_diag_i[n] = LD_nnz;
    U_diag_i[n] = UD_nnz;
// /* fill diag index/data from A (split into L/U and diagonal) */
// #pragma omp parallel num_threads(nt)
//     {
//         int tid = jxf_GetThreadNum();
//         int istart = tid * ROW_OMP;
//         int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
//         int cntL = 0, cntU = 0;
//         for (int r = istart; r < iend && r < n; r++)
//         {
//             L_diag_i[r] = workL[tid] + cntL;
//             U_diag_i[r] = workU[tid] + cntU;
//             for (int jj = A_diag_i[r]; jj < A_diag_i[r + 1]; jj++)
//             {
//                 JXF_Int col = A_diag_j[jj];
//                 JXF_Real aval = A_diag_data[jj];
//                 if (col < r)
//                 {
//                     L_diag_j[workL[tid] + cntL] = col;
//                     L_diag_data[workL[tid] + cntL] = D_data[col] * aval;
//                     cntL++;
//                 }
//                 else if (col > r)
//                 {
//                     U_diag_j[workU[tid] + cntU] = col;
//                     U_diag_data[workU[tid] + cntU] = aval;
//                     cntU++;
//                 }
//                 else
//                 {
//                     d_data[r] = aval;
//                     /* guard diagonal */
//                     if (fabs(d_data[r]) < MAT_TOL)
//                         d_data[r] = (JXF_Real)1e-6;
//                     D_data[r] = (JXF_Real)1.0 / d_data[r];
//                 }
//             }
//         }
//     }
    if (num_procs > 1)
    {
        /* reset work arrays for offd counting */
        for (i = 0; i <= nt; i++)
        {
            workLO[i] = 0;
            workUO[i] = 0;
        }

/* 2) Count offd lower/upper nnz per-thread (use global compare) */
#pragma omp parallel num_threads(nt)
        {
            int tid = jxf_GetThreadNum();
            int istart = tid * ROW_OMP;
            int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
            int localL = 0, localU = 0;
            for (int r = istart; r < iend && r < n; r++)
            {
                JXF_Int global_row_idx = r + first_col_diag_A;
                for (int jj = A_offd_i[r]; jj < A_offd_i[r + 1]; jj++)
                {
                    JXF_Int offd_col_idx = A_offd_j[jj]; /* index into col_map_offd_A */
                    JXF_Int global_col_idx = col_map_offd_A[offd_col_idx];
                    if (global_col_idx < global_row_idx)
                        localL++;
                    else if (global_col_idx > global_row_idx)
                        localU++;
                }
            }
            workLO[tid + 1] = localL;
            workUO[tid + 1] = localU;
#pragma omp barrier
#pragma omp single
            {
                for (int t = 1; t <= nt; t++)
                {
                    workLO[t] += workLO[t - 1];
                    workUO[t] += workUO[t - 1];
                }
            }
        }

        LO_nnz = workLO[nt];
        UO_nnz = workUO[nt];

        /* allocate offd arrays */
        L_offd_i = jxf_TAlloc(JXF_Int, n + 1);
        U_offd_i = jxf_TAlloc(JXF_Int, n + 1);
        L_offd_j = jxf_TAlloc(JXF_Int, LO_nnz);
        U_offd_j = jxf_TAlloc(JXF_Int, UO_nnz);
        L_offd_data = jxf_TAlloc(JXF_Real, LO_nnz * bnnz);
        U_offd_data = jxf_TAlloc(JXF_Real, UO_nnz * bnnz);
        // L_offd_data = jxf_TAlloc(JXF_Real, LO_nnz);
        // U_offd_data = jxf_TAlloc(JXF_Real, UO_nnz);

// C_offd_data = jxf_TAlloc(JXF_Real, nnz_A);

/* fill offd structures from A */
#pragma omp parallel num_threads(nt)
        {
            int tid = jxf_GetThreadNum();
            int istart = tid * ROW_OMP;
            int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
            int cntL = 0, cntU = 0;
            for (int r = istart; r < iend && r < n; r++)
            {
                L_offd_i[r] = workLO[tid] + cntL;
                U_offd_i[r] = workUO[tid] + cntU;
                JXF_Int global_row_idx = r + first_col_diag_A;
                for (int jj = A_offd_i[r]; jj < A_offd_i[r + 1]; jj++)
                {
                    JXF_Int offd_idx = A_offd_j[jj];
                    JXF_Int global_col_idx = col_map_offd_A[offd_idx];
                    JXF_Real* A_offd_blk = A_offd_data + jj * bnnz;
                    if (global_col_idx < global_row_idx)
                    {
                        L_offd_j[workLO[tid] + cntL] = offd_idx;
                        JXF_Real* L_offd_blk = L_offd_data + (workLO[tid] + cntL) * bnnz;
                        jxf_bsr_block_copy(A_offd_blk, L_offd_blk, 1.0, block_size);
                        cntL++;
                    }
                    else
                    {
                        U_offd_j[workUO[tid] + cntU] = offd_idx;
                        JXF_Real* U_offd_blk = U_offd_data + (workUO[tid] + cntU) * bnnz;
                        jxf_bsr_block_copy(A_offd_blk, U_offd_blk, 1.0, block_size);
                        cntU++;
                    }
                }
            }
        }
        L_offd_i[n] = LO_nnz;
        U_offd_i[n] = UO_nnz;
    }

    /* create ParCSR matrices matL and matU (structure only) */
//         L_offd_i[n] = LO_nnz;
//         U_offd_i[n] = UO_nnz;
//     }

    /* create ParBSR matrices matL and matU (structure only) */
    matL = jxf_ParBSRMatrixCreate(comm,
                                 jxf_ParBSRMatrixBlockSize(A),
                                 jxf_ParBSRMatrixGlobalNumRows(A),
                                 jxf_ParBSRMatrixGlobalNumCols(A),
                                 jxf_ParBSRMatrixRowStarts(A),
                                 jxf_ParBSRMatrixColStarts(A),
                                 L_offd_cols,
                                 LD_nnz,
                                 LO_nnz );
    /* attach diag/offd pointers for L */
    L_diag = jxf_ParBSRMatrixDiag(matL);
    jxf_BSRMatrixI(L_diag) = L_diag_i;
    jxf_BSRMatrixJ(L_diag) = L_diag_j;
    jxf_BSRMatrixData(L_diag) = L_diag_data;
    // jxf_CSRMatrixNumRows(L_diag); /* just ensure object exists */
    // /* offd */
    // jxf_ParCSRMatrixOffd(matL);       /* ensure offd exists if function provided */
    // jxf_ParCSRMatrixColMapOffd(matL); /* ensure colmap exists if needed */

    /* For U */
    matU = jxf_ParBSRMatrixCreate(comm,
                                 jxf_ParBSRMatrixBlockSize(A),
                                 jxf_ParBSRMatrixGlobalNumRows(A),
                                 jxf_ParBSRMatrixGlobalNumCols(A),
                                 jxf_ParBSRMatrixRowStarts(A),
                                 jxf_ParBSRMatrixColStarts(A),
                                 U_offd_cols,
                                 UD_nnz,
                                 UO_nnz );
    U_diag = jxf_ParBSRMatrixDiag(matU);
    jxf_BSRMatrixI(U_diag) = U_diag_i;
    jxf_BSRMatrixJ(U_diag) = U_diag_j;
    jxf_BSRMatrixData(U_diag) = U_diag_data;

    /* D_array stores inverse diagonal blocks: global size = block_rows * bnnz */
    JXF_Int global_block_rows = jxf_ParBSRMatrixGlobalNumRows(A);
    JXF_Int global_vec_size = global_block_rows * bnnz;
    jxf_ParVector *D_array = jxf_ParVectorCreate(jxf_ParBSRMatrixComm(A),
                                               global_vec_size,
                                               jxf_ParBSRMatrixRowStarts(A));
    jxf_ParVectorInitialize(D_array);
    jxf_Vector *D_local = jxf_ParVectorLocalVector(D_array);
    jxf_VectorData(D_local) = D_data;

    // if (num_procs > 1)
    // {
    //     jxf_CSRMatrix *L_offd = jxf_ParCSRMatrixOffd(matL);
    //     jxf_CSRMatrix *U_offd = jxf_ParCSRMatrixOffd(matU);
    //     jxf_CSRMatrixI(L_offd) = L_offd_i;
    //     jxf_CSRMatrixJ(L_offd) = L_offd_j;
    //     jxf_CSRMatrixData(L_offd) = L_offd_data;
    //     jxf_CSRMatrixI(U_offd) = U_offd_i;
    //     jxf_CSRMatrixJ(U_offd) = U_offd_j;
    //     jxf_CSRMatrixData(U_offd) = U_offd_data;
    // }

    if (num_procs > 1)
    {
        // JXF_Int num_cols_offd_A = jxf_ParCSRMatrixNumColsOffd(A);
        // JXF_Int *col_map_offd_A = jxf_ParCSRMatrixColMapOffd(A);

        /* offd 部分指针绑定 */
        jxf_BSRMatrix *L_offd = jxf_ParBSRMatrixOffd(matL);
        jxf_BSRMatrix *U_offd = jxf_ParBSRMatrixOffd(matU);

        jxf_BSRMatrixI(L_offd) = L_offd_i;
        jxf_BSRMatrixJ(L_offd) = L_offd_j;
        jxf_BSRMatrixData(L_offd) = L_offd_data;
        jxf_BSRMatrixNumCols(L_offd) = L_offd_cols;

        jxf_BSRMatrixI(U_offd) = U_offd_i;
        jxf_BSRMatrixJ(U_offd) = U_offd_j;
        jxf_BSRMatrixData(U_offd) = U_offd_data;
        jxf_BSRMatrixNumCols(U_offd) = U_offd_cols;

        /* col_map_offd 复制 */
        JXF_Int *col_map_offd_L = jxf_CTAlloc(JXF_Int, L_offd_cols);
        JXF_Int *col_map_offd_U = jxf_CTAlloc(JXF_Int, U_offd_cols);

        if (num_cols_offd_A > 0 && col_map_offd_A)
        {
            for (i = 0; i < num_cols_offd_A; i++)
            {
                col_map_offd_L[i] = col_map_offd_A[i];
                col_map_offd_U[i] = col_map_offd_A[i];
            }
        }
        jxf_ParCSRMatrixColMapOffd(matL) = col_map_offd_L;
        jxf_ParCSRMatrixColMapOffd(matU) = col_map_offd_U;

        /* 通信包创建 */
        jxf_BlockMatvecCommPkgCreate(matL);
        jxf_BlockMatvecCommPkgCreate(matU);
    }

    /* MAIN SWEEPS: ILU iterations */
    for (int sweep_it = 0; sweep_it < sweep; sweep_it++)
    {
        /* Step 1: Compute C_diag = A_diag - L_diag * d_data * U_diag at block level.
           The path product now includes the diagonal block D[k] = d_data[k]
           between L and U: C[r][c] -= L[r][k] * d_data[k] * U[k][c].
           This makes the sweep algebraically consistent with the
            reconstruction (I+L)*(D+U). */
#pragma omp parallel num_threads(nt)
        {
            int tid = jxf_GetThreadNum();
            int istart = tid * ROW_OMP;
            int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
            for (int r = istart; r < iend && r < n; r++)
            {
                for (int jj = A_diag_i[r]; jj < A_diag_i[r + 1]; jj++)
                {
                    JXF_Int col = A_diag_j[jj];
                    JXF_Real *C_blk = C_diag_data + jj * bnnz;
                    JXF_Real *A_blk = A_diag_data + jj * bnnz;
                    jxf_bsr_block_copy(A_blk, C_blk, 1.0, block_size);
                    for (int ljj = L_diag_i[r]; ljj < L_diag_i[r + 1]; ljj++)
                    {
                        JXF_Int k = L_diag_j[ljj];
                        for (int ujj = U_diag_i[k]; ujj < U_diag_i[k + 1]; ujj++)
                        {
                            if (U_diag_j[ujj] == col)
                            {
                                JXF_Real *L_blk = L_diag_data + ljj * bnnz;
                                JXF_Real *U_blk = U_diag_data + ujj * bnnz;
                                JXF_Real temp[25];
                                jxf_bsr_block_matmul(L_blk, U_blk, temp, block_size);
                                for (int b = 0; b < bnnz; b++)
                                    C_blk[b] -= temp[b];
                                break;
                            }
                        }
                    }
                }
            }
        }

        /* Step 2: Update L, D, U from C at block level.
           For each row r:
             1) Process diagonal first → compute D[r] = C[r][r], D_inv[r] = inv(D[r])
             2) Process lower (L[r][c] = C[r][c] * D_inv[c]) and
                upper (U[r][c] = C[r][c] (copy, no D_inv scaling)) */
#pragma omp parallel num_threads(nt)
        {
            int tid = jxf_GetThreadNum();
            int istart = tid * ROW_OMP;
            int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
            for (int r = istart; r < iend && r < n; r++)
            {
                /* --- Pass 1: find and process diagonal --- */
                JXF_Int diag_jj = -1;
                for (int jj = A_diag_i[r]; jj < A_diag_i[r + 1]; jj++)
                {
                    if (A_diag_j[jj] == r)
                    {
                        diag_jj = jj;
                        break;
                    }
                }
                if (diag_jj >= 0)
                {
                    JXF_Real *C_blk = C_diag_data + diag_jj * bnnz;
                    JXF_Real *diag_blk = d_data + r * bnnz;
                    jxf_bsr_block_copy(C_blk, diag_blk, 1.0, block_size);
                    JXF_Real det = jxf_bsr_block_det(diag_blk, block_size);
                    if (fabs(det) < MAT_TOL)
                    {
                        for (int b = 0; b < bnnz; b++)
                            diag_blk[b] = 0.0;
                        for (int kk = 0; kk < block_size; kk++)
                            diag_blk[kk * block_size + kk] = (JXF_Real)1.0;
                    }
                    JXF_Real *D_blk = D_data + r * bnnz;
                    jxf_bsr_block_copy(diag_blk, D_blk, 1.0, block_size);
                    jxf_bsr_block_inv(D_blk, block_size);
                }

                /* --- Pass 2: lower (col<r) and upper (col>r) --- */
                int localLpos = L_diag_i[r];
                int localUpos = U_diag_i[r];
                for (int jj = A_diag_i[r]; jj < A_diag_i[r + 1]; jj++)
                {
                    JXF_Int col = A_diag_j[jj];
                    JXF_Real *C_blk = C_diag_data + jj * bnnz;
                    if (col < r)
                    {
                        JXF_Real *L_blk = L_diag_data + localLpos * bnnz;
                        JXF_Real *D_col = D_data + col * bnnz;
                        jxf_bsr_block_matmul(C_blk, D_col, L_blk, block_size);
                        localLpos++;
                    }
                    else if (col > r)
                    {
                        JXF_Real *U_blk = U_diag_data + localUpos * bnnz;
                        jxf_bsr_block_copy(C_blk, U_blk, 1.0, block_size);
                        localUpos++;
                    }
                }
            }
        }

        if (num_procs > 1)
        {
            /* Step 3: Compute C_offd = A_offd (block copy) and split into L_offd/U_offd */
#pragma omp parallel num_threads(nt)
            {
                int tid = jxf_GetThreadNum();
                int istart = tid * ROW_OMP;
                int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
                for (int r = istart; r < iend && r < n; r++)
                {
                    for (int jj = A_offd_i[r]; jj < A_offd_i[r + 1]; jj++)
                    {
                        JXF_Real *C_blk = C_offd_data + jj * bnnz;
                        JXF_Real *A_blk = A_offd_data + jj * bnnz;
                        jxf_bsr_block_copy(A_blk, C_blk, 1.0, block_size);
                    }
                    int baseL = L_offd_i[r];
                    int baseU = U_offd_i[r];
                    JXF_Int global_row_idx = r + first_col_diag_A;
                    for (int jj = A_offd_i[r]; jj < A_offd_i[r + 1]; jj++)
                    {
                        JXF_Int col = A_offd_j[jj];
                        JXF_Int gcol = col_map_offd_A[col];
                        JXF_Real *C_blk = C_offd_data + jj * bnnz;
                        if (gcol > global_row_idx)
                        {
                            JXF_Real *U_blk = U_offd_data + baseU * bnnz;
                            jxf_bsr_block_copy(C_blk, U_blk, 1.0, block_size);
                            baseU++;
                        }
                        else if (gcol < global_row_idx)
                        {
                            JXF_Real *L_blk = L_offd_data + baseL * bnnz;
                            jxf_bsr_block_copy(C_blk, L_blk, 1.0, block_size);
                            baseL++;
                        }
                    }
                }
            }
            /* Note: the original CSR version called jxf_ParCSRMatrixM(matL, D_array)
               here, which was a parallel matvec overwriting D_array. For BSR this is
               incorrect (D_array != L * D_inv), and the comm package is unchanged
               during sweeps (ILU(0) has fixed sparsity), so we skip it. */
        }

    } /* end sweeps */
        // char FileNameCoaMat[256];
        // // // jxf_sprintf(FileNameCoaMat, "A_CSR_%d", num_procs);
        // // // jxf_ParCSRMatrixPrint(A, FileNameCoaMat);
        // jxf_sprintf(FileNameCoaMat, "L_CSR_%d", num_procs);
        // jxf_ParCSRMatrixPrint(matL, FileNameCoaMat);
        // jxf_sprintf(FileNameCoaMat, "U_CSR_%d", num_procs);
        // jxf_ParCSRMatrixPrint(matU, FileNameCoaMat);

        // if(my_id == 0 )
        // {
        //     for (JXF_Int i = 0; i < n ; i++) 
        //     {
        //     printf("%lf ", D_data[i]);
        //     printf("\n");
        //     }
        //     for (JXF_Int i = 0; i < nnz_AO ; i++) 
        //     {
        //     printf("%lf ", C_offd_data[i]);
        //     printf("\n");
        //     }
        //}



    /* compute and set global nnz counts for matU and matL (float stored) */
    local_nnz = (JXF_Real)(UD_nnz + UO_nnz);
    jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
    jxf_ParBSRMatrixDNumNonzeros(matU) = total_nnz;

    local_nnz = (JXF_Real)(LD_nnz + LO_nnz);
    jxf_MPI_Allreduce(&local_nnz, &total_nnz, 1, JXF_MPI_REAL, MPI_SUM, comm);
    jxf_ParBSRMatrixDNumNonzeros(matL) = total_nnz;

    // jxf_ParVector *D_array = jxf_ParVectorCreate(jxf_ParCSRMatrixComm(A),
    //                                            jxf_ParCSRMatrixGlobalNumRows(A),
    //                                            jxf_ParCSRMatrixRowStarts(A));
    // jxf_ParVectorInitialize(D_array);

    // jxf_Vector *D_local = jxf_ParVectorLocalVector(D_array);
    // jxf_VectorData(D_local) = D_data;

    /* outputs */
    *Lptr = matL;
    // *Dptr = D_data;
    *Uptr = matU;
    *Sptr = NULL;
    *u_end = NULL;
    *D = D_array;

    /* free temporaries */
    jxf_TFree(workL);
    workL = NULL;
    jxf_TFree(workU);
    workU = NULL;
    jxf_TFree(C_diag_data);
    C_diag_data = NULL;
    jxf_TFree(C_offd_data);
    C_offd_data = NULL;
    jxf_TFree(d_data);

    /* don't free L/U arrays: returned via matL/matU ownership */

    return 0;
}


void jxf_ParBSRMatrixMatmul_Apattern(jxf_ParBSRMatrix *A, jxf_ParBSRMatrix *L, jxf_ParBSRMatrix *B,
                                    JXF_Real *C_diag_data, JXF_Real *C_offd_data)
{

    MPI_Comm comm = jxf_ParBSRMatrixComm(A);
    JXF_Int num_procs;
    jxf_MPI_Comm_size(comm, &num_procs);

    /* Basic pointers to A and B blocks */
    jxf_BSRMatrix *A_diag = jxf_ParBSRMatrixDiag(A);
    jxf_BSRMatrix *A_offd = jxf_ParBSRMatrixOffd(A);
    JXF_Int *A_diag_i = jxf_BSRMatrixI(A_diag);
    JXF_Int *A_diag_j = jxf_BSRMatrixJ(A_diag);
    JXF_Real *A_diag_data = jxf_BSRMatrixData(A_diag);
    JXF_Int *A_offd_i = jxf_BSRMatrixI(A_offd);
    JXF_Int *A_offd_j = jxf_BSRMatrixJ(A_offd);
    JXF_Real *A_offd_data = jxf_BSRMatrixData(A_offd);

    JXF_Int *col_map_offd_A = jxf_ParBSRMatrixColMapOffd(A);

    JXF_Int *my_diag_array;
    JXF_Int *my_offd_array;
    JXF_Int max_num_threads = jxf_NumThreads();
    my_diag_array = jxf_CTAlloc(JXF_Int, max_num_threads);
    my_offd_array = jxf_CTAlloc(JXF_Int, max_num_threads);

    jxf_BSRMatrix *B_diag = jxf_ParBSRMatrixDiag(B);
    jxf_BSRMatrix *B_offd = jxf_ParBSRMatrixOffd(B);
    JXF_Int *B_diag_i = jxf_BSRMatrixI(B_diag);
    JXF_Int *B_diag_j = jxf_BSRMatrixJ(B_diag);
    JXF_Real *B_diag_data = jxf_CSRMatrixData(B_diag);
    JXF_Int *B_offd_i = jxf_BSRMatrixI(B_offd);
    JXF_Int *B_offd_j = jxf_BSRMatrixJ(B_offd);
    JXF_Real *B_offd_data = jxf_BSRMatrixData(B_offd);

    jxf_BSRMatrix *L_diag = jxf_ParBSRMatrixDiag(L);
    jxf_BSRMatrix *L_offd = jxf_ParBSRMatrixOffd(L);
    JXF_Int *L_diag_i = jxf_BSRMatrixI(L_diag);
    JXF_Int *L_diag_j = jxf_BSRMatrixJ(L_diag);
    JXF_Real *L_diag_data = jxf_BSRMatrixData(L_diag);
    JXF_Int *L_offd_i = jxf_BSRMatrixI(L_offd);
    JXF_Int *L_offd_j = jxf_BSRMatrixJ(L_offd);
    JXF_Real *L_offd_data = jxf_BSRMatrixData(L_offd);

    /* local sizes / meta */
    JXF_Int num_rows_diag_A = jxf_BSRMatrixNumRows(A_diag);
    JXF_Int num_cols_diag_A = jxf_BSRMatrixNumCols(A_diag);
    JXF_Int num_cols_offd_A = jxf_BSRMatrixNumCols(A_offd);
    JXF_Int first_col_diag_A = jxf_ParBSRMatrixFirstColDiag(A);
    JXF_Int last_col_diag_A = first_col_diag_A + num_cols_diag_A - 1;

    JXF_Int first_col_diag_B = jxf_ParBSRMatrixFirstColDiag(B);
    JXF_Int num_cols_diag_B = jxf_BSRMatrixNumCols(B_diag);
    JXF_Int last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

    jxf_BSRMatrix *Bs_ext = NULL;
    JXF_Real *Bs_ext_data = NULL;
    JXF_Int *Bs_ext_i = NULL;
    JXF_Int *Bs_ext_j = NULL;
    JXF_Int *B_ext_diag_i = NULL, *B_ext_offd_i = NULL;
    JXF_Int *B_ext_diag_j = NULL, *B_ext_offd_j = NULL;
    JXF_Real *B_ext_diag_data = NULL, *B_ext_offd_data = NULL;
    JXF_Int B_ext_diag_size = 0, B_ext_offd_size = 0;

    // /* For create C later (not used here, since C arrays provided) */
    // JXF_Int *row_starts_A = jxf_ParCSRMatrixRowStarts(A);
    if (num_procs > 1)
    {
        /*---------------------------------------------------------------------
         * If there exists no CommPkg for A, a CommPkg is generated using
         * equally load balanced partitionings within
         * jxf_ParBSRMatrixExtractBExt
         *--------------------------------------------------------------------*/
        Bs_ext = jxf_ParBSRMatrixExtractBExt(B, A, 1);
        Bs_ext_data = jxf_BSRMatrixData(Bs_ext);
        Bs_ext_i = jxf_BSRMatrixI(Bs_ext);
        Bs_ext_j = jxf_BSRMatrixJ(Bs_ext);
    }
    B_ext_diag_i = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
    B_ext_offd_i = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
    B_ext_diag_size = 0;
    B_ext_offd_size = 0;

    #if JXF_USING_OPENMP
    #pragma omp parallel
    #endif
    {
        JXF_Int size, rest, ii;
        JXF_Int ns, ne;
        JXF_Int i1, i, j;
        JXF_Int my_offd_size, my_diag_size;
        JXF_Int cnt_offd, cnt_diag;

        JXF_Int num_threads = jxf_NumActiveThreads();

        size = num_cols_offd_A / num_threads;
        rest = num_cols_offd_A - size * num_threads;
        ii = jxf_GetThreadNum();
        if (ii < rest)
        {
            ns = ii * size + ii;
            ne = (ii + 1) * size + ii + 1;
        }
        else
        {
            ns = ii * size + rest;
            ne = (ii + 1) * size + rest;
        }

        my_diag_size = 0;
        my_offd_size = 0;
        for (i = ns; i < ne; i++)
        {
            B_ext_diag_i[i] = my_diag_size;
            B_ext_offd_i[i] = my_offd_size;
            for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
            {
                if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
                {
                    my_offd_size++;
                }
                else
                {
                    my_diag_size++;
                }
            }
        }
        my_diag_array[ii] = my_diag_size;
        my_offd_array[ii] = my_offd_size;

        #if JXF_USING_OPENMP
        #pragma omp barrier
        #endif
        if (ii)
        {
            my_diag_size = my_diag_array[0];
            my_offd_size = my_offd_array[0];
            for (i1 = 1; i1 < ii; i1++)
            {
                my_diag_size += my_diag_array[i1];
                my_offd_size += my_offd_array[i1];
            }

            for (i1 = ns; i1 < ne; i1++)
            {
                B_ext_diag_i[i1] += my_diag_size;
                B_ext_offd_i[i1] += my_offd_size;
            }
        }
        else
        {
            B_ext_diag_size = 0;
            B_ext_offd_size = 0;
            for (i1 = 0; i1 < num_threads; i1++)
            {
                B_ext_diag_size += my_diag_array[i1];
                B_ext_offd_size += my_offd_array[i1];
            }
            B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
            B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

            if (B_ext_diag_size)
            {
                B_ext_diag_j = jxf_CTAlloc(JXF_Int, B_ext_diag_size);
                B_ext_diag_data = jxf_CTAlloc(JXF_Real, B_ext_diag_size);
            }
            if (B_ext_offd_size)
            {
                B_ext_offd_j = jxf_CTAlloc(JXF_Int, B_ext_offd_size);
                B_ext_offd_data = jxf_CTAlloc(JXF_Real, B_ext_offd_size);
            }
        }
        
        #if JXF_USING_OPENMP
        #pragma omp barrier
        #endif
        cnt_offd = B_ext_offd_i[ns];
        cnt_diag = B_ext_diag_i[ns];
        JXF_Int block_size = jxf_BSRMatrixBlockSize(A_diag);
        JXF_Int bnnz = block_size * block_size;
        for (i = ns; i < ne; i++)
        {
            for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
                if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
                {
                    B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
                    JXF_Real* src_blk = Bs_ext_data + j * bnnz;
                    JXF_Real* dst_blk = B_ext_offd_data + cnt_offd * bnnz;
                    for (int k = 0; k < bnnz; k++) dst_blk[k] = src_blk[k];
                    cnt_offd++;
                }
                else
                {
                    B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
                    JXF_Real* src_blk = Bs_ext_data + j * bnnz;
                    JXF_Real* dst_blk = B_ext_diag_data + cnt_diag * bnnz;
                    for (int k = 0; k < bnnz; k++) dst_blk[k] = src_blk[k];
                    cnt_diag++;
                }
        }
    #if JXF_USING_OPENMP
    #pragma omp barrier
    #endif
    }


    JXF_Int ii1, jj1, jj2, jj3, jj4;
    JXF_Int i2;
    JXF_Real a_entry;
    JXF_Int target_col;
    JXF_Int global_col;
    /* offd part */
    if (num_cols_offd_A)
    {
        JXF_Int block_size = jxf_BSRMatrixBlockSize(A_diag);
        JXF_Int bnnz = block_size * block_size;
        for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
        {
            for (jj1 = A_offd_i[ii1]; jj1 < A_offd_i[ii1 + 1]; jj1++)
            {
                jxf_bsr_block_copy(A_offd_data + jj1 * bnnz, C_offd_data + jj1 * bnnz, 1.0, block_size);
                target_col = A_offd_j[jj1];
                global_col = col_map_offd_A[target_col];

                // L_offd * B_ext_offd
                for (jj2 = L_offd_i[ii1]; jj2 < L_offd_i[ii1 + 1]; jj2++)
                {
                    i2 = L_offd_j[jj2];
                    for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
                    {
                        if (B_ext_offd_j[jj3] == global_col)
                        {
                            jxf_bsr_block_matmul_add(L_offd_data + jj2 * bnnz,
                                                     B_ext_offd_data + jj3 * bnnz,
                                                     -1.0,
                                                     C_offd_data + jj1 * bnnz,
                                                     block_size);
                        }
                    }
                }

                // L_diag * B_offd
                for (jj2 = L_diag_i[ii1]; jj2 < L_diag_i[ii1 + 1]; jj2++)
                {
                    i2 = L_diag_j[jj2];
                    for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
                    {
                        if (B_offd_j[jj3] == target_col)
                        {
                            jxf_bsr_block_matmul_add(L_diag_data + jj2 * bnnz,
                                                     B_offd_data + jj3 * bnnz,
                                                     -1.0,
                                                     C_offd_data + jj1 * bnnz,
                                                     block_size);
                        }
                    }
                }
            }
        }
    }
    /* diag part */
    for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
    {
        JXF_Int block_size = jxf_BSRMatrixBlockSize(A_diag);
        JXF_Int bnnz = block_size * block_size;
        for (jj1 = A_diag_i[ii1]; jj1 < A_diag_i[ii1 + 1]; jj1++)
        {
            jxf_bsr_block_copy(A_diag_data + jj1 * bnnz,
                               C_diag_data + jj1 * bnnz,
                               1.0, block_size);
            JXF_Int target_col = A_diag_j[jj1];

            if (num_cols_offd_A)
            {
                // L_offd * B_ext_diag
                for (jj2 = L_offd_i[ii1]; jj2 < L_offd_i[ii1 + 1]; jj2++)
                {
                    i2 = L_offd_j[jj2];
                    for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
                    {
                        if (B_ext_diag_j[jj3] == target_col)
                        {
                            jxf_bsr_block_matmul_add(L_offd_data + jj2 * bnnz,
                                                     B_ext_diag_data + jj3 * bnnz,
                                                     -1.0,
                                                     C_diag_data + jj1 * bnnz,
                                                     block_size);
                        }
                    }
                }
            }
            // L_diag * B_diag
            for (jj2 = L_diag_i[ii1]; jj2 < L_diag_i[ii1 + 1]; jj2++)
            {
                i2 = L_diag_j[jj2];
                for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
                {
                    if (B_diag_j[jj3] == target_col)
                    {
                        jxf_bsr_block_matmul_add(L_diag_data + jj2 * bnnz,
                                                 B_diag_data + jj3 * bnnz,
                                                 -1.0,
                                                 C_diag_data + jj1 * bnnz,
                                                 block_size);
                    }
                }
            }
        }
    }

    jxf_TFree(B_ext_diag_i);
        if (B_ext_diag_size)
        {
        jxf_TFree(B_ext_diag_j);
        jxf_TFree(B_ext_diag_data);
        }
    jxf_TFree(B_ext_offd_i);
    if (B_ext_offd_size)
        {
        jxf_TFree(B_ext_offd_j);
        jxf_TFree(B_ext_offd_data);
        }

    jxf_TFree(my_diag_array);
    jxf_TFree(my_offd_array);
    return;
}


JXF_Int
jxf_ParBSRMatrixM(  jxf_ParBSRMatrix *A,jxf_ParVector    *x ) //修改是否正确，需要进行确认
{
    jxf_ParCSRCommHandle **comm_handle;
    jxf_ParCSRCommPkg	*comm_pkg = jxf_ParBSRMatrixCommPkg(A);
    jxf_BSRMatrix         *diag     = jxf_ParBSRMatrixDiag(A);
    JXF_Int *diag_i = jxf_BSRMatrixI(diag);
    JXF_Int *diag_j = jxf_BSRMatrixJ(diag);
    JXF_Real *diag_data = jxf_BSRMatrixData(diag);
    JXF_Int n = jxf_BSRMatrixNumRows(diag);
    JXF_Int nt = jxf_NumThreads();
    JXF_Int ROW_OMP;
        
    jxf_BSRMatrix         *offd     = jxf_ParBSRMatrixOffd(A);
    JXF_Int *offd_i = jxf_BSRMatrixI(offd);
    JXF_Int *offd_j = jxf_BSRMatrixJ(offd);
    JXF_Real *offd_data = jxf_BSRMatrixData(offd);

    jxf_Vector            *x_local  = jxf_ParVectorLocalVector(x);   
    JXF_Int         num_rows = jxf_ParBSRMatrixGlobalNumRows(A);
    JXF_Int         num_cols = jxf_ParBSRMatrixGlobalNumCols(A);

    jxf_Vector     *x_tmp;
    JXF_Int        x_size = jxf_ParVectorGlobalSize(x);
    JXF_Int        num_vectors   = jxf_VectorNumVectors(x_local);
    JXF_Int	      num_cols_offd = jxf_BSRMatrixNumCols(offd);
    JXF_Int        ierr = 0;
    JXF_Int	      num_sends, i, j, jv, index, start;

    JXF_Int        vecstride = jxf_VectorVectorStride( x_local );
    JXF_Int        idxstride = jxf_VectorIndexStride( x_local );

    JXF_Real     *x_tmp_data, **x_buf_data;
    JXF_Real     *x_local_data = jxf_VectorData(x_local);

    JXF_Real      wall_time = 0.0;  /* for debugging instrumentation  */

    if (jxf__global_mvcpu_flag) wall_time = jxf_time_getWallclockSeconds();
    
    /*---------------------------------------------------------------------
    *  Check for size compatibility.  ParMatvec returns ierr = 11 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 12 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 13 if both are true.
    *
    *  Because temporary vectors are often used in ParMatvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    jxf_assert( idxstride > 0 );

    // jxf_assert( jxf_VectorNumVectors(y_local)==num_vectors );

    JXF_Int vec_per_offd_block = jxf_ParBSRMatrixBlockSize(A);
    if ( num_vectors == 1 )
    {
        x_tmp = jxf_SeqVectorCreate( num_cols_offd * vec_per_offd_block );
    }
    else
    {
        jxf_assert( num_vectors > 1 );
        x_tmp = jxf_SeqMultiVectorCreate( num_cols_offd * vec_per_offd_block, num_vectors );
    }
    jxf_SeqVectorInitialize(x_tmp);
    x_tmp_data = jxf_VectorData(x_tmp);

    comm_handle = jxf_CTAlloc(jxf_ParCSRCommHandle*, num_vectors);

    /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
    
    if (!comm_pkg)
    {
        jxf_BlockMatvecCommPkgCreate(A);
        comm_pkg = jxf_ParBSRMatrixCommPkg(A); 
    }

    num_sends = jxf_ParCSRCommPkgNumSends(comm_pkg);
    x_buf_data = jxf_CTAlloc(JXF_Real *, num_vectors);
    for (jv = 0; jv < num_vectors; ++ jv)
    {
        x_buf_data[jv] = jxf_CTAlloc( JXF_Real, jxf_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * vec_per_offd_block );
    }

    if ( num_vectors == 1 )
    {
        index = 0;
        for (i = 0; i < num_sends; i ++)
        {
            start = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < jxf_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
            {
                JXF_Int offd_col = jxf_ParCSRCommPkgSendMapElmt(comm_pkg,j);
                JXF_Real* src = x_local_data + offd_col * vec_per_offd_block;
                for (int k = 0; k < vec_per_offd_block; k++)
                    x_buf_data[0][index++] = src[k];
            }
        }
    }
    else
    {
        for (jv = 0; jv < num_vectors; ++ jv)
        {
            index = 0;
            for (i = 0; i < num_sends; i ++)
            {
                start = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
                for (j = start; j < jxf_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
                {
                    JXF_Int offd_col = jxf_ParCSRCommPkgSendMapElmt(comm_pkg,j);
                    JXF_Real* src = x_local_data + jv*vecstride + idxstride*offd_col * vec_per_offd_block;
                    for (int k = 0; k < vec_per_offd_block; k++)
                        x_buf_data[jv][index++] = src[k];
                }
            }
        }
    }

    jxf_assert( idxstride == 1 );
   
   /* >>> ... The assert is because the following loop only works for 'column' storage of a multivector <<<
      >>> This needs to be fixed to work more generally, at least for 'row' storage. <<<
      >>> This in turn, means either change CommPkg so num_sends is no.zones*no.vectors (not no.zones)
      >>> or, less dangerously, put a stride in the logic of CommHandleCreate (stride either from a
      >>> new arg or a new variable inside CommPkg).  Or put the num_vector iteration inside
      >>> CommHandleCreate (perhaps a new multivector variant of it). */
      
    for (jv = 0; jv < num_vectors; ++ jv)
    {
        comm_handle[jv]
        = jxf_ParCSRCommHandleCreate( 1, comm_pkg, x_buf_data[jv], &(x_tmp_data[jv*num_cols_offd*vec_per_offd_block]) );
    }

//    jxf_CSRMatrixMatvec(alpha, diag, x_local, beta, y_local);

    
    for (jv = 0; jv < num_vectors; ++ jv)
    {
        jxf_ParCSRCommHandleDestroy(comm_handle[jv]);
        comm_handle[jv] = NULL;
    }
    jxf_TFree(comm_handle);

    ROW_OMP = (n + nt - 1) / nt;

    if (num_cols_offd)
    {
    #pragma omp parallel num_threads(nt)
        {
            int tid = jxf_GetThreadNum();
            int istart = tid * ROW_OMP;
            int iend = (tid + 1 == nt) ? n : (tid + 1) * ROW_OMP;
            for (int r = istart; r < iend && r < n; r++)
            {
                for (int jj = offd_i[r]; jj < offd_i[r+1]; jj++)
                {
                    JXF_Int col = offd_j[jj];
                    JXF_Real* x_blk = x_tmp_data + col * vec_per_offd_block;
                    JXF_Real* o_blk = offd_data + jj * vec_per_offd_block;
                    for (int k = 0; k < vec_per_offd_block; k++)
                        o_blk[k] = x_blk[k] * o_blk[k];
                }
            }
        }
        // jxf_CSRMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);

    }

   jxf_SeqVectorDestroy(x_tmp);
   x_tmp = NULL;
   for (jv = 0; jv < num_vectors; ++ jv) 
   {
      jxf_TFree(x_buf_data[jv]);
   }
   jxf_TFree(x_buf_data);

   if (jxf__global_mvcpu_flag) jxf_total_elapsed_time_matvec += (jxf_time_getWallclockSeconds() - wall_time);

   return ierr;
}



// void jxf_ParCSRMatrixMatmul_Apattern(jxf_ParCSRMatrix *A, jxf_ParCSRMatrix *L, jxf_ParCSRMatrix *B,
//                                     JXF_Real *C_diag_data, JXF_Real *C_offd_data)
// {

//     MPI_Comm comm = jxf_ParCSRMatrixComm(A);
//     JXF_Int num_procs;
//     jxf_MPI_Comm_size(comm, &num_procs);

//     /* Basic pointers to A and B blocks */
//     jxf_CSRMatrix *A_diag = jxf_ParCSRMatrixDiag(A);
//     jxf_CSRMatrix *A_offd = jxf_ParCSRMatrixOffd(A);
//     JXF_Int *A_diag_i = jxf_CSRMatrixI(A_diag);
//     JXF_Int *A_diag_j = jxf_CSRMatrixJ(A_diag);
//     JXF_Real *A_diag_data = jxf_CSRMatrixData(A_diag);
//     JXF_Int A_diag_nnz = jxf_CSRMatrixNumNonzeros(A_diag);
//     JXF_Int *A_offd_i = jxf_CSRMatrixI(A_offd);
//     JXF_Int *A_offd_j = jxf_CSRMatrixJ(A_offd);
//     JXF_Real *A_offd_data = jxf_CSRMatrixData(A_offd);
//     JXF_Int A_offd_nnz = jxf_CSRMatrixNumNonzeros(A_offd);

//     JXF_Int *col_map_offd_A = jxf_ParCSRMatrixColMapOffd(A);

//     JXF_Int *my_diag_array;
//     JXF_Int *my_offd_array;
//     JXF_Int max_num_threads = jxf_NumThreads();
//     my_diag_array = jxf_CTAlloc(JXF_Int, max_num_threads);
//     my_offd_array = jxf_CTAlloc(JXF_Int, max_num_threads);

//     jxf_CSRMatrix *B_diag = jxf_ParCSRMatrixDiag(B);
//     jxf_CSRMatrix *B_offd = jxf_ParCSRMatrixOffd(B);
//     JXF_Int *B_diag_i = jxf_CSRMatrixI(B_diag);
//     JXF_Int *B_diag_j = jxf_CSRMatrixJ(B_diag);
//     JXF_Real *B_diag_data = jxf_CSRMatrixData(B_diag);
//     JXF_Int *B_offd_i = jxf_CSRMatrixI(B_offd);
//     JXF_Int *B_offd_j = jxf_CSRMatrixJ(B_offd);
//     JXF_Real *B_offd_data = jxf_CSRMatrixData(B_offd);

//     jxf_CSRMatrix *L_diag = jxf_ParCSRMatrixDiag(L);
//     jxf_CSRMatrix *L_offd = jxf_ParCSRMatrixOffd(L);
//     JXF_Int *L_diag_i = jxf_CSRMatrixI(L_diag);
//     JXF_Int *L_diag_j = jxf_CSRMatrixJ(L_diag);
//     JXF_Real *L_diag_data = jxf_CSRMatrixData(L_diag);
//     JXF_Int *L_offd_i = jxf_CSRMatrixI(L_offd);
//     JXF_Int *L_offd_j = jxf_CSRMatrixJ(L_offd);
//     JXF_Real *L_offd_data = jxf_CSRMatrixData(L_offd);

//     /* local sizes / meta */
//     JXF_Int num_rows_diag_A = jxf_CSRMatrixNumRows(A_diag);
//     JXF_Int num_cols_diag_A = jxf_CSRMatrixNumCols(A_diag);
//     JXF_Int num_cols_offd_A = jxf_CSRMatrixNumCols(A_offd);
//     JXF_Int first_col_diag_A = jxf_ParCSRMatrixFirstColDiag(A);
//     JXF_Int last_col_diag_A = first_col_diag_A + num_cols_diag_A - 1;

//     JXF_Int first_col_diag_B = jxf_ParCSRMatrixFirstColDiag(B);
//     JXF_Int num_cols_diag_B = jxf_CSRMatrixNumCols(B_diag);
//     JXF_Int last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

//     jxf_CSRMatrix *Bs_ext = NULL;
//     JXF_Real *Bs_ext_data = NULL;
//     JXF_Int *Bs_ext_i = NULL;
//     JXF_Int *Bs_ext_j = NULL;
//     JXF_Int *B_ext_diag_i = NULL, *B_ext_offd_i = NULL;
//     JXF_Int *B_ext_diag_j = NULL, *B_ext_offd_j = NULL;
//     JXF_Real *B_ext_diag_data = NULL, *B_ext_offd_data = NULL;
//     JXF_Int B_ext_diag_size = 0, B_ext_offd_size = 0;

//     // /* For create C later (not used here, since C arrays provided) */
//     // JXF_Int *row_starts_A = jxf_ParCSRMatrixRowStarts(A);

//     if (num_procs > 1)
//     {
//         /*---------------------------------------------------------------------
//          * If there exists no CommPkg for A, a CommPkg is generated using
//          * equally load balanced partitionings within
//          * jxf_ParCSRMatrixExtractBExt
//          *--------------------------------------------------------------------*/
//         Bs_ext = jxf_ParCSRMatrixExtractBExt(B, A, 1);
//         Bs_ext_data = jxf_CSRMatrixData(Bs_ext);
//         Bs_ext_i = jxf_CSRMatrixI(Bs_ext);
//         Bs_ext_j = jxf_CSRMatrixJ(Bs_ext);
//     }
//     B_ext_diag_i = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
//     B_ext_offd_i = jxf_CTAlloc(JXF_Int, num_cols_offd_A + 1);
//     B_ext_diag_size = 0;
//     B_ext_offd_size = 0;

//     #if JXF_USING_OPENMP
//     #pragma omp parallel
//     #endif
//     {
//         JXF_Int size, rest, ii;
//         JXF_Int ns, ne;
//         JXF_Int i1, i, j;
//         JXF_Int my_offd_size, my_diag_size;
//         JXF_Int cnt_offd, cnt_diag;

//         JXF_Int num_threads = jxf_NumActiveThreads();

//         size = num_cols_offd_A / num_threads;
//         rest = num_cols_offd_A - size * num_threads;
//         ii = jxf_GetThreadNum();
//         if (ii < rest)
//         {
//             ns = ii * size + ii;
//             ne = (ii + 1) * size + ii + 1;
//         }
//         else
//         {
//             ns = ii * size + rest;
//             ne = (ii + 1) * size + rest;
//         }

//         my_diag_size = 0;
//         my_offd_size = 0;
//         for (i = ns; i < ne; i++)
//         {
//             B_ext_diag_i[i] = my_diag_size;
//             B_ext_offd_i[i] = my_offd_size;
//             for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
//             {
//                 if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
//                 {
//                     my_offd_size++;
//                 }
//                 else
//                 {
//                     my_diag_size++;
//                 }
//             }
//         }
//         my_diag_array[ii] = my_diag_size;
//         my_offd_array[ii] = my_offd_size;

//         #if JXF_USING_OPENMP
//         #pragma omp barrier
//         #endif

//         if (ii)
//         {
//             my_diag_size = my_diag_array[0];
//             my_offd_size = my_offd_array[0];
//             for (i1 = 1; i1 < ii; i1++)
//             {
//                 my_diag_size += my_diag_array[i1];
//                 my_offd_size += my_offd_array[i1];
//             }

//             for (i1 = ns; i1 < ne; i1++)
//             {
//                 B_ext_diag_i[i1] += my_diag_size;
//                 B_ext_offd_i[i1] += my_offd_size;
//             }
//         }
//         else
//         {
//             B_ext_diag_size = 0;
//             B_ext_offd_size = 0;
//             for (i1 = 0; i1 < num_threads; i1++)
//             {
//                 B_ext_diag_size += my_diag_array[i1];
//                 B_ext_offd_size += my_offd_array[i1];
//             }
//             B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
//             B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

//             if (B_ext_diag_size)
//             {
//                 B_ext_diag_j = jxf_CTAlloc(JXF_Int, B_ext_diag_size);
//                 B_ext_diag_data = jxf_CTAlloc(JXF_Real, B_ext_diag_size);
//             }
//             if (B_ext_offd_size)
//             {
//                 B_ext_offd_j = jxf_CTAlloc(JXF_Int, B_ext_offd_size);
//                 B_ext_offd_data = jxf_CTAlloc(JXF_Real, B_ext_offd_size);
//             }
//         }
        
//         #if JXF_USING_OPENMP
//         #pragma omp barrier
//         #endif

//         cnt_offd = B_ext_offd_i[ns];
//         cnt_diag = B_ext_diag_i[ns];
//         for (i = ns; i < ne; i++)
//         {
//             for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
//                 if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
//                 {
//                     B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
//                     B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
//                 }
//                 else
//                 {
//                     B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
//                     B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
//                 }
//         }

//     #if JXF_USING_OPENMP
//     #pragma omp barrier
//     #endif
//     }

//     JXF_Int ii1, jj1, jj2, jj3, jj4;
//     JXF_Int i2;
//     JXF_Real a_entry;
//     JXF_Int target_col;
//     JXF_Int global_col;

//     memcpy(C_diag_data, A_diag_data, A_diag_nnz * sizeof(JXF_Real));
//     memcpy(C_offd_data, A_offd_data, A_offd_nnz * sizeof(JXF_Real));

//     /* offd part */
//     if (num_cols_offd_A)
//     {
//         for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
//         {
//             for (jj2 = L_offd_i[ii1]; jj2 < L_offd_i[ii1 + 1]; jj2++)
//             {
//                 i2 = L_offd_j[jj2];
//                 for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
//                 {
//                     for (jj1 = A_offd_i[ii1]; jj1 < A_offd_i[ii1 + 1]; jj1++)
//                     {
//                         target_col = A_offd_j[jj1];
//                         global_col = col_map_offd_A[target_col];
//                         if(B_ext_offd_j[jj3] == global_col)
//                         {
//                             C_offd_data[jj1] -= L_offd_data[jj2] * B_ext_offd_data[jj3];
//                             break;
//                         }
//                     }
//                 }
//             }
//         }
        
//         for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
//         {
//             // L_diag * B_offd
//             for (jj2 = L_diag_i[ii1]; jj2 < L_diag_i[ii1 + 1]; jj2++)
//             {
//                 i2 = L_diag_j[jj2];
//                 for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
//                 {
//                     for (jj1 = A_offd_i[ii1]; jj1 < A_offd_i[ii1 + 1]; jj1++)
//                     {
//                         if(B_offd_j[jj3] == A_offd_j[jj1])
//                         {
//                             C_offd_data[jj1] -= L_diag_data[jj2] * B_offd_data[jj3];
//                             break;
//                         }
//                     }
//                 }
//             }
//         }

//         for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
//         {
//             for (jj2 = L_offd_i[ii1]; jj2 < L_offd_i[ii1 + 1]; jj2++)
//             {
//                 i2 = L_offd_j[jj2];
//                 for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
//                 {
//                     target_col = B_ext_diag_j[jj3];
//                     for (jj1 = A_diag_i[ii1]; jj1 < A_diag_i[ii1 + 1]; jj1++)
//                     {
//                         if(A_diag_j[jj1] == target_col)
//                         {
//                             C_diag_data[jj1] -= L_offd_data[jj2] * B_ext_diag_data[jj3];
//                             break;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     // L_offd * B_ext_diag
//         for (ii1 = 0; ii1 < num_rows_diag_A; ii1++)
//         {
//             // L_diag * B_diag
//             for (jj2 = L_diag_i[ii1]; jj2 < L_diag_i[ii1 + 1]; jj2++)
//             {
//                 i2 = L_diag_j[jj2];
//                 for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
//                 {
//                     target_col = B_diag_j[jj3];
//                     for (jj1 = A_diag_i[ii1]; jj1 < A_diag_i[ii1 + 1]; jj1++)
//                     {
//                         if(A_diag_j[jj1] == target_col)
//                         {
//                             C_diag_data[jj1] -= L_diag_data[jj2] * B_diag_data[jj3];
//                             break;
//                         }
//                     }
//                 }
//             }
//         }

//     if (num_procs > 1)
//     {
//     jxf_CSRMatrixDestroy(Bs_ext);
//     Bs_ext = NULL;
//     }
    
//     jxf_TFree(B_ext_diag_i);
//     if (B_ext_diag_size)
//     {
//     jxf_TFree(B_ext_diag_j);
//     jxf_TFree(B_ext_diag_data);
//     }
//     jxf_TFree(B_ext_offd_i);
//     if (B_ext_offd_size)
//     {
//     jxf_TFree(B_ext_offd_j);
//     jxf_TFree(B_ext_offd_data);
//     }

//     jxf_TFree(my_diag_array);
//     jxf_TFree(my_offd_array);
//     return;
// }
