//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jxf_ilu.h -- head files for ILU Preconditioner
 *  Date: 2014/03/24
 *
 *  Created by Yue Xiaoqiang
 */

#ifndef JXF_ILU_HEADER
#define JXF_ILU_HEADER
 
#ifndef JXF_UTIL_HEADER 
#include "jxf_util.h"
#endif

#ifndef JXF_MV_HEADER 
#include "jxf_mv.h"
#endif

/*----------------------------------------------------------------*
 *                   Struct Declaration                           *
 *----------------------------------------------------------------*/

/*!
 * \struct jxf_GridPartitionData
 */
typedef struct
{
    MPI_Comm comm;
    
    JXF_Int part_type;
    JXF_Int x_part_len;
    JXF_Int y_part_len;
    JXF_Int x_lower_idx;
    JXF_Int x_upper_idx;
    JXF_Int y_lower_idx;
    JXF_Int y_upper_idx;
    JXF_Int num_sideproc;
    JXF_Int num_smallside;
    JXF_Int num_largeside;
    JXF_Int num_smallcross;
    JXF_Int num_largecross;
    JXF_Int num_nocrossside;
    
    JXF_Int *xlo_array;
    JXF_Int *xup_array;
    JXF_Int *ylo_array;
    JXF_Int *yup_array;
    JXF_Int *sideprocs;
    JXF_Int *sideprcpos;
    JXF_Int *sideprcxsrt;
    JXF_Int *sideprcysrt;
    JXF_Int *sideprclength;
    
} jxf_GridPartitionData;

/*!
 * \struct jxf_ILUZeroFactorData
 */
typedef struct
{
    MPI_Comm comm;
    
    JXF_Int nx;
    JXF_Int ny;
    JXF_Int npx;
    JXF_Int npy;
    JXF_Int num_equns;
    
    JXF_Int ex_len;
    JXF_Int ey_len;
    JXF_Int dx_len;
    JXF_Int kx_len;
    JXF_Int dy_len;
    JXF_Int lx_len;
    JXF_Int ly_len;
    JXF_Int length;
    JXF_Int max_iter;
    JXF_Int pos_int_end;
    JXF_Int pos_dwn_end;
    JXF_Int pos_lft_end;
    JXF_Int num_fill_in_drop;
    
    JXF_Real drop_tol;
    
    JXF_Int *index;
    JXF_Int *indexA;
    JXF_Int *indexD;
    JXF_Int *permute;
    
    JXF_Real *value;
    JXF_Real *senddown;
    
    MPI_Status *status;
    
    jxf_CSRMatrix *matA;
    
    jxf_Vector *aux_vec;
    jxf_Vector *res_vec;
    jxf_Vector *tmp_vec;
    
    jxf_ParVector *par_aux_vec;
    jxf_ParVector *par_res_vec;
    
    jxf_GridPartitionData *par_grid;
    
} jxf_ILUZeroFactorData;


/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/

/* csrc/ilu/csr_matrix.c */
jxf_CSRMatrix *jxf_CSRMatrixTDMGReorderByNodes( jxf_CSRMatrix *A, JXF_Int num_equns );
jxf_CSRMatrix *jxf_CSRMatrixTDMGReorderByVariables( jxf_CSRMatrix *A, JXF_Int num_equns );
jxf_CSRMatrix *jxf_CSRMatrixTDMGReorderAlongX( jxf_CSRMatrix *A, JXF_Int num_equns, JXF_Int nx, JXF_Int ny );
jxf_CSRMatrix *jxf_CSRMatrixTDMGReorderAlongY( jxf_CSRMatrix *A, JXF_Int num_equns, JXF_Int nx, JXF_Int ny );

/* csrc/ilu/decomposition.c */
JXF_Int jxf_ILUZeroDecompositionA( jxf_CSRMatrix *A,
                              JXF_Int **indexDP_ptr,
                              JXF_Int **indexLU_ptr,
                              JXF_Real **valueLU_ptr );
JXF_Int jxf_ILUZeroDecompositionB( jxf_CSRMatrix *A,
                              JXF_Real drop_tol,
                              JXF_Int **indexAP_ptr,
                              JXF_Int **indexDP_ptr,
                              JXF_Int **indexLU_ptr,
                              JXF_Real **valueLU_ptr );
JXF_Int *jxf_ILUZeroParallelDecompositionA( jxf_ParCSRMatrix *par_A,
                                       JXF_Int **indexDP_ptr,
                                       JXF_Int **indexLU_ptr,
                                       JXF_Real **valueLU_ptr,
                                       JXF_Int *num_nonzeros,
                                       JXF_Int *fill_in_drop,
                                       jxf_ILUZeroFactorData *ilu_data );
JXF_Int
jxf_ILUZeroLocalDecompositionIntURPntsA( JXF_Int *IA,
                                        JXF_Int *JA,
                                        JXF_Real *AA,
                                        JXF_Int *indexLU,
                                        JXF_Int *indexDP,
                                        JXF_Int *placeRC,
                                        JXF_Real *valueLU,
                                        JXF_Int int_uprgt_pnt,
                                        JXF_Int first_row_idx );
void
jxf_ILUZeroLocalDecompositionDPntsA( JXF_Int *IA,
                                    JXF_Int *JA,
                                    JXF_Real *AA,
                                    JXF_Int *indexLU,
                                    JXF_Int *indexDP,
                                    JXF_Int *placeRC,
                                    JXF_Real *valueLU,
                                    JXF_Int int_uprgt_pnt,
                                    JXF_Int first_row_idx,
                                    JXF_Int num_rows,
                                    JXF_Int ex_len,
                                    JXF_Int *fill_in_drop );
void
jxf_ILUZeroLocalDecompositionLPntsA( JXF_Int *IA,
                                    JXF_Int *JA,
                                    JXF_Real *AA,
                                    JXF_Int *indexLU,
                                    JXF_Int *indexDP,
                                    JXF_Int *placeRC,
                                    JXF_Real *valueLU,
                                    JXF_Int int_uprgtdwn_pnt,
                                    JXF_Int first_row_idx,
                                    JXF_Int num_rows,
                                    JXF_Int dwn_num_rows,
                                    JXF_Int ex_len,
                                    JXF_Int *fill_in_drop );
void
jxf_ILUZeroLocalDecompositionLDPntsA( JXF_Int *IA,
                                     JXF_Int *JA,
                                     JXF_Real *AA,
                                     JXF_Int *indexLU,
                                     JXF_Int *indexDP,
                                     JXF_Int *placeRC,
                                     JXF_Real *valueLU,
                                     JXF_Int int_urdl_pnt,
                                     JXF_Int first_row_idx,
                                     JXF_Int fst_row_idx,
                                     JXF_Int num_rows,
                                     JXF_Int lft_cnum_rows,
                                     JXF_Int dwn_cnum_rows,
                                     JXF_Int ex_len,
                                     JXF_Int *fill_in_drop );
void jxf_ILUZeroFactorDataParallelUPartIntURPntsA( JXF_Int *indexLU,
                                                  JXF_Int *indexAP,
                                                  JXF_Int *indexDP,
                                                  JXF_Real *valueLU,
                                                  JXF_Int int_uprgt_pnt,
                                                  JXF_Int recv_downsrt,
                                                  JXF_Int recv_leftsrt,
                                                  JXF_Int *permute,
                                                  JXF_Real *aux_data,
                                                  JXF_Real *res_data,
                                                  JXF_Real *app_data,
                                                  JXF_Int first_row_idx,
                                                  JXF_Int next_row_idx,
                                                  JXF_Int nnxt_row_idx );
void jxf_ILUZeroFactorDataParallelUPartIntURPntsB( JXF_Int *indexLU,
                                                  JXF_Int *indexAP,
                                                  JXF_Int *indexDP,
                                                  JXF_Real *valueLU,
                                                  JXF_Int int_uprgt_pnt,
                                                  JXF_Int recv_downsrt,
                                                  JXF_Int *permute,
                                                  JXF_Real *aux_data,
                                                  JXF_Real *res_data,
                                                  JXF_Real *app_data,
                                                  JXF_Int first_row_idx,
                                                  JXF_Int next_row_idx );
void jxf_ILUZeroFactorDataParallelUPartIntURPntsC( JXF_Int *indexLU,
                                                  JXF_Int *indexAP,
                                                  JXF_Int *indexDP,
                                                  JXF_Real *valueLU,
                                                  JXF_Int int_uprgt_pnt,
                                                  JXF_Int recv_leftsrt,
                                                  JXF_Int *permute,
                                                  JXF_Real *aux_data,
                                                  JXF_Real *res_data,
                                                  JXF_Real *app_data,
                                                  JXF_Int first_row_idx,
                                                  JXF_Int next_row_idx );
void jxf_ILUZeroFactorDataParallelUPartIntURPntsD( JXF_Int *indexLU,
                                                  JXF_Int *indexAP,
                                                  JXF_Int *indexDP,
                                                  JXF_Real *valueLU,
                                                  JXF_Int int_uprgt_pnt,
                                                  JXF_Int *permute,
                                                  JXF_Real *aux_data,
                                                  JXF_Real *res_data,
                                                  JXF_Real *app_data,
                                                  JXF_Int first_row_idx );

/* csrc/ilu/grid.c */
JXF_Int JXF_GridPartitionDataCreate( JXF_Solver *solver, MPI_Comm comm, JXF_Int nx, JXF_Int ny, JXF_Int npx, JXF_Int npy );
JXF_Int JXF_GridPartitionDataSetEachSides4Comm( JXF_Solver solver );
JXF_Int JXF_GridPartitionDataDestroy( JXF_Solver solver );
void *jxf_GridPartitionDataInitialize( MPI_Comm comm, JXF_Int nx, JXF_Int ny, JXF_Int npx, JXF_Int npy );
JXF_Int jxf_GridPartitionDataSetEachSides4Comm( void *grid_vdata );
JXF_Int jxf_GridPartitionDataFinalize( void *grid_vdata );

/* csrc/ilu/ilu.c */
JXF_Int JXF_ILUZeroFactorDataCreate( JXF_Solver *solver, MPI_Comm comm );
JXF_Int JXF_ILUZeroFactorDataSetMaxIter( JXF_Solver solver, JXF_Int max_iter );
JXF_Int JXF_ILUZeroFactorDataSetNxy( JXF_Solver solver, JXF_Int nx, JXF_Int ny );
JXF_Int JXF_ILUZeroFactorDataSetNpxy( JXF_Solver solver, JXF_Int npx, JXF_Int npy );
JXF_Int JXF_ILUZeroFactorDataSetNumEquns( JXF_Solver solver, JXF_Int num_equns );
JXF_Int JXF_ILUZeroFactorDataSetDropTol( JXF_Solver solver, JXF_Real drop_tol );
JXF_Int JXF_ILUZeroFactorDataSetMatA( JXF_Solver solver, jxf_CSRMatrix *matA );
JXF_Int JXF_ILUZeroFactorDataGetLULength( JXF_Solver solver, JXF_Int *lu_length );
JXF_Int JXF_ILUZeroFactorDataDestroy( JXF_Solver solver );
JXF_Int JXF_ILUZeroFactorDataGenerateParGrid( JXF_Solver solver );
JXF_Int JXF_ILUZeroFactorDataSetup( JXF_Solver solver, JXF_ParCSRMatrix par_matrix );
JXF_Int JXF_ILUZeroFactorDataPrecond( JXF_Solver       solver,
                                 JXF_ParCSRMatrix par_matrix,
                                 JXF_ParVector    par_rhs,
                                 JXF_ParVector    par_app  );
void *jxf_ILUZeroFactorDataInitialize( MPI_Comm comm );
JXF_Int jxf_ILUZeroFactorDataSetMaxIter( void *ilu_vdata, JXF_Int max_iter );
JXF_Int jxf_ILUZeroFactorDataSetNxy( void *ilu_vdata, JXF_Int nx, JXF_Int ny );
JXF_Int jxf_ILUZeroFactorDataSetNpxy( void *ilu_vdata, JXF_Int npx, JXF_Int npy );
JXF_Int jxf_ILUZeroFactorDataSetNumEquns( void *ilu_vdata, JXF_Int num_equns );
JXF_Int jxf_ILUZeroFactorDataSetDropTol( void *ilu_vdata, JXF_Real drop_tol );
JXF_Int jxf_ILUZeroFactorDataSetMatA( void *ilu_vdata, jxf_CSRMatrix *matA );
JXF_Int jxf_ILUZeroFactorDataGetLULength( void *ilu_vdata, JXF_Int *lu_length );
JXF_Int jxf_ILUZeroFactorDataFinalize( void *ilu_vdata );
JXF_Int jxf_ILUZeroFactorDataGenerateParGrid( void *ilu_vdata );

/* csrc/ilu/ilucycle.c */
JXF_Int jxf_ILUZeroFactorDataPrecond( void            *ilu_vdata,
                                 jxf_ParCSRMatrix *par_A,
                                 jxf_ParVector    *par_b,
                                 jxf_ParVector    *par_x );
void jxf_ILUZeroFactorDataCycleA( void *ilu_vdata, jxf_CSRMatrix *A, jxf_Vector *f, jxf_Vector *u );
void jxf_ILUZeroFactorDataCycleB( void *ilu_vdata, jxf_CSRMatrix *A, jxf_Vector *f, jxf_Vector *u );
void jxf_ILUZeroFactorDataParallelCycleA( jxf_ILUZeroFactorData *ilu_data,
                                         jxf_ParCSRMatrix *par_A,
                                         jxf_ParVector *par_b,
                                         jxf_ParVector *par_x );
void jxf_ILUZeroFactorDataParallelLPartIntURPntsA( JXF_Int *indexLU,
                                                  JXF_Int *indexAP,
                                                  JXF_Int *indexDP,
                                                  JXF_Real *valueLU,
                                                  JXF_Int *permute,
                                                  JXF_Int int_uprgt_pnt,
                                                  JXF_Real *aux_data,
                                                  JXF_Real *rhs_data,
                                                  JXF_Int first_row_idx );
void jxf_ILUZeroFactorDataParallelLPartDPntsA( JXF_Int *indexLU,
                                              JXF_Int *indexAP,
                                              JXF_Int *indexDP,
                                              JXF_Real *valueLU,
                                              JXF_Int int_uprgt_pnt,
                                              JXF_Int num_rows,
                                              JXF_Int *permute,
                                              JXF_Real *aux_data,
                                              JXF_Real *res_data,
                                              JXF_Real *rhs_data,
                                              JXF_Int first_row_idx );
void jxf_ILUZeroFactorDataParallelLPartLPntsA( JXF_Int *indexLU,
                                              JXF_Int *indexAP,
                                              JXF_Int *indexDP,
                                              JXF_Real *valueLU,
                                              JXF_Int int_uprgtdwn_pnt,
                                              JXF_Int num_rows,
                                              JXF_Int dwn_num_rows,
                                              JXF_Int *permute,
                                              JXF_Real *aux_data,
                                              JXF_Real *res_data,
                                              JXF_Real *rhs_data,
                                              JXF_Int first_row_idx );
void jxf_ILUZeroFactorDataParallelLPartLDPntsA( JXF_Int *indexLU,
                                               JXF_Int *indexAP,
                                               JXF_Int *indexDP,
                                               JXF_Real *valueLU,
                                               JXF_Int int_urdl_pnt,
                                               JXF_Int num_rows,
                                               JXF_Int fst_row_idx,
                                               JXF_Int *permute,
                                               JXF_Int lft_cnum_rows,
                                               JXF_Int dwn_dnum_rows,
                                               JXF_Real *aux_data,
                                               JXF_Real *res_data,
                                               JXF_Real *rhs_data,
                                               JXF_Int first_row_idx );
void jxf_ILUZeroFactorDataParallelUPartLDPntsA( JXF_Int *indexLU,
                                               JXF_Int *indexAP,
                                               JXF_Int *indexDP,
                                               JXF_Real *valueLU,
                                               JXF_Int int_urdl_pnt,
                                               JXF_Int num_rows,
                                               JXF_Int *permute,
                                               JXF_Real *aux_data,
                                               JXF_Real *app_data,
                                               JXF_Int first_row_idx );
void jxf_ILUZeroFactorDataParallelUPartLPntsA( JXF_Int *indexLU,
                                              JXF_Int *indexAP,
                                              JXF_Int *indexDP,
                                              JXF_Real *valueLU,
                                              JXF_Int int_urdl_pnt,
                                              JXF_Int num_rows,
                                              JXF_Int recv_downsrt,
                                              JXF_Int *permute,
                                              JXF_Real *aux_data,
                                              JXF_Real *res_data,
                                              JXF_Real *app_data,
                                              JXF_Int first_row_idx,
                                              JXF_Int next_row_idx );
void jxf_ILUZeroFactorDataParallelUPartDPntsA( JXF_Int *indexLU,
                                              JXF_Int *indexAP,
                                              JXF_Int *indexDP,
                                              JXF_Real *valueLU,
                                              JXF_Int int_uprgt_pnt,
                                              JXF_Int num_rows,
                                              JXF_Int recv_leftsrt,
                                              JXF_Int *permute,
                                              JXF_Real *aux_data,
                                              JXF_Real *res_data,
                                              JXF_Real *app_data,
                                              JXF_Int first_row_idx,
                                              JXF_Int next_row_idx );

/* csrc/ilu/ilusetup.c */
JXF_Int jxf_ILUZeroFactorDataSetup( void *ilu_vdata, jxf_ParCSRMatrix *par_A );

/* csrc/ilu/par_csr_matrix.c */
jxf_ParCSRMatrix *jxf_BuildMatParFromOneFile2( char *filename, 
                                             JXF_Int   file_base,
                                             JXF_Int  *row_part, 
                                             JXF_Int  *col_part,
                                             JXF_Int   num_equns,
                                             JXF_Int   nx,
                                             JXF_Int   ny );
jxf_ParCSRMatrix *jxf_BuildMatParFromOneFile3( char *filename, JXF_Int file_base, JXF_Int num_equns, JXF_Int nx, JXF_Int ny );
jxf_ParCSRMatrix *
jxf_ParCSRMatrixCreate2( MPI_Comm   comm,
                        JXF_Int        global_num_rows,
                        JXF_Int        global_num_cols,
                        JXF_Int       *row_starts,
                        JXF_Int       *col_starts,
                        JXF_Int        num_cols_offd,
                        JXF_Int        num_nonzeros_diag,
                        JXF_Int        num_nonzeros_offd,
                        JXF_Int        num_equns,
                        JXF_Int        nx,
                        JXF_Int        ny );
jxf_ParCSRMatrix *jxf_CSRMatrixToParCSRMatrix2( MPI_Comm comm, jxf_CSRMatrix *A, JXF_Int num_equns, JXF_Int nx, JXF_Int ny );
jxf_CSRMatrix *jxf_MergeDiagAndOffdDropSmall( jxf_ParCSRMatrix *par_matrix, JXF_Real drop_tol );
jxf_CSRMatrix *
jxf_CSRMatrixMergeReorderIntUpRtDnLtQuasiBdy( MPI_Comm              comm,
                                             JXF_Int                  *row_starts,
                                             JXF_Int                  *permute,
                                             jxf_CSRMatrix         *ser_B,
                                             JXF_Int                   nz_srt,
                                             JXF_Int                   ng_pt,
                                             jxf_GridPartitionData *grid_data,
                                             JXF_Int                  *ex_len,
                                             JXF_Int                  *ey_len,
                                             JXF_Int                  *dx_len,
                                             JXF_Int                  *kx_len,
                                             JXF_Int                  *dy_len,
                                             JXF_Int                  *lx_len,
                                             JXF_Int                  *ly_len,
                                             JXF_Int                  *postn_a,
                                             JXF_Int                  *postn_b,
                                             JXF_Int                  *postn_c );

/* csrc/ilu/par_vector.c */
jxf_ParVector *jxf_BuildRhsParFromOneFile2( char *filename, jxf_ParCSRMatrix *A, JXF_Int num_equns, JXF_Int nx, JXF_Int ny );
JXF_Int jxf_GeneratePartitioning2(JXF_Int num_procs, JXF_Int **part_ptr, JXF_Int num_equns, JXF_Int length, JXF_Int ny);

/* csrc/ilu/vector.c */
jxf_Vector *jxf_VectorTDMGReorderByNodes( jxf_Vector *x, JXF_Int num_equns );
jxf_Vector *jxf_VectorTDMGReorderByVariables( jxf_Vector *x, JXF_Int num_equns );
jxf_Vector *jxf_VectorTDMGReorderAlongX( jxf_Vector *x, JXF_Int num_equns, JXF_Int nx, JXF_Int ny );
jxf_Vector *jxf_VectorTDMGReorderAlongY( jxf_Vector *x, JXF_Int num_equns, JXF_Int nx, JXF_Int ny );

#endif
