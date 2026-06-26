//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_pgs.h -- head files for pgs preconditioner
 *  Date: 2013/10/27
 */ 
#include "jx_util.h"

#include "jx_mv.h"

/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/

/* csrc/pgs/pgs.c */
JX_Int 
JX_PGSSetup( JX_Solver        solver, 
                   JX_ParCSRMatrix  par_matrix );
JX_Int
jx_PGSSetup( void             *solver, 
                   jx_ParCSRMatrix  *par_matrix );
JX_Int 
JX_PGSPrecond( JX_Solver       solver,
                     JX_ParCSRMatrix par_matrix,
                     JX_ParVector    par_rhs,
                     JX_ParVector    par_app  );
JX_Int
jx_PGSPrecond( void            *solver,
                     jx_ParCSRMatrix *par_matrix,
                     jx_ParVector    *par_rhs,
                     jx_ParVector    *par_app  );

typedef struct {
   JX_Int times;
}jx_PGS_data; /**< Data for PGS */

JX_Int
JX_PGSSetTimes( JX_Solver solver, JX_Int  pgs_times );

JX_Int
jx_PGSSetTimes( void *data, JX_Int pgs_times );

JX_Int
jx_PGSTimes( void            *solver,
                     jx_ParCSRMatrix *par_matrix,
                     jx_ParVector    *par_rhs,
                     jx_ParVector    *par_app  );