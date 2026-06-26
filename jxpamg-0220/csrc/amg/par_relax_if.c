//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGRelaxIF
 * \brief This function was ignored before, because I took it as granted that 
 *        the array (or pointer) "grid_relax_points", as one member of the struct
 *        "jx_ParAMGData" must be set, however, it's not necessary.
 * \date 2011/09/03
 */
JX_Int  
jx_PAMGRelaxIF( jx_ParCSRMatrix *par_matrix,
                jx_ParVector    *par_rhs,
                JX_Int             *cf_marker,
                JX_Int              relax_type,
                JX_Int              relax_order,
                JX_Int              cycle_type,
                JX_Real           relax_weight,
                JX_Real           omega,
                jx_ParVector    *par_app,
                jx_ParVector    *Vtemp )
{
   JX_Int i, Solve_err_flag = 0;
   JX_Int relax_points[2];

   if (relax_order == 1 && cycle_type < 3)
   {
       if (cycle_type < 2)
       {
          relax_points[0] =  1;
          relax_points[1] = -1;
       }
       else
       {
	  relax_points[0] = -1;
	  relax_points[1] =  1;
       }

       for (i = 0; i < 2; i ++)
       {
          Solve_err_flag = jx_PAMGRelax( par_matrix,
                                         par_rhs,
                                         cf_marker,
                                         relax_type,
                                         relax_points[i],
                                         relax_weight,
                                         omega,
                                         par_app,
                                         Vtemp ); 
       }
   }
   else
   {
       Solve_err_flag = jx_PAMGRelax( par_matrix,
                                      par_rhs,
                                      cf_marker,
                                      relax_type,
                                      0,
                                      relax_weight,
                                      omega,
                                      par_app,
                                      Vtemp ); 
   }

   return Solve_err_flag;
}
