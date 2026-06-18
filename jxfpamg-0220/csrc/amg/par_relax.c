//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax.c
 *  Date: 2011/09/03
 */ 

#include "jxf_pamg.h"

/*!
 * \fn JXF_Int jxf_PAMGRelax
 * \brief Relaxation phase of PAMG.
 * \date 2011/09/03
 */
JXF_Int  
jxf_PAMGRelax( jxf_ParCSRMatrix *par_matrix,
              jxf_ParVector    *par_rhs,
              JXF_Int             *cf_marker,
              JXF_Int              relax_type,
              JXF_Int              relax_points,
              JXF_Real           relax_weight,
              JXF_Real           omega,
              jxf_ParVector    *par_app,
              jxf_ParVector    *Vtemp )
{
   JXF_Int relax_error = 0;
   
  /*-----------------------------------------------------------------------------
   * Switch statement to direct control based on relax_type:
   *     relax_type = 0 -> Jacobi or CF-Jacobi
   *     relax_type = 1 -> Gauss-Seidel <--- very slow, sequential
   *     relax_type = 2 -> Gauss_Seidel: interior points in parallel ,
   *			 	   	  boundary sequential 
   *     relax_type = 3 -> hybrid: SOR-J mix off-processor, SOR on-processor
   *     		    with outer relaxation parameters (forward solve)
   *     relax_type = 4 -> hybrid: SOR-J mix off-processor, SOR on-processor
   *     		    with outer relaxation parameters (backward solve)
   *     relax_type = 5 -> hybrid: GS-J mix off-processor, chaotic GS on-node
   *     relax_type = 6 -> hybrid: SSOR-J mix off-processor, SSOR on-processor
   *     		    with outer relaxation parameters 
   *     relax_type = 7 -> Jacobi (uses Matvec), only needed in CGNR
   *     relax_type = 9 -> Direct Solve
   *-----------------------------------------------------------------------------*/
   relax_points = 0; // zlj set for UA
   if (relax_type == 0)
   {
      relax_error = jxf_PAMGRelax0(par_matrix, par_rhs, cf_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 1)
   {
      relax_error = jxf_PAMGRelax1(par_matrix, par_rhs, cf_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 2)
   {
      relax_error = jxf_PAMGRelax2(par_matrix, par_rhs, cf_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 3)
   {
      relax_error = jxf_PAMGRelax3(par_matrix, par_rhs, cf_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 4)
   {
      relax_error = jxf_PAMGRelax4(par_matrix, par_rhs, cf_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 5)
   {
      relax_error = jxf_PAMGRelax5(par_matrix, par_rhs, cf_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 6)
   {
      relax_error = jxf_PAMGRelax6(par_matrix, par_rhs, cf_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 7)
   {
      relax_error = jxf_PAMGRelax7(par_matrix, par_rhs, cf_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 9)
   {
      relax_error = jxf_PAMGRelax9(par_matrix, par_rhs, cf_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }

   return(relax_error); 
}
