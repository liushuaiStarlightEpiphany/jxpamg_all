//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  partition.c -- some basic function to check partitions of vector and 
 *                 processor-grouping.
 *  Date: 2011/09/08
 *
 *  The 3T matrices are of the following form:
 *    /           \      /           \  
 *   | A11 A12  0  |    | Arr Are  0  |  
 *   | A21 A22 A23 | or | Aer Aee Aei |
 *   |  0  A32 A33 |    |  0  Aie Aii |
 *    \           /      \           /   
 * 
 *  Created by peghoty
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

/*!
 * \fn JXF_Int jxf_3tCheckNprocs
 * \brief Check whether number of processors is feasible.
 * \author peghoty 
 * \date 2011/09/22
 */
JXF_Int
jxf_3tCheckNprocs( MPI_Comm comm, JXF_Int print_level, JXF_Int *np_R_ptr, JXF_Int *np_E_ptr, JXF_Int *np_I_ptr )
{
   JXF_Int np_R = 0, np_E = 0, np_I = 0;
   JXF_Int np, myid;
   
   jxf_MPI_Comm_size(comm, &np);
   jxf_MPI_Comm_rank(comm, &myid);

   if (print_level && myid == 0) 
   {
      jxf_printf(" =========================================================\n");
      jxf_printf("  The number of processors should be 1 or a multiple of 3!\n");
      jxf_printf(" =========================================================\n\n");
   }
   
   if ( np % 3 == 0 && np > 0)
   {
      np_R = np / 3;
      np_E = np_R;
      np_I = np_R; 
      if (print_level && myid == 0) 
      {
         jxf_printf(" The number of processors (np = %d) is feasible!\n", np);
         jxf_printf(" (np_R np_E np_I) = (%d %d %d) \n\n", np_R, np_E, np_I);
      }
   }
   else if (np != 1)
   {
      if (myid == 0) 
      {
         jxf_printf(" Warning: np should be 1 or a multiple of 3, please try again!\n\n"); 
      } 
      exit(-2);
   }

   *np_R_ptr = np_R;
   *np_E_ptr = np_E;
   *np_I_ptr = np_I;

   return (0);
}

/*!
 * \fn JXF_Int jxf_3tGetMyPartition
 * \brief Get the partition based on processor-grouping information.
 * \author peghoty 
 * \date 2011/09/07
 */
JXF_Int 
jxf_3tGetMyPartition( JXF_Int   nprocs, 
                     JXF_Int   np_R, 
                     JXF_Int   np_E, 
                     JXF_Int   np_I, 
                     JXF_Int   N, 
                     JXF_Int **mypartition_ptr )
{
   JXF_Int n = N / 3;
   JXF_Int np_RE = np_R + np_E;
   JXF_Int i, div, res;
   
   JXF_Int *np_dof = NULL;
   JXF_Int *mypartition = jxf_CTAlloc(JXF_Int, nprocs + 1);

   if (nprocs > 1)
   {    
      np_dof = jxf_CTAlloc(JXF_Int, nprocs);
        
      /* R */ 
      div = n / np_R;
      res = n % np_R;
 
      for (i = 0; i < res; i ++)
      {
         np_dof[i] = div + 1; 
      }
      for (i = res; i < np_R; i ++)
      {
         np_dof[i] = div; 
      }

      /* E */
      div = n / np_E;
      res = n % np_E;
   
      for (i = 0; i < res; i ++)
      {
         np_dof[i+np_R] = div + 1;
      }
      for (i = res; i < np_E; i ++)
      {
         np_dof[i+np_R] = div; 
      }
     
      /* I */
      div = n / np_I;
      res = n % np_I;
   
      for (i = 0; i < res; i ++)
      {
         np_dof[i+np_RE] = div + 1;
      }
      for (i = res; i < np_I; i ++)
      {
         np_dof[i+np_RE] = div; 
      }
       
      /* Generate mypatition */      
      mypartition[0] = 0;
      for (i = 0; i < nprocs; i ++)
      {
         mypartition[i+1] = mypartition[i] + np_dof[i];
      }
      
      jxf_TFree(np_dof);
   }
   else // newly added for single-processor case, peghoty, 2012/02/15
   {
      mypartition[0] = 0;
      mypartition[1] = N;
   }
   
   *mypartition_ptr = mypartition;
   
   return (0);
}
