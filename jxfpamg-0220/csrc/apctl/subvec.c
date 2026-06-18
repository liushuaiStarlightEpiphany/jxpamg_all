//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  subvec.c -- subroutines to extract a or some sub-vectors from a 3tvector.
 *  Date: 2011/09/08
 *
 *  The 3T vectors are of the following form:
 *    /  \      /  \        
 *   | v1 |    | vr |  
 *   | v2 | or | ve |
 *   | v3 |    | vi |
 *    \  /      \  /   
 * 
 *  Created by peghoty
 *
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

/*!
 * \fn JXF_Int jxf_3tGetSubVecs
 * \brief Get the three sub vectors of a 3t vector.
 * \author peghoty
 * \date 2012/01/16
 */
JXF_Int
jxf_3tGetSubVecs( jxf_Vector  *f, 
                 jxf_Vector **fR_ptr, 
                 jxf_Vector **fE_ptr, 
                 jxf_Vector **fI_ptr )
{
   jxf_Vector *fR = NULL; 
   jxf_Vector *fE = NULL; 
   jxf_Vector *fI = NULL;

   JXF_Real *f_data = jxf_VectorData(f);
   JXF_Real *fR_data = NULL;
   JXF_Real *fE_data = NULL;
   JXF_Real *fI_data = NULL;
   JXF_Int size = jxf_VectorSize(f);
   JXF_Int n  = size / 3;
   JXF_Int n2 = 2*n;
   JXF_Int i;

   fR = jxf_SeqVectorCreate(n);
   fE = jxf_SeqVectorCreate(n);
   fI = jxf_SeqVectorCreate(n);

   jxf_SeqVectorInitialize(fR);
   jxf_SeqVectorInitialize(fE);
   jxf_SeqVectorInitialize(fI);

   fR_data = jxf_VectorData(fR);
   fE_data = jxf_VectorData(fE);
   fI_data = jxf_VectorData(fI);
   
   for (i = 0; i < n; i ++)
   {
      fR_data[i] = f_data[i];
      fE_data[i] = f_data[i+n];
      fI_data[i] = f_data[i+n2];
   }

   *fR_ptr = fR;
   *fE_ptr = fE;
   *fI_ptr = fI;

   return (0);
}

JXF_Int
jxf_mgGetSubVecs( jxf_Vector  *f,
                 jxf_Vector **fR,
                 jxf_Vector **fE_ptr,
                 jxf_Vector **fI_ptr,
                 JXF_Int      ng )
{
   jxf_Vector *fE = NULL;
   jxf_Vector *fI = NULL;

   JXF_Real *f_data = jxf_VectorData(f);
   JXF_Real *fE_data = NULL;
   JXF_Real *fI_data = NULL;
   JXF_Int size = jxf_VectorSize(f);
   JXF_Int i,j,n,e_pos,i_pos;

   n = size / (ng + 2);
   e_pos = ng * n;
   i_pos = e_pos + n;

   for (j = 0; j < ng; j ++)
   {
      fR[j] = jxf_SeqVectorCreate(n);
      jxf_SeqVectorInitialize(fR[j]);
   }
   fE = jxf_SeqVectorCreate(n);
   jxf_SeqVectorInitialize(fE);
   fI = jxf_SeqVectorCreate(n);
   jxf_SeqVectorInitialize(fI);

   fE_data = jxf_VectorData(fE);
   fI_data = jxf_VectorData(fI);

   for (i = 0; i < n; i ++)
   {
      for (j = 0; j < ng; j ++)
      {
         jxf_VectorData(fR[j])[i] = f_data[i+j*n];
      }
      fE_data[i] = f_data[i+e_pos];
      fI_data[i] = f_data[i+i_pos];
   }

  *fE_ptr = fE;
  *fI_ptr = fI;

   return (0);
}
