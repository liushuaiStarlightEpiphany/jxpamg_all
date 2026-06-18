//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//
// pbgmres.c - BSR版本的GMRES实现
#include "jxf_combined.h"
#include "jxf_krylov.h"

/*!
 * \fn JXF_Int JXF_ParBSRGMRESCreate
 * \brief 创建BSR矩阵的GMRES求解器
 */ 
JXF_Int
JXF_ParBSRGMRESCreate( MPI_Comm comm, JXF_Solver *solver )
{
   jxf_GMRESFunctions * gmres_functions =
      jxf_GMRESFunctionsCreate(
         jxf_CAlloc, 
         jxf_ParKrylovFree, 
         jxf_ParBSRKrylovCommInfo,
         jxf_ParBSRKrylovCreateVector,
         jxf_ParBSRKrylovCreateVectorArray,
         jxf_ParBSRKrylovDestroyVector, 
         jxf_ParBSRKrylovMatvecCreate,
         jxf_ParBSRKrylovMatvec, 
         jxf_ParBSRKrylovMatvecDestroy,
         jxf_ParBSRKrylovInnerProd, 
         jxf_ParBSRKrylovCopyVector,
         jxf_ParBSRKrylovClearVector,
         jxf_ParBSRKrylovScaleVector, 
         jxf_ParBSRKrylovAxpy,
         jxf_ParBSRKrylovIdentitySetup, 
         jxf_ParBSRKrylovIdentity );

   *solver = ( (JXF_Solver) jxf_GMRESCreate( gmres_functions ) );
   
   if (!solver) 
   {
      jxf_error_in_arg(2);
   }

   return jxf_error_flag;
}

/*!
 * \fn JXF_Int JXF_ParBSRGMRESDestroy
 */ 
JXF_Int 
JXF_ParBSRGMRESDestroy( JXF_Solver solver )
{
   return( jxf_GMRESDestroy( (void *) solver ) );
}

/*!
 * \fn void *jxf_ParBSRGMRESCreate
 */
void *
jxf_ParBSRGMRESCreate( MPI_Comm comm )
{
   jxf_GMRESData *solver = NULL;
   
   jxf_GMRESFunctions * gmres_functions =
      jxf_GMRESFunctionsCreate(
         jxf_CAlloc, 
         jxf_ParKrylovFree, 
         jxf_ParBSRKrylovCommInfo,
         jxf_ParBSRKrylovCreateVector,
         jxf_ParBSRKrylovCreateVectorArray,
         jxf_ParBSRKrylovDestroyVector, 
         jxf_ParBSRKrylovMatvecCreate,
         jxf_ParBSRKrylovMatvec, 
         jxf_ParBSRKrylovMatvecDestroy,
         jxf_ParBSRKrylovInnerProd, 
         jxf_ParBSRKrylovCopyVector,
         jxf_ParBSRKrylovClearVector,
         jxf_ParBSRKrylovScaleVector, 
         jxf_ParBSRKrylovAxpy,
         jxf_ParBSRKrylovIdentitySetup, 
         jxf_ParBSRKrylovIdentity );

   solver = ( (jxf_GMRESData *)jxf_GMRESCreate( gmres_functions ) );
   
   if (!solver) 
   {
      jxf_error_in_arg(2);
   }

   return solver;
}

/*!
 * \fn JXF_Int jxf_ParBSRGMRESSolve
 * \brief BSR版本的GMRES求解函数
 */
JXF_Int
jxf_ParBSRGMRESSolve( void *gmres_vdata, void *preOperator, void *matA, 
                    void *vecB, void *vecX )
{
   // 直接调用通用的GMRES求解函数
   return jxf_GMRESSolve(gmres_vdata, preOperator, matA, vecB, vecX);
}

/*!
 * \fn JXF_Int JXF_ParBSRGMRESSetup
 * \brief BSR版本的GMRES设置函数
 */
JXF_Int 
JXF_ParBSRGMRESSetup( JXF_Solver solver, JXF_Matrix matA, JXF_Vector vecB, JXF_Vector vecX )
{
   return( jxf_GMRESSetup( solver, matA, vecB, vecX ) );
}

/*!
 * \fn JXF_Int JXF_ParBSRGMRESSolve
 */
JXF_Int 
JXF_ParBSRGMRESSolve( JXF_Solver solver, JXF_Matrix preOperator, JXF_Matrix matA, 
                     JXF_Vector vecB, JXF_Vector vecX )
{
   return( jxf_GMRESSolve( solver, preOperator, matA, vecB, vecX ) );
}