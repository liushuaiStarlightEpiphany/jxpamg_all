//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  interp.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

/*!
 * \fn jxf_ParCSRMatrix *jxf_3tApctlInterpolation
 * \brief Build interpolation operator for PCTL. 
 * \author peghoty 
 * \date 2012/02/29
 */
jxf_ParCSRMatrix *
jxf_3tApctlInterpolation( MPI_Comm         comm, 
                         MPI_Comm         comm_x,
                         JXF_Int              N,
                         JXF_Int              groupid_x,
                         JXF_Int             *row_starts, 
                         JXF_Int             *col_starts,
                         jxf_ParVector    *WEE )
{        
   jxf_ParCSRMatrix *P     = NULL;     
   jxf_CSRMatrix    *P_rec = NULL;
   
   JXF_Int *ip = NULL;
   JXF_Int *jp = NULL;
   JXF_Real *pp = NULL;
   
   jxf_Vector *W = NULL;
   
   JXF_Int num_rows;
   JXF_Int num_cols;
   JXF_Int num_nonzeros;
   
   JXF_Int start;
   JXF_Int first_col_diag;
   JXF_Int last_col_diag;  
   
   JXF_Int owns_row_starts;
   JXF_Int owns_col_starts; 
   
   JXF_Int i, n = N / 3;
   JXF_Int nplusn = 2*n;
   JXF_Int myid, nprocs;
    
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);
   
   
   if (groupid_x == 0)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jxf_CSRMatrixInitialize(P_rec);
      pp = jxf_CSRMatrixData(P_rec);
      ip = jxf_CSRMatrixI(P_rec); 
      jp = jxf_CSRMatrixJ(P_rec);
      
      W = jxf_ParVectorLocalVector(WEE);
      
      start = row_starts[myid];
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = jxf_VectorData(W)[i];
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }
   else if (groupid_x == 1)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jxf_CSRMatrixInitialize(P_rec);
      pp = jxf_CSRMatrixData(P_rec);
      ip = jxf_CSRMatrixI(P_rec); 
      jp = jxf_CSRMatrixJ(P_rec);
            
      start = row_starts[myid] - n;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = 1.0;
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;      
   }
   else if (groupid_x == 2)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jxf_CSRMatrixInitialize(P_rec);
      pp = jxf_CSRMatrixData(P_rec);
      ip = jxf_CSRMatrixI(P_rec); 
      jp = jxf_CSRMatrixJ(P_rec);
      
      W = jxf_ParVectorLocalVector(WEE);
      
      start = row_starts[myid] - nplusn;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = jxf_VectorData(W)[i];
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }

   if (row_starts) 
   {
      owns_row_starts = 0;
   }
   else
   {
      owns_row_starts = 1;
   }
   
   if (col_starts) 
   {
      owns_col_starts = 0;
   }
   else
   {
      owns_col_starts = 1;
   }
        
   P = jxf_ParCSRMatrixCreate(comm, N, n, row_starts, col_starts, 0, 0, 0);
   jxf_ParCSRMatrixSetRowStartsOwner(P, owns_row_starts);
   jxf_ParCSRMatrixSetColStartsOwner(P, owns_col_starts);
 
   first_col_diag = jxf_ParCSRMatrixColStarts(P)[myid];
   last_col_diag  = jxf_ParCSRMatrixColStarts(P)[myid+1] - 1;
   jxf_GenerateDiagAndOffd(P_rec, P, first_col_diag, last_col_diag);

   jxf_CSRMatrixDestroy(P_rec);  

   return (P); 
}

jxf_ParCSRMatrix *
jxf_3tApctlmgInterpolation( MPI_Comm         comm, 
                           MPI_Comm         comm_x,
                           JXF_Int              N,
                           JXF_Int              groupid_x,
                           JXF_Int              ng,
                           JXF_Int             *row_starts, 
                           JXF_Int             *col_starts,
                           jxf_ParVector    *WEE )
{
   jxf_ParCSRMatrix *P     = NULL;
   jxf_CSRMatrix    *P_rec = NULL;

   JXF_Int *ip = NULL;
   JXF_Int *jp = NULL;
   JXF_Real *pp = NULL;

   jxf_Vector *W = NULL;

   JXF_Int num_rows;
   JXF_Int num_cols;
   JXF_Int num_nonzeros;

   JXF_Int start;
   JXF_Int first_col_diag;
   JXF_Int last_col_diag;

   JXF_Int owns_row_starts;
   JXF_Int owns_col_starts;

   JXF_Int i, n = N / (ng + 2);
   JXF_Int e_pos = ng * n;
   JXF_Int nplusn = e_pos + n;
   JXF_Int myid, nprocs;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (groupid_x < ng)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jxf_CSRMatrixInitialize(P_rec);
      pp = jxf_CSRMatrixData(P_rec);
      ip = jxf_CSRMatrixI(P_rec);
      jp = jxf_CSRMatrixJ(P_rec);
      W = jxf_ParVectorLocalVector(WEE);
      start = row_starts[myid] - groupid_x * n;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = jxf_VectorData(W)[i];
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }
   else if (groupid_x == ng)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jxf_CSRMatrixInitialize(P_rec);
      pp = jxf_CSRMatrixData(P_rec);
      ip = jxf_CSRMatrixI(P_rec); 
      jp = jxf_CSRMatrixJ(P_rec);
      start = row_starts[myid] - e_pos;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = 1.0;
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }
   else if (groupid_x == ng+1)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jxf_CSRMatrixInitialize(P_rec);
      pp = jxf_CSRMatrixData(P_rec);
      ip = jxf_CSRMatrixI(P_rec); 
      jp = jxf_CSRMatrixJ(P_rec);
      W = jxf_ParVectorLocalVector(WEE);
      start = row_starts[myid] - nplusn;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = jxf_VectorData(W)[i];
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }

   if (row_starts)
   {
      owns_row_starts = 0;
   }
   else
   {
      owns_row_starts = 1;
   }

   if (col_starts)
   {
      owns_col_starts = 0;
   }
   else
   {
      owns_col_starts = 1;
   }

   P = jxf_ParCSRMatrixCreate(comm, N, n, row_starts, col_starts, 0, 0, 0);
   jxf_ParCSRMatrixSetRowStartsOwner(P, owns_row_starts);
   jxf_ParCSRMatrixSetColStartsOwner(P, owns_col_starts);
 
   first_col_diag = jxf_ParCSRMatrixColStarts(P)[myid];
   last_col_diag  = jxf_ParCSRMatrixColStarts(P)[myid+1] - 1;
   jxf_GenerateDiagAndOffd(P_rec, P, first_col_diag, last_col_diag);

   jxf_CSRMatrixDestroy(P_rec);

   return (P);
}

/*!
 * \fn JXF_Int x_3tAPCTLParaInterpVec
 * \brief 利用进程组上的并行向量生成两个基于所有进程的并行向量(PRR, PII). 
 * \author peghoty 
 * \date 2012/02/29
 */
JXF_Int 
jxf_3tAPCTLParaInterpVec( MPI_Comm       comm,
                         MPI_Comm       comm_x, 
                         JXF_Int            groupid_x,
                         JXF_Int           *vec_starts, 
                         jxf_ParVector  *WEE, 
                         jxf_ParVector **PRR_ptr, 
                         jxf_ParVector **PII_ptr )
{
   JXF_Int nprocs;
   JXF_Int npeachgroup, rootid_R, rootid_I;
  
   jxf_ParVector *PRR = NULL;
   jxf_ParVector *PII = NULL;
   
   jxf_Vector *PRR_s = NULL;
   jxf_Vector *PII_s = NULL;   
 
   jxf_MPI_Comm_size(comm, &nprocs);

   npeachgroup = nprocs / 3;
   rootid_R = 0; 
   rootid_I = 2*npeachgroup;
   
   if (groupid_x == 0)
   { 
      PRR_s = jxf_ParVectorToVectorAll(WEE);
   }
   else if (groupid_x == 2)
   {    
      PII_s = jxf_ParVectorToVectorAll(WEE);
   }

   PRR = jxf_VectorToParVector_FromGivenPro(comm, rootid_R, PRR_s, vec_starts);
   PII = jxf_VectorToParVector_FromGivenPro(comm, rootid_I, PII_s, vec_starts);

   if (groupid_x == 0)
   {
      jxf_SeqVectorDestroy(PRR_s);
   }
   else if (groupid_x == 2)
   {
      jxf_SeqVectorDestroy(PII_s);
   }
   
   *PRR_ptr = PRR;
   *PII_ptr = PII;

   return (0);
}

JXF_Int 
jxf_3tAPCTLmgParaInterpVec( MPI_Comm        comm,
                           MPI_Comm        comm_x, 
                           JXF_Int            groupid_x,
                           JXF_Int           *vec_starts, 
                           JXF_Int          ng,
                           jxf_ParVector   *WEE, 
                           jxf_ParVector ***PRR_ptr, 
                           jxf_ParVector  **PII_ptr )
{
   JXF_Int nprocs, gidx, npeachgroup;

   jxf_ParVector **PRR = jxf_CTAlloc(jxf_ParVector *, ng);

   jxf_ParVector *PII = NULL;

   jxf_Vector *PRR_s = NULL;
   jxf_Vector *PII_s = NULL;
 
   jxf_MPI_Comm_size(comm, &nprocs);

   npeachgroup = nprocs / (ng + 2);

   if (groupid_x < ng)
   {
      PRR_s = jxf_ParVectorToVectorAll(WEE);
   }
   else if (groupid_x == ng+1)
   {
      PII_s = jxf_ParVectorToVectorAll(WEE);
   }

   for (gidx = 0; gidx < ng; gidx ++) PRR[gidx] = jxf_VectorToParVector_FromGivenPro(comm, gidx*npeachgroup, PRR_s, vec_starts);
   PII = jxf_VectorToParVector_FromGivenPro(comm, (ng+1)*npeachgroup, PII_s, vec_starts);

   if (groupid_x < ng)
   {
      jxf_SeqVectorDestroy(PRR_s);
   }
   else if (groupid_x == ng+1)
   {
      jxf_SeqVectorDestroy(PII_s);
   }

   *PRR_ptr = PRR;
   *PII_ptr = PII;

   return (0);
}
