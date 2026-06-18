//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  coarseop.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

/*!
 * \fn jxf_ParCSRMatrix *jxf_3tApctlCoarseOperator
 * \brief Build interpolation operator for PCTL. 
 * \note The sparse pattern of ARR_s, AEE_s, AII_s should be exactly the same.
 *       If not, something needs to be done to guarantee this condition.
 *       For example, reorder each row ascendingly based on the clumn indices.
 * \author peghoty 
 * \date 2012/02/28
 */
jxf_ParCSRMatrix *
jxf_3tApctlCoarseOperator_mp( MPI_Comm          comm,
                             jxf_ParCSRMatrix  *ARR, 
                             jxf_ParCSRMatrix  *AEE, 
                             jxf_ParCSRMatrix  *AII, 
                             jxf_ParVector     *VRE, 
                             jxf_ParVector     *VER, 
                             jxf_ParVector     *VEI, 
                             jxf_ParVector     *VIE,
                             jxf_ParVector     *PRR, 
                             jxf_ParVector     *PII )
{
   jxf_ParCSRMatrix  *ACC = NULL;
   
   jxf_CSRMatrix   *diagR = NULL;
   jxf_CSRMatrix   *diagE = NULL;
   jxf_CSRMatrix   *diagI = NULL;
   jxf_CSRMatrix   *diagC = NULL;
   
   jxf_CSRMatrix   *offdR = NULL;
   jxf_CSRMatrix   *offdE = NULL;
   jxf_CSRMatrix   *offdI = NULL;
   jxf_CSRMatrix   *offdC = NULL;

   JXF_Int n = jxf_ParCSRMatrixGlobalNumRows(ARR);
  
   JXF_Real *send_buf_r = NULL;
   JXF_Real *send_buf_i = NULL;

   jxf_ParCSRCommHandle  *comm_handle_R = NULL;
   jxf_ParCSRCommHandle  *comm_handle_I = NULL;
   jxf_ParCSRCommPkg	*comm_pkg      = NULL;
   
   JXF_Int  myid;
   JXF_Int  i, j, k;   
   JXF_Int  index, start, end;
   JXF_Int  num_sends;
   JXF_Int  send_buf_size;
   JXF_Int  local_size;
   JXF_Int  num_cols_offd;
   JXF_Real diag_add;
   
   JXF_Real *pr_local_data = NULL;   
   JXF_Real *pi_local_data = NULL;

   JXF_Real *pr_other_data = NULL;   
   JXF_Real *pi_other_data = NULL;
   
   JXF_Int    *diag_ia = NULL;
   JXF_Int    *diag_ja = NULL;
   JXF_Int    *offd_ia = NULL;
   JXF_Int    *offd_ja = NULL;   
   
   JXF_Real *diag_aa_R = NULL;
   JXF_Real *diag_aa_I = NULL;
   JXF_Real *diag_aa_C = NULL;
   
   JXF_Real *offd_aa_R = NULL;
   JXF_Real *offd_aa_I = NULL;
   JXF_Real *offd_aa_C = NULL;
   
   JXF_Real *Vre_local_data = NULL;
   JXF_Real *Ver_local_data = NULL;
   JXF_Real *Vei_local_data = NULL;
   JXF_Real *Vie_local_data = NULL;    
   
   JXF_Int *row_starts = NULL;
   JXF_Int *col_starts = NULL;
   JXF_Int  num_nonzeros_diag;
   JXF_Int  num_nonzeros_offd;    
 
   jxf_MPI_Comm_rank(comm, &myid);   

   diagR = jxf_ParCSRMatrixDiag(ARR);
   diagE = jxf_ParCSRMatrixDiag(AEE);
   diagI = jxf_ParCSRMatrixDiag(AII);
   offdR = jxf_ParCSRMatrixOffd(ARR);
   offdE = jxf_ParCSRMatrixOffd(AEE);
   offdI = jxf_ParCSRMatrixOffd(AII); 
  
#if 1
   jxf_CSRMatrixReorderColumnNumber12(diagR);
   jxf_CSRMatrixReorderColumnNumber12(diagE);
   jxf_CSRMatrixReorderColumnNumber12(diagI);
   jxf_CSRMatrixReorderColumnNumberAll(offdR);
   jxf_CSRMatrixReorderColumnNumberAll(offdE);
   jxf_CSRMatrixReorderColumnNumberAll(offdI);
#endif

   num_cols_offd = jxf_CSRMatrixNumCols(offdE);
   num_nonzeros_diag = jxf_CSRMatrixNumNonzeros(diagE);
   num_nonzeros_offd = jxf_CSRMatrixNumNonzeros(offdE);
   jxf_ParCSRMatrixGetRowPartitioning(AEE, &row_starts);
   jxf_ParCSRMatrixGetColPartitioning(AEE, &col_starts);
   ACC = jxf_ParCSRMatrixCreate( comm, n, n, row_starts, col_starts,
                                num_cols_offd, num_nonzeros_diag, num_nonzeros_offd );
                          
   jxf_ParCSRMatrixInitialize(ACC);
   jxf_ParCSRMatrixCopy(AEE, ACC, 1); 

   diagC = jxf_ParCSRMatrixDiag(ACC);
   offdC = jxf_ParCSRMatrixOffd(ACC); 

   comm_pkg = jxf_ParCSRMatrixCommPkg(ARR);
   if (!comm_pkg)
   {
      jxf_MatvecCommPkgCreate(ARR);
      comm_pkg = jxf_ParCSRMatrixCommPkg(ARR); 
   }

   num_sends = jxf_ParCSRCommPkgNumSends(comm_pkg);
   send_buf_size = jxf_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   send_buf_r = jxf_CTAlloc(JXF_Real, send_buf_size);
   send_buf_i = jxf_CTAlloc(JXF_Real, send_buf_size);
   pr_local_data = jxf_VectorData(jxf_ParVectorLocalVector(PRR));
   pi_local_data = jxf_VectorData(jxf_ParVectorLocalVector(PII));
   
   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
      end   = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      for (j = start; j < end; j ++)
      {
         send_buf_r[index] = pr_local_data[jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         send_buf_i[index] = pi_local_data[jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         index ++;
      }
   }

   num_cols_offd = jxf_CSRMatrixNumCols(offdR);
   
   pr_other_data = jxf_CTAlloc(JXF_Real, num_cols_offd);
   pi_other_data = jxf_CTAlloc(JXF_Real, num_cols_offd);

   comm_handle_R = jxf_ParCSRCommHandleCreate(1, comm_pkg, send_buf_r, pr_other_data);
   comm_handle_I = jxf_ParCSRCommHandleCreate(1, comm_pkg, send_buf_i, pi_other_data);

   diag_ia = jxf_CSRMatrixI(diagC);
   diag_ja = jxf_CSRMatrixJ(diagC);

   diag_aa_R = jxf_CSRMatrixData(diagR);
   diag_aa_I = jxf_CSRMatrixData(diagI);
   diag_aa_C = jxf_CSRMatrixData(diagC);
   
   local_size = jxf_CSRMatrixNumRows(diagC);

   for (i = 0; i < local_size; i ++)
   {
      for (k = diag_ia[i]; k < diag_ia[i+1]; k ++)
      {
         j = diag_ja[k];
         /*diag_aa_C[k] += (   diag_aa_R[k]*pr_local_data[i]*pr_local_data[j]
                           + diag_aa_I[k]*pi_local_data[i]*pi_local_data[j] );*/
         diag_aa_C[k] += (diag_aa_R[k] * pr_local_data[i] * pr_local_data[j]);
         diag_aa_C[k] += (diag_aa_I[k] * pi_local_data[i] * pi_local_data[j]);
      }
   }

   jxf_ParCSRCommHandleDestroy(comm_handle_R);
   jxf_ParCSRCommHandleDestroy(comm_handle_I);

   offd_ia = jxf_CSRMatrixI(offdC);
   offd_ja = jxf_CSRMatrixJ(offdC);     

   offd_aa_R = jxf_CSRMatrixData(offdR);
   offd_aa_I = jxf_CSRMatrixData(offdI);
   offd_aa_C = jxf_CSRMatrixData(offdC);   

   for (i = 0; i < local_size; i ++)
   {
      for (k = offd_ia[i]; k < offd_ia[i+1]; k ++)
      {
         j = offd_ja[k];
         /*offd_aa_C[k] += (   offd_aa_R[k]*pr_local_data[i]*pr_other_data[j]
                           + offd_aa_I[k]*pi_local_data[i]*pi_other_data[j] );*/
         offd_aa_C[k] += (offd_aa_R[k] * pr_local_data[i] * pr_other_data[j]);
         offd_aa_C[k] += (offd_aa_I[k] * pi_local_data[i] * pi_other_data[j]);
      }
   }

   Vre_local_data = jxf_VectorData(jxf_ParVectorLocalVector(VRE));
   Ver_local_data = jxf_VectorData(jxf_ParVectorLocalVector(VER));
   Vei_local_data = jxf_VectorData(jxf_ParVectorLocalVector(VEI));
   Vie_local_data = jxf_VectorData(jxf_ParVectorLocalVector(VIE));    

   for (i = 0; i < local_size; i ++)
   {
      k = diag_ia[i];
      
      /*diag_add =   (Vre_local_data[i] + Ver_local_data[i])*pr_local_data[i]
                 + (Vei_local_data[i] + Vie_local_data[i])*pi_local_data[i];*/
      diag_add = 0.0;
      diag_add += ((Vre_local_data[i] + Ver_local_data[i]) * pr_local_data[i]);
      diag_add += ((Vei_local_data[i] + Vie_local_data[i]) * pi_local_data[i]);
      
      diag_aa_C[k] += diag_add;
   } 
   
   jxf_TFree(send_buf_r);
   jxf_TFree(send_buf_i);
   jxf_TFree(pr_other_data);
   jxf_TFree(pi_other_data);

   return (ACC);
}

jxf_ParCSRMatrix *
jxf_3tApctlmgCoarseOperator_mp( MPI_Comm          comm,
                               JXF_Int            ng,
                               jxf_ParCSRMatrix **ARR, 
                               jxf_ParCSRMatrix  *AEE, 
                               jxf_ParCSRMatrix  *AII, 
                               jxf_ParVector    **VRE, 
                               jxf_ParVector    **VER, 
                               jxf_ParVector     *VEI, 
                               jxf_ParVector     *VIE,
                               jxf_ParVector    **PRR, 
                               jxf_ParVector     *PII )
{
   jxf_ParCSRMatrix *ACC = NULL;
 
   jxf_CSRMatrix *diagE = NULL;
   jxf_CSRMatrix *diagI = NULL;
   jxf_CSRMatrix *diagC = NULL;
   jxf_CSRMatrix *offdE = NULL;
   jxf_CSRMatrix *offdI = NULL;
   jxf_CSRMatrix *offdC = NULL;

   JXF_Int n = jxf_ParCSRMatrixGlobalNumRows(ARR[0]);

   JXF_Real **send_buf_r = jxf_CTAlloc(JXF_Real *, ng);

   JXF_Real *send_buf_i = NULL;

   jxf_ParCSRCommHandle **comm_handle_R = jxf_CTAlloc(jxf_ParCSRCommHandle *, ng);

   jxf_ParCSRCommHandle *comm_handle_I = NULL;

   jxf_ParCSRCommPkg	*comm_pkg = NULL;
 
   JXF_Int  myid;
   JXF_Int  i, j, k, gidx;
   JXF_Int  index, start, end;
   JXF_Int  num_sends;
   JXF_Int  send_buf_size;
   JXF_Int  local_size;
   JXF_Int  num_cols_offd;
   JXF_Real diag_add;
 
   JXF_Real *pi_local_data = NULL;

   JXF_Real **pr_other_data = jxf_CTAlloc(JXF_Real *, ng);

   JXF_Real *pi_other_data = NULL;
 
   JXF_Int *diag_ia = NULL;
   JXF_Int *diag_ja = NULL;
   JXF_Int *offd_ia = NULL;
   JXF_Int *offd_ja = NULL;

   JXF_Real *diag_aa_I = NULL;
   JXF_Real *diag_aa_C = NULL;
   JXF_Real *offd_aa_I = NULL;
   JXF_Real *offd_aa_C = NULL;

   JXF_Real *Vei_local_data = NULL;
   JXF_Real *Vie_local_data = NULL;

   JXF_Int *row_starts = NULL;
   JXF_Int *col_starts = NULL;

   JXF_Int num_nonzeros_diag;
   JXF_Int num_nonzeros_offd;
 
   jxf_MPI_Comm_rank(comm, &myid);

#if 0
   for (gidx = 0; gidx < ng; gidx ++)
   {
      jxf_CSRMatrixReorderColumnNumber12(jxf_ParCSRMatrixDiag(ARR[gidx]));
      jxf_CSRMatrixReorderColumnNumberAll(jxf_ParCSRMatrixOffd(ARR[gidx]));
   }
   jxf_CSRMatrixReorderColumnNumber12(diagE);
   jxf_CSRMatrixReorderColumnNumber12(diagI);
   jxf_CSRMatrixReorderColumnNumberAll(offdE);
   jxf_CSRMatrixReorderColumnNumberAll(offdI);
#endif

   diagE = jxf_ParCSRMatrixDiag(AEE);
   offdE = jxf_ParCSRMatrixOffd(AEE);
   diagI = jxf_ParCSRMatrixDiag(AII);
   offdI = jxf_ParCSRMatrixOffd(AII); 

   num_cols_offd = jxf_CSRMatrixNumCols(offdE);
   num_nonzeros_diag = jxf_CSRMatrixNumNonzeros(diagE);
   num_nonzeros_offd = jxf_CSRMatrixNumNonzeros(offdE);
   jxf_ParCSRMatrixGetRowPartitioning(AEE, &row_starts);
   jxf_ParCSRMatrixGetColPartitioning(AEE, &col_starts);
   ACC = jxf_ParCSRMatrixCreate( comm, n, n, row_starts, col_starts, num_cols_offd, num_nonzeros_diag, num_nonzeros_offd );
   jxf_ParCSRMatrixInitialize(ACC);
   jxf_ParCSRMatrixCopy(AEE, ACC, 1);
   diagC = jxf_ParCSRMatrixDiag(ACC);
   offdC = jxf_ParCSRMatrixOffd(ACC);

   comm_pkg = jxf_ParCSRMatrixCommPkg(ARR[0]);
   if (!comm_pkg)
   {
      jxf_MatvecCommPkgCreate(ARR[0]);
      comm_pkg = jxf_ParCSRMatrixCommPkg(ARR[0]);
   }

   num_sends = jxf_ParCSRCommPkgNumSends(comm_pkg);
   send_buf_size = jxf_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   for (gidx = 0; gidx < ng; gidx ++) send_buf_r[gidx] = jxf_CTAlloc(JXF_Real, send_buf_size);
   send_buf_i = jxf_CTAlloc(JXF_Real, send_buf_size);
   pi_local_data = jxf_VectorData(jxf_ParVectorLocalVector(PII));

   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i);
      end   = jxf_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      for (j = start; j < end; j ++)
      {
         for (gidx = 0; gidx < ng; gidx ++)
         {
            send_buf_r[gidx][index] = jxf_VectorData(jxf_ParVectorLocalVector(PRR[gidx]))[jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
         send_buf_i[index] = pi_local_data[jxf_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         index ++;
      }
   }

   num_cols_offd = jxf_CSRMatrixNumCols(jxf_ParCSRMatrixOffd(ARR[0]));

   for (gidx = 0; gidx < ng; gidx ++) pr_other_data[gidx] = jxf_CTAlloc(JXF_Real, num_cols_offd);
   pi_other_data = jxf_CTAlloc(JXF_Real, num_cols_offd);

   for (gidx = 0; gidx < ng; gidx ++)
   {
      comm_handle_R[gidx] = jxf_ParCSRCommHandleCreate(1, comm_pkg, send_buf_r[gidx], pr_other_data[gidx]);
   }
   comm_handle_I = jxf_ParCSRCommHandleCreate(1, comm_pkg, send_buf_i, pi_other_data);

   diag_ia = jxf_CSRMatrixI(diagC);
   diag_ja = jxf_CSRMatrixJ(diagC);

   diag_aa_I = jxf_CSRMatrixData(diagI);
   diag_aa_C = jxf_CSRMatrixData(diagC);
 
   local_size = jxf_CSRMatrixNumRows(diagC);

   for (i = 0; i < local_size; i ++)
   {
      for (k = diag_ia[i]; k < diag_ia[i+1]; k ++)
      {
         j = diag_ja[k];
         for (gidx = 0; gidx < ng; gidx ++)
         {
            diag_aa_C[k] += ( jxf_CSRMatrixData(jxf_ParCSRMatrixDiag(ARR[gidx]))[k] *
                                jxf_VectorData(jxf_ParVectorLocalVector(PRR[gidx]))[i] *
                                  jxf_VectorData(jxf_ParVectorLocalVector(PRR[gidx]))[j] );
         }
         diag_aa_C[k] += ( diag_aa_I[k] * pi_local_data[i] * pi_local_data[j] );
      }
   }

   for (gidx = 0; gidx < ng; gidx ++) jxf_ParCSRCommHandleDestroy(comm_handle_R[gidx]);
   jxf_TFree(comm_handle_R);
   jxf_ParCSRCommHandleDestroy(comm_handle_I);

   offd_ia = jxf_CSRMatrixI(offdC);
   offd_ja = jxf_CSRMatrixJ(offdC);

   offd_aa_I = jxf_CSRMatrixData(offdI);
   offd_aa_C = jxf_CSRMatrixData(offdC);   

   for (i = 0; i < local_size; i ++)
   {
      for (k = offd_ia[i]; k < offd_ia[i+1]; k ++)
      {
         j = offd_ja[k];
         for (gidx = 0; gidx < ng; gidx ++)
         {
            offd_aa_C[k] += ( jxf_CSRMatrixData(jxf_ParCSRMatrixOffd(ARR[gidx]))[k] *
                                jxf_VectorData(jxf_ParVectorLocalVector(PRR[gidx]))[i] * pr_other_data[gidx][j] );
         }
         offd_aa_C[k] += ( offd_aa_I[k] * pi_local_data[i] * pi_other_data[j] );
      }
   }

   Vei_local_data = jxf_VectorData(jxf_ParVectorLocalVector(VEI));
   Vie_local_data = jxf_VectorData(jxf_ParVectorLocalVector(VIE));

   for (i = 0; i < local_size; i ++)
   {
      k = diag_ia[i];
      diag_add = 0.0;
      for (gidx = 0; gidx < ng; gidx ++)
      {
         diag_add += ((jxf_VectorData(jxf_ParVectorLocalVector(VRE[gidx]))[i] +
                         jxf_VectorData(jxf_ParVectorLocalVector(VER[gidx]))[i]) *
                           jxf_VectorData(jxf_ParVectorLocalVector(PRR[gidx]))[i]);
      }
      diag_add += (Vei_local_data[i] + Vie_local_data[i]) * pi_local_data[i];
      diag_aa_C[k] += diag_add;
   }

   for (gidx = 0; gidx < ng; gidx ++) jxf_TFree(send_buf_r[gidx]);
   jxf_TFree(send_buf_r);
   jxf_TFree(send_buf_i);
   for (gidx = 0; gidx < ng; gidx ++) jxf_TFree(pr_other_data[gidx]);
   jxf_TFree(pr_other_data);
   jxf_TFree(pi_other_data);

   return (ACC);
}

/*!
 * \fn jxf_ParCSRMatrix *jxf_3tApctlCoarseOperator_sp
 * \brief Build interpolation operator for PCTL(single processor case). 
 * \note The sparse pattern of ARR_s, AEE_s, AII_s should be exactly the same.
 *       If not, something needs to be done to guarantee this condition.
 *       For example, reorder each row ascendingly based on the clumn indices.
 * \author peghoty 
 * \date 2011/09/26
 */
jxf_ParCSRMatrix *
jxf_3tApctlCoarseOperator_sp( MPI_Comm       comm,
                             jxf_CSRMatrix  *ARR_s, 
                             jxf_CSRMatrix  *AEE_s, 
                             jxf_CSRMatrix  *AII_s, 
                             jxf_Vector     *VRE_s, 
                             jxf_Vector     *VER_s, 
                             jxf_Vector     *VEI_s, 
                             jxf_Vector     *VIE_s,
                             jxf_ParVector  *PRR,
                             jxf_ParVector  *PII  )
{
   jxf_ParCSRMatrix  *ACC   = NULL;
   jxf_CSRMatrix     *ACC_s = NULL;

   JXF_Int    *ia = NULL;
   JXF_Int    *ja = NULL;
   JXF_Real *aa = NULL;   
   
   JXF_Real *aa_R = jxf_CSRMatrixData(ARR_s);
   JXF_Real *aa_I = jxf_CSRMatrixData(AII_s);

   JXF_Real *Vre_data = jxf_VectorData(VRE_s);
   JXF_Real *Ver_data = jxf_VectorData(VER_s);
   JXF_Real *Vei_data = jxf_VectorData(VEI_s);
   JXF_Real *Vie_data = jxf_VectorData(VIE_s); 
   
   JXF_Real *pr_data = jxf_VectorData(jxf_ParVectorLocalVector(PRR));
   JXF_Real *pi_data = jxf_VectorData(jxf_ParVectorLocalVector(PII));

   JXF_Int i, j, k;  
   JXF_Int n, nz;
   JXF_Real diag_add = 0.0;

#if 1
   jxf_CSRMatrixReorderColumnNumber12(ARR_s);
   jxf_CSRMatrixReorderColumnNumber12(AEE_s);
   jxf_CSRMatrixReorderColumnNumber12(AII_s);  
#endif

   n  = jxf_CSRMatrixNumRows(AEE_s);
   nz = jxf_CSRMatrixNumNonzeros(AEE_s);
   ACC_s = jxf_CSRMatrixCreate(n,n,nz);
   jxf_CSRMatrixInitialize(ACC_s);
   jxf_CSRMatrixCopy(AEE_s, ACC_s, 1);
   
   ia = jxf_CSRMatrixI(ACC_s);
   ja = jxf_CSRMatrixJ(ACC_s);
   aa = jxf_CSRMatrixData(ACC_s); 

   for (i = 0; i < n; i ++)
   {
      for (k = ia[i]; k < ia[i+1]; k ++)
      {
         j = ja[k];
         aa[k] += ( aa_R[k]*pr_data[i]*pr_data[j] + aa_I[k]*pi_data[i]*pi_data[j] );
      }
   }

   for (i = 0; i < n; i ++)
   {
      k = ia[i];
      diag_add = (Vre_data[i] + Ver_data[i])*pr_data[i] + (Vei_data[i] + Vie_data[i])*pi_data[i];
      aa[k] += diag_add;
   } 

   ACC = jxf_CSRMatrixToParCSRMatrix_sp(comm, ACC_s);

   jxf_TFree(ACC_s); 
   
   return (ACC);
}
