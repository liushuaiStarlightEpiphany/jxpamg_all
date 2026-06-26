#include "jx_pamg.h"
#if JX_USING_MUMPS
#include "dmumps_c.h"
#include "smumps_c.h"

#define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */

JX_Int
jx_Mumps_float( JX_Solver *solve,jx_ParCSRMatrix *pA, jx_ParVector *pb, jx_ParVector *px )
{
    JX_Int relax_error = 0;

    MPI_Comm comm = jx_ParCSRMatrixComm(pA);
    jx_CSRMatrix *A = jx_ParCSRMatrixToCSRMatrixAll(pA);
    JX_Real *ADATA = jx_CSRMatrixData(A);
    JX_Int *AIA = jx_CSRMatrixI(A);
    JX_Int *AJA = jx_CSRMatrixJ(A);
    JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int num_nonzeros = jx_CSRMatrixNumNonzeros(A);
    jx_Vector *b = jx_ParVectorLocalVector(pb);
    JX_Real *bDATA = jx_VectorData(b);
    jx_Vector *x = jx_ParVectorLocalVector(px);
    JX_Real *xDATA = jx_VectorData(x);

    mumps_Interface_float(comm, num_rows,num_nonzeros, AIA, AJA,ADATA,xDATA, bDATA);
    
    // jx_SeqVectorDestroy(b);
    // jx_SeqVectorDestroy(x);
    // jx_CSRMatrixDestroy(A);
    return 0;
}



JX_Int mumps_Interface_float(MPI_Comm comm, int row, int nnz, int *ia, int  *ja, JX_Real *aa, JX_Real *x, JX_Real *b)  
{
    int myid, num_procs;
    SMUMPS_STRUC_C id;
    const  int prtlvl = 1;
    const  int n =  row;
    const  int nz = nnz;
    
    int *irn;
    int *jcn;
    float *a;
    float *rhs;
    int i,j;
    int begin_row, end_row;
    double start_time, end_time;
       
//    MPI_Comm comm = MPI_COMM_WORLD;
    jx_MPI_Comm_rank(comm, &myid );
    jx_MPI_Comm_size(comm, &num_procs);

    if(myid == 0)
    {   
        // First check the matrix format
        if ( ia[0] != 0 && ia[0] != 1 ) {
            printf("### ERROR: Matrix format is wrong -- IA[0] = %d\n", ia[0]);
            return -1;
        }
    }

    if(myid == 0)
    {  
        if ( prtlvl > -1 ) start_time = MPI_Wtime();
        /* Define A and rhs */
        irn = (int *)malloc( sizeof(int)*nz );
        jcn = (int *)malloc( sizeof(int)*nz );
        a   = (float *)malloc( sizeof(float)*nz );
        rhs = (float *)malloc( sizeof(float)*n );
        
        if ( ia[0] == 0 ) { // C-convention
            for (i=0; i<n; i++) {
                begin_row = ia[i]; end_row = ia[i+1];
                for (j=begin_row; j< end_row; j++) {
                    irn[j] = i + 1;
                    jcn[j] = ja[j]+1;
                    a[j]   = aa[j];
                }
            }
        }
        else { // For-convention
            for (i=0; i<n; i++) {
                begin_row = ia[i]-1; end_row = ia[i+1]-1;
                for (j=begin_row; j< end_row; j++) {
                    irn[j] = i + 1;
                    jcn[j] = ja[j];
                    a[j]   = aa[j];
                }
            }
        }
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    /* Initialize a MUMPS instance. */
    id.job=-1; id.par=1; id.sym=0; id.comm_fortran=comm;//0   
    smumps_c(&id);

    if(myid == 0)
    {      
        /* Define the problem on the host */
        id.n = n; id.nz =nz; id.irn=irn; id.jcn=jcn;
        id.a = a; id.rhs = rhs;
        // /* No outputs */
        // id.ICNTL(1) = -1;
        // id.ICNTL(2) = -1;
        // id.ICNTL(3) = -1;
        // id.ICNTL(4) =  0;
    }

    
    /* Call the MUMPS package */
    if(myid == 0)
    {
        for (i=0; i<n; i++) rhs[i] = b[i];
    }

    // id.job = 6; 
    // smumps_c(&id);
     
    id.job = 1; 
    smumps_c(&id);

    id.job = 2; 
    smumps_c(&id);
    if(myid == 0)
    { 
        printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    id.job = 3; 
    smumps_c(&id);

    if(myid == 0)
    { 
        printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    if(myid == 0)
    {
        for (i=0; i<n; i++) x[i] = id.rhs[i];
    }

    if(myid == 0)
    { 
        printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    id.job = -2;
    smumps_c(&id); /* Terminate instance */

    if(myid == 0)
    { 
        printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }
   
    if(myid == 0)
    {
        free(irn);
        free(jcn);
        free(a);
        free(rhs);
        
        if ( prtlvl > -1 ) {
            end_time = MPI_Wtime();
            double solve_time = end_time - start_time;
            printf("MUMPS costs %f seconds.\n", solve_time);
        }
    
#if DEBUG_MODE
    printf("### DEBUG: MUMPS_Interface ... [Finish]\n");
#endif
    }

    return 1;
}

JX_Int
jx_Mumps_double( JX_Solver *solve,jx_ParCSRMatrix *pA, jx_ParVector *pb, jx_ParVector *px )
{
    JX_Int relax_error = 0;

    MPI_Comm comm = jx_ParCSRMatrixComm(pA);
    jx_CSRMatrix *A = jx_ParCSRMatrixToCSRMatrixAll(pA);
    JX_Real *ADATA = jx_CSRMatrixData(A);
    JX_Int *AIA = jx_CSRMatrixI(A);
    JX_Int *AJA = jx_CSRMatrixJ(A);
    JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int num_nonzeros = jx_CSRMatrixNumNonzeros(A);
    jx_Vector *b = jx_ParVectorLocalVector(pb);
    JX_Real *bDATA = jx_VectorData(b);
    jx_Vector *x = jx_ParVectorLocalVector(px);
    JX_Real *xDATA = jx_VectorData(x);

    mumps_Interface_double(comm, num_rows,num_nonzeros, AIA, AJA,ADATA,xDATA, bDATA);
    
    return 0;
}

int mumps_Interface_double(MPI_Comm comm, int row, int nnz, int *ia, int  *ja, JX_Real *aa, JX_Real *x, JX_Real *b)  
{
    int myid, num_procs;
    DMUMPS_STRUC_C id;
    const  int prtlvl = 1;
    const  int n =  row;
    const  int nz = nnz;
    int *IA = ia;
    int *JA = ja;
    JX_Real *AA =  aa;
    
    int *irn;
    int *jcn;
    double *a;
    double *rhs;
    int i,j;
    int begin_row, end_row;
    double start_time, end_time;
       
 //    MPI_Comm comm = MPI_COMM_WORLD;
    jx_MPI_Comm_rank(comm, &myid );
    jx_MPI_Comm_size(comm, &num_procs);

    if(myid == 0)
    {
        // First check the matrix format
        if ( IA[0] != 0 && IA[0] != 1 ) {
            printf("### ERROR: Matrix format is wrong -- IA[0] = %d\n", IA[0]);
            return -1;
        }
    }

    if(myid == 0)
    {  
        if ( prtlvl > -1 ) start_time = MPI_Wtime();
        /* Define A and rhs */
        irn = (int *)malloc( sizeof(int)*nz );
        jcn = (int *)malloc( sizeof(int)*nz );
        a   = (double *)malloc( sizeof(double)*nz );
        rhs = (double *)malloc( sizeof(double)*n );
        
        if ( IA[0] == 0 ) { // C-convention
            for (i=0; i<n; i++) {
                begin_row = IA[i]; end_row = IA[i+1];
                for (j=begin_row; j< end_row; j++) {
                    irn[j] = i + 1;
                    jcn[j] = JA[j]+1;
                    a[j]   = AA[j];
                }
            }
        }
        else { // For-convention
            for (i=0; i<n; i++) {
                begin_row = IA[i]-1; end_row = IA[i+1]-1;
                for (j=begin_row; j< end_row; j++) {
                    irn[j] = i + 1;
                    jcn[j] = JA[j];
                    a[j]   = AA[j];
                }
            }
        }
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    /* Initialize a MUMPS instance. */
    id.job=-1; id.par=1; id.sym=0; id.comm_fortran=comm;//0   
    dmumps_c(&id);

    if(myid == 0)
    {      
        /* Define the problem on the host */
        id.n = n; id.nz =nz; id.irn=irn; id.jcn=jcn;
        id.a = a; 
    }

    /* No outputs */
    id.ICNTL(1) = -1;
    id.ICNTL(2) = -1;
    id.ICNTL(3) = -1;
    id.ICNTL(4) =  0;
    
    

    // id.job = 6; 
    // dmumps_c(&id);
    id.job = 1; 
    dmumps_c(&id);

    id.job = 2; 
    dmumps_c(&id);

    /* Call the MUMPS package */
    if(myid == 0)
    {
        id.rhs = rhs;
        for (i=0; i<n; i++) rhs[i] = b[i];
    }

    id.job = 3; 
    dmumps_c(&id); 


    if(myid == 0)
    {
        for (i=0; i<n; i++) x[i] = id.rhs[i];
    }

    id.job = -2;
    dmumps_c(&id); /* Terminate instance */
   
    if(myid == 0)
    {
        free(irn);
        free(jcn);
        free(a);
        free(rhs);
        
        if ( prtlvl > -1 ) {
            end_time = MPI_Wtime();
            double solve_time = end_time - start_time;
            printf("MUMPS costs %f seconds.\n", solve_time);
        }
    
#if DEBUG_MODE
    printf("### DEBUG: MUMPS_Interface ... [Finish]\n");
#endif
    }

    return 1;
}


JX_Int
jx_Mumps_double_setup(jx_DMUMPS_data *mumps_data,jx_ParCSRMatrix *pA)
{
    MPI_Comm comm = jx_ParCSRMatrixComm(pA);
    jx_CSRMatrix *A = jx_ParCSRMatrixToCSRMatrixAll(pA);
    JX_Real *ADATA = jx_CSRMatrixData(A);
    JX_Int *AIA = jx_CSRMatrixI(A);
    JX_Int *AJA = jx_CSRMatrixJ(A);
    // JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int num_rows = (A) -> num_rows;
    JX_Int num_nonzeros = jx_CSRMatrixNumNonzeros(A);

    int myid, num_procs;
    const  int prtlvl = 1;
    (mumps_data)->num_rows = num_rows;
    (mumps_data)->num_nonzeros = num_nonzeros;
 
    int *irn;
    int *jcn;
    double *a;
    double *rhs;

    int i,j;
    int begin_row, end_row;
    double start_time, end_time;
       
    jx_MPI_Comm_rank(comm, &myid);
    jx_MPI_Comm_size(comm, &num_procs);



    if(myid == 0)
    {
        // First check the matrix format
        if ( AIA[0] != 0 && AIA[0] != 1 ) {
            printf("### ERROR: Matrix format is wrong -- IA[0] = %d\n", AIA[0]);
            return -1;
        }
    }

    if(myid == 0)
    {  
        if ( prtlvl > -1 ) start_time = MPI_Wtime();
        /* Define A and rhs */
        irn = (int *)malloc( sizeof(int)*num_nonzeros);
        jcn = (int *)malloc( sizeof(int)*num_nonzeros );
        a   = (double *)malloc( sizeof(double)*num_nonzeros);
        rhs   = (double *)malloc( sizeof(double)*num_rows);
        
        if ( AIA[0] == 0 ) { // C-convention
            for (i=0; i<num_rows; i++) {
                begin_row = AIA[i]; end_row = AIA[i+1];
                for (j=begin_row; j< end_row; j++) {
                    irn[j] = i + 1;
                    jcn[j] = AJA[j]+1;
                    a[j]   = ADATA[j];
                }
            }
        }
        else { // For-convention
            for (i=0; i<num_rows; i++) {
                begin_row = AIA[i]-1; end_row = AIA[i+1]-1;
                for (j=begin_row; j< end_row; j++) {
                    irn[j] = i + 1;
                    jcn[j] = AJA[j];
                    a[j]   = ADATA[j];
                }
            }
        }
        printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    /* Initialize a MUMPS instance. */
    mumps_data->id.job=-1; mumps_data->id.par=1; mumps_data->id.sym=0; mumps_data->id.comm_fortran=comm;//0   
    dmumps_c(&mumps_data->id);
    
    printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );



    if(myid == 0)
    {      
        /* Define the problem on the host */
        mumps_data->id.n = num_rows; 
        mumps_data->id.nz =num_nonzeros; 
        mumps_data->id.irn=irn; 
        mumps_data->id.jcn=jcn;
        mumps_data->id.a = a; 
        mumps_data->id.rhs = rhs; 
    }

    /* No outputs */
    mumps_data->id.ICNTL(1) = -1;
    mumps_data->id.ICNTL(2) = -1;
    mumps_data->id.ICNTL(3) = -1;
    mumps_data->id.ICNTL(4) =  0;
    
    /* Call the MUMPS package */
    mumps_data->id.job = 1; 
    dmumps_c(&mumps_data->id);

    mumps_data->id.job = 2; 
    dmumps_c(&mumps_data->id);

    if(myid == 0)
    {
        if ( prtlvl > -1 ) {
            end_time = MPI_Wtime();
            double solve_time = end_time - start_time;
            printf("MUMPS SETUP costs %f seconds.\n", solve_time);
        }
    
#if DEBUG_MODE
    printf("### DEBUG: MUMPS_Interface ... [Finish]\n");
#endif
    }

    
    return 0;
}

JX_Int
jx_Mumps_double_solve(jx_DMUMPS_data *mumps_data,jx_ParCSRMatrix *pA,jx_ParVector *pb, jx_ParVector *px)
{
    MPI_Comm comm = jx_ParCSRMatrixComm(pA);
    int myid, num_procs;
    jx_MPI_Comm_rank(comm, &myid);
    jx_MPI_Comm_size(comm, &num_procs);

    jx_Vector *b = jx_ParVectorLocalVector(pb);
    JX_Real *bDATA = jx_VectorData(b);
    jx_Vector *x = jx_ParVectorLocalVector(px);
    JX_Real *xDATA = jx_VectorData(x);
    double end_time;
    double start_time;
    
    int i;

    JX_Int n = mumps_data->num_rows;
    JX_Int nz = mumps_data->num_nonzeros;
    // printf("%d seconds.\n", n);
    // printf("%d seconds.\n", nz);

    // DMUMPS_STRUC_C id = mumps_data->id;
    const  int prtlvl = 1;


    if(myid == 0)
    {  
        if ( prtlvl > -1 ) start_time = MPI_Wtime();

        for (i=0; i<n; i++) mumps_data->id.rhs[i] = bDATA[i];
    }

    mumps_data->id.comm_fortran=comm;
    mumps_data->id.job = 3; 
    dmumps_c(&mumps_data->id); 


    if(myid == 0)
    {
        for (i=0; i<n; i++) xDATA[i] = mumps_data->id.rhs[i];
    }

    // mumps_data->id.job = -2;
    // dmumps_c(&mumps_data->id); /* Terminate instance */

    if(myid == 0)
    {   
        if ( prtlvl > -1 ) {
            end_time = MPI_Wtime();
            double solve_time = end_time - start_time;
            printf("MUMPS solve costs %f seconds.\n", solve_time);
        }
    
#if DEBUG_MODE
    printf("### DEBUG: MUMPS_Interface ... [Finish]\n");
#endif
    }  
    return 0;
}


JX_Int
jx_Mumps_float_setup(jx_SMUMPS_data *mumps_data,jx_ParCSRMatrix *pA)
{
    MPI_Comm comm = jx_ParCSRMatrixComm(pA);
    jx_CSRMatrix *A = jx_ParCSRMatrixToCSRMatrixAll(pA);
    JX_Real *ADATA = jx_CSRMatrixData(A);
    JX_Int *AIA = jx_CSRMatrixI(A);
    JX_Int *AJA = jx_CSRMatrixJ(A);
    // JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int num_rows = (A) -> num_rows;
    JX_Int num_nonzeros = jx_CSRMatrixNumNonzeros(A);

    int myid, num_procs;
    const  int prtlvl = 1;
    (mumps_data)->num_rows = num_rows;
    (mumps_data)->num_nonzeros = num_nonzeros;
 
    int *irn;
    int *jcn;
    float *a;
    float *rhs;
    int i,j;
    int begin_row, end_row;
    double start_time, end_time;
       
    jx_MPI_Comm_rank(comm, &myid);
    jx_MPI_Comm_size(comm, &num_procs);



    if(myid == 0)
    {
        // First check the matrix format
        if ( AIA[0] != 0 && AIA[0] != 1 ) {
            printf("### ERROR: Matrix format is wrong -- IA[0] = %d\n", AIA[0]);
            return -1;
        }
    }

    if(myid == 0)
    {  
        if ( prtlvl > -1 ) start_time = MPI_Wtime();
        /* Define A and rhs */
        irn = (int *)malloc( sizeof(int)*num_nonzeros);
        jcn = (int *)malloc( sizeof(int)*num_nonzeros );
        a   = (float *)malloc( sizeof(float)*num_nonzeros);
        rhs   = (float *)malloc( sizeof(float)*num_rows);

        if ( AIA[0] == 0 ) { // C-convention
            for (i=0; i<num_rows; i++) {
                begin_row = AIA[i]; end_row = AIA[i+1];
                for (j=begin_row; j< end_row; j++) {
                    irn[j] = i + 1;
                    jcn[j] = AJA[j]+1;
                    a[j]   = ADATA[j];
                }
            }
        }
        else { // For-convention
            for (i=0; i<num_rows; i++) {
                begin_row = AIA[i]-1; end_row = AIA[i+1]-1;
                for (j=begin_row; j< end_row; j++) {
                    irn[j] = i + 1;
                    jcn[j] = AJA[j];
                    a[j]   = ADATA[j];
                }
            }
        }
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    /* Initialize a MUMPS instance. */
    mumps_data->id.job=-1; mumps_data->id.par=1; mumps_data->id.sym=0; mumps_data->id.comm_fortran=comm;//0   
    smumps_c(&mumps_data->id);

    if(myid == 0)
    {      
        /* Define the problem on the host */
        mumps_data->id.n = num_rows; 
        mumps_data->id.nz =num_nonzeros; 
        mumps_data->id.irn=irn; 
        mumps_data->id.jcn=jcn;
        mumps_data->id.a = a; 
        mumps_data->id.rhs = rhs; 
        
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    /* No outputs */
    mumps_data->id.ICNTL(1) = -1;
    mumps_data->id.ICNTL(2) = -1;
    mumps_data->id.ICNTL(3) = -1;
    mumps_data->id.ICNTL(4) =  0;
    
    /* Call the MUMPS package */
    mumps_data->id.job = 1; 
    smumps_c(&mumps_data->id);

    if(myid == 0)
    { 
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    mumps_data->id.job = 2; 
    smumps_c(&mumps_data->id);

    if(myid == 0)
    { 
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    if(myid == 0)
    {
        if ( prtlvl > -1 ) {
            end_time = MPI_Wtime();
            double solve_time = end_time - start_time;
            printf("MUMPS SETUP costs %f seconds.\n", solve_time);
        }
    
#if DEBUG_MODE
    printf("### DEBUG: MUMPS_Interface ... [Finish]\n");
#endif
    }

    
    return 0;
}

JX_Int
jx_Mumps_float_solve(jx_SMUMPS_data *mumps_data,jx_ParCSRMatrix *pA,jx_ParVector *pb, jx_ParVector *px)
{
    MPI_Comm comm = jx_ParCSRMatrixComm(pA);

    jx_Vector *b = jx_ParVectorLocalVector(pb);
    JX_Real *bDATA = jx_VectorData(b);
    jx_Vector *x = jx_ParVectorLocalVector(px);
    JX_Real *xDATA = jx_VectorData(x);
    double end_time;
    double start_time;
    
    int i;

    JX_Int n = mumps_data->num_rows;
    JX_Int nz = mumps_data->num_nonzeros;
    // printf("%d seconds.\n", n);
    // printf("%d seconds.\n", nz);

    // DMUMPS_STRUC_C id = mumps_data->id;
    const  int prtlvl = 1;

    int myid, num_procs;
    jx_MPI_Comm_rank(comm, &myid);
    jx_MPI_Comm_size(comm, &num_procs);

    if(myid == 0)
    { 
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    if(myid == 0)
    {  
        if ( prtlvl > -1 ) start_time = MPI_Wtime();
        // rhs = (float *)malloc( sizeof(float)*n );
        // mumps_data->id.rhs = rhs;

        for (i=0; i<n; i++) mumps_data->id.rhs[i] = bDATA[i];

        // mumps_data->id.rhs = rhs;
    }

    if(myid == 0)
    { 
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }


    mumps_data->id.job = 3; 
    smumps_c(&mumps_data->id); 

    if(myid == 0)
    { 
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }


    if(myid == 0)
    {
        for (i=0; i<n; i++) xDATA[i] = mumps_data->id.rhs[i];
    }

    if(myid == 0)
    { 
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
    }

    // mumps_data->id.job = -2;
    // smumps_c(&mumps_data->id); /* Terminate instance */

    if(myid == 0)
    {   
        // free(rhs);
        // printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );

        if ( prtlvl > -1 ) {
            end_time = MPI_Wtime();
            double solve_time = end_time - start_time;
            printf("MUMPS solve costs %f seconds.\n", solve_time);
        }
    
#if DEBUG_MODE
    printf("### DEBUG: MUMPS_Interface ... [Finish]\n");
#endif
    }  
    return 0;
}



#endif