
typedef double Real;
typedef int Int;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "mmio_highlevel.h"
#include "my_solver.h"
#include "mpi.h"

void check_correctness(int n, int *row_ptr, int *col_idx, Real *val, Real *x, Real *b);
void store_x(Int n, Real *x, char *filename);
void load_b(Int n, Real *b, char *filename);

Int     global_rank, size;


Int main(Int argc, char **argv)
{
    Int     *row_ptr; // the csr row pointer array of matrix A
    Int     *col_idx; // the csr column index array of matrix A
    Real  *val;  // the csr value array of matrix A
    Real  *x, *b; 

    char    *filename_matrix = argv[1];  // the filename of matrix A
    char    *nrows = argv[2];       // type 0: CSR matrix and vector; type 1: binary matrix and vector
    char    *number_of_zeros = argv[3];       // type 0: CSR matrix and vector; type 1: binary matrix and vector
    char    *filename_x = argv[4];       // the filename of solution vector x
    char    *matrix_tolerance = argv[5]; // the tolerance of input matrix
    char    *pbb = argv[6];              // the number of blocks
    char    *filename_b = argv[7];
    char    *load_t = argv[8];

    Int     n           = atoi(nrows);
    Int     nnzA        = atoi(number_of_zeros);
    Int     pb          = atoi(pbb);
    Int     load_type          = atoi(load_t);
    // char    *sid = argv[5];              // solver id
    Int     iter = 0;                    // the number of iterations
    Real  residule = 0.0, RRN;
    Real  tolerance = atof(matrix_tolerance);
    Real  coff = 1;

    Int     i;
    // Int     solver_id   = atoi(sid);
    Int     solver_id   = 12;
    Int     precond_id  = 3;
    Int     num_functions = 1;
    MPI_Comm comm = MPI_COMM_WORLD;
   
    MPI_Init(NULL,NULL); 
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    
    //-----------------------
    //  命令行修改参数
    //-----------------------
    Int arg_index = 1;
    while (arg_index < argc)
    {
        if ( strcmp(argv[arg_index], "-sid") == 0 )
        {
            arg_index ++;
            solver_id = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-nf") == 0 )
        {
            arg_index ++;
            num_functions = atoi(argv[arg_index++]);
        }else{
            arg_index ++;
        }
    }

    if (global_rank == 0)
    {
        printf("===================================================================\n");
        printf("A  : %s\n", filename_matrix);
        printf("x  : %s\n", filename_x);
        printf("tol: %s\n", matrix_tolerance);
        printf("sid: %d\n", solver_id);
        printf("nf : %d\n", num_functions);
        printf("===================================================================\n");
        
        // int load_type=0;// 0: load fasp CSR matrix; 1: load 3 binary matrix; 2: load MatrixMarket matrix
        if(load_type==1){
            row_ptr = (int *)malloc(sizeof(int) * n+1);
            col_idx = (int *)malloc(sizeof(int) * nnzA);
            val = (double *)malloc(sizeof(double) * nnzA);
            b = (double *)malloc(sizeof(double) * n);
            //load matrix
            printf("load matrix:\n");
            char target[200];
            int size;
            FILE *fp;

            sprintf(target, "%s/Ai.bin\0", filename_matrix);
            fp = fopen(target, "rb");
            if (fp == NULL) {
                printf("cannot open %s \n", target);
                return -1;
            }
            size = fread(row_ptr, sizeof(int), n+1, fp);
            fclose(fp);

            sprintf(target, "%s/Aj.bin\0", filename_matrix);
            fp = fopen(target, "rb");
            if (fp == NULL) {
                printf("cannot open %s \n", target);
                return -1;
            }
            size = fread(col_idx, sizeof(int), nnzA, fp);
            fclose(fp);

            sprintf(target, "%s/Av.bin\0", filename_matrix);
            fp = fopen(target, "rb");
            if (fp == NULL) {
                printf("cannot open %s \n", target);
                return -1;
            }
            size = fread(val, sizeof(double), nnzA, fp);
            fclose(fp);

            printf("pb=%d\n",pb);
            // return 0;
            if(pb==7){
                sprintf(target, "%s/b_t0.bin\0", filename_matrix);
                fp = fopen(target, "rb");
                if (fp == NULL) {
                    printf("cannot open %s \n", target);
                    return -1;
                }
                size = fread(b, sizeof(double), n, fp);
                fclose(fp);
            }
            else{
                sprintf(target, "%s/b.bin\0", filename_matrix);
                fp = fopen(target, "rb");
                if (fp == NULL) {
                    printf("cannot open %s \n", target);
                    return -1;
                }
                size = fread(b, sizeof(double), n, fp);
                fclose(fp);
            }
        }
        else if(load_type==2)
        {
            Int isSymmetricA,m;
            row_ptr = (int *)malloc(sizeof(int) * n+1);
            col_idx = (int *)malloc(sizeof(int) * nnzA);
            val = (double *)malloc(sizeof(double) * nnzA);
            b = (double *)malloc(sizeof(double) * n);
            //load matrix
            printf("load MatrixMarket:\n");
            mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix);
             
            printf("matrix info: m: %d, n: %d, nnz: %d, isSymmertirc: %d\n", m, n, nnzA, isSymmetricA);
            if (m != n)
            {
                printf("Invalid matrix size.\n");
                return 0;
            }
            // load right-hand side vector b
            printf("load right-hand side vector:\n");
            load_b(n, b, filename_b);
        }
        else
        {
            int  i,m,idata;
            double ddata;
            
            // Open input disk file
            FILE *fp = fopen(filename_matrix, "r");
            
            // if ( fp == NULL ) fasp_chkerr(ERROR_OPEN_FILE, filename);
            
            // printf("%s: reading file %s...\n", __FUNCTION__, filename);
            
            // Read CSR matrix
            if ( fscanf(fp, "%d", &m) > 0 ) n = m;
                // printf("idata info: %d\n", m);

            // else {
            //     fasp_chkerr(ERROR_WRONG_FILE, filename);
            // }

            // A->IA = (INT *)fasp_mem_calloc(m+1, sizeof(INT));
            row_ptr = (int *)malloc(sizeof(int) * n+1);
            b = (double *)malloc(sizeof(double) * n);
            load_b(m, b, filename_b);
            // printf("idata info: %d\n", m);
            for ( i = 0; i < n+1; i++ ) {
                if ( fscanf(fp, "%d", &idata) > 0 ) row_ptr[i] = idata;
                // printf("idata info: %d\n", idata);

            }
            // printf("idata info: %d\n", m);
            int nnz = row_ptr[n]-row_ptr[0];
            nnzA = nnz;
            // printf("matrix info: m: %d, n: %d, nnz: %d\n", row_ptr[0], row_ptr[n], nnz);
            
            col_idx = (int *)malloc(sizeof(int) * nnzA);
            val = (double *)malloc(sizeof(double) * nnzA);
            
            for ( i = 0; i < nnz; ++i ) {
                if ( fscanf(fp, "%d", &idata) > 0 ) col_idx[i] = idata;
            }
            
            for ( i = 0; i < nnz; ++i ) {
                if ( fscanf(fp, "%lf", &ddata) > 0 ) val[i]= ddata;
            }
            fclose(fp);
        }

        printf("matrix info: m: %d, n: %d, nnz: %d\n", n, n, nnzA);
        x = (Real *)malloc(sizeof(Real) * n);
        for (i = 0; i < n; i++) {
            x[i] = 0; // 初始化为 0 
            // b[i] = 1; // 初始化为 0 
        }
    }


    struct timeval t_start, t_stop;

    gettimeofday(&t_start, NULL);
       
    JXPAMG_Solver_Interface(argc, argv, comm, n, row_ptr, col_idx, val, x, b, tolerance, solver_id, num_functions, &iter);

    gettimeofday(&t_stop, NULL);
    double total_time = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;

    if (global_rank == 0)
    {
        //store x to a file
        // store_x(n, x, filename_x);

        //check the correctness
        //print the #iteration, residual and total_time
        //RRN = check_correctness(n, row_ptr, col_idx, val, x, b, &residule);
        // check_correctness(n, row_ptr, col_idx, val, x, b);
        // printf("\n\n===============================================================================\n");
        // printf("iteration = %d, residual = %Le, RRN = %Le, total_time = %.5Lf sec\n", iter, residule, RRN, total_time / 1000.0);
        // printf("===============================================================================\n");
        // printf("total_time = %.5lf sec\n", total_time / 1000.0);
    }
    
    MPI_Finalize();
    return 0;
}

//validate the x by b-A*x
// Real check_correctness(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *b, Real *Res)
// {
//     Real *b_new = (Real *)malloc(sizeof(Real) * n);
//     Real *check_b = (Real *)malloc(sizeof(Real) * n);
//     spmv(n, row_ptr, col_idx, val, x, b_new);
//     Int i;
//     for (i = 0; i < n; i++)
//         check_b[i] = b_new[i] - b[i];

//     *Res =  vec2norm(check_b, n);
//     Real b_L2 = vec2norm(b, n);
//     // printf("\nZL: ||b-Ax||_2: %Le, ||b||_2: %Le\n", vec2norm(check_b, n), b_L2);
//     return *Res / b_L2;
// }
void check_correctness(int n, int *row_ptr, int *col_idx, Real *val, Real *x, Real *b)
{
    Real *b_new = (Real *)malloc(sizeof(Real) * n);
    Real *check_b = (Real *)malloc(sizeof(Real) * n);
    spmv(n, row_ptr, col_idx, val, x, b_new);
    int i ;
    for (i = 0; i < n; i++)
        check_b[i] = b_new[i] - b[i];

    double answer1 = vec2norm(check_b, n);
    //double answer2 = max_check(check_b, n);
    double answer3 = answer1 / vec2norm(b, n);
    fprintf(stdout, "Check || AX - B || 2             =  %12.6e\n", answer1);
    //fprintf(stdout, "Check || AX - B || MAX           =  %12.6e\n", answer2);
    fprintf(stdout, "Check || AX - B || 2 / || B || 2 =  %12.6e\n", answer3);

    free(b_new);
    free(check_b);
}
// void check_correctness(int n, int *row_ptr, int *col_idx, long double *val, long double *x, long double *b)
// {
//     long double *b_new = (long double *)malloc(sizeof(long double) * n);
//     long double *check_b = (long double *)malloc(sizeof(long double) * n);
//     spmv(n, row_ptr, col_idx, val, x, b_new);
//     int i ;
//     for (i = 0; i < n; i++)
//         check_b[i] = b_new[i] - b[i];

//     double answer1 = vec2norm(check_b, n);
//     //double answer2 = max_check(check_b, n);
//     double answer3 = answer1 / vec2norm(b, n);
//     fprintf(stdout, "Check || AX - B || 2             =  %12.6e\n", answer1);
//     //fprintf(stdout, "Check || AX - B || MAX           =  %12.6e\n", answer2);
//     fprintf(stdout, "Check || AX - B || 2 / || B || 2 =  %12.6e\n", answer3);

//     free(b_new);
//     free(check_b);
// }


//store x to a file
void store_x(Int n, Real *x, char *filename)
{
    FILE *p = fopen(filename, "w");
    fprintf(p, "%d\n", n);
    Int i;
    for (i = 0; i < n; i++)
        fprintf(p, "%Lf\n", x[i]);
    fclose(p);
}

//load right-hand side vector b
void load_b(Int n, Real *b, char *filename)
{
    FILE *p = fopen(filename, "r");
    Int n_right;
    Int r = fscanf(p, "%d", &n_right);
    // printf("n_right: %d, n: %d.\n", n_right, n);
    if (n_right != n)
    {
        fclose(p);
        printf("Invalid size of b.\n");
        return;
    }
    Int i;
    for (i = 0; i < n_right; i++)
        r = fscanf(p, "%lf", &b[i]);
    fclose(p);
}
