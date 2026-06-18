/*
This is the reference code of SolverChallenge to optimize by the participants
@Code version: 1.0
@Update date: 2021/5/17
@Author: Dechuang Yang, Haocheng Lian

Added some parallel packages: jxfpamg
@Update date: 2021/5/28
@Author: Li Zhao
*/

// typedef long double Real;
typedef double Real;
typedef int Int;
// typedef long long int Int;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "mmio_highlevel.h"
#include "my_solver.h"
#include "mpi.h"


// Real check_correctness(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *b, Real *Res);
// void check_correctness(int n, int *row_ptr, int *col_idx, long double *val, long double *x, long double *b);
void check_correctness(int n, int *row_ptr, int *col_idx, Real *val, Real *x, Real *b);
void store_x(Int n, Real *x, char *filename);
void load_b(Int n, Real *b, char *filename);

Int main(Int argc, char **argv)
{
    Int     m, n, nnzA, isSymmetricA;
    Int     *row_ptr; // the csr row pointer array of matrix A
    Int     *col_idx; // the csr column index array of matrix A
    Real  *val;  // the csr value array of matrix A
    Real  *x, *b; 

    char    *filename_matrix = argv[1];  // the filename of matrix A
    char    *filename_b = argv[2];       // the filename of right-hand side vector b
    char    *filename_x = argv[3];       // the filename of solution vector x
    char    *matrix_tolerance = argv[4]; // the tolerance of input matrix
    // char    *sid = argv[5];              // solver id
    Int     iter = 0;                    // the number of iterations
    Real  residule = 0.0, RRN;
    Real  tolerance = atof(matrix_tolerance);
    Real  coff = 1;

    int     rank;
    int     size;
    Int     i;
    // Int     solver_id   = atoi(sid);
    Int     solver_id   = 12;
    Int     precond_id  = 3;
    Int     num_functions = 1;
   
    
    // system("free -h");
    MPI_Init(NULL,NULL); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
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

    if (rank == 0)
    {
        printf("===================================================================\n");
        printf("A  : %s\n", filename_matrix);
        printf("b  : %s\n", filename_b);
        printf("x  : %s\n", filename_x);
        printf("tol: %s\n", matrix_tolerance);
        printf("sid: %d\n", solver_id);
        printf("nf : %d\n", num_functions);
        printf("===================================================================\n");

        printf("\n\n+++++++++++++++++++++ MPI using %d processors +++++++++++++++++++++\n\n", size);

        //load matrix
        printf("load matrix:\n");
        mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix);
    //     n = 5;
    //     m = n;
	//     nnzA = 13;
    // 	row_ptr    = (int*)malloc(sizeof(int) * (n+1));
    // 	col_idx    = (int*)malloc(sizeof(int) * nnzA);
    // 	val    = (Real*)malloc(sizeof(Real) * nnzA);	
    // 	x    = (Real*)malloc(sizeof(Real) * n);
    // 	b    = (Real*)malloc(sizeof(Real) * n);
	// *(row_ptr+0) = 0;
	// *(row_ptr+1) = 2;
	// *(row_ptr+2) = 5;
	// *(row_ptr+3) = 8;
	// *(row_ptr+4) = 11;
	// *(row_ptr+5) = 13;
	
	// *(col_idx+0) = 0;
	// *(col_idx+1) = 1;
	// *(col_idx+2) = 0;
	// *(col_idx+3) = 1;
	// *(col_idx+4) = 2;
	// *(col_idx+5) = 1;
	// *(col_idx+6) = 2;
	// *(col_idx+7) = 3;
	// *(col_idx+8) = 2;
	// *(col_idx+9) = 3;
	// *(col_idx+10) = 4;
	// *(col_idx+11) = 3;
	// *(col_idx+12) = 4;
	
	// *(val+0) = 2;
	// *(val+1) = -1;
	// *(val+2) = -1;
	// *(val+3) = 2;
	// *(val+4) = -1;
	// *(val+5) = -1;
	// *(val+6) = 2;
	// *(val+7) = -1;
	// *(val+8) = -1;
	// *(val+9) = 2;
	// *(val+10) = -1;
	// *(val+11) = -1;
	// *(val+12) = 2;
	
	// *(b+0) = 1;
	// *(b+1) = 0;
	// *(b+2) = 0;
	// *(b+3) = 0;
	// *(b+4) = 1;
 
        printf("matrix info: m: %lld, n: %lld, nnz: %lld, isSymmertirc: %lld\n", m, n, nnzA, isSymmetricA);
        if (m != n)
        {
            printf("Invalid matrix size.\n");
            return 0;
        }

        x = (Real *)malloc(sizeof(Real) * n);
        b = (Real *)malloc(sizeof(Real) * m);

        // load right-hand side vector b
        printf("load right-hand side vector:\n");
        load_b(n, b, filename_b);
        
        for (i = 0; i < n; i++) x[i] = 0; // 初始化为 0 

    }



    // solve x and record wall-time
    struct timeval t_start, t_stop;
    gettimeofday(&t_start, NULL);
    


    //-------------------------------------------------------------------------
    //  boomer_amg
    //-------------------------------------------------------------------------  
    if (rank == 0) printf("\nJXFPAMG_Solver_Interface(int, long double -- version):\n");    
    JXFPAMG_Solver_Interface(argc, argv, n, row_ptr, col_idx, val, x, b, tolerance, solver_id, num_functions, &iter);
    


    gettimeofday(&t_stop, NULL);
    double total_time = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
    
    if (rank == 0)
    {
        //store x to a file
        // store_x(n, x, filename_x);

        //check the correctness
        //print the #iteration, residual and total_time
        //RRN = check_correctness(n, row_ptr, col_idx, val, x, b, &residule);
        check_correctness(n, row_ptr, col_idx, val, x, b);
        // printf("\n\n===============================================================================\n");
        // printf("iteration = %d, residual = %Le, RRN = %Le, total_time = %.5Lf sec\n", iter, residule, RRN, total_time / 1000.0);
        // printf("===============================================================================\n");
        printf("total_time = %.5lf sec\n", total_time / 1000.0);
    }
    
    ///
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
    fprintf(p, "%lld\n", n);
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
    Int r = fscanf(p, "%lld", &n_right);
    // printf("n_right: %lld, n: %lld.\n", n_right, n);
    if (n_right != n)
    {
        fclose(p);
        printf("Invalid size of b.\n");
        return;
    }
    Int i;
    for (i = 0; i < n_right; i++)
        r = fscanf(p, "%Lf", &b[i]);
    fclose(p);
}
