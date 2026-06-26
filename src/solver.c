/*
This is the reference code of SolverChallenge to optimize by the participants
@Code version: 1.0
@Update date: 2021/5/17
@Author: Dechuang Yang, Haocheng Lian

Added some parallel packages: jxpamg
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


Real check_correctness(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *b, Real *Res);
void store_x(Int n, Real *x, char *filename);
void load_b(Int n, Real *b, char *filename);
void read_coo_to_csr(int *m, int *n, int *nnzA, 
    int **row_ptr, int **col_idx, double **val, 
    const char *filename);
void read_csr(int *m, int *n, int *nnzA, 
        int **row_ptr, int **col_idx, double **val, 
        const char *filename);

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
        // mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix);
        read_coo_to_csr(&m, &n, &nnzA, &row_ptr, &col_idx, &val, filename_matrix);
        printf("matrix info: m: %lld, n: %lld, nnz: %lld", m, n, nnzA);
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
    // if (rank == 0) printf("\nJXPAMG_Solver_Interface(int, double -- version):\n");
    JXPAMG_Solver_Interface(argc, argv, MPI_COMM_WORLD, n, row_ptr, col_idx, val, 
                               x, b, tolerance, solver_id, 1, &iter);
    
    // printf("1\n");

    // gettimeofday(&t_stop, NULL);
    // Real total_time = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
    
//     if (rank == 0)
//     {
//         //store x to a file
//         store_x(n, x, filename_x);
// //	store_x(n, RRN, filename_RRN);

//         //check the correctness
//         //print the #iteration, residual and total_time
//         RRN = check_correctness(n, row_ptr, col_idx, val, x, b, &residule);
//         printf("\n\n===============================================================================\n");
//         printf("iteration = %d, residual = %Le, RRN = %Le, total_time = %.5Lf sec\n", iter, residule, RRN, total_time / 1000.0);
//         printf("===============================================================================\n");
//     }
    
    ///
    MPI_Finalize();
    return 0;
}

//validate the x by b-A*x
Real check_correctness(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *b, Real *Res)
{
    Real *b_new = (Real *)malloc(sizeof(Real) * n);
    Real *check_b = (Real *)malloc(sizeof(Real) * n);
    spmv(n, row_ptr, col_idx, val, x, b_new);
    Int i;
    for (i = 0; i < n; i++)
        check_b[i] = b_new[i] - b[i];

    *Res =  vec2norm(check_b, n);
    Real b_L2 = vec2norm(b, n);
    // printf("\nZL: ||b-Ax||_2: %Le, ||b||_2: %Le\n", vec2norm(check_b, n), b_L2);
    return *Res / b_L2;
}

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
        r = fscanf(p, "%lf", &b[i]);
    fclose(p);
}

// Function to read COO format matrix file and convert to CSR
void read_coo_to_csr(int *m, int *n, int *nnzA, 
                    int **row_ptr, int **col_idx, double **val, 
                    const char *filename) {
    FILE *fp;
    int i, row, col;
    double value;
    
    // Open file
    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    // Read dimensions and nnz
    if (fscanf(fp, "%d %d %d\n", m, n, nnzA) != 3) {
        printf("Error: Invalid file format\n");
        fclose(fp);
        exit(1);
    }

    // Allocate temporary arrays for COO
    int *temp_row = (int *)malloc(*nnzA * sizeof(int));
    int *temp_col = (int *)malloc(*nnzA * sizeof(int));
    double *temp_val = (double *)malloc(*nnzA * sizeof(double));

    // Read COO entries
    for (i = 0; i < *nnzA; i++) {
        if (fscanf(fp, "%d %d %lf\n", &row, &col, &value) != 3) {
            printf("Error: Invalid entry at line %d\n", i + 2);
            free(temp_row);
            free(temp_col);
            free(temp_val);
            fclose(fp);
            exit(1);
        }
        temp_row[i] = row - 1;  // Convert to 0-based indexing
        temp_col[i] = col - 1;
        temp_val[i] = value;
    }
    fclose(fp);

    // Allocate CSR arrays
    *row_ptr = (int *)calloc((*m + 1), sizeof(int));
    *col_idx = (int *)malloc(*nnzA * sizeof(int));
    *val = (double *)malloc(*nnzA * sizeof(double));

    // Count entries per row
    for (i = 0; i < *nnzA; i++) {
        (*row_ptr)[temp_row[i] + 1]++;
    }

    // Cumulative sum for row_ptr
    for (i = 1; i <= *m; i++) {
        (*row_ptr)[i] += (*row_ptr)[i - 1];
    }

    // Fill col_idx and val
    int *row_count = (int *)calloc(*m, sizeof(int));
    for (i = 0; i < *nnzA; i++) {
        row = temp_row[i];
        int pos = (*row_ptr)[row] + row_count[row];
        (*col_idx)[pos] = temp_col[i];
        (*val)[pos] = temp_val[i];
        row_count[row]++;
    }

    // Free temporary arrays
    free(temp_row);
    free(temp_col);
    free(temp_val);
    free(row_count);
}

void read_csr(int *m, int *n, int *nnzA, 
    int **row_ptr, int **col_idx, double **val, 
    const char *filename) 
{
FILE *fp;
int i;

// Open file
fp = fopen(filename, "r");
if (fp == NULL) {
printf("Error: Cannot open file %s\n", filename);
exit(1);
}

// Read number of rows and set columns equal to rows
if (fscanf(fp, "%d\n", m) != 1) {
printf("Error: Invalid file format\n");
fclose(fp);
exit(1);
}
*n = *m; // Set columns equal to rows

// Read nnz
if (fscanf(fp, "%d\n", nnzA) != 1) {
printf("Error: Invalid nnz format\n");
fclose(fp);
exit(1);
}

// Allocate CSR arrays
*row_ptr = (int *)malloc((*m + 1) * sizeof(int));
*col_idx = (int *)malloc(*nnzA * sizeof(int));
*val = (double *)malloc(*nnzA * sizeof(double));

// Read row_ptr
for (i = 0; i <= *m; i++) {
if (fscanf(fp, "%d", &(*row_ptr)[i]) != 1) {
  printf("Error: Invalid row_ptr entry at index %d\n", i);
  free(*row_ptr);
  free(*col_idx);
  free(*val);
  fclose(fp);
  exit(1);
}
}

// Read col_idx
for (i = 0; i < *nnzA; i++) {
if (fscanf(fp, "%d", &(*col_idx)[i]) != 1) {
  printf("Error: Invalid col_idx entry at index %d\n", i);
  free(*row_ptr);
  free(*col_idx);
  free(*val);
  fclose(fp);
  exit(1);
}
(*col_idx)[i]--; // Convert to 0-based indexing
}

// Read values
for (i = 0; i < *nnzA; i++) {
if (fscanf(fp, "%lf", &(*val)[i]) != 1) {
  printf("Error: Invalid value entry at index %d\n", i);
  free(*row_ptr);
  free(*col_idx);
  free(*val);
  fclose(fp);
  exit(1);
}
}

fclose(fp);
}