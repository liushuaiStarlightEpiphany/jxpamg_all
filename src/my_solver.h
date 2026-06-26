/*
A simple serial CG iterative method to solve a linear system
@Code version: 1.0
@Update date: 2021/5/17
@Author: Dechuang Yang,Haocheng Lian
*/
// Multiply a csr matrix with a vector x, and get the resulting vector y
// void spmv(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *y)
// {
//     Int i, j;
//     for (i = 0; i < n; i++)
//     {
//         y[i] = 0.0;
//         for (j = row_ptr[i]; j < row_ptr[i + 1]; j++)
//             y[i] += val[j] * x[col_idx[j]];
//     }
// }
void spmv(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *y)
{
    int i,j;
    for (i = 0; i < n; i++)
    {
        y[i] = 0.0;
        Real c = 0.0;
        for (j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            long double num = val[j] * x[col_idx[j]];
            long double z = num - c;
            long double t = y[i] + z;
            c = (t - y[i]) - z;
            y[i] = t;
        }
    }
}

// Calculate the 2-norm of a vector
// Real vec2norm(Real *x, Int n)
// {
//     Real sum = 0.0;
//     Int i;
//     for (i = 0; i < n; i++)
//         sum += x[i] * x[i];
//     return sqrt(sum);
// }

Real vec2norm(Real *x, int n)
{
    Real sum = 0.0;
    Real c = 0.0;
    int i;
    for (i = 0; i < n; i++)
    {
        long double num = x[i] * x[i];
        long double z = num - c;
        long double t = sum + z;
        c = (t - sum) - z;
        sum = t;
    }

    return sqrt(sum);
}

// Compute dot product of two vectors, and return the result
Real dotproduct(Real *x1, Real *x2, Int n)
{
    Real sum = 0.0;
    Int i;
    for (i = 0; i < n; i++)
        sum += x1[i] * x2[i];
    return sum;
}

// Solve a linear system by using a simple CG iterative method
// void my_solver(Int n, Int *row_ptr, Int *col_idx, Real *val,
//                Real *x, Real *b, Int *iter, Real tolerance)
// {
//     Int maxiter = 1000;
//     memset(x, 0, sizeof(Real) * n);
//     Real *r = (Real *)malloc(sizeof(Real) * n);
//     Real *y = (Real *)malloc(sizeof(Real) * n);
//     Real *p = (Real *)malloc(sizeof(Real) * n);
//     Real *q = (Real *)malloc(sizeof(Real) * n);
//     *iter = 0;
//     Real norm = 0.0;
//     Real rho = 0.0;
//     Real rho_1 = 0.0;
//     Real error = 0.0;
//     Int i;
//     spmv(n, row_ptr, col_idx, val, x, y);
//     for (i = 0; i < n; i++)
//         r[i] = b[i] - y[i];

//     do
//     {
//         rho = dotproduct(r, r, n);
//         if (*iter == 0)
//         {
//             for (i = 0; i < n; i++)
//                 p[i] = r[i];
//         }
//         else
//         {
//             Real beta = rho / rho_1;
//             for (i = 0; i < n; i++)
//                 p[i] = r[i] + beta * p[i];
//         }

//         spmv(n, row_ptr, col_idx, val, p, q);
//         Real alpha = rho / dotproduct(p, q, n);

//         for (i = 0; i < n; i++)
//             x[i] += alpha * p[i];
//         for (i = 0; i < n; i++)
//             r[i] += -alpha * q[i];

//         rho_1 = rho;
//         error = vec2norm(r, n);
//         *iter += 1;

//         if (error < tolerance)
//             break;
//     } while (*iter < maxiter);

//     free(r);
//     free(y);
//     free(p);
//     free(q);
// }