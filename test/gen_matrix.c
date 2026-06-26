#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char *argv[])
{
    int n, i;
    char mat_file[256] = "matrix.mtx";
    char b_file[256] = "b.txt";

    if (argc < 2) {
        printf("Usage: %s <size> [matrix_file] [b_file]\n", argv[0]);
        printf("  e.g.: %s 100\n", argv[0]);
        printf("  e.g.: %s 100 mymat.mtx myb.txt\n", argv[0]);
        return 1;
    }

    n = atoi(argv[1]);
    if (n < 2) { printf("size must be >= 2\n"); return 1; }
    if (argc >= 3) strcpy(mat_file, argv[2]);
    if (argc >= 4) strcpy(b_file, argv[3]);

    /* known solution x[i] = i+1 */
    double *x = (double *)malloc(n * sizeof(double));
    for (i = 0; i < n; i++) x[i] = (double)(i + 1);

    int nnz = 3 * n - 2;
    double diag = 5.0;
    double off  = 1.0;

    FILE *fp = fopen(mat_file, "w");
    if (!fp) { perror(mat_file); return 1; }
    fprintf(fp, "%d %d %d\n", n, n, nnz);
    for (i = 0; i < n; i++) {
        if (i > 0)     fprintf(fp, "%d %d %lg\n", i + 1, i,     off);
        fprintf(fp, "%d %d %lg\n", i + 1, i + 1, diag);
        if (i < n - 1) fprintf(fp, "%d %d %lg\n", i + 1, i + 2, off);
    }
    fclose(fp);
    printf("Wrote %s (%d x %d, %d nnz)\n", mat_file, n, n, nnz);

    /* b = A * x */
    double *b = (double *)calloc(n, sizeof(double));
    for (i = 0; i < n; i++) {
        if (i > 0)     b[i] += off * x[i - 1];
        b[i] += diag * x[i];
        if (i < n - 1) b[i] += off * x[i + 1];
    }

    fp = fopen(b_file, "w");
    if (!fp) { perror(b_file); return 1; }
    fprintf(fp, "%d\n", n);
    for (i = 0; i < n; i++) fprintf(fp, "%.15lg\n", b[i]);
    fclose(fp);
    printf("Wrote %s\n", b_file);

    free(x);
    free(b);
    return 0;
}
