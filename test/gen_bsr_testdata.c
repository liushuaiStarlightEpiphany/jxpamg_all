#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <n_blocks> <block_size> <bsr_mat> <bsr_rhs>\n", prog);
    fprintf(stderr, "       [type] [sol_type] [sol_file] [csr_mat] [csr_rhs]\n");
    fprintf(stderr, "  type:     0 = 1D Laplacian (tridiagonal, default)\n");
    fprintf(stderr, "            1 = 2D 5-pt Laplacian (n_blocks must be a perfect square)\n");
    fprintf(stderr, "            2 = random SPD (diagonal dominant)\n");
    fprintf(stderr, "            3 = full dense block matrix\n");
    fprintf(stderr, "            4 = 3D 7-pt Laplacian (n_blocks must be a perfect cube)\n");
    fprintf(stderr, "  sol_type: 0 = x = (1, 1, ..., 1)^T (default)\n");
    fprintf(stderr, "            1 = x_i = (i+1)/dim  (linear ramp)\n");
    fprintf(stderr, "            2 = x_i = sin(2*pi*i/dim)\n");
    fprintf(stderr, "            3 = x_i = random uniform [0,1]\n");
    fprintf(stderr, "            4 = read x from sol_file\n");
    fprintf(stderr, "  csr_mat:  optional CSR matrix output file (read_csr format)\n");
    fprintf(stderr, "  csr_rhs:  optional CSR RHS output file (load_b format)\n");
}

static void fill_scale(double *blk, int bs, double s) {
    for (int r = 0; r < bs; r++)
        for (int c = 0; c < bs; c++)
            blk[r * bs + c] = (r == c) ? s : 0.0;
}

static void write_bsr(FILE *fp, int n, int bs, int nnz, int *IA, int *JA, double *val) {
    int bnnz = bs * bs;
    int total_vals = nnz * bnnz;
    fprintf(fp, "%d %d %d\n", n, n, nnz);
    fprintf(fp, "%d\n", bs);
    fprintf(fp, "0\n");
    fprintf(fp, "%d\n", n + 1);
    for (int i = 0; i <= n; i++) fprintf(fp, "%d\n", IA[i]);
    fprintf(fp, "%d\n", nnz);
    for (int i = 0; i < nnz; i++) fprintf(fp, "%d\n", JA[i]);
    fprintf(fp, "%d\n", total_vals);
    for (int i = 0; i < total_vals; i++) fprintf(fp, "%.14e\n", val[i]);
}

static void write_rhs(FILE *fp, int dim, double *rhs) {
    fprintf(fp, "%d\n", dim);
    for (int i = 0; i < dim; i++)
        fprintf(fp, "%d %.14e\n", i, rhs[i]);
}

/* y = A * x  (scalar-level SpMV: A is n*bs x n*bs, stored in BSR format) */
static void bsr_spmv(int n, int bs, int nnz, int *IA, int *JA, double *val,
                     const double *x, double *y) {
    int bnnz = bs * bs;
    for (int br = 0; br < n; br++) {
        for (int p = IA[br] - 1; p < IA[br + 1] - 1; p++) {
            int bc = JA[p] - 1;
            double *blk = val + p * bnnz;
            for (int r = 0; r < bs; r++) {
                double sum = 0.0;
                for (int c = 0; c < bs; c++)
                    sum += blk[r * bs + c] * x[bc * bs + c];
                y[br * bs + r] += sum;
            }
        }
    }
}

/* ========== CSR output (expand BSR blocks to scalar CSR) ========== */

static void write_coo(FILE *fp, int n, int bs, int nnz, int *IA, int *JA, double *val) {
    int dim = n * bs;
    int bnnz = bs * bs;

    /* Count and write COO entries (read_coo_to_csr format: 1-based row/col) */
    int coo_nnz = 0;
    for (int br = 0; br < n; br++)
        for (int p = IA[br] - 1; p < IA[br + 1] - 1; p++)
            for (int r = 0; r < bs; r++)
                for (int c = 0; c < bs; c++)
                    if (val[p * bnnz + r * bs + c] != 0.0)
                        coo_nnz++;

    fprintf(fp, "%d %d %d\n", dim, dim, coo_nnz);
    for (int br = 0; br < n; br++) {
        for (int p = IA[br] - 1; p < IA[br + 1] - 1; p++) {
            int bc = JA[p] - 1;
            double *blk = val + p * bnnz;
            for (int r = 0; r < bs; r++) {
                int scal_row = br * bs + r;
                for (int c = 0; c < bs; c++) {
                    double v = blk[r * bs + c];
                    if (v != 0.0)
                        fprintf(fp, "%d %d %.14e\n", scal_row + 1, bc * bs + c + 1, v);
                }
            }
        }
    }
}

static void write_rhs_csr(FILE *fp, int dim, double *rhs) {
    fprintf(fp, "%d\n", dim);
    for (int i = 0; i < dim; i++)
        fprintf(fp, "%.14e\n", rhs[i]);
}

/* ========== matrix generators (adapted from gen_bsr_matrix.c) ========== */

static int gen_1d_laplacian(int n, int bs, int **pIA, int **pJA, double **pval) {
    int nnz = 3 * n - 2;
    int bnnz = bs * bs;
    int *IA = calloc(n + 1, sizeof(int));
    int *JA = malloc(nnz * sizeof(int));
    double *val = calloc(nnz * bnnz, sizeof(double));
    if (!IA || !JA || !val) { free(IA); free(JA); free(val); return -1; }
    int pos = 0;
    IA[0] = 1;
    for (int i = 0; i < n; i++) {
        if (i > 0) { JA[pos] = i;     pos++; }
        JA[pos] = i + 1; pos++;
        if (i < n - 1) { JA[pos] = i + 2; pos++; }
        IA[i + 1] = pos + 1;
    }
    pos = 0;
    for (int i = 0; i < n; i++) {
        if (i > 0) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
        fill_scale(val + pos * bnnz, bs, 2.0); pos++;
        if (i < n - 1) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
    }
    *pIA = IA; *pJA = JA; *pval = val;
    return nnz;
}

static int gen_2d_laplacian(int n, int bs, int **pIA, int **pJA, double **pval) {
    int g = (int)(sqrt(n) + 0.5);
    if (g * g != n) {
        fprintf(stderr, "n_blocks must be a perfect square for 2D Laplacian\n");
        return -1;
    }
    int nnz = 0;
    for (int r = 0; r < n; r++) {
        int i = r / g, j = r % g;
        nnz++;
        if (i > 0) nnz++;
        if (i < g - 1) nnz++;
        if (j > 0) nnz++;
        if (j < g - 1) nnz++;
    }
    int bnnz = bs * bs;
    int *IA = calloc(n + 1, sizeof(int));
    int *JA = malloc(nnz * sizeof(int));
    double *val = calloc(nnz * bnnz, sizeof(double));
    if (!IA || !JA || !val) { free(IA); free(JA); free(val); return -1; }
    int pos = 0;
    IA[0] = 1;
    for (int r = 0; r < n; r++) {
        int i = r / g, j = r % g;
        if (i > 0) { JA[pos] = (i - 1) * g + j + 1; pos++; }
        if (j > 0) { JA[pos] = r;                    pos++; }
        JA[pos] = r + 1; pos++;
        if (j < g - 1) { JA[pos] = r + 2;                pos++; }
        if (i < g - 1) { JA[pos] = (i + 1) * g + j + 1; pos++; }
        IA[r + 1] = pos + 1;
    }
    pos = 0;
    for (int r = 0; r < n; r++) {
        int i = r / g, j = r % g;
        if (i > 0) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
        if (j > 0) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
        fill_scale(val + pos * bnnz, bs, 4.0); pos++;
        if (j < g - 1) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
        if (i < g - 1) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
    }
    *pIA = IA; *pJA = JA; *pval = val;
    return nnz;
}

static int gen_3d_laplacian(int n, int bs, int **pIA, int **pJA, double **pval) {
    int g = (int)(cbrt(n) + 0.5);
    if (g * g * g != n) {
        fprintf(stderr, "n_blocks must be a perfect cube for 3D Laplacian\n");
        return -1;
    }
    int nnz = 0;
    for (int r = 0; r < n; r++) {
        int i = r / (g * g);
        int rem = r % (g * g);
        int j = rem / g, k = rem % g;
        nnz++;
        if (i > 0) nnz++;
        if (i < g - 1) nnz++;
        if (j > 0) nnz++;
        if (j < g - 1) nnz++;
        if (k > 0) nnz++;
        if (k < g - 1) nnz++;
    }
    int bnnz = bs * bs;
    int *IA = calloc(n + 1, sizeof(int));
    int *JA = malloc(nnz * sizeof(int));
    double *val = calloc(nnz * bnnz, sizeof(double));
    if (!IA || !JA || !val) { free(IA); free(JA); free(val); return -1; }
    int pos = 0;
    IA[0] = 1;
    for (int r = 0; r < n; r++) {
        int i = r / (g * g);
        int rem = r % (g * g);
        int j = rem / g, k = rem % g;
        if (i > 0) { JA[pos] = (i - 1) * g * g + j * g + k + 1; pos++; }
        if (j > 0) { JA[pos] = i * g * g + (j - 1) * g + k + 1; pos++; }
        if (k > 0) { JA[pos] = r;                              pos++; }
        JA[pos] = r + 1; pos++;
        if (k < g - 1) { JA[pos] = r + 2;                          pos++; }
        if (j < g - 1) { JA[pos] = i * g * g + (j + 1) * g + k + 1; pos++; }
        if (i < g - 1) { JA[pos] = (i + 1) * g * g + j * g + k + 1; pos++; }
        IA[r + 1] = pos + 1;
    }
    pos = 0;
    for (int r = 0; r < n; r++) {
        int i = r / (g * g);
        int rem = r % (g * g);
        int j = rem / g, k = rem % g;
        if (i > 0) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
        if (j > 0) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
        if (k > 0) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
        fill_scale(val + pos * bnnz, bs, 6.0); pos++;
        if (k < g - 1) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
        if (j < g - 1) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
        if (i < g - 1) { fill_scale(val + pos * bnnz, bs, -1.0); pos++; }
    }
    *pIA = IA; *pJA = JA; *pval = val;
    return nnz;
}

static int gen_random_spd(int n, int bs, int **pIA, int **pJA, double **pval) {
    int max_off_per_side = 3;
    int max_nnz = n * (1 + 2 * max_off_per_side);
    int bnnz = bs * bs;
    int *IA = calloc(n + 1, sizeof(int));
    int *JA = malloc(max_nnz * sizeof(int));
    double *val = calloc(max_nnz * bnnz, sizeof(double));
    if (!IA || !JA || !val) { free(IA); free(JA); free(val); return -1; }
    srand(42);
    int pos = 0;
    IA[0] = 1;
    for (int r = 0; r < n; r++) {
        int left_avail = r, right_avail = n - r - 1;
        int n_left = 0, n_right = 0;
        if (left_avail > 0)
            n_left = 1 + (rand() % (left_avail > max_off_per_side ? max_off_per_side : left_avail));
        if (right_avail > 0)
            n_right = 1 + (rand() % (right_avail > max_off_per_side ? max_off_per_side : right_avail));
        int *cols_left = NULL;
        if (n_left > 0) {
            cols_left = malloc(n_left * sizeof(int));
            for (int k = 0; k < n_left; k++) {
                int c = rand() % left_avail;
                int dup;
                do { dup = 0;
                    for (int t = 0; t < k; t++) if (cols_left[t] == c) { dup = 1; break; }
                    if (dup) c = rand() % left_avail;
                } while (dup);
                cols_left[k] = c;
                JA[pos] = c + 1;
                double v = ((double)rand() / RAND_MAX) * 0.5;
                for (int r2 = 0; r2 < bs; r2++)
                    for (int c2 = 0; c2 < bs; c2++)
                        val[pos * bnnz + r2 * bs + c2] = (r2 == c2) ? v : v * 0.1;
                pos++;
            }
        }
        int *cols_right = NULL;
        if (n_right > 0) {
            cols_right = malloc(n_right * sizeof(int));
            for (int k = 0; k < n_right; k++) {
                int c = r + 1 + (rand() % right_avail);
                int dup;
                do { dup = 0;
                    for (int t = 0; t < k; t++) if (cols_right[t] == c) { dup = 1; break; }
                    if (dup) c = r + 1 + (rand() % right_avail);
                } while (dup);
                cols_right[k] = c;
                JA[pos] = c + 1;
                double v = ((double)rand() / RAND_MAX) * 0.5;
                for (int r2 = 0; r2 < bs; r2++)
                    for (int c2 = 0; c2 < bs; c2++)
                        val[pos * bnnz + r2 * bs + c2] = (r2 == c2) ? v : v * 0.1;
                pos++;
            }
        }
        free(cols_left);
        free(cols_right);
        double diag_val = 2.0 + (double)rand() / RAND_MAX * (n_left + n_right > 0 ? n_left + n_right : 1);
        JA[pos] = r + 1;
        for (int r2 = 0; r2 < bs; r2++)
            for (int c2 = 0; c2 < bs; c2++)
                val[pos * bnnz + r2 * bs + c2] = (r2 == c2) ? diag_val : 0.0;
        pos++;
        IA[r + 1] = pos + 1;
    }
    *pIA = IA; *pJA = JA; *pval = val;
    return pos;
}

static int gen_full_dense(int n, int bs, int **pIA, int **pJA, double **pval) {
    int nnz = n * n;
    int bnnz = bs * bs;
    int total_vals = nnz * bnnz;
    int *IA = calloc(n + 1, sizeof(int));
    int *JA = malloc(nnz * sizeof(int));
    double *val = calloc(total_vals, sizeof(double));
    if (!IA || !JA || !val) { free(IA); free(JA); free(val); return -1; }
    srand(42);
    int pos = 0;
    IA[0] = 1;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) { JA[pos] = c + 1; pos++; }
        IA[r + 1] = pos + 1;
    }
    pos = 0;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            double *blk = val + pos * bnnz;
            if (r == c) {
                for (int r2 = 0; r2 < bs; r2++)
                    for (int c2 = 0; c2 < bs; c2++)
                        blk[r2 * bs + c2] = (r2 == c2) ? (n + 1.0) : ((double)rand() / RAND_MAX) * 0.01;
            } else {
                for (int r2 = 0; r2 < bs; r2++)
                    for (int c2 = 0; c2 < bs; c2++)
                        blk[r2 * bs + c2] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            }
            pos++;
        }
    }
    *pIA = IA; *pJA = JA; *pval = val;
    return nnz;
}

int main(int argc, char **argv) {
    if (argc < 5) { print_usage(argv[0]); return 1; }

    int n   = atoi(argv[1]);
    int bs  = atoi(argv[2]);
    const char *bsr_matfile  = argv[3];
    const char *bsr_rhsfile  = argv[4];
    int type       = (argc > 5) ? atoi(argv[5]) : 0;
    int sol_type   = (argc > 6) ? atoi(argv[6]) : 0;
    const char *solfile   = NULL;
    const char *csr_matfile = NULL;
    const char *csr_rhsfile = NULL;
    if (argc > 7) {
        if (argc == 8) {
            solfile = argv[7];
        } else if (argc == 9) {
            csr_matfile = argv[7];
            csr_rhsfile = argv[8];
        } else {
            solfile = argv[7];
            csr_matfile = argv[8];
            csr_rhsfile = argv[9];
        }
    }

    if (n < 1 || bs < 1) {
        fprintf(stderr, "n_blocks >= 1, block_size >= 1\n");
        return 1;
    }

    int *IA = NULL, *JA = NULL;
    double *val = NULL;
    int nnz = 0;

    switch (type) {
    case 0: nnz = gen_1d_laplacian(n, bs, &IA, &JA, &val); break;
    case 1: nnz = gen_2d_laplacian(n, bs, &IA, &JA, &val); break;
    case 2: nnz = gen_random_spd(n, bs, &IA, &JA, &val); break;
    case 3: nnz = gen_full_dense(n, bs, &IA, &JA, &val); break;
    case 4: nnz = gen_3d_laplacian(n, bs, &IA, &JA, &val); break;
    default: fprintf(stderr, "Unknown type %d\n", type); return 1;
    }

    if (nnz <= 0) { fprintf(stderr, "Generation failed\n"); return 1; }

    /* Write BSR matrix */
    FILE *fp = fopen(bsr_matfile, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", bsr_matfile); return 1; }
    write_bsr(fp, n, bs, nnz, IA, JA, val);
    fclose(fp);
    printf("Wrote BSR matrix : %s  (%d blocks, block_size=%d, nnz=%d)\n",
           bsr_matfile, n, bs, nnz);

    /* Build RHS = A * x, with x determined by sol_type */
    int dim = n * bs;
    double *x = malloc(dim * sizeof(double));
    double *b = calloc(dim, sizeof(double));
    if (!x || !b) { free(x); free(b); return 1; }

    switch (sol_type) {
    case 0:
        for (int i = 0; i < dim; i++) x[i] = 1.0;
        break;
    case 1:
        for (int i = 0; i < dim; i++) x[i] = (double)(i + 1) / dim;
        break;
    case 2:
        for (int i = 0; i < dim; i++) x[i] = sin(2.0 * M_PI * i / dim);
        break;
    case 3:
        srand(123);
        for (int i = 0; i < dim; i++) x[i] = (double)rand() / RAND_MAX;
        break;
    case 4:
        if (!solfile) {
            fprintf(stderr, "sol_type 4 requires a sol_file path\n");
            free(x); free(b); return 1;
        }
        break;
    default:
        fprintf(stderr, "Unknown sol_type %d\n", sol_type);
        free(x); free(b); return 1;
    }

    if (sol_type == 4) {
        FILE *sf = fopen(solfile, "r");
        if (!sf) { fprintf(stderr, "Cannot open solution file %s\n", solfile); return 1; }
        int nvals;
        if (fscanf(sf, "%d", &nvals) != 1) { fclose(sf); return 1; }
        for (int i = 0; i < dim && i < nvals; i++) {
            int idx; double v;
            if (fscanf(sf, "%d %lf", &idx, &v) == 2)
                if (idx >= 0 && idx < dim) x[idx] = v;
        }
        fclose(sf);
    }

    bsr_spmv(n, bs, nnz, IA, JA, val, x, b);

    /* Write BSR RHS */
    fp = fopen(bsr_rhsfile, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", bsr_rhsfile); return 1; }
    write_rhs(fp, dim, b);
    fclose(fp);
    printf("Wrote BSR RHS : %s  (dim=%d)\n", bsr_rhsfile, dim);

    /* Write CSR matrix and RHS if requested */
    if (csr_matfile) {
        fp = fopen(csr_matfile, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", csr_matfile); return 1; }
        write_coo(fp, n, bs, nnz, IA, JA, val);
        fclose(fp);
        printf("Wrote COO matrix : %s  (read_coo_to_csr format)\n", csr_matfile);
    }
    if (csr_rhsfile) {
        fp = fopen(csr_rhsfile, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", csr_rhsfile); return 1; }
        write_rhs_csr(fp, dim, b);
        fclose(fp);
        printf("Wrote CSR RHS : %s  (dim=%d)\n", csr_rhsfile, dim);
    }

    /* Cleanup */
    free(IA); free(JA); free(val); free(x); free(b);
    return 0;
}
