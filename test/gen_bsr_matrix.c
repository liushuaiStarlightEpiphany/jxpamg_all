#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s <n_blocks> <block_size> <file> [type]\n", prog);
    fprintf(stderr, "  type: 0 = 1D Laplacian (tridiagonal block, default)\n");
    fprintf(stderr, "        1 = 2D 5-pt Laplacian (sizeof block_grid)\n");
    fprintf(stderr, "        2 = random SPD (diag-dominant, bilateral off-diagonals)\n");
    fprintf(stderr, "        3 = full dense block matrix (no zero entries)\n");
    fprintf(stderr, "       %s -p <file>      print existing BSR file as dense matrix\n", prog);
}

static int print_dense_bsr(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return 1; }

    int nrows, ncols, nnz, bs, storage;
    if (fscanf(fp, "%d %d %d", &nrows, &ncols, &nnz) != 3) {
        fprintf(stderr, "Error: failed to read header\n"); fclose(fp); return 1;
    }
    if (fscanf(fp, "%d", &bs) != 1) {
        fprintf(stderr, "Error: failed to read block_size\n"); fclose(fp); return 1;
    }
    if (fscanf(fp, "%d", &storage) != 1) {
        fprintf(stderr, "Error: failed to read storage_manner\n"); fclose(fp); return 1;
    }

    int bnnz = bs * bs;

    /* Read IA */
    int tmp;
    if (fscanf(fp, "%d", &tmp) != 1) { fclose(fp); return 1; }
    int* IA = malloc((nrows + 1) * sizeof(int));
    if (!IA) { fclose(fp); return 1; }
    for (int i = 0; i <= nrows; i++) {
        if (fscanf(fp, "%d", &IA[i]) != 1) { free(IA); fclose(fp); return 1; }
    }
    int file_base = (IA[0] == 1) ? 1 : 0;
    for (int i = 0; i <= nrows; i++) IA[i] -= file_base;

    /* Read JA */
    if (fscanf(fp, "%d", &tmp) != 1) { free(IA); fclose(fp); return 1; }
    int* JA = malloc(nnz * sizeof(int));
    if (!JA) { free(IA); fclose(fp); return 1; }
    for (int i = 0; i < nnz; i++) {
        if (fscanf(fp, "%d", &JA[i]) != 1) { free(IA); free(JA); fclose(fp); return 1; }
        JA[i] -= file_base;
    }

    /* Read val (row-major within each block) */
    if (fscanf(fp, "%d", &tmp) != 1) { free(IA); free(JA); fclose(fp); return 1; }
    int total = nnz * bnnz;
    double* val = malloc(total * sizeof(double));
    if (!val) { free(IA); free(JA); fclose(fp); return 1; }
    for (int i = 0; i < total; i++) {
        if (fscanf(fp, "%lf", &val[i]) != 1) { free(IA); free(JA); free(val); fclose(fp); return 1; }
    }
    fclose(fp);

    /* Expand BSR to dense scalar matrix (nrows*bs) x (ncols*bs) */
    int sn = nrows * bs;
    double* dense = calloc(sn * sn, sizeof(double));
    if (!dense) { free(IA); free(JA); free(val); return 1; }

    for (int br = 0; br < nrows; br++) {
        for (int p = IA[br]; p < IA[br + 1]; p++) {
            int bc = JA[p];
            for (int r = 0; r < bs; r++)
                for (int c = 0; c < bs; c++)
                    dense[(br * bs + r) * sn + (bc * bs + c)] = val[p * bnnz + r * bs + c];
        }
    }

    printf("=== Dense BSR matrix: %d x %d blocks, block_size=%d, scalar %d x %d ===\n",
           nrows, ncols, bs, sn, sn);
    for (int i = 0; i < sn; i++) {
        for (int j = 0; j < sn; j++)
            printf("%14.8f", dense[i * sn + j]);
        printf("\n");
    }

    free(IA); free(JA); free(val); free(dense);
    return 0;
}

static void fill_identity(double* blk, int bs) {
    for (int r = 0; r < bs; r++)
        for (int c = 0; c < bs; c++)
            blk[r * bs + c] = (r == c) ? 1.0 : 0.0;
}

static void fill_scale(double* blk, int bs, double s) {
    for (int r = 0; r < bs; r++)
        for (int c = 0; c < bs; c++)
            blk[r * bs + c] = (r == c) ? s : 0.0;
}

static void fill_rand_diag_dominant(double* blk, int bs, unsigned int seed) {
    srand(seed);
    double sum = 0.0;
    for (int r = 0; r < bs; r++) {
        for (int c = 0; c < bs; c++) {
            if (r != c) {
                double v = ((double)rand() / RAND_MAX) * 0.1;
                blk[r * bs + c] = v;
                sum += fabs(v);
            }
        }
    }
    for (int r = 0; r < bs; r++)
        blk[r * bs + r] = sum + 1.0 + ((double)rand() / RAND_MAX);
}

static void write_bsr(FILE* fp, int n, int bs, int nnz, int* IA, int* JA, double* val) {
    int bnnz = bs * bs;
    int total_vals = nnz * bnnz;

    fprintf(fp, "%d %d %d\n", n, n, nnz);
    fprintf(fp, "%d\n", bs);
    fprintf(fp, "0\n");

    fprintf(fp, "%d\n", n + 1);
    for (int i = 0; i <= n; i++)
        fprintf(fp, "%d\n", IA[i]);

    fprintf(fp, "%d\n", nnz);
    for (int i = 0; i < nnz; i++)
        fprintf(fp, "%d\n", JA[i]);

    fprintf(fp, "%d\n", total_vals);
    for (int i = 0; i < total_vals; i++)
        fprintf(fp, "%.14e\n", val[i]);
}

/*
 * Type 0: 1D Laplacian
 *   Block tridiagonal: diag=2*I, off-diag=-I
 */
static int gen_1d_laplacian(int n, int bs, FILE* fp) {
    int nnz = 3 * n - 2;
    int bnnz = bs * bs;

    int* IA = calloc(n + 1, sizeof(int));
    int* JA = malloc(nnz * sizeof(int));
    double* val = calloc(nnz * bnnz, sizeof(double));
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

    write_bsr(fp, n, bs, nnz, IA, JA, val);
    free(IA); free(JA); free(val);
    return 0;
}

/*
 * Type 1: 2D 5-pt Laplacian on sqrt(n) x sqrt(n) grid
 *   Each block is bs x bs
 *   Pattern: each row has up to 5 blocks (center, left, right, down, up)
 */
static int gen_2d_laplacian(int n, int bs, FILE* fp) {
    int g = (int)(sqrt(n) + 0.5);
    if (g * g != n) {
        fprintf(stderr, "Error: n_blocks must be a perfect square for 2D Laplacian\n");
        return -1;
    }

    /* Count nnz */
    int nnz = 0;
    for (int r = 0; r < n; r++) {
        int i = r / g, j = r % g;
        nnz++;                           /* diagonal */
        if (i > 0) nnz++;                /* north */
        if (i < g - 1) nnz++;            /* south */
        if (j > 0) nnz++;                /* west */
        if (j < g - 1) nnz++;            /* east */
    }

    int bnnz = bs * bs;
    int* IA = calloc(n + 1, sizeof(int));
    int* JA = malloc(nnz * sizeof(int));
    double* val = calloc(nnz * bnnz, sizeof(double));
    if (!IA || !JA || !val) { free(IA); free(JA); free(val); return -1; }

    int pos = 0;
    IA[0] = 1;
    for (int r = 0; r < n; r++) {
        int i = r / g, j = r % g;
        if (i > 0) { JA[pos] = (i - 1) * g + j + 1; pos++; }
        if (j > 0) { JA[pos] = r;                    pos++; } /* west = same row */
        JA[pos] = r + 1; pos++; /* diagonal */
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

    write_bsr(fp, n, bs, nnz, IA, JA, val);
    free(IA); free(JA); free(val);
    return 0;
}

/*
 * Type 2: Random SPD (diagonal dominant)
 *   Both lower and upper off-diagonals (bilateral), so L and U are both non-empty.
 *   Each diag block = identity scaled by (2 + rand)
 *   Each off-diag block = random small values
 */
static int gen_random_spd(int n, int bs, FILE* fp) {
    /* Symmetric random sparse pattern: each row has
       up to 3 left off-diagonals + up to 3 right off-diagonals + diagonal */
    int max_off_per_side = 3;
    int max_nnz = n * (1 + 2 * max_off_per_side);

    int* IA = calloc(n + 1, sizeof(int));
    int* JA = malloc(max_nnz * sizeof(int));
    double* val = calloc(max_nnz * bs * bs, sizeof(double));
    if (!IA || !JA || !val) { free(IA); free(JA); free(val); return -1; }

    srand(42);
    int bnnz = bs * bs;

    int pos = 0;
    IA[0] = 1;
    for (int r = 0; r < n; r++) {
        int n_left = 0, n_right = 0;
        int left_avail = r;
        int right_avail = n - r - 1;

        if (left_avail > 0)
            n_left = 1 + (rand() % (left_avail > max_off_per_side ? max_off_per_side : left_avail));
        if (right_avail > 0)
            n_right = 1 + (rand() % (right_avail > max_off_per_side ? max_off_per_side : right_avail));

        int* cols_left = NULL;
        if (n_left > 0) {
            cols_left = malloc(n_left * sizeof(int));
            for (int k = 0; k < n_left; k++) {
                int c = rand() % left_avail;
                int dup;
                do {
                    dup = 0;
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

        int* cols_right = NULL;
        if (n_right > 0) {
            cols_right = malloc(n_right * sizeof(int));
            for (int k = 0; k < n_right; k++) {
                int c = r + 1 + (rand() % right_avail);
                int dup;
                do {
                    dup = 0;
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

    int nnz = pos;
    write_bsr(fp, n, bs, nnz, IA, JA, val);
    free(IA); free(JA); free(val);
    return 0;
}

/*
 * Type 3: Full dense block matrix — no zero entries.
 *   Every block position (r,c) is non-zero.
 *   Diagonal blocks: n*I + small random (SPD guaranteed by strict diagonal dominance)
 *   Off-diagonal blocks: small random values
 *   Purpose: ILU(0) pattern = full LU pattern → after enough sweeps, residual ≈ 0
 */
static int gen_full_dense(int n, int bs, FILE* fp) {
    int nnz = n * n;
    int bnnz = bs * bs;
    int total_vals = nnz * bnnz;

    int* IA = calloc(n + 1, sizeof(int));
    int* JA = malloc(nnz * sizeof(int));
    double* val = calloc(total_vals, sizeof(double));
    if (!IA || !JA || !val) { free(IA); free(JA); free(val); return -1; }

    srand(42);

    /* Build IA, JA — all n*n positions */
    int pos = 0;
    IA[0] = 1;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            JA[pos] = c + 1;
            pos++;
        }
        IA[r + 1] = pos + 1;
    }

    /* Fill values */
    pos = 0;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            double* blk = val + pos * bnnz;
            if (r == c) {
                /* Diagonal: n*I + random perturbation for SPD guarantee */
                for (int r2 = 0; r2 < bs; r2++)
                    for (int c2 = 0; c2 < bs; c2++)
                        blk[r2 * bs + c2] = (r2 == c2) ? (n + 1.0) : ((double)rand() / RAND_MAX) * 0.01;
            } else {
                /* Off-diagonal: small random */
                for (int r2 = 0; r2 < bs; r2++)
                    for (int c2 = 0; c2 < bs; c2++)
                        blk[r2 * bs + c2] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            }
            pos++;
        }
    }

    write_bsr(fp, n, bs, nnz, IA, JA, val);
    free(IA); free(JA); free(val);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    /* Print dense mode: gen_bsr_matrix -p <file> */
    if (strcmp(argv[1], "-p") == 0) {
        if (argc < 3) { fprintf(stderr, "Error: missing filename\n"); return 1; }
        return print_dense_bsr(argv[2]);
    }

    if (argc < 4) { print_usage(argv[0]); return 1; }

    int n = atoi(argv[1]);
    int bs = atoi(argv[2]);
    const char* outfile = argv[3];
    int type = (argc > 4) ? atoi(argv[4]) : 0;

    if (n < 1 || bs < 1 || bs > 20) {
        fprintf(stderr, "Error: n_blocks >= 1, 1 <= block_size <= 20\n");
        return 1;
    }

    FILE* fp = fopen(outfile, "w");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", outfile); return 1; }

    int ret;
    switch (type) {
    case 0: ret = gen_1d_laplacian(n, bs, fp); break;
    case 1: ret = gen_2d_laplacian(n, bs, fp); break;
    case 2: ret = gen_random_spd(n, bs, fp); break;
    case 3: ret = gen_full_dense(n, bs, fp); break;
    default: fprintf(stderr, "Unknown type %d\n", type); fclose(fp); return 1;
    }

    fclose(fp);
    if (ret) { fprintf(stderr, "Generation failed\n"); return 1; }

    printf("Wrote BSR matrix: n_blocks=%d block_size=%d type=%d -> %s\n",
           n, bs, type, outfile);
    return 0;
}
