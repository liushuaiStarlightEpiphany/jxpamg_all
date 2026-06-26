#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "jx_mv.h"
#include "jx_bsr_mv.h"

static int is_binary_file(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (ext != NULL) {
        if (strcmp(ext, ".bin") == 0 || strcmp(ext, ".BIN") == 0) {
            return 1;
        }
    }
    return 0;
}

static const char* get_basename(const char *path) {
    const char *last_slash = strrchr(path, '/');
    if (last_slash != NULL) {
        return last_slash + 1;
    }
    return path;
}

static void analyze_csr_matrix(jx_CSRMatrix *csr, const char *name) {
    JX_Int n = jx_CSRMatrixNumRows(csr);
    JX_Int nnz = jx_CSRMatrixNumNonzeros(csr);
    JX_Int *csr_i = jx_CSRMatrixI(csr);
    JX_Int *csr_j = jx_CSRMatrixJ(csr);
    
    printf("============================================================\n");
    printf("  Matrix Analysis: %s\n", name);
    printf("============================================================\n");
    printf("  Format:         CSR (Compressed Sparse Row)\n");
    printf("  Unknowns (n):   %d\n", n);
    printf("  Nonzeros (nnz): %d\n", nnz);
    
    if (n == 0) {
        printf("  Warning: Empty matrix!\n");
        printf("============================================================\n");
        return;
    }
    
    double avg_nnz_per_row = (double)nnz / n;
    printf("  nnz/row:        %.2f\n", avg_nnz_per_row);
    
    JX_Int min_nnz = nnz, max_nnz = 0;
    
    for (JX_Int i = 0; i < n; i++) {
        JX_Int rnnz = csr_i[i + 1] - csr_i[i];
        if (rnnz < min_nnz) min_nnz = rnnz;
        if (rnnz > max_nnz) max_nnz = rnnz;
    }
    printf("  nnz/row range: [%d, %d]\n", min_nnz, max_nnz);
    
    JX_Int min_bw = n, max_bw = 0;
    long long total_lower_bw = 0, total_upper_bw = 0;
    JX_Int max_lower_bw = 0, max_upper_bw = 0;
    
    for (JX_Int i = 0; i < n; i++) {
        JX_Int row_lower_bw = 0;
        JX_Int row_upper_bw = 0;
        
        for (JX_Int k = csr_i[i]; k < csr_i[i + 1]; k++) {
            JX_Int j = csr_j[k];
            if (j < i) {
                JX_Int dist = i - j;
                if (dist > row_lower_bw) row_lower_bw = dist;
            } else if (j > i) {
                JX_Int dist = j - i;
                if (dist > row_upper_bw) row_upper_bw = dist;
            }
        }
        
        JX_Int row_bw = row_lower_bw + row_upper_bw + 1;
        if (row_bw < min_bw) min_bw = row_bw;
        if (row_bw > max_bw) max_bw = row_bw;
        
        total_lower_bw += row_lower_bw;
        total_upper_bw += row_upper_bw;
        if (row_lower_bw > max_lower_bw) max_lower_bw = row_lower_bw;
        if (row_upper_bw > max_upper_bw) max_upper_bw = row_upper_bw;
    }
    
    double avg_lower_bw = (double)total_lower_bw / n;
    double avg_upper_bw = (double)total_upper_bw / n;
    double avg_bw = avg_lower_bw + avg_upper_bw + 1;
    
    printf("  Bandwidth:\n");
    printf("    Average:      %.1f\n", avg_bw);
    printf("    Min/Max:      [%d, %d]\n", min_bw, max_bw);
    printf("    Lower (max):  %d\n", max_lower_bw);
    printf("    Upper (max):  %d\n", max_upper_bw);
    printf("    Half-bandwidth: %d\n", max_lower_bw > max_upper_bw ? max_lower_bw : max_upper_bw);
    
    long long total = (long long)n * n;
    double density = (double)nnz / total * 100.0;
    printf("  Density:        %.6f %%\n", density);
    printf("============================================================\n");
}

static void analyze_bsr_matrix(jx_BSRMatrix *bsr_mat, const char *name) {
    JX_Int bs = jx_BSRMatrixBlockSize(bsr_mat);
    JX_Int n_blocks = jx_BSRMatrixNumRows(bsr_mat);
    JX_Int block_nnz = jx_BSRMatrixNumNonzeros(bsr_mat);
    JX_Int *bsr_i = jx_BSRMatrixI(bsr_mat);
    JX_Int *bsr_j = jx_BSRMatrixJ(bsr_mat);
    
    JX_Int n = n_blocks * bs;
    JX_Int scalar_nnz = block_nnz * bs * bs;
    
    printf("============================================================\n");
    printf("  Matrix Analysis: %s\n", name);
    printf("============================================================\n");
    printf("  Format:         BSR (Block Sparse Row)\n");
    printf("  Block size:     %d\n", bs);
    printf("  Block rows:     %d\n", n_blocks);
    printf("  Block nnz:      %d\n", block_nnz);
    printf("  Unknowns (n):   %d\n", n);
    printf("  Nonzeros (nnz): %d (scalar)\n", scalar_nnz);
    
    if (n_blocks == 0) {
        printf("  Warning: Empty matrix!\n");
        printf("============================================================\n");
        return;
    }
    
    double avg_block_nnz_per_row = (double)block_nnz / n_blocks;
    double avg_nnz_per_row = (double)scalar_nnz / n;
    printf("  block-nnz/row:  %.2f\n", avg_block_nnz_per_row);
    printf("  nnz/row:        %.2f (scalar)\n", avg_nnz_per_row);
    
    JX_Int min_block_nnz = block_nnz, max_block_nnz = 0;
    
    for (JX_Int i = 0; i < n_blocks; i++) {
        JX_Int rnnz = bsr_i[i + 1] - bsr_i[i];
        if (rnnz < min_block_nnz) min_block_nnz = rnnz;
        if (rnnz > max_block_nnz) max_block_nnz = rnnz;
    }
    printf("  block-nnz/row range: [%d, %d]\n", min_block_nnz, max_block_nnz);
    
    JX_Int min_block_bw = n_blocks, max_block_bw = 0;
    long long total_lower_bw = 0, total_upper_bw = 0;
    JX_Int max_lower_bw = 0, max_upper_bw = 0;
    
    for (JX_Int i = 0; i < n_blocks; i++) {
        JX_Int row_lower_bw = 0;
        JX_Int row_upper_bw = 0;
        
        for (JX_Int k = bsr_i[i]; k < bsr_i[i + 1]; k++) {
            JX_Int j = bsr_j[k];
            if (j < i) {
                JX_Int dist = i - j;
                if (dist > row_lower_bw) row_lower_bw = dist;
            } else if (j > i) {
                JX_Int dist = j - i;
                if (dist > row_upper_bw) row_upper_bw = dist;
            }
        }
        
        JX_Int row_bw = row_lower_bw + row_upper_bw + 1;
        if (row_bw < min_block_bw) min_block_bw = row_bw;
        if (row_bw > max_block_bw) max_block_bw = row_bw;
        
        total_lower_bw += row_lower_bw;
        total_upper_bw += row_upper_bw;
        if (row_lower_bw > max_lower_bw) max_lower_bw = row_lower_bw;
        if (row_upper_bw > max_upper_bw) max_upper_bw = row_upper_bw;
    }
    
    double avg_lower_bw = (double)total_lower_bw / n_blocks;
    double avg_upper_bw = (double)total_upper_bw / n_blocks;
    double avg_block_bw = avg_lower_bw + avg_upper_bw + 1;
    double avg_scalar_bw = avg_block_bw * bs;
    JX_Int max_scalar_bw = max_block_bw * bs;
    
    printf("  Bandwidth (block):\n");
    printf("    Average:      %.1f\n", avg_block_bw);
    printf("    Min/Max:      [%d, %d]\n", min_block_bw, max_block_bw);
    printf("    Lower (max):  %d\n", max_lower_bw);
    printf("    Upper (max):  %d\n", max_upper_bw);
    printf("    Half-bandwidth: %d\n", max_lower_bw > max_upper_bw ? max_lower_bw : max_upper_bw);
    
    printf("  Bandwidth (scalar, estimated):\n");
    printf("    Average:      %.1f\n", avg_scalar_bw);
    printf("    Max:          %d\n", max_scalar_bw);
    
    long long total_blocks = (long long)n_blocks * n_blocks;
    long long total_scalar = (long long)n * n;
    double block_density = (double)block_nnz / total_blocks * 100.0;
    double scalar_density = (double)scalar_nnz / total_scalar * 100.0;
    printf("  Block density:  %.6f %%\n", block_density);
    printf("  Scalar density: %.6f %%\n", scalar_density);
    printf("============================================================\n");
}

static void print_usage(const char *prog) {
    printf("Usage: %s <matrix_file> [format] [binary]\n", prog);
    printf("\nOptions:\n");
    printf("  matrix_file:  Input matrix file\n");
    printf("  format:       0=CSR, 1=BSR (default: 1, BSR)\n");
    printf("  binary:       0=text, 1=binary (default: 0, auto-detect .bin)\n");
    printf("\nExamples:\n");
    printf("  %s ./data/A_bsr_302X302X3.dat          (BSR text)\n", prog);
    printf("  %s ./data/spe10_bsr/A_bsr_spe10.bin 1 1 (BSR binary)\n", prog);
    printf("  %s ./data/matrix.csr 0               (CSR text)\n", prog);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "-help") == 0) {
        print_usage(argv[0]);
        return 0;
    }
    
    const char *matrix_file = argv[1];
    JX_Int format = 1;
    JX_Int binary = is_binary_file(matrix_file);
    
    if (argc >= 3) format = atoi(argv[2]);
    if (argc >= 4) binary = atoi(argv[3]);
    
    FILE *fp = fopen(matrix_file, "r");
    if (!fp) {
        perror("Error opening matrix file");
        return 1;
    }
    fclose(fp);
    
    const char *name = get_basename(matrix_file);
    
    printf("\n");
    printf("============================================================\n");
    printf("  Matrix Statistics Analyzer\n");
    printf("============================================================\n");
    printf("  File:           %s\n", matrix_file);
    printf("  Format:         %s\n", format == 1 ? "BSR" : "CSR");
    printf("  Binary:         %s\n", binary ? "yes" : "no");
    printf("============================================================\n");
    printf("\n");
    
    if (format == 1) {
        jx_BSRMatrix *bsr;
        printf("Reading BSR matrix...\n");
        if (binary) {
            bsr = jx_BSRMatrixRead_Binary(matrix_file);
        } else {
            bsr = jx_BSRMatrixRead((char*)matrix_file);
        }
        
        if (!bsr) {
            printf("Error: Cannot read BSR matrix from %s\n", matrix_file);
            printf("Note: If this is a CSR matrix, use format=0\n");
            return 1;
        }
        
        analyze_bsr_matrix(bsr, name);
        jx_BSRMatrixDestroy(bsr);
    } else {
        printf("Reading CSR matrix...\n");
        jx_CSRMatrix *csr = jx_CSRMatrixRead((char*)matrix_file, 0);
        
        if (!csr) {
            printf("Error: Cannot read CSR matrix from %s\n", matrix_file);
            printf("Note: If this is a BSR matrix, use format=1\n");
            return 1;
        }
        
        analyze_csr_matrix(csr, name);
        jx_CSRMatrixDestroy(csr);
    }
    
    return 0;
}
