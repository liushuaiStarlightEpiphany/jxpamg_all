#!/bin/bash
set -e

# ============================================================
# 可修改参数
# ============================================================
# 矩阵 / RHS / 输出
MATRIX_FILE="/home/spring/l_s/jxpamg_all/JXPAMG-bsr/data/spe10_bsr/A_bsr_spe10.bin"
RHS_FILE="/home/spring/l_s/jxpamg_all/JXPAMG-bsr/data/spe10_bsr/b_spe10.dat"
OUTPUT_FILE="./output/sol.txt"

# MPI 进程数
NP=16

# 求解器参数
SID=27
KDIM=30
MAXIT=20
TOL=1e-6
BINARY=1              # 0=文本, 1=二进制

# 是否预先生成测试矩阵（留空则不生成；若设非空会覆盖 MATRIX_FILE / RHS_FILE）
# 示例（3D Laplacian, 27 blocks, block_size=3, sol_type=2 sine）:
#   GEN_NBLOCKS=27; GEN_BLOCKSIZE=3; GEN_TYPE=4; GEN_SOL_TYPE=2
GEN_NBLOCKS=
GEN_BLOCKSIZE=
GEN_TYPE=
GEN_SOL_TYPE=
GEN_SOL_FILE=
# ============================================================

# ----- 从路径提取文件名（去后缀）-----
mat_base=$(basename "$MATRIX_FILE")
mat_name="${mat_base%.*}"
rhs_base=$(basename "$RHS_FILE")
rhs_name="${rhs_base%.*}"

# ----- 构建参数标签 ----
tag="mat-${mat_name}_rhs-${rhs_name}"
tag="${tag}_sid${SID}_kdim${KDIM}_maxit${MAXIT}_tol${TOL}"
if [ "$BINARY" = "1" ]; then tag="${tag}_binary"; fi
tag="${tag}_np${NP}"

# ----- 日期目录 ----
mydate=$(date +%Y-%m-%d)
log_dir="./log/$mydate"
mkdir -p "$log_dir"

# ----- 输出文件 ----
log_file="$log_dir/$tag.log"
echo "Log: $log_file"

# ----- 预生成测试矩阵 ----
if [ -n "$GEN_NBLOCKS" ] && [ -n "$GEN_BLOCKSIZE" ]; then
    coo_mat="${MATRIX_FILE%.*}_coo.${MATRIX_FILE##*.}"
    coo_rhs="${RHS_FILE%.*}_coo.${RHS_FILE##*.}"
    gen_args=("$GEN_NBLOCKS" "$GEN_BLOCKSIZE" "$MATRIX_FILE" "$RHS_FILE" "$GEN_TYPE" "$GEN_SOL_TYPE")
    if [ -n "$GEN_SOL_FILE" ]; then
        gen_args+=("$GEN_SOL_FILE")
    fi
    gen_args+=("$coo_mat" "$coo_rhs")
    echo "Generating test matrix (BSR→$MATRIX_FILE, COO→$coo_mat)..."
    ./test/gen_bsr_testdata "${gen_args[@]}" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
fi

# ----- 构建 mpirun 命令 ----
cmd="mpirun -np $NP ./solver"
cmd="$cmd $MATRIX_FILE $RHS_FILE $OUTPUT_FILE"
cmd="$cmd -sid $SID -kdim $KDIM -maxit $MAXIT -tol $TOL"
if [ "$BINARY" = "1" ]; then
    cmd="$cmd -binary"
fi

# ----- 执行 ----
echo "Running: $cmd"
echo "" | tee -a "$log_file"
eval "$cmd" 2>&1 | tee -a "$log_file"

exit ${PIPESTATUS[0]}
