#!/bin/bash
# 编译并运行 CPR-GMRES 混合精度测试程序
# 用法: ./run_test.sh [np]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# MPI 进程数，默认为 1
NP="${1:-1}"

# 数据文件路径
DATA_DIR="../JXPAMG-bsr/data/spe10_bsr"
MATRIX_FILE="$DATA_DIR/A_bsr_spe10.bin"
RHS_FILE="$DATA_DIR/b_spe10.dat"

# 日志目录（按日期分目录）
DATE_DIR=$(date "+%Y%m%d")
LOG_DIR="$SCRIPT_DIR/log/$DATE_DIR"
mkdir -p "$LOG_DIR"

# 提取测试矩阵简称（取倒数第二个带下划线的字段）
MATRIX_NAME="$(basename "$(dirname "$MATRIX_FILE")" | sed 's/_bsr$//')"

echo "========================================"
echo " JXFPAMG-BSR 混合精度测试"
echo " 时间: $(date)"
echo " MPI 进程数: $NP"
echo "========================================"
echo ""

# ========== 编译 ==========
echo "[1/2] 编译测试程序..."

echo "  -> 编译 test_bsr_cprgmres_mixed ..."
make -j4 test_mixed 2>&1

echo "  -> 编译 test_bsr_cprgmres_hybrid ..."
make -j4 test_hybrid 2>&1

echo "编译完成。"
echo ""

# ========== 运行 ==========
echo "[2/2] 运行测试..."
echo ""

echo "----------------------------------------"
echo " test_bsr_cprgmres_mixed"
echo "----------------------------------------"
MIXED_LOG="$LOG_DIR/test_bsr_cprgmres_mixed-np${NP}-${MATRIX_NAME}.log"
echo " 日志: $MIXED_LOG"
mpirun -np "$NP" ./test_bsr_cprgmres_mixed "$MATRIX_FILE" 1 1 "$RHS_FILE" 2>&1 | tee "$MIXED_LOG"
echo ""

echo "----------------------------------------"
echo " test_bsr_cprgmres_hybrid"
echo "----------------------------------------"
HYBRID_LOG="$LOG_DIR/test_bsr_cprgmres_hybrid-np${NP}-${MATRIX_NAME}.log"
echo " 日志: $HYBRID_LOG"
mpirun -np "$NP" ./test_bsr_cprgmres_hybrid "$MATRIX_FILE" 1 1 "$RHS_FILE" 2>&1 | tee "$HYBRID_LOG"
echo ""

echo "========================================"
echo " 所有测试完成"
echo " 日志目录: $LOG_DIR"
echo "========================================"
