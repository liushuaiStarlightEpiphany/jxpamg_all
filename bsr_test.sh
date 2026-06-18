#!/bin/bash
#SBATCH --job-name=test_bsr        # 作业名
#SBATCH -N 1                        # 节点数：1 到 4 个节点
#SBATCH -n 16                       # MPI任务数
#SBATCH --partition=mt_test         # 计算分区
#SBATCH --exclusive                 # 不共享节点

export UCX_TLS=^cma

# for np in 1 2
# do 
#     yhrun --mpi=pmix -n $np ./test_bsr_mv ./data/A_bsr.dat 1 0
# done

# for np in 1 
# do 
#     yhrun --mpi=pmix -n $np ./test_bsr_cprgmres ./data/A_bsr_302X302X3.dat 1 0 ./data/b_bsr_302X302X3.dat
# done

# for np in 1 
# do 
#     yhrun --mpi=pmix -n $np ./test_bsr_cprgmres ../data/spe10_bsr/A_bsr_spe10.bin 1 1 ../data/spe10_bsr/b_spe10.dat 
# done

# for np in 1 
# do 
#     yhrun --mpi=pmix -n $np ./test_bsr_cprgmres_mixed ../data/spe10_bsr/A_bsr_spe10.bin 1 1 ../data/spe10_bsr/b_spe10.dat 
# done

for np in 1 2
do 
    yhrun --mpi=pmix -n $np ./test_bsr_cprgmres_hybrid ../data/spe10_bsr/A_bsr_spe10.bin 1 1 ../data/spe10_bsr/b_spe10.dat 
done
 