#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH -J myFirstJob
#SBATCH -p partMath
#SBATCH --ntasks=32          # 共 32 个任务（即 32 个 MPI 进程）
#SBATCH --cpus-per-task=1    # 每个任务用 1 核（按需调整）

source /opt/intel/oneapi/setvars.sh --force


# 确保日志目录存在
mydate=$(date +%m-%d)
mkdir -p ./log/$mydate

# for pid in 3
# do
#     for ilu_IR_iters in 1
#     do
#         for np  in 1 2
#         do
#             for ilu_ujac_iters in 1 2 3 4 5
#             do
#                 mpirun  -n $np ./solver_strong  -pid $pid -sid 81 -ilu_type 0 -ilu_tri_solve 5 -ilu_ljac_iters $ilu_ujac_iters \
#                 -ilu_ujac_iters $ilu_ujac_iters -nts 1 -ilu_IR_iters $ilu_IR_iters -sweep 5
#             done
#         done
#     done
# done


# for pid in 3
# do
#     for ilu_IR_iters in 1
#     do
#         for np  in 1 2
#         do
#             for ilu_ujac_iters in 1 2 3 4 5
#             do
#                 mpirun  -n $np ./solver_strong  -pid $pid -sid 81 -ilu_type 0 -ilu_tri_solve 20 -ilu_ljac_iters $ilu_ujac_iters \
#                 -ilu_ujac_iters $ilu_ujac_iters -nts 1 -ilu_IR_iters $ilu_IR_iters -sweep 5
#             done
#         done
#     done
# done


# for pid in 3
# do
#     for ilu_IR_iters in 1
#     do
#         for np  in 1 2
#         do
#             for ilu_ujac_iters in 1 2 3 4 5
#             do
#                 mpirun  -n $np ./solver_strong  -pid $pid -sid 81 -ilu_type 0 -ilu_tri_solve 21 -ilu_ljac_iters $ilu_ujac_iters \
#                 -ilu_ujac_iters $ilu_ujac_iters -nts 1 -ilu_IR_iters $ilu_IR_iters -sweep 5
#             done
#         done
#     done
# done


for pid in 3
do
    for ilu_IR_iters in 1
    do
        for np  in 1 2 4 8 16 32
        do
            for ilu_ujac_iters in 3
            do
                mpirun  -n $np ./solver_strong  -pid $pid -sid 81 -ilu_type 0 -ilu_tri_solve 14 -ilu_ljac_iters $ilu_ujac_iters \
                -ilu_ujac_iters $ilu_ujac_iters -nts 1 -ilu_IR_iters $ilu_IR_iters -sweep 3 -ilu_type 3
            done
        done
    done
done
