#!/bin/sh
module load mpich/mpi-x

# cd jxpamg-trunk-vv
cd jxpamg-0220

make clean

make -j32

cd ..

make clean
# make -j 32
# make test_bsr_mv
make test_cpr