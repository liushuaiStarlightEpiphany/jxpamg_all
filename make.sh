#!/bin/sh
module load mpich/mpi-x

# cd jxfpamg-trunk-vv
cd jxfpamg-0220

#make clean

make -j32

cd ..

make clean
# make -j 32
# make test_bsr_mv
# make test_cpr
# make test_mixed
make test_hybrid