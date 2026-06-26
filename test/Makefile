JXPAMGDIR = ../jxpamg-0220

JXPAMG_USE_OMP = YES
JX_USE_MUMPS = #YES
JX_USE_BIG_INT = #YES
JX_USE_BIG_DOUBLE = #YES

ifdef JX_USE_BIG_INT
JX_BIG_INT_OPTION = -DJX_USING_BIG_INT=1
else
JX_BIG_INT_OPTION = -DJX_USING_BIG_INT=0
endif

ifdef JX_USE_BIG_DOUBLE
JX_BIG_DOUBLE_OPTION = -DJX_USING_BIG_DOUBLE=1
else
JX_BIG_DOUBLE_OPTION = -DJX_USING_BIG_DOUBLE=0
endif

ifdef JXPAMG_USE_OMP
CC = mpicc -fopenmp
else
CC = mpicc
endif

ifdef JX_USE_MUMPS
JX_MUMPS_OPTION = -DJX_USING_MUMPS=1
else
JX_MUMPS_OPTION = -DJX_USING_MUMPS=0
endif

JX_OMP_CP_OPTION = -DJX_USING_OPENMP=1
C_COMPILE_FLAGS = ${JX_BIG_INT_OPTION} ${JX_BIG_DOUBLE_OPTION} ${JX_OMP_CP_OPTION}

INCLUDES = -I $(JXPAMGDIR)/include
LIB      = -L $(JXPAMGDIR)/lib -lJXPAMG -lmpi -lm
CFLAGS   = ${C_COMPILE_FLAGS} ${INCLUDES}

test_bsr_fgpbilu_setup: test_bsr_fgpbilu_setup.o
	${CC} -O3 -no-pie -o $@ $< ${LIB} ${CFLAGS}

gen_bsr_testdata: gen_bsr_testdata.o
	${CC} -O3 -no-pie -o $@ $< -lm

matrix_stats: matrix_stats.o
	${CC} -O3 -no-pie -o $@ $< ${LIB} ${CFLAGS}

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

clean:
	rm -rf test_bsr_fgpbilu_setup gen_bsr_testdata matrix_stats *.o
