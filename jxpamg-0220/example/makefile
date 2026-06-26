include ../makefile.pub

ALLPGS = solver_strong apctl_mg schur_mg

all: $(ALLPGS)

default: all

apctl_mg: apctl_mg.o
	$(CC) -o $@ ${CFLAGS} $^ $(ALLLIBS) -lm

schur_mg: schur_mg.o
	$(CC) -o $@ ${CFLAGS} $^ $(ALLLIBS) -lm
	
solver_strong: solver_strong.o
	$(CC) -o $@ ${CFLAGS} $^ $(ALLLIBS) -lm

clean:
	-rm -f *.o *~
	-rm -f solver_strong apctl_mg schur_mg
