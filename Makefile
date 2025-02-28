include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

heatEquation: heatEquation.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}
