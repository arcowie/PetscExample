#include <petsc.h>



static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
    PetscFunctionBeginUser;
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
    PetscCall(DMSetFromOptions(*dm));
    PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
    PetscFunctionReturn(PETSC_SUCCESS);
    
}

static void f0Constant(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
    f0[0] = 2. * dim;
}

static void f1Constant(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
    for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static PetscErrorCode exact(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt nc, PetscScalar *u, void *ctx)
{
    u[0] = 0.;
    for (PetscInt d = 0; d < dim; ++d) u[0] += PetscSqr(x[d]);
    return 0;
}

//Jacobian
static PetscErrorCode g3Laplace(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
    for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.;
    
}
static PetscErrorCode SetupProblem(DM dm)
{
    PetscFE        fe;
    PetscInt       cStart, dim, id = 1;
    DMPolytopeType ct;
    PetscDS        ds;
    DMLabel        label;

    PetscFunctionBeginUser;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetCellType(dm, cStart, &ct));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, NULL, PETSC_DECIDE, &fe));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(DMCreateDS(dm));
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetResidual(ds, 0, f0Constant, f1Constant));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3Laplace));
    PetscCall(PetscDSSetExactSolution(ds, 0, exact, NULL));
    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "Wall", label, 1, &id, 0, 0, NULL, (void (*)(void))exact, NULL, NULL, NULL));
    PetscFunctionReturn(PETSC_SUCCESS);
}



int main(int argc, char **argv)
{
    DM  dm;
    Vec b, u, uexact;
    Mat A;    
    SNES snes;

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
    PetscCall(SetupProblem(dm));

    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
    PetscCall(SNESSetDM(snes, dm));
    PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, NULL));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(VecSet(u, 0.));
    PetscCall(SNESSolve(snes, NULL, u));
    PetscCall(VecViewFromOptions(u, NULL, "-sol_view"));

    PetscCall(SNESDestroy(&snes));
    
    PetscCall(VecDestroy(&u));
    
    PetscCall(DMDestroy(&dm));
    
    PetscCall(PetscFinalize());
    return 0;
    
}