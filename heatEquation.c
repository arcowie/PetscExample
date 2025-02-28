#include "petsc.h"

typedef struct {
    PetscScalar thermalConductivity[3];
} AppCtx;

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
    PetscFunctionBegin;
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
    PetscCall(DMSetFromOptions(*dm));
    PetscCall(DMViewFromOptions(*
    dm, NULL, "-dm_view"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/*
* Exact 2D solution
* u     = 2t + x^2 + y^2
* u_t   = 2
* u_x   = 2 + 2 =4
* f     = 2
* */
static PetscErrorCode exact2D(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
    *u = 2 * time;
    for(PetscInt d = 0; d < dim ; ++d) *u += PetscSqr(x[d]);
    return 0;
}

static PetscErrorCode exact2D_t(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
    *u = 2.;
    return PETSC_SUCCESS;
}
//Seperated the f0 terms into the source/sink and the full f0 which
//includes the time derivative component.  This is not required,
//however, this is useful since typically the source/sink term
//is the one that is of interest for changing.
static void f0Const2DSource(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
    f0[0] = - 2.;
}

static void f0Const2DFull(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
    PetscScalar source = 0.;
    
    f0Const2DSource(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, x, numConstants, constants, &source);

    f0[0] = u_t[0] + source;
}


static void f1Const2D(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
    for(PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}
static PetscErrorCode exact2D_t(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
    *u = 2.;
    return PETSC_SUCCESS;
}];
}

static void g3_temp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

static void g0_temp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_tShift * 1.0;
}



static PetscErrorCode SetupProblem(DM dm)
{
    PetscFE         fe;
    PetscInt        cStart, dim id = 1;
    DMPolytopeType  ct;
    PetscDS         ds;
    DMLabel         label;

    PetscFunctionBeginUser;
    //Get the values needed to supply the arguments to
    //PetscFECreateByCell to create the FE object
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetCellType(dm, cStart, &ct));
    //create the FE object
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, NULL, PETSC_DECIDE, &fe));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    //No longer need the local fe object
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(DMCreateDS(dm));
    PetscCall(DMGetDS(dm, &ds)); 
    //Setup the residuals
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))exact2D, (void (*)(void))exact2D_t, NULL, NULL));
    
    PetscCall(PetscDSSetResidual(ds, 0, f0Const2DFull, f1Const2D));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_temp, NULL, NULL, g3_temp));
    PetscCall(PetscDSSetExactSolution(ds, 0, exact2D, NULL));
    PetscCall(PetscDSSetExactSolutionTimeDerivative(ds, 0, exact2D_t, NULL));
    
    
    PetscFunctionReturn(PETSC_SUCCESS);

    
     
}

static PetscErrorCode SetInitialConditions(TS ts, Vec u)
{
  DM        dm;
  PetscReal t;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(DMComputeExactSolution(dm, t, u, NULL));
  PetscCall(VecSetOptionsPrefix(u, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}



int main(int argc, char **argv)
{
    DM  dm;
    Vec u;
    TS  ts;

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
    PetscCall(SetupProblem(dm));

    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetDM(ts, dm));
    //Get functions from PLEX
    PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL));
    PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, NULL));
    PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, NULL));

    PetscCall(TSSetMaxTime(ts, 1.0));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSSetComputeInitialCondition(ts, SetInitialConditions));
    
    
    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(DMTSCheckFromOptions(ts, u));
    PetscCall(SetInitialConditions(ts, u));
    PetscCall(PetscObjectSetName((PetscObject)u, "temperature"));


    PetscCall(TSSetSolution(ts, u));
    PetscCall(VecViewFromOptions(u, NULL, "-u0_view"));
    
    //Done because the vector u that goes in may not be the final u 
    PetscCall(VecDestroy(&u));
    PetscCall(TSSolve(ts, NULL));

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(VecViewFromOptions(u, NULL, "-uf_view"));
    PetscCall(DMTSCheckFromOptions(ts, u));
    
    
    
    
    
    
    PetscCall(TSDestroy(&ts));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    return 0;
    
}