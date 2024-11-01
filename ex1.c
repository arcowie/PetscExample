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

static PetscErrorCode SetupProblem(DM dm)
{
    PetscFE        fe;
    PetscInt       cStart, dim;
    DMPolytopeType ct;
    PetscDS        ds;

    PetscFunctionBeginUser;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetCellType(dm, cStart, &ct));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, NULL, PETSC_DECIDE, &fe));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(DMCreateDS(dm));
    PetscCall(DMGetDS(dm, &ds));
    
    PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **argv)
{
    DM  dm;
    Vec b, u, uexact;
    Mat A;    

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
    PetscCall(SetupProblem(dm));

    PetscCall(DMCreateGlobalVector(dm, &u));
    
    
    


    PetscCall(VecDestroy(&u));
    
    PetscCall(DMDestroy(&dm));
    
    PetscCall(PetscFinalize());
    return 0;
    
}