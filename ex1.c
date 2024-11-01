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

int main(int argc, char **argv)
{
    DM  dm;
    Vec b, u, uexact;
    Mat A;    

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
    

    
    
    PetscCall(DMDestroy(&dm));
    
    PetscCall(PetscFinalize());
    return 0;
    
}