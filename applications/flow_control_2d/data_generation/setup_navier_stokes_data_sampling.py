import sys, os
import dolfin as dl

import hippylib as hp
import hippyflow as hf
import soupy 

from navier_stokes_problem import build_navier_stokes_problem

import mr_dino.spoon as spoon 

def setup_navier_stokes_data_sampling(settings, timing=False):
    mesh, Vh, pde, prior, basis_all, geometry, geo_specs = build_navier_stokes_problem(settings)
    parameter_sampler = spoon.BiLaplacianSampler(prior, rng=hp.Random())
    control_sampler = spoon.FiniteIndependentGaussian(Vh[soupy.CONTROL], sigma=settings["control_sd"])

    print("State dim: %g" %(Vh[soupy.STATE].dim()))
    print("Parameter dim: %g" %(Vh[soupy.PARAMETER].dim()))
    print("Control dim: %g" %(Vh[soupy.CONTROL].dim()))

    u_trial = dl.TrialFunction(Vh[soupy.STATE])
    u_test = dl.TestFunction(Vh[soupy.STATE])

    M = dl.PETScMatrix(mesh.mpi_comm())
    dl.assemble(dl.inner(u_trial,u_test)*dl.dx, tensor=M)

    B = hf.StateSpaceIdentityOperator(M)
    observable = hf.LinearStateObservable(pde,B)

    ds = geometry.ds(geometry.LEFT)
    observable_sampler = spoon.ControlProblemSamplerWithBoundaryKLE(Vh, observable, parameter_sampler, control_sampler, ds) 

    return observable_sampler
