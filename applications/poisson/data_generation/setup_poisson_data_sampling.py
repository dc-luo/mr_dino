import sys, os
import dolfin as dl

import hippylib as hp
import hippyflow as hf
import soupy 

from poisson_pde import build_poisson_problem

import mr_dino.spoon as spoon 


def setup_poisson_data_sampling(settings, timing=False):
    mesh, pde, Vh, prior, control_sampler = build_poisson_problem(settings, timing=timing)
    parameter_sampler = spoon.BiLaplacianSampler(prior, rng=hp.Random())

    u_trial = dl.TrialFunction(Vh[soupy.STATE])
    u_test = dl.TestFunction(Vh[soupy.STATE])

    M = dl.PETScMatrix(mesh.mpi_comm())
    dl.assemble(u_trial*u_test*dl.dx, tensor=M)

    B = hf.StateSpaceIdentityOperator(M)
    observable = hf.LinearStateObservable(pde,B)
    observable_sampler = spoon.ControlProblemSampler(Vh, observable, parameter_sampler, control_sampler) 

    return observable_sampler


def data_directory(location_type):
    if location_type == "local":
        return "../data/flow_control_state_control_g1.0_d5.0/"
    elif location_type == "storage":
        return "/storage/dingcheng/controldata/flow_control_state_control_g1.0_d5.0/"
    else:
        raise ValueError("No location of this type available")

def make_directory_handler(location_type):
    base_dir = data_directory(location_type)
    dir_handler = spoon.DirectoryHandler(base_dir)
    return dir_handler







