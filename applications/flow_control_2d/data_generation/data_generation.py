import argparse
import time
import pickle
import sys, os

import dolfin as dl
import numpy as np
from mpi4py import MPI

sys.path.append( os.environ.get('HIPPYLIB_PATH'))
sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))

import hippylib as hp
import hippyflow as hf
import soupy

sys.path.append("../../../")
import mr_dino.spoon as spoon 

sys.path.append("../")
from navier_stokes_problem import defaultBluffBodySettings
from setup_navier_stokes_data_sampling import setup_navier_stokes_data_sampling


parser = argparse.ArgumentParser()
parser.add_argument('-N', '--n_data', type=int, required=False, help="Number of samples for training data")
parser.add_argument('-n', '--n_pod', type=int, required=False, help="Number of samples for computing POD")
parser.add_argument('-m', '--m_rank', type=int, required=False, help="Input rank (KLE)")
parser.add_argument('-u', '--u_rank', type=int, required=False, help="Output rank (POD)")
parser.add_argument('-s', '--save_dir', type=str, default='data', help="Save directory for data")
parser.add_argument('--mesh_level', type=str, default='medium', help="Save directory for data")

args = parser.parse_args()
save_dir = args.save_dir
dir_handler = spoon.DirectoryHandler(save_dir)
dir_handler.make_all_directories()
problem_settings = defaultBluffBodySettings()
problem_settings["mesh_base_directory"] = "../"
problem_settings["mesh_resolution"] = args.mesh_level

observable_sampler = setup_navier_stokes_data_sampling(problem_settings)
driver_settings = spoon.standard_data_generation_settings()
driver_settings.update_if_not_none('n_data', args.n_data)
driver_settings.update_if_not_none('n_pod', args.n_pod)
driver_settings.update_if_not_none('pod_rank', args.u_rank)
driver_settings.update_if_not_none('kle_rank', args.m_rank)

sample_driver = spoon.StandardDataGenerationDriver(observable_sampler, dir_handler, driver_settings)
sample_driver.sample_state_data()
sample_driver.sample_control_jacobian_at_data()
sample_driver.compute_pod_basis()
sample_driver.compute_kle_basis()
sample_driver.collect_and_project_jacobian()

