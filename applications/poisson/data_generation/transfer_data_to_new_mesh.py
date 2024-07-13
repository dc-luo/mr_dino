import argparse
import time
import pickle
import sys, os

import dolfin as dl
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt 

sys.path.append( os.environ.get('HIPPYLIB_PATH'))
sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))

import hippylib as hp
import hippyflow as hf
import soupy

sys.path.append("../../../")
import mr_dino.spoon as spoon 

sys.path.append("../")
from poisson_pde import poisson_control_settings 
from setup_poisson_data_sampling import setup_poisson_data_sampling


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-nf', type=int, required=True, help="Fine resolution") 
    parser.add_argument('-nc', type=int, required=True, help="Coarse resolution") 
    parser.add_argument('-n', '--n_pod', type=int, required=False, help="Number of samples for computing POD")
    parser.add_argument('-m', '--m_rank', type=int, required=False, help="Input rank (KLE)")
    parser.add_argument('-u', '--u_rank', type=int, required=False, help="Output rank (POD)")
    parser.add_argument('-s', '--save_dir', type=str, default='data', help="Save directory for data")
    parser.add_argument('-v', '--visualize', default=False, action='store_true', help="Compare saved samples")
    args = parser.parse_args()

    nx_fine = args.nf
    nx_coarse = args.nc

    # Existing save dir 
    save_dir = args.save_dir
    dir_handler = spoon.DirectoryHandler(save_dir)

    # New save dir for new data 
    new_save_dir = args.save_dir + "%g" %(nx_coarse)
    new_dir_handler = spoon.DirectoryHandler(new_save_dir)
    new_dir_handler.make_all_directories()

    # Existing problem settings 
    problem_settings = poisson_control_settings()
    problem_settings['nx'] = nx_fine
    problem_settings['ny'] = nx_fine
    observable_sampler = setup_poisson_data_sampling(problem_settings)
    Vh = observable_sampler.Vh 

    # Existing data settings 
    data_settings = spoon.standard_data_generation_settings()
    data_settings.update_if_not_none('n_pod', args.n_pod)
    data_settings.update_if_not_none('pod_rank', args.u_rank)
    data_settings.update_if_not_none('kle_rank', args.m_rank)
    data_loader = spoon.StandardDataLoader(dir_handler, data_settings)

    # New problem settings 
    problem_settings_coarse = poisson_control_settings()
    problem_settings_coarse['nx'] = nx_coarse
    problem_settings_coarse['ny'] = nx_coarse
    observable_sampler_coarse = setup_poisson_data_sampling(problem_settings_coarse)
    Vh_coarse = observable_sampler_coarse.Vh

    spoon.convert_data_to_new_mesh(Vh, Vh_coarse, data_loader, new_dir_handler)
    new_driver = spoon.StandardDataGenerationDriver(observable_sampler_coarse, new_dir_handler, data_settings)
    new_driver.compute_pod_basis()
    new_driver.compute_kle_basis()
    new_driver.collect_and_project_jacobian()
    
    if args.visualize:
        new_data_loader = spoon.StandardDataLoader(new_dir_handler, data_settings)
        spoon.compare_data_on_meshes(data_loader, new_data_loader, observable_sampler, observable_sampler_coarse)
        plt.show()
