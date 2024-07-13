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
from navier_stokes_problem import defaultBluffBodySettings
from setup_navier_stokes_data_sampling import setup_navier_stokes_data_sampling

def compare_navier_stokes_data_on_meshes(data_loader1, data_loader2, sampler1, sampler2, 
                           data_inds=[0,1,10], 
                           jacobian_inds=[0, 1, 5]):
    
    m_fun1 = dl.Function(sampler1.Vh[soupy.PARAMETER])
    z_fun1 = dl.Function(sampler1.Vh[soupy.CONTROL])
    u_fun1 = dl.Function(sampler1.Vh[soupy.STATE])
    m1, z1, u1  = data_loader1.load_state_data()

    m_fun2 = dl.Function(sampler2.Vh[soupy.PARAMETER])
    z_fun2 = dl.Function(sampler2.Vh[soupy.CONTROL])
    u_fun2 = dl.Function(sampler2.Vh[soupy.STATE])
    m2, z2, u2  = data_loader2.load_state_data()

    for data_ind in data_inds:
        m_fun1.vector().set_local(m1[data_ind])
        m_fun2.vector().set_local(m2[data_ind])

        z_fun1.vector().set_local(z1[data_ind])
        z_fun2.vector().set_local(z2[data_ind])

        u_fun1.vector().set_local(u1[data_ind])
        u_fun2.vector().set_local(u2[data_ind])

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        tri = dl.plot(m_fun1)
        plt.colorbar(tri)
        plt.title('Parameter %d' %(data_ind))

        plt.subplot(122)
        tri = dl.plot(m_fun2)
        plt.colorbar(tri)
        plt.title('Parameter %d' %(data_ind))

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.plot(z_fun1.vector().get_local())
        plt.title('Control %d' %(data_ind))

        plt.subplot(122)
        plt.plot(z_fun2.vector().get_local())
        plt.title('Control %d' %(data_ind))


        plt.figure(figsize=(12,10))
        plt.subplot(221)
        tri = dl.plot(u_fun1.sub(0))
        plt.colorbar(tri)
        plt.title('Velocity %d' %(data_ind))

        plt.subplot(222)
        tri = dl.plot(u_fun2.sub(0))
        plt.colorbar(tri)
        plt.title('Velocity %d' %(data_ind))

        plt.subplot(223)
        tri = dl.plot(u_fun1.sub(1))
        plt.colorbar(tri)
        plt.title('Pressure %d' %(data_ind))

        plt.subplot(224)
        tri = dl.plot(u_fun2.sub(1))
        plt.colorbar(tri)
        plt.title('Pressure %d' %(data_ind))

        Jz_1 = data_loader1.load_full_jacobian_data(data_ind)
        Jz_2 = data_loader2.load_full_jacobian_data(data_ind)


        for jacobian_ind in jacobian_inds:
            u_fun1.vector().set_local(Jz_1[:, jacobian_ind])
            u_fun2.vector().set_local(Jz_2[:, jacobian_ind])

            plt.figure(figsize=(12,10))
            plt.subplot(221)
            tri = dl.plot(u_fun1.sub(0))
            plt.colorbar(tri)
            plt.title('Jacobian Velocity col %d sample %d' %(jacobian_ind, data_ind))

            plt.subplot(222)
            tri = dl.plot(u_fun2.sub(0))
            plt.colorbar(tri)
            plt.title('Jacobian Velocity col %d sample %d' %(jacobian_ind, data_ind))

            plt.subplot(223)
            tri = dl.plot(u_fun1.sub(1))
            plt.colorbar(tri)
            plt.title('Jacobian Pressure col %d sample %d' %(jacobian_ind, data_ind))

            plt.subplot(224)
            tri = dl.plot(u_fun2.sub(1))
            plt.colorbar(tri)
            plt.title('Jacobian Pressure col %d sample %d' %(jacobian_ind, data_ind))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', type=str, required=True, help="Fine resolution") 
    parser.add_argument('-mc', type=str, required=True, help="Coarse resolution") 
    parser.add_argument('-n', '--n_pod', type=int, required=False, help="Number of samples for computing POD")
    parser.add_argument('-m', '--m_rank', type=int, required=False, help="Input rank (KLE)")
    parser.add_argument('-u', '--u_rank', type=int, required=False, help="Output rank (POD)")
    parser.add_argument('-s', '--save_dir', type=str, default='data', help="Save directory for data")
    parser.add_argument('-v', '--visualize', default=False, action='store_true', help="Compare saved samples")
    args = parser.parse_args()


    # Existing save dir 
    save_dir = args.save_dir
    dir_handler = spoon.DirectoryHandler(save_dir)

    # New save dir for new data 
    new_save_dir = args.save_dir + "_%s" %(args.mc)
    new_dir_handler = spoon.DirectoryHandler(new_save_dir)
    new_dir_handler.make_all_directories()

    # Existing problem setting    
    problem_settings = defaultBluffBodySettings()
    problem_settings["mesh_base_directory"] = "../"
    problem_settings["mesh_resolution"] = args.mf
    observable_sampler = setup_navier_stokes_data_sampling(problem_settings)
    Vh = observable_sampler.Vh 

    # Existing data settings 
    data_settings = spoon.standard_data_generation_settings()
    data_settings.update_if_not_none('n_pod', args.n_pod)
    data_settings.update_if_not_none('pod_rank', args.u_rank)
    data_settings.update_if_not_none('kle_rank', args.m_rank)
    data_loader = spoon.StandardDataLoader(dir_handler, data_settings)

    # New problem settings 
    problem_settings_coarse = defaultBluffBodySettings()
    problem_settings_coarse["mesh_base_directory"] = "../"
    problem_settings_coarse["mesh_resolution"] = args.mc
    observable_sampler_coarse = setup_navier_stokes_data_sampling(problem_settings_coarse)
    Vh_coarse = observable_sampler_coarse.Vh

    spoon.convert_data_to_new_mesh(Vh, Vh_coarse, data_loader, new_dir_handler)
    new_driver = spoon.StandardDataGenerationDriver(observable_sampler_coarse, new_dir_handler, data_settings)
    new_driver.compute_pod_basis()
    new_driver.compute_kle_basis()
    new_driver.collect_and_project_jacobian()
    
    if args.visualize:
        new_data_loader = spoon.StandardDataLoader(new_dir_handler, data_settings)
        compare_navier_stokes_data_on_meshes(data_loader, new_data_loader, observable_sampler, observable_sampler_coarse)
        plt.show()
