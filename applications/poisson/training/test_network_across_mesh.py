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

sys.path.append('../../../')
import mr_dino.neural_network as nn 
import mr_dino.spoon as spoon

from local_utils import configure_gpu, save_dir_and_layers_from_args

sys.path.append("../")
from poisson_pde import poisson_control_settings, build_poisson_problem

def load_test_data(data_loader, n_test, load_from_temp=False):
    if load_from_temp:
        print("Loading data from temp")
        m_data = np.load('temp/m_test.npy')
        z_data = np.load('temp/z_test.npy')
        q_data = np.load('temp/q_test.npy')
        Jz_data = np.load('temp/Jz_test.npy')
        dM = m_data.shape[1]
        dZ = z_data.shape[1]
        dQ = q_data.shape[1]
    
    else:
        print("Loading data from storage")
        m_data, z_data, q_data, dM, dZ, dQ = data_loader.load_state_data(return_sizes=True)
        n_all = m_data.shape[0]

        # reduce to test data 
        m_data = m_data[n_all-n_test:]
        z_data = z_data[n_all-n_test:]
        q_data = q_data[n_all-n_test:]

        Jz_data = np.zeros((n_test, dQ, dZ))
        i_start = n_all - n_test 
        i_end = n_all
        count = 0 
        for i_test in range(i_start, i_end):
            Jz_data [count] = data_loader.load_full_jacobian_data(i_test)
            count += 1
            print("loading jacobian %d" %(i_test))

    return m_data, z_data, q_data, Jz_data, dM, dZ, dQ



def transfer_network_to_new_mesh(Vh_old, Vh_new, metadata, weights, basis=None):
    """
    Transfers the outputs to new mesh 
    """
    assert weights['u_shift'] is not None, "Only supports shifted networks for now"
    dU = Vh_new[soupy.STATE].dim()
    metadata['dU'] = dU
    
    u_basis = weights['decoder_network']['reduced_basis_u'][0]
    u_shift = weights['decoder_network']['reduced_basis_u'][1]

    if basis is None:
        print("Coverting")
        converter = spoon.FunctionSpaceConverter(Vh_old[soupy.STATE], Vh_new[soupy.STATE], method='superlu_dist')
        u_basis_new = converter.convert_batch(u_basis)
        u_shift_new = converter.convert_batch(np.array([u_shift]))[0]
        u_projector_new = 0*u_basis_new.T # Not needed, dummy 
    else:
        print("Loading in")
        u_basis_new = basis['reduced_basis_u']
        u_projector_new = basis['u_projector']
        u_shift_new = basis['u_shift']

    weights['decoder_network']['reduced_basis_u'][0] = u_basis_new
    weights['decoder_network']['reduced_basis_u'][1] = u_shift_new
    weights['u_projector'] = u_projector_new
    weights['u_shift'] = u_shift_new

    basis_new = dict()
    basis_new['reduced_basis_u'] = u_basis_new
    basis_new['u_projector'] = u_projector_new
    basis_new['u_shift'] = u_shift_new

    return metadata, weights, basis_new



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN training")

    parser.add_argument('-nf', type=int, required=True, help="Fine resolution") 
    parser.add_argument('-nc', type=int, required=True, help="Coarse resolution") 

    parser.add_argument('--n_train', type=int, default=1024, help="Number of training data")
    parser.add_argument('--n_test', type=int, default=1024, help="Number of test data")
    parser.add_argument('--l2_only', default=False, action='store_true', help="Train with l2 only")
    parser.add_argument('--scheduler', default=False, action='store_true', help="Use learning rate scheduler")
    parser.add_argument('--m_rank', type=int, default=100, help="Rank for random parameter")
    parser.add_argument('--u_rank', type=int, default=100, help="Rank for state")

    parser.add_argument('--pre_layers_m', type=int, nargs='+', help="Pre concatenation layer breadths for m")
    parser.add_argument('--pre_layers_z', type=int, nargs='+', help="Pre concatenation layer breadths for z")
    parser.add_argument('--post_layers', type=int, nargs='+', help="Post concatenation layer breadths")
    parser.add_argument('--index', default=0, type=int, help="Index for different neural network initializations")
    parser.add_argument('--data_dir', type=str, default='../data_generation/data', help="Location where data is stored")
    parser.add_argument('--training_base_dir', type=str, default='./', help="Location where network is stored")

    parser.add_argument('--verbosity', default=0, type=int, help="Verbosity for prints in training")
    parser.add_argument('-g', '--enable_gpu', default="", type=str, help="Enable GPU for tensorflow")
    parser.add_argument('-M', '--mass_weighting', default=False, action='store_true', help="Use mass weighted POD modes")
    parser.add_argument('-S', '--shifted', default=False, action='store_true', help="Use POD with shift")

    args = parser.parse_args()

    # --------------------------------- Preprocessing --------------------------------- # 
    evaluation_dir = args.training_base_dir + "/accuracy_on_fine"
    os.makedirs(evaluation_dir, exist_ok=True)

    # Configure the GPU for tensorflow based on machine
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    configure_gpu(args.enable_gpu)

    print("Loading trained neural network")
    save_dir, pre_layers_m, pre_layers_z, post_layers = save_dir_and_layers_from_args(args)
    save_name = save_dir.split('/')[-1]
    print(save_dir, save_name)


    if args.nf == args.nc:
        rbnet = nn.load_multi_input_reduced_basis_network(save_dir)
    else:
        # Convert networks  
        settings = poisson_control_settings()
        settings['nx'] = args.nc 
        settings['ny'] = args.nc 
        _, _, Vh_old, _, _ = build_poisson_problem(settings)


        settings = poisson_control_settings()
        settings['nx'] = args.nf
        settings['ny'] = args.nf 
        _, _, Vh_new, _, _ = build_poisson_problem(settings)

        
        basis_file = f'temp_basis_u{args.u_rank}_nx{args.nc}.p'
        basis_exist = os.path.isfile(basis_file)
        if basis_exist:
            print("Basis exists, loading")
            with open(basis_file, 'rb') as bf:
                basis_dict = pickle.load(bf)
        else:
            print("Basis does not exist")
            basis_dict = None

        metadata, weights = nn.load_multi_input_reduced_basis_network_data(save_dir)
        metadata, weights, basis_new = transfer_network_to_new_mesh(Vh_old, Vh_new, metadata, weights, basis=basis_dict)

        if not basis_exist:
            with open(basis_file, 'wb') as bf:
                pickle.dump(basis_new, bf)

        rbnet = nn.build_multi_input_reduced_basis_network_from_data(metadata, weights)





    data_dir = args.data_dir
    dir_handler = spoon.DirectoryHandler(data_dir)
    data_settings = spoon.standard_data_generation_settings()
    data_loader = spoon.StandardDataLoader(dir_handler, data_settings)

    print("Loading data")
    m_test, z_test, q_test, Jz_test, dM, dZ, dU = load_test_data(data_loader, args.n_test, False)

    print("Mass matrix")
    M = data_loader.load_mass_csr()


    if args.nf != args.nc:
        print("Load input data on coarse mesh")
        dir_handler = spoon.DirectoryHandler(args.training_base_dir)
        data_settings = spoon.standard_data_generation_settings()
        data_loader = spoon.StandardDataLoader(dir_handler, data_settings)
        m_test, _, _, _, _, _, _ = load_test_data(data_loader, args.n_test, False)


    print("Equip model with Jacobian")
    full_network = rbnet.full_network
    full_network_with_jacobian = nn.equip_model_with_full_control_jacobian(full_network)

    print("-" * 80)
    print("Prediction")
    t0 = time.time()
    u_pred, Jz_pred = full_network_with_jacobian.predict([m_test, z_test])
    t1 = time.time()
    print("Prediction time: %g s" %(t1 - t0))
    print("-" * 80)

    l2_error = nn.relative_l2_prediction_error(u_pred, q_test, M)
    F_error = nn.relative_fro_prediction_error(Jz_pred, Jz_test, M)
    print("L2 error: %g" %(l2_error))
    print("H1 error: %g" %(F_error))
    np.save("%s/%s.npy" %(evaluation_dir, save_name), [l2_error, F_error])
