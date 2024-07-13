import pickle 
import time 
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from local_utils import configure_gpu, save_dir_and_layers_from_args

sys.path.append( os.environ.get('SOUPY_PATH'))
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
sys.path.append( os.environ.get('HIPPYFLOW_PATH'))

sys.path.append('../../../')
import mr_dino.neural_network as nn 
import mr_dino.spoon as spoon


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN training")

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

    parser.add_argument('--load_from_temp', default=False, action='store_true', help="Load test data from temp")
    parser.add_argument('--save_temp', default=False, action='store_true', help="Save test data to temp")
    args = parser.parse_args()

    # --------------------------------- Preprocessing --------------------------------- # 
    evaluation_dir = args.training_base_dir + "/accuracy"
    os.makedirs(evaluation_dir, exist_ok=True)

    # Configure the GPU for tensorflow based on machine
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    configure_gpu(args.enable_gpu)

    print("Loading trained neural network")
    save_dir, pre_layers_m, pre_layers_z, post_layers = save_dir_and_layers_from_args(args)
    save_name = save_dir.split('/')[-1]
    print(save_name)
    rbnet = nn.load_multi_input_reduced_basis_network(save_dir)

    data_dir = args.data_dir
    dir_handler = spoon.DirectoryHandler(data_dir)
    data_settings = spoon.standard_data_generation_settings()
    data_loader = spoon.StandardDataLoader(dir_handler, data_settings)

    # Load training data
    print("Loading data")
    m_test, z_test, q_test, Jz_test, dM, dZ, dU = load_test_data(data_loader, args.n_test, args.load_from_temp)

    print("Mass matrix")
    M = data_loader.load_mass_csr()

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

    np.save("%s/%s.npy" %(evaluation_dir, save_name), [l2_error, F_error])

    if args.save_temp:
        os.makedirs('temp', exist_ok=True)
        np.save('temp/m_test.npy', m_test)
        np.save('temp/z_test.npy', z_test)
        np.save('temp/q_test.npy', q_test)
        np.save('temp/Jz_test.npy', Jz_test)
