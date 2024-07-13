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

sys.path.append( os.environ.get('HIPPYLIB_PATH'))
sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))

sys.path.append('../../../')
import mr_dino.spoon as spoon
import mr_dino.neural_network as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN training")
    parser.add_argument('--n_train', type=int, default=1024, help="Number of training data")
    parser.add_argument('--n_test', type=int, default=512, help="Number of test data")
    parser.add_argument('--n_pod', type=int, default=512, help="Number of training samples for computing POD")
    parser.add_argument('--epochs_l2', type=int, default=100, help="Number of epochs for l2 pretraining")
    parser.add_argument('--alpha_l2', type=float, default=1e-3, help="Learning rate for l2 pretraining")
    parser.add_argument('--epochs_h1', type=int, default=100, help="Number of epochs for control Jacobian training")
    parser.add_argument('--alpha_h1', type=float, default=1e-3, help="Learning rate for control Jacobian training")
    parser.add_argument('--l2_only', default=False, action='store_true', help="Train with l2 only")
    parser.add_argument('--scheduler', default=False, action='store_true', help="Use learning rate scheduler")
    parser.add_argument('--scheduler_epoch', default=500, type=int, help="Epoch at which scheduler switches on")
    parser.add_argument('--scheduler_scale', default=0.25, type=float, help="Ratio to scale learning rate by")
    parser.add_argument('--no_shuffle', default=False, action='store_true', help="Do not shuffle data")

    parser.add_argument('--m_rank', type=int, default=100, help="Rank for random parameter")
    parser.add_argument('--u_rank', type=int, default=100, help="Rank for state")
    parser.add_argument('--m_rank_save', type=int, default=100, help="Rank for random parameter")
    parser.add_argument('--u_rank_save', type=int, default=100, help="Rank for state")
    parser.add_argument('--pre_layers_m', type=int, nargs='+', help="Pre concatenation layer breadths for m")
    parser.add_argument('--pre_layers_z', type=int, nargs='+', help="Pre concatenation layer breadths for z")
    parser.add_argument('--post_layers', type=int, nargs='+', help="Post concatenation layer breadths")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size for training")
    parser.add_argument('--index', default=0, type=int, help="Index for different neural network initializations")
    parser.add_argument('--data_dir', type=str, default='../data_generation/data', help="Location where data is stored")
    parser.add_argument('--training_base_dir', type=str, default='./', help="Location where network is stored")
    # parser.add_argument('--data_location', type=str, default='storage', choices=['local', 'storage'], help="Location where data is stored")

    parser.add_argument('--verbosity', default=0, type=int, help="Verbosity for prints in training")
    parser.add_argument('-g', '--enable_gpu', default="", type=str, help="Enable GPU for tensorflow")
    parser.add_argument('-M', '--mass_weighting', default=False, action='store_true', help="Use mass weighted POD modes")
    parser.add_argument('-S', '--shifted', default=False, action='store_true', help="Use POD with shift")
    args = parser.parse_args()

    # --------------------------------- Preprocessing --------------------------------- # 

    # Configure the GPU for tensorflow based on machine
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    configure_gpu(args.enable_gpu)

    # Make save directory
    save_dir, pre_layers_m, pre_layers_z, post_layers = save_dir_and_layers_from_args(args)
    os.makedirs(save_dir, exist_ok=True)

    # Load training data
    data_dir = args.data_dir
    dir_handler = spoon.DirectoryHandler(data_dir)
    data_settings = spoon.standard_data_generation_settings()
    data_settings['n_pod'] = args.n_pod 
    data_settings['kle_rank'] = args.m_rank_save
    data_settings['pod_rank'] = args.u_rank_save
    data_loader = spoon.StandardDataLoader(dir_handler, data_settings)

    m_data, z_data, q_data, dM, dZ, dQ = data_loader.load_state_data(return_sizes=True)
    reduced_Jz_data = data_loader.load_projected_jacobian_data(rank = args.u_rank)
    _, pod_basis, pod_projector, pod_shift = data_loader.load_pod(rank = args.u_rank)
    _, kle_basis, kle_projector, kle_shift = data_loader.load_kle(rank = args.m_rank)

    # Split training and testing data including jacobian
    train_data, test_data = nn.split_train_test_arrays([m_data, z_data, q_data, reduced_Jz_data], args.n_train, args.n_test,
                                                       seed=args.index, shuffle=not args.no_shuffle)
    m_train, z_train, u_train, reduced_Jz_train = train_data
    m_test, z_test, u_test, reduced_Jz_test = test_data

    # Project training and testing data
    if args.shifted:
        print("Project data with shifted POD")
        print(kle_shift.shape, pod_shift.shape)
        reduced_m_train, reduced_u_train = nn.project_data([m_train, u_train], [kle_projector, pod_projector], shifts=[kle_shift, pod_shift])
        reduced_m_test, reduced_u_test = nn.project_data([m_test, u_test], [kle_projector, pod_projector], shifts=[kle_shift, pod_shift])
    else:
        print("Project data with unshifted POD")
        reduced_m_train, reduced_u_train = nn.project_data([m_train, u_train], [kle_projector, pod_projector])
        reduced_m_test, reduced_u_test = nn.project_data([m_test, u_test], [kle_projector, pod_projector])

    # Compute normalization of data
    m_normalization, z_normalization, u_normalization = nn.compute_data_normalization([reduced_m_train, z_train, reduced_u_train])

    # Define training data
    reduced_input_train = [reduced_m_train, z_train]
    reduced_output_train = [reduced_u_train, ]
    
    # Define testing data (output will be defined later)
    reduced_input_test = [reduced_m_test, z_test]

    # --------------------------------- Make neural network --------------------------------- # 
    # Make reduced basis network and equip with jacobian
    if args.shifted:
        print("Build RB-network with shifted POD")
        rbnet = nn.MultiInputReducedBasisNetwork(dM, dZ, dQ, args.m_rank, args.u_rank, 
                m_projector=kle_projector, m_shift=kle_shift, u_basis=pod_basis, u_projector=pod_projector, u_shift=pod_shift, post_layers=post_layers, 
                m_normalization=m_normalization, z_normalization=z_normalization, u_normalization=u_normalization)
    else:
        print("Build RB-network with unshifted POD")
        rbnet = nn.MultiInputReducedBasisNetwork(dM, dZ, dQ, args.m_rank, args.u_rank, 
                m_projector=kle_projector, u_basis=pod_basis, u_projector=pod_projector, post_layers=post_layers, 
                m_normalization=m_normalization, z_normalization=z_normalization, u_normalization=u_normalization)
    reduced_network = rbnet.reduced_network


    # --------------------------------- Train neural network --------------------------------- # 
    # Do l2 training
    print("L2 Training data sizes")
    for i, input_data in enumerate(reduced_input_train):
        print("Input ", i, ": ", input_data.shape)

    for i, output_data in enumerate(reduced_output_train):
        print("Output ", i, ": ", output_data.shape)

    if args.scheduler:
        scheduler = nn.SwitchScheduler(epoch_tol=args.scheduler_epoch, scale=args.scheduler_scale)
    else:
        scheduler = None

    optimizer = tf.keras.optimizers.Adam(learning_rate = args.alpha_l2)
    l2_training_time, l2_training_history = nn.train_network_l2(rbnet.reduced_network, 
            reduced_input_train, reduced_u_train,
            optimizer, args.epochs_l2, args.batch_size, 
            validation_data=(reduced_input_test, reduced_u_test),
            scheduler=scheduler,
            verbosity=args.verbosity)
    
    if not args.l2_only:
        # Do H1 training
        # Append jacobian to output data 
        reduced_output_train = [reduced_u_train, reduced_Jz_train]
        reduced_output_test = [reduced_u_test, reduced_Jz_test]

        print("H1 Training data sizes")
        for i, input_data in enumerate(reduced_input_train):
            print("Input ", i, ": ", input_data.shape)

        for i, output_data in enumerate(reduced_output_train):
            print("Output ", i, ": ", output_data.shape)

        # Define training details
        optimizer = tf.keras.optimizers.Adam(learning_rate = args.alpha_h1)

        if args.scheduler:
            scheduler = nn.SwitchScheduler(epoch_tol=args.scheduler_epoch, scale=args.scheduler_scale)
        else:
            scheduler = None
        
        h1_training_time, h1_training_history = nn.train_network_mr_dino(rbnet.reduced_network, 
                reduced_input_train, reduced_output_train, 
                optimizer, args.epochs_h1, args.batch_size,
                validation_data=(reduced_input_test, reduced_output_test),
                scheduler=scheduler,
                verbosity=args.verbosity)
    else:
        h1_training_time = 0
        
    # --------------------------------- Save and evaluate --------------------------------- # 
    rbnet.save_weights(save_dir)
    rbnet.save_metadata(save_dir)
    mean_error = nn.evaluate_prediction_error(rbnet.full_network, [m_test, z_test], u_test)

    with open(save_dir + '/log.txt', 'w') as logfile:
        print("Mean testing L2 error: %g" %(mean_error), file=logfile)
        print("L2 training time: %g" %(l2_training_time), file=logfile)
        print("H1 training time: %g" %(h1_training_time), file=logfile)

    with open(save_dir + '/log.p', 'wb') as logfile:
        training_log = {'l2_training_time' : l2_training_time, 'h1_training_time' : h1_training_time, 'mean_l2_error' : mean_error}
        pickle.dump(training_log, logfile)

    print("L2 training took %g s" %(l2_training_time))
    if args.l2_only:
        nn.plot_and_save_history_all(l2_training_history, save_dir)
    else:
        nn.plot_and_save_history_all(h1_training_history, save_dir)
        print("H1 training took %g s" %(h1_training_time))
    print("#" * 80 )
