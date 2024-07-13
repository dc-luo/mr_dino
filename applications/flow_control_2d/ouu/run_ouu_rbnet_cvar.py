import random
import time
import os, sys
import platform
import pickle
import argparse
import scipy.optimize

import matplotlib.pyplot as plt
import dolfin as dl
import numpy as np

import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))

import hippylib as hp
import hippyflow as hf
import soupy

sys.path.append('../../../')
import mr_dino.spoon as spoon 
import mr_dino.neural_network as nn

sys.path.append("../")
from navier_stokes_problem import defaultBluffBodySettings, build_navier_stokes_problem
from navier_stokes_ouu_utils import setup_qoi, L2VelocityPenalization
from training.local_utils import configure_gpu


if __name__ == "__main__":
    dl.set_log_active(False)

    parser = argparse.ArgumentParser(description="Solving OUU problem")
    parser.add_argument('-b', '--beta', type=float, default=0.95, help="CVaR quantile")
    parser.add_argument('-N', '--N_sample', type=int, default=64, help="Number of samples for SAA")
    parser.add_argument('-n', '--max_iter', type=int, default=10000, help="Maximum number of iterations")
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help="Step size for SD iterations (not used anymore)")


    parser.add_argument('-p', '--penalty', type=float, default=0.01, help="Weight on penalization")
    parser.add_argument('-w', '--weights_name', type=str, default="Srbnet_data60_kle-pod_m15u15_200_200", help="name for weights")
    parser.add_argument('-d', '--weights_dir', type=str, default="../training/trained_weights", help="Directory for weights")
    parser.add_argument('-q', '--qoi_type', type=str, default="dissipation", choices=["dissipation", "tracking_up", "tracking"], help="Type of QoI")
    parser.add_argument('-o', '--obs_type', type=str, default="full", choices=["full", "full_back", "windowed_back", "windowed_body"], help="Window of observation")
    parser.add_argument('--epsilon', type=float, default=1e-4, help="CVaR sharpness")

    parser.add_argument('--mesh_size', type=str, default="medium", choices=["medium", "medium_coarse", "coarse", "coarsest"],  help="Mesh size specification")
    parser.add_argument('--mesh_base_directory', default="../", help="Base directory for mesh")
    parser.add_argument('--mesh_format', default="xdmf", help="Format for mesh")

    parser.add_argument('--gtol', type=float, default=1e-5, help="Gradient tolerance")
    parser.add_argument('--display', default=False, action="store_true", help="Display optimization iterations")

    parser.add_argument('--random_index', default=0, type=int, help="random index of parameter samples" )
    parser.add_argument('--load_initial_guess', default=False, action="store_true", help="Load initial guess for control")
    parser.add_argument('--initial_guess_directory', default=".", help="Initial guess directory for loading")
    parser.add_argument('--enable_gpu', default="", type=str, help="Enable GPU for tensorflow (0, 1, or all)")

    args = parser.parse_args()
    configure_gpu(args.enable_gpu)

    case_name = "pellet_%s" %(args.mesh_size)
    results_dir = "results_rbnet_cvar_%s_%s_eps%g" %(args.qoi_type, args.obs_type, args.epsilon)

    save_dir = results_dir + "/NN_%s_%s_penalty%g_beta%g_SAA_ns%d_alpha%g_maxiter%g" %(args.weights_name,
            case_name, args.penalty, args.beta, args.N_sample,
            args.alpha, args.max_iter)

    if args.random_index > 0:
        save_dir += "_random%d" %(args.random_index)
        random.seed(1)
        for i in range(args.random_index):
            seed = random.randrange(100_000_000)
    else:
        seed = 1

    os.makedirs(save_dir, exist_ok=True)

    settings = defaultBluffBodySettings()
    settings["mesh_format"] = args.mesh_format
    settings["mesh_base_directory"] = args.mesh_base_directory
    settings["mesh_resolution"] = args.mesh_size 

    print("Making problem")
    mesh, Vh, ns_problem, prior, basis_all, geometry, geo_specs = build_navier_stokes_problem(settings)
    qoi = setup_qoi(args.qoi_type, args.obs_type, mesh, Vh, geo_specs, settings)
    control_model = soupy.ControlModel(ns_problem, qoi)
    x = control_model.generate_vector()
    weights_path = "%s/%s" %(args.weights_dir, args.weights_name)
    print(weights_path)
    print(qoi)

    # Load neural operator
    rbnet = nn.load_multi_input_reduced_basis_network(weights_path)
    # rbqoi = spoon.ReducedBasisL2QoI(qoi.K, rbnet.u_basis())
    # nn_control_model = spoon.ReducedBasisNetworkQoIControlModel(Vh, rbnet, rbqoi)

    u_basis = rbnet.u_basis()
    u_projector = rbnet.u_projector()
    r = u_basis.shape[1]
    print("Check orthogonality of projectors: ", np.linalg.norm(u_basis.T @ u_projector - np.eye(r)))
    
    if args.qoi_type == "tracking" or args.qoi_type == "tracking_up":
        print("Using tracking objective")
        rbqoi = spoon.ReducedBasisL2QoI(qoi.K, rbnet.u_basis(), d=qoi.d, u_projector=rbnet.u_projector(), u_shift=rbnet.u_shift())
    else:
        rbqoi = spoon.ReducedBasisL2QoI(qoi.K, rbnet.u_basis(), u_projector=rbnet.u_projector(), u_shift=rbnet.u_shift())

    nn_control_model = spoon.ReducedBasisNetworkQoIControlModel(Vh, rbnet, rbqoi)

    print("STATE dimension is %d" %(len(x[soupy.STATE].get_local())))
    print("PARAMETER dimension is %d" %(len(x[soupy.PARAMETER].get_local())))
    print("CONTROL dimension is %d" %(len(x[soupy.CONTROL].get_local())))

    dim = len(basis_all)
    # Make the risk measure
    rm_param = spoon.NNSuperquantileRiskMeasureSAASettings()
    rm_param["beta"] = args.beta
    rm_param["sample_size"] = args.N_sample
    rm_param["seed"] = seed
    rm_param["epsilon"] = args.epsilon
    nn_rm = spoon.NNSuperquantileRiskMeasureSAA(nn_control_model, prior, rm_param, project_m=True)

    penalty = L2VelocityPenalization(Vh, basis_all, geometry, args.penalty)

    print("Solution by NN")
    # Cost functional
    hc_cost = soupy.RiskMeasureControlCostFunctional(nn_rm, penalty)
    scipy_cost = soupy.ScipyCostWrapper(hc_cost, verbose=True)


    # Optimize
    print("Begin optimization")
    z0 = nn_rm.generate_vector(soupy.CONTROL)
    z0_np = z0.get_local()

    options = {'disp' : args.display, 'gtol' : args.gtol, 'maxiter' : args.max_iter}

    tsolve_0 = time.time()
    results = scipy.optimize.minimize(scipy_cost.function(), z0_np, method="BFGS", jac=scipy_cost.jac(), options=options)
    tsolve_1 = time.time()
    zt_opt_np = results["x"]
    tsolve = tsolve_1 - tsolve_0
    print("Time to solve optimization problem (s): %g" %(tsolve))
    with open("%s/timing.txt" %(save_dir), "w") as time_file:
        print("Time to solve optimization problem (s): %g" %(tsolve), file=time_file)

    np.save("%s/z_opt.npy" %(save_dir), zt_opt_np[:-1])
    with open("%s/results.p" %(save_dir), "wb") as f:
        pickle.dump(results, f)
