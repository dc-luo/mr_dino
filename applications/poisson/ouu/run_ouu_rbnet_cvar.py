import random
import time
import os, sys
import pickle
import argparse
import platform

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import tensorflow as tf
import scipy.optimize

if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
sys.path.append("../../../")
sys.path.append("../")

import hippylib as hp
import soupy
import mr_dino.spoon as spoon 
import mr_dino.neural_network as nn 

from poisson_pde import build_poisson_problem, poisson_control_settings
from poisson_ouu_utils import get_target, plot_wells
from training.local_utils import configure_gpu

dl.set_log_active(False)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Solving OUU problem")
    parser.add_argument('-t', '--target', type=str, default="sinusoid", help="target case")
    parser.add_argument('-p', '--param', type=float, default=1.0, help="Parameter for target definition")
    parser.add_argument('-b', '--beta', type=float, default=0.95, help="Quantile for CVaR")
    parser.add_argument('--epsilon', type=float, default=1e-4, help="CVaR approximation parameter")

    parser.add_argument('-N', '--N_sample', type=int, default=64, help="Number of samples for SAA")
    parser.add_argument('-n', '--max_iter', type=int, default=100, help="Maximum number of iterations")
    parser.add_argument('-a', '--alpha', type=float, default=1, help="Step size for SD iterations")
    parser.add_argument('-o', '--optimizer', type=str, default="lbfgsb", choices=["lbfgsb", "sd"],  help="Optimization method")
    parser.add_argument('-w', '--weights_name', type=str, default="Srbnet_data80_kle-pod_m20u20_200_200", help="name for weights")
    parser.add_argument('-d', '--weights_dir', type=str, default="../training/trained_weights", help="Directory for weights")

    parser.add_argument('--nx', type=int, default=64, help="Number of elements in x direction")
    parser.add_argument('--ny', type=int, default=64, help="Number of elements in y direction")
    parser.add_argument('--N_sources', type=int, default=7, help="Number of sources per side")
    parser.add_argument('--loc_lower', type=float, default=0.1, help="Lower bound of location of sources")
    parser.add_argument('--loc_upper', type=float, default=0.9, help="Upper bound of location of sources")
    parser.add_argument('--well_width', type=float, default=0.08, help="Width of wells")

    parser.add_argument('--random_index', type=int, default=0, help="index of randomized initial guess")
    parser.add_argument('--initial_guess', type=str, default="", help="Initial guess file if not randomized")
    parser.add_argument('--randomize_initial_guess', default=False, action="store_true",  help="Randomize initial guess")
    parser.add_argument('--save_iterations', default=False, action="store_true", help="Save the iterations")

    parser.add_argument('--ftol', type=float, default=1e-7, help="Incremental function change tolerance for LBFGSB")
    parser.add_argument('--gtol', type=float, default=1e-5, help="Projected gradient tolerance")
    parser.add_argument('--display', default=False, action="store_true", help="Print iterates in L-BFGS-B optimization")
    parser.add_argument('--load_samples', default=False, action="store_true", help="Load samples for computing SAA")
    parser.add_argument('--results_dir', type=str, default="./", help="Directory for saving results")

    parser.add_argument('-g', '--enable_gpu', default="", type=str, help="Enable GPU for tensorflow. (0, 1, or all)")
    args = parser.parse_args()
    configure_gpu(args.enable_gpu)

    # Set parsed parameters
    MODEL = "NN"

    # Poisson problem parameters
    poisson_settings = poisson_control_settings()
    poisson_settings['nx'] = args.nx
    poisson_settings['ny'] = args.ny
    poisson_settings['WELL_WIDTH'] = args.well_width
    poisson_settings['LOC_LOWER'] = args.loc_lower
    poisson_settings['LOC_UPPER'] = args.loc_upper
    poisson_settings['N_WELLS_PER_SIDE'] = args.N_sources

    weights_path = args.weights_dir + "/" + args.weights_name

    if args.random_index >= 1:
        save_dir = "%s/results_rbnet_cvar_eps%g/NN_%s_mesh%dx%d_sources%d_%gto%g_width%g_target_%s_p%g_beta%g_SAA_alpha%g_ns%d_maxiter%d_%s_random%d" %(args.results_dir, 
             args.epsilon,
            args.weights_name,
            args.nx, args.ny,
            args.N_sources, args.loc_lower, args.loc_upper, args.well_width,
            args.target, args.param,
            args.beta, args.alpha, args.N_sample, args.max_iter, args.optimizer, args.random_index)
        random.seed(1)
        for i in range(args.random_index):
            seed = random.randrange(100_000_000)
    else:
        save_dir = "%s/results_rbnet_cvar_eps%g/NN_%s_mesh%dx%d_sources%d_%gto%g_width%g_target_%s_p%g_beta%g_SAA_alpha%g_ns%d_maxiter%d_%s" %(args.results_dir, 
            args.epsilon,
            args.weights_name,
            args.nx, args.ny,
            args.N_sources, args.loc_lower, args.loc_upper, args.well_width,
            args.target, args.param,
            args.beta, args.alpha, args.N_sample, args.max_iter, args.optimizer)
        seed = 1

    os.makedirs(save_dir, exist_ok=True)

    mesh, pde, Vh, prior, control_distribution = build_poisson_problem(poisson_settings)

    # Define the control model
    u_target_expr = get_target(args.target, args.param)
    u_target = dl.interpolate(u_target_expr, Vh[hp.STATE])
    qoi = soupy.L2MisfitControlQoI(Vh, u_target.vector())
    control_model = soupy.ControlModel(pde, qoi)

    print("Solution by neural network")

    # Load neural operator
    u_trial = dl.TrialFunction(Vh[soupy.STATE])
    u_test = dl.TestFunction(Vh[soupy.STATE])
    M = dl.assemble(u_trial * u_test * dl.dx)
    rbnet = nn.load_multi_input_reduced_basis_network(weights_path)

    u_basis = rbnet.u_basis()
    u_projector = rbnet.u_projector()
    r = u_basis.shape[1]
    print(np.linalg.norm(u_basis.T @ u_projector - np.eye(r)))
    
    rbqoi = spoon.ReducedBasisL2QoI(M, rbnet.u_basis(), u_projector=rbnet.u_projector(), u_shift=rbnet.u_shift(), d=u_target.vector())
    nn_control_model = spoon.ReducedBasisNetworkQoIControlModel(Vh, rbnet, rbqoi)

    # Make the NN risk measure for optimization
    nn_rm_param = spoon.NNSuperquantileRiskMeasureSAASettings()
    nn_rm_param["beta"] = args.beta
    nn_rm_param["sample_size"] = args.N_sample
    nn_rm_param["seed"] = seed
    nn_rm_param["epsilon"] = args.epsilon

    if args.load_samples:
        m_samples = np.load('temp/m_n%d_run%d.npy' %(args.N_sample, args.random_index))
        nn_rm = spoon.NNSuperquantileRiskMeasureSAA(nn_control_model, prior, settings=nn_rm_param, m_samples=m_samples, project_m=True)
    else:
        nn_rm = spoon.NNSuperquantileRiskMeasureSAA(nn_control_model, prior, settings=nn_rm_param, project_m=True)

    # Cost functional
    nn_cost = soupy.RiskMeasureControlCostFunctional(nn_rm, None)

    # Box constraint with CVaR. t is unconstrained
    dim = poisson_settings["N_WELLS_PER_SIDE"]**2
    lb = np.ones(dim) * poisson_settings["STRENGTH_LOWER"]
    ub = np.ones(dim) * poisson_settings["STRENGTH_UPPER"]
    lb = np.append(lb, np.array([-np.infty]))
    ub = np.append(ub, np.array([np.infty]))

    # ----------------- Optimize ----------------- #
    z0 = nn_cost.generate_vector(soupy.CONTROL)
    z0_np = z0.get_local()
    if args.random_index >= 1 and args.randomize_initial_guess:
        # Randomize the initial guess
        print("Using random initial guess")
        z0_np = np.random.randn(len(z0_np))

    elif args.initial_guess != "":
        print("Loading initial guess")
        z0_np = np.load(args.initial_guess)
    else:
        print("Using zero initial guess")

    if args.optimizer == "lbfgsb":
        scipy_cost = soupy.ScipyCostWrapper(nn_cost)
        box_bounds = scipy.optimize.Bounds(lb=lb, ub=ub)
        opt_options = {'disp' : args.display, 'ftol' : args.ftol, 'gtol' : args.gtol}

        tsolve_0 = time.time()
        results = scipy.optimize.minimize(scipy_cost.function(), z0_np, method="L-BFGS-B", jac=scipy_cost.jac(), bounds=box_bounds, options=opt_options)
        tsolve_1 = time.time()

        zt_opt_np = results["x"]
        z_opt_np = zt_opt_np[:-1]
    else:
        box_bounds = [lb, ub]
        sd_param = soupy.SteepestDescent_ParameterList()
        sd_param["alpha"] = args.alpha 
        sd_param["max_iter"] = args.max_iter
        sd_param["print_level"] = -1
        sd_solver = soupy.SteepestDescent(nn_cost, sd_param)
        z0.set_local(z0_np)

        tsolve_0 = time.time()
        z1, results = sd_solver.solve(z0, box_bounds)
        tsolve_1 = time.time()

        zt_opt_np = z1.get_local()
        z_opt_np = zt_opt_np[:-1]

    tsolve = tsolve_1 - tsolve_0
    print("Solving done after %g s" %(tsolve))

    with open("%s/results.p" %(save_dir), "wb") as results_file:
        pickle.dump(results, results_file)

    np.save("%s/z_opt.npy" %(save_dir), z_opt_np)

    with open("%s/run_info.txt" %(save_dir), "w") as f:
        print("Model: %s" %(MODEL), file=f)
        print("Target: %s -- %g" %(args.target, args.param), file=f)
        print("CVaR quantile: %g" %(args.beta), file=f)
        print("%s: %d samples, %d steps, %g step size" %(args.optimizer, args.N_sample, args.max_iter, args.alpha), file=f)
        print("Solution time (s): %g" %(tsolve), file=f)
        print("Initial guess %d" %(args.random_index))

    # ----- Post processing ----- #
    N_PLOTS = 4
    u_fun = dl.Function(Vh[hp.STATE])
    u_nn_fun = dl.Function(Vh[hp.STATE])
    m_fun = dl.Function(Vh[hp.PARAMETER])
    z_fun = dl.Function(Vh[soupy.CONTROL])
    z_opt = z_fun.vector()
    z_opt.set_local(z_opt_np)

    x = [u_fun.vector(), m_fun.vector(), None, z_opt]
    plot_wells(z_opt, poisson_settings["N_WELLS_PER_SIDE"], poisson_settings["LOC_LOWER"], poisson_settings["LOC_UPPER"])
    plt.savefig("%s/control.png" %(save_dir))

    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    z_data = np.expand_dims(z_opt.get_local(), axis=0)
    print(z_opt.get_local())

    for i in range(N_PLOTS):
        # Sample from prior
        hp.parRandom.normal(1.0, noise)
        prior.sample(noise, m_fun.vector())

        # Solve forward
        control_model.solveFwd(x[hp.STATE], x)

        # Solve by NN
        m_data = np.expand_dims(m_fun.vector().get_local(), axis=0)
        u_nn = nn_control_model.solveFwd(m_data, z_data)
        u_nn_fun.vector().set_local(u_nn[0,:])

        plt.figure(figsize=(15,10))
        hp.nb.plot(u_target, subplot_loc=221, mytitle="Target state")
        hp.nb.plot(m_fun, subplot_loc=222, mytitle="Random parameter")
        hp.nb.plot(u_fun, subplot_loc=223, mytitle="PDE Solution")
        hp.nb.plot(u_nn_fun, subplot_loc=224, mytitle="NN Solution")
        plt.savefig("%s/optimal_state%d.png" %(save_dir, i))
