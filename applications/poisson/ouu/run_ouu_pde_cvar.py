import random 
import time
import os, sys
import pickle
import argparse

import scipy.optimize
import numpy as np 
import matplotlib.pyplot as plt
import dolfin as dl
from mpi4py import MPI

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))
sys.path.append("../")
sys.path.append("../../../")

import hippylib as hp
import soupy
from poisson_ouu_utils import get_target, plot_wells
from poisson_pde import build_poisson_problem, poisson_control_settings

dl.set_log_active(False)

def print_on_root(print_str, mpi_comm=MPI.COMM_WORLD):
    if mpi_comm.Get_rank() == 0:
        print(print_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solving OUU problem")
    parser.add_argument('-t', '--target', type=str, default="sinusoid", help="target case")
    parser.add_argument('-p', '--param', type=float, default=1.0, help="Parameter for target definition")
    parser.add_argument('-b', '--beta', type=float, default=0.95, help="CVaR percentile value")
    parser.add_argument('--epsilon', type=float, default=1e-4, help="CVaR approximation parameter")

    parser.add_argument('-N', '--N_sample', type=int, default=64, help="Number of samples for SAA")
    parser.add_argument('-n', '--max_iter', type=int, default=100, help="Maximum number of optimization iterations")
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help="Step size for SD iterations")

    parser.add_argument('--nx', type=int, default=64, help="Number of elements in x direction")
    parser.add_argument('--ny', type=int, default=64, help="Number of elements in y direction")
    parser.add_argument('--N_sources', type=int, default=7, help="Number of sources per side")
    parser.add_argument('--loc_lower', type=float, default=0.1, help="Lower bound of location of sources")
    parser.add_argument('--loc_upper', type=float, default=0.9, help="Upper bound of location of sources")
    parser.add_argument('--well_width', type=float, default=0.08, help="Width of sources")

    parser.add_argument('--ftol', type=float, default=1e-7, help="Incremental function change tolerance for LBFGSB")
    parser.add_argument('--gtol', type=float, default=1e-5, help="Projected gradient tolerance")
    parser.add_argument('--display', default=False, action="store_true", help="Display optimization iterations")

    parser.add_argument('--seed', type=int, default=0, help="Random seed for sampling the initial guess and random parameter")
    args = parser.parse_args()
        
    # MPI 
    comm_world = MPI.COMM_WORLD
    comm_self = MPI.COMM_SELF
    world_rank = comm_world.Get_rank()
    world_size = comm_world.Get_size()

    comm_sampler = comm_world
    comm_mesh = comm_self 
    
    print_on_root("Split communicators", comm_world)
    comm_world.Barrier()


    # ----------------- Initialize parameters ----------------- # 
    # Set parsed parameters

    # Model choice
    MODEL = "PDE" 

    # Poisson problem parameters
    poisson_settings = poisson_control_settings()
    poisson_settings['nx'] = args.nx
    poisson_settings['ny'] = args.ny
    poisson_settings['WELL_WIDTH'] = args.well_width
    poisson_settings['LOC_LOWER'] = args.loc_lower
    poisson_settings['LOC_UPPER'] = args.loc_upper
    poisson_settings['N_WELLS_PER_SIDE'] = args.N_sources

    # Objective parameters
    TARGET_CASE = args.target
    TARGET_PARAM = args.param

    # SD parameters
    save_dir = "results_cvar_eps%g/%s_mesh%dx%d_sources%d_%gto%g_width%g_target_%s_p%g_beta%g_SAA_alpha%g_ns%d_maxiter%d" %(args.epsilon,
        MODEL, args.nx, args.ny, 
        args.N_sources, args.loc_lower, args.loc_upper, args.well_width,
        args.target, args.param, 
        args.beta, args.alpha, args.N_sample,  args.max_iter)

    if args.seed > 0:
        # Do the sampling randomly 
        save_dir += "_seed%d" %(args.seed)
        random.seed(1)
        for i in range(args.seed):
            seed = random.randrange(100_000_000)
    else:
        seed = 1 

    if world_rank == 0:
        print("Make directory: ", save_dir)
        os.makedirs(save_dir, exist_ok=True)

    comm_world.Barrier()

    # ----------------- Make problem ----------------- # 
    print_on_root("Making poisson problem", comm_world)
    mesh, pde, Vh, prior, control_distribution = build_poisson_problem(poisson_settings, comm_mesh=comm_mesh)
    comm_world.Barrier()

    print_on_root("Making target, QoI, and problem", comm_world)
    # Define the control model
    u_target_expr = get_target(TARGET_CASE, TARGET_PARAM)
    u_target = dl.interpolate(u_target_expr, Vh[hp.STATE])

    qoi = soupy.L2MisfitControlQoI(Vh, u_target.vector())
    control_model = soupy.ControlModel(pde, qoi)

    comm_world.Barrier()

    # Make the risk measure
    print_on_root("Making SAA risk measure and cost", comm_world)

    rm_param = soupy.superquantileRiskMeasureSAASettings()
    rm_param["beta"] = args.beta
    rm_param["sample_size"] = args.N_sample
    rm_param["seed"] = seed 
    rm_param["epsilon"] = 1e-4
    pde_rm = soupy.SuperquantileRiskMeasureSAA(control_model, prior, settings=rm_param, comm_sampler=comm_sampler)
    pde_cost = soupy.RiskMeasureControlCostFunctional(pde_rm, None)

    # Scipy cost 
    print_on_root("Convert to scipy cost")
    scipy_cost = soupy.ScipyCostWrapper(pde_cost)
    comm_world.Barrier()

    # Box constraint with CVaR. t is unconstrained 
    dim = poisson_settings["N_WELLS_PER_SIDE"]**2
    lb = np.ones(dim) * poisson_settings["STRENGTH_LOWER"]
    ub = np.ones(dim) * poisson_settings["STRENGTH_UPPER"]
    lb = np.append(lb, np.array([-np.infty]))
    ub = np.append(ub, np.array([np.infty]))
    box_bounds = scipy.optimize.Bounds(lb=lb, ub=ub)

    # ----------------- Optimize ----------------- # 
    print_on_root("Start optimization")
    opt_options = {"maxiter" : args.max_iter, "disp" : args.display, 'ftol' : args.ftol, 'gtol' : args.gtol}

    z0 = pde_cost.generate_vector(soupy.CONTROL)
    z0_np = z0.get_local()
    comm_world.Barrier()
    tsolve_0 = time.time()
    results = scipy.optimize.minimize(scipy_cost.function(), z0_np, method="L-BFGS-B", jac=scipy_cost.jac(), bounds=box_bounds, options=opt_options)
    tsolve_1 = time.time()
    tsolve = tsolve_1 - tsolve_0
    zt_opt_np = results["x"]
    z_opt_np = zt_opt_np[:-1]

    print_on_root("Time to solve: %g" %tsolve)
    print_on_root("Number of function evals: %d" %(scipy_cost.n_func))
    print_on_root("Number of gradient evals: %d" %(scipy_cost.n_grad))
    
    z0.set_local(zt_opt_np)
    pde_rm.computeComponents(z0)
    risk_opt = pde_rm.cost()
    cvar_opt = pde_rm.superquantile()

    # -----------------  Post processing ----------------- #
    if world_rank == 0:
        with open("%s/results.p" %(save_dir), "wb") as results_file:
            pickle.dump(results, results_file)

        costs = dict() 
        costs["n_func"] = scipy_cost.n_func 
        costs["n_grad"] = scipy_cost.n_grad
        costs["risk_opt"] = risk_opt
        costs["cvar_opt"] = cvar_opt

        with open("%s/costs.p" %(save_dir), "wb") as costs_file:
            pickle.dump(costs, costs_file)

        plot_wells(z_opt_np, poisson_settings["N_WELLS_PER_SIDE"], poisson_settings["LOC_LOWER"], poisson_settings["LOC_UPPER"])
        plt.savefig("%s/control.png" %(save_dir))
        np.save("%s/z_opt.npy" %(save_dir), z_opt_np)
