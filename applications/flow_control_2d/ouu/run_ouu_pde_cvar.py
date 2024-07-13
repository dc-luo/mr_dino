import random 
import argparse
import time
import pickle
import sys, os

sys.path.append( os.environ.get('HIPPYLIB_PATH'))
sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
sys.path.append( os.environ.get('SOUPY_PATH'))

import hippylib as hp
import soupy

import scipy.optimize
import numpy as np
import ufl
import dolfin as dl
import matplotlib.pyplot as plt
from mpi4py import MPI

sys.path.append("../")
from navier_stokes_problem import defaultBluffBodySettings, build_navier_stokes_problem
from navier_stokes_ouu_utils import L2VelocityPenalization, setup_qoi

if __name__ == "__main__":
    dl.parameters["form_compiler"]["quadrature_degree"] = 5

    parser = argparse.ArgumentParser(description="Solving OUU problem")
    parser.add_argument('-b', '--beta', type=float, default=0.95, help="CVaR quantile")
    parser.add_argument('-N', '--N_sample', type=int, default=64, help="Number of samples for SAA")
    parser.add_argument('-n', '--max_iter', type=int, default=10000, help="Maximum number of optimization iterations")
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help="Step size SD iterations (not used anymore)")
    parser.add_argument('-p', '--penalty', type=float, default=0.01, help="Weight on penalization")
    parser.add_argument('-c', '--constrained', default=False, action="store_true", help="Use constrained optimization or not")
    parser.add_argument('-q', '--qoi_type', type=str, default="dissipation", choices=["dissipation", "tracking_up", "tracking"], help="Type of QoI")
    parser.add_argument('-o', '--obs_type', type=str, default="full", choices=["full", "full_back", "windowed_back", "windowed_body"], help="Window of observation")
    parser.add_argument('--epsilon', type=float, default=1e-4, help="CVaR sharpness")

    parser.add_argument('--mesh_size', type=str, default="medium", choices=["medium", "medium_coarse", "coarse", "coarsest"],  help="Mesh size specification")
    parser.add_argument('--mesh_base_directory', default="../", help="Base directory for mesh")
    parser.add_argument('--mesh_format', default="xdmf", help="Format for mesh")

    parser.add_argument('--ftol', type=float, default=1e-7, help="Incremental function change tolerance for LBFGSB")
    parser.add_argument('--gtol', type=float, default=1e-5, help="Gradient tolerance")
    parser.add_argument('--display', default=False, action="store_true", help="Display optimization iterations")

    parser.add_argument('--N_plots', type=int, default=4, help="Number of solves for plotting")
    parser.add_argument('--postprocess', default=False, action="store_true", help="run post processing solves")
    parser.add_argument('--load_initial_guess', default=False, action="store_true", help="Load initial guess for control")
    parser.add_argument('--initial_guess_directory', default=".", help="Initial guess directory for loading")
    parser.add_argument('--load_from_results', default=False, action="store_true", help="Loading from results.p")
    parser.add_argument('--seed', type=int, default=0, help="Number of solves for plotting")
    args = parser.parse_args()
    
    case_name = "pellet_%s" %(args.mesh_size)
    if args.constrained:
        results_dir = "results_cvar_constrained_%s_%s_eps%g" %(args.qoi_type, args.obs_type, args.epsilon)
    else:
        results_dir = "results_cvar_%s_%s_eps%g" %(args.qoi_type, args.obs_type, args.epsilon)

    if args.seed > 0:
        save_dir = results_dir + "/%s_penalty%g_beta%g_SAA_ns%d_alpha%g_maxiter%g_seed%d" %(case_name, args.penalty, args.beta, args.N_sample, args.alpha, args.max_iter, args.seed)
    else:
        save_dir = results_dir + "/%s_penalty%g_beta%g_SAA_ns%d_alpha%g_maxiter%g" %(case_name, args.penalty, args.beta, args.N_sample, args.alpha, args.max_iter)
    os.makedirs(save_dir, exist_ok=True)

    # MPI
    comm_world = MPI.COMM_WORLD
    world_rank = comm_world.Get_rank()
    world_size = comm_world.Get_size()
    comm_mesh = comm_world.Split(color = world_rank, key=0)
    comm_sampler = comm_world

    settings = defaultBluffBodySettings()
    settings["mesh_format"] = args.mesh_format
    settings["mesh_base_directory"] = args.mesh_base_directory
    settings["mesh_resolution"] = args.mesh_size 

    # ----------------- Make problem ----------------- #
    if comm_world.Get_rank() == 0:
        print("Making problem")
    mesh, Vh, ns_problem, prior, basis_all, geometry, geo_specs = build_navier_stokes_problem(settings)
    qoi = setup_qoi(args.qoi_type, args.obs_type, mesh, Vh, geo_specs, settings)
    control_model = soupy.ControlModel(ns_problem, qoi)
    x = control_model.generate_vector()

    if comm_world.Get_rank() == 0:
        print("STATE dimension is %d" %(len(x[soupy.STATE].get_local())))
        print("PARAMETER dimension is %d" %(len(x[soupy.PARAMETER].get_local())))
        print("CONTROL dimension is %d" %(len(x[soupy.CONTROL].get_local())))

    if args.seed > 0:
        # Do the sampling randomly 
        random.seed(1)
        for i in range(args.seed):
            seed = random.randrange(100_000_000)
    else:
        seed = 1 


    # Make the PDE risk measure
    rm_param = soupy.superquantileRiskMeasureSAASettings()
    rm_param["beta"] = args.beta
    rm_param["sample_size"] = args.N_sample
    rm_param["seed"] = seed
    rm_param["epsilon"] = args.epsilon
    pde_rm = soupy.SuperquantileRiskMeasureSAA(control_model, prior, settings=rm_param, comm_sampler=comm_sampler)

    # Cost functional
    penalty = L2VelocityPenalization(Vh, basis_all, geometry, args.penalty)
    pde_cost = soupy.RiskMeasureControlCostFunctional(pde_rm, penalty)
    scipy_cost = soupy.ScipyCostWrapper(pde_cost, verbose=True)


    # ----------------- Before optimization ----------------- #
    comm_world.Barrier()
    u_fun = dl.Function(Vh[hp.STATE])
    m_fun = dl.Function(Vh[hp.PARAMETER])
    z_fun = dl.Function(Vh[soupy.CONTROL])
    x = [u_fun.vector(), m_fun.vector(), None, z_fun.vector()]

    noise = dl.Vector(comm_mesh)
    prior.init_vector(noise, "noise")

    v, p = u_fun.split()
    if comm_world.Get_rank() == 0 and args.postprocess:
        print("Solving for state before optimization")
        v_file = dl.File(comm_mesh, "%s/v_uncontrolled.pvd" %(save_dir))
        p_file = dl.File(comm_mesh, "%s/p_uncontrolled.pvd" %(save_dir))

        rng = hp.Random()
        for i in range(args.N_plots):
            print("Solving %d of %d" %(i, args.N_plots))
            # Sample from prior
            rng.normal(1.0, noise)
            prior.sample(noise, m_fun.vector())
            # Solve forward
            control_model.solveFwd(u_fun.vector(), x)
            v_file << v
            p_file << p

    # ----------------- Optimize ----------------- #
    comm_world.Barrier()
    if comm_world.Get_rank() == 0:
        print("Begin optimization")

    z0 = pde_rm.generate_vector(soupy.CONTROL)
    if args.load_initial_guess:
        # Loading initial guess from numpy array
        z0_np = z0.get_local()
        z0_np[:-1] = np.load("%s/z_opt.npy" %(args.initial_guess_directory))
    elif args.load_from_results:
        with open("%s/results.p" %(args.initial_guess_directory), "rb") as f:
            results = pickle.load(f)
        z0_np = results['x']
    else:
        z0_np = z0.get_local()

    comm_world.Barrier()
    tsolve_0 = time.time()

    dim_CONTROL = Vh[soupy.CONTROL].dim()
    if args.constrained:
        lb = np.ones(dim_CONTROL + 1) * settings["control_min"]
        ub = np.ones(dim_CONTROL + 1) * settings["control_max"]
        lb[-1] = None
        ub[-1] = None 
        box_bounds = scipy.optimize.Bounds(lb=lb, ub=ub)
        options = {'disp' : args.display, 'ftol' : args.ftol, 'gtol' : args.gtol, 'maxiter' : args.max_iter}
        results = scipy.optimize.minimize(scipy_cost.function(), z0_np, method="L-BFGS-B", jac=scipy_cost.jac(), bounds=box_bounds, options=options)
    else:
        options = {'disp' : args.display, 'gtol' : args.gtol, 'maxiter' : args.max_iter}
        results = scipy.optimize.minimize(scipy_cost.function(), z0_np, method="BFGS", jac=scipy_cost.jac(), options=options)

    zt_opt_np = results["x"]
    z_opt_np = zt_opt_np[:-1]
    tsolve_1 = time.time()
    tsolve = tsolve_1 - tsolve_0

    # -----------------  Post processing ----------------- #
    # Saving the cost evaluations at optimal
    z = control_model.generate_vector(soupy.CONTROL)
    z.set_local(z_opt_np)
    penalization_opt = penalty.cost(z)

    zt = soupy.AugmentedVector(z, copy_vector=True)
    zt.set_local(zt_opt_np)
    pde_rm.computeComponents(zt)
    risk_opt = pde_rm.cost()
    cvar_opt = pde_rm.superquantile()

    if comm_world.Get_rank() == 0:
        print("Optimal distribution: ", z_opt_np)
        print("Time to solve: %g" %tsolve)
        np.save("%s/z_opt.npy" %(save_dir), z_opt_np)

        costs = dict()
        costs["penalization"] = penalization_opt
        costs["risk_opt"] = risk_opt
        costs["cvar_opt"] = cvar_opt
        costs["n_func"] = scipy_cost.n_func
        costs["n_grad"] = scipy_cost.n_grad

        with open("%s/costs.p" %(save_dir), "wb") as f:
            pickle.dump(costs, f)

        with open("%s/results.p" %(save_dir), "wb") as f:
            pickle.dump(results, f)

        if args.postprocess:
            x[soupy.CONTROL].set_local(z_opt_np)
            v_file = dl.File(comm_mesh, "%s/v_controlled.pvd" %(save_dir))
            p_file = dl.File(comm_mesh, "%s/p_controlled.pvd" %(save_dir))
            rng = hp.Random()
            for i in range(args.N_plots):
                # Sample from prior
                rng.normal(1.0, noise)
                prior.sample(noise, m_fun.vector())
                # Solve forward
                control_model.solveFwd(u_fun.vector(), x)
                v_file << v
                p_file << p

