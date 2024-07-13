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
from navier_stokes_problem import build_navier_stokes_problem, defaultBluffBodySettings
from navier_stokes_ouu_utils import setup_qoi, L2VelocityPenalization

def evaluate_penalty_mpi(penalty, z, save_dict, is_uncontrolled=False):
    """
    Evaluats the penalty for optimal control `z` and saves results to `save_dict`
    """
    penalty_cost = penalty.cost(z)
    
    if isinstance(z, soupy.AugmentedVector):
        z_np = z.get_local()[:-1]
    else:
        z_np = z.get_local()

    if args.uncontrolled:
        raw_penalty = 0.0
        raw_penalty_grad = np.zeros_like(z_np)
    else:
        M_np = penalty.M.array()
        raw_penalty = np.inner(z_np, M_np @ z_np)
        raw_penalty_grad = 2 * M_np @ z_np 

    save_dict["alpha"] = penalty.alpha
    save_dict["penalty"] = penalty_cost
    save_dict["raw_penalty"] = raw_penalty
    save_dict["raw_penalty_grad"] = raw_penalty_grad



def evaluate_risk_mpi(risk_measure_mpi, z, save_dict, is_cvar=False):
    """
    Evaluats the risk measure for optimal control `z` and saves results to `save_dict`
    """

    # print("Evaluating risk at optimal control")
    risk_measure_mpi.computeComponents(z, order=1)
    sample_size = risk_measure_mpi.sample_size
    comm_sampler = risk_measure_mpi.comm_sampler

    save_dict["cost"] = risk_measure_mpi.cost()
    save_dict["N"] = sample_size
        
    # Save gradient 
    g = risk_measure_mpi.generate_vector(soupy.CONTROL)
    risk_measure_mpi.grad(g)
    save_dict["grad"] = g.get_local()
    save_dict["gradnorm"] = np.linalg.norm(g.get_local())
    
    if comm_sampler.Get_rank() == 0: 
        print("-" * 80)
        print("Risk at optimal solution by %d PDE solves: %g" %(sample_size, risk_measure_mpi.cost()))
        print("Risk grad norm: %g" %(save_dict["gradnorm"]))

    if is_cvar:
        # Also compute gradient for t = quantile 
        beta = risk_measure_mpi.beta
        epsilon = risk_measure_mpi.settings["epsilon"]
        q_samples = risk_measure_mpi.gather_samples()
        q_sample_quantile = 0.0 
        q_min_quantile = 0.0 

        if comm_sampler.Get_rank() == 0:
            q_sample_quantile = np.quantile(q_samples, beta)
            q_sample_superquantile = soupy.sample_superquantile(q_samples, beta)
            q_min_quantile, q_min_superquantile = soupy.sample_superquantile_by_minimization(q_samples, beta, epsilon=epsilon)
            print("-" * 20)
            print("Sample quantile = %g" %(q_sample_quantile))
            print("Minimization quantile = %g" %(q_min_quantile))

            print("-" * 20)
            print("Sample superquantile = %g" %(q_sample_superquantile))
            print("Minimization superquantile = %g" %(q_min_superquantile))
            print("-" * 20)
        
        q_sample_quantile = comm_sampler.bcast(q_sample_quantile, root=0)
        q_min_quantile = comm_sampler.bcast(q_min_quantile, root=0)

        z.set_scalar(q_sample_quantile)
        risk_measure_mpi.computeComponents(z, order=1)
        risk_measure_mpi.grad(g)
        save_dict["grad_at_sample_quantile"] = g.get_local()
        save_dict["gradnorm_at_sample_quantile"] = np.linalg.norm(g.get_local())

        z.set_scalar(q_min_quantile)
        risk_measure_mpi.computeComponents(z, order=1)
        risk_measure_mpi.grad(g)
        save_dict["grad_at_minimization_quantile"] = g.get_local()
        save_dict["gradnorm_at_minimization_quantile"] = np.linalg.norm(g.get_local())

        if comm_sampler.Get_rank() == 0: 
            print("With t = quantile from sample")
            print("Risk gradnorm wrt z: %g" %(np.linalg.norm(save_dict["grad_at_sample_quantile"][:-1])))
            print("Risk gradnorm wrt t: %g" %(np.linalg.norm(save_dict["grad_at_sample_quantile"][-1])))

            print("With t = quantile from minimization")
            print("Risk gradnorm wrt z: %g" %(np.linalg.norm(save_dict["grad_at_minimization_quantile"][:-1])))
            print("Risk gradnorm wrt t: %g" %(np.linalg.norm(save_dict["grad_at_minimization_quantile"][-1])))
            print("-" * 80)
    
        sys.stdout.flush()




if __name__ == "__main__":
    dl.parameters["form_compiler"]["quadrature_degree"] = 5

    parser = argparse.ArgumentParser(description="Solving OUU problem")
    parser.add_argument('-b', '--beta', type=float, default=0.0, help="Variance weight")
    parser.add_argument('-N', '--N_sample', type=int, default=64, help="Number of samples for SAA")
    parser.add_argument('-n', '--max_iter', type=int, default=100, help="Maximum number of optimizerion iterations")
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help="Step size for optimization (not used anymore)")
    parser.add_argument('-p', '--penalty', type=float, default=0.0, help="Number of samples for evaluation of optimum")
    parser.add_argument('-u', '--uncertainty', type=str, default="SAA", help="Type of uncertainty")
    parser.add_argument('-m', '--model_prefix', type=str, default="", help="Model to prefix in front of directory")
    parser.add_argument('-s', '--suffix', type=str, default="", help="Additional suffix in run name")
    parser.add_argument('-q', '--qoi_type', type=str, default="dissipation", choices=["dissipation", "tracking_up", "tracking"], help="Type of QoI")
    parser.add_argument('-o', '--obs_type', type=str, default="full", choices=["full", "full_back", "windowed_back", "windowed_body"], help="Window of observation")
    parser.add_argument('--epsilon', type=float, default=1e-4, help="CVaR sharpness")

    parser.add_argument('--mesh_size', type=str, default="medium", choices=["medium", "medium_coarse", "coarse", "coarsest"],  help="Mesh size specification")
    parser.add_argument('--mesh_base_directory', default="../", help="Base directory for mesh")
    parser.add_argument('--mesh_format', default="xdmf", help="Format for mesh")

    parser.add_argument('-e', '--N_eval', type=int, default=100, help="Number of samples for evaluation of optimum")
    parser.add_argument('--N_plots', type=int, default=4, help="Number of solves for plotting")
    parser.add_argument('--results_directory', default="results_cvar", help="directory where all cases are stored")
    parser.add_argument('--uncontrolled', default=False, action="store_true", help="Compute for uncontrolled")

    args = parser.parse_args()

    # ---------------------------------- Parse inputs ---------------------------------- #

    case_name = "pellet_%s" %(args.mesh_size)
    if args.uncontrolled:
        save_dir = "uncontrolled"
    else:
        if args.uncertainty == "SAA":
            save_dir = "%s/%s%s_penalty%g_beta%g_SAA_ns%d_alpha%g_maxiter%g" %(args.results_directory, args.model_prefix, case_name,
                    args.penalty, args.beta, args.N_sample, args.alpha, args.max_iter)
        else:
            save_dir = "%s/%s_deterministic_penalty%g_alpha%g_maxiter%g" %(args.results_directory, case_name, args.penalty, args.alpha, args.max_iter)
        save_dir += args.suffix
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

    # ---------------------------------- Make problem ---------------------------------- #
    t0_setup = time.time()
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

    # ----------------- Make risk measure ----------------- #
    
    if 'cvar' in args.results_directory:
        is_cvar = True
        if comm_sampler.Get_rank() == 0:
            print("Using a CVaR risk measure")

        rm_param = soupy.superquantileRiskMeasureSAASettings()
        rm_param["sample_size"] = args.N_eval
        rm_param["beta"] = args.beta 
        rm_param["epsilon"] = args.epsilon
        pde_rm = soupy.SuperquantileRiskMeasureSAA(control_model, prior, settings=rm_param, comm_sampler=comm_sampler)

        z = pde_rm.generate_vector(soupy.CONTROL)
        if not args.uncontrolled:
            with open("%s/results.p" %(save_dir), "rb") as results_file:
                opt_results = pickle.load(results_file)
                zt_opt_np = opt_results['x']

            z.set_local(zt_opt_np)
            z_np = zt_opt_np[:-1]
    else:
        is_cvar = False 
        rm_param = soupy.meanVarRiskMeasureSAASettings()
        rm_param["sample_size"] = args.N_eval
        rm_param["beta"] = args.beta
        pde_rm = soupy.MeanVarRiskMeasureSAA(control_model, prior, settings=rm_param, comm_sampler=comm_sampler)
        z = pde_rm.generate_vector(soupy.CONTROL)

        if not args.uncontrolled:
            z_np = np.load("%s/z_opt.npy" %(save_dir))
            z.set_local(z_np)

    penalty = L2VelocityPenalization(Vh, basis_all, geometry, args.penalty)

    t1_setup = time.time()
    t_setup = t1_setup - t0_setup 

    # ---------------------------------- Start evaluation ---------------------------------- #
    comm_world.Barrier()


    costs = dict()
    
    # PDE SAA risk measure estimate 
    t0_solve = time.time()
    evaluate_risk_mpi(pde_rm, z, costs, is_cvar)
    t1_solve = time.time()
    t_solve = t1_solve - t0_solve
    q_samples = pde_rm.gather_samples()
    
    # Penalization 
    evaluate_penalty_mpi(penalty, z, costs, args.uncontrolled)
    
    sys.stdout.flush()
    # ---------------------------------- Output results ---------------------------------- #

    if world_rank == 0:
        print("Setup time: %g seconds" %(t_setup))
        print("Solves time %g seconds" %(t_solve))
        np.save("%s/q_opt.npy" %(save_dir), q_samples)

        plt.figure()
        plt.hist(q_samples, bins=30)
        plt.xlabel("r$Q$")
        plt.ylabel("Frequency")
        plt.savefig("%s/q_opt.png" %(save_dir))

        with open("%s/cost_opt.p" %(save_dir), "wb") as f_costs:
            pickle.dump(costs, f_costs)

        u_fun = dl.Function(Vh[hp.STATE])
        m_fun = dl.Function(Vh[hp.PARAMETER])
        v_file = dl.File(comm_mesh, "%s/paraview/v_controlled.pvd" %(save_dir))
        p_file = dl.File(comm_mesh, "%s/paraview/p_controlled.pvd" %(save_dir))

        v_fun, p_fun = u_fun.split()
        for i in range(args.N_plots):
            u_fun.vector().zero()
            u_fun.vector().axpy(1.0, pde_rm.x_mc[i][soupy.STATE])
            v_file << v_fun
            p_file << p_fun
