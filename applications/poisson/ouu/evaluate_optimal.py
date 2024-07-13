import time
import os, sys
import pickle
import argparse

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))

import hippylib as hp

import hippyflow as hf

import soupy

sys.path.append("../../../")
sys.path.append("../")

from poisson_pde import build_poisson_problem, poisson_control_settings
dl.set_log_active(False)

from poisson_ouu_utils import get_target, plot_wells


def evaluate_risk_mpi(risk_measure_mpi, z, is_cvar=False):
    # print("Evaluating risk at optimal control")
    risk_measure_mpi.computeComponents(z, order=1)
    sample_size = risk_measure_mpi.sample_size
    comm_sampler = risk_measure_mpi.comm_sampler

    risk_opt = dict()
    risk_opt["cost"] = risk_measure_mpi.cost()
    risk_opt["N"] = sample_size
        
    # Save gradient 
    g = risk_measure_mpi.generate_vector(soupy.CONTROL)
    risk_measure_mpi.grad(g)
    risk_opt["grad"] = g.get_local()
    risk_opt["gradnorm"] = np.linalg.norm(g.get_local())
    
    if comm_sampler.Get_rank() == 0: 
        print("-" * 80)
        print("Risk at optimal solution by %d PDE solves: %g" %(sample_size, risk_measure_mpi.cost()))
        print("Risk grad norm: %g" %(risk_opt["gradnorm"]))

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
        risk_opt["grad_at_sample_quantile"] = g.get_local()
        risk_opt["gradnorm_at_sample_quantile"] = np.linalg.norm(g.get_local())

        z.set_scalar(q_min_quantile)
        risk_measure_mpi.computeComponents(z, order=1)
        risk_measure_mpi.grad(g)
        risk_opt["grad_at_minimization_quantile"] = g.get_local()
        risk_opt["gradnorm_at_minimization_quantile"] = np.linalg.norm(g.get_local())

        if comm_sampler.Get_rank() == 0: 
            print("With t = quantile from sample")
            print("Risk gradnorm wrt z: %g" %(np.linalg.norm(risk_opt["grad_at_sample_quantile"][:-1])))
            print("Risk gradnorm wrt t: %g" %(np.linalg.norm(risk_opt["grad_at_sample_quantile"][-1])))

            print("With t = quantile from minimization")
            print("Risk gradnorm wrt z: %g" %(np.linalg.norm(risk_opt["grad_at_minimization_quantile"][:-1])))
            print("Risk gradnorm wrt t: %g" %(np.linalg.norm(risk_opt["grad_at_minimization_quantile"][-1])))
            print("-" * 80)

    return risk_opt



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Solving OUU problem")
    parser.add_argument('-t', '--target', type=str, default="sinusoid", help="target case")
    parser.add_argument('-p', '--param', type=float, default=1.0, help="Parameter for target definition")
    parser.add_argument('-b', '--beta', type=float, default=0.95, help="Variance weight or CVaR quantile")
    parser.add_argument('-m', '--model', type=str, default="PDE", help="Model used for optimization")
    parser.add_argument('-o', '--optimization', type=str, default="SAA", help="Optimization algorithm")
    parser.add_argument('--epsilon', type=float, default=1e-4, help="Sharpness of CVaR of approximation, if applicable")

    parser.add_argument('-N', '--N_sample', type=int, default=64, help="Number of samples for SD")
    parser.add_argument('-n', '--max_sd_iter', type=int, default=100, help="Maximum number of SD iterations")
    parser.add_argument('-a', '--alpha', type=float, default=1, help="Step size for SD iterations")
    parser.add_argument('-e', '--N_eval', type=int, default=100, help="Number of samples for evaluation of optimum")
    parser.add_argument('-d', '--results_dir', type=str, default="results_cvar_eps0.0001", help="Directory of the results")
    parser.add_argument('-s', '--suffix', type=str, default="", help="additional suffixes")

    parser.add_argument('--nx', type=int, default=64, help="Number of elements in x direction (for the evaluation)")
    parser.add_argument('--ny', type=int, default=64, help="Number of elements in y direction (for the evaluation)")
    parser.add_argument('--nx_run', type=int, default=64, help="Number of elements in x direction (running the optimizer)")
    parser.add_argument('--ny_run', type=int, default=64, help="Number of elements in y direction (running the optimizer)")
    parser.add_argument('--N_sources', type=int, default=7, help="Number of sources per side")
    parser.add_argument('--loc_lower', type=float, default=0.1, help="Lower bound of location of sources")
    parser.add_argument('--loc_upper', type=float, default=0.9, help="Upper bound of location of sources")
    parser.add_argument('--well_width', type=float, default=0.08, help="Upper bound of location of sources")


    parser.add_argument("--N_plot", type=int, default=4, help="Number of plots")
    args = parser.parse_args()

    # MPI
    comm_world = MPI.COMM_WORLD
    world_rank = comm_world.Get_rank()
    world_size = comm_world.Get_size()
    comm_mesh = comm_world.Split(color = world_rank, key=0)
    comm_sampler = comm_world

    # ----------------- Initialize parameters ----------------- #
    # Set parsed parameters
    # Model choice

    # Poisson settings
    poisson_settings = poisson_control_settings()
    poisson_settings['nx'] = args.nx
    poisson_settings['ny'] = args.ny
    poisson_settings['WELL_WIDTH'] = args.well_width
    poisson_settings['LOC_LOWER'] = args.loc_lower
    poisson_settings['LOC_UPPER'] = args.loc_upper
    poisson_settings['N_WELLS_PER_SIDE'] = args.N_sources

    # Objective parameters

    if args.optimization == "deterministic":
        save_dir = "%s/%s_mesh%dx%d_sources%d_%gto%g_width%g_target_%s_p%g_beta%g_%s_alpha%g_maxiter%d" %(
            args.results_dir, args.model,
            args.nx_run, args.ny_run,
            args.N_sources, args.loc_lower, args.loc_upper, args.well_width,
            args.target, args.param,
            args.beta, args.optimization, args.alpha, args.max_sd_iter)
    else:
        save_dir = "%s/%s_mesh%dx%d_sources%d_%gto%g_width%g_target_%s_p%g_beta%g_%s_alpha%g_ns%d_maxiter%d" %(
            args.results_dir, args.model,
            args.nx_run, args.ny_run,
            args.N_sources, args.loc_lower, args.loc_upper, args.well_width,
            args.target, args.param,
            args.beta, args.optimization, args.alpha, args.N_sample, args.max_sd_iter)

    save_dir += args.suffix
    
    z_opt_np = np.load("%s/z_opt.npy" %(save_dir))

    # for post processing
    N_EVAL = args.N_eval

    # ----------------- Make problem ----------------- #
    mesh, pde, Vh, prior, control_distribution = build_poisson_problem(poisson_settings, comm_mesh=comm_mesh)

    # Define the control model
    u_target_expr = get_target(args.target, args.param)
    u_target = dl.interpolate(u_target_expr, Vh[hp.STATE])

    qoi = soupy.L2MisfitControlQoI(Vh, u_target.vector())
    control_model = soupy.ControlModel(pde, qoi)
    

    if 'cvar' in args.results_dir:
        if comm_sampler.Get_rank() == 0:
            print("Using a CVaR risk measure")
        rm_param_saa = soupy.superquantileRiskMeasureSAASettings()
        rm_param_saa["sample_size"] = args.N_eval
        rm_param_saa["beta"] = args.beta 
        rm_param_saa["epsilon"] = args.epsilon
        eval_rm = soupy.SuperquantileRiskMeasureSAA(control_model, prior, 
                                                        settings=rm_param_saa, 
                                                        comm_sampler=comm_sampler)

        with open("%s/results.p" %(save_dir), "rb") as results_file:
            opt_results = pickle.load(results_file)
            zt_opt_np = opt_results['x']

        z_opt = eval_rm.generate_vector(soupy.CONTROL)
        z_opt.set_local(zt_opt_np)
        is_cvar = True

    else:
        if comm_sampler.Get_rank() == 0:
            print("Using a Mean + Var risk measure")
        rm_param_saa = soupy.meanVarRiskMeasureSAASettings()
        rm_param_saa["sample_size"] = args.N_eval
        rm_param_saa["beta"] = args.beta
        eval_rm = soupy.MeanVarRiskMeasureSAA(control_model, prior, settings=rm_param_saa, comm_sampler=comm_sampler)

        z_opt = eval_rm.generate_vector(soupy.CONTROL)
        z_opt.set_local(z_opt_np)
        is_cvar = False
    
    sys.stdout.flush()
    # -----------------  Post processing ----------------- #

    # 1. Risk evaluation
    t0 = time.time()
    risk_opt = evaluate_risk_mpi(eval_rm, z_opt, is_cvar=is_cvar)
    t1 = time.time()

    q_opt = eval_rm.gather_samples()
    u_fun = dl.Function(Vh[hp.STATE])
    m_fun = dl.Function(Vh[hp.PARAMETER])
    z_fun = dl.Function(Vh[soupy.CONTROL])

    if comm_sampler.Get_rank() == 0:
        for i in range(args.N_plot):
            u_fun.vector().set_local(eval_rm.x_mc[i][soupy.STATE].get_local())
            m_fun.vector().set_local(eval_rm.x_mc[i][soupy.PARAMETER].get_local())

            plt.figure(figsize=(15,10))
            hp.nb.plot(u_target, subplot_loc=221, mytitle="Target state")
            hp.nb.plot(m_fun, subplot_loc=222, mytitle="Random parameter")
            hp.nb.plot(u_fun, subplot_loc=223, mytitle="PDE Solution")
            plt.savefig("%s/optimal_state%d.png" %(save_dir, i))
            plt.close()

        # output results
        print("Risk evaluation time: %g s" %(t1- t0))

        np.save("%s/q_opt.npy" %(save_dir), q_opt)

        plt.figure()
        plt.hist(q_opt, bins=30)
        plt.xlabel(r"$Q$")
        plt.savefig("%s/q_opt.png" %(save_dir))
        plt.close()

        with open("%s/risk_opt.p" %(save_dir), "wb") as f:
            pickle.dump(risk_opt, f)
