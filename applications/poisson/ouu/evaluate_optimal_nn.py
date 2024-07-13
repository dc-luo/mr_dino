import time
import os, sys
import pickle
import argparse

from mpi4py import MPI 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import dolfin as dl

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))

import hippylib as hp
import hippyflow as hf
import soupy

sys.path.append("../../../")
import mr_dino.spoon as spoon  
import mr_dino.neural_network as nn 

sys.path.append("../")
from poisson_pde import build_poisson_problem, poisson_control_settings
dl.set_log_active(False)

from training.local_utils import configure_gpu

from poisson_ouu_utils import get_target, plot_wells

normi = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
 
plt.rcParams.update({"font.size" : 28})
plt.rcParams.update({"font.family" : "serif"})
plt.rcParams.update({"mathtext.fontset" : "cm"})

def get_colorbar_ticks(vmin, vmax):
    vmin = round(vmin, 1) - 0.1
    vmax = round(vmax, 1) + 0.1
    ticks = np.arange(vmin, 0.1, vmax)
    return vmin, vmax, ticks 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Solving OUU problem")
    parser.add_argument('-t', '--target', type=str, default="sinusoid", help="target case")
    parser.add_argument('-p', '--param', type=float, default=1.0, help="Parameter for target definition")
    parser.add_argument('-b', '--beta', type=float, default=0.95, help="Variance weight or CVaR quantile")
    parser.add_argument('-m', '--model', type=str, default="Srbnet_data80_kle-pod_m20u20_200_200", help="NN Model used for optimization")
    parser.add_argument('-o', '--optimization', type=str, default="SAA", help="Optimization algorithm")
    parser.add_argument('-N', '--N_sample', type=int, default=2048, help="Number of samples for SD")
    parser.add_argument('-n', '--max_sd_iter', type=int, default=100, help="Maximum number of SD iterations")
    parser.add_argument('-a', '--alpha', type=float, default=1, help="Step size for SD iterations")
    parser.add_argument('-e', '--N_eval', type=int, default=4096, help="Number of samples for evaluation of optimum")
    parser.add_argument('-d', '--results_dir', type=str, default="results_rbnet_cvar_eps0.0001", help="Directory of the results")
    parser.add_argument('-s', '--suffix', type=str, default="_lbfgsb_random1", help="Additional suffixes")

    parser.add_argument('--nx', type=int, default=64, help="Number of elements in x direction")
    parser.add_argument('--ny', type=int, default=64, help="Number of elements in y direction")
    parser.add_argument('--N_sources', type=int, default=7, help="Number of sources per side")
    parser.add_argument('--loc_lower', type=float, default=0.1, help="Lower bound of location of sources")
    parser.add_argument('--loc_upper', type=float, default=0.9, help="Upper bound of location of sources")
    parser.add_argument('--well_width', type=float, default=0.08, help="Upper bound of location of sources")
    parser.add_argument('-g', '--enable_gpu', default="", type=str, help="Enable GPU for tensorflow")
    parser.add_argument("--N_plot", type=int, default=4, help="Number of plots")
    parser.add_argument("--usetitle", default=False, action="store_true", help="Title on plots")
    args = parser.parse_args()

    configure_gpu(args.enable_gpu)

    # ----------------- Initialize parameters ----------------- # 
    # Set parsed parameters
    poisson_settings = poisson_control_settings()
    poisson_settings['nx'] = args.nx
    poisson_settings['ny'] = args.ny
    poisson_settings['WELL_WIDTH'] = args.well_width
    poisson_settings['LOC_LOWER'] = args.loc_lower
    poisson_settings['LOC_UPPER'] = args.loc_upper
    poisson_settings['N_WELLS_PER_SIDE'] = args.N_sources

    save_dir = "%s/NN_%s_mesh%dx%d_sources%d_%gto%g_width%g_target_%s_p%g_beta%g_%s_alpha%g_ns%d_maxiter%d" %(
        args.results_dir, args.model, 
        args.nx, args.ny, 
        args.N_sources, args.loc_lower, args.loc_upper, args.well_width,
        args.target, args.param,
        args.beta, args.optimization, args.alpha, args.N_sample, args.max_sd_iter)
    save_dir += args.suffix

    z_opt_np = np.load("%s/z_opt.npy" %(save_dir)) 

    # ----------------- Make problem ----------------- # 
    mesh, pde, Vh, prior, control_distribution = build_poisson_problem(poisson_settings)

    u_target_expr = get_target(args.target, args.param)
    u_target = dl.interpolate(u_target_expr, Vh[hp.STATE])
    qoi = soupy.L2MisfitControlQoI(Vh, u_target.vector())

    # Define the PDE control model
    control_model = soupy.ControlModel(pde, qoi)

    # Define the NN control model 
    weights_path = "../training/trained_weights/" + args.model
    rbnet = nn.load_multi_input_reduced_basis_network(weights_path)

    # Load neural operator
    u_trial = dl.TrialFunction(Vh[soupy.STATE])
    u_test = dl.TestFunction(Vh[soupy.STATE])
    M = dl.assemble(u_trial * u_test * dl.dx)

    neural_operator = spoon.ControlNeuralOperator(Vh, rbnet.full_network)
    neural_qoi = spoon.NNSparseL2QoI(M, d=u_target.vector())
    nn_control_model = spoon.NNOperatorQoIControlModel(Vh, neural_operator, neural_qoi)

    # -----------------  Post processing ----------------- #
    u_fun = dl.Function(Vh[hp.STATE])
    m_fun = dl.Function(Vh[hp.PARAMETER])
    z_fun = dl.Function(Vh[soupy.CONTROL])
    x = [u_fun.vector(), m_fun.vector(), None, z_fun.vector()]

    if args.results_dir == "results_cvar":
        x[soupy.CONTROL].set_local(z_opt_np[:-1])
    else:
        x[soupy.CONTROL].set_local(z_opt_np)

    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    
    os.makedirs("%s/figures" %(save_dir), exist_ok=True)

    figsize = (6,6)
    colorbar_scale = 0.65
    colorbar_ticks = 10

    plt.figure(figsize=figsize)
    tri = hp.nb.plot(u_target, colorbar=False)
    plt.colorbar(tri, shrink=colorbar_scale, format='%.1f')

    if args.usetitle:
        plt.title("Target state")

    plt.tight_layout()
    plt.savefig("%s/figures/target.png" %(save_dir))
    plt.savefig("%s/figures/target.pdf" %(save_dir))
    plt.close()

    fig, ax = plot_wells(z_opt_np, poisson_settings["N_WELLS_PER_SIDE"], poisson_settings["LOC_LOWER"], poisson_settings["LOC_UPPER"])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0.5, 1])
    ax.set_zticks([-1, 0.0, 1])
    if args.usetitle:
        plt.title("Optimal control")

    plt.tight_layout()
    plt.savefig("%s/figures/control.pdf" %(save_dir))
    plt.savefig("%s/figures/control.png" %(save_dir))

    for i in range(args.N_plot):
        # Sample from prior
        hp.parRandom.normal(1.0, noise)
        prior.sample(noise, m_fun.vector())

        # Solve forward PDE 
        control_model.solveFwd(x[hp.STATE], x)
        m_np = np.reshape(m_fun.vector().get_local(), (1,-1))
        z_np = np.reshape(z_opt_np, (1,-1))
        print(m_np.shape[0], z_np.shape[0])
        u_np = nn_control_model.solveFwd(m_np, z_np)
        u_pde_np = u_fun.vector().get_local()

        u_max_pde = np.max(u_pde_np)
        u_min_pde = np.min(u_pde_np)
        u_max_nn = np.max(u_np)
        u_min_nn = np.min(u_np)

        vmin = np.min([u_min_nn, u_min_pde]) 
        vmax = np.max([u_max_nn, u_max_pde])
        # vmin, vmax, cbticks = get_colorbar_ticks(vmin, vmax)
        print(vmin, vmax)

        plt.figure(figsize=figsize)
        tri = hp.nb.plot(m_fun, colorbar=False)
        plt.colorbar(tri, shrink=colorbar_scale, format="%.1f")
        if args.usetitle:
            plt.title("Random parameter")
        plt.tight_layout()
        plt.savefig("%s/figures/parameter_%d.png" %(save_dir, i))
        plt.savefig("%s/figures/parameter_%d.pdf" %(save_dir, i))

        plt.figure(figsize=figsize)
        tri = hp.nb.plot(u_fun, vmin=vmin, vmax=vmax, colorbar=False)
        cb = plt.colorbar(tri, shrink=colorbar_scale, format="%.1f")

        if args.usetitle:
            plt.title("PDE prediction")

        plt.tight_layout()
        plt.savefig("%s/figures/pde_state_%d.png" %(save_dir, i))
        plt.savefig("%s/figures/pde_state_%d.pdf" %(save_dir, i))

        u_fun.vector().set_local(u_np[0,:])
        
        fig = plt.figure(figsize=figsize)
        tri2 = hp.nb.plot(u_fun, vmin=vmin, vmax=vmax, colorbar=False)
        ax = plt.gca()
        cb = plt.colorbar(tri, ax=ax, shrink=colorbar_scale, format="%.1f")
        if args.usetitle:
            plt.title("NN prediction")
        plt.tight_layout()
        plt.savefig("%s/figures/nn_state_%d.png" %(save_dir, i))
        plt.savefig("%s/figures/nn_state_%d.pdf" %(save_dir, i))
        plt.close()
   
    m_np = [] 
    z_np = [] 
    for i in range(args.N_eval):
        # Sample from prior
        hp.parRandom.normal(1.0, noise)
        prior.sample(noise, m_fun.vector())
        m_np.append(m_fun.vector().get_local())
        z_np.append(z_opt_np)
    m_np = np.array(m_np) 
    z_np = np.array(z_np)
    q_np = nn_control_model.cost(m_np, z_np)

    np.save("%s/nn_q_opt.npy" %(save_dir), q_np)