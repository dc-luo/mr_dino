import os
import platform
import dolfin as dl
import numpy as np

def configure_gpu(enable_gpu):
    if enable_gpu == "0":
        print("Enabling GPU 0")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif enable_gpu == "1":
        print("Enabling GPU 1")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif enable_gpu == "all":
        print("Enabling all GPUs")
    else:
        print("Disabling GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def projected_stiffness_matrix(u_basis):
    mesh = dl.UnitSquareMesh(64, 64)
    Vh = dl.FunctionSpace(mesh, "CG", 1)
    u_trial = dl.TrialFunction(Vh)
    u_test = dl.TestFunction(Vh)

    W = dl.assemble(u_trial*u_test*dl.dx + dl.Constant(10.0) * dl.inner(dl.grad(u_trial), dl.grad(u_test))*dl.dx).array()
    Wr = u_basis.T @ W @ u_basis
    return Wr



def layers_from_args(args):
    """
    From parsed arguments, extract layer breadths
    Where if none is specified, use default sizes
    """
    DEFAULT_PRE_LAYERS_M = []
    DEFAULT_PRE_LAYERS_Z = []
    DEFAULT_POST_LAYERS = [200, 200]

    if args.pre_layers_m is None:
        pre_layers_m = DEFAULT_PRE_LAYERS_M
    else:
        pre_layers_m = args.pre_layers_m

    if args.pre_layers_z is None:
        pre_layers_z = DEFAULT_PRE_LAYERS_Z
    else:
        pre_layers_z = args.pre_layers_z

    if args.post_layers is None:
        post_layers = DEFAULT_POST_LAYERS
    else:
        post_layers = args.post_layers
    return pre_layers_m, pre_layers_z, post_layers



def save_dir_and_layers_from_args(args):
    # Make save directory
    pre_layers_m, pre_layers_z, post_layers = layers_from_args(args)

    save_dir = '%s/trained_weights/' %(args.training_base_dir)

    if args.mass_weighting:
        save_dir += 'M'
    if args.shifted:
        save_dir += 'S'

    save_dir += 'rbnet_data%d' %(args.n_train)

    if args.l2_only:
        save_dir += '_l2_only'
    
    save_dir += '_kle-pod_m%du%d' %(args.m_rank, args.u_rank)

    # if args.Cinv:
    #     save_dir += "_Cinv"

    for layer in post_layers:
        save_dir += '_%d' %(layer)

    if args.scheduler:
        save_dir += "_scheduler"

    if args.index > 0:
        save_dir += '_run%d' %(args.index)

    return save_dir, pre_layers_m, pre_layers_z, post_layers

